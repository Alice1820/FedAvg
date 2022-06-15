import copy
import gc
import itertools
import logging
from operator import mod
from typing_extensions import assert_type
from grpc import local_channel_credentials

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

# backbones
from .models.models import *
from .models.tsm import TSN
from .utils import *
from .client import Client

# import sys
# sys.path.append("..")

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, 
                server_config={}, clients_config={},
                init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.modals = server_config["modals"]
        self.learn_strategy = server_config["learn_strategy"]

        self.models = {}
        self.eval_models = {}
        self.model_config = model_config

        self.clients_modals = clients_config["clients_modals"]
        self.clients_learn_strategy = clients_config["clients_learn_strategy"]
        self.is_central = "N" in self.clients_learn_strategy

        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]
        self.use_ema = global_config["use_ema"]
        if self.use_ema: 
            self.ema_models = {}
            self.ema_decay = global_config["ema_decay"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_server_subjects = fed_config["SS"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.global_epoch = fed_config["SE"]
        self.batch_size = fed_config["B"]
        self.mu = fed_config["mu"]
        if self.learn_strategy == "SX": self.mu = 1 # server supervised learning

        # if self.learn_strategy != 'N': 
        #     assert self.num_server_subjects > 0
        #     self.is_central = True
        # else:
        #     self.is_central = False

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # config models
        # assert number of modals and models
        assert self.model_config['num_modals'] == len(self.model_config["modals"]) == len(self.model_config["backbone_config"])
        for modal, backbone_config in zip(self.model_config["modals"], self.model_config["backbone_config"]):
            assert modal == backbone_config["modal"]
        # init every model in self.models
        for idx, backbone_config in enumerate(self.model_config['backbone_config']): # backbone_config is a list of dictionaries
            model = eval(backbone_config["name"])(**backbone_config)
            modal = backbone_config["modal"]
            # initialize weights of the model
            torch.manual_seed(self.seed)
            # init_single_net(model, **self.init_config)
            # self.models.append(model)
            # self.models[modal] = nn.DataParallel(model)
            self.models[modal] = model

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized backbone {backbone_config}"
            print (message); gc.collect()

        if self.use_ema:
            from models.ema import ModelEMA
            for _modal in self.modals: self.ema_models[_modal] = ModelEMA(self.models[_modal], ema_decay=self.ema_decay)

        # print (self.models["RGB"])
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized {self.model_config['num_modals']} models \
            (# parameters: {[str(sum(p.numel() for p in model.parameters())) for model in self.models.values()]})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        # TODO: different num_local_epochs for every client
        local_datasets, global_dataset, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid, 
                                                        num_server_subjects=self.num_server_subjects, seed=self.seed)
        
        local_modals, local_modals_index, local_learn_strategy = create_modals(self.num_clients, self.modals, self.clients_modals, self.clients_learn_strategy)

        message = f"[Round: {str(self._round).zfill(4)}] Created server: DATA {str(len(global_dataset)).rjust(6, ' ')} , MODALS: {str(self.modals).rjust(20, ' ')},  LEARN: {self.learn_strategy.rjust(4, ' ')}!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign central_client
        # if self.is_central:
        #     self.central_client = Client(client_id=self.num_clients, local_data=global_dataset, local_modals=self.modals, local_modals_index=self.modals_index, 
        #                         local_learn_strategy=self.learn_strategy, device=self.device)
        #     message = f"[Round: {str(self._round).zfill(4)}] Created client: DATA: {str(len(self.global_dataset)).rjust(6, ' ')}, MODALS: {str(self.modals).rjust(20, ' ')},  LEARN: {self.learn_strategy.rjust(4, ' ')}!"
        #     print(message); logging.info(message)
        #     del message; gc.collect()

        # prepare hold-out dataset for evaluation
        self.train_data = global_dataset
        self.test_data = test_dataset
        if self.learn_strategy == "X":
            self.global_dataloader = DataLoader(global_dataset, batch_size=self.batch_size*self.mu, shuffle=True, num_workers=8)
        else:
            self.global_dataloader = DataLoader(global_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size*self.mu, shuffle=False, num_workers=8)
        
        self.modals_index = range(len(self.modals))

        if self.is_central:
            return;

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets=local_datasets, local_modals=local_modals, 
                                        local_modals_index=local_modals_index, local_learn_strategy=local_learn_strategy)

        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config,
            mu = self.mu
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets, local_modals, local_modals_index, local_learn_strategy):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in enumerate(local_datasets):
            # ***create clients
            client = Client(client_id=k, local_data=dataset, local_modals=local_modals[k], local_modals_index=local_modals_index[k], 
                            local_learn_strategy=local_learn_strategy[k], device=self.device, global_dataloader=self.global_dataloader)
            clients.append(client)
            message = f"[Round: {str(self._round).zfill(4)}] Created client: DATA: {str(len(dataset)).rjust(6, ' ')}, MODALS: {str(local_modals[k]).rjust(20, ' ')},  LEARN: {local_learn_strategy[k].rjust(4, ' ')}!"
            print(message); logging.info(message)
            del message; gc.collect()

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for client in self.clients:
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in self.clients: #
                # print (client.modals)
                for _modal in client.modals:
                    client.models[_modal] = copy.deepcopy(self.models[_modal])
                    client.models[_modal] = nn.DataParallel(client.models[_modal])

            # # move all models to cpu
            # for _model in self.models:
            #     _model.to("cpu")
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in sampled_client_indices:
                for _modal in self.clients[idx].modals:
                    self.clients[idx].models[_modal] = copy.deepcopy(self.models[_modal])
                    self.clients[idx].models[_modal] = nn.DataParallel(self.models[_modal])
            # move all models to cpu
            # for _model in self.models:
            #     _model.to("cpu")
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        for idx in sampled_client_indices:
            message = f"[Round: {str(self._round).zfill(4)}] Sampled Client NO: {str(idx).zfill(4)}, DATA: {str(len(self.clients[idx].data)).rjust(6, ' ')}, MODALS: {str(self.clients[idx].modals).rjust(20, ' ')},  LEARN: {self.clients[idx].learn_strategy.rjust(4, ' ')} ...!"
            print(message); logging.info(message)
            del message; gc.collect()

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = {}
        for _modal in self.modals:
            selected_total_size[_modal] = 0
        # for idx in tqdm(sampled_client_indices, leave=True):
        for idx in sampled_client_indices:
            # print (self.learn_strategy, self.clients[idx].learn_strategy)
            if self.learn_strategy == 'N' and self.clients[idx].learn_strategy in ['F', 'M']:
                self.clients[idx].client_joint_update()
            else:
                self.clients[idx].client_update()
            # update selected total size for every modal
            for _modal in self.modals:
                if _modal in self.clients[idx].modals:
                    selected_total_size[_modal] += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # two configs
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        # print (self.models.keys())
        # assert self.models.keys() == self.modals
        # iterate over all models
        for _modal in self.modals:
            # print (_modal)
            averaged_weights = OrderedDict()
            count = 0
            for it, idx in enumerate(sampled_client_indices):
                # print ("check length of client models", len(self.clients[idx].models)) # 2
                # parallel training
                # local_weights = self.clients[idx].models[model_index].state_dict()
                if _modal in self.clients[idx].modals:
                    local_weights = self.clients[idx].models[_modal].module.state_dict()
                    # print (local_weights.keys())
                    for key in self.models[_modal].state_dict().keys():
                        # print (local_weights.keys())
                        # if it == 0:
                        if count == 0: # is empty
                            averaged_weights[key] = coefficients[_modal][it] * local_weights[key]
                        else:
                            averaged_weights[key] += coefficients[_modal][it] * local_weights[key]
                    count += 1
            # print (_modal)
            if count > 0: 
                # ema
                if self.use_ema: self.ema_models[_modal].update(self.models[_modal])
                self.models[_modal].load_state_dict(averaged_weights)
                message = f"[Round: {str(self._round).zfill(4)}] MODAL: {str(_modal).rjust(6, ' ')} ...updated weights of {count} clients are successfully averaged!"
                print(message); logging.info(message)
                del message; gc.collect()
            else:
                message = f"[Round: {str(self._round).zfill(4)}] MODAL: {str(_modal).rjust(6, ' ')} ...not averaged in this round (no clients)!"
                print(message); logging.info(message)
                del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        # displays = [False for _ in sampled_client_indices]
        # displays[0] = True
        # map_configs = list(itertools.product(sampled_client_indices, displays))
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
                # selected_total_size = workhorse.map(self.mp_update_selected_clients, map_configs)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # if self.clientval_flag:
        #     if self.mp_flag:
        #         message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        #         print(message); logging.info(message)
        #         del message; gc.collect()

        #         with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
        #             workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        #     else:
        #         self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = {}
        for _modal in self.modals:
            mixing_coefficients[_modal] = []
            for idx in sampled_client_indices:
                if _modal in self.clients[idx].modals:
                    mixing_coefficients[_modal].append(len(self.clients[idx]) / selected_total_size[_modal])
                else:
                    mixing_coefficients[_modal].append(0)

        # average each updated model parameters of the selected clients and update the global model
        return sampled_client_indices, mixing_coefficients
    
    def train_global_model(self):
        """Update the global model using the global holdout training dataset (self.train_dataloader)."""
        optimizers = {}
        for _modal in self.models.keys():
            self.models[_modal].train()
            self.models[_modal].to(self.device)
            optimizers[_modal] = (eval(self.optimizer)(self.models[_modal].parameters(), **self.optim_config))
            # self.parallel_models.append(nn.DataParallel(_model))
        # self.models = [nn.DataParallel(_model) for _model in self.models]
        # training
        for e in range(self.global_epoch):
            for inputs, labels in tqdm(self.global_dataloader):
            # for inputs, labels in self.dataloader:
                # check len(inputs) = 2*num_modals
                data = {}
                if self.learn_strategy == 'X':
                    for _modal, _modal_index in zip(self.modals, self.modals_index):
                        # print (_modal, _modal_index) # RGB, 0 Depth, 1
                        data[_modal] = inputs[_modal_index * 2].float().to(self.device) # select every two elements, all weakly supervised models
                else:
                    raise Exception("Not implemented.")
                # data = [modala_w.float().to(self.device), modalb_w.float().to(self.device)]
                labels = labels.long().to(self.device)
                # iterate over every model with same data
                for _modal in self.modals:
                    optimizers[_modal].zero_grad()
                    outputs = self.models[_modal](data[_modal])
                    torch.cuda.synchronize()
                    loss = eval(self.criterion)()(outputs, labels)
                    loss.backward()
                    # gradient clipping
                    nn.utils.clip_grad_norm_(self.models[_modal].parameters(), max_norm=20, norm_type=2)
                    optimizers[_modal].step() 

                if self.device == "cuda": torch.cuda.empty_cache()

        for _model in self.models.values():
            _model.to("cpu")
        
        # ema training
        if self.use_ema:
            for _modal in self.modals:
                self.ema_models[_modal].update(self.models[_modal])
        
        message = f"[Round: {str(self._round).zfill(4)}] ...updated global model using the global holdout training dataset!"
        print(message); logging.info(message)
        del message; gc.collect()

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        for _modal in self.modals:
            if self.use_ema:
                self.eval_models[_modal] = self.ema_models[_modal].ema
            else:
                self.eval_models[_modal] = self.models[_modal]

        test_losses, test_accuracys = {}, {}
        for _modal in self.modals:
            self.models[_modal].eval()
            self.models[_modal].to(self.device)
            test_losses[_modal] = test_accuracys[_modal] =  0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader):
                data = {}
                if self.learn_strategy in ["N", "X"]:
                    for _modal, _modal_index in zip(self.modals, self.modals_index):
                        data[_modal] = inputs[_modal_index * 2].float().to(self.device) # inputs[0], inputs[2]
                else:
                    raise Exception("Not implemented.")
                labels = labels.long().to(self.device)
                # foreward pass through all models
                for _modal in self.modals:
                    outputs = self.eval_models[_modal](data[_modal])
                    test_losses[_modal] += eval(self.criterion)()(outputs, labels).item()
                
                    predicted = outputs.argmax(dim=1, keepdim=True)
                    test_accuracys[_modal] += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        # move all models to cpu
        for _model in self.eval_models.values():
            _model.to("cpu")

        for _modal in self.modals:
            test_losses[_modal] =  test_losses[_modal]  / len(self.test_dataloader)
            test_accuracys[_modal]  = test_accuracys[_modal] / len(self.test_data)

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Modal: {_modal}\
                \n\t=> Loss: {test_losses[_modal]:.4f}\
                \n\t=> Accuracy: {100. * test_accuracys[_modal]:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()

        return test_losses, test_accuracys

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1
            
            if self.learn_strategy == 'X':
                self.train_global_model()
                test_loss, test_accuracy = self.evaluate_global_model()
            if not self.is_central:
                sampled_client_indices, mixing_coefficients = self.train_federated_model()
                self.average_model(sampled_client_indices, mixing_coefficients)
                # TODO: ema model
                test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            for _modal in self.modals:
                self.writer.add_scalars(
                    'Loss/{}'.format(_modal),
                    {f"{self.learn_strategy}_[{self.clients_learn_strategy}]_{self.models[_modal].name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}": test_loss[_modal]},
                    self._round
                    )
                self.writer.add_scalars(
                    'Accuracy/{}'.format(_modal), 
                    {f"{self.learn_strategy}_[{self.clients_learn_strategy}]_{self.models[_modal].name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}": test_accuracy[_modal]},
                    self._round
                    )
        self.transmit_model()
