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
        self.model_config = model_config

        self.clients_modals = clients_config["clients_modals"]
        self.clients_learn_strategy = clients_config["clients_learn_strategy"]

        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

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
            self.models[modal] = model
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized backbone {backbone_config}"
            print (message); gc.collect()
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

        message = f"[Round: {str(self._round).zfill(4)}] Created server: DATA {len(global_dataset)}, MODALS: {self.modals},  LEARN: {self.learn_strategy}!"
        print(message); logging.info(message)
        del message; gc.collect()

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets, local_modals, local_modals_index, local_learn_strategy)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.global_dataloader = DataLoader(global_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
        self.modals_index = range(len(self.modals))
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config,
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets, local_modals, local_modals_index, local_learn_strategy):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in enumerate(local_datasets):
            # ***create clients
            client = Client(client_id=k, local_data=dataset, local_modals=local_modals[k], local_modals_index=local_modals_index[k], 
                            local_learn_strategy=local_learn_strategy[k], device=self.device)
            clients.append(client)
            message = f"[Round: {str(self._round).zfill(4)}] Created client: DATA {len(dataset)}, MODALS: {local_modals[k]},  LEARN: {local_learn_strategy[k]}!"
            print(message); logging.info(message)
            del message; gc.collect()

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False): #
                # print (client.modals)
                for _modal in client.modals:
                    # print (type(_modal), type(client.models), type(self.models))
                    client.models[_modal] = copy.deepcopy(self.models[_modal])
                    # client.models[_modal] = self.models[_modal]
                    # client.models = [nn.DataParallel(_model) for _model in client.models]
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

            for k, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                for _modal in self.clients[idx].modals:
                    self.clients[idx].models[_modal] = copy.deepcopy(self.models[_modal])
                    # client.models = [nn.DataParallel(_model) for _model in client.models]
                    self.clients[idx].models[_modal] = nn.DataParallel(self.models[_modal])
            # # move all models to cpu
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

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=True):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

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
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                # print ("check length of client models", len(self.clients[idx].models)) # 2
                # parallel training
                # local_weights = self.clients[idx].models[model_index].state_dict()
                if _modal in self.clients[idx].modals:
                    present_flag = True
                    local_weights = self.clients[idx].models[_modal].module.state_dict()
                    # print (local_weights.keys())
                    for key in self.models[_modal].state_dict().keys():
                        # print (local_weights.keys())
                        # if it == 0:
                        if key not in averaged_weights.keys(): # is empty
                            averaged_weights[key] = coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += coefficients[it] * local_weights[key]
            # print (_modal)
            if averaged_weights: self.models[_modal].load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
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
        displays = [False for _ in sampled_client_indices]
        displays[0] = True
        map_configs = list(itertools.product(sampled_client_indices, displays))
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
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
    
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
                    optimizers[_modal].step() 

                if self.device == "cuda": torch.cuda.empty_cache()
        for _model in self.models.values():
            _model.to("cpu")
        
        message = f"[Round: {str(self._round).zfill(4)}] ...updated global model using the global holdout training dataset!"
        print(message); logging.info(message)
        del message; gc.collect()

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        for _model in self.models.values():
            _model.eval()
            _model.to(self.device)

        test_losses, corrects = [0 for _ in self.models.keys()], [0 for _ in self.models.keys()]
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader):
                data = {}
                if self.learn_strategy in ["N", "X"]:
                    for _modal, _modal_index in zip(self.modals, range(len(self.modals))):
                        data[_modal] = inputs[_modal_index * 2].float().to(self.device)
                else:
                    raise Exception("Not implemented.")
                labels = labels.long().to(self.device)
                # foreward pass through all models
                for idx, _modal in enumerate(self.modals):
                    outputs = self.models[_modal](data[_modal])
                    test_losses[idx] += eval(self.criterion)()(outputs, labels).item()
                
                    predicted = outputs.argmax(dim=1, keepdim=True)
                    corrects[idx] += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        # move all models to cpu
        for _model in self.models.values():
            _model.to("cpu")

        test_losses = [test_loss / len(self.test_dataloader) for test_loss in test_losses]
        test_accuracys = [correct / len(self.data) for correct in corrects]

        return test_losses, test_accuracys

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            if self.learn_strategy != 'N':
                self.train_global_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            for model_idx, _model in enumerate(self.models.values()):
                self.writer.add_scalars(
                    'Loss',
                    {f"[{self.dataset_name}]_{_model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss[model_idx]},
                    self._round
                    )
                self.writer.add_scalars(
                    'Accuracy', 
                    {f"[{self.dataset_name}]_{_model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy[model_idx]},
                    self._round
                    )
                
                message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                    \n\t[Server] ...finished evaluation!\
                    \n\t=> Modal: {self.modals[model_idx]}\
                    \n\t=> Loss: {test_loss[model_idx]:.4f}\
                    \n\t=> Accuracy: {100. * test_accuracy[model_idx]:.2f}%\n"            
                print(message); logging.info(message)
                del message; gc.collect()
        self.transmit_model()
