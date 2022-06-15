import copy
import gc
from math import floor
import pickle
import logging
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .utils import *
from .criterion import *

logger = logging.getLogger(__name__)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, local_modals, local_modals_index, local_learn_strategy, device, global_dataloader=None):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        # device_id = self.id % torch.cuda.device_count()
        # self.device = torch.device("cuda: {}".format(device_id + 1))
        # self.device = torch.device("cuda: {}".format(device_id + 1))
        # self.__models = []
        self.__models = {}
        self.modals = local_modals
        self.modals_index = local_modals_index
        self.learn_strategy = local_learn_strategy
        self.fixmatch = FixMatchLoss()
        self.multimatch = MultiMatchLoss()
        self.global_dataloader =global_dataloader
    
    @property
    def models(self):
        """Local model getter for parameter aggregation."""
        # print ("called models.property")
        return self.__models
    
    @models.setter
    def models(self, models):
        """Local model setter for passing globally aggregated model parameters."""
        # iterate over models
        # for __model, model in zip(self.__models, models):
            # __model = model
        # print ("called models.setter")
        self.__models = models


    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.mu = client_config["mu"]
        self.batch_size = client_config["batch_size"]
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size*self.mu, shuffle=True, num_workers=8)
        # self.local_epoch = client_config["num_local_epochs"] * int(MAX_INS / len(self.data))
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self):
        """Update local model using local dataset."""
        optimizers = {}
        for _modal in self.models.keys():
            self.models[_modal].train()
            self.models[_modal].to(self.device)
            optimizers[_modal] = (eval(self.optimizer)(self.models[_modal].parameters(), **self.optim_config))
            # self.parallel_models.append(nn.DataParallel(_model))
        # self.models = [nn.DataParallel(_model) for _model in self.models]
        # training
        loss_meter = {}
        loss_x_meter = {}
        loss_u_meter = {}
        mask_meter = {}
        for _modal in self.modals:
            loss_meter[_modal] = AverageMeter()
            loss_x_meter[_modal] = AverageMeter()
            loss_u_meter[_modal] = AverageMeter()
            mask_meter[_modal] = AverageMeter()
        for e in range(self.local_epoch):
            p_bar = tqdm(range(len(self.dataloader)))
            for inputs, labels in self.dataloader:
            # for inputs, labels in self.dataloader:
                # check len(inputs) = 2*num_modals
                data = {}
                if self.learn_strategy == 'X':
                    for _modal, _modal_index in zip(self.modals, self.modals_index):
                        data[_modal] = inputs[_modal_index * 2].float().to(self.device)
                elif self.learn_strategy in ['F', 'M']:
                    for _modal, _modal_index in zip(self.modals, self.modals_index):
                        data[_modal] = interleave(
                        torch.cat((inputs[_modal_index * 2], inputs[_modal_index * 2 + 1])), 2).to(self.device)
                else:
                    raise Exception("Not implemented.")
                # data = [modala_w.float().to(self.device), modalb_w.float().to(self.device)]
                labels = labels.long().to(self.device)
                # iterate over every model with same data
                outputs = {}
                loss = {}
                loss_x = {}
                loss_u = {}
                mask = {}
                outputs = {}
                for _modal in self.modals:
                    loss_x[_modal] = loss_u[_modal] = 0
                    # forward
                    optimizers[_modal].zero_grad()
                    outputs[_modal] = self.models[_modal](data[_modal])
                    torch.cuda.synchronize()
                    # de_interleave outputs
                    outputs[_modal] = de_interleave(outputs[_modal], 2)
                    outputs[_modal + 'w'], outputs[_modal + 's'] = outputs[_modal].chunk(2)
                        # print (outputs[_modal + 'w'].shape) # torch.Size([16, 60]
                    if self.learn_strategy == 'X':
                        loss_x[_modal] = eval(self.criterion)()(outputs[_modal], labels)
                        # update average meter
                        loss_x_meter[_modal].update(loss_x[_modal])
                    if self.learn_strategy == "F":
                        mask[_modal], loss_u[_modal] = self.fixmatch(outputs[_modal + 'w'], outputs[_modal + 's'])
                        # update average meter
                        mask_meter[_modal].update(mask[_modal])
                        loss_u_meter[_modal].update(loss_u[_modal])
                # jointly compute multimatch loss
                if self.learn_strategy == "M":
                    mask[self.modals[0]], mask[self.modals[1]], loss_u[self.modals[0]], loss_u[self.modals[1]] = \
                            self.multimatch(outputs[self.modals[0] + 'w'], outputs[self.modals[0] + 's'], 
                            outputs[self.modals[1] + 'w'], outputs[self.modals[1] + 's'])
                    # update average meter
                    mask_meter[self.modals[0]].update(mask[self.modals[0]])
                    mask_meter[self.modals[1]].update(mask[self.modals[1]])
                    loss_u_meter[self.modals[0]].update(loss_u[self.modals[0]])
                    loss_u_meter[self.modals[1]].update(loss_u[self.modals[1]])

                # backward
                for _modal in self.modals:
                    loss[_modal] = loss_x[_modal] + loss_u[_modal]
                    loss[_modal].backward()
                    # gradient clipping
                    nn.utils.clip_grad_norm_(self.models[_modal].parameters(), max_norm=20, norm_type=2)
                    optimizers[_modal].step()

                descrip = "Train Client Id: {:3} ".format(self.id)
                for _modal in self.modals:
                    # descrip += "LX_{}: {:.2f} LU_{}: {:.2f} M_{}: {:.2f} ".format(_modal, loss_x[_modal], _modal, loss_u[_modal], _modal, mask[_modal])
                    descrip += "LX_{}: {:.2f} LU_{}: {:.2f} M_{}: {:.2f} ".format(_modal, loss_x_meter[_modal].avg, _modal, loss_u_meter[_modal].avg, _modal, mask_meter[_modal].avg)
                p_bar.set_description(descrip)
                p_bar.update()

                if self.device == "cuda": torch.cuda.empty_cache()

            p_bar.close()
        for _model in self.models.values():
            _model.to("cpu")

    def client_joint_update(self):
        """Update local model using local dataset."""
        optimizers = {}
        for _modal in self.models.keys():
            self.models[_modal].train()
            self.models[_modal].to(self.device)
            optimizers[_modal] = (eval(self.optimizer)(self.models[_modal].parameters(), **self.optim_config))
            # self.parallel_models.append(nn.DataParallel(_model))
        # self.models = [nn.DataParallel(_model) for _model in self.models]
        # training
        iterator = iter(self.dataloader)
        loss_meter = {}
        loss_x_meter = {}
        loss_u_meter = {}
        mask_meter = {}
        for _modal in self.modals:
            loss_meter[_modal] = AverageMeter()
            loss_x_meter[_modal] = AverageMeter()
            loss_u_meter[_modal] = AverageMeter()
            mask_meter[_modal] = AverageMeter()
        for _ in range(self.local_epoch):
            # initialize
            # p_bar = tqdm(range(len(self.dataloader)))
            p_bar = tqdm(range(len(self.global_dataloader)))

            # for _inputs_u in self.dataloader:
            for inputs_x, labels in self.global_dataloader:
                # inputs_u, _ = _inputs_u
                try:
                    # inputs_x, labels = global_iter.next()
                    inputs_u, _ = iterator.next()
                except:
                    # global_iter = iter(self.global_dataloader)
                    # inputs_x, labels = global_iter.next()
                    iterator = iter(self.dataloader)
                    inputs_u, _ = iterator.next()
            # for inputs, labels in self.dataloader:
                # check len(inputs) = 2*num_modals
                # print (torch.min(modala_w), torch.max(modala_w))
                data = {}
                for _modal, _modal_index in zip(self.modals, self.modals_index):
                    # print (inputs_x[_modal_index * 2].shape)
                    # print (inputs_u[_modal_index * 2].shape)
                    # print (inputs_u[_modal_index * 2 + 1].shape)
                    data[_modal] = interleave(torch.cat((inputs_x[_modal_index * 2], inputs_u[_modal_index * 2], inputs_u[_modal_index * 2 + 1])), 2*self.mu+1).to(self.device)
                labels = labels.long().to(self.device)
                # iterate over every model with same data
                loss = {}
                loss_x = {}
                loss_u = {}
                mask = {}
                outputs = {}
                for _modal in self.modals:
                    optimizers[_modal].zero_grad()
                    # loss[_modal] = loss_x[_modal] = loss_u[_modal]= mask[_modal] = 0
                    # forward
                    outputs[_modal] = self.models[_modal](data[_modal])
                    torch.cuda.synchronize()
                    outputs[_modal] = de_interleave(outputs[_modal], 2*self.mu+1)
                    outputs[_modal + 'x'] = outputs[_modal][:self.batch_size]
                    outputs[_modal + 'w'], outputs[_modal + 's'] = outputs[_modal][self.batch_size:].chunk(2)
                    # print (outputs[_modal + 'w'].shape) # torch.Size([16, 60]
                    loss_x[_modal] = eval(self.criterion)()(outputs[_modal + 'x'], labels)
                    # update average meter
                    loss_x_meter[_modal].update(loss_x[_modal])
                    if self.learn_strategy == "F":
                        mask[_modal], loss_u[_modal] = self.fixmatch(outputs[_modal + 'w'], outputs[_modal + 's'])
                        # update average meter
                        mask_meter[_modal].update(mask[_modal])
                        loss_u_meter[_modal].update(loss_u[_modal])
                if self.learn_strategy == "M":
                    mask[self.modals[0]], mask[self.modals[1]], loss_u[self.modals[0]], loss_u[self.modals[1]] = \
                                                    self.multimatch(outputs[self.modals[0] + 'w'], outputs[self.modals[0] + 's'], 
                                                    outputs[self.modals[1] + 'w'], outputs[self.modals[1] + 's'])
                    # update average meter
                    mask_meter[self.modals[0]].update(mask[self.modals[0]])
                    mask_meter[self.modals[1]].update(mask[self.modals[1]])
                    loss_u_meter[self.modals[0]].update(loss_u[self.modals[0]])
                    loss_u_meter[self.modals[1]].update(loss_u[self.modals[1]])
                    

                # print (loss)

                for _modal in self.modals:
                    loss[_modal] = loss_x[_modal] + loss_u[_modal]
                    loss[_modal].backward()
                    # gradient clipping
                    nn.utils.clip_grad_norm_(self.models[_modal].parameters(), max_norm=20, norm_type=2)
                    optimizers[_modal].step()

                # display
                descrip = "Train Client Id: {:3} ".format(self.id)
                for _modal in self.modals:
                    descrip += "LX_{}: {:.2f} LU_{}: {:.2f} M_{}: {:.2f} ".format(_modal, loss_x_meter[_modal].avg, _modal, loss_u_meter[_modal].avg, _modal, mask_meter[_modal].avg)
                p_bar.set_description(descrip)
                p_bar.update()
                if self.device == "cuda": torch.cuda.empty_cache()
            p_bar.close()
        for _model in self.models.values():
            _model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        for _model in self.models.values:
            _model.eval()
            _model.to(self.device)

        test_losses, corrects = [0 for _ in self.models], [0 for _ in self.models]
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                data = {}
                if self.learn_strategy == 'X':
                    for _modal, _modal_index in zip(self.modals, self.modals_index):
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

        test_losses = [test_loss / len(self.dataloader) for test_loss in test_losses]
        test_accuracys = [correct / len(self.data) for correct in corrects]

        # message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!"
        # print(message, flush=True); logging.info(message)
        # del message; gc.collect()

        for test_loss, test_accuracy in zip(test_losses, test_accuracys):
            message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
            print(message, flush=True); logging.info(message)
            del message; gc.collect()

        # print (test_losses, test_accuracys)
        return test_losses, test_accuracys
