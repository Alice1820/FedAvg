import copy
import gc
from math import floor
import pickle
import logging
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

MAX_INS = 5760 * 2

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
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        # device_id = self.id % torch.cuda.device_count()
        # self.device = torch.device("cuda: {}".format(device_id + 1))
        # self.device = torch.device("cuda: {}".format(device_id + 1))
        self.__models = []
    
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
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True, num_workers=4)
        # self.local_epoch = client_config["num_local_epochs"] * int(MAX_INS / len(self.data))
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self, display=False):
        """Update local model using local dataset."""
        optimizers = []
        self.parallel_models = []
        for _model in self.models:
            _model.train()
            _model.to(self.device)
            optimizers.append(eval(self.optimizer)(_model.parameters(), **self.optim_config))
            # self.parallel_models.append(nn.DataParallel(_model))
        # self.models = [nn.DataParallel(_model) for _model in self.models]
        # training
        for e in range(self.local_epoch):
            for idx, (inputs, labels) in enumerate(self.dataloader):
            # for inputs, labels in self.dataloader:
                # check len(inputs) = 2*num_modals
                modala_w, modala_s, modalb_w, modalb_s = inputs
                # print (torch.min(modala_w), torch.max(modala_w))
                # print (torch.min(modalb_w), torch.max(modalb_w))
                data = [modala_w.float().to(self.device), modalb_w.float().to(self.device)]
                labels = labels.long().to(self.device)
                # iterate over every model with same data
                for _optimizer, _model, _data in zip(optimizers, self.models, data):
                    _optimizer.zero_grad()
                    outputs = _model(_data)
                    torch.cuda.synchronize()
                    loss = eval(self.criterion)()(outputs, labels)

                    # display
                    if display: print("Id: {} Epoch: {} / {} Iter: {} / {} Loss: {:.2f}".format(self.id, e, self.local_epoch, idx, len(self.dataloader), loss.data.cpu().numpy()), end='\r')

                    loss.backward()
                    _optimizer.step() 

                if self.device == "cuda": torch.cuda.empty_cache()
        for _model in self.models:
            _model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        for _model in self.models:
            _model.eval()
            _model.to(self.device)

        test_losses, corrects = [0 for _ in self.models], [0 for _ in self.models]
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                modala_w, modala_s, modalb_w, modalb_s = inputs
                labels = labels.long().to(self.device)
                data = [modala_w.float().to(self.device), modalb_w.float().to(self.device)]

                # foreward pass through all models
                for idx, _model in enumerate(self.models):
                    outputs = _model(data[idx])
                    test_losses[idx] += eval(self.criterion)()(outputs, labels).item()
                
                    predicted = outputs.argmax(dim=1, keepdim=True)
                    corrects[idx] += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        # move all models to cpu
        for _model in self.models:
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
