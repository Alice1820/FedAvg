import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


#######################
# Criterion #
#######################
class SingleFixmatchLoss(torch.nn.CrossEntropyLoss):
    r"""This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
        - Output: If reduction is 'none', same shape as the target. Otherwise, scalar.

        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}

    Examples::

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(SingleFixmatchLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input_w: Tensor, input_s: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = transforms.ToTensor()
        
        # prepare raw training & test datasets
        training_dataset = datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
    
    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]
    else:
        # sort data by labels
        sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
        training_inputs = training_dataset.data[sorted_indices]
        training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

        # partition data into shards first
        shard_size = len(training_dataset) // num_shards #300
        shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
        shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

        # sort the list to conveniently assign samples to each clients from at least two classes
        shard_inputs_sorted, shard_labels_sorted = [], []
        for i in range(num_shards // num_categories):
            for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                shard_inputs_sorted.append(shard_inputs[i + j])
                shard_labels_sorted.append(shard_labels[i + j])
                
        # finalize local datasets by assigning shards to each client
        shards_per_clients = num_shards // num_clients
        local_datasets = [
            CustomTensorDataset(
                (
                    torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                    torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                ),
                transform=transform
            ) 
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ]
    return local_datasets, test_dataset
