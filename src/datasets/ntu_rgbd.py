from torchvideotransforms import video_transforms, volume_transforms, tensor_transforms
import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from scipy.stats import norm
import logging

from .randaugment_vid import vidRandAugmentMC, vidRGB2Lab
from sklearn.utils import shuffle
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
lab_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
lab_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
dep_mean = [2500]
dep_std = [0.5]

# video transformation
# %% tools
def load_video(path, vid_len=32):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    # image list
    video = []
    # np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        # if cap.isOpened() and fr_idx in taken:
        if cap.isOpened() and fr_idx in taken:
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb.astype(np.uint8))
                video.append(img)
            # np_idx += 1
    cap.release()

    return video

# %% tools
def load_video(path, vid_len=32):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # image list
    video = []
    # np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        # if cap.isOpened() and fr_idx in taken:
        if cap.isOpened():
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb.astype(np.uint8))
                video.append(img)
            # np_idx += 1
    cap.release()

    return video

def load_depth(path, vid_len=32):
    img_list = sorted(os.listdir(path))
    num_frames = len(img_list)
    # Init the numpy array
    taken = np.linspace(0, num_frames, vid_len).astype(int)
    
    video = []
    np_idx = 0
    for fr_idx in range(num_frames):
        if fr_idx in taken: # 24 frames
            img_path = os.path.join(path, img_list[fr_idx])
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) # 16bit
            if not img is None: # skip empty frame
                # img = cv2.resize(img, dim) # 310*256
                img = Image.fromarray(img.astype(np.uint16))
                video.append(img)
            np_idx += 1
    return video
    
# %%
class TemNormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
        _rgb = []
        for i, f in enumerate(rgb):
            if i in indices_rgb:
                _rgb.append(f)

        num_frames_dep = len(dep)
        indices_dep = np.linspace(0, num_frames_dep - 1, self.vid_len[0]).astype(int)
        _dep = []
        for i, f in enumerate(dep):
            if i in indices_dep:
                _dep.append(f)

        return {'rgb': _rgb,
                'dep': _dep,
                'label': label}


def interpole(data, cropped_length, vid_len):
    C, T, V, M = data.shape
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, :, :, None]
    data = F.interpolate(data, size=(vid_len, 1), mode='bilinear', align_corners=False).squeeze(dim=3).squeeze(dim=0)
    data = data.contiguous().view(C, V, M, vid_len).permute(0, 3, 1, 2).contiguous().numpy()
    return data

# %%

class TemCenterCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.9):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        bias = int((1 - self.p_interval) * num_frames_rgb / 2)
        rgb = rgb[bias:num_frames_rgb - bias]

        num_frames_dep = len(dep)
        bias = int((1 - self.p_interval) * num_frames_dep / 2)
        dep = dep[bias:num_frames_dep - bias]

        return {'rgb': rgb,
                'dep': dep,
                'label': label}


class TemAugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        label = sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        
        num_frames_rgb = len(rgb)
        begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
        rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb)]

        num_frames_dep = len(dep)
        begin_dep = (num_frames_dep - int(num_frames_dep * ratio)) // 2
        dep = dep[begin_dep:(num_frames_dep - begin_dep)]

        return {'rgb': rgb,
                'dep': dep,
                'label': label}

def depth_transform(np_clip):
    ####### depth ######
    # histogram, fit the first frame of each video to a gauss distribution
    p_min = 500.
    p_max = 4500.
    np_clip[(np_clip < p_min)] = 0.0
    np_clip[(np_clip > p_max)] = 0.0
    return np_clip

# %%
class NTU_X(Dataset):
    r""" Supervised RGBD Dataset
        Args:
            root_dir (string): Directory where data is.
            subjects (list): List of subject number
            
    """
    def __init__(self, root_dir='/data0/xfzhang/data/NTU_RGBD_60',  # /data0/xifan/NTU_RGBD_60
                 stage='train',
                 subjects=[],
                 temTransform=None,
                 spaTransform=None,
                 depTransform=None,
                 vid_len=(8, 8)):
        # check subjects list is not empty
        assert len(subjects) != 0

        # basename
        self.basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
        self.basename_dep = os.path.join(root_dir, 'nturgbd_depth_masked/dep_310*256') 
    
        self.vid_len = vid_len # (8, 8)
        self.subjects = subjects
        self.rgb_list = []
        self.dep_list = []
        self.labels = []

        self.rgb_list += [os.path.join(self.basename_rgb, f) for f in sorted(os.listdir(self.basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in self.subjects]
        self.dep_list += [os.path.join(self.basename_dep, f) for f in sorted(os.listdir(self.basename_dep)) if 
                        int(f[9:12]) in self.subjects]
        self.labels += [int(f[17:20]) for f in sorted(os.listdir(self.basename_rgb)) if
                    f.split(".")[-1] == "avi" and int(f[9:12]) in self.subjects]

        self.rgb_list, self.dep_list, self.labels = shuffle(self.rgb_list, self.dep_list, self.labels)
        self.temTransform = temTransform
        self.spaTransform = spaTransform
        self.depTransform = depTransform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        deppath = self.dep_list[idx]
        label = self.labels[idx]
        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        depth = load_depth(deppath)

        sample = {'rgb': video, 'dep': depth,  'label': label - 1}
        if self.temTransform:
            sample = self.temTransform(sample)
        # print (sample['rgb'][0].size, len(sample['rgb']))
        if self.spaTransform:
            sample['rgb'] = self.spaTransform(sample['rgb'])
        if self.depTransform:
            sample['dep'] = self.depTransform(sample['dep'])
        return sample['rgb'], sample['dep'], sample['label']

class NTU_U(NTU_X):
    r""" Unsupervised RGBD Dataset
        Args:
            root_dir (string): Directory where data is.
            subjects (list): List of subject number

    """
    def __init__(self, root_dir='/data0/xfzhang/data/NTU_RGBD_60',  # /data0/xifan/NTU_RGBD_60
                 subjects=[],
                 temTransform=None,
                 spaTransform_w=None,
                 spaTransform_s=None,
                 depTransform_w=None,
                 depTransform_s=None,
                 vid_len=(8, 8)):
        super(NTU_X, self).__init__(root_dir=root_dir, 
                                    subjects=subjects, 
                                    temTransform=temTransform, 
                                    spaTransform=spaTransform_w, 
                                    depTransform=depTransform_w
                                    )
        self.spaTransform_s = spaTransform_s
        self.depTransform_s = depTransform_s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        deppath = self.dep_list[idx]
        label = self.labels[idx]
        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        depth = load_depth(deppath)

        sample = {'rgb': video, 'dep': depth, 'label': label - 1}
        if self.temTransform:
            sample = self.temTransform(sample)
        # print (sample['rgb'][0].size, len(sample['rgb']))
       #  print (sample['rgb'].size(), 'rgb')
        samples = {}
        samples['label'] = sample['label']
        samples['rgb_w'] = self.spaTransform(sample['rgb'])
        # print (sample['rgb_w'].size(), 'rgb_w')
        samples['rgb_s'] = self.spaTransform_s(sample['rgb'])
        # print (sample['rgb_s'].size(), 'rgb_s')
        samples['dep_w'] = self.depTransform(sample['dep'])
        # print (sample['dep_w'].size(), 'dep_w')
        samples['dep_s'] = self.depTransform_s(sample['dep'])
        # print (sample['dep_s'].size(), 'dep_s')
     
        return samples['rgb_w'], samples['rgb_s'], samples['dep_w'], samples['dep_s']


def get_ntu_rgbd(args):
    # RGB
    spaTransform_unlabeled_w = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            volume_transforms.ClipToTensor(div_255=True),
            video_transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
    spaTransform_unlabeled_s = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            vidRandAugmentMC(n=2, m=10),
            volume_transforms.ClipToTensor(div_255=True),
            video_transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
    spaTransform_labeled = transforms.Compose([
        # extra augmentation
        # video_transforms.RandomGrayscale(),
        # video_transforms.RandomRotation(30),
        # video_transforms.ColorJitter(0.2, 0.2, 0.2),
        # regular augmentation
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomCrop((224, 224)),
        volume_transforms.ClipToTensor(div_255=True),
        video_transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    spaTransform_val = transforms.Compose([
        video_transforms.CenterCrop((224, 224)),
        volume_transforms.ClipToTensor(div_255=True),
        video_transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Depth
    depTransform_unlabeled_w = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            volume_transforms.ClipToTensor(channel_nb=1, div_255=False),
            video_transforms.Normalize(mean=dep_mean, std=dep_std),
            ])

    depTransform_unlabeled_s = transforms.Compose([
            # extra augmentation
            video_transforms.RandomRotation(30),
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            volume_transforms.ClipToTensor(channel_nb=1, div_255=False),
            video_transforms.Normalize(mean=dep_mean, std=dep_std),
            ])
    depTransform_labeled = transforms.Compose([
        # extra augmentation
        # regular augmentation
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomCrop((224, 224)),
        volume_transforms.ClipToTensor(channel_nb=1, div_255=False),
        video_transforms.Normalize(mean=dep_mean, std=dep_std)
    ])
    depTransform_val = transforms.Compose([
        video_transforms.CenterCrop((224, 224)),
        volume_transforms.ClipToTensor(channel_nb=1, div_255=False),
        video_transforms.Normalize(mean=dep_mean, std=dep_std)
    ])

    temTransform_labeled = transforms.Compose([TemAugCrop(), TemNormalizeLen((args.num_segments, args.num_segments))])
    temTransform_val = transforms.Compose([TemCenterCrop(), TemNormalizeLen((args.num_segments, args.num_segments))])

    train_labeled_dataset = NTU_x(root_dir=args.data_folder, stage=args.dataset, temTransform=temTransform_labeled, spaTransform=spaTransform_labeled, depTransform=depTransform_labeled)
    train_unlabeled_dataset = NTU_u(root_dir=args.data_folder, stage='train', temTransform=temTransform_labeled, spaTransform_w=spaTransform_unlabeled_w, spaTransform_s=spaTransform_unlabeled_s, depTransform_w=depTransform_unlabeled_w, depTransform_s=depTransform_unlabeled_s)    
    eval_dataset = NTU_x(root_dir=args.data_folder, stage='dev', temTransform=temTransform_val, spaTransform=spaTransform_val, depTransform=depTransform_val)
    test_dataset = NTU_x(root_dir=args.data_folder, stage='test', temTransform=temTransform_val, spaTransform=spaTransform_val, depTransform=depTransform_val)

    print('number of labeled train: {}'.format(len(train_labeled_dataset)))
    print('number of unlabeled train: {}'.format(len(train_unlabeled_dataset)))
    print('number of val: {}'.format(len(eval_dataset)))
    print('number of test: {}'.format(len(test_dataset)))

    return train_labeled_dataset, train_unlabeled_dataset, eval_dataset, test_dataset


class vidTransformFixMatch(object):
    def __init__(self, mean, std, rgb2lab, div_255):
        self.weak = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224))])
        self.strong = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            vidRandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            # volume_transforms.ClipToTensor(div_255=False),
            volume_transforms.ClipToTensor(div_255=div_255),
            video_transforms.Normalize(mean=mean, std=std)
        ])
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class depTransformFixMatch(object):
    def __init__(self, mean, std, rgb2lab, div_255):
        self.weak = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224))])
        self.strong = transforms.Compose([
            # extra augmentation
            video_transforms.RandomRotation(30),
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224)),
            # labRandAugmentMC(n=2, m=10),
            ])
        self.normalize = transforms.Compose([
            vidRGB2Lab(rgb2lab),
            # volume_transforms.ClipToTensor(div_255=False),
            volume_transforms.ClipToTensor(div_255=div_255),
            video_transforms.Normalize(mean=mean, std=std)
        ])
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# %%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="store",
                        dest="folder",
                        help="Path to the data",
                        default="NTU")
    parser.add_argument('--outputdir', type=str, help='output base dir', default='checkpoints/')
    parser.add_argument('--datadir', type=str, help='data directory', default='NTU')
    parser.add_argument("--j", action="store", default=12, dest="num_workers", type=int,
                        help="Num of workers for dataset preprocessing.")

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 8), dest="vid_len", type=int, help="length of video")
    parser.add_argument('--modality', type=str, help='modality: rgb, skeleton, both', default='rgb')
    parser.add_argument("--hp", action="store_true", default=False, dest="hp", help="random search on hp")
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument('--num_classes', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument("--clip", action="store", default=None, dest="clip", type=float,
                        help="if using gradient clipping")
    parser.add_argument("--lr", action="store", default=0.001, dest="learning_rate", type=float,
                        help="initial learning rate")
    parser.add_argument("--lr_decay", action="store_true", default=False, dest="lr_decay",
                        help="learning rate exponential decay")
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")
    parser.add_argument('--epochs', type=int, help='training epochs', default=10)

    args = parser.parse_args()
