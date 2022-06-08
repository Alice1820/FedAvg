import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
import cv2

import torchvision.transforms as transforms
from scipy.stats import norm
from sklearn.utils import shuffle
from PIL import Image

import logging
from torchvideotransforms import video_transforms, volume_transforms, tensor_transforms

from .randaugment_vid import vidRandAugmentMC, labRandAugmentMC, vidRGB2Lab

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
# video transformation

# %% tools
def load_video(path, vid_len=32):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
    video = np.zeros((vid_len, height, width, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    # image list
    video = []
    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        if cap.isOpened() and fr_idx in taken:
            if frame is not None:
                # img = np.array(frame).astype(np.float32)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print (img_rgb)
                # img = Image.fromarray((img * 255).astype(np.uint8))
                # print (img)
                img = Image.fromarray(img_rgb.astype(np.uint8))
                video.append(img)
            np_idx += 1
    cap.release()
    # print (len(video))
    return video

# %%
class TemNormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb = sample['rgb']
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
        _rgb = []
        for i, f in enumerate(rgb):
            if i in indices_rgb:
                _rgb.append(f)
    
        return {'rgb': _rgb,
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
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        bias = int((1 - self.p_interval) * num_frames_rgb / 2)
        rgb = rgb[bias:num_frames_rgb - bias]

        return {'rgb': rgb,
                'label': label}


class TemAugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb = sample['rgb']
        label = sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        
        num_frames_rgb = len(rgb)
        begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
        rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb)]

        return {'rgb': rgb,
                'label': label}

def depth_transform(np_clip):
    ####### depth ######
    # histogram, fit the first frame of each video to a gauss distribution
    p_min = 500.
    p_max = 4500.
    np_clip[(np_clip < p_min)] = 0.0
    np_clip[(np_clip > p_max)] = 0.0
    frame = np_clip[(np_clip >= 500) * (np_clip <= 4500)] # range for skeleton detection
    mu, std = norm.fit(frame)
    # print (mu, std)
    # select certain range
    r_min = mu - std
    r_max = mu + std
    np_clip[(np_clip < r_min)] = 0.0
    np_clip[(np_clip > r_max)] = 0.0
    np_clip = np_clip - mu
    np_clip = np_clip / std # -3~3
    # repeat to BGR to fit pretrained resnet parameters
    # np_clip = np.repeat(np_clip[:, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3
    return np_clip
# %%
class NTU(Dataset):
    # RGBDI Dataset
    def __init__(self, root_dir='/data/NTU_RGBD_60',  # /data0/xifan/NTU_RGBD_60
                 split='cross_subject', # 40 subjects, 3 cameras
                 stage='train',
                 temTransform=None,
                 spaTransform=None,
                 vid_len=(8, 8),
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
    
        self.vid_len = vid_len

        self.rgb_list = []
        self.labels = []

        if split == 'cross_subject':
            if stage == 'train':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            elif stage == 'train100':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 2, 5, 9, 14, 3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
            elif stage == 'traindev':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 2, 5, 9, 14]
            elif stage == 'train5':
                subjects = [1]
            elif stage == 'train5b':
                subjects = [4]
            elif stage == 'train5c':
                subjects = [8]
            elif stage == 'train10':
                subjects = [1, 4]
            elif stage == 'train25':
                subjects = [1, 4, 8, 13]
            elif stage == 'train25b':
                subjects = [15, 16, 17, 18]
            elif stage == 'train25c':
                subjects = [19, 25, 27, 28]
            elif stage == 'train25d':
                subjects = [31, 34, 35, 38]
            elif stage == 'train50':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18]
            elif stage == 'train50':
                subjects = [1, 4, 8, 13, 15, 16, 17, 18]
            elif stage == 'trainexp':
                subjects = [1]
            elif stage == 'test':
                subjects = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
            elif stage == 'dev':  # smaller train datase for exploration
                subjects = [2, 5, 9, 14]
                # subjects = [2]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                          f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
        elif split == 'cross_view':
            if stage == 'train':
                cameras = [2, 3]
            elif stage == 'trainss':  # self-supervised training
                cameras = [2, 3]
                # cameras = [3]
            elif stage == 'trains':
                cameras = [2]
            elif stage == 'dev':
                cameras = [1]
            elif stage == 'test':
                cameras = [1]
            else:
                raise Exception('wrong stage: ' + stage)
            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if 
                            f.split(".")[-1] == "avi" and int(f[5:8]) in cameras]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if int(f[5:8]) in cameras]
        else:
            raise Exception('wrong mode: ' + args.mode)

        self.rgb_list, self.labels = shuffle(self.rgb_list, self.labels)
        self.root_dir = root_dir
        self.stage = stage
        self.args = args
        self.temTransform = temTransform
        self.spaTransform = spaTransform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        label = self.labels[idx]
        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)

        sample = {'rgb': video, 'label': label - 1}
        if self.temTransform:
            sample = self.temTransform(sample)
        # print (sample['rgb'][0].size, len(sample['rgb']))
        if self.spaTransform:
            sample['rgb'] = self.spaTransform(sample['rgb'])
        # print (np.max(np.array(depth[0])), 'np max')
        # print (np.min(np.array(depth[0])), 'np min')
        # print (np.mean(np.array(depth[0])), 'np mean')
        # print (np.max(np.array(depth[0])), 'np max')
        # print (np.min(np.array(depth[0])), 'np min')
        # print (np.mean(np.array(depth[0])), 'np mean')
        # print (torch.max(sample['dep']), 'torch max')
        # print (torch.min(sample['dep']), 'torch min')
        # print (torch.mean(sample['dep']), 'torch mean')
        return sample['rgb'], sample['label']

def get_ntu(args):
    if args.view == 'RGB':
        mean = imagenet_mean
        std = imagenet_std
        div_255 = True
        rgb2lab = False
        spaTransform_unlabeled = vidTransformFixMatch(mean=mean, std=std, rgb2lab=rgb2lab, div_255=div_255)
    elif args.view == 'Lab':
        mean = lab_mean
        std = lab_std
        div_255 = False
        rgb2lab = True
        spaTransform_unlabeled = labTransformFixMatch(mean=mean, std=std, rgb2lab=rgb2lab, div_255=div_255)
    elif args.view == 'RGBD':
        pass

    spaTransform_labeled = transforms.Compose([
        # extra augmentation
        # video_transforms.RandomGrayscale(),
        # video_transforms.RandomRotation(30),
        # video_transforms.ColorJitter(0.2, 0.2, 0.2),
        # regular augmentation
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomCrop((224, 224)),
        vidRGB2Lab(rgb2lab),
        volume_transforms.ClipToTensor(div_255=div_255),
        video_transforms.Normalize(mean=mean, std=std)
    ])
    spaTransform_val = transforms.Compose([
        video_transforms.CenterCrop((224, 224)),
        vidRGB2Lab(rgb2lab),
        # volume_transforms.ClipToTensor(div_255=False),
        volume_transforms.ClipToTensor(div_255=div_255),
        video_transforms.Normalize(mean=mean, std=std)
    ])
    temTransform_labeled = transforms.Compose([TemAugCrop(), TemNormalizeLen((args.num_segments, args.num_segments))])
    temTransform_val = transforms.Compose([TemCenterCrop(), TemNormalizeLen((args.num_segments, args.num_segments))])

    train_labeled_dataset = NTU(stage=args.dataset, temTransform=temTransform_labeled, spaTransform=spaTransform_labeled)
    train_unlabeled_dataset = NTU(stage='traindev', temTransform=temTransform_labeled, spaTransform=spaTransform_unlabeled)    
    eval_dataset = NTU(stage='dev', temTransform=temTransform_val, spaTransform=spaTransform_val)
    test_dataset = NTU(stage='test', temTransform=temTransform_val, spaTransform=spaTransform_val)

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
            vidRGB2Lab(rgb2lab),
            # volume_transforms.ClipToTensor(div_255=False),
            volume_transforms.ClipToTensor(div_255=div_255),
            video_transforms.Normalize(mean=mean, std=std)
        ])
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
class labTransformFixMatch(object):
    def __init__(self, mean, std, rgb2lab, div_255):
        self.weak = transforms.Compose([
            # regular augmentation
            video_transforms.RandomHorizontalFlip(),
            video_transforms.RandomCrop((224, 224))])
        self.strong = transforms.Compose([
            # extra augmentation
            video_transforms.RandomRotation(30),
            video_transforms.ColorJitter(0.2, 0.2, 0.2),
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
