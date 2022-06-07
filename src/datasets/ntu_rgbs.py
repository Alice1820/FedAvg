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
from .tools import valid_crop_resize, rand_rotate, random_choose, random_move, random_shift, valid_choose

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
# tools
def load_video(path, vid_len=32):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
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

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def get_3D_skeleton(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

# %%
class TemNormalizeLen(object):
    """ Return a normalized number of frames as list. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb = sample['rgb']
        skel = sample['skel']
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
        _rgb = []
        for i, f in enumerate(rgb):
            if i in indices_rgb:
                _rgb.append(f)

        return {'rgb': _rgb,
                'skel': skel,
                'label': label}

# %%

class TemCenterCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.9):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb = sample['rgb']
        skel = sample['skel']
        label = sample['label']
        
        num_frames_rgb = len(rgb)
        bias = int((1 - self.p_interval) * num_frames_rgb / 2)
        rgb = rgb[bias:num_frames_rgb - bias]

        return {'rgb': rgb,
                'skel': skel,
                'label': label}


class TemAugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb = sample['rgb']
        skel = sample['skel']
        label = sample['label']
        
        ratio = (1.0 - self.p_interval * np.random.rand())
        num_frames_rgb = len(rgb)
        begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
        rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb)]

        return {'rgb': rgb,
                'skel': skel,
                'label': label}

# %%
class NTU_x(Dataset):
    # RGBDI Dataset
    def __init__(self, root_dir='/data/NTU_RGBD_60',  # /data0/xifan/NTU_RGBD_60
                 split='cross_subject', # 40 subjects, 3 cameras
                 stage='train',
                 temTransform=None,
                 spaTransform=None,
                 skeTransform=None,
                 vid_len=(8, 32),
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
        basename_ske = os.path.join(root_dir, 'nturgbd_skeleton/nturgb+d_skeletons')

        self.vid_len = vid_len

        self.rgb_list = []
        self.ske_list = []
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
            self.ske_list += [os.path.join(basename_ske, f) for f in sorted(os.listdir(basename_ske)) if
                          f.split(".")[-1] == "skeleton" and int(f[9:12]) in subjects]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]

        with open("bad_skel.txt", "r") as f:
                for line in f.readlines():
                    if os.path.join(basename_ske, line[:-1] + ".skeleton") in self.ske_list:
                        i = self.ske_list.index(os.path.join(basename_ske, line[:-1] + ".skeleton"))
                        self.ske_list.pop(i)
                        self.rgb_list.pop(i)
                        self.labels.pop(i)

        self.rgb_list, self.ske_list, self.labels = shuffle(self.rgb_list, self.ske_list, self.labels)
        self.root_dir = root_dir
        self.stage = stage
        self.args = args
        self.temTransform = temTransform
        self.spaTransform = spaTransform
        self.skeTransform = skeTransform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        skepath = self.ske_list[idx]
        label = self.labels[idx]
        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        skel = get_3D_skeleton(skepath)
        # print (skel) # numpy array
        # print (skel, 'preprocess') # float tensor
        valid_frame_num = skel.shape[1]
        skel = valid_crop_resize(skel, valid_frame_num, [0.5, 1], 32)
        sample = {'rgb': video, 'skel': skel,  'label': label - 1}
        if self.temTransform:
            sample = self.temTransform(sample)
        # print (sample['rgb'][0].size, len(sample['rgb']))
        if self.spaTransform:
            sample['rgb'] = self.spaTransform(sample['rgb'])
        if self.skeTransform:
            sample['skel'] = self.skeTransform(sample['skel'])
        # print (sample['skel'], 'sample')
        # exit()
        # preprocess
        return sample

class NTU_u(Dataset):
    # RGBDI Dataset
    def __init__(self, root_dir='/data/NTU_RGBD_60',  # /data0/xifan/NTU_RGBD_60
                 split='cross_subject', # 40 subjects, 3 cameras
                 stage='train',
                 temTransform=None,
                 spaTransform_w=None,
                 spaTransform_s=None,
                 skeTransform_w=None,
                 skeTransform_s=None,
                 vid_len=(8, 32),
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_310x256_30') 
        basename_ske = os.path.join(root_dir, 'nturgbd_skeleton/nturgb+d_skeletons')
    
        self.vid_len = vid_len

        self.rgb_list = []
        self.ske_list = []
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
            elif stage == 'train50b':
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
            self.ske_list += [os.path.join(basename_ske, f) for f in sorted(os.listdir(basename_ske)) if
                          f.split(".")[-1] == "skeleton" and int(f[9:12]) in subjects]
            self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]

        with open("bad_skel.txt", "r") as f:
                for line in f.readlines():
                    if os.path.join(basename_ske, line[:-1] + ".skeleton") in self.ske_list:
                        i = self.ske_list.index(os.path.join(basename_ske, line[:-1] + ".skeleton"))
                        self.ske_list.pop(i)
                        self.rgb_list.pop(i)
                        self.labels.pop(i)

        self.rgb_list, self.ske_list, self.labels = shuffle(self.rgb_list, self.ske_list, self.labels)
        self.root_dir = root_dir
        self.stage = stage
        self.args = args
        self.temTransform = temTransform
        self.spaTransform_w = spaTransform_w
        self.spaTransform_s = spaTransform_s
        self.skeTransform_w = skeTransform_w
        self.skeTransform_s = skeTransform_s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        skepath = self.ske_list[idx]
        label = self.labels[idx]
        # if self.args.modality == "rgb" or self.args.modality == "both":
        video = load_video(rgbpath)
        skel = get_3D_skeleton(skepath)
        # normalize
        valid_frame_num = skel.shape[1]
        skel = valid_crop_resize(skel, valid_frame_num, [0.5, 1], 32)
        sample = {'rgb': video, 'skel': skel,  'label': label - 1}
        if self.temTransform:
            sample = self.temTransform(sample)
        samples = {}
        samples['label'] = sample['label']
        samples['rgb_w'] = self.spaTransform_w(sample['rgb'])
        # print (sample['rgb_w'].size(), 'rgb_w')
        samples['rgb_s'] = self.spaTransform_s(sample['rgb'])
        # print (sample['rgb_s'].size(), 'rgb_s')
        if self.skeTransform_w:
            samples['ske_w'] = self.skeTransform_w(sample['skel'])
        else:
            samples['ske_w'] = sample['skel']
        if self.skeTransform_s:
            samples['ske_s'] = self.skeTransform_s(sample['skel'])
        else:
            samples['ske_s'] = sample['skel']
        return samples


def get_ntu_rgbs(args):
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

    # Skeleton
    skeTransform_unlabeled_w = skeTransform_labeled = transforms.Compose([
        skelRandRotate(rand_rotate=False),
        skelValidChoose(random_valid_choose=False, window_size=32, crop_resize=True, random_choose=False),
    ])
    skeTransform_unlabeled_s = transforms.Compose([
        skelRandRotate(rand_rotate=True),
        skelRandChoose(window_size=32),
        skelValidChoose(random_valid_choose=False, window_size=32, crop_resize=True, random_choose=True),
        skelRandMove(),
        skelRandShift(),
    ])
    skeTransform_val = None
    # skeTransform_val = skeTransform_unlabeled_s = skeTransform_unlabeled_w = skeTransform_labeled = None

    temTransform_labeled = transforms.Compose([TemAugCrop(), TemNormalizeLen((8, 32))])
    temTransform_val = transforms.Compose([TemCenterCrop(), TemNormalizeLen((8, 32))])

    train_labeled_dataset = NTU_x(root_dir=args.data_folder, stage=args.dataset, temTransform=temTransform_labeled, spaTransform=spaTransform_labeled, skeTransform=skeTransform_labeled)
    train_unlabeled_dataset = NTU_u(root_dir=args.data_folder, stage='train', temTransform=temTransform_labeled, spaTransform_w=spaTransform_unlabeled_w, spaTransform_s=spaTransform_unlabeled_s, skeTransform_w=skeTransform_unlabeled_w, skeTransform_s=skeTransform_unlabeled_s)    
    eval_dataset = NTU_x(root_dir=args.data_folder, stage='dev', temTransform=temTransform_val, spaTransform=spaTransform_val, skeTransform=skeTransform_val)
    test_dataset = NTU_x(root_dir=args.data_folder, stage='test', temTransform=temTransform_val, spaTransform=spaTransform_val, skeTransform=skeTransform_val)

    print('number of labeled train: {}'.format(len(train_labeled_dataset)))
    print('number of unlabeled train: {}'.format(len(train_unlabeled_dataset)))
    print('number of val: {}'.format(len(eval_dataset)))
    print('number of test: {}'.format(len(test_dataset)))

    return train_labeled_dataset, train_unlabeled_dataset, eval_dataset, test_dataset

class skelNormalize(object):
    def __init__(self, origin_transfer):
        self.origin_transfer = origin_transfer
    def __call__(self, data_numpy):
        # data_numpy = (data_numpy - self.mean_map) / self.std_map
        # be careful the value is for NTU_RGB-D, for other dataset, please replace with value from function 'get_mean_map'
        if self.origin_transfer == 0:
            min_map, max_map = np.array([-4.9881773, -2.939787, -4.728529]), np.array(
                [5.826573, 2.391671, 4.824233])
        elif self.origin_transfer == 1:
            min_map, max_map = np.array([-5.836631, -2.793758, -4.574943]), np.array([5.2021008, 2.362596, 5.1607])
        elif self.origin_transfer == 2:
            min_map, max_map = np.array([-2.965678, -1.8587272, -4.574943]), np.array(
                [2.908885, 2.0095677, 4.843938])
        else:
            min_map, max_map = np.array([-3.602826, -2.716611, 0.]), np.array([3.635367, 1.888282, 5.209939])

        data_numpy = np.floor(255 * (data_numpy - min_map[:, None, None, None]) / \
                                (max_map[:, None, None, None] - min_map[:, None, None, None])) / 255
        return data_numpy

class skelCropResize(object):
    def __init__(self, p_interval, window_size):
        self.p_interval = p_interval
        self.window_size = window_size
    
    def __call__(self, data_numpy, valid_frame_num):
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        return data_numpy

class skelRandRotate(object):
    def __init__(self, rand_rotate):
        self.rand_rotate = rand_rotate

    def __call__(self, data_numpy):
        data_numpy = rand_rotate(data_numpy, self.rand_rotate)
        return data_numpy

class skelRandChoose(object):
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, data_numpy):
        data_numpy = random_choose(data_numpy, self.window_size, auto_pad=True)
        return data_numpy

class skelValidChoose(object):
    def __init__(self, random_valid_choose, window_size, crop_resize, random_choose):
        self.random_valid_choose = random_valid_choose
        self.window_size = window_size
        self.crop_resize = crop_resize
        self.random_choose = random_choose

    def __call__(self, data_numpy):
        if self.random_valid_choose:
            data_numpy = valid_choose(data_numpy, self.window_size, random_pad = True)
        elif self.window_size > 0 and (not self.crop_resize) and (not self.random_choose):
            data_numpy = valid_choose(data_numpy, self.window_size, random_pad=False)
        return data_numpy

class skelRandShift(object):
    def __call__(self, data_numpy):
        data_numpy = random_shift(data_numpy)
        return data_numpy

class skelRandMove(object):
    def __call__(self, data_numpy):
        data_numpy = random_move(data_numpy)
        return data_numpy

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
