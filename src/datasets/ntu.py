import logging
from torchvision import transforms
from torchvideotransforms import video_transforms, volume_transforms, tensor_transforms

from .ntu_rgb import NTU, TemAugCrop, TemNormalizeLen, TemCenterCrop
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