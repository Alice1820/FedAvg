# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def vidAutoContrast(vid, **kwarg):
    """
    args: vid: list of PIL Image
    return: _vid: list of PIL Image
    """
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageOps.autocontrast(img))
    return _vid


def vidBrightness(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageEnhance.Brightness(img).enhance(v))
    return _vid


def vidColor(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageEnhance.Color(img).enhance(v))
    return _vid


def vidContrast(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageEnhance.Contrast(img).enhance(v))
    return _vid


def vidCutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return vidCutoutAbs(img, v)


def vidCutoutAbs(vid, v, **kwarg):
    _vid = []
    for img in vid:
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        xy = (x0, y0, x1, y1)
        # gray
        color = (127, 127, 127)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        _vid.append(img)
    return _vid


def vidEqualize(vid, **kwarg):
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageOps.equalize(img))
    return _vid


def vidIdentity(vid, **kwarg):
    return vid


def vidInvert(vid, **kwarg):
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageOps.invert(img))
    return _vid

def vidPosterize(vid, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageOps.posterize(img, v))
    return _vid


def vidRotate(vid, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    _vid = []
    for img in vid:
        _vid.append(img.rotate(v))
    return _vid


def vidSharpness(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageEnhance.Sharpness(img).enhance(v))
    return _vid


def vidShearX(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    _vid = []
    for img in vid:
        _vid.append(img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)))
    return _vid


def vidShearY(vid, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    _vid = []
    if random.random() < 0.5:
            v = -v
    for img in vid:
        _vid.append(img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)))
    return _vid


def vidSolarize(vid, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    _vid = []
    for img in vid:
        _vid.append(PIL.ImageOps.solarize(img, 256 - v))
    return _vid

def vidSolarizeAdd(vid, v, max_v, bias=0, threshold=128):
    _vid = []
    rand = random.random()
    v = _float_parameter(v, max_v) + bias
    if rand < 0.5:
        v = -v
    v = int(v * vid[0].size[1])
    for img in vid:
        img_np = np.array(img).astype(np.int)
        img_np = img_np + v
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        _vid.append(PIL.ImageOps.solarize(img, threshold))
    return _vid


def vidTranslateX(vid, v, max_v, bias=0):
    _vid = []
    rand = random.random()
    v = _float_parameter(v, max_v) + bias
    if rand < 0.5:
        v = -v
    v = int(v * vid[0].size[1])
    for img in vid:
        _vid.append(img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)))
    return _vid


def vidTranslateY(vid, v, max_v, bias=0):
    _vid = []
    rand = random.random()
    v = _float_parameter(v, max_v) + bias
    if rand < 0.5:
        v = -v
    v = int(v * vid[0].size[1])
    for img in vid:
        _vid.append(img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)))
    return _vid


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(vidAutoContrast, None, None),
            (vidBrightness, 0.9, 0.05),
            (vidColor, 0.9, 0.05),
            (vidContrast, 0.9, 0.05),
            (vidEqualize, None, None),
            (vidIdentity, None, None),
            (vidPosterize, 4, 4),
            (vidRotate, 30, 0),
            (vidSharpness, 0.9, 0.05),
            (vidShearX, 0.3, 0),
            (vidShearY, 0.3, 0),
            (vidSolarize, 256, 0),
            (vidTranslateX, 0.3, 0),
            (vidTranslateY, 0.3, 0)]
    return augs

def dep_augment_pool():
    # FixMatch paper
    augs = [
            # (vidAutoContrast, None, None),
            # (vidBrightness, 0.9, 0.05),
            # (vidColor, 0.9, 0.05),
            # (vidContrast, 0.9, 0.05),
            # (vidEqualize, None, None),
            # (vidIdentity, None, None),
            # (vidPosterize, 4, 4),
            (vidRotate, 30, 0),
            # (vidSharpness, 0.9, 0.05),
            # (vidShearX, 0.3, 0),
            # (vidShearY, 0.3, 0),
            # (vidSolarize, 256, 0),
            # (vidTranslateX, 0.3, 0),
            # (vidTranslateY, 0.3, 0)
            ]
    return augs

def my_augment_pool():
    # Test
    augs = [(vidAutoContrast, None, None),
            (vidBrightness, 1.8, 0.1),
            (vidColor, 1.8, 0.1),
            (vidContrast, 1.8, 0.1),
            (vidCutout, 0.2, 0),
            (vidEqualize, None, None),
            (vidInvert, None, None),
            (vidPosterize, 4, 4),
            (vidRotate, 30, 0),
            (vidSharpness, 1.8, 0.1),
            (vidShearX, 0.3, 0),
            (vidShearY, 0.3, 0),
            (vidSolarize, 256, 0),
            (vidSolarizeAdd, 110, 0),
            (vidTranslateX, 0.45, 0),
            (vidTranslateY, 0.45, 0)]
    return augs

def lab_augment_pool():
    # Test
    augs = [
            # (vidAutoContrast, None, None),
            # (vidBrightness, 1.8, 0.1),
            # (vidColor, 1.8, 0.1),
            # (vidContrast, 1.8, 0.1),
            # (vidCutout, 0.2, 0),
            # (vidEqualize, None, None),
            # (vidInvert, None, None),
            # (vidPosterize, 4, 4),
            (vidRotate, 30, 0),
            # (vidSharpness, 1.8, 0.1),
            (vidShearX, 0.3, 0),
            (vidShearY, 0.3, 0),
            # (vidSolarize, 256, 0),
            # (vidSolarizeAdd, 110, 0),
            (vidTranslateX, 0.45, 0),
            (vidTranslateY, 0.45, 0)]
    return augs

class vidRandAugmentMC(object):
    # TODO: video transform
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, vid): # img: PILImage, video: list of PILImage
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                vid = op(vid, v=v, max_v=max_v, bias=bias)
        vid = vidCutoutAbs(vid, int(32*0.5))
        return vid

class labRandAugmentMC(object):
    # TODO: video transform
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = lab_augment_pool()

    def __call__(self, vid): # img: PILImage, video: list of PILImage
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                vid = op(vid, v=v, max_v=max_v, bias=bias)
        vid = vidCutoutAbs(vid, int(32*0.5))
        return vid

class vidRGB2Lab(object):
    # TODO: video transform
    def __init__(self, flag=True):
        self.flag = flag
        # print (self.flag)

    def __call__(self, vid): # img: PILImage, video: list of PILImage
        w, h = vid[0].size
        # print (len(vid), w, h)
        if self.flag is True:
            _vid = []
            srgb_profile = ImageCms.createProfile("sRGB")
            lab_profile  = ImageCms.createProfile("LAB")
            rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
            # lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
            for idx, img in enumerate(vid):
                # print (img.mode)
                # img = img.convert("RGB")
                img = ImageCms.applyTransform(img, rgb2lab_transform)
                _vid.append(img)
                # img = ImageCms.applyTransform(img, lab2rgb_transform)
                # img.save('/home/zhangxifan/fixmatch/logs/img{:02d}.jpg'.format(idx))
            return _vid
        else:
            return vid