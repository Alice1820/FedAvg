import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

def inflated_resnet(**kwargs):
    list_block = [Bottleneck3D, Bottleneck3D, Bottleneck3D, Bottleneck3D]
    list_layers = [3, 4, 6, 3]

    # Create the model
    model = ResNet(list_block,
                   list_layers,
                   **kwargs)

    # Pretrained from imagenet weights
    load_pretrained_2D_weights('resnet50', model, inflation='center')

    return model

def _inflate_weight(w, new_temporal_size, inflation='center'):
    w_up = w.unsqueeze(2).repeat(1, 1, new_temporal_size, 1, 1)
    if inflation == 'center':
        w_up = central_inflate_3D_conv(w_up)  # center
    elif inflation == 'mean':
        w_up /= new_temporal_size  # mean
    return w_up


def central_inflate_3D_conv(w):
    new_temporal_size = w.size(2)
    middle_timestep = int(new_temporal_size / 2.)
    before, after = list(range(middle_timestep)), list(range(middle_timestep + 1, new_temporal_size))
    if len(before) > 0:
        w[:, :, before] = torch.zeros_like(w[:, :, before])
    if len(after):
        w[:, :, after] = torch.zeros_like(w[:, :, after])
    return w


def _update_pretrained_weights(model, pretrained_W, inflation='center'):
    pretrained_W_updated = pretrained_W.copy()
    model_dict = model.state_dict()
    for k, v in pretrained_W.items():
        if k in model_dict.keys():
            if len(model_dict[k].shape) == 5:
                new_temporal_size = model_dict[k].size(2)
                v_updated = _inflate_weight(v, new_temporal_size, inflation)
            else:
                v_updated = v

            if isinstance(v, torch.autograd.Variable):
                pretrained_W_updated.update({k: v_updated.data})
            else:
                pretrained_W_updated.update({k: v_updated})
        elif "fc.weight" in k:
            pretrained_W_updated.pop('fc.weight', None)
        elif "fc.bias" in k:
            pretrained_W_updated.pop('fc.bias', None)
        else:
            print('{} cannot be init with Imagenet weighst'.format(k))

    # update the state dict
    model_dict.update(pretrained_W_updated)

    return model_dict


def _keep_only_existing_keys(model, pretrained_weights_inflated):
    # Loop over the model_dict and update W
    model_dict = model.state_dict()  # Take the initial weights
    for k, v in model_dict.items():
        if k in pretrained_weights_inflated.keys():
            model_dict[k] = pretrained_weights_inflated[k]
    return model_dict


def load_pretrained_2D_weights(arch, model, inflation):
    pretrained_weights = model_zoo.load_url(model_urls[arch])
    pretrained_weights_inflated = _update_pretrained_weights(model, pretrained_weights, inflation)
    model.load_state_dict(pretrained_weights_inflated)
    print("---> Imagenet initialization - 3D from 2D (inflation = {})".format(inflation))

# %%
class I3D(nn.Module):
    def __init__(self, low_dim=128, in_channel=3, width=1, vid_len=8):
        super(I3D, self).__init__()
        self.cnn = inflated_resnet()
        self.avgpool_Tx7x7 = nn.AvgPool3d((vid_len, 7, 7))
        self.D = 2048
        self.classifier = nn.Linear(self.D, low_dim)

    def temporal_pooling(self, x):
        B, D, T, W, H = x.size()
        # print (B, D, T, W, H) # 1 2048 8 8 10
        if self.D == D:
            final_representation = self.avgpool_Tx7x7(x)
            final_representation = final_representation.view(B, self.D)
            return final_representation
        else:
            print("Temporal pooling is not possible due to invalid channels dimensions:", self.D, D)

    def forward(self, x):
        # Changing temporal and channel dim to fit the inflated resnet input requirements
        # print (x.size())
        # B, T, W, H, C = x.size()
        B, T, C, W, H = x.size()
        # x = x.view(B, 1, T, W, H, C)
        # x = x.transpose(1, -1)
        # x = x.view(B, C, T, W, H)
        x = x.transpose(1, 2)
        x = x.contiguous()

        # Inflated ResNet
        out_1, out_2, out_3, out_4 = self.cnn.get_feature_maps(x)

        # Temporal pooling
        out_5 = self.temporal_pooling(out_4)
        out_6 = self.classifier(out_5)

        return out_5, out_6

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

K_1st_CONV = 3


class ResNet(nn.Module):
    def __init__(self, list_block, layers,
                 **kwargs):
        self.inplanes = 64
        self.input_dim = 4
        super(ResNet, self).__init__()
        self._first_conv()
        self.relu = nn.ReLU(inplace=True)
        self.list_channels = [64, 128, 256, 512]
        self.layer1 = self._make_layer(list_block[0], self.list_channels[0], layers[0])
        self.layer2 = self._make_layer(list_block[1], self.list_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(list_block[2], self.list_channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(list_block[3], self.list_channels[3], layers[3], stride=2)
        self.out_dim = 5

        # Init of the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _first_conv(self):
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.input_dim = 4

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        # Upgrade the stride is spatio-temporal kernel
        stride = (1, stride, stride)

        if stride != 1 or self.inplanes != planes * block.expansion:
            conv, batchnorm = nn.Conv3d, nn.BatchNorm3d

            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False, dilation=dilation),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feature_maps(self, x):

        B, C, T, W, H = x.size()

        # 5D -> 4D if 2D conv at the beginning
        x = transform_input(x, self.input_dim, T=T)

        # 1st conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 1st residual block
        x = transform_input(x, self.layer1[0].input_dim, T=T)
        x = self.layer1(x)
        fm1 = x

        # 2nd residual block
        x = transform_input(x, self.layer2[0].input_dim, T=T)
        x = self.layer2(x)
        fm2 = x

        # 3rd residual block
        x = transform_input(x, self.layer3[0].input_dim, T=T)
        x = self.layer3(x)
        fm3 = x

        # 4th residual block
        x = transform_input(x, self.layer4[0].input_dim, T=T)
        x = self.layer4(x)
        final_fm = transform_input(x, self.out_dim, T=T)

        return fm1, fm2, fm3, final_fm

def transform_input(x, dim, T=12):
    diff = len(x.size()) - dim

    if diff > 0:
        B, C, T, W, H = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, C, W, H)
    elif diff < 0:
        _, C, W, H = x.size()
        x = x.view(-1, T, C, W, H)
        x = x.transpose(1, 2)

    return x

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dilation=(1, dilation, dilation))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.input_dim = 5
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class I3DV1(nn.Module):
    def __init__(self, name='resnet50'):
        super(I3DV1, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = I3D()
            self.ab_to_l = I3D()
        # elif name == 'resnet18':
        #     self.l_to_ab = resnet18(in_channel=1, width=2)
        #     self.ab_to_l = resnet18(in_channel=2, width=2)
        # elif name == 'resnet101':
        #     self.l_to_ab = resnet101(in_channel=1, width=2)
        #     self.ab_to_l = resnet101(in_channel=2, width=2)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, l, ab): # x: [bs, 3, width, height]
        # l: [bs, 3, width, height]
        # ab: [bs, 1, width, height]
        # l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l) # 

        feat_ab = self.ab_to_l(ab)
        return feat_l, feat_ab


class MyI3DCMC(nn.Module):
    def __init__(self, name='resnet50v1'):
        super(MyI3DCMC, self).__init__()
        if name.endswith('v1'):
            self.encoder = I3DV1(name[:-2])
        elif name.endswith('v2'):
            self.encoder = I3DV2(name[:-2])
        elif name.endswith('v3'):
            self.encoder = I3DV3(name[:-2])
        else:
            raise NotImplementedError('model not support: {}'.format(name))

        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, l, ab, layer=7):
        return self.encoder(l, ab)