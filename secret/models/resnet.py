from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = ['resnet50']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.reset_params()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ResNet(nn.Module):
    __factory = {
        50: torchvision.models.resnet50,
    }

    def __init__(self, depth, cfg, num_classes):
        super(ResNet, self).__init__()

        self.pretrained = cfg.MODEL.BACKBONE.PRETRAIN
        self.depth = depth
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=self.pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_features = resnet.fc.in_features
        self.num_classes = num_classes

        out_planes = resnet.fc.in_features

        # Append new layers
        self.num_features = out_planes
        self.part_detach = cfg.MODEL.PART_DETACH

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        norm_layer = nn.BatchNorm2d
        block = Bottleneck
        planes = 512
        self.planes = planes
        downsample = nn.Sequential(
            conv1x1(out_planes, planes * block.expansion),
            norm_layer(planes * block.expansion),
        )
        self.part_bottleneck = block(
                out_planes, planes, downsample = downsample, norm_layer = norm_layer
            )

        self.part_num_features = planes * block.expansion
        self.part_pool = nn.AdaptiveAvgPool2d((2,1))

        self.partup_feat_bn = nn.BatchNorm1d(self.part_num_features)
        self.partup_feat_bn.bias.requires_grad_(False)
        init.constant_(self.partup_feat_bn.weight, 1)
        init.constant_(self.partup_feat_bn.bias, 0)

        self.partdown_feat_bn = nn.BatchNorm1d(self.part_num_features)
        self.partdown_feat_bn.bias.requires_grad_(False)
        init.constant_(self.partdown_feat_bn.weight, 1)
        init.constant_(self.partdown_feat_bn.bias, 0)

        self.classifier_partup = nn.Linear(self.part_num_features, self.num_classes, bias = False)
        init.normal_(self.classifier_partup.weight, std=0.001)
        self.classifier_partdown = nn.Linear(self.part_num_features, self.num_classes, bias = False)
        init.normal_(self.classifier_partdown.weight, std=0.001)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, finetune = False):
        featuremap = self.base(x)

        x = self.gap(featuremap)
        x = x.view(x.size(0), -1)

        bn_x = self.feat_bn(x)

        if self.part_detach:
            part_x = self.part_bottleneck(featuremap.detach())
        else:
            part_x = self.part_bottleneck(featuremap)

        part_x = self.part_pool(part_x)
        part_up = part_x[:, :, 0, :]
        part_up = part_up.view(part_up.size(0), -1)
        bn_part_up = self.partup_feat_bn(part_up)

        part_down = part_x[:, :, 1, :]
        part_down = part_down.view(part_down.size(0), -1)
        bn_part_down = self.partdown_feat_bn(part_down)

        if self.training is False and finetune is False:
            bn_x = F.normalize(bn_x)
            return [bn_x]

        prob = self.classifier(bn_x)
        prob_part_up = self.classifier_partup(bn_part_up)
        prob_part_down = self.classifier_partdown(bn_part_down)

        if finetune is True:
            bn_x = F.normalize(bn_x)
            bn_part_up = F.normalize(bn_part_up)
            bn_part_down = F.normalize(bn_part_down)
            return [x, part_up, part_down], [bn_x, bn_part_up, bn_part_down], [prob, prob_part_up, prob_part_down]
        else:
            return [x, part_up, part_down], [prob, prob_part_up, prob_part_down]

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = PartResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())

def resnet50(cfg, num_classes, **kwargs):
    return ResNet(50, cfg, num_classes, **kwargs)
