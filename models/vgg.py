import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class DRS_learnable(nn.Module):
    """ 
    DRS learnable setting
    hyperparameter X , additional training paramters O 
    """
    def __init__(self, channel):
        super(DRS_learnable, self).__init__()
        self.relu = nn.ReLU()
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()

        x = self.relu(x)
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.global_avg_pool(x).view(b, c)
        control = self.fc(control).view(b, c, 1, 1)
        control = control.expand_as(x)

        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
            
        return x
        
    
class DRS(nn.Module):
    """ 
    DRS non-learnable setting
    hyperparameter O , additional training paramters X
    """
    def __init__(self, delta):
        super(DRS, self).__init__()
        self.relu = nn.ReLU()
        self.delta = delta
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        x = self.relu(x)
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.delta
        
        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
        
        return x

    
class VGG(nn.Module):
    def __init__(self, features, delta=0, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        
        self.features = features
        
        self.layer1_conv1 = features[0]
        self.layer1_relu1 = DRS_learnable(64) if delta == 0 else DRS(delta)
        self.layer1_conv2 = features[2]
        self.layer1_relu2 = DRS_learnable(64) if delta == 0 else DRS(delta)
        self.layer1_maxpool = features[4]
        
        self.layer2_conv1 = features[5]
        self.layer2_relu1 = DRS_learnable(128) if delta == 0 else DRS(delta)
        self.layer2_conv2 = features[7]
        self.layer2_relu2 = DRS_learnable(128) if delta == 0 else DRS(delta)
        self.layer2_maxpool = features[9]
        
        self.layer3_conv1 = features[10]
        self.layer3_relu1 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_conv2 = features[12]
        self.layer3_relu2 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_conv3 = features[14]
        self.layer3_relu3 = DRS_learnable(256) if delta == 0 else DRS(delta)
        self.layer3_maxpool = features[16]
        
        self.layer4_conv1 = features[17]
        self.layer4_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_conv2 = features[19]
        self.layer4_relu2 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_conv3 = features[21]
        self.layer4_relu3 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer4_maxpool = features[23]
        
        self.layer5_conv1 = features[24]
        self.layer5_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer5_conv2 = features[26]
        self.layer5_relu2 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.layer5_conv3 = features[28]
        self.layer5_relu3 = DRS_learnable(512) if delta == 0 else DRS(delta)
        
        self.extra_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_relu1 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.extra_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_relu2 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.extra_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_relu3 = DRS_learnable(512) if delta == 0 else DRS(delta)
        self.extra_conv4 = nn.Conv2d(512, 20, kernel_size=1)
        
        if init_weights:
            self._initialize_weights(self.extra_conv1)
            self._initialize_weights(self.extra_relu1)
            self._initialize_weights(self.extra_conv2)
            self._initialize_weights(self.extra_relu2)
            self._initialize_weights(self.extra_conv3)
            self._initialize_weights(self.extra_relu3)
            self._initialize_weights(self.extra_conv4)
            
            self._initialize_weights(self.layer1_relu1)
            self._initialize_weights(self.layer1_relu2)
            self._initialize_weights(self.layer2_relu1)
            self._initialize_weights(self.layer2_relu2)
            self._initialize_weights(self.layer3_relu1)
            self._initialize_weights(self.layer3_relu2)
            self._initialize_weights(self.layer3_relu3)
            self._initialize_weights(self.layer4_relu1)
            self._initialize_weights(self.layer4_relu2)
            self._initialize_weights(self.layer4_relu3)
            self._initialize_weights(self.layer5_relu1)
            self._initialize_weights(self.layer5_relu2)
            self._initialize_weights(self.layer5_relu3)
        

    def forward(self, x, label=None, size=None):
        if size is None:
            size = x.size()[2:]
        
        # layer1
        x = self.layer1_conv1(x)
        x = self.layer1_relu1(x)
        x = self.layer1_conv2(x)
        x = self.layer1_relu2(x)
        x = self.layer1_maxpool(x)
        
        # layer2
        x = self.layer2_conv1(x)
        x = self.layer2_relu1(x)
        x = self.layer2_conv2(x)
        x = self.layer2_relu2(x)
        x = self.layer2_maxpool(x)
        
        # layer3
        x = self.layer3_conv1(x)
        x = self.layer3_relu1(x)
        x = self.layer3_conv2(x)
        x = self.layer3_relu2(x)
        x = self.layer3_conv3(x)
        x = self.layer3_relu3(x)
        x = self.layer3_maxpool(x)
        
        # layer4
        x = self.layer4_conv1(x)
        x = self.layer4_relu1(x)
        x = self.layer4_conv2(x)
        x = self.layer4_relu2(x)
        x = self.layer4_conv3(x)
        x = self.layer4_relu3(x)
        x = self.layer4_maxpool(x)
        
        # layer5
        x = self.layer5_conv1(x)
        x = self.layer5_relu1(x)
        x = self.layer5_conv2(x)
        x = self.layer5_relu2(x)
        x = self.layer5_conv3(x)
        x = self.layer5_relu3(x)
        
        # extra layer
        x = self.extra_conv1(x)
        x = self.extra_relu1(x)
        x = self.extra_conv2(x)
        x = self.extra_relu2(x)
        x = self.extra_conv3(x)
        x = self.extra_relu3(x)
        x = self.extra_conv4(x)
        # ==============================
        
        logit = self.fc(x)
        
        if label is None:
            # for training
            return logit
        
        else:
            # for validation
            cam = self.cam_normalize(x.detach(), size, label)

            return logit, cam

    
    def fc(self, x):
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        return x
    
    
    def cam_normalize(self, cam, size, label):
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5
        
        cam = cam * label[:, :, None, None] # clean
        
        return cam
    
    
    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                    

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name or 'fc' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

        
        
        
#######################################################################################################
        
    
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=True, delta=0):
    model = VGG(make_layers(cfg['D1']), delta=delta)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
        
    return model


if __name__ == '__main__':
    import copy
    
    model = vgg16(pretrained=True, delta=0.6)

    print(model)
    
    input = torch.randn(2, 3, 321, 321)

    out = model(input)
    
    model.get_parameter_groups()
