import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.init as init
import math
import cv2
import numpy as np
import os

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

    
class VGG(nn.Module):

    def __init__(self, features, num_classes=20, init_weights=True):
        
        super(VGG, self).__init__()
        
        self.features = features
        
        self.layer1_conv1 = features[0]
        self.layer1_relu1 = features[1]
        self.layer1_conv2 = features[2]
        self.layer1_relu2 = features[3]
        self.layer1_maxpool = features[4]
        
        self.layer2_conv1 = features[5]
        self.layer2_relu1 = features[6]
        self.layer2_conv2 = features[7]
        self.layer2_relu2 = features[8]
        self.layer2_maxpool = features[9]
        
        self.layer3_conv1 = features[10]
        self.layer3_relu1 = features[11]
        self.layer3_conv2 = features[12]
        self.layer3_relu2 = features[13]
        self.layer3_conv3 = features[14]
        self.layer3_relu3 = features[15]
        self.layer3_maxpool = features[16]
        
        self.layer4_conv1 = features[17]
        self.layer4_relu1 = features[18]
        self.layer4_conv2 = features[19]
        self.layer4_relu2 = features[20]
        self.layer4_conv3 = features[21]
        self.layer4_relu3 = features[22]
        self.layer4_maxpool = features[23]
        
        self.layer5_conv1 = features[24]
        self.layer5_relu1 = features[25]
        self.layer5_conv2 = features[26]
        self.layer5_relu2 = features[27]
        self.layer5_conv3 = features[28]
        self.layer5_relu3 = features[29]
        
        self.extra_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.activation = nn.ReLU()
        
        if init_weights:
            self._initialize_weights(self.extra_conv1)
            self._initialize_weights(self.extra_conv2)
            self._initialize_weights(self.extra_conv3)
            self._initialize_weights(self.extra_conv4)

            
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
        x = self.activation(x)
        x = self.extra_conv2(x)
        x = self.activation(x)
        x = self.extra_conv3(x)
        x = self.activation(x)
        x = self.extra_conv4(x)
            
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        if label is not None:
            x = x * label[:, :, None, None] # clean
            
        return x

    
    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        
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


def vgg16(pretrained=True):
    model = VGG(make_layers(cfg['D1']))
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


if __name__ == '__main__':
    import copy
    
    model = vgg16(pretrained=True)
    
    input = torch.randn(1, 3, 321, 321)

    out = model(input)
    
