import sys
sys.path.append('/data/siit/Weakly-Supervised-Learning-for-Semantic-Segmentation')

import numpy as np
import torch
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import vgg16
from utils.decode import decode_seg_map_sequence
from utils.LoadData import valid_data_loader
from utils.decode import decode_segmap

parser = argparse.ArgumentParser(description='The Pytorch code of OAA')
parser.add_argument("--input_size", type=int, default=321)
parser.add_argument("--crop_size", type=int, default=321)
parser.add_argument("--img_dir", type=str, default="/data/DB/VOCdevkit/VOC2012/")
parser.add_argument("--train_list", type=str, default="/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt")
parser.add_argument("--test_list", type=str, default="/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, default="pascal_voc")
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--checkpoint", type=str, default="DRS_th55_epoch15.pth")
parser.add_argument("--threshold", type=float, default=0.55)

args = parser.parse_args()
print(args)

device = 'cuda'

""" model load """
model = vgg16(pretrained=True, thresh=args.threshold)
model = model.to(device)

model.eval()
    
state = torch.load("../checkpoints/%s.pth" % args.checkpoint)

model.load_state_dict(state['model'], strict=True)


""" dataloader """
test_loader = valid_data_loader(args)

for idx, dat in enumerate(test_loader):
    print("[%03d/%03d]" % (idx, len(test_loader)), end="\r")

    img, label, gt_map, sal_map, img_name = dat
    
    label = label.to(device)
    img = img.to(device)

    _, H, W = sal_map.shape
    CAM = np.zeros((20, H, W), dtype=np.float32)
        
    # multi-scale testing
    for s in [256, 321, 384]:
        _img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)

        logit, cam = model(_img)

        """ obtain CAMs """
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam * label[:, :, None, None].to(device)

        cam = cam[0].cpu().detach().numpy()
        cam = cam / (np.max(cam, (1,2), keepdims=True) + 1e-5)  # [20, H, W]

        CAM = np.maximum(CAM, cam)
        
    gt_map = gt_map[0].detach().numpy()
    sal_map = sal_map[0].detach().numpy()
    label = label[0].cpu().detach().numpy()
    img_name = img_name[0].split("/")[-1][:-4]
    
    """ save localization map """
    attention_map = CAM
    attention_map = np.uint8(attention_map * 255)
    
    np.save("/data/DB/VOCdevkit/VOC2012/localization_maps/%s.npy" % (img_name), attention_map)

    