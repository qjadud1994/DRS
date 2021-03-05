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

from models.vgg_refine import vgg16
from utils.decode import decode_seg_map_sequence
from utils.LoadData_refine import test_data_loader
from utils.Metrics import Cls_Accuracy, RunningConfusionMatrix, IOUMetric
from utils.decode import decode_segmap

parser = argparse.ArgumentParser(description='DRS pytorch implement')
parser.add_argument("--input_size", type=int, default=321)
parser.add_argument("--crop_size", type=int, default=321)
parser.add_argument("--img_dir", type=str, default="/data/DB/VOCdevkit/VOC2012/")
parser.add_argument("--train_list", type=str, default="/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt")
parser.add_argument("--test_list", type=str, default="/data/DB/VOCdevkit/VOC2012/ImageSets/Segmentation/train_labeled_cls.txt")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset", type=str, default="pascal_voc")
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--checkpoint", type=str, default="Refinement.pth")
parser.add_argument("--alpha", type=float, default=0.20)
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" model load """
model = vgg16(pretrained=True)
model = model.to(device)

model.eval()
    
state = torch.load("../checkpoints/%s.pth" % args.checkpoint)

model.load_state_dict(state['model'], strict=True)


""" dataloader """
test_loader = test_data_loader(args)

""" metric """
mIOU = IOUMetric(num_classes=21)
running_confusion_matrix = RunningConfusionMatrix(labels=range(21))

global_counter = 0

for idx, dat in enumerate(test_loader):
    print("[%03d/%03d]" % (idx, len(test_loader)), end="\r")

    img, label, sal_map, gt_map, att_map, img_name = dat
    
    label = label.to(device)
    img = img.to(device)

    _, H, W = sal_map.shape

    CAM = np.zeros((20, H, W), dtype=np.float32)
        
    """ single-scale testing """
    for s in [321, ]:
        _img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)

        cam = model(_img)

        """ obtain CAMs """
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam * label[:, :, None, None].to(device)

        cam = cam[0].cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)

        CAM = np.maximum(CAM, cam)

    img = img[0].cpu().detach().numpy()
    gt_map = gt_map[0].detach().numpy()
    sal_map = sal_map[0].detach().numpy()

    
    """ segmentation label generation """
    CAM[CAM < args.cut_thresh] = 0 # object cue

    bg = np.zeros((1, H, W), dtype=np.float32)
    pred_map = np.concatenate([bg, CAM], axis=0)  # [21, H, W]
    
    pred_map[0, :, :] = (1. - sal_map) # backgroudn cue
    
    pred_map = pred_map.argmax(0)
    mIOU.add_batch(pred_map[None, ...], gt_map[None, ...])
    
    
""" performance """
res = mIOU.evaluate()

val_miou = res["Mean_IoU"]
val_pixel_acc = res["Pixel_Accuracy"]

print("ckpt : ". args.checkpoint)
print("val_miou : %.4f" % val_miou)
print("val_pixel_acc : %.4f" % val_pixel_acc)
