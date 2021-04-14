import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

from models.vgg_refine import vgg16
#from models.vgg import vgg16
from utils.Metrics import IOUMetric
from utils.LoadData import test_data_loader
from utils.decode import get_palette

parser = argparse.ArgumentParser(description='DRS pytorch implementation')
parser.add_argument("--input_size", type=int, default=320)
parser.add_argument("--crop_size", type=int, default=320)
parser.add_argument("--img_dir", type=str, default="/data/DB/VOC2012/")
parser.add_argument("--test_list", type=str, default='VOC2012_list/train_aug_cls.txt')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--delta", type=float, default=0, help='set 0 for the learnable DRS')
parser.add_argument("--alpha", type=float, default=0.20)

args = parser.parse_args()
print(args)

output_dir = os.path.join(args.img_dir, "refined_pseudo_segmentation_labels")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" model load """
#model = vgg16(pretrained=True, delta=args.delta)
model = vgg16()
model = model.cuda()
model.eval()
    
ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)


""" dataloader """
data_loader = test_data_loader(args)
palette = get_palette()

for idx, dat in enumerate(data_loader):
    print("[%03d/%03d]" % (idx, len(data_loader)), end="\r")

    img, label, sal_map, _, img_name = dat
    
    label = label.cuda()
    img = img.cuda()

    _, H, W = sal_map.shape
    localization_maps = np.zeros((20, H, W), dtype=np.float32)

    """ single-scale testing """
    for s in [256, 320, 384]:
        _img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)

        #_, cam = model(_img, label, size=(H, W))
        cam = model(_img, label, size=(H, W))

        """ obtain CAMs """
        cam = cam[0].cpu().detach().numpy()
        localization_maps = np.maximum(localization_maps, cam)

    sal_map = sal_map[0].detach().numpy()
    img_name = img_name[0].split("/")[-1].split(".")[0]
    
    """ segmentation label generation """
    localization_maps[localization_maps < args.alpha] = 0 # object cue

    bg = np.zeros((1, H, W), dtype=np.float32)
    pred_map = np.concatenate([bg, localization_maps], axis=0)  # [21, H, W]
    
    pred_map[0, :, :] = (1. - sal_map) # backgroudn cue
    
    # conflict pixels with multiple confidence values
    bg = np.array(pred_map > 0.9, dtype=np.uint8)
    bg = np.sum(bg, axis=0)
    pred_map = pred_map.argmax(0).astype(np.uint8)
    pred_map[bg > 2] = 255

    # pixels regarded as background but confidence saliency values 
    bg = (sal_map == 1).astype(np.uint8) * (pred_map == 0).astype(np.uint8)
    pred_map[bg > 0] = 255
    
    """ save pseudo segmentation label """
    pred_map = Image.fromarray(pred_map)
    pred_map.putpalette(palette)
    pred_map.save(os.path.join(output_dir, "%s.png" % img_name))
    
print("done!")