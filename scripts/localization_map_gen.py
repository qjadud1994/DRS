import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import torch.nn.functional as F

from models.vgg import vgg16
from utils.decode import decode_seg_map_sequence
from utils.LoadData import test_data_loader
from utils.decode import decode_segmap

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

args = parser.parse_args()
print(args)

output_dir = os.path.join(args.img_dir, "localization_maps")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

""" model load """
model = vgg16(pretrained=True, delta=args.delta)
model = model.cuda()
model.eval()
    
ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)


""" dataloader """
data_loader = test_data_loader(args)

for idx, dat in enumerate(data_loader):
    print("[%04d/%04d]" % (idx, len(data_loader)), end="\r")

    img, label, _, _, img_name = dat
    
    label = label.cuda()
    img = img.cuda()

    H = W = args.crop_size
    localization_maps = np.zeros((20, H, W), dtype=np.float32)
        
    # multi-scale testing
    for s in [256, 320, 384]:
        _img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)

        _, cam = model(_img, label, size=(H, W))

        cam = cam[0].detach().cpu().numpy()
        localization_maps = np.maximum(localization_maps, cam)
        
    img_name = img_name[0].split("/")[-1].split(".")[0]
    
    """ save localization map """
    localization_maps = np.uint8(localization_maps * 255)
    
    np.save(os.path.join(output_dir, "%s.npy" % img_name), localization_maps)

    