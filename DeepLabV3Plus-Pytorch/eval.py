from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time
import joblib
import multiprocessing

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizer import Visualizer
from utils.utils import AverageMeter

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from utils.crf import DenseCRF

torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="num classes 21 for VOC")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=2,
                        help="random seed (default: 2)")
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012_aug',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--logit_dir", type=str, default='./logits')
    
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                              image_set='val', download=False, 
                              transform=val_transform, ret_fname=True)

    return val_dst


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    
    with torch.no_grad():
        for i, (images, labels, fnames) in enumerate(loader):
            print("[%04d/%04d] " % (i, len(loader)), end="\r")
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
            for b in range(outputs.size(0)):
                fname = fnames[b]
                np.save(os.path.join(opts.logit_dir, fname + ".npy"), outputs[b].detach().cpu().numpy().astype(np.float16))

        score = metrics.get_results()
        
    return score


def crf_inference(opts, dataset, metrics):
    metrics.reset()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
        
    def process(i):
        image, gt_label, fname = dataset.__getitem__(i)

        filename = os.path.join(opts.logit_dir, fname + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()
        gt_label = gt_label.cpu().numpy()
        
        image = image.permute(1, 2, 0).cpu().numpy()
        image *= std
        image += mean
        image *= 255
        image = image.astype(np.uint8)
        prob = postprocessor(image, prob)
        pred_label = np.argmax(prob, axis=0)

        return pred_label, gt_label
    
    # CRF in multi-process
    results = joblib.Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts = zip(*results)

    for pred, gt in zip(preds, gts):
        metrics.update(gt, pred)
        
    score = metrics.get_results()
    
    return score



def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    os.makedirs(opts.logit_dir, exist_ok=True)
    
    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4)
    
    print("Dataset: voc, Val set: %d" %
          ( len(val_dst)) )

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        assert "no checkpoint"

    #==========   Eval   ==========#
    model.eval()
    val_score = validate(
        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
    print(metrics.to_str(val_score))

    print("\n\n----------- crf -------------")
    crf_score = crf_inference(opts, val_dst, metrics)
    print(metrics.to_str(crf_score))
    
    os.system(f"rm -rf {opts.logit_dir}")
    
    
if __name__ == '__main__':
    main()
