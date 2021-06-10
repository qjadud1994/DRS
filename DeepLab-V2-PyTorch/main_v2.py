#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import random
import os
import argparse
import cv2
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image

from libs.datasets import get_dataset
from libs.models import *
from libs.utils import PolynomialLR
from libs.utils.stream_metrics import StreamSegMetrics, AverageMeter


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--config_path", type=str, help="config file path")
    parser.add_argument("--gt_path", type=str, help="gt label path")
    parser.add_argument("--log_dir", type=str, help="training log path")
    parser.add_argument("--cuda", type=bool, default=True, help="GPU")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--val_interval", type=int, default=100, help="val_interval")
    
    return parser
                        

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def main():
    opts = get_argparser().parse_args()
    print(opts)
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    """
    Training DeepLab by v2 protocol
    """
    # Configuration
    
    with open(opts.config_path) as f:
        CONFIG = Dict(yaml.load(f))
    
    device = get_device(opts.cuda)
    torch.backends.cudnn.benchmark = True

    # Dataset
    train_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
        gt_path=opts.gt_path,
    )
    print(train_dataset)
    print()
    
    valid_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
        gt_path="SegmentationClassAug",
    )
    print(valid_dataset)  

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )
    
    # Model check
    print("Model:", CONFIG.MODEL.NAME)
    assert (
        CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC"
    ), 'Currently support only "DeepLabV2_ResNet101_MSC"'

    # Model setup
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL, map_location='cpu')
    
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
            
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        opts.log_dir,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TRAIN,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    model.train()

    metrics = StreamSegMetrics(CONFIG.DATASET.N_CLASSES)
    
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    
    curr_iter = 0
    best_score = 0
    end_time = time.time()
    
    while True:
        for _, images, labels, cls_labels in train_loader:
            curr_iter += 1
            loss = 0
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=opts.amp):
                # Propagate forward
                logits = model(images.to(device))

                # Loss
                for logit in logits:
                    # Resize labels for {100%, 75%, 50%, Max} logits
                    _, _, H, W = logit.shape
                    labels_ = resize_labels(labels, size=(H, W))

                    pseudo_labels = logit.detach() * cls_labels[:, :, None, None].to(device)
                    pseudo_labels = pseudo_labels.argmax(dim=1)

                    _loss = criterion(logit, labels_.to(device)) + criterion(logit, pseudo_labels)

                    loss += _loss

                # Propagate backward (just compute gradients wrt the loss)
                loss = (loss / len(logits))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            avg_loss.update(loss.item())
            avg_time.update(time.time() - end_time)
            end_time = time.time()

            # TensorBoard
            if curr_iter % 10 == 0:
                print(" Itrs %d/%d, Loss=%6f, Time=%.2f , LR=%.8f" %
                  (curr_iter, CONFIG.SOLVER.ITER_MAX, 
                   avg_loss.avg, avg_time.avg*1000, optimizer.param_groups[0]['lr']))
                
            # validation
            if curr_iter % opts.val_interval == 0:
                print("... validation")
                metrics.reset()
                with torch.no_grad():
                    for _, images, labels, _ in valid_loader:
                        images = images.to(device)

                        # Forward propagation
                        logits = model(images)

                        # Pixel-wise labeling
                        _, H, W = labels.shape
                        logits = F.interpolate(logits, size=(H, W), 
                                               mode="bilinear", align_corners=False)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        targets = labels.cpu().numpy()

                        metrics.update(targets, preds)

                score = metrics.get_results()
                print(metrics.to_str(score))
                
                if score['Mean IoU'] > best_score:  # save best model
                    best_score = score['Mean IoU']
                    torch.save(
                        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_best.pth")
                    )
                
            if curr_iter > CONFIG.SOLVER.ITER_MAX:
                return
            

if __name__ == "__main__":
    
    main()
