import sys
sys.path.append('/data/Weakly-Supervised-Learning-for-Semantic-Segmentation')

import numpy as np
import torch
import argparse
import os
import cv2
import time

import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from models.vgg_learnable import vgg16
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.decode import decode_seg_map_sequence
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter


ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def get_arguments():
    parser = argparse.ArgumentParser(description='DRS pytorch implement')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='None')
    parser.add_argument("--test_list", type=str, default='None')
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument('--logdir', default='logs/test1', type=str, help='Tensorboard log dir')
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument('--save_folder', default='checkpoints/', help='Location to save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.20)

    return parser.parse_args()

def get_model(args):
    model = vgg16(pretrained=True, thresh=args.threshold) 

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    return  model, optimizer


def validate(current_epoch):

    print('\nvalidating ... ', flush=True, end='')
    
    mIOU = IOUMetric(num_classes=21)
    cls_acc_matrix = Cls_Accuracy()
    
    val_loss = AverageMeter()
    
    model.eval()
    
    visual_image = None
    visual_cam = None
    visual_label = None
    visual_gt_map = None
    visual_pred_map = None
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
                
            img, label, sal_map, gt_map, _ = dat
            
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit, cam = model(img)

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            """ obtain CAMs """
            cam = cam * label[:, :, None, None].to('cuda')
            cam = F.interpolate(cam, size=img.shape[2:], mode='bilinear', align_corners=False)

            cam = cam.cpu().detach().numpy()
            cam_max = np.max(cam, (2, 3), keepdims=True)
            cam = cam / (cam_max + 1e-5)  # [B, 20, 256, 256]

            val_loss.update(loss.data.item(), img.size()[0])
            gt_map = gt_map.detach().numpy()
            sal_map = sal_map.detach().numpy()

            """ segmentation label generation """
            cam[cam < args.alpha] = 0  # object cue
            bg = np.zeros((B, 1, H, W), dtype=np.float32)
            pred_map = np.concatenate([bg, cam], axis=1)  # [B, 21, H, W]

            pred_map[:, 0, :, :] = (1. - sal_map) # background cue
            pred_map = pred_map.argmax(1)

            mIOU.add_batch(pred_map, gt_map)

            """ for tensorboard visualization """
            visual_image = img[0]
            visual_cam = cam[0]
            visual_label = label[0]
            visual_pred_map = pred_map[0]
            visual_gt_map = gt_map[0]
    
    
    """ validation performance """
    res = mIOU.evaluate()
    val_miou = res["Mean_IoU"]
    val_pixel_acc = res["Pixel_Accuracy"]
    val_cls_acc = cls_acc_matrix.compute_avg_acc()
    
    """ tensorboard visualization """
    result_map, result_cam  = visualize(visual_image, visual_cam, visual_label, visual_gt_map, visual_pred_map)

    writer.add_images('valid CAM', result_cam, current_epoch)
    writer.add_image('valid Seg', result_map, current_epoch)
    writer.add_scalar('valid loss', val_loss.avg, current_epoch)
    writer.add_scalar('valid acc', val_cls_acc, current_epoch)
    writer.add_scalar('valid mIoU', val_miou, current_epoch)
    writer.add_scalar('valid Pixel Acc', val_pixel_acc, current_epoch)
    
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    print('validating mIoU: %.4f' % val_miou)
    print('validating Pixel Acc: %.4f' % val_pixel_acc)
    

def train(current_epoch):
    train_loss = AverageMeter()
    cls_acc_matrix = Cls_Accuracy()

    model.train()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):
        img, label = dat
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit, cam = model(img)

        """ classification loss """
        loss = F.multilabel_soft_margin_loss(logit, label)

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_acc_matrix.update(logit, label)
        train_loss.update(loss.data.item(), img.size()[0])
        
        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            train_cls_acc = cls_acc_matrix.compute_avg_acc()

            _, _, H, W = img.size()
            
            """ obtain CAM """
            cam = cam * label[:, :, None, None].to('cuda')
            cam = F.interpolate(cam, size=img.shape[2:], mode='bilinear', align_corners=False)[0]

            cam = cam.cpu().detach().numpy()
            cam_max = np.max(cam, (1, 2), keepdims=True)
            cam = cam / (cam_max + 1e-5)  # [B, 20, 256, 256]

            """ for tensorboard visualization """
            result_cam = visualize(img[0], cam, label[0])
            
            writer.add_images('train CAM', result_cam, global_counter)
            writer.add_scalar('train loss', train_loss.avg, global_counter)
            writer.add_scalar('train acc', train_cls_acc, global_counter)

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'ACC: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, global_counter%len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], train_cls_acc, loss=train_loss))

    args.global_counter = global_counter
    

def visualize(image, cam, label, gt_map=None, pred_map=None):
    
    image = np.transpose(image.clone().cpu().detach().numpy(), (1,2,0))  # H, W, C
    cam = np.transpose(cam, (1,2,0)) # H, W, C
    
    """ image denormalize """
    image *= [0.229, 0.224, 0.225]
    image += [0.485, 0.456, 0.406]
    image *= 255
    image = np.clip(image.transpose(2,0,1), 0, 255).astype(np.uint8) # C, H, W

    size = image.shape[1]

    """ visualize selected CAM outputs """
    label = label.clone().cpu().detach().numpy()
    label = np.nonzero(label)[0]

    selected_cam_image = np.zeros((len(label)+1, 3, size, size), dtype=np.uint8)
    selected_cam_image[0] = image
    
    for n, i in enumerate(label):
        cam_img = cam[:, :, i] # H, W
        cam_img *= 255
        cam_img = np.clip(cam_img, 0, 255)

        cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cv2.COLORMAP_JET) # H, W, 3
        cam_img = cam_img[:, :, ::-1]

        """ border """
        cam_img[0:3, :] = 128
        cam_img[size-3:size, :] = 128
        cam_img[:, 0:3] = 128
        cam_img[:, size-3:size] = 128

        selected_cam_image[n+1] = cam_img.transpose(2, 0, 1)

    selected_cam_image = selected_cam_image.astype(np.float32) /255.
        
    if gt_map is not None:
        seg_image = np.zeros((3, size*2, 2*size), dtype=np.uint8)
        seg_image[:, :size, :size] = image
        seg_image[:, :size, size:2*size] = 255 * decode_seg_map_sequence(gt_map)

        seg_image[:, size:, :size] = image
        seg_image[:, size:, size:2*size] = 255 * decode_seg_map_sequence(pred_map)
        
        return seg_image, selected_cam_image
        
    return selected_cam_image

        
    
if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    model, optimizer = get_model(args)

    for current_epoch in range(args.epoch+1):
        train(current_epoch)
        validate(current_epoch)

        """ save checkpoint """
        if current_epoch % args.save_interval == 0 and current_epoch > 0:
            print('\nSaving state, epoch : %d \n' % current_epoch)
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter,
            }
            model_file = args.save_folder + '/ckpt_' + repr(current_epoch) + '.pth'
            torch.save(state, model_file)
