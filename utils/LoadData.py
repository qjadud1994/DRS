from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong, RandomCrop
import os
from PIL import Image
import random

def train_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, mode='train')

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader


def valid_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='valid')

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return val_loader



def test_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset(args.test_list, crop_size, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, mode='test')

    test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader


class VOCDataset(Dataset):
    def __init__(self, datalist_file, input_size, root_dir, num_classes=20, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes

        self.image_list, self.label_list, self.gt_map_list, self.sal_map_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
            
        if self.mode == 'valid':
            self.map_transform = transforms.Compose([transforms.Resize(input_size, Image.NEAREST), transforms.From_Numpy()])
        elif self.mode == 'test':
            self.map_transform = transforms.Compose([transforms.From_Numpy(), ])  # pseudo gt gen
        
        
    def __len__(self):
        return len(self.image_list)

    
    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.mode != 'train':
            gt_map = Image.open(self.gt_map_list[idx])
            sal_map = Image.open(self.sal_map_list[idx])
            
            gt_map = self.map_transform(gt_map)
            sal_map = self.map_transform(sal_map)
            
            sal_map = (sal_map > 50).float()
            
            return image, self.label_list[idx], sal_map, gt_map, img_name
        
        return image, self.label_list[idx], img_name

    
    def read_labeled_image_list(self, data_dir, data_list):
        img_dir = os.path.join(data_dir, "JPEGImages")
        gt_map_dir = os.path.join(data_dir, "SegmentationClassAug")
        sal_map_dir = os.path.join(data_dir, "saliency_map")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        gt_map_list = []
        sal_map_list = []
        
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            gt_map = fields[0] + '.png'
            sal_map = fields[0] + '.png'
            
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(img_dir, image))
            gt_map_list.append(os.path.join(gt_map_dir, gt_map))
            sal_map_list.append(os.path.join(sal_map_dir, sal_map))
            img_labels.append(labels)
            
        return img_name_list, img_labels, gt_map_list, sal_map_list
