import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
import torch_dct as DCT

def remove_small_components(pil_image, min_area=50):

    img = np.array(pil_image)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    
    return Image.fromarray(mask)

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def cv_random_hflip(image, mask):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return image, mask

def cv_random_vflip(image, mask):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    return image, mask


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode=Image.NEAREST
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        
    return image, label


def colorEnhance(image):
    enhance_flag = random.randint(0, 1)
    if enhance_flag == 1:
        bright_intensity = random.randint(5, 20) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 20) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(5, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(5, 20) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    channel,width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([channel,width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)



class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)        

        self.filter_files()
        self.size = len(self.images)

        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


        self.split_rate=1/4

    def __getitem__(self, index):

 
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        gt=remove_small_components(gt,100)

        image,gt=cv_random_hflip(image,gt)
        image,gt=cv_random_vflip(image,gt)

        image = colorEnhance(image)
        
        image = np.array(image)[:, :, ::-1]            
        image = Image.fromarray(np.uint8(image)) 
        
        
        image=self.img_transform(image)
        
        gt_0=np.asarray(gt,dtype=np.uint8)

        gt=self.gt_transform(gt)
        
        
        size = image.shape[2]
        image_dct8 = image.reshape( 3, size // 8, 8, size // 8, 8).permute(1,3,0,2,4)
        image_dct8 = DCT.dct_2d(image_dct8,norm='ortho')
        image_dct8 = image_dct8.reshape(size // 8, size // 8, -1).permute(2,0,1)

        masks= self.dct_mask_gen(gt_0)

        masks = [self.gt_transform(Image.fromarray(mask_i,mode='L')) for mask_i in masks]

        masks = torch.cat(masks, dim=0)

        
        
        return image, gt,  masks, image_dct8

    def dct_mask_gen(self, gt):

        dt2=cv2.distanceTransform(255-gt,cv2.DIST_L2,3)
        dt2[dt2>255]=255
        dt=cv2.distanceTransform(gt,cv2.DIST_L2,3)
        
        dt[dt>255]=255
        dist_max=dt.max()
        dt, gt ,dt2= np.array(dt) * 1.0, np.array(gt) * 1.0,np.array(dt2) * 1.0

        masks = []

        boundary_in= np.logical_and(dt > 0, dt <= self.split_rate*dist_max)

        interior = np.where(dt, 1, 0)-boundary_in
        interior  = interior.astype(np.float32)

        boundary_out = np.logical_and(dt2 > 0, dt2 <= self.split_rate*dist_max)

        contour = 1-np.logical_xor(dt,dt2)

        boundary=np.logical_or(boundary_out, boundary_in)
        boundary=np.logical_or(contour, boundary)
        boundary = boundary.astype(np.float32)
        
        background = np.where(dt2, 1, 0) - boundary_out
        background  = background.astype(np.float32)
        
        masks.append(boundary)
        masks.append(interior)
        masks.append(background)

        return masks
    

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size



def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=16, pin_memory=True):

    dataset = PolypDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        

        self.size = len(self.images)
        self.index = 0

    def load_data(self):

        
 
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            
        image = np.array(image)[:, :, ::-1]            
        image = Image.fromarray(np.uint8(image)) 

        image=self.img_transform(image)
        
        size = image.shape[2]
        image_dct8 = image.reshape( 3, size // 8, 8, size // 8, 8).permute(1,3,0,2,4)
        image_dct8 = DCT.dct_2d(image_dct8,norm='ortho')
        image_dct8 = image_dct8.reshape(size // 8, size // 8, -1).permute( 2, 0,1)

        image=image.unsqueeze(0)
        image_dct8=image_dct8.unsqueeze(0)
        self.index += 1
        self.index = self.index % self.size
        return image, gt, image_dct8 , name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size