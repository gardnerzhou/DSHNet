
import os

import torch
import random
from datetime import datetime
import argparse

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from lib.network import DSHNet
from dataloader import get_loader,test_dataset

from torch import optim 
from torch.autograd import Variable


def seed_torch(seed=11219116):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if epoch%decay_epoch==1 and epoch>1:

        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

            
def test(model, data_path,opt):
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    b=0.0
    smooth = 1e-8
    for i in range(test_loader.size):
        image, gt,image_dct,name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image_dct=image_dct.cuda()

        pred,masks,  mid_preds  = model(image,image_dct)

        res=pred
        res = F.interpolate(res, size=gt.shape,  mode='bilinear')
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))

        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
 
        a = float(loss)
        b = b + a
        
    return b/test_loader.size*100 

def train(train_loader, model,  optimizer, epoch,  opt):
    
    model.train()
    loss_all_sum = 0.
    loss_pred_sum = 0.
    loss_mask_sum = 0.

    epoch_step = 0

    for i, data_batch in enumerate(train_loader, start=1):
        
        optimizer.zero_grad()
        images, gts,dt_masks,image_dct = data_batch

        images, gts,dt_masks,image_dct= Variable(images.cuda()),Variable(gts.cuda()),Variable(dt_masks.cuda()),Variable(image_dct.cuda())
                                        
        pred, masks,  mid_preds = model(images,image_dct)

        loss_pred = structure_loss(pred, gts)
        loss_pred = structure_loss(F.interpolate(mid_preds[0],gts.size()[2:],mode='bilinear'),gts) + loss_pred
        loss_pred = structure_loss(F.interpolate(mid_preds[1],gts.size()[2:],mode='bilinear'),gts) + loss_pred
        loss_pred = structure_loss(F.interpolate(mid_preds[2],gts.size()[2:],mode='bilinear'),gts) + loss_pred

        loss_mask = 0.0

        for p_mask in masks:
            loss_mask = F.mse_loss(F.interpolate(p_mask, size=dt_masks.size()[2:], mode="bilinear"), dt_masks) + loss_mask

        loss_mask = loss_mask * opt.mask_loss
        
        loss = loss_pred + loss_mask

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        
        epoch_step += 1
        loss_all_sum += loss.cpu().data.item()
        loss_pred_sum += loss_pred.cpu().data.item()
        loss_mask_sum += loss_mask.cpu().data.item()
                                                                        
    print('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_all_AVG: {:.4f}, Loss_pred_AVG: {:.4f}, Loss_mask_AVG: {:.4f}'.format(epoch,
                                                                                            opt.epochs,
                                                                                            loss_all_sum / epoch_step,
                                                                                            loss_pred_sum / epoch_step,
                                                                                            loss_mask_sum / epoch_step
                                                                                            ))

    if epoch== opt.epochs:
        
        ckpt_path = os.path.join(opt.save_path,'Model.pth')
        
        torch.save(model.state_dict(),  ckpt_path)
        print('Training finished, model saved at '+ ckpt_path)
        
        dice_cvc_clinic = test(model,opt.test_path1,opt)
        dice_kva = test(model,opt.test_path2,opt)
        dice_cvc_colon = test(model,opt.test_path3,opt)
        dice_etis = test(model,opt.test_path4,opt)
        dice_cvc_300 = test(model,opt.test_path5,opt)
        
        print('#TEST#: dice_cvc_clinic: {:.4f}, dice_kva: {:.4f}, dice_cvc_colon: {:.4f}, dice_etis: {:.4f}, dice_cvc300: {:.4f}'\
                                                                                    .format(dice_cvc_clinic,dice_kva,dice_cvc_colon,dice_etis,dice_cvc_300)) 


if __name__ == '__main__':

    seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./save/', type=str, help='path for output')
    parser.add_argument('--data_root', default='/data/whl/204_whl/Datasets/mydataset', type=str, help='data path')
    parser.add_argument('--img_size', default=352, type=int, help='size of input image')
    parser.add_argument('--test_size', type=int, default=352, help='testing size')
    parser.add_argument('--test_path1', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-ClinicDB', help='path to test dataset')
    parser.add_argument('--test_path2', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/Kvasir', help='path to test dataset')
    parser.add_argument('--test_path3', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-ColonDB',help='path to test dataset')
    parser.add_argument('--test_path4', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/ETIS-LaribPolypDB',help='path to test dataset')
    parser.add_argument('--test_path5', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-300',help='path to test dataset')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay of adamw')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--mask_loss', default=1.0, type=float, help='weight of mask loss')
    parser.add_argument('--trainset', default='TrainDataset', type=str, help='Trainging set')
    parser.add_argument('--decay_rate', default=0.1, type=float, help='dacay rate of learning rate')
    parser.add_argument('--decay_epoch', default=50, type=int, help='decay eopoch of learning rate')

    opt = parser.parse_args()

    # ---- build models ----

    model = DSHNet()
   
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model.cuda()
        

    params = model.parameters()
    
    optimizer = optim.AdamW(  params=params, lr=opt.lr,weight_decay=opt.weight_decay,betas=[0.5,0.999])

    image_root = opt.data_root +'/' +opt.trainset + '/image/'
    gt_root = opt.data_root  +'/'+ opt.trainset + '/mask/'
    
    train_loader = get_loader(image_root, gt_root,batchsize=opt.batch_size,trainsize=opt.img_size)
    total_step = len(train_loader)
    
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epochs+1):
        adjust_lr(optimizer, opt.lr, epoch,opt.decay_rate,opt.decay_epoch)
        print('epoch:{0}-------lr:{1}    {2}'.format(epoch, optimizer.param_groups[0]['lr'],datetime.now()))
        train(train_loader, model, optimizer, epoch, opt)


