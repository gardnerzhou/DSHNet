import os

import random
from datetime import datetime

# import Testing
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
# from Evaluation import main as eval_main
from Models.network import DSHNet
from dataloader import get_loader,test_dataset
# from tensorboardX import SummaryWriter
from torch import optim 
from torch.autograd import Variable
# import torch_dct as DCT
# import wandb

# from scipy.spatial.distance import directed_hausdorff
step = 0
best_mae = 1
best_epoch = 0
project_name='MyNet_record'

def test(model, path,args):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, args.img_size)
    b=0.0
    hd=0.0
    smooth = 1e-8
    # print('[validation_size]',test_loader.size)
    for i in range(test_loader.size):
        image, gt,image_dct = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image_dct=image_dct.cuda()
        


        mask, mid_pre = model(image,image_dct)
        pred_mask, pred_masks = mask
        res=pred_mask
        # res5= model(image)
        res = F.interpolate(res, size=gt.shape,  mode='bilinear')
        # res = F.interpolate(res, size=gt.shape,  mode='nearest')
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


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if epoch%decay_epoch==1 and epoch>1:
        # decay = decay_rate ** (epoch // decay_epoch)
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
mse_loss = nn.MSELoss().cuda()

def seed_torch():
    # seed = int(time.time()*256) % (2**32-1)
    seed = 11219116
    # 保存随机种子
    print("~~~random_seed:", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True



def main( args,best_dice):
    
    print(">>>>>>> Training")
    seed_torch()

    args.save_model_dir = args.save_path + '/{}/'.format(project_name)
    os.makedirs(args.save_model_dir,exist_ok=True)


    model = DSHNet()
    
    os.makedirs('./save/{}/best_cvc_clinic'.format(project_name),exist_ok=True)
    os.makedirs('./save/{}/best_kva'.format(project_name),exist_ok=True)
    os.makedirs('./save/{}/best_cvc_colon'.format(project_name),exist_ok=True)
    os.makedirs('./save/{}/best_etis'.format(project_name),exist_ok=True)
    os.makedirs('./save/{}/best_cvc_300'.format(project_name),exist_ok=True)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("all_model_parameters: ", num_params / 1e6)

    model.train()
    model.cuda()

    params = model.parameters()

    optimizer = optim.AdamW(  params=params, lr=args.lr,weight_decay=args.weight_decay,betas=[0.5,0.999])


    image_root = args.data_root +'/' +args.trainset + '/image/'
    gt_root = args.data_root  +'/'+ args.trainset + '/mask/'

    train_loader = get_loader(image_root, gt_root,batchsize=args.batch_size,trainsize=args.img_size)


    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)


    for epoch in range(1, args.epochs + 1):
        adjust_lr(optimizer, args.lr, epoch,args.decay_rate,args.decay_epoch)



        print('epoch:{0}-------lr:{1}    {2}'.format(epoch, optimizer.param_groups[0]['lr'],datetime.now()))
        train_epoch(train_loader, model, optimizer, epoch, args,best_dice,project_name)



def train_epoch(train_loader, model,  optimizer, epoch,  args,best_dice,project_name):
    
    global step
    model.train()
    loss_all_sum = 0.
    loss_sal_sum = 0.
    loss_freq_sum = 0.
    epoch_step = 0

    for i, data_batch in enumerate(train_loader, start=1):
        
        optimizer.zero_grad()
        images, gts,dt_masks,image_dct = data_batch
        images = images.cuda()
        gts = gts.cuda()
        # contours = contours.cuda()
        dt_masks = dt_masks.cuda()
        image_dct=image_dct.cuda()
        images, gts,dt_masks,image_dct= Variable(images.cuda()), \
                                        Variable(gts.cuda()),\
                                        Variable(dt_masks.cuda()),\
                                        Variable(image_dct.cuda())
                                        
        mask,  mid_pre = model(images,image_dct)
        pred_mask, pred_masks = mask
        

        loss_sal = structure_loss(pred_mask, gts)
        # loss_sal += structure_loss(mid_pre[0], F.interpolate(gts,mid_pre[0].size()[2:], mode='nearest')) #* 0.5
        # loss_sal += structure_loss(mid_pre[1], F.interpolate(gts,mid_pre[1].size()[2:], mode='nearest')) #* 0.5
        # loss_sal += structure_loss(mid_pre[2], F.interpolate(gts,mid_pre[2].size()[2:], mode='nearest')) #* 0.8
        # loss_sal += structure_loss(mid_pre[3], F.interpolate(gts,mid_pre[3].size()[2:], mode='nearest')) #* 0.8
        loss_sal += structure_loss(F.interpolate(mid_pre[0],gts.size()[2:],mode='bilinear',align_corners=False),gts) #* 0.5
        loss_sal += structure_loss(F.interpolate(mid_pre[1],gts.size()[2:],mode='bilinear',align_corners=False),gts)# * 0.5
        loss_sal += structure_loss(F.interpolate(mid_pre[2],gts.size()[2:],mode='bilinear',align_corners=False),gts) #* 0.8
        # loss_sal += structure_loss(mid_pre[3], F.interpolate(gts,mid_pre[3].size()[2:],mode='bilinear',align_corners=False)) * 0.8
        
        loss_sal_aux = 0.0

        for m in pred_masks:
            map_i = m
            mask_i = dt_masks
            # mask_i = F.interpolate(dt_masks, size=[m.shape[-2], m.shape[-1]], mode="nearest")
            # mask_i = F.interpolate(dt_masks, size=[m.shape[-2], m.shape[-1]], mode="bilinear")
            map_i = F.interpolate(map_i, size=[mask_i.shape[-2], mask_i.shape[-1]], mode="bilinear")
            loss_sal_aux += mse_loss(map_i, mask_i)

        loss_sal_aux = loss_sal_aux * args.aux_loss

        loss = loss_sal + loss_sal_aux

        loss.backward()

        clip_gradient(optimizer, args.clip)
        optimizer.step()
        
        step += 1
        epoch_step += 1
        loss_all_sum += loss.cpu().data.item()
        loss_sal_sum += loss_sal.cpu().data.item()
                                                                        
    print('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_all_AVG: {:.4f},Loss_sal_AVG: {:.4f}, Loss_freq_AVG: {:.4f}'.format(epoch,
                                                                                            args.epochs,
                                                                                            loss_all_sum / epoch_step,
                                                                                            loss_sal_sum / epoch_step,
                                                                                            loss_freq_sum/epoch_step
                                                                                            ))

    if epoch== args.epochs:
        dice_cvc_clinic=test(model,args.val_path1,args)
        dice_kva=test(model,args.val_path2,args)
        dice_cvc_colon=test(model,args.val_path3,args)
        dice_etis=test(model,args.val_path4,args)
        dice_cvc_300=test(model,args.val_path5,args)
        results=[dice_cvc_clinic,dice_kva,dice_cvc_colon,dice_etis,dice_cvc_300]
        
        ns=['cvc_clinic','kva','cvc_colon','etis','cvc_300']
        for i in range(5):
            if results[i]>best_dice[i]:
                best_dice[i]=results[i]
                torch.save(model.state_dict(),  './save/{0}/best_{1}/Model_best_{1}.pth'.format(project_name,ns[i]))
                fp = open('./save/{}/best_{}/best.txt'.format(project_name,ns[i]),'w')
                fp.write(' {:0.5f} '.format(results[i])+'\n')
                fp.close()

        print('#TEST#: dice_cvc_clinic: {:.4f}, dice_kva: {:.4f}, dice_cvc_colon: {:.4f}, dice_etis: {:.4f}, dice_cvc300: {:.4f}'\
                                                                                    .format(dice_cvc_clinic,dice_kva,dice_cvc_colon,dice_etis,dice_cvc_300)) 

