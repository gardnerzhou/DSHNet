import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from lib.network import DSHNet
from dataloader import test_dataset
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./save/Model.pth')


if __name__=='__main__':    
     
    opt = parser.parse_args()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = '/path/to/dataset/TestDataset/{}/'.format(_data_name)
        save_path = './results/{}/'.format(_data_name)
        
        model = DSHNet()
        
        model.cuda()

        model.load_state_dict(torch.load(opt.pth_path))
        model.eval()

        os.makedirs(save_path, exist_ok=True)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

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
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)*255
            res=np.uint8(res)

            imageio.imsave(save_path+name, res)
