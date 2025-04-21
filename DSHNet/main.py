import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3" 


import Training as train
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='./save/', type=str, help='path for output')

    parser.add_argument('--init_method', default='tcp://127.0.0.1:32222', type=str, help='init_method')
    parser.add_argument('--data_root', default='/data/whl/204_whl/Datasets/mydataset', type=str, help='data path')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--test_size', type=int, default=352, help='testing size')
    
    parser.add_argument('--val_path1', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-ClinicDB', help='path to train dataset')
    parser.add_argument('--val_path2', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/Kvasir', help='path to train dataset')
    parser.add_argument('--val_path3', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-ColonDB',help='path to train dataset')
    parser.add_argument('--val_path4', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/ETIS-LaribPolypDB',help='path to train dataset')
    parser.add_argument('--val_path5', type=str,
                        default='/data/whl/204_whl/Datasets/mydataset/TestDataset/CVC-300',help='path to train dataset')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--query_num', default=3, type=int, help='query-num')

    parser.add_argument('--aux_loss', default=1.0, type=float, help='batch_size')
    parser.add_argument('--freq_loss', default=0, type=float, help='batch_size')

    parser.add_argument('--trainset', default='TrainDataset', type=str, help='Trainging set')

    parser.add_argument('--decay_rate', default=0.1, type=float, help='batych_size')
    parser.add_argument('--decay_epoch', default=50, type=int, help='batch_size')

    args = parser.parse_args()
    torch.cuda.set_device(0)

    best_dice=[0,0,0,0,0]

    ns=['cvc_clinic','kva','cvc_colon','etis','cvc_300']


    for i in range(10):
        print('training iter {}'.format(i+1))
        train.main(args,best_dice)

