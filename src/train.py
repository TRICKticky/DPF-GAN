import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
import pprint

import torch
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils
import multiprocessing as mp
#code文件夹
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
#该根目录优先被搜索导入
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_model_opt,save_models,load_npz, params_count
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import sample_one_batch as sample, test as test, train as train, get_groundTruth_attn as attn
from lib.datasets import get_fix_data

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    # default = '../cfg/model/coco.yml'
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/bird.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: 4)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,
                        help='input imsize')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--train', type=bool, default=True,
                        help='if train model')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='resume epoch')
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the model for resume training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if multi-gpu training under ddp')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args):
    time_stamp = get_time_stamp()
    #标记
    stamp = '_'.join([str(args.model),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])

    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME), stamp)
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        #创建日志文件夹、模型参数保存文件夹、采样图片保存文件夹
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        #tensorboard会在log_dir下创建事件文件，记录writer的记录事件数据
        writer = SummaryWriter(log_dir)
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words
    #其image_encoder和text_encoder为预训练好的模型，模型加载于cpu中，不占gpu显存
    new_NetC = False
    new_NetG = True
    image_encoder, text_encoder, netG, netD, netC = prepare_models(args, new_NetG=new_NetG, new_NetC=new_NetC)
    #将进行裁剪、旋转操作的图片，句子编码和噪声提取出来(其中图片是训练和测试集拼接的（24(batchsize)*2），句子编码也是拼接的（24*2），噪声是截取的（24*2）)
    #这里的fixed_img对象和后面每次迭代后的sample(fixed_sent, fixed_z)输出的对象，在tensorboard界面用作对比
    fixed_img, fixed_sent, fixed_z, fixed_words, fixed_mask = get_fix_data(train_dl, valid_dl, text_encoder, args)
    # fixed_real_local_features, _  = image_encoder(fixed_img)
    # fixed_attn = attn(fixed_real_local_features, fixed_words, fixed_mask)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(fixed_img.cpu(), nrow=8, normalize=True)
        writer.add_image('fixed images', fixed_grid, 0)
        #在tensorboard上展示为fixed images，本地保存为z.png
        #这个图片为ground truth，与后面的每个迭代的sample函数（用fixed_sent, fixed_z去生成）的结果作对比
        img_name = 'z.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
    # prepare optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=0.0004, betas=(0.0, 0.9))
    m1, s1 = load_npz(args.npz_path)
    # load from checkpoint
    strat_epoch = 1
    #中断训练后重新迭代训练，修改resume_epoch为对应的上次保存模型参数的epoch数值，并从对应的模型参数文件读取参数加载入网络
    if args.resume_epoch!=1:
        strat_epoch = args.resume_epoch+1
        path = osp.join(args.resume_model_path, 'state_epoch_%03d.pth'%(args.resume_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_model_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        #打印当前的各个超参并保存在yaml文件中（logs文件夹下）
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")
    # Start training
    test_interval,gen_interval,save_interval = args.test_interval,args.gen_interval,args.save_interval
    #torch.cuda.empty_cache()
    for epoch in range(strat_epoch, args.max_epoch, 1):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        # training 
        args.current_epoch = epoch
        torch.cuda.empty_cache()
        train(train_dl, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, args, writer, new_NetG=new_NetG, new_NetC=new_NetC)
        #torch.cuda.empty_cache()
        # save 根据间隔(10)进行保存模型参数
        if epoch%save_interval==0:
            save_models(netG, netD, netC, optimizerG, optimizerD, epoch, args.multi_gpus, args.model_save_file)
        # sample 根据间隔(1)采样生成的图像
        if epoch%gen_interval==0:
            #一批次的采样(这里的一批次包括24*2，训练集和测试集一个集合抽取一批次)
            sample(fixed_z, fixed_sent, fixed_words, fixed_mask, netG, args.multi_gpus, epoch, args.img_save_dir, writer)
        # end epoch
        # test 根据间隔(10)计算fid指标
        if epoch%test_interval==0:
            #释放torch缓存器占有但未使用的显存（第一次进入cuda时，缓存器会占用一部分显存），方便其他程序使用显存
            torch.cuda.empty_cache()
            #用测试集计算fid(在测试集中的采样次数为10次)
            # fid = test(valid_dl, text_encoder, image_encoder, netG, args.device, m1, s1, epoch, args.max_epoch, \
            #             args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
            fid, IS_mean, IS_std = test(valid_dl, text_encoder, image_encoder, netG, args.device, m1, s1, epoch, args.max_epoch, \
                        args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
        
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            if epoch%test_interval==0:
                writer.add_scalar('FID', fid, epoch)
                print('The %d epoch FID: %.2f'%(epoch,fid))
                writer.add_scalars('IS', {'IS_mean': IS_mean, 'IS_std': IS_std}, epoch)
                print('The %d epoch IS: {{mean:%.2f std:%.2f}}'%(epoch,IS_mean,IS_std))
                
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print('*'*40)
        #torch.cuda.empty_cache()
        

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    # print(args)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            #gpu_id为0，可查看nvidia-smi查看设备id（单个gpu）
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

