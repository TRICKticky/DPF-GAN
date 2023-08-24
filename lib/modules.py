from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np

from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from .utils import truncated_noise
from .utils import mkdir_p, get_rank
#sys.path.append("..")
from .datasets import TextImgDataset as Dataset
from .datasets import prepare_data, encode_tokens, prepare_embs
from models.inception import InceptionV3

from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
#
# def conv_context(in_planes, out_planes):
#     "1x1 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
#                      padding=0, bias=False)


def get_groundTruth_attn(input, context, mask):
    """
        input: batch x idf x ih x iw (queryL=ihxiw)
        context: batch x cdf x sourceL
    """
    ih, iw = input.size(2), input.size(3)
    queryL = ih * iw
    batch_size, sourceL = context.size(0), context.size(2)

    # --> batch x queryL x idf
    target = input.view(batch_size, -1, queryL)
    targetT = torch.transpose(target, 1, 2).contiguous()
    # batch x cdf x sourceL --> batch x cdf x sourceL x 1
    # sourceT = context.unsqueeze(3)
    sourceT = context

    #局部图像特征和单词特征维度均为256，无需改变
    # # --> batch x idf x sourceL
    # sourceT = self.conv_context(sourceT).squeeze(3)

    # Get attention
    # (batch x queryL x idf)(batch x idf x sourceL)
    # -->batch x queryL x sourceL
    attn = torch.bmm(targetT, sourceT)
    # --> batch*queryL x sourceL
    attn = attn.view(batch_size * queryL, sourceL)
    if mask is not None:
        # batch_size x sourceL --> batch_size*queryL x sourceL
        mask = torch.transpose(mask, 1, 2)
        mask = mask.repeat(1, queryL, 1).cuda()
        # mask = mask.repeat(1, queryL, 1)

        mask = mask.view(batch_size * queryL, sourceL)
        # 将0替换为负无穷
        attn.data.masked_fill_(mask.data, -float('inf'))
    # my_changed: 加入dim参数
    attn = nn.Softmax()(attn)  # Eq. (2)
    # --> batch x queryL x sourceL
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL（batch_size * sentence_len * (17 * 17））
    attn = torch.transpose(attn, 1, 2).contiguous()

    attn = attn.view(batch_size, -1, ih, iw)

    return attn

from torch.autograd import Variable
############   modules   ############
def train(dataloader, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, args, writer, new_NetG=False, new_NetC=False):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC = netG.train(), netD.train(), netC.train()

    #addition
    loss_writer = writer
    G_loss = 0.0
    D_loss = 0.0

    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        # # prepare_data，keys为该批次的所有图像的名字字符串
        imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)

        # imgs = data[0]
        # imgs = Variable(imgs).cuda()
        # keys = data[3]
        # # n * batch * 18 * 1
        # captions = data[1]
        # # print(np.array([item.cpu().numpy() for item in captions]).shape)
        # # captions = torch.tensor(np.array([item.cpu().numpy() for item in captions]))
        # captions = torch.stack(captions)
        # # n * batch
        # caption_lens = data[2]
        # # print(np.array([item.cpu().numpy() for item in caption_lens]).shape)
        # # caption_lens = torch.tensor(np.array([item.cpu().numpy() for item in caption_lens]))
        # caption_lens = torch.stack(caption_lens)
        # # n * batch * 256
        # sent_emb = []
        # for cap, lens in zip(captions, caption_lens):
        #     sent_embs, _ = prepare_embs(cap, lens, text_encoder)
        #     sent_emb.append(sent_embs)
        # # n * batch * 256
        # # sent_emb = torch.tensor(np.array([item.cpu().numpy() for item in sent_emb]))
        # sent_emb = torch.stack(sent_emb)
        # # print(sent_emb.shape)
        # # batch * n * 256
        # sent_emb = torch.transpose(sent_emb, 1, 0)
        # # print(sent_emb.shape)

        # # 使用image_encoder获取局部图像编码 256 * 17 * 17
        # real_local_features, _ = image_encoder(imgs)
        # imgs, captions, caption_lens, keys = data
        captions = data[1]
        # batch_size * dataset_max_len
        mask = (captions == 0)
        # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
        # num_words为这一批次的最长句子长度
        num_words = words_embs.size(2)
        # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        # # batch_size * sentence_len * (17 * 17）
        # attn = get_groundTruth_attn(real_local_features, words_embs, mask)

        #3 * 256 * 256
        imgs = imgs.to(device).requires_grad_()
        #256 * 1
        sent_emb = sent_emb.to(device).requires_grad_()
        words_embs = words_embs.to(device).requires_grad_()

        # synthesize fake images
        # 训练网络时的z是非截断的正态分布(训练的数据量相对测试的多很多)，而测试的时候使用的是截断的正态分布，因为测试的数据量较小，为了防止破坏分布的对称性，所以采用截断分布
        noise = torch.randn(batch_size, z_dim).to(device)
        if new_NetG:
            fake = netG(noise, sent_emb)
        else:
            fake = netG(noise, sent_emb)


        # # 256 * (本批次中句子最长的len)
        # words_embs = words_embs.to(device).requires_grad_()
        # # batch_size * sentence_len * (17 * 17）
        # attn = attn.to(device).requires_grad_()
        # mask = mask.to(device).requires_grad_()
        # predict real
        #256 * 4 * 4，256 * 16 * 16
        # real_features, real_features_local = netD(imgs)
        # real_features, real_features_local = netD(imgs)
        real_features = netD(imgs)
        real_features_local=None
        # #使用image_encoder获取局部图像编码 256 * 17 * 17
        # real_local_features = image_encoder(imgs)
        #返回netC的输出（对图像的打分）和损失loss
        pred_real, errD_real = predict_loss(netC, real_features, img_feature_local=real_features_local, text_feature=sent_emb, text_feature_local=None, mask=None, negtive=False, new_NetC=new_NetC)

        #使用简单的移位篡改一个批次里的顺序，使得配对紊乱(这一操作有利于模型判断不同图片的区别，提升健硕性，学到多样性，有些图片比较相似，但是有细微的区别，增加这个可以帮助模型放大这个区别)
        mis_features = torch.cat((real_features[1:], real_features[0:1]), dim=0)
        mis_features_local=None
        # mis_features_local = torch.cat((real_features_local[1:], real_features_local[0:1]), dim=0)
        _, errD_mis = predict_loss(netC, mis_features, img_feature_local=mis_features_local, text_feature=sent_emb, text_feature_local=None, mask=None, negtive=True, new_NetC=new_NetC)

        # # synthesize fake images
        # #训练网络时的z是非截断的正态分布(训练的数据量相对测试的多很多)，而测试的时候使用的是截断的正态分布，因为测试的数据量较小，为了防止破坏分布的对称性，所以采用截断分布
        # noise = torch.randn(batch_size, z_dim).to(device)
        # if new_NetG:
        #     fake, s = netG(noise, sent_emb, sent_ix)
        # else:
        #     fake = netG(noise, sent_emb)

        # detach创建一个新的指向同一tensor的变量，使用这个变量不会反响传播，不修改图结构(但是修改这个值bp会报错，.data就不会报错，所以detach更安全)
        # detach_直接分离出这个tensor节点，将其作为叶子节点，修改了图结构
        # 即这个地方的fake_features只对后面的netD有传播作用，对前面的netG没有传播作用
        # fake_features, fake_features_local = netD(fake.detach())
        fake_features = netD(fake.detach())

        _, errD_fake = predict_loss(netC, fake_features, img_feature_local=None, text_feature=sent_emb, text_feature_local=None, mask=None, negtive=True, new_NetC=new_NetC)

        # MA-GP
        errD_MAGP = MA_GP(imgs, sent_emb, pred_real)
        #
        # whole D loss
        errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        #addition：追加显示loss
        D_loss += errD.item()

        # update D
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()
        # update G
        # 交互式的迭代，先根据G生成的图像更新D，在用更新后的D对G进行更新
        fake_features = netD(fake)
        fake_features_local=None
        # fake_features, fake_features_local = netD(fake)

        if new_NetC:
            # output = netC(fake_features, fake_features_local, sent_emb, words_embs)
            output = netC(fake_features, sent_emb)
        else:
            output = netC(fake_features, sent_emb)
        # sim = MAP(image_encoder, fake, sent_emb).mean()
        # 取一个批次的平均打分值为loss，打分越低，loss越大
        errG = -output.mean()# - sim
        # addition：追加显示loss
        G_loss += errG.item()

        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()

    #addition：在tensorboard显示loss
    loss_writer.add_scalar('D_loss', D_loss/len(dataloader), epoch)
    loss_writer.add_scalar('G_loss', G_loss/len(dataloader), epoch)

    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()



def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp, truncation, trunc_rate, times):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
        sent_emb = sent_emb.to(device)
        words_embs = words_embs.to(device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            #截断噪声
            if truncation==True:
                noise = truncated_noise(batch_size, z_dim, trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size, z_dim).to(device)
            # fake_imgs = netG(noise, sent_emb)
            fake_imgs = netG(noise, sent_emb, words_embs)
        for j in range(batch_size):
            s_tmp = '%s/single/%s' % (save_dir, keys[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            ######################################################
            # (3) Save fake images
            ######################################################            
            if multi_gpus==True:
                filename = 'd%d_s%s.png' % (get_rank(),times)
            else:
                filename = 's%s.%d' % (stamp, data[4][j])
            fullpath = '%s_%s.png' % (s_tmp, filename)
            im.save(fullpath)


def test(dataloader, text_encoder, img_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    # fid = calculate_fid(dataloader, text_encoder, img_encoder, netG, device, m1, s1, epoch, max_epoch, \
    #                     times, z_dim, batch_size, truncation, trunc_rate)
    #times为10.即采样10次，每次采样整个数据集，计算整个测试集的fid，计算10次
    fid, IS_mean, IS_std = calculate_fid_is(dataloader, text_encoder, img_encoder, netG, device, m1, s1, epoch, max_epoch, \
                        times, z_dim, batch_size, truncation, trunc_rate)

    return fid, IS_mean, IS_std


def calculate_fid(dataloader, text_encoder, img_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size() error
    n_gpu = 1
    dl_length = dataloader.__len__()#总的batch次数（注意：dataloader的长度和dataset的长度不是一个东西，dataset的长度是整个数据集的大小，而dataloader的长度为dataset的长度/batch_size）
    imgs_num = dl_length * n_gpu * batch_size * times
    #所有图片的输出特征向量
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):#总共10次，因为dataloader在采样的时候是丢弃了最后一个凑不够一个batch_size的batch，所以采样整个数据集多次以达到公平性（每一次采样的数据有差别）
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)

            # # n * batch * 18 * 1
            # captions = data[1]
            # # print(np.array([item.cpu().numpy() for item in captions]).shape)
            # # captions = torch.tensor(np.array([item.cpu().numpy() for item in captions]))
            # captions = torch.stack(captions)
            #
            # # n * batch
            # caption_lens = data[2]
            # # print(np.array([item.cpu().numpy() for item in caption_lens]).shape)
            # # caption_lens = torch.tensor(np.array([item.cpu().numpy() for item in caption_lens]))
            # caption_lens = torch.stack(caption_lens)
            # # n * batch * 256
            # sent_emb = []
            # for cap, lens in zip(captions, caption_lens):
            #     sent_embs, _ = prepare_embs(cap, lens, text_encoder)
            #     sent_emb.append(sent_embs)
            #
            # # n * batch * 256
            # # sent_emb = torch.tensor(np.array([item.cpu().numpy() for item in sent_emb]))
            # sent_emb = torch.stack(sent_emb)
            # # print(sent_emb.shape)
            # # batch * n * 256
            # sent_emb = torch.transpose(sent_emb, 1, 0)


            # # 使用image_encoder获取局部图像编码 256 * 17 * 17
            # real_local_features, _ = img_encoder(imgs)
            # imgs, captions, caption_lens, keys = data
            # captions = data[1]
            # # batch_size * dataset_max_len
            # mask = (captions == 0)
            # # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
            # # num_words为这一批次的最长句子长度
            # num_words = words_embs.size(2)
            # # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]
            # # batch_size * sentence_len * (17 * 17）
            # attn = get_groundTruth_attn(real_local_features, words_embs, mask)

            sent_emb = sent_emb.to(device)
            words_embs = words_embs.to(device)
            # attn = attn.to(device)
            captions = data[1]
            # batch_size * dataset_max_len
            mask = (captions == 0)
            # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
            # num_words为这一批次的最长句子长度
            num_words = words_embs.size(2)
            # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            ######################################################
            # (2) Generate fake images
            ######################################################
            # sent_ix = [random.randint(0, sent_emb.shape[1] - 1) for i in range(sent_emb.shape[0])]
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else: 
                    noise = torch.randn(batch_size, z_dim).to(device)
                # fake_imgs = netG(noise, sent_emb, words_embs, mask)
                fake_imgs = netG(noise, sent_emb)
                

                #规范化生成图像并将其输入V3网络
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    #将每个通道上的size变为1*1（batch_size*channel_size*w*h）
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # # concat pred from multi GPUs
                # output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                # pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                # pred_arr[start:end] = pred_all.cpu().data.numpy()
                pred_arr[start:end] = pred.squeeze(-1).squeeze(-1).cpu().data.numpy()

            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


#######################################
#########inception_score.py
#######################################
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""


from PIL import Image

from inception.slim import slim
import numpy as np
import tensorflow as tf


import math
import os.path
import scipy.misc
import imageio
# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir',
                           '../birds_valid299/model.ckpt',
                           """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string('image_folder', 
							'../test/valid/single',
							"""Path where to load the images """)

tf.app.flags.DEFINE_integer('num_classes', 50,      # 20 for flowers
                            """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10,
                            """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999



 

def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)


def get_inception_score(sess, images, pred_op):
    splits = FLAGS.splits
    # assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        # print("%d of %d batches" % (i, n_batches))
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) -
              np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)




def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
              images,
              dropout_keep_prob=0.8,
              num_classes=num_classes,
              is_training=for_training,
              restore_logits=restore_logits,
              scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


def IS(is_imgs):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [FLAGS.batch_size, 299, 299, 3],
                    name='inputs')
                # print(inputs)

                logits, _ = inference(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = \
                    tf.slice(logits, [0, 1],
                             [FLAGS.batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, FLAGS.checkpoint_dir)
                print('Restore the model from %s).' % FLAGS.checkpoint_dir)
                images = is_imgs

                return get_inception_score(sess, images, pred_op)

#######################################
#########inception_score.py
#######################################



def calculate_fid_is(dataloader, text_encoder, img_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size() error
    n_gpu = 1
    dl_length = dataloader.__len__()#总的batch次数（注意：dataloader的长度和dataset的长度不是一个东西，dataset的长度是整个数据集的大小，而dataloader的长度为dataset的长度/batch_size）
    imgs_num = dl_length * n_gpu * batch_size * times
    #所有图片的输出特征向量
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    is_imgs = []
    for time in range(times):#总共10次，因为dataloader在采样的时候是丢弃了最后一个凑不够一个batch_size的batch，所以采样整个数据集多次以达到公平性（每一次采样的数据有差别）
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)

            # # n * batch * 18 * 1
            # captions = data[1]
            # # print(np.array([item.cpu().numpy() for item in captions]).shape)
            # # captions = torch.tensor(np.array([item.cpu().numpy() for item in captions]))
            # captions = torch.stack(captions)
            #
            # # n * batch
            # caption_lens = data[2]
            # # print(np.array([item.cpu().numpy() for item in caption_lens]).shape)
            # # caption_lens = torch.tensor(np.array([item.cpu().numpy() for item in caption_lens]))
            # caption_lens = torch.stack(caption_lens)
            # # n * batch * 256
            # sent_emb = []
            # for cap, lens in zip(captions, caption_lens):
            #     sent_embs, _ = prepare_embs(cap, lens, text_encoder)
            #     sent_emb.append(sent_embs)
            #
            # # n * batch * 256
            # # sent_emb = torch.tensor(np.array([item.cpu().numpy() for item in sent_emb]))
            # sent_emb = torch.stack(sent_emb)
            # # print(sent_emb.shape)
            # # batch * n * 256
            # sent_emb = torch.transpose(sent_emb, 1, 0)


            # # 使用image_encoder获取局部图像编码 256 * 17 * 17
            # real_local_features, _ = img_encoder(imgs)
            # imgs, captions, caption_lens, keys = data
            # captions = data[1]
            # # batch_size * dataset_max_len
            # mask = (captions == 0)
            # # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
            # # num_words为这一批次的最长句子长度
            # num_words = words_embs.size(2)
            # # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]
            # # batch_size * sentence_len * (17 * 17）
            # attn = get_groundTruth_attn(real_local_features, words_embs, mask)

            sent_emb = sent_emb.to(device)
            words_embs = words_embs.to(device)
            # attn = attn.to(device)
            captions = data[1]
            # batch_size * dataset_max_len
            mask = (captions == 0)
            # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
            # num_words为这一批次的最长句子长度
            num_words = words_embs.size(2)
            # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            ######################################################
            # (2) Generate fake images
            ######################################################
            # sent_ix = [random.randint(0, sent_emb.shape[1] - 1) for i in range(sent_emb.shape[0])]
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else: 
                    noise = torch.randn(batch_size, z_dim).to(device)
                # fake_imgs = netG(noise, sent_emb, words_embs, mask)
                fake_imgs = netG(noise, sent_emb)
            
                #规范化生成图像并将其输入V3网络
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    #将每个通道上的size变为1*1（batch_size*channel_size*w*h）
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # # concat pred from multi GPUs
                # output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                # pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                # pred_arr[start:end] = pred_all.cpu().data.numpy()
                pred_arr[start:end] = pred.squeeze(-1).squeeze(-1).cpu().data.numpy()

            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()

            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                is_imgs.append(im)


    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)


    IS_mean, IS_std = IS(is_imgs=is_imgs)

    return fid_value, IS_mean, IS_std


def eval(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
                times, z_dim, batch_size, truncation=True, trunc_rate=0.86):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    # n_gpu = dist.get_world_size()
    n_gpu = 1

    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)

            # # n * batch * 18 * 1
            # captions = data[1]
            # # print(np.array([item.cpu().numpy() for item in captions]).shape)
            # # captions = torch.tensor(np.array([item.cpu().numpy() for item in captions]))
            # captions = torch.stack(captions)
            #
            # # n * batch
            # caption_lens = data[2]
            # # print(np.array([item.cpu().numpy() for item in caption_lens]).shape)
            # # caption_lens = torch.tensor(np.array([item.cpu().numpy() for item in caption_lens]))
            # caption_lens = torch.stack(caption_lens)
            # # n * batch * 256
            # sent_emb = []
            # for cap, lens in zip(captions, caption_lens):
            #     sent_embs, _ = prepare_embs(cap, lens, text_encoder)
            #     sent_emb.append(sent_embs)
            #
            # # n * batch * 256
            # # sent_emb = torch.tensor(np.array([item.cpu().numpy() for item in sent_emb]))
            # sent_emb = torch.stack(sent_emb)
            # # print(sent_emb.shape)
            # # batch * n * 256
            # sent_emb = torch.transpose(sent_emb, 1, 0)

            sent_emb = sent_emb.to(device)
            # words_embs = words_embs.to(device)
            # captions = data[1]
            # # batch_size * dataset_max_len
            # mask = (captions == 0)
            # # 文本编码器输出的单词级别的向量：batch * f_dim * len_sentence
            # # num_words为这一批次的最长句子长度
            # num_words = words_embs.size(2)
            # # 将mask与words_embs对齐(将尾部多余的0清除，之前添加补全至最大长度是为了dataloader统一处理)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]
            ######################################################
            # (2) Generate fake images
            ######################################################
            # sent_ix = [random.randint(0, sent_emb.shape[1] - 1) for i in range(sent_emb.shape[0])]
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise, sent_emb)
                # fake_imgs = netG(noise, sent_emb)
                if save_imgs==True:
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # # concat pred from multi GPUs
                # output = list(torch.empty_like(pred) for _ in range(n_gpu))
                # dist.all_gather(output, pred)
                # pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                # pred_arr[start:end] = pred_all.cpu().data.numpy()
                pred_arr[start:end] = pred.squeeze(-1).squeeze(-1).cpu().data.numpy()

            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            #print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(time*dl_len*batch_size+batch_size*batch_n+j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

#按批次采样
def sample_one_batch(noise, sent, words, mask, netG, multi_gpus, epoch, img_save_dir, writer):
    #生成采样图像
    fixed_results = generate_samples(noise, sent, words, mask, netG)
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        if writer!=None:
            #以网格形式整合展示一批次的生成图像结果
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, value_range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)


def generate_samples(noise, caption, words, mask, model):
    #model为netG
    with torch.no_grad():
        # fake = model(noise, caption.cuda(), words.cuda(), mask)
        fake = model(noise, caption.cuda())

    return fake


#########   MAGP   ########
#使用netC的输出对ground_truth进行偏导
def MA_GP(img, sent, out):
    #out关于(img, sent)求导，out = f（img，sent）
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),#预计算的参数值
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)
    #将两个梯度融合为一个梯度（取两个梯度的平方然后开方），梯度越小，loss越小
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

#计算real_image和text（正例）、fake_image和text（负例）、mis_image和text（负例）之间语义对齐性
#直接进行语义对齐，收敛更快
def predict_loss(predictor, img_feature, img_feature_local, text_feature, text_feature_local, mask=None, negtive=None, new_NetC=False):
    if new_NetC == True:
        output = predictor(img_feature, text_feature)
    else:
        output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err

#这里的合页损失的超参dis设置为1.0，距离大于这个值的就是正例，损失均为0（因为关心的是近距离容易混淆的，远距离的损失直接为0），小于这个值的就是负例，距离越远loss越大
#参考svm的二分类问题的合页损失函数
def hinge_loss(output, negtive):
    if negtive==False:
        #如果是正例，则1-output应该<0（out_put>1），损失err就会为0，反之，就是归类错误，距离越远，loss越大
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        #如果是负例，则out_put<-1.0，即1.0 + output<0代表分类正确，损失err就会为0，反之，就是归类错误，距离越远，loss越大
        err = torch.nn.ReLU()(1.0 + output).mean()
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


