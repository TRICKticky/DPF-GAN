from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import time
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from .utils import truncated_noise

#使用dataloader只获取一个批次的数据，在tensorboard进行同样样本的展示对比
def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
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

    return imgs, sent_emb, words_embs, mask

# def get_one_batch_data(dataloader, text_encoder, args):
#     data = next(iter(dataloader))
#     # imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
#     imgs = data[0]
#     imgs = Variable(imgs).cuda()
#     keys = data[3]
#     # n * batch * 18 * 1
#     captions = data[1]
#     # print(np.array([item.cpu().numpy() for item in captions]).shape)
#     # captions = torch.tensor(np.array([item.cpu().numpy() for item in captions]))
#     captions = torch.stack(captions)
#
#     # n * batch
#     caption_lens = data[2]
#     # print(np.array([item.cpu().numpy() for item in caption_lens]).shape)
#     # caption_lens = torch.tensor(np.array([item.cpu().numpy() for item in caption_lens]))
#     caption_lens = torch.stack(caption_lens)
#     # n * batch * 256
#     sent_emb = []
#     for cap, len in zip(captions, caption_lens):
#         sent_embs, _ = prepare_embs(cap, len, text_encoder)
#         sent_emb.append(sent_embs)
#
#     # n * batch * 256
#     # sent_emb = torch.tensor(np.array([item.cpu().numpy() for item in sent_emb]))
#     sent_emb = torch.stack(sent_emb)
#     # print(sent_emb.shape)
#     # batch * n * 256
#     sent_emb = torch.transpose(sent_emb, 1, 0)
#     # sent_emb = torch.from_numpy(sent_emb)
#
#
#     return imgs, sent_emb


def get_fix_data(train_dl, test_dl, text_encoder, args):
    fixed_image_train, fixed_sent_train, fixed_word_train, fixed_mask_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, fixed_sent_test, fixed_word_test, fixed_mask_test = get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)

    if fixed_word_train.shape[2] != fixed_word_test.shape[2]:
        if fixed_word_train.shape[2] > fixed_word_test.shape[2]:
            padding = fixed_word_train.shape[2] - fixed_word_test.shape[2]
            padding0 = torch.zeros((fixed_word_train.shape[0], fixed_word_train.shape[1], padding))
            padding1 = torch.zeros((fixed_word_train.shape[0], padding))
            fixed_word_test = torch.cat((fixed_word_test, padding0), dim=2)
            fixed_mask_test = torch.cat((fixed_mask_test, padding1==0), dim=1)
            fixed_words = torch.cat((fixed_word_train, fixed_word_test), dim=0)
            fixed_mask = torch.cat((fixed_mask_train, fixed_mask_test), dim=0)
        else:
            padding = fixed_word_test.shape[2] - fixed_word_train.shape[2]
            padding0 = torch.zeros((fixed_word_train.shape[0], fixed_word_train.shape[1], padding))
            padding1 = torch.zeros((fixed_word_train.shape[0], padding))
            fixed_word_train = torch.cat((fixed_word_train, padding0), dim=2)
            fixed_mask_train = torch.cat((fixed_mask_train, padding1==0), dim=1)
            fixed_words = torch.cat((fixed_word_train, fixed_word_test), dim=0)
            fixed_mask = torch.cat((fixed_mask_train, fixed_mask_test), dim=0)
    else:
        fixed_words = torch.cat((fixed_word_train, fixed_word_test), dim=0)
        fixed_mask = torch.cat((fixed_mask_train, fixed_mask_test), dim=0)

    if args.truncation==True:
        noise = truncated_noise(fixed_image.size(0), args.z_dim, args.trunc_rate)
        fixed_noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)

    return fixed_image, fixed_sent, fixed_noise, fixed_words, fixed_mask

#change
def prepare_data(data, text_encoder):
    imgs, captions, caption_lens, keys, _ = data
    #根据句子的长度进行排序(降序)，利于输入rnn，sorted_cap_idxs记录了原来的位置
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    #进行降序排列是方便输入rnn，所以最终得到句子特征和单词特征后还需要还原顺序
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    # imgs = Variable(imgs).cuda()
    imgs = Variable(imgs)

    return imgs, sent_emb, words_embs, keys

def prepare_embs(captions, caption_lens, text_encoder):

    #根据句子的长度进行排序(降序)，利于输入rnn，sorted_cap_idxs记录了原来的位置
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    #进行降序排列是方便输入rnn，所以最终得到句子特征和单词特征后还需要还原顺序
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)

    return sent_emb, words_embs
#change
def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)#排序并返回与原来位置的索引
    captions = captions[sorted_cap_indices].squeeze()#根据索引，将原来的captions进行排序
    # captions = Variable(captions).cuda()
    captions = Variable(captions)
    # sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens)

    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        #判断text_encoder是否有'module'这个属性
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))#初始化为batch_size个rnn序列，每个序列长度由len决定
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        #自然语言处理由于句子长短不一，所以在形成一个batch的时候需要对齐，不然无法形成batch_size * len * 1这样的张量
        #而图像经过随机裁剪至固定大小使得可以让一个batch的size对齐融合形成一个张量
        #注意这里的words_embs的第二维度不唯一，因为一个批次的len不一致
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split='train', transform=None, args=None):
        self.transform = transform
        self.word_num = args.TEXT.WORDS_NUM
        self.embeddings_num = args.TEXT.CAPTIONS_PER_IMAGE
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(self.data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(self.data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    #将非法字符变成空格
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                #数据集每个图像对应的描述句子的数量是一致的，不一致的话数据集就有问题
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
                #返回一个列表的列表，每个子列表为一个句子，每个子列表的元素为一个单词，即将整个数据集不同图像的描述全部按filenames的顺序打包整合在一堆，获取的时候10个为一组的获取即可
        return all_captions
    #将数据集的所有文字囊括为一个字典
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1
        #提取所有key，即单词字符串
        vocab = [w for w in word_counts if word_counts[w] >= 0]
        #将单词以数字编码表示，编码从1开始
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        #将描述全部替换为字典里的数字索引
        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_DAMSM.pickle')
        #一个filename对应一个图像
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        #如果下载了别人预先整理好的包，直接加载即可，否则重新本地处理数据集
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
                #class_id：用数字来表示不同类别
        else:#没有图像-类别索引，就将每个图像认定为一个类别
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        #用0填充，即代表字典里的<end>
        x = np.zeros((self.word_num, 1), dtype='int64')
        x_len = num_words
        #按照最大单词数量来对齐，不足的补0到self.num_words，多的随机抽取剪切到self.num_words
        if num_words <= self.word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.word_num
        return x, x_len

    def __getitem__(self, index):
        #key为图像的名字
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #bounding box不是辅助网络学习的，没有融入网络，是预处理（learn what and where to draw作者表示这个标签和key point是可以辅助网络学习图像生成的，如果融入网络的话）
        #预先裁取图像的主干(bounding box是数据集固定的)，然后将其作为ground truth，再进行一些transform(随机裁剪)
        #裁取主干只是为了在transform随机裁剪的时候能更大概率裁剪到图像的主要部分
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        if self.dataset_name.find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
        elif self.dataset_name.find('flower') != -1:
            if self.split=='train':
                img_name = '%s/oxford-102-flowers/images/%s.jpg' % (data_dir, key)
            else:
                img_name = '%s/oxford-102-flowers/images/%s.jpg' % (data_dir, key)
        elif self.dataset_name.find('CelebA') != -1:
            if self.split=='train':
                img_name = '%s/image/CelebA-HQ-img/%s.jpg' % (data_dir, key)
            else:
                img_name = '%s/image/CelebA-HQ-img/%s.jpg' % (data_dir, key)
        else:
            img_name = '%s/images/%s.jpg' % (data_dir, key)

        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        #index代表第几个图片，一个图片n个句子，那么新的句子索引应该为n * index + random(0, n)
        new_sent_ix = index * self.embeddings_num + sent_ix
        #caps为固定长度，即self.word_num，但是cap_len不固定，小于等于这个self.word_num阈值的是多少就是多少，大于的全部按照这个阈值来算(随机剪切提取)
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, key, sent_ix

    # #获取一个图片的所有描述句子
    # def __getitem__(self, index):
    #     #key为图像的名字
    #     key = self.filenames[index]
    #     cls_id = self.class_id[index]
    #     #bounding box不是辅助网络学习的，没有融入网络，是预处理（learn what and where to draw作者表示这个标签和key point是可以辅助网络学习图像生成的，如果融入网络的话）
    #     #预先裁取图像的主干(bounding box是数据集固定的)，然后将其作为ground truth，再进行一些transform(随机裁剪)
    #     #裁取主干只是为了在transform随机裁剪的时候能更大概率裁剪到图像的主要部分
    #     if self.bbox is not None:
    #         bbox = self.bbox[key]
    #         data_dir = '%s/CUB_200_2011' % self.data_dir
    #     else:
    #         bbox = None
    #         data_dir = self.data_dir
    #     #
    #     if self.dataset_name.find('coco') != -1:
    #         if self.split=='train':
    #             img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
    #         else:
    #             img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
    #     elif self.dataset_name.find('flower') != -1:
    #         if self.split=='train':
    #             img_name = '%s/oxford-102-flowers/images/%s.jpg' % (data_dir, key)
    #         else:
    #             img_name = '%s/oxford-102-flowers/images/%s.jpg' % (data_dir, key)
    #     elif self.dataset_name.find('CelebA') != -1:
    #         if self.split=='train':
    #             img_name = '%s/image/CelebA-HQ-img/%s.jpg' % (data_dir, key)
    #         else:
    #             img_name = '%s/image/CelebA-HQ-img/%s.jpg' % (data_dir, key)
    #     else:
    #         img_name = '%s/images/%s.jpg' % (data_dir, key)
    #
    #     imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
    #     # # random select a sentence
    #     # sent_ix = random.randint(0, self.embeddings_num)
    #     # #index代表第几个图片，一个图片n个句子，那么新的句子索引应该为n * index + random(0, n)
    #     # new_sent_ix = index * self.embeddings_num + sent_ix
    #
    #     sent_ix = [ index * self.embeddings_num+i for i in range(self.embeddings_num) ]
    #
    #     caps = []
    #     cap_len = []
    #     #caps为固定长度，即self.word_num，但是cap_len不固定，小于等于这个self.word_num阈值的是多少就是多少，大于的全部按照这个阈值来算(随机剪切提取)
    #     for ix in sent_ix:
    #         cap, len = self.get_caption(ix)
    #         caps.append(cap)
    #         cap_len.append(len)
    #
    #     return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)


