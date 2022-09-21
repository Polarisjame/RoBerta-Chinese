#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dataset.py 
@File    ：main.py
@Author  ：Polaris
@Date    ：2022-05-14 1:18
'''

# import sys
#
# sys.path.extend(['D:\\TOOL\\jupyter program', 'D:/TOOL/jupyter program/ShopCLass'])
import torch
from transformers import AutoModel, AutoTokenizer
import re
from model import BertBiLSTMModel
from readfile import loadset
from train import train

pretrained_name = 'uer/roberta-base-finetuned-jd-full-chinese'
tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

filename = r'D:\TOOL\jupyter program\nlpdataset\shops\data_202106c.csv'
batch_size = 16
num_epoch = 3  # 训练轮次
train_ratio = 0.95  # 训练集比例
learning_rate = 0.00003  # 优化器的学习率
use_finetune = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model_name = 'uer/roberta-base-finetuned-jd-full-chinese'
use_dropout = True

trainloader, testloader, tag2id = loadset(filename, batch_size=batch_size, train_rate=train_ratio)
bert_model = BertBiLSTMModel(class_size=len(tag2id),pretrained_name=pretrained_model_name, use_finetune=use_finetune, use_dropout=use_dropout)
train(bert_model, trainloader, testloader, device, learning_rate, num_epoch)

# Predict
bert_model = torch.load(r'D:\TOOL\jupyter program\ShopClass\bert_shop_model2.pth', map_location='cpu')
tag2id = {'文化玩乐': 0, '母婴用品': 1, '百货食品': 2, '服装鞋包': 3, '汽配摩托': 4, '美妆饰品': 5, '家居建材': 6, '运动户外': 7, '手机数码': 8, '家用电器': 9,
          '游戏话费': 10, '其他': 11, '生活服务': 12, '其他商品': 13, '盒马': 14, '数字阅读': 15}
id2tag = {id: tag for tag, id in tag2id.items()}


def coffate_fn(examples):
    sents = []
    for itemname in examples:
        itemname = re.sub('[^\u4e00-\u9fa5]+', '', itemname)
        sents.append(itemname)
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 # return_offsets_mapping=True,
                                 is_split_into_words=False,
                                 max_length=100,
                                 return_tensors="pt")
    return tokenized_inputs


waits = []
with open(filename, encoding='gb18030', errors='ignore') as f:
    for ind, i in enumerate(f):
        if ind == 0:
            continue
        elif ind == 100:
            break
        texts = i.split('\t')
        waits.append((texts[2], texts[8]))

for text, tag in waits:
    inputs = coffate_fn([text])
    bert_output = bert_model(inputs)
    output = bert_output
    pred = torch.max(output, 1)[1]
    print(text, ':', tag, '  predicted: ', id2tag[int(pred)])



