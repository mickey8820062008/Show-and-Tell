#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

import collections

class Cache:
    def __init__(self, m, cache_size):
        self.cache_size = cache_size
        self.cache = {}
        self.que = collections.deque()
        self.m = m
    def __getitem__(self, x):
        if x not in self.cache:
            y = self.m(x)
            self.cache[x] = y
            self.que.append(x)
        else: y = self.cache[x]
        
        if len(self.que) > self.cache_size:
            xp = self.que.popleft()
            self.cache.pop(xp)
        return y
        

from vision import *
import numpy as np


import json

def Loader(transform, seq_len, batch_size, shuffle, name):
    config = json.load(open('../config.json'))[name]
    imgRt = config['imgRt']
    
    token_pool = torch.load(config['load']['id2tok'])
    flickr8k_data = torch.load(config['load']['prLis'])
    
    class FlickrDataset(VisionDataset):
        def __len__(self): return len(self.prLis)
        def __init__(self, transform, seq_len, prLis):
            self.seq_len = seq_len
            self.transform = transform
            self.prLis = prLis
            self.toTensor = torchvision.transforms.ToTensor()
            super(FlickrDataset, self).__init__(imgRt)
        def __getitem__(self, idx):
            img_path = self.root+self.prLis[idx][0]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            cap = self.prLis[idx][1]
            cap = [cap[i] if i<len(cap) else 0 for i in range(self.seq_len)]
            return img, torch.LongTensor(cap)
    
    from torchvision import transforms
    
    
    data = FlickrDataset(transform,seq_len,flickr8k_data)
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# Usage
# train_loader, test_loader = Loader(..., name='flickr8k')
# for e in range(epoch):
#   for i,(xs,ys) in enumerate(loader):
#     do_stuff()
