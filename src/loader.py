#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

# Re-using a generator
class Reuse(object):
    def __init__(self, g, *p1, **p2):
        self.p1 = p1
        self.p2 = p2
        self.g = g
        self._g = g(*p1,**p2)
    def __iter__(self): return self
    def __next__(self):
        try: return self._g.__next__()
        except StopIteration:
            self._g = self.g(*self.p1,**self.p2)
            raise StopIteration()

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
        

def Loader(data, tok_num, seq_len=50, shuffle=True, batch_size=1):
    toTensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
        toTensor
    ])
    
    def getImg(img_id): return transform(Image.open('./dataset/Flickr8k/Images/'+img_id).convert('RGB')).reshape(1,3, 224, 224)
    cache = Cache(getImg,1000)
    def inst(img_id, cap):
        cap = [tok if i<len(cap) else 0 for i,tok in enumerate(cap)]
        return cache[img_id], cap

    def _load():
        random.shuffle(data)
        idata = iter(data)
        try: 
            while True:
                imgs, caps = zip(*[inst(*idata.__next__()) for i in range(batch_size)])
                yield torch.cat(imgs), torch.cat(caps)
        except StopIteration: pass
    return Reuse(_load)

