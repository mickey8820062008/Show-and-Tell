#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import torch
import torch.utils.data as data
from collections import defaultdict
from PIL import Image
import glob
import random
import torchvision
import torchvision.transforms as transforms
#from Loader import Loader


# In[2]:


# load flickr8k dataset
prefix = '../Flickr_Data/'
flickr8k_dataset = torchvision.datasets.Flickr30k(
    root=prefix+'Images',
    ann_file=prefix+'Flickr_TextData/Flickr8k.token'
)


# In[3]:


# count and validate tokens
token_pool = ['<undefined>', '<unknown>', '<start>', '<end>']
token_count = {}

for img_id in flickr8k_dataset.ids:
    captions = flickr8k_dataset.annotations[img_id]
    for caption in captions:
        tokens = caption.lower().split()
        for token in tokens:
            if token not in token_count:
                token_count[token] = 1
            else:
                token_count[token] += 1

token_pool.extend([key for key, value in token_count.items() if value >= 5])
token_to_id = {token: id for id, token in enumerate(token_pool)}


# In[4]:


# change tokens into ids
flickr8k_data = []

for img_id in flickr8k_dataset.ids:
    captions = flickr8k_dataset.annotations[img_id]
    for caption in captions:
        tokens = caption.lower().split()
        ids = [token_to_id[token] if token in token_to_id else token_to_id['<unknown>'] for token in tokens]
        ids = [token_to_id['<start>']] + ids + [token_to_id['<end>']]
        flickr8k_data.append((img_id, ids))


# In[5]:

from vision import *
import numpy as np

class FlickrDataset(VisionDataset):
    def __len__(self): return len(self.prLis)
    def __init__(self, transform, seq_len, root, prLis):
        self.seq_len = seq_len
        self.transform = transform
        self.prLis = prLis
        self.root = root
        self.toTensor = torchvision.transforms.ToTensor()
        super(FlickrDataset, self).__init__(root+'Images/')
    def __getitem__(self, idx):
        img_path = self.root+self.prLis[idx][0]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        cap = self.prLis[idx][1]
        cap = [cap[i] if i<len(cap) else 0 for i in range(self.seq_len)]
        return img, torch.LongTensor(cap)

from torchvision import transforms

toTensor = transforms.ToTensor()
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    toTensor
])



train = FlickrDataset(transform,50,prefix,flickr8k_data)
dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=4)
# # check caption length
# import numpy as np
# caption_length = []
# for img_id, caption in flickr8k_data:
#     caption_length.append(len(caption))
    
# print('Min: {}'.format(np.min(caption_length)))
# print('Max: {}'.format(np.max(caption_length)))

### USAGE!!!!!
#for i, (xs,ys) in enumerate(dataloader):
#    pass

# In[6]:


# save preprocessed data
#torch.save(token_pool, 'preprocessed_data/flickr8k_id_to_word.pylist')
#torch.save(flickr8k_data, 'preprocessed_data/flickr8k_data.pylist')

