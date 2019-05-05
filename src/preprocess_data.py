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

imgRt = '../Flickr_Data/Images/'
annRt = '../Flickr_Data/Flickr_TextData/Flickr8k.token'

flickr8k_dataset = torchvision.datasets.Flickr30k(root=imgRt, ann_file=annRt)


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

# save preprocessed data
torch.save(token_pool, './flickr8k_id_to_word.pylist')
torch.save(flickr8k_data, './flickr8k_data.pylist')

# Usage:
# token_pool = torch.load(...)
# flickr8k_data = torch.load(...)
