#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision


# In[2]:


# load flickr8k dataset
flickr8k_dataset = torchvision.datasets.Flickr30k(
    root='daaset/Flickr8k/Images',
    ann_file='dataset/Flickr8k/Flickr8k.token'
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


# # check caption length
# import numpy as np
# caption_length = []
# for img_id, caption in flickr8k_data:
#     caption_length.append(len(caption))
    
# print('Min: {}'.format(np.min(caption_length)))
# print('Max: {}'.format(np.max(caption_length)))


# In[6]:


# save preprocessed data
torch.save(token_pool, 'preprocessed_data/flickr8k_id_to_word.pylist')
torch.save(flickr8k_data, 'preprocessed_data/flickr8k_data.pylist')

