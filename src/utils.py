#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch
import json


# In[86]:


# e.g
# name = 'flickr8k'
# train = whether using training dataset for id_to_word? (no difference for now)
# prob = input as a probablisitic vector?
# clip = disgard token after '<end>'?
def resolve_caption(outputs, name, train, prob=True, clip=False):
    # outputs: (batch_size, timesteps, features)
    if not train: name+='.test'
    config = json.load(open('../config.json'))[name]
    id_to_word = torch.load(config['load']['id2tok'])
    
    captions = []
    for output in outputs:
        if prob: caption = ' '.join([id_to_word[torch.argmax(id)] for id in output])
        else: caption = ' '.join([id_to_word[id] for id in output])
        if clip:
            try: caption = caption[:caption.index('<end>')+5]
            except: pass
        captions.append(caption)
        
    # captions: (batch_size)
    return captions

