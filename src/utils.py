#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch
import json
config = json.load(open('../config.json'))['flickr8k']


# In[86]:


id_to_word = torch.load(config['load']['id2tok'])

def resolve_caption(outputs, prob=True, clip=False):
    # outputs: (batch_size, timesteps, features)
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

