#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch


# In[86]:


id_to_word = torch.load('./flickr8k_id_to_word.pylist')

def resolve_caption(outputs, prob=True):
    # outputs: (batch_size, timesteps, features)
    captions = []
    
    for output in outputs:
        if prob: caption = ' '.join([id_to_word[torch.argmax(id)] for id in output])
        else: caption = ' '.join([id_to_word[id] for id in output]) 
        captions.append(caption)
        
    # captions: (batch_size)
    return captions

