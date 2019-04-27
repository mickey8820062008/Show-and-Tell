#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch


# In[86]:


def resolve_caption(outputs, reference='./preprocessed_data/flickr8k_id_to_word.pylist'):
    # outputs: (batch_size, timesteps, features)
    id_to_word = torch.load(reference)
    
    captions = []
    
    for output in outputs:
        caption = ' '.join([id_to_word[id] for id in output])
        captions.append(caption)
    # captions: (batch_size)
    return captions

