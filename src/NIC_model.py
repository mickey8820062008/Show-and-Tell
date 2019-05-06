#!/usr/bin/env python
# coding: utf-8

# In[36]:
import json
config = json.load(open('../config.json'))['flickr8k']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# In[38]:
tok2id = torch.load(config['save']['tok2id'])
tokNum = len(tok2id)

def one_hot(batch_size, tok):
    m = torch.zeros(batch_size, tokNum)
    for i in range(batch_size):
        m[i,tok2id[tok] if tok in tok2id else tok2id['<unknown>']]=1.0
    return m

class NIC(nn.Module):
    def __init__(self, num_token, num_hidden=1000):
        super(NIC, self).__init__()
        self.num_token = num_token
        self.num_hidden = num_hidden
        self.cnn = torchvision.models.resnet34(pretrained=True)
        self.embedding = nn.Embedding(num_token, self.num_hidden)
        self.lstm = nn.LSTM(self.num_hidden, self.num_token)
        self.softmax = nn.Softmax(2)
        self.ht = None
        self.ct = None
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, image, target):
        batch_size, timesteps = target.shape
        with torch.no_grad(): image_features = self.cnn(image)
        _, (self.ht, self.ct) = self.lstm(image_features.view(1, batch_size, -1))
        
        embedded_target = self.embedding(target)
        
        outputs = [one_hot(batch_size,'<start>').view(1,batch_size,-1).cuda()]
        for t in range(timesteps-1):
            embedded_target_time_t = embedded_target[:, t, :].view(1, batch_size, -1)
            output, (self.ht, self.ct) = self.lstm(embedded_target_time_t)
            output = self.softmax(output)
            outputs.append(output)
        outputs = torch.cat(outputs) # (time_steps,batch_size,features)
        outputs = outputs.permute(1,0, 2).contiguous() # (batch_size,time_steps,features)
        loss = self.loss(outputs.view(batch_size*timesteps, -1), target.view(batch_size*timesteps))
            
        return loss, outputs
        

# In[40]:

learning_rate = 0.001
if False:
    sv = torch.load('../model/NIC.model')
    model, opt = sv['model'].cuda(), sv['opt']
else:
    model = NIC(tokNum).cuda()
    opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# In[41]:
import loader
import torchvision.transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor()
])
batch_size, seq_len = 128, 50
loader = loader.Loader(
    transform=transform, seq_len=seq_len, batch_size=batch_size,shuffle=True)

import utils

epoch_num = 15
print('Train:')
for epoch in range(epoch_num):
    for i, (xs,ys) in enumerate(loader):
        xs, ys = xs.cuda(), ys.cuda()
        loss, out = model(xs,ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('epoch: ',epoch,utils.resolve_caption(out[:3],True))

torch.save({'opt':opt, 'model':model},'../model/NIC.model')

# Test
print('Test:')
model.eval()
for i, (xs,ys) in enumerate(loader):
    xs, ys = xs.cuda(), ys.cuda()
    _, out = model(xs,ys)
    if i%50 == 49:
        print('gen/true:',i)
        for a,b in zip(utils.resolve_caption(out,True),
                utils.resolve_caption(ys,False)):
            print('gen: ',a)
            print('true: ',b)

