#!/usr/bin/env python
# coding: utf-8

# In[36]:
import json
name = 'flickr8k'
config = json.load(open('../config.json'))[name]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

# In[38]:
import preJSON
preJSON.preprocess(name,True)
preJSON.preprocess(name,False)

id2tok = torch.load(config['load']['id2tok'])
tok2id = torch.load(config['load']['tok2id'])
tokNum = len(tok2id)

def one_hot(batch_size, tok):
    m = torch.zeros(batch_size, tokNum)
    for i in range(batch_size):
        m[i,tok2id[tok] if tok in tok2id else tok2id['<unknown>']]=1.0
    return m

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None or x.shape[0] != self.m.shape[0]):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = torch.autograd.Variable(self.m, requires_grad=False) / (1 - dropout)
        return mask * x

class NIC(nn.Module):
    def __init__(self, num_token, num_hidden=1000):
        super(NIC, self).__init__()
        self.num_token = num_token
        self.num_hidden = num_hidden
        self.cnn = torchvision.models.resnet101(pretrained=True)
        self.fc = nn.Linear(1000,1000)
        self.embedding = nn.Embedding(num_token, self.num_hidden)
        self.lstm1 = nn.LSTM(self.num_hidden, 2000)
        self.lstm1_fc = nn.Linear(2000,num_token)
        self.lstm1_bn = nn.BatchNorm1d(num_token)
        #self.lstm1_drop = nn.Dropout(p=0.5)
        self.lstm1_drop = LockedDropout() #nn.Dropout(p=0.5)

        #self.lstm2 = nn.LSTM(self.num_hidden, 1000)
        #self.lstm2_fc = nn.Linear(1000,num_token)
        #self.lstm2_bn = nn.BatchNorm1d(num_token)
        #self.lstm2_drop = nn.Dropout(p=0.5)

        self.softmax = nn.Softmax(2)
        self.ht = None
        self.ct = None
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, image, target, schedule):
        # schedule = prob of choosing previous input
        # typically while testing, schedule=1
        # while training, schedule = closer to 0 (e.g. 0.25)
        self.lstm1_drop.reset_state()
        batch_size = len(image)
        with torch.no_grad(): image_features = self.cnn(image)
        image_features = self.fc(image_features)
        _, (self.ht, self.ct) = self.lstm1(image_features.view(1, batch_size, -1))
        
        if target is not None: embedded_target = self.embedding(target)
        outputs = [one_hot(batch_size,'<start>').view(1,batch_size,-1).cuda()]
        for t in range(seq_len-1):
            if random.uniform(0,1)<schedule:
                feed = torch.argmax(outputs[-1],2).view(batch_size,1)
                feed = self.embedding(feed)
            else: feed = embedded_target[:, t, :]
            feed = feed.view(1, batch_size, -1)
            #embedded_target_time_t = embedded_target[:, t, :].view(1, batch_size, -1)
            output, (self.ht, self.ct) = self.lstm1(feed, (self.ht, self.ct))
            output = self.lstm1_fc(output.view(batch_size,-1))
            output = self.lstm1_bn(output)
            output = self.lstm1_drop(output).view(1,batch_size,-1)
            #output = self.softmax(output)
            outputs.append(output)
        outputs = torch.cat(outputs) # (time_steps,batch_size,features)
        outputs = outputs.permute(1,0, 2).contiguous() # (batch_size,time_steps,features)
        if target is not None: loss = self.loss(outputs.view(batch_size*seq_len, -1),
                target.view(batch_size*seq_len))
        else: loss = 0
        return loss, outputs
    def beam(self, image, k, idx, st):
        # pass in a single image
        sht, sct = self.ht[:,idx,:].view(1,1,-1), self.ct[:,idx,:].view(1,1,-1)
        front = [[st,1,sht,sct,None]]
        for t in range(seq_len-1):
            que = []
            for i,(y, prob, ht, ct, prev) in enumerate(front):
                h = self.embedding(y.view(1,1))
                h, (ht, ct) = self.lstm1(h.view(1,1,-1),(ht,ct))
                h = self.lstm1_fc(h.view(1,-1))
                h = self.lstm1_bn(h)
                h = self.lstm1_drop(h).view(1,1,-1)
                h = self.softmax(h).view(-1)
                que.append([[j,h[j]*prob,ht,ct,front[i]] for j in h.argsort()[-k:]])
            que = [x for lis in que for x in lis]
            front = sorted(que,key=lambda x:x[1])[-k:]
        pvt = front[-1]
        out = []
        while pvt != None:
            out.append(pvt[0])
            pvt = pvt[4]
        return torch.tensor(out[::-1])
    def batch_beam(self, image, k):
        st = torch.tensor(tok2id['<start>']).cuda()
        batch_size = len(image)
        with torch.no_grad(): h = self.cnn(image)
        h = self.fc(h)
        _, (self.ht, self.ct) = self.lstm1(h.view(1,batch_size,-1))
        return torch.stack([self.beam(img,k,idx,st) for idx, img in enumerate(image)])
        

# In[40]:
modelRt = '../model/NIC.model'
learning_rate = 0.1
if False:
    sv = torch.load(modelRt)
    model, opt = sv['model'].cuda(), sv['opt']
else:
    model = NIC(tokNum).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate,momentum=0.9)

# In[41]:
import Loader
from torchvision import transforms
import random

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*random.uniform(0, 1),
            contrast=0.1*random.uniform(0, 1),
            saturation=0.1*random.uniform(0, 1),
            hue=0.1*random.uniform(0, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#transform = transforms.Compose([
#    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
#    transforms.ToTensor()
#])
#test_transform = transforms.Compose([
#    transforms.CenterCrop(224),
#    transforms.ToTensor()
#])

batch_size, seq_len = 256, 50
loader = Loader.Loader(name=name,
    transform=transform, seq_len=seq_len, batch_size=batch_size,shuffle=True)
tester = Loader.Loader(name=name+'.test',
    transform=test_transform, seq_len=seq_len, batch_size=batch_size,shuffle=True)


import utils

epoch_num = 0
print('Train:')
for epoch in range(epoch_num):
    for i, (xs,ys) in enumerate(loader):
        xs, ys = xs.cuda(), ys.cuda()
        loss, out = model(xs,ys,schedule=0.25)
        opt.zero_grad()
        loss.backward()
        opt.step()
    idxLis = np.random.choice(batch_size,3)
    print('epoch: ', epoch,utils.resolve_caption(out[:3],name,True,True,True))
    print('loss: ', loss)
    if epoch%3==2: torch.save({'opt':opt, 'model':model},modelRt)

# Test
beam_num = 50
print('Test:')
model.eval()

for i, (xs,ys) in enumerate(tester):
    xs, ys = xs.cuda(), ys.cuda()
    _, out = model(xs,ys,schedule=1.0)
    #out = model.batch_beam(xs,beam_num)
    if i%3 == 1:
        print('gen/true',i,':')
        idxLis = np.random.choice(len(out),3)
        for gen,truth in zip(utils.resolve_caption(out[idxLis],name,False,True,True),
                utils.resolve_caption(ys[idxLis],name,False,False,True)):
            print("gen: ",gen)
            print('truth: ',truth)

import BLEU
import dataset
# Eval
print('BLEU:')
total, score = 0, 0
bleu = BLEU.BLEU(name)
evalTester = dataset.get_test_loader(name, batch_size)
for i, (idLis,xs) in enumerate(evalTester):
    xs = xs.cuda()
    _, ys = model(xs,None,schedule=1)
    #ys = model.batch_beam(xs,beam_num)
    for i, y in zip(idLis,ys.cpu()):
        score += bleu(i, torch.argmax(y,1))
        #score += bleu(i, y)
        total += 1
score /= total
print(score)


