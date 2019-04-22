import torch
import torchvision
import itertools
prefix = '../Flickr_Data/Images/'
flickr_data = torchvision.datasets.Flickr30k(prefix,'../Flickr_Data/Flickr_TextData/Flickr8k.token')
#for i,e in enumerate(flickr_data):
#    if i%10 == 9: print(i//10)

#raise Exception()

cnt, idx = {}, {}
def cntTok(tok, flt=False):
    if flt:
        if tok not in idx: return 0
        return idx[tok]
    if tok not in cnt:
        cnt[tok] = 1
    else:
        cnt[tok] += 1
        if cnt[tok]>=5: idx[tok] = len(idx)+1
    return tok

data = [(i, [cntTok(tok) for tok in sent.lower().split() ]) for i in flickr_data.ids for sent in flickr_data.annotations[i]]
data = [(i, [cntTok(tok,True) for tok in sent]) for i,sent in data]

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

def Loader(data,shuffle=True, batch_size=1):
    toTensor = torchvision.transforms.ToTensor()
    def inst(img, cap):
        return toTensor(Image.open(prefix+img).convert('RGB')), cap
    def _load():
        random.shuffle(data)
        idata = iter(data)
        while True: yield zip(*[inst(*idata.__next__()) for i in range(batch_size)])
    return Reuse(_load)

loader = Loader(data, shuffle=True, batch_size=10)

for epoch in range(2):
    for i, (xs,ys) in enumerate(loader):
        print(i)
