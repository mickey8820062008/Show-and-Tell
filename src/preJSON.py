import torch
import json

capRt = '../ann/captions.txt'

f = open(capRt,'r')
imgToCap = {k:v for line in f.readlines() for k,v in json.loads(line).items()}
prLis = [(img, cap.lower()) for img,capLis in imgToCap.items() for cap in capLis]

# count and validate tokens
token_pool = ['<undefined>', '<unknown>', '<start>', '<end>']
token_count = {}

for i, (img, cap) in enumerate(prLis):
    cap = cap.split()
    for tok in cap:
        if tok in token_count: token_count[tok] += 1
        else: token_count[tok] = 1
    prLis[i] = (img,cap)

token_pool.extend([key for key, value in token_count.items() if value >= 5])
tok2id = {token: id for id, token in enumerate(token_pool)}

def parse(img, cap):
    def getIdx(tok):
        if tok in tok2id: return tok2id[tok]
        else: return tok2id['<unknown>']
    cap = [tok2id['<start>']]+[getIdx(tok) for tok in cap]+[tok2id['<end>']]
    return img, cap

prLis = [parse(*x) for x in prLis]

# save preprocessed data
torch.save(tok2id, '../preJSON/tok2id.pylist')
torch.save(token_pool, '../preJSON/id2tok.pylist')
torch.save(prLis, '../preJSON/prLis.pylist')

