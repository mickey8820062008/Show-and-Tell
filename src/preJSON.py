import torch
import json
    
def preprocess(dataset='flickr8k', train=True):
    if not train: dataset+='.test'
    config = json.load(open('../config.json'))[dataset]

    f = open(config['annRt'],'r')
    imgToCap = {k:v for line in f.readlines() for k,v in json.loads(line).items()}
    prLis = [(img, cap.lower().split())
            for img,capLis in imgToCap.items() for cap in capLis]
    if train:
        # count and validate tokens
        token_pool = ['<undefined>', '<unknown>', '<start>', '<end>']
        token_count = {}
        
        for i, (img, cap) in enumerate(prLis):
            for tok in cap:
                if tok in token_count: token_count[tok] += 1
                else: token_count[tok] = 1
        
        token_pool.extend([key for key, value in token_count.items() if value >= 5])
        tok2id = {token: id for id, token in enumerate(token_pool)}
    else:
        token_pool = torch.load(config['load']['id2tok'])
        tok2id = torch.load(config['load']['tok2id'])
    
    def parse(img, cap):

        def getIdx(tok):
            if tok in tok2id: return tok2id[tok]
            else: return tok2id['<unknown>']
        cap = [tok2id['<start>']]+[getIdx(tok) for tok in cap]+[tok2id['<end>']]
        return img, cap
    prLis = [parse(*x) for x in prLis]

    # save preprocessed data
    torch.save(tok2id, config['save']['tok2id'])
    torch.save(token_pool, config['save']['id2tok'])
    torch.save(prLis, config['save']['prLis'])

