import torch
import json
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

class BLEU:
    def __init__(self, name):
        config = json.load(open('../config.json','r'))[name+'.test']
        self.img2caps = torch.load(config['load']['img2cap'])

    def __call__(self, imgid, predict):
        references = self.img2caps[imgid]
        candidate = np.array(predict)
        return sentence_bleu(references, candidate)
