import torch
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

class BLEU:
    def __init__(self, name):
        config = json.load(open('../config.json','r'))[name+'.test']
        self.img2caps = torch.load(config['load']['img2cap'])
        self.c = SmoothingFunction()
    def __call__(self, imgid, predict, method=1):
        references = self.img2caps[imgid]
        candidate = np.array(predict)
        return sentence_bleu(references, candidate,smoothing_function=[
                self.c.method1,self.c.method2,self.c.method3,self.c.method4,
                self.c.method5,self.c.method6,self.c.method7][method+1])
