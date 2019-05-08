import torch
import json
from nltk.translate.bleu_score import sentence_bleu

class BLEU:
    def __init__(self, name):
        self.name = name
        self.config = json.load('CONFIG_PATH')[name+'.test']
        self.img2caps = torch.load(config['load']['img2caps'])

    def __call__(self, imgid, predict):
        references = self.img2caps[imgid]
        candidate = predict
        return sentence_bleu(references, candidate)
