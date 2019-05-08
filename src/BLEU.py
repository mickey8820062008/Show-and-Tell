import torch
import json
from nltk.translate.bleu_score import sentence_bleu

class BLEU:
    def __init__(self, name):
        self.name = name
        self.config = json.load('../config.json')[name+'.test']
        self.img2caps = torch.load(config['load']['img2cap'])

    def __call__(self, imgid, predict):
        references = self.img2caps[imgid]
        candidate = predict
        return sentence_bleu(references, candidate)
