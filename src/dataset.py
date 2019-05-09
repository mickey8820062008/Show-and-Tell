import torch
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image
from vision import VisionDataset

class TestLoader(VisionDataset):
    def __init__(self, root, ann_file):
        super(TestLoader, self).__init__(root)
        self.root = root
        self.ann_file = ann_file
        f = open(ann_file, 'r')
        self.data = [k for line in f.readlines() for k,v in json.loads(line).items()]
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image = Image.open(self.root + self.data[idx]).convert('RGB')
        image = self.transform(image)
        
        return image
        
def get_test_loader(name, batch_size):
    config = json.load(open('../config.json'))[name+'.test']
    dataset = TestLoader(root=config['imgRt'], ann_file=config['annRt'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader
