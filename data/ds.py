from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob

import xml.etree.ElementTree as ET
import random
import torch
import PIL
import os


class DogDS(Dataset):
    def __init__(self, root:str, size = 64) -> None:
        super().__init__()

        self.root = root
        self.size = size
        self.img_dir = os.path.join(self.root, 'Images')
        self.ann_dir = os.path.join(self.root, 'Annotation')

        self.classes = sorted(os.listdir(self.img_dir))
        self.cls2idx = {x : idx for idx, x in enumerate(self.classes)}

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor()
            ]
        )

        self.imgs = sorted(glob(os.path.join(self.img_dir, '*/*')), key=lambda x: random.random())
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]

        img = PIL.Image.open(img_path)
        img = self.transform(img)

        raw_cls = img_path.split('/')[-2]
        cls = self.cls2idx[raw_cls]

        filename = os.path.basename(img_path).split('.')[0]

        annot_path = os.path.join(self.ann_dir, raw_cls, filename)

        # xmin, xmax, ymin, ymax = self.get_bbox(annot_path=annot_path)

        return img, cls


    def get_bbox(self, annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()

        bbox_object = root.find('object').find('bndbox')

        xmin = int(bbox_object.find('xmin').text)
        xmax = int(bbox_object.find('xmax').text)
        ymin = int(bbox_object.find('ymin').text)
        ymax = int(bbox_object.find('ymax').text)

        return xmin, xmax, ymin, ymax
