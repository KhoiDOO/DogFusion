from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from PIL import Image
from glob import glob
from torch import nn

import xml.etree.ElementTree as ET
import random
import torch
import PIL
import os


class DogDS(Dataset):
    def __init__(self, root:str, size:int = 64, horflip:bool=False) -> None:
        super().__init__()

        self.root = root
        self.size = size
        self.horflip = horflip

        self.img_dir = os.path.join(self.root, 'Images')
        self.ann_dir = os.path.join(self.root, 'Annotation')

        self.classes = sorted(os.listdir(self.img_dir))
        self.cls2idx = {x : idx for idx, x in enumerate(self.classes)}

        self.transform = T.Compose(
            [
                T.ToImage(),
                T.RandomHorizontalFlip(),
                T.Resize([self.size, self.size]),
                T.ToDtype(torch.float32, scale=True),
                T.PILToTensor()
            ]
        ) if self.horflip else T.Compose(
            [
                T.ToImage(),
                T.Resize([self.size, self.size]),
                T.ToDtype(torch.float32, scale=True),
                T.PILToTensor()
            ]
        )

        self.imgs = sorted(glob(os.path.join(self.img_dir, '*/*.jpg')), key=lambda x: random.random())
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, BoundingBoxes, int]:
        img_path = self.imgs[index]

        img = Image.open(img_path)

        raw_cls = img_path.split('/')[-2]
        cls = self.cls2idx[raw_cls]

        # filename = os.path.basename(img_path).split('.')[0]

        # annot_path = os.path.join(self.ann_dir, raw_cls, filename)

        # xmin, ymin, xmax, ymax = self.get_bbox(annot_path=annot_path)

        # bboxes = BoundingBoxes([[xmin, ymin, xmax, ymax]], format=BoundingBoxFormat.XYXY, canvas_size=img.size)

        # out_img, out_bboxes = self.transform(img, bboxes)

        out_img = self.transform(img)

        return out_img, cls


    def get_bbox(self, annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()

        bbox_object = root.find('object').find('bndbox')

        xmin = int(bbox_object.find('xmin').text)
        xmax = int(bbox_object.find('xmax').text)
        ymin = int(bbox_object.find('ymin').text)
        ymax = int(bbox_object.find('ymax').text)

        return xmin, ymin, xmax, ymax
