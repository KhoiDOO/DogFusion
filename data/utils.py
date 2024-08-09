from torch.utils.data import DataLoader
from .ds import DogDS

import numpy as np
import torch

def denorm(img:torch.Tensor):
    iv_img = (img * 255).permute(1, -1, 0).numpy().astype(np.uint8)

    return iv_img

def get_ds(args):
    ds = DogDS(root=args.root, size=args.size, horflip=args.horflip)
    return DataLoader(ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)