from glob import glob
from tqdm import tqdm
from ds import DogDS

import PIL.Image
import xml.etree.ElementTree as ET
import scipy.io as sio
import PIL
import os

if __name__ == '__main__':
    ds_dir = '/media/mountHDD3/data_storage/standford_dog'
    train_data_path = os.path.join(ds_dir, 'train_data.mat')
    test_data_path = os.path.join(ds_dir, 'test_data.mat')

    # train_mat = sio.loadmat(train_data_path)
    # test_mat = sio.loadmat(test_data_path)

    # train_feature_data = train_mat['train_fg_data']
    # train_data_content = train_mat['train_data']

    # print(train_feature_data.shape, train_data_content.shape)

    # annos = glob(os.path.join(ds_dir, 'Annotation', '*' , '*'))

    # sample_anno = annos[0]
    # print(sample_anno)

    # tree = ET.parse(sample_anno)
    # root = tree.getroot()

    # print(tree)
    # print(root.find('object').find('bndbox').find('xmin').text)

    # imgs = glob(os.path.join(ds_dir, 'Images/*/*'))

    # Hs, Ws = [], []

    # for img in tqdm(imgs):
    #     x = PIL.Image.open(img)

    #     H, W = x.size

    #     Hs.append(H)
    #     Ws.append(W)
    
    # print(f'Hmax: {max(Hs)} - Wmax: {max(Ws)}')
    # print(f'Hmin: {min(Hs)} - Wmin: {min(Ws)}')

    # ds = DogDS(root=ds_dir)

    # dct = {}

    # for idx in tqdm(range(len(ds))):
    #     _, lbl = ds[idx]

    #     if lbl in dct:
    #         dct[lbl] += 1
    #     else:
    #         dct[lbl] = 0
    
    # print(dct.keys())

    # dct = {}

    # ds = DogDS(root=ds_dir)

    # for idx in tqdm(range(len(ds))):
    #     img, lbl, path = ds[idx]
    
    #     C, _, _ = img.shape

    #     if C in dct:
    #         dct[C] += 1
    #     else:
    #         dct[C] = 1
        
    #     if C == 4:
    #         print(path)
    
    # print(dct)