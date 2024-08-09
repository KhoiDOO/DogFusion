from ds import DogDS
from utils import denorm

import imageio.v3 as iio
import cv2
import os

COLOR = (0, 255, 0)
THICKNESS = 2


if __name__ == '__main__':
    sv_dir = os.path.join(os.getcwd(), 'viz')
    os.makedirs(sv_dir, exist_ok=True)

    ds = DogDS(
        root='/media/mountHDD3/data_storage/standford_dog',
        size=128,
        horflip=False
    )

    print(f'Number of Sample: {len(ds)}')

    for idx in range(10):
        img, bbox, lbl = ds[idx]

        bbox = bbox.numpy()[0]

        start_point = [bbox[0], bbox[1]]
        end_point = [bbox[2], bbox[3]]

        iv_img = denorm(img=img)

        iv_img = cv2.rectangle(iv_img, start_point, end_point, COLOR, THICKNESS)

        iio.imwrite(os.path.join(sv_dir, f'{idx}.jpg'), iv_img)