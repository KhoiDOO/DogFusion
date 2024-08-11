from glob import glob
from tqdm import tqdm
from PIL import Image  

import imageio.v3 as iio
import argparse
import zipfile
import cv2
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, default=None)

    args = parser.parse_args()

    runs_dir = os.path.join('../runs', f'{args.name}')

    vid_path = os.path.join('../runs', f'{args.name}', f'{args.name}.avi')

    imgs = glob(f'{runs_dir}/*.png')

    im = Image.open(imgs[0]) 
    width, height = im.size 

    video = cv2.VideoWriter(vid_path, 0, 1, (width, height)) 

    for img in tqdm(imgs):
        video.write(iio.imread(img))  

    cv2.destroyAllWindows() 
    video.release()