from glob import glob
from tqdm import tqdm

import argparse
import zipfile
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, default=None)

    args = parser.parse_args()

    runs_dir = os.path.join('../runs', f'{args.name}')

    zip_path = os.path.join('../runs', f'{args.name}', f'{args.name}.zip')

    zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    imgs = glob(f'{runs_dir}/*.png')

    for img in tqdm(imgs):
        zip.write(img)

    zip.close()