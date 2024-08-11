from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm

import numpy as np
import torch
import math
import os


class CFGFIDEvaluation(FIDEvaluation):
    def __init__(self, 
                 batch_size, 
                 dl, 
                 sampler, 
                 channels=3, 
                 accelerator=None, 
                 stats_dir="./results", 
                 device="cuda", 
                 num_fid_samples=50000, 
                 inception_block_idx=2048):
        assert num_fid_samples >= batch_size, 'the number of sample taken into account for FID must be larger than batch size'
        super().__init__(batch_size, dl, sampler, channels, accelerator, stats_dir, device, num_fid_samples, inception_block_idx)
    
    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(f"Stacking Inception features for {self.n_samples} samples from the real dataset.")
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples, _ = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True
    
    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        self.sampler.eval()

        classes = torch.randint(0, 120, (self.n_samples, )).to(self.device)

        batch_size = self.n_samples if self.n_samples < self.batch_size else self.batch_size

        batches = num_to_groups(self.n_samples, batch_size)

        stacked_fake_features = []
        self.print_fn(f"Stacking Inception features for {self.n_samples} generated samples.")
        for idx in tqdm(range(len(batches))):
            fake_samples = self.sampler.sample(classes[idx*batch_size:(idx+1)*batch_size])
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def divisible_by(numer, denom):
    return (numer % denom) == 0

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    