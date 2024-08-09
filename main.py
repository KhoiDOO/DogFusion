from data import get_ds
from datetime import datetime

import argparse
import os


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="PyTorch DataLoader with Augmentation, Diffusion, U-Net, Training, and Weights & Biases Parameters")

    # Add basic arguments
    parser.add_argument('--root', type=str, help='Root directory where data is located')
    parser.add_argument('--size', type=int, required=True, help='Size of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader')
    parser.add_argument('--horflip', action='store_true', help='Apply horizontal flip augmentation')

    # Add diffusion-related arguments
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps for diffusion process')
    parser.add_argument('--sampling_timesteps', type=int, default=None, help='Number of timesteps for sampling process')
    parser.add_argument('--objective', type=str, default='pred_noise', choices=['pred_noise', 'pred_x0', 'pred_v'], help='Objective for diffusion (pred_noise, pred_x0, pred_v)')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Beta schedule type (linear, cosine)')
    parser.add_argument('--ddim_sampling_eta', type=float, default=1.0, help='Eta for DDIM sampling')
    parser.add_argument('--offset_noise_strength', type=float, default=0.0, help='Strength of the offset noise')
    parser.add_argument('--min_snr_loss_weight', action='store_true', help='Enable min SNR loss weight')
    parser.add_argument('--min_snr_gamma', type=float, default=5.0, help='Gamma value for min SNR loss weight')

    # Add U-Net architecture arguments
    parser.add_argument('--dim', type=int, required=True, help='Base dimension of the U-Net')
    parser.add_argument('--cond_drop_prob', type=float, default=0.5, help='Conditional dropout probability')
    parser.add_argument('--init_dim', type=int, default=None, help='Initial dimension after the first convolution')
    parser.add_argument('--out_dim', type=int, default=None, help='Output dimension')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Multipliers for each level of the U-Net')
    parser.add_argument('--channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--learned_variance', action='store_true', help='Enable learning of variance')
    parser.add_argument('--learned_sinusoidal_cond', action='store_true', help='Enable learned sinusoidal conditioning')
    parser.add_argument('--random_fourier_features', action='store_true', help='Enable random Fourier features')
    parser.add_argument('--learned_sinusoidal_dim', type=int, default=16, help='Dimension for learned sinusoidal embedding')
    parser.add_argument('--attn_dim_head', type=int, default=32, help='Dimension per attention head')
    parser.add_argument('--attn_heads', type=int, default=4, help='Number of attention heads')

    # Add training-related arguments
    parser.add_argument('--mode', type=str, default='cfg', choices=['cfg'], help='training mode')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, help='Accumulate gradients every N steps')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--train_num_steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--ema_update_every', type=int, default=10, help='Update the EMA every N steps')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='Decay factor for EMA')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=(0.9, 0.99), help='Betas for Adam optimizer')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Save and sample every N steps')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--mixed_precision_type', type=str, default='fp16', help='Type of mixed precision (e.g., fp16)')
    parser.add_argument('--split_batches', action='store_true', help='Split batches for large models')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID score')
    parser.add_argument('--inception_block_idx', type=int, default=2048, help='Block index for Inception model')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for gradient clipping')
    parser.add_argument('--num_fid_samples', type=int, default=50000, help='Number of samples for FID calculation')
    parser.add_argument('--save_best_and_latest_only', action='store_true', help='Save only the best and latest models')

    # Add Weights & Biases (wandb) arguments
    parser.add_argument('--wandb', action='store_true', help='toggle wandb')
    parser.add_argument('--wandb_project', type=str, default='dogfusion', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (team) name')

    # Parse the arguments
    args = parser.parse_args()

    print('INFO: Dataset Preparation')
    dl = get_ds(args)
    print(f'INFO: #Batches: {len(dl)}')

    from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

    model = Unet(
        dim = args.dim,
        num_classes = 120,
        cond_drop_prob = args.cond_drop_prob,
        init_dim = args.init_dim,
        out_dim = args.out_dim,
        dim_mults= args.dim_mults,
        channels = args.channels,
        learned_variance = args.learned_variance,
        learned_sinusoidal_cond = args.learned_sinusoidal_cond,
        random_fourier_features = args.random_fourier_features,
        learned_sinusoidal_dim = args.learned_sinusoidal_dim,
        attn_dim_head = args.attn_dim_head,
        attn_heads = args.attn_heads
    )
    print(f'INFO: Unet Preparation - #Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    diffusion = GaussianDiffusion(
        model = model,
        image_size=args.size,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,
        objective = args.objective,
        beta_schedule = args.beta_schedule,
        ddim_sampling_eta = args.ddim_sampling_eta,
        offset_noise_strength = args.offset_noise_strength,
        min_snr_loss_weight = args.min_snr_loss_weight,
        min_snr_gamma = args.min_snr_gamma
    )
    print(f'INFO: Diffusion Preparation')

    if args.mode == 'cfg':
        from trainer.cfg import Trainer

        time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        sv_dir = os.path.join(os.getcwd(), 'runs', f'{time}')
        os.makedirs(sv_dir, exist_ok=True)

        trainer = Trainer(
            diffusion_model = diffusion,
            dl = dl,
            train_batch_size = args.batch_size,
            gradient_accumulate_every = args.gradient_accumulate_every,
            train_lr = args.train_lr,
            train_num_steps = args.train_num_steps,
            ema_update_every = args.ema_update_every,
            ema_decay = args.ema_decay,
            adam_betas = args.adam_betas,
            save_and_sample_every = args.save_and_sample_every,
            results_folder = sv_dir,
            amp = args.amp,
            mixed_precision_type = args.mixed_precision_type,
            split_batches = args.split_batches,
            calculate_fid = args.calculate_fid,
            inception_block_idx = args.inception_block_idx,
            max_grad_norm = args.max_grad_norm,
            num_fid_samples = args.num_fid_samples,
            save_best_and_latest_only = args.save_best_and_latest_only,
            wandb_dct={'project': args.wandb_project, 'config': args, 
                       'entity': args.wandb_entity, 'name': os.path.basename(sv_dir)} if args.wandb else None
        )

        trainer.train()