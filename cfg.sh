export CUDA_VISIBLE_DEVICES=1
python main.py \
    --root /media/mountHDD3/data_storage/standford_dog \
    --size 128 --batch_size 32 --shuffle --num_workers 12 --pin_memory --horflip \
    \
    --timesteps 1000 --sampling_timesteps 250 --objective pred_noise --beta_schedule cosine \
    --ddim_sampling_eta 1.0 --offset_noise_strength 0.0 --min_snr_loss_weight --min_snr_gamma 5.0 \
    \
    --dim 64 --cond_drop_prob 0.5 --dim_mults 1 2 4 8 --channels 3  --attn_dim_head 32 --attn_heads 4 \
    \
    --mode cfg --gradient_accumulate_every 2 --train_lr 1e-4 --train_num_steps 100000 --ema_update_every 10 \
    --ema_decay 0.995 --adam_betas 0.9 0.99 --save_and_sample_every 1000 --amp --mixed_precision_type fp16 \
    --split_batches --calculate_fid --inception_block_idx 2048 --max_grad_norm 1.0 \
    --num_fid_samples 120 --save_best_and_latest_only \
    \
    --wandb \
    --wandb_project dogfusion