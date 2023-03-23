#!/bin/bash

# # MNIST fraction6000
# ## EMA
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10011 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str ema1-total10 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10022 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str ema2-total10 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# ## Fixed
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10033 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed1-total10 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --data_fraction 6000 \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10044 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed2-total10 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --data_fraction 6000 \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# echo "waiting all background jobs..."
# for job in `jobs -p`
# do
#     wait $job
# done
# sleep 60


# # MNIST fraction6000 n_inner1
# ## EMA
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10055 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str ema1-n_inner1-fraction6000 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --n_inner 1 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10066 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str ema2-n_inner1-fraction6000 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --n_inner 1 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# ## Fixed
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10077 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed1-n_inner1-fraction6000 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --n_inner 1 \
#     --data_fraction 6000 \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10088 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed2-n_inner1-fraction6000 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --n_inner 1 \
#     --data_fraction 6000 \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# echo "waiting all background jobs..."
# for job in `jobs -p`
# do
#     wait $job
# done
# sleep 60


# MNIST fraction6000 n_inner50 or 100 ema_rate 0.5
## EMA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10055 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-rate05-n_inner50-fraction6000 \
    --seed 1 \
    --sampling_freq 0 \
    --ema_rate 0.5 \
    --n_epochs 400 \
    --n_inner 50 \
    --data_fraction 6000 \
    --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset MNIST \
    --yaml ./config/MNIST/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10066 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-rate05-n_inner50-fraction6000 \
    --seed 2 \
    --sampling_freq 0 \
    --ema_rate 0.5 \
    --n_epochs 400 \
    --n_inner 50 \
    --data_fraction 6000 \
    --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset MNIST \
    --yaml ./config/MNIST/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10077 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-rate05-n_inner100-fraction6000 \
    --seed 1 \
    --sampling_freq 0 \
    --ema_rate 0.5 \
    --n_epochs 400 \
    --n_inner 100 \
    --data_fraction 6000 \
    --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset MNIST \
    --yaml ./config/MNIST/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10088 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-rate05-n_inner100-fraction6000 \
    --seed 2 \
    --sampling_freq 0 \
    --ema_rate 0.5 \
    --n_epochs 400 \
    --n_inner 100 \
    --data_fraction 6000 \
    --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset MNIST \
    --yaml ./config/MNIST/wrn-28-10.yaml &
sleep 3

echo "waiting all background jobs..."
for job in `jobs -p`
do
    wait $job
done
sleep 60


# # MNIST fraction6000 n_inner50 or 100 ema_rate 0.5
# ## Fixed
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10011 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed1-rate05-n_inner50-fraction6000 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --ema_rate 0.5 \
#     --n_epochs 400 \
#     --n_inner 50 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10022 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed2-rate05-n_inner50-fraction6000 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --ema_rate 0.5 \
#     --n_epochs 400 \
#     --n_inner 50 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10033 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed1-rate05-n_inner100-fraction6000 \
#     --seed 1 \
#     --sampling_freq 0 \
#     --ema_rate 0.5 \
#     --n_epochs 400 \
#     --n_inner 100 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4  \
#   --master_port=10044 main.py \
#     --dist \
#     --wandb_store_image \
#     --wandb_str fixed2-rate05-n_inner100-fraction6000 \
#     --seed 2 \
#     --sampling_freq 0 \
#     --ema_rate 0.5 \
#     --n_epochs 400 \
#     --n_inner 100 \
#     --data_fraction 6000 \
#     --root /hdd/hdd4/lsj/torchvision_dataset/MNIST \
#     --log_dir /hdd/hdd4/lsj/teach_augment \
#     --fixed_teacher /hdd/hdd4/lsj/teach_augment/MNIST-overfit-to-aug/MNIST-one-per-class-wrn-28-10-epoch200.pt \
#     --dataset MNIST \
#     --yaml ./config/MNIST/wrn-28-10.yaml &
# sleep 3

# echo "waiting all background jobs..."
# for job in `jobs -p`
# do
#     wait $job
# done
# sleep 60
