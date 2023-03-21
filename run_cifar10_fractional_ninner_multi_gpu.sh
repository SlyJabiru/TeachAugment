#!/bin/bash

# CIFAR 10
## EMA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10011 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10022 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-fraction50 \
    --seed 2 \
    --sampling_freq 0 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

## Fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10033 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed1-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10044 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed2-fraction50 \
    --seed 2 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

echo "waiting all background jobs..."
for job in `jobs -p`
do
    wait $job
done
sleep 60


# CIFAR 10
## EMA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10055 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-n_inner1-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 1 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10066 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-n_inner1-fraction50 \
    --seed 2 \
    --sampling_freq 0 \
    --n_inner 1 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

## Fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10077 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed1-n_inner1-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 1 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10088 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed2-n_inner1-fraction50 \
    --seed 2 \
    --n_inner 1 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

echo "waiting all background jobs..."
for job in `jobs -p`
do
    wait $job
done
sleep 60


# CIFAR 10
## EMA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10011 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-n_inner100-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 100 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10022 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-n_inner100-fraction50 \
    --seed 2 \
    --sampling_freq 0 \
    --n_inner 100 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

## Fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10033 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed1-n_inner100-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 100 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10044 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed2-n_inner100-fraction50 \
    --seed 2 \
    --n_inner 100 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

echo "waiting all background jobs..."
for job in `jobs -p`
do
    wait $job
done
sleep 60


# CIFAR 10 data fraction and n_inner
## EMA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10055 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema1-n_inner200-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 200 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10066 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str ema2-n_inner200-fraction50 \
    --seed 2 \
    --sampling_freq 0 \
    --n_inner 200 \
    --data_fraction 50 \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

## Fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10077 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed1-n_inner200-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 200 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4  \
  --master_port=10088 main.py \
    --dist \
    --wandb_store_image \
    --wandb_str fixed2-n_inner200-fraction50 \
    --seed 1 \
    --sampling_freq 0 \
    --n_inner 200 \
    --data_fraction 50 \
    --fixed_teacher /hdd/hdd4/lsj/teach_augment/20230309-17:38:01/withoutDA-wrn-28-10-epoch480.pt \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR10 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR10 \
    --yaml ./config/CIFAR10/wrn-28-10.yaml &
sleep 3

echo "waiting all background jobs..."
for job in `jobs -p`
do
    wait $job
done
sleep 60

