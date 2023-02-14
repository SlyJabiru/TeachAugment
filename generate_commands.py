def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

def remove_postfix(text, postfix):
    return text[:-len(postfix)] if text.endswith(postfix) else text


g_scale = None # default: 0.5
g_scale_command = f'--g_scale {g_scale}' if g_scale else ''
c_scale = None # default: 0.8
c_scale_command = f'--c_scale {c_scale}' if c_scale else ''

g_offset = None # default -0.5
g_offset_command = f'--g_offset {g_offset}' if g_offset else ''

g_scale_unlimited = True
g_scale_unlimited_command = f'--g_scale_unlimited' if g_scale_unlimited else ''

c_scale_unlimited = True
c_scale_unlimited_command = f'--c_scale_unlimited' if c_scale_unlimited else ''

c_shift_unlimited = True
c_shift_unlimited_command = f'--c_shift_unlimited' if c_shift_unlimited else ''



gpu_start = 4
gpu_end = 7

seed_start = 0
seed_end = 3

seeds = list(range(seed_start, seed_end+1))
gpu_ids = list(range(gpu_start, gpu_end+1))


for (seed, gpu) in list(zip(seeds, gpu_ids)):
    new_seed = str(seed).zfill(3)
    g_scale_tag = f"_gscale{str(g_scale).replace('.', '')}_" if g_scale else ""
    g_offset_tag = f"_goffset{str(g_offset).replace('.', '')}_" if g_offset else ""
    g_scale_unlimited_tag = f"_gscale_unlimited_" if g_scale_unlimited else ""

    c_scale_tag = f"_cscale{str(c_scale).replace('.', '')}_" if c_scale else ""
    c_scale_unlimited_tag = f"_cscale_unlimited_" if c_scale_unlimited else ""
    c_shift_unlimited_tag = f"_cshift_unlimited_" if c_shift_unlimited else ""

    new_tag = f'CIFAR100_wrn28x10_gradual_warm_scheduler'
    new_tag = f'{new_tag}{g_scale_tag}{g_offset_tag}{g_scale_unlimited_tag}'
    new_tag = f'{new_tag}{c_scale_tag}{c_scale_unlimited_tag}{c_shift_unlimited_tag}{new_seed}'

    commands = f"""
tmux new -s {new_tag}
conda activate trivial_augment
CUDA_VISIBLE_DEVICES={gpu} python main.py \
    --seed {seed} \
    --scheduler gradual_warm \
    --root /hdd/hdd4/lsj/torchvision_dataset/CIFAR100 \
    --log_dir /hdd/hdd4/lsj/teach_augment \
    --dataset CIFAR100 \
    {g_scale_command} \
    {g_offset_command} \
    {c_scale_command} \
    {g_scale_unlimited_command} \
    {c_scale_unlimited_command} \
    {c_shift_unlimited_command} \
    --yaml ./config/CIFAR100/wrn-28-10.yaml --vis 2>&1 | tee stdouts/{new_tag}.log
"""

    # print(new_tag)
    print(commands)
    print()
