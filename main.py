import os
import logging
import warnings
import datetime
import os

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', UserWarning)

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from lib import augmentation, build_dataset, teachaugment
from lib.utils import utils, lr_scheduler
from lib.models import build_model
from lib.losses import non_saturating_loss

import wandb
import numpy as np

def main(args):
    main_process = args.local_rank == 0
    if main_process:
        logger.info(args)
    # Setup GPU
    if torch.cuda.is_available():
        device = 'cuda'
        if args.disable_cudnn:
            # torch.nn.functional.grid_sample, which is used for geometric augmentation, is non-deterministic
            # so, reproducibility is not ensured even though following option is True
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
    else:
        raise NotImplementedError('CUDA is unavailable.')
    # Dataset
    base_aug, train_trans, val_trans, normalizer = augmentation.get_transforms(args.dataset)
    train_data, eval_data, n_classes = build_dataset(args.dataset, args.root, train_trans, val_trans)
    print(f'len(subset_train) before fraction: {len(train_data)}')

    subset_indices = list(range(0, len(train_data), args.data_fraction))
    train_data = torch.utils.data.Subset(train_data, subset_indices)
    print(f'len(subset_train) after fraction: {len(train_data)}')

    sampler = torch.utils.data.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.local_rank) if args.dist else None
    train_loader = DataLoader(train_data, args.batch_size, not args.dist, sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    eval_loader = DataLoader(eval_data, 1)
    n_channel = 1 if args.dataset == 'MNIST' else 3
    # Model
    model = build_model(args.model, n_classes, n_channel).to(device)
    model.train()
    # EMA Teacher
    avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                args.ema_rate * averaged_model_parameter + (1 - args.ema_rate) * model_parameter
    
    if args.fixed_teacher:
        print(f'Make fixed teacher model')
        ema_model = build_model(args.model, n_classes, n_channel)
        ema_model.load_state_dict(torch.load(args.fixed_teacher))
        ema_model = ema_model.to(device)

        fixed_model_for_check = build_model(args.model, n_classes, n_channel)
        fixed_model_for_check.load_state_dict(torch.load(args.fixed_teacher))
        fixed_model_for_check = fixed_model_for_check.to(device)
    else:
        print(f'Make EMA Teacher')
        ema_model = optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)
    for ema_p in ema_model.parameters():
        ema_p.requires_grad_(False)
    ema_model.train()

    # Trainable Augmentation
    rbuffer = augmentation.replay_buffer.ReplayBuffer(args.rb_decay)
    trainable_aug = augmentation.build_augmentation(n_classes, n_channel,
                                                    args.g_offset, args.g_scale, args.g_scale_unlimited,
                                                    args.c_scale, args.c_scale_unlimited, args.c_shift_unlimited,
                                                    args.c_reg_coef, normalizer, rbuffer,
                                                    args.batch_size // args.group_size,
                                                    not args.wo_context).to(device)
    # Baseline augmentation
    base_aug = torch.nn.Sequential(*base_aug).to(device)
    if main_process:
        logger.info('augmentation')
        logger.info(trainable_aug)
        logger.info(base_aug)
    # Optimizer
    optim_cls = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)
    optim_aug = optim.AdamW(trainable_aug.parameters(), lr=args.aug_lr, weight_decay=args.aug_weight_decay)

    if args.scheduler == 'original':
        if args.dataset == 'ImageNet':
            scheduler = lr_scheduler.MultiStepLRWithLinearWarmup(optim_cls, 5, [90, 180, 240], 0.1)
        else:
            scheduler = lr_scheduler.CosineAnnealingWithLinearWarmup(optim_cls, 5, args.n_epochs)
    elif args.scheduler == 'gradual_warm':
        from warmup_scheduler import GradualWarmupScheduler
        if args.dataset == 'ImageNet':
            base_scheduler = optim.lr_scheduler.MultiStepLR(optim_cls, [90, 180, 240], 0.1)
            scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)
        else:
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_cls, args.n_epochs)
            scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)
    else:
        raise Exception(f'args.scheduler should be "original" or "gradual_warm", current value: {args.scheduler}')

    # Following Fast AutoAugment (https://github.com/kakaobrain/fast-autoaugment),
    # pytorch-gradual-warmup-lr (https://github.com/ildoonet/pytorch-gradual-warmup-lr) was used for the paper experiments.
    # The implementation of our "*WithLinearWarmup" is slightly different from GradualWarmupScheduler.
    # Thus, to reproduce experimental results strictly, please use following scheduler, instead of above scheduler.

    # Don't forget to install pytorch-gradual-warmup-lr
    #     pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

    # from warmup_scheduler import GradualWarmupScheduler
    # if args.dataset == 'ImageNet':
    #     base_scheduler = optim.lr_scheduler.MultiStepLR(optim_cls, [90, 180, 240], 0.1)
    #     scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)
    # else:
    #     base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_cls, args.n_epochs)
    #     scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)

    # Objective function
    adv_criterion = non_saturating_loss.NonSaturatingLoss(args.epsilon)
    objective = teachaugment.TeachAugment(model, ema_model, trainable_aug,
                                          adv_criterion, args.teacher_loss_coeff, args.weight_decay,
                                          base_aug, normalizer, not args.dist and args.save_memory).to(device)
    # DDP
    if args.dist:
        objective = torch.nn.parallel.DistributedDataParallel(objective, device_ids=[args.local_rank],
                                                              find_unused_parameters=True,
                                                              output_device=args.local_rank)
    # Resume
    st_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoint.pth'))
        st_epoch += checkpoint['epoch']
        if main_process:
            logger.info(f'resume from epoch {st_epoch}')
        buffer_length = checkpoint['epoch'] // args.sampling_freq
        rbuffer.initialize(buffer_length, trainable_aug.get_augmentation_model()) # define placeholder for load_state_dict
        objective.load_state_dict(checkpoint['objective']) # including model, ema teacher, trainable_aug, and replay buffer
        optim_cls.load_state_dict(checkpoint['optim_cls'])
        optim_aug.load_state_dict(checkpoint['optim_aug'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    # Training loop
    if main_process:
        logger.info('training')
    meter = utils.AvgMeter()
    num_update_aug = 0
    global_iter_idx = 0
    for epoch in range(st_epoch, args.n_epochs + 1):
        model.train()
        ema_model.train()
        trainable_aug.train()
        if args.dist:
            train_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader):
            torch.cuda.synchronize()
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            if args.wo_context:
                context = None
            else:
                context = targets
            # update teacher model
            if not args.fixed_teacher:
                # print(f'Update teacher')
                ema_model.update_parameters(model)
            
            global_iter_idx += 1
            # Update augmentation
            if global_iter_idx % args.n_inner == 0:
                if main_process:
                    num_update_aug += 1
                    # print(f'num_update_aug: {num_update_aug}')

                optim_aug.zero_grad()
                if args.dist and args.save_memory: # computating gradient independently for saving memory
                    loss_adv, c_reg, acc_tar = objective(inputs, targets, context, 'loss_adv')
                    (loss_adv + 0.5 * c_reg).backward()
                    loss_tea, c_reg, acc_tea = objective(inputs, targets, context, 'loss_tea')
                    (loss_tea + 0.5 * c_reg).backward()
                    res = {'loss adv.': loss_adv.item(),
                           'loss teacher': loss_tea.item(),
                           'color reg.': c_reg.item(),
                           'acc.': acc_tar.item(),
                           'acc. teacher': acc_tea.item()}
                else:
                    loss_aug, res = objective(inputs, targets, context, 'aug')
                    loss_aug.backward()
                optim_aug.step()
                meter.add(res)
            # Update target model
            optim_cls.zero_grad()
            loss_cls, res, aug_img = objective(inputs, targets, context, 'cls')
            loss_cls.backward()
            if args.dataset != 'ImageNet':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim_cls.step()
            # Adjust learning rate
            scheduler.step(epoch - 1. + (i + 1.) / len(train_loader))
            # Print losses and accuracy
            meter.add(res)
            # if main_process and (i + 1) % args.print_freq == 0:
            #     logger.info(meter.state(f'epoch {epoch} [{i+1}/{len(train_loader)}]',
            #                             f'lr {optim_cls.param_groups[0]["lr"]:.4e}'))

        # Log performances each epoch
        if main_process:
            avg_dict = meter.make_avg_dict()
            # print(f'epoch:{epoch}, global_iter_idx: {global_iter_idx}')
            print(avg_dict)
            if 'loss adv.' in avg_dict:
                # update of augmentation network is done
                loss_aug = avg_dict['loss adv.'] + avg_dict['loss teacher'] + avg_dict['color reg.'] # 이 크기가 작다.
                wandb.log({
                    'step': epoch,
                    'global_iter_idx': global_iter_idx,
                    'num_update_aug': num_update_aug,
                    'train/loss_adv': avg_dict['loss adv.'], # 작다. 키우고 싶지만, 안 커진다.
                    'train/loss_tea': avg_dict['loss teacher'], # 크다. 작게 하고 싶지만, 크다.
                    'train/loss_color': avg_dict['color reg.'],
                    'train/loss_aug': loss_aug, # 작다.
                    'train/acc_adv': avg_dict['acc.'],
                    'train/acc_tea': avg_dict['acc. teacher'],
                    'train/loss_cls': avg_dict['loss cls.'],
                })
            else:
                # only update of target classifier is done
                wandb.log({
                    'step': epoch,
                    'global_iter_idx': global_iter_idx,
                    'num_update_aug': num_update_aug,
                    'train/loss_cls': avg_dict['loss cls.'],
                })

        # Store augmentation in buffer
        if args.sampling_freq > 0 and epoch % args.sampling_freq == 0:
            rbuffer.store(trainable_aug.get_augmentation_model())
            if main_process:
                logger.info(f'store augmentation (buffer length: {len(rbuffer)})')
        # Save checkpoint
        if main_process:
            logger.info(meter.mean_state(f'epoch [{epoch}/{args.n_epochs}]',
                                         f'lr {optim_cls.param_groups[0]["lr"]:.4e}'))
            # checkpoint = {'model': model.state_dict(),
            #               'objective': objective.state_dict(), # including ema model and replay buffer
            #               'optim_cls': optim_cls.state_dict(),
            #               'optim_aug': optim_aug.state_dict(),
            #               'scheduler': scheduler.state_dict(),
            #               'epoch': epoch}
            # torch.save(checkpoint, os.path.join(args.log_dir, 'checkpoint.pth'))

        # Check parameter and gradients of the target model and augmentation model
        if main_process and epoch % args.n_check_norm == 0:
            if args.fixed_teacher:
                for p1, p2 in zip(ema_model.parameters(), fixed_model_for_check.parameters()):
                    if p1.data.ne(p2.data).sum() > 0:
                        print(f'The teacher model is not fixed!')
                        print(f'p1.data.ne(p2.data).sum(): {p1.data.ne(p2.data).sum()}')

            sum_norm_model = 0.0
            sum_norm_grad_model = 0.0
            for name, param in model.named_parameters():
                sum_norm_model += torch.linalg.norm(param).item() if param is not None else 0.0
                sum_norm_grad_model += torch.linalg.norm(param.grad).item() if param.grad is not None else 0.0

            sum_norm_c_aug = 0.0
            sum_norm_grad_c_aug = 0.0
            sum_norm_g_aug = 0.0
            sum_norm_grad_g_aug = 0.0
            for name, param in trainable_aug.named_parameters():
                # print(f'name: {name}')
                if 'c_aug' in name:
                    sum_norm_c_aug += torch.linalg.norm(param).item() if param is not None else 0.0
                    sum_norm_grad_c_aug += torch.linalg.norm(param.grad).item() if param.grad is not None else 0.0
                elif 'g_aug' in name:
                    sum_norm_g_aug += torch.linalg.norm(param).item() if param is not None else 0.0
                    sum_norm_grad_g_aug += torch.linalg.norm(param.grad).item() if param.grad is not None else 0.0
                else:
                    pass

            wandb.log({
                'step': epoch,
                'train/norm_model': sum_norm_model,
                'train/norm_grad_model': sum_norm_grad_model,
                'train/norm_c_aug': sum_norm_c_aug,
                'train/norm_grad_c_aug': sum_norm_grad_c_aug,
                'train/norm_g_aug': sum_norm_g_aug,
                'train/norm_grad_g_aug': sum_norm_grad_g_aug,
            })

        # Check augmentation parameters time to time
        # TODO: Here? because of trainable_aug.get_params(inputs, context)??
        if main_process and epoch % args.n_check_aug_param == 0:
            with torch.no_grad():
                trainable_aug.eval()
                c_param, g_param, A = trainable_aug.get_params(inputs, context)
                scale = c_param[0] # scale should go to 1 to make identity function
                shift = c_param[1] # shift should go to 0 to make identity function
                # A should go to [[1, 0, 0], [0, 1, 0]]^T to make identity function

                A_mean = torch.mean(A, dim=0) # torch.Size([2, 3])
                A_txt_path = os.path.join(args.log_dir, f'{epoch}_A.txt')
                np.savetxt(A_txt_path, A_mean.detach().cpu().numpy(), fmt='%.3f')

                artifact = wandb.Artifact(f'aug-affine-matrix', type='dataset')
                artifact.add_file(A_txt_path, f'{epoch}_A.txt')
                wandb.log_artifact(artifact)

                identity = torch.Tensor([[1,0,0],[0,1,0]]).to(device)
            wandb.log({
                'step': epoch,
                'd1(A, I)': (A_mean - identity).abs().sum().item(),
                'scale': torch.mean(scale).item(),
                'shift': torch.mean(shift).item(),
            })

        # Save augmented images
        # TODO: because of this??
        # print(f'args.wandb_store_image: {args.wandb_store_image}')
        if main_process and args.wandb_store_image and epoch % args.n_save_image == 0:
            print(f'save image on wandb...')
            columns=['image', 'augmented', 'gt', 'target_pred', 'teacher_pred']
            image_table = wandb.Table(columns=columns)

            with torch.no_grad():
                model.eval()
                ema_model.eval()
                trainable_aug.eval()

                outputs_model = model(aug_img)
                _, pred_model = torch.max(outputs_model.data, 1)
                outputs_teacher = ema_model(aug_img)
                _, pred_teacher = torch.max(outputs_teacher.data, 1)

            utils.fill_wandb_table(inputs, aug_img, targets,
                                   pred_model, pred_teacher,
                                   image_table)
            wandb.log({f'{epoch}_image_table' : image_table})

            # save_image(aug_img, os.path.join(args.log_dir, f'{epoch}epoch_aug_img.png'))
            # save_image(inputs, os.path.join(args.log_dir, f'{epoch}epoch_img.png'))


        if main_process and epoch % args.n_eval == 0:
            eval_meter = utils.AvgMeter()
            model.eval()
            ema_model.eval()
            trainable_aug.eval()
            n_samples = len(eval_data)
            with torch.no_grad():
                for data in eval_loader:
                    input, target = data
                    input = input.to(device)
                    target = target.to(device)

                    model_pred = model(input)
                    model_eval_loss = F.cross_entropy(model_pred, target)
                    model_eval_accs = utils.accuracy(model_pred, target, (1, 5))

                    ema_model_pred = ema_model(input)
                    ema_model_eval_loss = F.cross_entropy(ema_model_pred, target)
                    ema_model_eval_accs = utils.accuracy(ema_model_pred, target, (1, 5))

                    eval_meter.add({
                        'model_eval_loss': model_eval_loss.item(),
                        'model_eval_acc1': model_eval_accs[0],
                        'model_eval_acc5': model_eval_accs[1],
                        'ema_model_eval_loss': ema_model_eval_loss.item(),
                        'ema_model_eval_acc1': ema_model_eval_accs[0],
                        'ema_model_eval_acc5': ema_model_eval_accs[1],
                    })

            eval_avg_dict = eval_meter.make_avg_dict()
            wandb.log({
                'step': epoch,
                'eval/loss_cls': eval_avg_dict['model_eval_loss'],
                'eval/acc1_cls': eval_avg_dict['model_eval_acc1'],
                'eval/acc5_cls': eval_avg_dict['model_eval_acc5'],
                'eval/loss_tea': eval_avg_dict['ema_model_eval_loss'],
                'eval/acc1_tea': eval_avg_dict['ema_model_eval_acc1'],
                'eval/acc5_tea': eval_avg_dict['ema_model_eval_acc5'],
            })
            logger.info(eval_meter.mean_state(f'[Eval] epoch [{epoch}/{args.n_epochs}]',''))

    # Evaluation
    if main_process:
        logger.info('Final Evaluation')
        model_acc1, model_acc5 = 0, 0
        ema_model_acc1, ema_model_acc5 = 0, 0
        model.eval()
        ema_model.eval()
        n_samples = len(eval_data)
        eval_meter = utils.AvgMeter()

        with torch.no_grad():
            for data in eval_loader:
                input, target = data
                input = input.to(device)
                target = target.to(device)

                model_pred = model(input)
                model_eval_loss = F.cross_entropy(model_pred, target)
                model_eval_accs = utils.accuracy(model_pred, target, (1, 5))

                ema_model_pred = ema_model(input)
                ema_model_eval_loss = F.cross_entropy(ema_model_pred, target)
                ema_model_eval_accs = utils.accuracy(ema_model_pred, target, (1, 5))

                eval_meter.add({
                    'model_eval_loss': model_eval_loss.item(),
                    'model_eval_acc1': model_eval_accs[0],
                    'model_eval_acc5': model_eval_accs[1],
                    'ema_model_eval_loss': ema_model_eval_loss.item(),
                    'ema_model_eval_acc1': ema_model_eval_accs[0],
                    'ema_model_eval_acc5': ema_model_eval_accs[1],
                })

                model_acc1 += model_eval_accs[0]
                model_acc5 += model_eval_accs[1]
                ema_model_acc1 += ema_model_eval_accs[0]
                ema_model_acc5 += ema_model_eval_accs[1]
        eval_avg_dict = eval_meter.make_avg_dict()
        wandb.log({
            'step': epoch+1,
            'eval/loss_cls': eval_avg_dict['model_eval_loss'],
            'eval/acc1_cls': eval_avg_dict['model_eval_acc1'],
            'eval/acc5_cls': eval_avg_dict['model_eval_acc5'],
            'eval/loss_tea': eval_avg_dict['ema_model_eval_loss'],
            'eval/acc1_tea': eval_avg_dict['ema_model_eval_acc1'],
            'eval/acc5_tea': eval_avg_dict['ema_model_eval_acc5'],
        })
        print('eval_avg_dict')
        print(eval_avg_dict)
        print()

        print(f"eval_avg_dict['model_eval_acc1']: {eval_avg_dict['model_eval_acc1']}")
        print(f"eval_avg_dict['ema_model_eval_acc1']: {eval_avg_dict['ema_model_eval_acc1']}")
        print(f"model_acc1/n_samples: {model_acc1/n_samples}")
        print(f"ema_model_acc1/n_samples: {ema_model_acc1/n_samples}")
        logger.info(f'Final Evaluation: {args.dataset} error rate (%) | Target model: Top1 {100 - model_acc1/n_samples}, Top5 {100 - model_acc5/n_samples}')
        logger.info(f'Final Evaluation: {args.dataset} error rate (%) | Teacher model: Top1 {100 - ema_model_acc1/n_samples}, Top5 {100 - ema_model_acc5/n_samples}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--root', default='./data', type=str,
                        help='/path/to/dataset')
    parser.add_argument('--data_fraction', default=1, type=int)
    # Model
    parser.add_argument('--model', default='wrn-28-10', type=str)
    parser.add_argument('--fixed_teacher', default='', type=str)
    # Optimization
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=5e-4, type=float)
    parser.add_argument('--teacher_loss_coeff', default=1.0, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--aug_lr', default=1e-3, type=float,
                        help='learning rate for augmentation model')
    parser.add_argument('--aug_weight_decay', '-awd', default=1e-2, type=float,
                        help='weight decay for augmentation model')
    parser.add_argument('--scheduler', default='gradual_warm', choices=['original', 'gradual_warm'], type=str)
    # Augmentation
    parser.add_argument('--g_offset', default=-0.5, type=float,
                        help='the search range offset of the magnitude of geometric augmantation')
    parser.add_argument('--g_scale', default=0.5, type=float,
                        help='the search range of the magnitude of geometric augmantation')
    parser.add_argument('--g_scale_unlimited', default=False, action='store_true',
                        help='if true, the search range of the magnitude of geometric augmantation is (-inf, inf)')
    parser.add_argument('--c_scale', default=0.8, type=float,
                        help='the search range of the magnitude of color augmantation')
    parser.add_argument('--c_scale_unlimited', default=False, action='store_true',
                        help='if true, the search range of the magnitude of color scale augmentation is (-inf, inf)')
    parser.add_argument('--c_shift_unlimited', default=False, action='store_true',
                        help='if true, the search range of the magnitude of color shift augmentation is (-inf, inf)')
    parser.add_argument('--group_size', default=8, type=int)
    parser.add_argument('--wo_context', action='store_true',
                        help='without context vector as input')
    # TeachAugment
    parser.add_argument('--n_inner', default=5, type=int,
                        help='the number of iterations for inner loop (i.e., updating classifier)')
    parser.add_argument('--ema_rate', default=0.999, type=float,
                        help='decay rate for the ema teacher')
    # Improvement techniques
    parser.add_argument('--c_reg_coef', default=10, type=float,
                        help='coefficient of the color regularization')
    parser.add_argument('--rb_decay', default=0.9, type=float,
                        help='decay rate for replay buffer')
    parser.add_argument('--sampling_freq', default=10, type=int,
                        help='sampling augmentation frequency')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='epsilon for the label smoothing')
    # Distributed data parallel
    parser.add_argument('--dist', action='store_true',
                        help='use distributed data parallel')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', '-ws', default=1, type=int)
    parser.add_argument('--port', default=None, type=str)
    # Misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--disable_cudnn', action='store_true',
                        help='disable cudnn for reproducibility')
    parser.add_argument('--resume', action='store_true',
                        help='resume training')
    parser.add_argument('--num_workers', '-j', default=16, type=int,
                        help='the number of data loading workers')
    # parser.add_argument('--vis', action='store_true',
    #                     help='visualize augmented images')
    parser.add_argument('--save_memory', action='store_true',
                        help='independently calculate adversarial loss \
                            and teacher loss for saving memory')
    parser.add_argument('--yaml', default=None, type=str,
                        help='given path to .json, parse from .yaml')
    parser.add_argument('--json', default=None, type=str,
                        help='given path to .json, parse from .json')

    # Logging
    parser.add_argument('--wandb_project', default='TeachAugment', type=str, help='a string to use as a wandb project name.')
    parser.add_argument('--wandb_str', default='', type=str, help='a string to use in wandb run name.')
    parser.add_argument('--wandb_store_image', action='store_true', help='a flag whether images are saved in wandb or not')
    parser.add_argument('--n_check_aug_param', default=5, type=int, help='an integer indicating how frequently augmentation parameters are checked.')
    parser.add_argument('--n_eval', default=5, type=int, help='an integer indicating how frequently evaluations are done during training.')
    parser.add_argument('--n_save_image', default=5, type=int, help='an integer indicating how frequently images are saved during training.')
    parser.add_argument('--n_check_norm', default=5, type=int, help='an integer indicating how frequently model norms are checked.')

    args = parser.parse_args()

    # override args
    if args.yaml is not None:
        yaml_cfg = utils.load_yaml(args.yaml)
        args = utils.override_config(args, yaml_cfg)
    if args.json is not None:
        json_cfg = utils.load_json(args.json)
        args = utils.override_config(args, json_cfg)

    utils.set_seed(args.seed)

    # now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    # args.log_dir = os.path.join(args.log_dir, now)
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir) 

    if args.local_rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        args.log_dir = os.path.join(args.log_dir, now)

        utils.setup_logger(args.log_dir, args.resume)
        
        wandb_name = args.dataset + '-' + args.wandb_str + '-' + now
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            save_code=True
            # sync_tensorboard=True
        )
        wandb.run.log_code(".")
        wandb.config.update(args)
        # wandb.define_metric("val_accuracy", step_metric="epoch")
    if args.dist:
        utils.setup_ddp(args)

    main(args)
