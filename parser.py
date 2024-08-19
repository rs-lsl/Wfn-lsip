# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_parser():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=True,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--res_dir', default=os.path.join(base_dir, 'checkpoints'), type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--diff_seed', action='store_true', default=False,
                        help='Whether to set different seeds for different ranks')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--empty_cache', action='store_true', default=True,
                        help='Whether to empty cuda cache after GPU training')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Whether to find unused parameters in forward during DDP training')  # for the RNN-based methods, it is true
    parser.add_argument('--broadcast_buffers', action='store_false', default=True,
                        help='Whether to set broadcast_buffers to false during DDP training')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', default=0, type=int, help='Only performs testing')   # *****
    parser.add_argument('--inference', '-i', action='store_true', default=False, help='Only performs inference')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')
    parser.add_argument('--launcher', default='none', type=str,
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='job launcher for distributed training')
    parser.add_argument('--local_rank', type=str, default='0')  # local_rank
    parser.add_argument('--dis_url', type=str, default='env://')
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')
    parser.add_argument('--half_precision', default=0, type=int, help='if half_precision training')   # *****

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', default=64, type=int, help='Validation batch size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='/data01/lisl/Rain_estimation/')
    parser.add_argument('--dataname', default='_64_32', type=str,
                        help='Dataset name (default: "era5_6432")')  # era5_12864
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--aft_seq_length_train', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--aft_seq_length_test', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--input_time_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--shrink', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--val_dataset_step', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--test_dataset_step', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--channel_num', default=None, type=int, help='channel number')
    parser.add_argument('--channel_num_const', default=None, type=int, help='channel number of const var')
    parser.add_argument('--time_emb_num', default=None, type=int, help='feature number of the time embedding')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,   # for the crevnet method, it must be true
                        help='Whether to drop the last batch in the val data loading')

    parser.add_argument('--min_max_array', default=[0,1,0,1], type=float, help='min_max_array')
    parser.add_argument('--eps', default=1e-11, type=float, help='the factor of the log transform')
    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'CrevNet', 'crevnet', 'DMVFN', 'dmvfn', 'E3DLSTM', 'e3dlstm',
                                 'MAU', 'mau', 'MIM', 'mim', 'PhyDNet', 'phydnet', 'PredNet', 'prednet',
                                 'PredRNN', 'predrnn', 'PredRNNpp', 'predrnnpp', 'PredRNNv2', 'predrnnv2',
                                 'SimVP', 'simvp', 'TAU', 'simvp_x', 'tau'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default='/configs/mmnist/simvp/SimVP_gSTA.py', type=str,
                        help='Path to the default config file')
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to allow overwriting the provided config file with args')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--eval_iter', default=1, type=int, help='eval interval in training process')
    parser.add_argument('--save_iter', default=10, type=int, help='save interval in training process')
    parser.add_argument('--display_step', default=1, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--log_step', default=1000, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')  # 该值越大，对梯度的约束越强，越不容易引发梯度爆炸
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--early_stop_epoch', default=-1, type=int,
                        help='Check to early stop after this epoch')
    parser.add_argument('--no_display_method_info', action='store_true', default=True,
                        help='Do not display method info')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight')

    # Training parameters (scheduler)
    # parser.add_argument('--sched', default='onecycle', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "onecycle"')
    # parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
    #                     help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')
    parser.add_argument('--loss_weight_latent', type=float, default=1,
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--loss_weight_in_recon', type=float, default=1,
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--target_dim', type=list, default=[],
                        help='LR decay rate (default: 0.1)')

    # fourcastnet
    # parser = argparse.ArgumentParser('FourCastNet training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--pretrain-epochs', default=80, type=int)
    parser.add_argument('--fintune-epochs', default=25, type=int)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')

    # parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--decay_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_milestones', type=list, default=[5], metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--time_inte', type=list, default=[1,2,4,8], metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # fno parameters
    parser.add_argument('--fno-bias', action='store_true')
    parser.add_argument('--fno-blocks', type=int, default=4)
    parser.add_argument('--fno-softshrink', type=float, default=0.00)
    parser.add_argument('--double-skip', action='store_true')
    parser.add_argument('--tensorboard-dir', type=str, default=None)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--checkpoint-activations', action='store_true')
    parser.add_argument('--autoresume', action='store_true')

    # attention parameters
    parser.add_argument('--num-attention-heads', type=int, default=1)

    # long short parameters
    parser.add_argument('--ls-w', type=int, default=4)
    parser.add_argument('--ls-dp-rank', type=int, default=16)

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'display_step': 10,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'use_gpu': True,
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'diff_seed': False,
        'fps': False,
        'empty_cache': True,
        'find_unused_parameters': False,
        'broadcast_buffers': True,
        'resume_from': None,  # *****
        'auto_resume': False,
        'test': False,    # *****
        'inference': False,
        'deterministic': False,
        'launcher': 'pytorch',
        'local_rank': 0,    # *****
        'port': 29500,
        # dataset parameters
        'batch_size': 64,     # *****
        'val_batch_size': 64,  # *****
        'num_workers': 4,
        'data_root': './data',  # *****
        'dataname': 'mmnist',  # *****
        'pre_seq_length': 10,  # *****
        'aft_seq_length': 10,  # *****
        'total_length': 20,    # *****
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',  # *****
        'config_file': '/home/lisl/OpenSTL_master/configs/mmnist/simvp/SimVP_gSTA.py',  # *****
        'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,  # *****
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'early_stop_epoch': -1,
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,  # *****
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
    }
    return default_values
