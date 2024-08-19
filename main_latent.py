import os
import os.path as osp
import numpy as np
import xarray as xr
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(root_directory2_path)
# sys.path.append(root_directory2_path)
import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
# from torch.cuda.amp import GradScaler
import time

from parser import create_parser
from DDP_latent import Pred_model#, Pred_model_auto
from DDP import init_distributed_mode#, cleanup, train_one_epoch, evaluate, reduce_value, clip_grads

from model import Model_x
from utils.utils import create_folder_if_not_exists, copy_all_files

if __name__ == '__main__':
    time0 = time.time()
    # 集成预报
    # export ESMFMKFILE=/home/lisl/.conda/envs/py39gdal/lib/python3.9/site-packages/esmf/lib/libg/Linux.gfortran.64.mpiuni.default/esmf.mk
    # export CUDA_VISIBLE_DEVICES=1
    # torchrun --nproc_per_node=1 weatherbench2_main/model2023/main_latent.py --epoch 65 --ex_name '240424_baseline'
    # --batch_size 32 --val_batch_size 32 --lr 1e-4 --test 1 --drop 0.2 --weight_decay 1 --clip_grad 5

    # export CUDA_VISIBLE_DEVICES=2
    #  torchrun --nproc_per_node=1 weatherbench2_main/model2023/main_latent.py --epoch 65 --ex_name '240418_wo_mslp'
    #  --batch_size 32 --val_batch_size 32 --lr 1e-4 --test 1 --drop 0.2 --weight_decay 1 --clip_grad 5
    # setting the gpu index
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
    import torch
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.benchmark = True

    args = create_parser().parse_args()
    config = args.__dict__
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")

    # 启动多GPU环境
    init_distributed_mode(args)
    rank = args.rank
    batch_size = args.batch_size
    args.lr *= args.world_size

    # 获得gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print('args.world_size', args.world_size)
        print('rank', rank)
        print('device', device)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.dataname = '_64_32'  #  '_64_32', '_240_121_cn', '_240_121'
    if args.dataname == '_64_32':
        from datasets.dataloader_ERA5 import load_ERA5_dataset_per

        args.minlat = None
        args.pre_seq_length = 1   # per output of the model
        args.aft_seq_length = 1  # total time length to be predicted
        args.aft_seq_length_train = 2  # total time length to be predicted
        args.aft_seq_length_val = 60  # total time length to be predicted
        args.aft_seq_length_test = 60  # total time length to be predicted, set to 60 for insert to fuxi
        args.input_time_length = 2  # input time length of the model, change it, try 4
        args.in_len_val = 5
        args.shrink = 1 if args.input_time_length > args.pre_seq_length else 0
        args.time_emb_num = 15
        args.val_dataset_step = 1  # the step to unfold the val dataset
        args.test_dataset_step = 2  # the step to unfold the test dataset
        args.eps = 1e-3
        # if predict the rainfall, set them to 0 and 1
        args.p_dim = 3
        args.other_dim = 4

        args.pred_tp = False
        args.resume_epoch = None
        # args.epoch_auto = 50
        # args.bs_auto = 36

        args.weight_decay = 1
        args.time_inte = [1,2,4]
        args.iter_len_epoch = [0, 50, 55, 60, 65]
        args.pred_len = list(range(2, 10, 2))
        assert args.iter_len_epoch[-1] <= args.epoch
        assert len(args.pred_len) == len(args.iter_len_epoch) - 1
        args.sched = 'multistep'  # cosine: mae 11.56  ssim 0.216
        # self.args.decay_epochs = 10
        args.decay_milestones = list(range(10, 60, 10))
        args.decay_rate = 0.5
        # args.channel_num = 7
        hid_S, hid_T, N_S, N_T = 24, 256, 4, 4
        dataset_dir = '/data02/lisl/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
        save_dir = '/home/lisl/weatherbench2_main/model_show/'
        results_dir = os.path.join('./results_64_32')
        cp_dir = os.path.join(results_dir, 'checkpoints/', args.ex_name)
        args.save_dir = save_dir

        if rank == 0 and 0:
            # save the datasets in data_npy path
            write_ERA5_dataset_per(dataset_dir=dataset_dir, save_dir=save_dir, args=args)
            exit()

        dataloader_train, sampler_train, dataloader_val, dataloader_test = \
                load_ERA5_dataset_per(batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size,
                      test_batch_size=args.test_batch_size, lon_len=64, lat_len=32,
                      save_dir=save_dir, num_workers=4, distributed=True,
                                      use_prefetcher=False, test=args.test, args=args)

        # modify this path
        min_max_array = np.load(os.path.join(save_dir, "little_files", "mean_std_2000.npy"))

        args.pred_more = True
        if args.pred_more:  #  [ 209.32118 1177.17755]
            mean_std_idx = [0,1,3,4,5,10,12,19,22,24,32,35,45,48,58,61,63,71,74,76]
            args.mean_std_array = np.concatenate([min_max_array[:, mean_std_idx]], 1)  # *************************
            args.target_dim = [i+1 for i in mean_std_idx] # *************************  [0]

            args.var_name_abb = ['U10m', 'V10m', 'T2m', 'mslp', 'sp', 'TCWV', 'Z50', 'Z500', 'Z850', 'Z1000',
                                 'T500', 'T850', 'RH500', 'RH850', 'U500', 'U850', 'U1000', 'V500', 'V850', 'V1000']

        const_data = np.load(osp.join(save_dir, 'little_files', 'var_const_data.npy'))
        args.ch_num = 104 # min_max_array.shape[1] + 1
        args.ch_num_const = const_data.shape[0]
        print('args.ch_num:', args.ch_num)
        print('args.ch_num_const:', args.ch_num_const)
        # args.time_num_val = time_num_val
        const_data = torch.Tensor(const_data)
        in_shape = [args.batch_size, args.input_time_length, args.ch_num, 64, 32]

    model = Model_x(in_shape, out_ch=len(args.target_dim), hid_S=hid_S, hid_T=hid_T, N_S=N_S, drop=args.drop, spatio_kernel_enc=3,
                               spatio_kernel_dec=3, time_emb_num=args.time_emb_num, args=args).to(device)
    # print(sum(para.numel() for para in model.parameters() if para.requires_grad))
    if args.half_precision:
        model = model.half()

    if args.resume_epoch:
        checkpoint_path = os.path.join("/data02/lisl/results/results_64_32/checkpoints/240812_wo_ch_att/weight_50.pth")
        # print('checkpoint_path', self.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 尝试移除下方代码
    if rank == 0:
        print(args)
        # print('checkpoint_path', os.path.join(cp_dir, "initial_weight.pt"))
    # if rank == 0:
    #     torch.save(model.state_dict(), os.path.join(cp_dir, "initial_weight.pt"))
    # dist.barrier()
    time.sleep(1)
    # model.load_state_dict(torch.load(os.path.join(cp_dir, "initial_weight.pt"), map_location=device))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                           output_device=local_rank,
                                                           find_unused_parameters=True)  # device[args.device]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)   # 把这一步放进pred_model中

    pred_model = Pred_model(model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test, const_data,
                            in_shape=[args.batch_size, args.input_time_length,
                                    args.ch_num, *in_shape[-2:]], hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
                                 time_emb_num=args.time_emb_num, results_dir=results_dir, device=device, rank=rank,
                                 local_rank=local_rank, cp_dir=cp_dir, args=args)

    if rank == 0:
        # pred_res = pred_model.test(mode='val')
        pred_res = pred_model.test(mode='test')
