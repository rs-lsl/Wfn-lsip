import os
import os.path as osp
import numpy as np
import xarray as xr
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# root_directory2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../graph_weather'))
# print(root_directory2_path)
# sys.path.append(root_directory2_path)
import warnings
warnings.filterwarnings('ignore')

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.distributed as dist
# from torch.cuda.amp import GradScaler
import time
# import gc
# import tqdm
# 从 visualdl 库中引入 LogWriter 类
# from visualdl import LogWriter
# import timm
# from timm.scheduler import create_scheduler
from datasets.utils_dataset import read_img, write_img_gdal
from parser import create_parser

from model2023.DDP import init_distributed_mode, cleanup, train_one_epoch, evaluate, reduce_value, clip_grads

# from graph_weather_main.graph_weather.models.forecast import GraphWeatherForecaster
# from model2023.metrics import diff_div_reg

# from model2023.optim_scheduler import get_optim_scheduler
from utils.utils0 import create_folder_if_not_exists, copy_all_files

if __name__ == '__main__':
    time0 = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
    import torch
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.benchmark = True

    args = create_parser().parse_args()
    config = args.__dict__
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.local_rank
    # os.environ['OMP_NUM_THREADS'] = 8
    # 启动多GPU环境
    init_distributed_mode(args)
    # 拿到当前进程的rank，注意是当前进程，就像之前说的，每个进程都会执行该段代码，进程0的rank就是0，进程1的rank就是1
    rank = args.rank
    batch_size = args.batch_size
    # 这里学习率需要随着GPU数量成倍增长。具体可以参考B站视频讲解。不过也不一定，自己可以尝试调一调这个参数
    args.lr *= args.world_size

    # 获得gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)

    # 我们只让进程0输出信息，进程1不执行这一部分。这样进程1就不会输出信息，避免重复输出。
    if rank == 0:
        print('args.world_size', args.world_size)
        print('rank', rank)
        print('device', device)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.dataname = '_64_32'  #  '_1440_721_cn', '_240_121_cn', '_240_121' , '_1440_721_cn'   _64_32
    if args.dataname == '_64_32':
        from model2023.datasets.dataloader_ERA5 import write_ERA5_dataset, load_ERA5_dataset2, \
            write_ERA5_dataset_per, load_ERA5_dataset_per#, write_zarr, load_ERA5_dataset_zarr
        from model2023.DDP_latent import Pred_model, Pred_model_auto
        from model import Model_x, Discriminator, Encoder, Decoder
        args.minlat = None
        args.pre_seq_length = 1   # per output of the model
        args.aft_seq_length = 1  # total time length to be predicted
        args.aft_seq_length_train = 2  # total time length to be predicted
        args.aft_seq_length_val = 60  # total time length to be predicted
        args.aft_seq_length_test = 60  # total time length to be predicted, set to 60 for insert to fuxi
        args.input_time_length = 1  # input time length of the model, change it, try 4
        args.in_len_val = 1
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
        args.compute_std_diff = False
        args.compute_climate_mean = False
        # args.epoch_auto = 50
        # args.bs_auto = 36

        args.pin_memory = False
        args.weight_decay = 1
        args.time_inte = [1,2,4]
        args.iter_len_epoch = [0, 5, 10, 15, 20]
        args.pred_len = list(range(2, 10, 2))
        assert args.iter_len_epoch[-1] <= args.epoch
        assert len(args.pred_len) == len(args.iter_len_epoch) - 1
        args.sched = 'multistep'  # cosine: mae 11.56  ssim 0.216
        # self.args.decay_epochs = 10
        args.decay_milestones = list(range(10, 60, 10))
        args.decay_rate = 0.5
        # args.channel_num = 7
        hid_S, hid_T, N_S, N_T = 32, 256, 4, 4
        loss_type_adv = 'binary_cross_entropy'

        root_dir = 'root_dir'
        results_dir = os.path.join('results_dir', 'results/', 'results' + args.dataname)
        save_dir = os.path.join(root_dir, 'era5_post/_64_32/')
        cp_dir = os.path.join(results_dir, 'checkpoints/', args.ex_name)
        if rank == 0:
            create_folder_if_not_exists(cp_dir)
            create_folder_if_not_exists(os.path.join(results_dir, 'logs', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quanti_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'csv_results', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quali_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'pred_results', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'nc_results', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'singular_results', args.ex_name))
        if rank == 0 and 0:
            # write_zarr(save_dir)
            # write_ERA5_dataset(dataset_dir=dataset_dir, base_dir=save_dir, args=args)
            write_ERA5_dataset_per(dataset_dir=dataset_dir, save_dir=save_dir, args=args)
            exit()

        per_read = True
        zarr_read = False
        dataloader_train, sampler_train, dataloader_val, dataloader_test = \
                load_ERA5_dataset_per(batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size,
                      test_batch_size=args.test_batch_size, lon_len=64, lat_len=32,
                      save_dir=save_dir, num_workers=4, distributed=True,
                                      use_prefetcher=False, test=args.test, args=args)

        # min_max_array = np.load(osp.join(save_dir, 'mean_std.npy'))
        min_max_array = np.load(osp.join(save_dir, 'mean_std_2000.npy'))
        # print(min_max_array-min_max_array2)
        # print(min_max_array)
        # print(min_max_array2)
        args.pred_mslp = False
        args.pred_more = True
        if args.pred_more:  #  [ 209.32118 1177.17755]
            mean_std_idx = [0,1,3,4,5,10,12,19,22,24,32,35,45,48,58,61,63,71,74,76]
            args.mean_std_array = np.concatenate([min_max_array[:, mean_std_idx]], 1)  # *************************
            args.target_dim = [i+1 for i in mean_std_idx] # *************************  [0]

            args.var_name_abb = ['U10m', 'V10m', 'T2m', 'mslp', 'sp', 'TCWV', 'Z50', 'Z500', 'Z850', 'Z1000',
                                 'T500', 'T850', 'SH500', 'SH850', 'U500', 'U850', 'U1000', 'V500', 'V850', 'V1000']

        const_data = np.load(osp.join(save_dir, 'var_const_data.npy'))
        args.ch_num = 104  #len(args.target_dim) # min_max_array.shape[1] + 1
        args.ch_num_const = const_data.shape[0]
        print('args.ch_num:', args.ch_num)
        print('args.ch_num_const:', args.ch_num_const)
        # args.time_num_val = time_num_val
        const_data = torch.Tensor(const_data)
        in_shape = [args.batch_size, args.input_time_length, args.ch_num, 64, 32]

    elif args.dataname == '_1440_721_cn':   #  [ 61.56464 216.08519]
        from datasets.dataloader_ERA5 import Write_ZARR_files, load_ERA5_dataset2, \
            load_ERA5_dataset_1440_721_cn, write_zarr, write_ERA5_dataset_1440_721_cn
        from DDP_1440_721_cn import Pred_model, Pred_model_auto
        from model_1440_721_cn import Model_x, Encoder, Decoder

        args.maxlat, args.minlat, args.minlon, args.maxlon = 57, 15, 84, 130  # target index, not true lat, lon
        args.pre_seq_length = 1  # per output of the model
        args.aft_seq_length = 1  # total time length to be predicted
        args.aft_seq_length_train = 2  # total time length to be predicted
        args.aft_seq_length_val = 60  # total time length to be predicted
        args.aft_seq_length_test = 60  # total time length to be predicted, set to 60 for insert to fuxi
        args.input_time_length = 1  # input time length of the model, change it, try 4
        args.in_len_val = 1 if args.input_time_length == 1 else 5
        args.shrink = 1 if args.input_time_length > args.pre_seq_length else 0
        args.time_emb_num = 12
        args.val_dataset_step = 1  # the step to unfold the val dataset
        args.test_dataset_step = 2  # the step to unfold the test dataset
        args.eps = 1e-3
        # if predict the rainfall, set them to 0 and 1
        args.p_dim = 3
        args.other_dim = 4

        args.pred_tp = False
        args.resume_epoch = None
        args.compute_mean_std = False  # --nproc_per_node=1 ************************
        args.compute_climate_mean = False
        args.compute_std_diff = False

        args.trainset_ratio = 0.666667  # 0.666667
        args.pin_memory = False
        args.tar_dim = [[40, -41], [40, -41]]
        args.weight_decay = 1e-6
        args.H_d, args.W_d = 64, 64
        args.sample_inte_test = 12
        args.tar_size = 64
        args.time_inte = [1, 2, 4]
        args.iter_len_epoch = [0, 50]
        args.pred_len = list(range(2, 4, 2))
        assert args.iter_len_epoch[-1] <= args.epoch
        assert len(args.pred_len) == len(args.iter_len_epoch) - 1
        args.sched = 'multistep'  # cosine: mae 11.56  ssim 0.216
        # self.args.decay_epochs = 10
        args.decay_milestones = list(range(10, 60, 10))
        args.decay_rate = 0.5
        # args.channel_num = 7
        hid_S, hid_T, N_S, N_T = 32, 256, 4, 4
        loss_type_adv = 'binary_cross_entropy'
        root_dir = 'root_dir'
        results_dir = os.path.join('results_dir', 'results/', 'results' + args.dataname)
        save_dir = os.path.join(root_dir, 'era5/2021_2023_asia/')
        cp_dir = os.path.join(results_dir, 'checkpoints/', args.ex_name)
        if rank == 0:
            create_folder_if_not_exists(cp_dir)
            create_folder_if_not_exists(os.path.join(results_dir, 'logs', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quanti_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'csv_results', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'quali_figures', args.ex_name))
            create_folder_if_not_exists(os.path.join(results_dir, 'pred_results', args.ex_name))
        if rank == 0 and 0:
            # write_zarr(save_dir)
            # write_ERA5_dataset(dataset_dir=dataset_dir, base_dir=save_dir, args=args)
            write_ERA5_dataset_1440_721_cn(dataset_dir=dataset_dir, save_dir=save_dir, args=args)
            # Write_ZARR_files(args=args)
            exit()
        num_workers = 24 if not args.test else 0
        dataloader_train, sampler_train, dataloader_val, dataloader_test = \
            load_ERA5_dataset_1440_721_cn(batch_size=args.batch_size,
                                  val_batch_size=args.val_batch_size,
                                  test_batch_size=args.test_batch_size, lon_len=145, lat_len=145,
                                  era5_data_path=save_dir, num_workers=num_workers, distributed=True,
                                  use_prefetcher=True, test=args.test, root_dir=root_dir, args=args)

        args.pred_more = True
        args.pred_104 = False
        if args.pred_more:  # [ 209.32118 1177.17755]
            args.var_name_abb = ['Z50', 'Z500', 'Z850', 'Z1000',
                                 'T500', 'T850', 'U500', 'U850', 'U1000', 'V500', 'V850', 'V1000',
                                 't2m', 'msl', 'sst', 'u100', 'v100']
            args.target_dim = [139, 126, 117, 111, 52, 43, 163, 154, 148, 200, 191, 185, 224, 225, 226, 229, 230]

        # var_name = ['q', 't', 'w', 'z', 'u', 'v']
        # "pressure_level": [
        #     "1", "2", "3",
        #     "5", "7", "10",
        #     "20", "30", "50",
        #     "70", "100", "125",
        #     "150", "175", "200",
        #     "225", "250", "300",
        #     "350", "400", "450",
        #     "500", "550", "600",
        #     "650", "700", "750",
        #     "775", "800", "825",
        #     "850", "875", "900",
        #     "925", "950", "975",
        #     "1000"

        const_data = None #np.load(osp.join(save_dir, 'var_const_data.npy'))
        args.ch_num = 242  # min_max_array.shape[1] + 1
        # args.ch_num_const = const_data.shape[0]
        print('args.ch_num:', args.ch_num)
        # print('args.ch_num_const:', args.ch_num_const)
        # args.time_num_val = time_num_val
        # const_data = torch.Tensor(const_data)
        in_shape = [args.batch_size, args.input_time_length, args.ch_num, 145, 145]


    # act_inplace = False
    # enc = Encoder(args.ch_num, hid_S, N_S, 3,
    #                    act_inplace=act_inplace).to(device)  # C_in, C_hid, N_S, spatio_kernel, act_inplace=True
    # dec = Decoder(hid_S, args.ch_num, N_S, 3,
    #                    act_inplace=act_inplace).to(device)  # 1 means the total_precipitation_6hr var
    #
    # in_shape = [args.batch_size, args.input_time_length, args.ch_num, 64, 32]
    # # model = SimVP_Model_x(in_shape, out_ch=2 + args.other_dim - args.p_dim, hid_S=hid_S, hid_T=hid_T, N_S=N_S,
    # #                       drop=args.drop, spatio_kernel_enc=3,
    # #                       spatio_kernel_dec=3, time_emb_num=args.time_emb_num, args=args).to(device)
    #
    # enc = torch.nn.parallel.DistributedDataParallel(enc, device_ids=[local_rank],
    #                                                   output_device=local_rank,
    #                                                   find_unused_parameters=False)  # device[args.device]
    # dec = torch.nn.parallel.DistributedDataParallel(dec, device_ids=[local_rank],
    #                                                   output_device=local_rank,
    #                                                   find_unused_parameters=False)  # device[args.device]
    # optimizer_auto = torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()), lr=args.lr, betas=(0.9, 0.95),
    #                               weight_decay=args.weight_decay)  # 把这一步放进pred_model中
    #
    # pred_model = Pred_model_auto(enc, dec, optimizer_auto, dataloader_train_auto, sampler_train_auto, const_data,
    #                         in_shape=[args.batch_size, args.input_time_length,
    #                                   args.ch_num, 64, 32], hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
    #                         time_emb_num=args.time_emb_num, results_dir=results_dir, device=device, rank=rank,
    #                         local_rank=local_rank, loss_type=loss_type_adv, cp_dir=cp_dir, args=args)
    # if not args.test:
    #     pred_model.train()

    # main training process
    # act_inplace = False
    # enc = Encoder(args.ch_num, hid_S, N_S, 3,
    #                    act_inplace=act_inplace).to(device)  # C_in, C_hid, N_S, spatio_kernel, act_inplace=True
    # # dec = Decoder(hid_S, 2+args.other_dim-args.p_dim, N_S, 3,
    # #                    act_inplace=act_inplace).to(device)  # 1 means the total_precipitation_6hr var
    # # print(torch.load(os.path.join(cp_dir, "weight_enc.pth")))
    # enc.load_state_dict(torch.load(os.path.join(cp_dir, "weight_enc.pth"), map_location=device))
    # # enc.load_state_dict(torch.load(os.path.join(cp_dir, "weight_dec.pth"), map_location=device))

    model = Model_x(in_shape, out_ch=len(args.target_dim), hid_S=hid_S, hid_T=hid_T, N_S=N_S, drop=args.drop, spatio_kernel_enc=3,
                               spatio_kernel_dec=3, time_emb_num=args.time_emb_num, args=args).to(device)
    # print(sum(para.numel() for para in model.parameters() if para.requires_grad))
    if args.half_precision:
        model = model.half()

    if args.resume_epoch:
        checkpoint_path = os.path.join(results_dir, "checkpoints", args.ex_name , "weight_"+str(args.resume_epoch)+".pth")
        # print('checkpoint_path', self.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 尝试移除下方代码
    if rank == 0:
        print(args)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                           output_device=local_rank,
                                                           find_unused_parameters=True)  # device[args.device]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)   # 把这一步放进pred_model中

    pred_model = Pred_model(model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test, const_data,
                            in_shape=[args.batch_size, args.input_time_length,
                                    args.ch_num, *in_shape[-2:]], hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
                                 time_emb_num=args.time_emb_num, results_dir=results_dir, device=device, rank=rank,
                                 local_rank=local_rank, loss_type=loss_type_adv, cp_dir=cp_dir, args=args)

    if not args.test:
        print('args.epoch:', args.epoch)
        pred_model.train()
        print(f'Using time: {time.time()-time0}') #

    save_path = os.path.join(results_dir, 'pred_results', args.ex_name, 'pred_res_ours.npy') #if args.pred_mslp else \
                #os.path.join(results_dir, 'pred_results', 'result_ours_wo_mslp.npy')
    if rank == 0:
        # pred_res = pred_model.test(mode='val')
        pred_res = pred_model.test(mode='test')

