
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import mnist
from torch.autograd import Variable
from tqdm import tqdm
import sys
import pandas as pd
import tempfile
import time
import numpy as np
from timm.scheduler import create_scheduler
# from visualdl import LogWriter
# from weatherbench2 import config
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from metrics import metric



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return
    # print(args)
    #
    args.distributed = True

    # 这里需要设定使用的GPU
    torch.cuda.set_device(args.gpu)
    # 这里是GPU之间的通信方式，有好几种的，nccl比较快也比较推荐使用。
    args.dis_backend = 'nccl'
    # 启动多GPU
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        rank=args.rank
    )
    # 这个是：多GPU之间进行同步，也就是有的GPU跑的快，有的跑的慢（比如当你判断if RANK == 0: do something， 那么进程0就会多执行代码速度慢）
    # 所以这个代码就是等待所有进程运行到此处。
    dist.barrier()


def cleanup():
    # 这里不同我多说，看名字就知道啥意思
    dist.destroy_process_group()

# 判断多GPU是否启动
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
# 拿到你有几个GPU，数量。主要是用来all_reduce计算的。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 拿到进程的rank
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# 这个主要就是和all_reduce差不多，每一个进程都会算到一个值，把不同进程的值回合起来。
# 比如对于loss而言，进程0是1、2样本得到的loss是0.1，进程1是3、4样本得到的loss是0.2，那么
# 该批次的all_loss就是0.1+0.2=0.3
# 同样 你在计算分类正确时候 比如batch是100，进程0得到的正确个数是50 进程1得到的正确率60，那么整体的准确率就是110/200
def reduce_value(value, average=True):
    # 拿到GPU个数，主要是判断我们有几个进程
    world_size = get_world_size()
    # 如果单进程就返回
    if world_size < 2:
        return value

    with torch.no_grad():
        # 这个就是all_reduce把不同进程的值都汇总返回。
        dist.all_reduce(value)
        if average:
            # 是否取均值
            value /= world_size
        return value

# 判断是否是主进程，主进程的意思就是rank=0，
# 严格意义上来说没有主进程之分，你想进程1是主进程，那么你就 get_rank() == 1就行。
def is_main_process():
    return get_rank() == 0

def clip_grads(params, args, norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    args.clip_mode = args.clip_mode if args.clip_grad is not None else None
    if args.clip_mode is None:
        return
    if args.clip_mode == 'norm':
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad, norm_type=norm_type)
    elif args.clip_mode == 'value':
        torch.nn.utils.clip_grad_value_(params, args.clip_grad)
    else:
        assert False, f"Unknown clip mode ({args.clip_mode})."

class Pred_model(nn.Module):
    def __init__(self, model, optimizer, dataloader_train, sampler_train, dataloader_val, dataloader_test, const_data,
                 in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True,
                 time_emb_num=10, results_dir='', device=None, rank=0,
                 local_rank=0, loss_type='', cp_dir=None, args=None, **kwargs):
        super(Pred_model, self).__init__()
        self.args = args
        self.results_dir = results_dir
        self.cp_dir = cp_dir
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        # self.loss_weight = loss_weight  #
        B, T, C, H, W = in_shape  # T is input_time_length
        self.shape_val = [H, W]
        self.bs = B
        self.ch = C
        # self.target_dim = args.target_dim # list(range(args.p_dim, args.other_dim))+[15]  # ****************************

        self.dataloader_train, self.sampler_train, self.dataloader_val, self.dataloader_test = \
            dataloader_train, sampler_train, dataloader_val, dataloader_test
        # self.sampler_train = sampler_train
        self.const_data = const_data.type(torch.float32).to(self.device, non_blocking=True)

        self.model = model
        # self.checkpoint_path = os.path.join(self.cp_dir, "initial_weight.pt")
        # print(args)

        if rank == 0:
            print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))

        log_path = os.path.join(self.results_dir, 'logs', args.ex_name)
        # self.logwriter = LogWriter(logdir=log_path)

        self.init_lat_weight()
        # self.init_model()
        # self.adv_loss = AdversarialLoss(discriminator, loss_type=loss_type)

    def init_lat_weight(self):
        # modify this path
        a_weight = np.load(os.path.join(self.args.save_dir, "little_files", "lat_weight.npy"))
        lat_weight = torch.from_numpy(a_weight).reshape((1, 1, 1, 1, -1))#.clamp(0)
        self.lat_weight = lat_weight.type(torch.float32).to(self.device)

        time_weight = torch.from_numpy(np.ones(10)).reshape([1,10,1,1,1])
        # if self.rank == 0:
        #     print(f'time weight: {time_weight.squeeze()}')
        self.time_weight = time_weight.type(torch.float32).to(self.device)

        self.height_data = torch.from_numpy(np.array([1000] * 13 + list(range(300, 800, 100))  # **********************
                                                    + [600, 700, 850, 925, 1000] * 6)).type(torch.float32).to(self.device, non_blocking=True)
        mean0 = torch.mean(self.height_data)
        std0 = torch.std(self.height_data)
        self.height_data = (self.height_data - mean0) / std0

        # 大气变量的权重均为1，表面变量中，T2m为1，其他为0.1
        var_weight = torch.from_numpy(np.array([0.1, 0.1, 1.0, 1.0])).reshape([1,1,4,1,1]) if not self.args.pred_more else \
                            torch.from_numpy(np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1]+[1]*(len(self.args.var_name_abb)-6))).reshape([1, 1, len(self.args.var_name_abb), 1, 1])
        self.var_weight = var_weight.type(torch.float32).to(self.device)
        print(self.var_weight.flatten())

    def ACC(self, pred, true):
        value = torch.mean(torch.sum(pred * true * self.lat_weight, dim=(-1, -2)) / torch.sqrt(
            torch.sum((pred ** 2) * self.lat_weight, dim=(-1, -2)) * torch.sum((true ** 2) * self.lat_weight, dim=(-1, -2))))
        return value

    def test(self, mode='val'):
        state_dict = torch.load(
            os.path.join(os.path.join(self.args.save_dir, 'weights', 'weight.pth')))
        if self.args.dist:
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        # grad = self.evaluate_grad(
        #     metric_list=['mae', 'rmse'], mode=mode # the validation dataset
        # )

        sum_num, pred_res = self.evaluate(
            metric_list=['mae', 'rmse'], mode=mode # the validation dataset
        )

        # if self.rank == 0:
        print('metrics:    ', '    mae   ', ' rmse ')  # , 'snr', 'lpips'
        print('Eval results:', sum_num)

        # output_list = [torch.zeros(2)[None, ...].to(self.device) for _ in range(self.args.world_size)]
        # dist.all_gather(output_list, torch.Tensor(sum_num)[None, ...].to(self.device))
        # if self.rank == 0:
        #     print(torch.mean(torch.cat(output_list), 0).cpu().numpy())
        return pred_res

    def evaluate(self, epoch=None, metric_list=['mae', 'mse', 'rmse', 'ssim'], mode='val'):
        if mode == 'val':
            forcast_len = self.args.aft_seq_length_val
            dataloader = self.dataloader_val
        elif mode == 'test':
            forcast_len = self.args.aft_seq_length_test
            dataloader = self.dataloader_test
        # self.args.aft_seq_length = self.args.aft_seq_length_test
        spatial_norm = True
        self.model.eval()
        eval_res_list = []
        # pred_res = []
        pred_res = np.empty([0, 60, 20, 64, 32])
        label = np.empty([0, 60, 20, 64, 32])
        mean = []
        std = []
        time_list = []
        # fuxi_path = '/data02/lisl/forcast/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr'
        # dataset = xr.open_zarr(fuxi_path, chunks=None)
        # step_len = (dataset['10m_wind_speed'].data).shape[0]
        # forecast_path = "/data02/lisl/forcast/ours/2020-240x121_equiangular_conservative.zarr"

        with torch.no_grad():
            for step, (images, time_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)
                bs_idx = step * time_data.shape[0]
                inputs = images[:, :self.args.in_len_val, ...].type(torch.float32).to(self.device, non_blocking=True)#.clone()
                labels = images[:, self.args.in_len_val:, ...].type(torch.float32).to(self.device, non_blocking=True)#.clone()
                time_data = time_data.type(torch.float32).to(self.device, non_blocking=True)

                if self.args.half_precision:
                    inputs = inputs.half()  # .half()
                    labels = labels.half()  # .half()
                pred, _, _, _, _, _, _ = self.model(inputs, self.const_data, time_data, labels=None,
                                                    aft_seq_length=forcast_len,
                                                    shrink=self.args.shrink, mode=mode)
                if mode == 'test':  # and (step % 2) == 0 *************************************
                    pred_res = np.concatenate([pred_res, pred.cpu().numpy()], 0)  # = np.concatenate([pred_res, pred.cpu().numpy()], 0)
                    label = np.concatenate([label, labels[:, :, self.args.target_dim].clone().cpu().numpy()],
                                              0)  # = np.concatenate([pred_res, pred.cpu().numpy()], 0)

                eval_res = metric(self.trans_mean_std(pred),
                                  self.trans_mean_std(labels[:, :, self.args.target_dim]),
                                  weight=self.lat_weight.cpu().numpy())

                eval_res_list.append(eval_res['rmse'])
                if self.args.empty_cache:
                    torch.cuda.empty_cache()

        eval_res_list = np.array(eval_res_list).mean(0)
        print(eval_res_list.shape)
        # plot the results, for example, plot the first variable U10m with index 0
        #  args.var_name_abb = ['U10m', 'V10m', 'T2m', 'mslp', 'sp', 'TCWV', 'Z50', 'Z500', 'Z850', 'Z1000',
        #                       'T500', 'T850', 'RH500', 'RH850', 'U500', 'U850', 'U1000', 'V500', 'V850', 'V1000']
        plt.plot(eval_res_list[:, 0])
        plt.xlabel('Lead time step')
        plt.savefig(os.path.join(self.args.save_dir, 'rmse_U10m.png'), dpi=100)  # 300
        plt.close()
        exit()

        return pred_res

    def trans_mean_std(self, res):
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()
        mean = self.args.mean_std_array[0].reshape([1,1,-1,1,1])
        std = self.args.mean_std_array[1].reshape([1,1,-1,1,1])

        res = res * std + mean
        return res

    def transform_log(self, res):
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()
        # res = res * (self.args.min_max_array[-1] - self.args.min_max_array[-2]) + self.args.min_max_array[-2]
        res = (np.exp(res + np.log(self.args.eps)) - self.args.eps) * self.args.min_max_array[1] + self.args.min_max_array[0]
        # res = res * (self.args.min_max_array[1] - self.args.min_max_array[0]) + self.args.min_max_array[0]

        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    rank = '0, 1'
    print(f'cuda:{rank}')
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['LOCAL_RANK'] = '0'
    # os.environ['MASTER_ADDR'] = '10.102.105.91'
    # os.environ['MASTER_PORT'] = '5678'
    # print(os.environ['RANK'])
    # print(os.environ['WORLD_SIZE'])
    # print(os.environ['LOCAL_RANK'])
    # print(os.environ['MASTER_ADDR'])
    # print(os.environ['MASTER_PORT'])

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--dis_url', type=str, default='env://')
    args = parser.parse_args()

    main(args)