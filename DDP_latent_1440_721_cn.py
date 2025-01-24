import os
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
from weatherbench2 import config
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from model2023.metrics import metric
from model2023.utils.utils0 import find_min_value, find_max_value, add_diff_to_strings

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
        # print('        self.ch = C',         self.ch)
        # self.target_dim = args.target_dim # list(range(args.p_dim, args.other_dim))+[15]  # ****************************

        self.dataloader_train, self.sampler_train, self.dataloader_val, self.dataloader_test = \
            dataloader_train, sampler_train, dataloader_val, dataloader_test
        # self.sampler_train = sampler_train
        self.const_data = const_data#.type(torch.float32).to(self.device, non_blocking=True)

        self.model = model
        # self.checkpoint_path = os.path.join(self.cp_dir, "initial_weight.pt")
        # print(args)

        if rank == 0:
            print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))

        log_path = os.path.join(self.results_dir, 'logs', args.ex_name)
        # self.logwriter = LogWriter(logdir=log_path)

        if not args.test:
            # self.steps_per_epoch = len(dataloader_train)
            self.init_optim(optimizer)

        self.init_lat_weight()
        # self.init_model()
        # self.adv_loss = AdversarialLoss(discriminator, loss_type=loss_type)

    def init_lat_weight(self):
        a_weight = np.load(os.path.join(self.results_dir, 'lat_weight.npy'))
        lat_weight = torch.from_numpy(a_weight).reshape((1, 1, 1, -1, 1))#.clamp(0)
        self.lat_weight = lat_weight.type(torch.float32).to(self.device)
        self.lat_weight_tar = self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]]

        self.mean_std = torch.from_numpy(np.load(os.path.join(self.results_dir, 'mean_std.npy'))).type(
            torch.float32).to(self.device)
        self.mean_std_climate = torch.from_numpy(np.load(os.path.join(self.results_dir, 'mean_std_climate.npy'))[:, :, self.args.target_dim, None, None]).type(
            torch.float32).to(self.device)
        self.mean_std_climate = torch.cat([self.mean_std_climate, self.mean_std_climate, self.mean_std_climate[:, :12]], 1)
        # print(self.mean_std_climate.shape)
        # if self.rank == 0:
        #     print(self.mean_std)

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
        var_weight = torch.from_numpy(np.ones(len(self.args.target_dim))).reshape([1, 1, len(self.args.target_dim), 1, 1])
        # var_weight = torch.from_numpy(np.array([1]*13+[0.1]*3)).reshape([1,1,len(self.args.target_dim),1,1]) #if not self.args.pred_more else \
                            # torch.from_numpy(np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1]+[1]*(len(self.args.var_name_abb)-6))).reshape([1, 1, len(self.args.var_name_abb), 1, 1])
        if self.args.pred_104:
            var_weight = torch.from_numpy(np.array([0.1]*3+[1]+[0.1]*8+[1]*(len(self.args.target_dim)-12))).reshape([1, 1, len(self.args.target_dim), 1, 1])
        self.var_weight = var_weight.type(torch.float32).to(self.device)
        # print(self.var_weight.flatten())

    def init_optim(self, optimizer):
        # param_groups = timm.optim.optim_factory.param_groups_weight_decay(model, args.weight_decay)
        self.optimizer = optimizer   #  torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.95), weight_decay=self.args.weight_decay)
        # loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.scheduler, _ = create_scheduler(self.args, self.optimizer)
        if self.args.resume_epoch is not None:
            self.scheduler.step(self.args.resume_epoch)
        self.criterion = nn.L1Loss()
        # self.criterion_latent = nn.L1Loss()
        self.latent_weight_epoch = np.arange(0.1, 2, (2 - 0.1)/self.args.epoch)

    def init_model(self):

        if self.args.half_precision:
            self.model = self.model.half()

        # 尝试移除下方代码
        self.checkpoint_path = os.path.join(self.cp_dir, "initial_weight.pt")
        # print('checkpoint_path', self.checkpoint_path)
        if self.rank == 0:
            torch.save(self.model.state_dict(), self.cp_dir)
        dist.barrier()
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                                                          find_unused_parameters=True)  # device[args.device]

    def train(self):
        # print(self.args.iter_len_epoch)
        start_epoch = 0 if self.args.resume_epoch is None else self.args.resume_epoch
        for epoch in range(start_epoch, self.args.epoch):

            for i in range(len(self.args.iter_len_epoch) - 1):
                if self.args.iter_len_epoch[i] <= epoch < self.args.iter_len_epoch[i + 1]:
                    dataloader_train = self.dataloader_train[i]
                    self.sampler_train[i].set_epoch(epoch)  # 先不加
                    aft_seq_length = self.args.pred_len[i]
                    break

            time0 = time.time()
            # try to return the model, optimizer, scheduler
            loss_total = self.train_one_epoch(epoch, dataloader_train, aft_seq_length=aft_seq_length)

            self.scheduler.step(epoch)

            if self.rank == 0 and (epoch + 1) % self.args.save_iter == 0:
                save_path = os.path.join(self.cp_dir, "weight_"+str(epoch+1)+".pth")
                torch.save(self.model.module.state_dict(), save_path)

            if self.rank == 0 and (epoch + 1) == self.args.epoch:
                # print("[epoch {}] accuracy {}".format(epoch, sum_num))
                save_path = os.path.join(self.cp_dir, "weight.pth")
                torch.save(self.model.module.state_dict(), save_path)

            if self.rank == 0 and (epoch + 1) % self.args.display_step == 0:
                print("[epoch {}/{}] train_loss: {}, using time {}".format(epoch + 1, self.args.epoch, loss_total,
                                                                           time.time() - time0))

            dist.barrier()  # 先不加


        if self.rank == 0:
            # print("[epoch {}] accuracy {}".format(epoch, sum_num))
            save_path = os.path.join(self.cp_dir, "weight.pth")
            torch.save(self.model.module.state_dict(), save_path)

        # if self.rank == 0:
        #     if os.path.exists(self.checkpoint_path) is True:
        #         os.remove(self.checkpoint_path)

        cleanup()

    def train_one_epoch(self, epoch, dataloader_train, aft_seq_length=2):
        # self.args.aft_seq_length = self.args.aft_seq_length_train
        # torch.autograd.set_detect_anomaly(True)  # check
        self.model.train()

        loss_total = 0.0
        time0 = time.time()
        # train_pbar = tqdm(dataloader_train) if rank == 0 else dataloader_train
        for step, (images, images_surface, time_data, rand_idx) in enumerate(dataloader_train):
            self.optimizer.zero_grad()
            images_surface = torch.nan_to_num(images_surface).type(torch.float32)
            images = images.flatten(2,3)
            images = torch.cat([images, images_surface], 2)

            if self.args.compute_std_diff:
                if rand_idx[0].int() == 0:
                    tmp_img = torch.cat([(images[:, i] - images[:, i-1])[:, None] for i in range(1, (self.args.input_time_length+aft_seq_length))], 1)
                    mean_list1.append(torch.mean(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                    std_list1.append(torch.std(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                elif rand_idx[0].int() == 1:
                    tmp_img = torch.cat([(images[:, i] - images[:, i-1])[:, None] for i in
                                            range(1, (self.args.input_time_length + aft_seq_length))], 1)
                    mean_list2.append(torch.mean(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                    std_list2.append(torch.std(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                elif rand_idx[0].int() == 2:
                    tmp_img = torch.cat([(images[:, i] - images[:, i-1])[:, None] for i in
                                                range(1, (self.args.input_time_length + aft_seq_length))], 1)
                    mean_list4.append(torch.mean(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                    std_list4.append(torch.std(tmp_img, dim=[0, 1, 3, 4], keepdim=False)[None, ...])
                if step == len(dataloader_train) - 1 and epoch == 0:  # drop last = False
                    self.save_std_diff(mean_list1, mean_list2, mean_list4, std_list1, std_list2, std_list4)
                    exit()

            inputs = (images[:, :self.args.input_time_length, ...].clone()+ \
                      (torch.randn(size=[self.bs, self.args.input_time_length, self.ch, *self.shape_val],
                                 dtype=torch.float32)/100.0).to(self.device, non_blocking=True))#.to(self.device, non_blocking=True)
            # inputs = inputs + ().to(self.device, non_blocking=True)
            labels = images[:, self.args.input_time_length:(self.args.input_time_length+aft_seq_length), ...].clone()#.type(torch.float32).to(self.device, non_blocking=True)
            # time_data = time_data.type(torch.float32).to(self.device, non_blocking=True)
            with autocast():  # 混合精度训练/半精度

                pred, pred_latent, true_latent, label_pred, embed, embed_diff, embed_label_dec\
                    = self.model(inputs, self.const_data, time_data, labels,
                                                                        aft_seq_length=aft_seq_length, hid_i=rand_idx[0].int(),
                                                                        shrink=self.args.shrink, mode='train', device=self.device)  # 尝试更换latent的维度T

                loss0 = self.time_weighted_L1_loss(pred, labels[:, :, self.args.target_dim,
                                                         self.args.tar_dim[0][0]:self.args.tar_dim[0][1],
                     self.args.tar_dim[1][0]:self.args.tar_dim[1][1]].clone(), latent=False, aft_seq_length=aft_seq_length)  # [:, :, self.target_dim]

                loss1 = self.time_weighted_L1_loss(pred_latent, true_latent, latent=True,
                                                         aft_seq_length=aft_seq_length)
                loss2 = self.weighted_L1_loss(label_pred, images[:, :self.args.input_time_length].clone())
                loss4 = self.weighted_L1_loss(labels, embed_label_dec)

                loss = loss0 + loss1 + loss2 + loss4 #+ args.alpha * diff_div_reg(pred, labels)   # 维度加权损失函数

            with torch.autograd.detect_anomaly():
                loss.backward()
            # clip_grads(self.model.parameters(), self.args, norm_type=2.0)   #  adjust
            self.optimizer.step()

            loss_total += loss.item()
            torch.cuda.synchronize()  # 尝试去掉

        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)
        return loss_total

    def weighted_L1_loss(self, output, target):

        l1_loss = F.l1_loss(output, target, reduction='none')
        return torch.mean(l1_loss * self.lat_weight)

    def time_weighted_L1_loss(self, output, target, latent, input=None, aft_seq_length=2):
        if not latent:
            var_weight = self.var_weight
            std_tar = torch.std(target, dim=[0,1,3,4], keepdim=True)    #    .reshape(1, 1, target.shape[2], 1, 1)
            lat_weight = self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]]
            # var_tmp = [(target[:, i+1] - target[:, i])[:, None, ...] for i in range(aft_seq_length-1)]
            # var_tmp.insert(0, (target[:, 0][:, None, ...] - input))
            # # print((target[:, 0][:, None, ...] - input).shape)
            # # print(np.array(var_tmp.insert(0, (target[:, 0][:, None, ...] - input))).shape)
            # a = torch.tensor(torch.cat(var_tmp, 1))
            # time_diff_std = torch.std(a, dim=[0,2,3,4], keepdim=True)
        else:
            var_weight = torch.ones(1).to(self.device)
            std_tar = torch.ones(1).to(self.device)
            lat_weight = torch.ones(1).to(self.device)
            # time_diff_std = torch.ones(1).to(self.device)

        # tmp_weight = self.lat_weight if latent is False else self.lat_weight_60
        if latent:
            l1_loss = F.mse_loss(output, target, reduction='none')
            return torch.mean(l1_loss * lat_weight * self.time_weight[:, :aft_seq_length] * var_weight / (std_tar))
        else:
            l1_loss = F.l1_loss(output, target, reduction='none')
            return torch.mean(l1_loss * lat_weight * self.time_weight[:, :aft_seq_length] * var_weight / (std_tar))

    def weighted_L2_loss(self, output, target):
        l2_loss = F.mse_loss(output, target, reduction='none')
        return torch.mean(l2_loss * self.lat_weight)

    def ACC(self, pred, true):
        pred = (pred - torch.mean(pred, [0,1,3,4], keepdim=True)) / torch.std(pred, [0,1,3,4], keepdim=True)
        true = (true - torch.mean(true, [0, 1, 3, 4], keepdim=True)) / torch.std(true, [0, 1, 3, 4], keepdim=True)
        value = torch.sum(pred * true * self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]], dim=(0, -1, -2)) / torch.sqrt(
            torch.sum((pred ** 2) * self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]],
                      dim=(0, -1, -2)) * torch.sum((true ** 2) * self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]], dim=(0, -1, -2)))
        return value  # time len * ch_num

    def ACC2(self, pred, true):
        pred = (pred - self.mean_std_climate) #/ torch.std(pred, [0,1,3,4], keepdim=True)
        true = (true - self.mean_std_climate) #/ torch.std(true, [0, 1, 3, 4], keepdim=True)
        value = torch.mean(pred * true * (self.lat_weight_tar ** 2), dim=(0, -1, -2)) / torch.sqrt(
            torch.mean(((pred*self.lat_weight_tar) ** 2) ,
                      dim=(0, -1, -2)) * torch.mean(((true*self.lat_weight_tar) ** 2) ,dim=(0, -1, -2)))
        return value  # time len * ch_num

    def RMSE(self, pred, true, weight=None, spatial_norm=False):
        mse = (pred - true) ** 2
        # 使用权重进行加权
        weighted_mse = torch.mean(mse * self.lat_weight[:, :, :, self.args.tar_dim[0][0]:self.args.tar_dim[0][1]], dim=[0, -1, -2])
        # 计算 RMSE
        return torch.sqrt(weighted_mse)  # time len * ch_num

    def test(self, mode='val'):
        state_dict = torch.load(
            os.path.join(self.cp_dir, 'weight_30.pth')) #
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
        rmse_res_ours_path = os.path.join(self.results_dir, 'pred_results', self.args.ex_name , 'rmse_res_ours.npy') #
        acc_res_ours_path = os.path.join(self.results_dir, 'pred_results', self.args.ex_name , 'acc_res_ours.npy')
        # pred_path = os.path.join(self.results_dir, 'pred_results', self.args.ex_name , 'pred_res_ours.npy')
        # fourcastnet_path = "/data02/lisl/results_cv/results_1440_721_cn/pred_results/pred_res_afnonet.npy"
        # obs_path = os.path.join(self.results_dir, 'pred_results/obs_res.npy')
        tmp_base_dir = '/data/lisl/results/results_1440_721_cn/pred_results/'
        pred_path = os.path.join(tmp_base_dir, 'pred_res_ours.npy')
        fourcastnet_path = os.path.join(tmp_base_dir, "pred_res_afnonet.npy")
        obs_path = os.path.join(tmp_base_dir, 'obs_res.npy')

        pred_res = np.empty([0, forcast_len, len(self.args.target_dim), self.args.tar_size, self.args.tar_size])
        obs_res = np.empty([0, forcast_len, len(self.args.target_dim), self.args.tar_size, self.args.tar_size])
        rmse_res = np.empty([0, forcast_len, len(self.args.target_dim)])
        acc_res = np.empty([0, forcast_len, len(self.args.target_dim)])
        mean = []
        std = []
        time_list = []
        clim_mean_list = []
        # fuxi_path = '/data02/lisl/forcast/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr'
        # dataset = xr.open_zarr(fuxi_path, chunks=None)
        # step_len = (dataset['10m_wind_speed'].data).shape[0]
        # forecast_path = "/data02/lisl/forcast/ours/2020-240x121_equiangular_conservative.zarr"
        print(len(dataloader))
        with torch.no_grad():
            for step, (images, images_surface, time_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)

                images_surface = torch.nan_to_num(images_surface).type(torch.float32)
                images = images.flatten(2, 3)
                images = torch.cat([images, images_surface], 2)
                # print(self.mean_std)
                # print(images.shape)
                images = (images.type(torch.float32).to(self.device, non_blocking=True) - self.mean_std[0][None, None, ..., None, None]) / self.mean_std[1][
                    None, None, ..., None, None]

                if self.args.compute_climate_mean:
                    if step % 2 == 0:
                        clim_mean_list.append(torch.mean(images[:, :24], dim=[0, 3, 4], keepdim=False)[None, ...])
                        if step == len(dataloader) - 1:  # drop last = False
                            mean_all = torch.mean(torch.cat(clim_mean_list, 0), 0, keepdim=True)
                            np.save(os.path.join(self.results_dir, 'mean_std_climate.npy'), mean_all.cpu().numpy())
                            exit()

                bs_idx = step * time_data.shape[0]
                inputs = images[:, :self.args.in_len_val, ...].clone()#.type(torch.float32).to(self.device, non_blocking=True)#.clone()
                labels = images[:, self.args.in_len_val:, ...].clone()#.type(torch.float32).to(self.device, non_blocking=True)#.clone()
                time_data = time_data.type(torch.float32).to(self.device, non_blocking=True)

                if self.args.half_precision:
                    inputs = inputs.half()  # .half()
                    labels = labels.half()  # .half()
                time0 = time.time()
                pred, _, _, _, _, _, _ = self.model(inputs, self.const_data, time_data, labels,
                                                    aft_seq_length=forcast_len,
                                                    shrink=self.args.shrink, mode=mode)
                time_list.append(time.time() - time0)
                if mode == 'test':  # and (step % 2) == 0 *************************************
                    pred_res = np.concatenate([pred_res, pred.cpu().numpy()], 0)  # = np.concatenate([pred_res, pred.cpu().numpy()], 0)
                    obs_res = np.concatenate([obs_res, labels[:, :, self.args.target_dim, self.args.tar_dim[0][0]:self.args.tar_dim[0][1],
                     self.args.tar_dim[1][0]:self.args.tar_dim[1][1]].cpu().numpy()],
                                              0)  # = np.concatenate([pred_res, pred.cpu().numpy()], 0)

                label_last = labels[:, :, self.args.target_dim, self.args.tar_dim[0][0]:self.args.tar_dim[0][1],
                     self.args.tar_dim[1][0]:self.args.tar_dim[1][1]]

                rmse_res = np.concatenate([rmse_res, self.RMSE(self.trans_mean_std(pred),
                                                          self.trans_mean_std(labels[:, :, self.args.target_dim, self.args.tar_dim[0][0]:self.args.tar_dim[0][1],
                     self.args.tar_dim[1][0]:self.args.tar_dim[1][1]])).cpu().numpy()[None, ...]], 0)
                acc_res = np.concatenate([acc_res, self.ACC2(pred,
                                                          labels[:, :, self.args.target_dim, self.args.tar_dim[0][0]:self.args.tar_dim[0][1],
                     self.args.tar_dim[1][0]:self.args.tar_dim[1][1]]).cpu().numpy()[None, ...]], 0)

                if self.args.empty_cache:
                    torch.cuda.empty_cache()
        print(f'In mode {mode}, inference time per 60 frames is {np.mean(np.array(time_list))}')
        # exit()
        rmse_res = np.mean(rmse_res, 0)  # lead * ch_num
        acc_res = np.mean(acc_res, 0)    # lead * ch_num

        np.save(rmse_res_ours_path, rmse_res)
        np.save(acc_res_ours_path, acc_res)
        np.save(pred_path, pred_res) #

        rmse_res = np.load(rmse_res_ours_path)
        acc_res = np.load(acc_res_ours_path)

        save_quli_map = False
        if save_quli_map:
            # pred_res = np.load(pred_path)
            # pred_res_fct = np.load(fourcastnet_path)
            # obs_res = np.load(obs_path)
            # np.save(os.path.join(tmp_base_dir, 'pred_res_ours_100.npy'), pred_res[:100])
            # np.save(os.path.join(tmp_base_dir, 'pred_res_afnonet_100.npy'), pred_res_fct[:100])
            # np.save(os.path.join(tmp_base_dir, 'obs_res_100.npy'), obs_res[:100])
            pred_res = np.load(os.path.join(tmp_base_dir, 'pred_res_ours_100.npy'))
            pred_res_fct = np.load(os.path.join(tmp_base_dir, 'pred_res_afnonet_100.npy'))
            obs_res = np.load(os.path.join(tmp_base_dir, 'obs_res_100.npy'))
        else:
            pred_res, pred_res_fct, obs_res = None, None, None
        pred_metric_ours = [rmse_res, acc_res]  # lead * ch_num
        # results_fuxi = [rmse_res_list, acc_res_list]  # lead * ch_num

        base_dir = "results_cv/results_1440_721_cn/pred_results/"
        method_name_code = ['simvp', 'simvp_HTA', 'afnonet', 'afnonet_HTA', 'predrnnv2', 'TLS_MWP', 'stormer']
        rmse_res = [np.load(os.path.join(base_dir, "rmse_res_"+i+".npy")) for i in method_name_code]
        acc_res = [np.load(os.path.join(base_dir, "acc_res_"+i+".npy")) for i in method_name_code]

        pred_metric_other = [[i,j] for i,j in zip(rmse_res, acc_res)]
        # print(pred_metric_other[0])
        # exit()

        results_list = [pred_metric_ours]+pred_metric_other  # pred_metric_stormer  results_fuxi, results_graphcast, results_era5_forcast,
        method_name = ['Ours', 'SimVP', 'SimVP_HTA', 'FourCastNet', 'FourCastNet_HTA', 'PredRNNv2', 'TLS_MWP', 'Stormer']  # 'Fuxi'

        pred_res_fourcastnet = None  # np.load(fourcastnet_path)
        self.plot_rmse_acc(results_list, method_name, pred_res=pred_res, pred_res_fct=pred_res_fct, obs_res=obs_res, save_quli_map=save_quli_map)  # pred_res_fourcastnet, obs_res,

        exit()

        print(f'In mode {mode}, inference time per 60 frames is {np.mean(np.array(time_list))}')
        eval_res_list = torch.stack(eval_res_list, 0)
        eval_res_last = torch.mean(eval_res_list, 0)

        np.set_printoptions(precision=5, suppress=True)
        sum_num = np.array([round(i, 5) for i in eval_res_last.numpy()])

        return sum_num, pred_res

    def plot_rmse_acc(self, results_list,  method_name, pred_res=None, pred_res_fct=None, obs_res=None, save_quli_map=False): # pred_res_fourcastnet, obs_res,
        metrics_list = ['rmse', 'acc']  # , 'mae', 'mse'
        color_codes = [
            "red",  # 红色
            "green",  # 绿色
            "blue",  # 蓝色
            # "gray",  # 黄色
            "purple",  # 紫色
            "cyan",  # 青色
            # "orange",  # 橙色
            "olive",  # 橄榄色
            "brown",
            "violet"]  # 棕色

        # results_simvp, results_simvp_HTA, results_afnonet, results_afnonet_HTA, results_predrnnv2]

        show_len = self.args.aft_seq_length_test
        # 'U10m', 'V10m', 'T2m'
        pred_Z500_show = True
        pred_U500_show = False
        pred_t2m_show = False
        pred_Z50 = False
        pred_U1000 = False if pred_Z50 else True
        if pred_Z500_show:
            var_name = ['geopotential', 'geopotential', 'temperature', 'temperature']
            var_name_abb = ['Z500', 'Z850', 'T500', 'T850']
            temp_level = [500, 850]
            index_var = [1, 2, 4, 5]
            pred_name = 'Z500_show'
        elif pred_U500_show:
            var_name = ['u_component_of_wind', 'u_component_of_wind', 'v_component_of_wind', 'v_component_of_wind']
            var_name_abb = ['U500', 'U850', 'V500', 'V850']
            temp_level = [500, 850]
            index_var = [6, 7, 9, 10]
            pred_name = 'U500_show'
        elif pred_t2m_show:
            var_name = ['2m_temperature', 'mean_sea_level_pressure', '100m_u_component_of_wind', '100m_v_component_of_wind']
            var_name_abb = ['T2m', 'mslp', 'U100m', 'V100m']
            temp_level = [500, 850]
            index_var = [12, 13, 15, 16]
            pred_name = 't2m_show'
        elif pred_Z50:
            var_name = ['geopotential', 'geopotential', 'geopotential', 'geopotential', 'temperature', 'temperature',
                        'u_component_of_wind', 'u_component_of_wind']
            var_name_abb = ['Z50', 'Z500', 'Z850', 'Z1000', 'T500', 'T850', 'U500', 'U850']
            temp_level = [500, 850]
            index_var = [0, 1, 2, 3, 4, 5, 6, 7]
            pred_name = 'Z50'
        elif pred_U1000:
            var_name = ['u_component_of_wind', 'v_component_of_wind', 'v_component_of_wind', 'v_component_of_wind',
                        'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']
            var_name_abb = ['U1000', 'V500', 'V850', 'V1000', 'T2m', 'mslp', 'U100m', 'V100m']
            temp_level = [500, 850, 500, 850]
            index_var = [8, 9, 10, 11, 12, 13, 15, 16]
            pred_name = 'U1000'
            results_list.pop(-1)
            method_name.pop(-1)

        fig, axes = plt.subplots(nrows=2, ncols=len(var_name), figsize=(5 * len(var_name), 5*(len(var_name)//3)))

        for idx_i, i in enumerate(metrics_list):
            for idx, j in enumerate(var_name):

                a_res = []
                for ii, results in enumerate(results_list):
                    if method_name[ii] == 'Stormer' and j == 'u_component_of_wind' and i == 'rmse' and pred_name == 'Z50':
                        pass
                    else:
                        a_res.append(np.array(results[idx_i][:, index_var[idx]]).squeeze())

                line = []
                color_idx = 0
                for results, name in list(zip(a_res, method_name)):
                    if name == 'GraphCast':
                        tmp_show_len = 40
                        line1, = axes[idx_i, idx].plot(np.arange(1, tmp_show_len + 1), results[:tmp_show_len], label=name)
                    elif name == 'era5_forecasts':
                        tmp_show_len = 31
                        line1, = axes[idx_i, idx].plot(np.arange(1, tmp_show_len + 1), results[:tmp_show_len], label=name)
                    elif name == 'Ours':
                        line1, = axes[idx_i, idx].plot(np.arange(1, show_len + 1), results[:show_len], label=name, linewidth=2, color=color_codes[color_idx])
                    elif name == 'Stormer':
                        if j == 'u_component_of_wind' and i == 'rmse' and pred_name == 'Z50':
                            pass
                        else:
                            line1, = axes[idx_i, idx].plot(np.arange(1, show_len + 1), results[:show_len], label=name, color=color_codes[color_idx])
                    else:
                        if pred_name == 'Z50':
                            line1, = axes[idx_i, idx].plot(np.arange(1, show_len + 1), results[:show_len], label=name, color=color_codes[color_idx])
                        elif pred_name == 'U1000':
                            line1, = axes[idx_i, idx].plot(np.arange(1, show_len + 1), results[:show_len], label=name, color=color_codes[color_idx])
                    line.append(line1)
                    color_idx += 1

                if i == 'rmse':
                    # print(find_min_value(a_res), find_max_value(a_res))
                    if pred_name == 'Z50':
                        axes[idx_i, idx].set_ylim(find_min_value(a_res), find_max_value(a_res))
                    elif pred_name == 'U1000':
                        axes[idx_i, idx].set_ylim(find_min_value(a_res), find_max_value(a_res))
                    if j == 'geopotential':
                        axes[idx_i, idx].set_ylabel(r'$m^{2} s^{-2}$')
                    elif j == 'temperature' or j == '2m_temperature':
                        axes[idx_i, idx].set_ylabel('K')
                    elif j == '10m_wind_speed' or j == '10m_u_component_of_wind' or j == '10m_v_component_of_wind' \
                            or j == 'u_component_of_wind' or j == 'v_component_of_wind':
                        axes[idx_i, idx].set_ylabel(r'$m s^{-1}$')
                    elif j == 'mean_sea_level_pressure':
                        axes[idx_i, idx].set_ylabel(r'$pa$')
                    elif j == 'mean_sea_level_pressure':
                        axes[idx_i, idx].set_ylabel(r'$pa$')
                    elif j == 'relative_humidity':
                        axes[idx_i, idx].set_ylabel(r'$\%$')
                    axes[idx_i, idx].set_xlabel('Forecast time (hours)', loc='center', fontsize=14)
                elif i == 'acc':
                    axes[idx_i, idx].set_ylim(0, 1.1)
                    axes[idx_i, idx].set_xlabel('Forecast time (hours)', loc='center', fontsize=14)

                axes[idx_i, idx].set_title(var_name_abb[idx], loc='center', fontsize=28)

        fig.text(0.48, 0.98, 'rmse'.upper(), fontsize=18, fontweight="bold")
        # elif idx_i == 1:
        fig.text(0.48, 0.515, 'acc'.upper(), fontsize=18, fontweight="bold")
        fig.legend(line, method_name, loc='lower center', fontsize=24, bbox_to_anchor=(0.5, 0.00), shadow=True, ncol=9)  # 调整字体大小
        plt.subplots_adjust(left = 0.03, right = 0.98, top = 0.96, bottom = 0.15, wspace = 0.15, hspace = 0.25)  # left=0.03, right=0.98, top=0.96, bottom=0.1, wspace=0.15

        save_path = os.path.join(self.results_dir, 'quanti_figures', self.args.ex_name, 'metric_'+ pred_name + '.png')
        plt.savefig(save_path, dpi=100)  # 300
        plt.close()

    def trans_mean_std(self, res):
        # if isinstance(res, torch.Tensor):
        #     res = res.cpu().numpy()
        mean = self.mean_std[0][None, None, self.args.target_dim, None, None]#.cpu().numpy()
        std = self.mean_std[1][None, None, self.args.target_dim, None, None]#.cpu().numpy()

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