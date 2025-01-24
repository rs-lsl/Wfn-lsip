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
from model2023.model import SimVP_Model_x, Discriminator, AdversarialLoss
from model2023.utils.lat_weight import get_lat_weights

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

        if not args.test:
            # self.steps_per_epoch = len(dataloader_train)
            self.init_optim(optimizer)

        self.init_lat_weight()
        # self.init_model()
        # self.adv_loss = AdversarialLoss(discriminator, loss_type=loss_type)

    def init_lat_weight(self):
        a_weight = np.load(os.path.join(self.results_dir, 'lat_weight.npy'))
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
        # var_weight = torch.from_numpy(np.array([0.1, 0.1,0.1, 0.1,1.0, 0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1]+[1.0]*(104-13))).reshape([1, 1, -1, 1, 1])
        if self.args.pred_104:
            var_weight = torch.from_numpy(np.array([0.1]*3+[1]+[0.1]*8+[1]*(len(self.args.target_dim)-12))).reshape([1, 1, len(self.args.target_dim), 1, 1])
        self.var_weight = var_weight.type(torch.float32).to(self.device)
        print(self.var_weight.flatten())

        self.base_dir = 'base_dir'
        self.mean_std = torch.from_numpy(np.load("./little_files/mean_std_2000.npy")).type(torch.float32).to(self.device).reshape([2, 1, -1, 1, 1])

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

    def save_std_diff(self, mean_list1, mean_list2, mean_list4, std_list1, std_list2, std_list4):
        if self.args.compute_std_diff:
            std_last = []
            for (mean_list, std_list) in [(mean_list1, std_list1), (mean_list2, std_list2), (mean_list4, std_list4)]:
                mean_all = torch.mean(torch.cat(mean_list, 0), 0, keepdim=True)
                std_all = torch.sqrt((torch.sum((self.bs - 1) * (torch.cat(std_list, 0) ** 2), 0)
                                      + torch.sum(self.bs * ((torch.cat(mean_list, 0) - mean_all) ** 2), 0)) / (
                                                 (self.bs - 1) * len(mean_list)))
                std_last.append(std_all[None, ...])
            print(np.array(torch.cat(std_last, 0).cpu().numpy()).shape)
            np.save(os.path.join(self.results_dir, 'diff_std.npy'), np.array(torch.cat(std_last, 0).cpu().numpy()))

        # self.mean_std = torch.from_numpy(np.load(os.path.join(self.results_dir, 'mean_std.npy'))).type(
        #     torch.float32).to(self.device)

    def train_one_epoch(self, epoch, dataloader_train, aft_seq_length=2):
        self.model.train()

        loss_total = 0.0
        time0 = time.time()
        # train_pbar = tqdm(dataloader_train) if rank == 0 else dataloader_train
        for step, (images, time_data, rand_idx) in enumerate(dataloader_train):

            self.optimizer.zero_grad()

            inputs = (images[:, :self.args.input_time_length, ...]+ \
                     torch.randn(size=[self.bs, self.args.input_time_length, self.ch, *self.shape_val], dtype=torch.float32)/100.0).to(self.device, non_blocking=True)
            # inputs = inputs + ().to(self.device, non_blocking=True)
            labels = images[:, self.args.input_time_length:(self.args.input_time_length+aft_seq_length), ...].type(torch.float32).to(self.device, non_blocking=True)
            time_data = time_data.type(torch.float32).to(self.device, non_blocking=True)
            with autocast():  # 混合精度训练/半精度

                # time0 = time.time()

                pred, pred_latent, true_latent, label_pred, embed, embed_diff, embed_label_dec\
                    = self.model(inputs, self.const_data, time_data, labels,
                                                                        aft_seq_length=aft_seq_length, hid_i=rand_idx[0].int(),
                                                                        shrink=self.args.shrink, mode='train', device=self.device)  # 尝试更换latent的维度T

                temp_label = labels[:, :, self.args.target_dim]

                loss0 = self.time_weighted_L1_loss(pred, temp_label, latent=False, aft_seq_length=aft_seq_length)  # [:, :, self.target_dim]
                loss1 = self.time_weighted_L1_loss(pred_latent, true_latent, latent=True,
                                                         aft_seq_length=aft_seq_length)
                loss2 = self.weighted_L1_loss(label_pred, images[:, :self.args.input_time_length, self.args.target_dim].to(self.device))
                loss4 = self.weighted_L1_loss(labels[:, :, self.args.target_dim], embed_label_dec)
                loss = loss0 + loss1 + self.args.loss_weight_in_recon * loss2 + loss4 #+ args.alpha * diff_div_reg(pred, labels)   # 维度加权损失函数

            loss.backward()
            # clip_grads(self.model.parameters(), self.args, norm_type=2.0)   #  adjust
            self.optimizer.step()

            loss_total += loss.item()
            torch.cuda.synchronize()  # 尝试去掉

        # GPU之间同步，
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)
        return loss_total

    def weighted_L1_loss(self, output, target):

        l1_loss = F.l1_loss(output, target, reduction='none')
        return torch.mean(l1_loss * self.lat_weight * self.var_weight)

    def time_weighted_L1_loss(self, output, target, latent, input=None, aft_seq_length=2):
        if not latent:
            var_weight = self.var_weight
            std_tar = torch.std(target, dim=[0,1,3,4], keepdim=True)    #    .reshape(1, 1, target.shape[2], 1, 1)
        else:
            var_weight = torch.ones(1).to(self.device)
            std_tar = torch.ones(1).to(self.device)

        if latent:
            l1_loss = F.mse_loss(output, target, reduction='none')
            return torch.mean(l1_loss * self.lat_weight * self.time_weight[:, :aft_seq_length] * var_weight / (std_tar))
        else:
            l1_loss = F.l1_loss(output, target, reduction='none')
            return torch.mean(l1_loss * self.lat_weight * self.time_weight[:, :aft_seq_length] * var_weight / (std_tar))


    def weighted_L2_loss(self, output, target):
        l2_loss = F.mse_loss(output, target, reduction='none')
        return torch.mean(l2_loss * self.lat_weight)

    def ACC(self, pred, true):
        value = torch.mean(torch.sum(pred * true * self.lat_weight, dim=(-1, -2)) / torch.sqrt(
            torch.sum((pred ** 2) * self.lat_weight, dim=(-1, -2)) * torch.sum((true ** 2) * self.lat_weight, dim=(-1, -2))))
        return value

    def test(self, mode='val'):
        print(os.path.join(self.cp_dir, 'weight.pth'))
        state_dict = torch.load(
            os.path.join(self.cp_dir, 'weight.pth'))
        if self.args.dist:
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


        sum_num, pred_res = self.evaluate(
            metric_list=['mae', 'rmse'], mode=mode # the validation dataset
        )

        # if self.rank == 0:
        print('metrics:    ', '    mae   ', ' rmse ')  # , 'snr', 'lpips'
        print('Eval results:', sum_num)

        return pred_res

    def ACC2(self, pred, true):
        pred = (pred - self.mean_std_climate) #/ torch.std(pred, [0,1,3,4], keepdim=True)
        true = (true - self.mean_std_climate) #/ torch.std(true, [0, 1, 3, 4], keepdim=True)
        value = torch.mean(pred * true * (self.lat_weight ** 2), dim=(0, -1, -2)) / torch.sqrt(
            torch.mean(((pred*self.lat_weight) ** 2) ,
                      dim=(0, -1, -2)) * torch.mean(((true*self.lat_weight) ** 2) ,dim=(0, -1, -2)))
        return value  # time len * ch_num

    def RMSE(self, pred, true, weight=None, spatial_norm=False):
        mse = (pred - true) ** 2
        # 使用权重进行加权
        weighted_mse = torch.mean(mse * self.lat_weight, dim=[0, -1, -2])
        # 计算 RMSE
        return torch.sqrt(weighted_mse)  # time len * ch_num

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
        singular_value_list = []
        # pred_res = []
        pred_res = np.empty([0, 60, 20, 64, 32])
        mean = []
        std = []
        time_list = []

        with torch.no_grad():
            for step, (images, time_data) in enumerate(dataloader):
                if step % 1 == 0:
                    print(step)
                bs_idx = step * time_data.shape[0]
                inputs = images[:, :self.args.in_len_val, ...].clone().type(torch.float32).to(self.device, non_blocking=True)#.clone()
                labels = images[:, self.args.in_len_val:, ...].clone().type(torch.float32).to(self.device, non_blocking=True)#.clone()
                time_data = time_data.type(torch.float32).to(self.device, non_blocking=True)


                if self.args.half_precision:
                    inputs = inputs.half()  # .half()
                    labels = labels.half()  # .half()
                time0 = time.time()
                pred, _, _, _, _, _, _ = self.model(inputs, self.const_data, time_data, labels,
                                                    aft_seq_length=forcast_len,
                                                    shrink=self.args.shrink, mode=mode)

                time_list.append((time.time() - time0))

                if mode == 'test':  # and (step % 2) == 0 *************************************
                    pred_res = np.concatenate([pred_res, pred.cpu().numpy()], 0)  # = np.concatenate([pred_res, pred.cpu().numpy()], 0)

                eval_res = metric(self.trans_mean_std(pred),
                                  self.trans_mean_std(labels[:, :, self.args.target_dim]),
                                  weight=self.lat_weight.cpu().numpy())

                eval_res_list.append(torch.tensor(list(eval_res.values())))
                if self.args.empty_cache:
                    torch.cuda.empty_cache()

        print(f'In mode {mode}, inference time per 60 frames is {np.mean(np.array(time_list))}')
        # exit()
        eval_res_list = torch.stack(eval_res_list, 0)
        eval_res_last = torch.mean(eval_res_list, 0)

        np.set_printoptions(precision=5, suppress=True)
        sum_num = np.array([round(i, 5) for i in eval_res_last.numpy()])

        return sum_num, pred_res

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