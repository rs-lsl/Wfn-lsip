
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import mnist
from torch.autograd import Variable
from tqdm import tqdm
import sys
import tempfile
import time

import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from metrics import metric
from datetime import timedelta


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
        timeout=timedelta(seconds=7200000),
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

def _dist_forward_collect(data_loader, device, length=None, gather_data=False, rank=0, metric_list=['mae', 'mse', 'rmse', 'ssim']):
    """Forward and collect predictios in a distributed manner.

    Args:
        data_loader: dataloader of evaluation.
        length (int): Expected length of output arrays.
        gather_data (bool): Whether to gather raw predictions and inputs.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    # preparation
    results = []
    length = len(data_loader.dataset) if length is None else length
    if rank == 0:
        prog_bar = ProgressBar(len(data_loader))

    # loop
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        if idx == 0:
            part_size = batch_x.shape[0]
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_y = _predict(batch_x, batch_y)

        if gather_data:  # return raw datas
            results.append(dict(zip(['inputs', 'preds', 'trues'],
                                    [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
        else:  # return metrics
            eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                 data_loader.dataset.mean, data_loader.dataset.std,
                                 metrics=metric_list, spatial_norm=spatial_norm, return_log=False)
            eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
            for k in eval_res.keys():
                eval_res[k] = eval_res[k].reshape(1)
            results.append(eval_res)

        if self.args.empty_cache:
            torch.cuda.empty_cache()
        if self.rank == 0:
            prog_bar.update()

    # post gather tensors
    results_all = {}
    for k in results[0].keys():
        results_cat = np.concatenate([batch[k] for batch in results], axis=0)
        # gether tensors by GPU (it's no need to empty cache)
        results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
        results_strip = np.concatenate(results_gathered, axis=0)[:length]
        results_all[k] = results_strip
    return results_all

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 启动多GPU环境
    init_distributed_mode(args)
    # 拿到当前进程的rank，注意是当前进程，就像之前说的，每个进程都会执行该段代码，进程0的rank就是0，进程1的rank就是1
    rank = args.rank
    # 获得gpu
    device = torch.device(args.device)
    batch_size = args.batch_size
    # 这里学习率需要随着GPU数量成倍增长。具体可以参考B站视频讲解。不过也不一定，自己可以尝试调一调这个参数
    args.lr *= args.world_size
    # 我们只让进程0输出信息，进程1不执行这一部分。这样进程1就不会输出信息，避免重复输出。
    if rank == 0:
        print(args)
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=False)
    test_dataset = mnist.MNIST("./data", train=False, transform=transform, download=False)
    # 分布式过程
    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sample, batch_size, drop_last=True
    )
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_sampler=train_batch_sampler,pin_memory=True,num_workers=nw,
    )
    test_lodaer = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sample, pin_memory=True,num_workers=nw,
    )
    # 定义模型以及保证所有进程的模型参数一致
    model = CNNNet(1, 16, 32, 32, 128, 64, 10)
    model = model.to(device)
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weight.pt")
    if rank==0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = torch.nn.parallel.DistributedDataParallel(model, device[args.device])
    # pg = [p for p in model.parameters() if ]
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # 训练
    for epoch in range(args.epochs):
        # 因为存在随机行，每个进程对数据采样会不一样，这个就是保证进程之间采样信息是相通的一致的。
        train_sample.set_epoch(epoch)
        # 训练
        mean_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        #  评估
        sum_num = evaluate(
            model, test_lodaer, device
        )
        acc = sum_num / test_sample.total_size
        # 直在 进程0中打印信息。
        if rank == 0:
            print("[epoch {}] accuracy {}".format(epoch, acc))
            # 保存模型。由于所有的进程模型参数都一样，所以只需要选择其中一个进程保存就行。
            # 非常注意，我们对model进行DistributedDataParallel封装后，虽然model叫model，但本质上model == model.module
            # model(单卡) == model.module（多卡的model）。单卡获取模型的参数是model.state_dict()，多卡获取模型参数model.module.state_dict()
            # 需要加一个module。才能获取到真正的模型。
            torch.save(model.module.state_dict(), "weight.pth")

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


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