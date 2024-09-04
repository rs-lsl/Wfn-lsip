import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import partial
from itertools import repeat
from typing import Callable
import xarray as xr
import matplotlib.pyplot as plt
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
import torch.utils.data
import numpy as np


def compare_diff_resolution():
    fuxi_240 = xr.open_zarr('/data02/lisl/forcast/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr/')
    z500_240 = np.array(fuxi_240['geopotential'].data)[0, :, 0]
    print(z500_240.shape)

    fuxi_64 = xr.open_zarr('/data01/lisl/forcast/fuxi/2020-64x32_equiangular_conservative.zarr/')
    z500_64 = np.array(fuxi_64['geopotential'].data)[0, :, 0]
    print(z500_64.shape)

# 读图像文件
def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    # im_width =   # 栅格矩阵的列数
    # im_height =   # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    return dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    # return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def npy2tif(filename=''):
    ms_ratio = 3
    filename = "/data02/lisl/HSI/ms_last.npy"
    a = np.load(filename)[:, 2*ms_ratio:, :-6*ms_ratio]
    np.save("/data02/lisl/HSI/MSI_tmp.npy", a)
    print(f'msi_shape:{a.shape}')

    filename = "/data02/lisl/HSI/HS_ziyuan.npy"
    a = np.load(filename)[:, :-1, 1:]
    np.save("/data02/lisl/HSI/HSI_tmp.npy", a)
    print(f'hsi_shape:{a.shape}')

    filename = "/data02/lisl/HSI/pan_ziyuan.npy"
    a = np.load(filename)[:-12, 12:]
    np.save("/data02/lisl/HSI/PAN_tmp.npy", a)
    print(f'pan_shape:{a.shape}')


    # write_img_gdal("/data02/lisl/HSI/pan_ziyuan.tif", a)

def tp_hist():
    # dataset_dir1 = '/data02/lisl/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr/'
    dataset_dir2 = '/data02/lisl/era5/1959-2022-6h-64x32_equiangular_conservative.zarr/'
    a = xr.open_zarr(dataset_dir2)
    a = a.sel(time=slice('2000-01-01', '2020-12-31'))
    # a = dataset0.sel(time=slice('1959-01-01', '2022-12-31'))
    wind10m = np.array(a['total_precipitation_6hr']).flatten()
    print(wind10m.shape)
    plt.hist(wind10m, bins=100)
    # plt.xticks(np.linspace(0, 0.01, 20))
    plt.xlabel('total precipitation (m/6hr)')
    plt.savefig('/data02/lisl/results/results_64_32/tp_hist.png', dpi=100)
    plt.close()
    eps = 1e-3 #1e-7 优于 1e-3
    wind10m_log = np.log(wind10m + eps) - np.log(eps)
    plt.hist(wind10m_log, bins=100)
    plt.xlabel('total precipitation')
    plt.savefig('/data02/lisl/results/results_64_32/tp_hist_log.png', dpi=100)
    plt.close()

import os
def crop_lat_lon():
    fuxi_path = '/data01/lisl/forcast/fuxi/2020-64x32_equiangular_conservative.zarr'
    dataset = xr.open_zarr(fuxi_path, chunks=None)
    print(dataset['10m_wind_speed'].data.shape)
    exit()
    maxlat, minlat, minlon, maxlon = 57, 15, 72, 143
    dataset_dir2 = '/data02/lisl/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr/'  #  经度上0指数组index为0，维度上0指代数组index60
    dataset0 = xr.open_zarr(dataset_dir2)
    a = dataset0.sel(time=slice('2020-01-01', '2020-01-01'))
    b = a.sel(latitude = slice(minlat, maxlat),
    longitude = slice(minlon, maxlon))
    aa = np.array(a['10m_u_component_of_wind'])
    bb = np.array(b['10m_u_component_of_wind'])
    print(bb.shape)


def per_year_data(dataset_dir):
    dataset_dir = '/data01/lisl/forcast/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
    dataset0 = xr.open_zarr(dataset_dir)
    for year in range(1979, 2022):
        timestamps = np.arange('1959-01-01', '2022-01-01', dtype='datetime64[6h]')
        year_mask = (timestamps >= np.datetime64(f'{year}-01-01')) & (timestamps <= np.datetime64(f'{year}-12-31'))
        year_data = np.array(dataset0['total_precipitation_6hr'])[year_mask]
        print(np.isnan(year_data).any())

def read_img(filename):
    # 读取tif影像
    dataset = gdal.Open(filename)  # "H:/worldview2/result/origin_pan.tif"
    # print(dataset)
    im_cols = dataset.RasterXSize  # 栅格矩阵的列数
    im_rows = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount

    # band = dataset.GetRasterBand(1)
    im_data = np.float32(dataset.ReadAsArray(0, 0, im_cols, im_rows))  # 获取数据

    return im_data

def read_img_gdal(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    # del dataset
    return im_data


# 写文件，以写成tif为例
def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def fast_collate_for_prediction(batch):
    """ A fast collation function optimized for float32 images (np array or torch)
        and float32 targets (video prediction labels) in video prediction tasks"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into
        # one tensor ordered by position such that all tuple of position n will end up
        # in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.float32)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            # all input tensor tuples must be same length
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.zeros((batch_size, *batch[1][0].shape), dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x

# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(torch.utils.data.DataLoader):
#
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_input2, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_input2 = next_input2.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_input2 = next_input2.float()
                        next_target = next_target.float()

            if not first:
                yield input, input2, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            input2 = next_input2
            target = next_target

        yield input, input2, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=True,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=False,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                # sampler = torch.utils.data.BatchSampler(
                #     sampler, batch_size, drop_last=True
                # )
                print('setting the dis sampler')
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    # for images, labels in loader:
    #     print(images.shape)
    # print(loader)
    return loader, sampler


def reshape_patch(img_tensor, patch_size):
    assert 4 == img_tensor.ndim
    seq_length = np.shape(img_tensor)[0]
    img_height = np.shape(img_tensor)[1]
    img_width = np.shape(img_tensor)[2]
    num_channels = np.shape(img_tensor)[3]
    a = np.reshape(img_tensor, [seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 3, 2, 4, 5])
    patch_tensor = np.reshape(b, [seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor


def reshape_patch_back_tensor(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    patch_narray = patch_tensor.detach().cpu().numpy()
    batch_size = np.shape(patch_narray)[0]
    seq_length = np.shape(patch_narray)[1]
    patch_height = np.shape(patch_narray)[2]
    patch_width = np.shape(patch_narray)[3]
    channels = np.shape(patch_narray)[4]
    img_channels = channels // (patch_size * patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                     patch_height, patch_width,
                                     patch_size, patch_size,
                                     img_channels])
    b = a.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                   patch_height * patch_size,
                                   patch_width * patch_size,
                                   img_channels])
    return img_tensor.permute(0, 1, 4, 2, 3)

if __name__ == '__main__':
    # down_zarr_nc()
    test_down_fun()
    # compare_diff_resolution()
    # var_corr_cross()
    # a = list(np.arange(92,105))
    # b = '['+', '.join(a) + ']'
    # print(a)