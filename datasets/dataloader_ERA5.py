import io
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import time
import gc
# import h5py
import pickle
from utils.utils0 import get_days_in_year, create_folder_if_not_exists, sort_by_last_digit
from datasets.utils_dataset import create_loader
from datasets.utils_dataset import read_img, write_img_gdal
import torchvision.transforms as trans

# level, array([  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,  1000]
Coord_name = ['latitude', 'longitude', 'time']
var_name = ['total_precipitation_6hr', '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', '2m_temperature',
            'angle_of_sub_gridscale_orography',
            'anisotropy_of_sub_gridscale_orography', 'geopotential_at_surface', 'high_vegetation_cover', 'lake_cover',
            'lake_depth', 'land_sea_mask', 'low_vegetation_cover', 'mean_sea_level_pressure', 'sea_ice_cover', 'sea_surface_temperature',
            'slope_of_sub_gridscale_orography', 'soil_type', 'standard_deviation_of_filtered_subgrid_orography',
            'standard_deviation_of_orography', 'surface_pressure', 'toa_incident_solar_radiation',
            'toa_incident_solar_radiation_12hr', 'toa_incident_solar_radiation_24hr', 'toa_incident_solar_radiation_6hr',
            'total_cloud_cover', 'total_column_water_vapour', 'total_precipitation_12hr', 'total_precipitation_24hr',
            'type_of_high_vegetation', 'type_of_low_vegetation', 'geopotential', 'temperature', 'specific_humidity',
            'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'wind_speed']
            #
ignore_name = ['sea_ice_cover', 'sea_surface_temperature', 'toa_incident_solar_radiation_24hr', 'total_precipitation_24hr']
idx_dim = np.array([[0, 13], [5, 10], [-5, 13], [-5, 13], [-5, 13], [-5, 13], [-5, 13], [-5, 13]])  # 8*2
train_year = [2000, 2018]
val_year = [2018, 2020]
test_year = [2020, 2021]

def list_files(directory):
    list_file = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_file.append(os.path.join(root, file))

    return list_file

def list_subdirectories(main_folder):
    subdirectories = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    return subdirectories

def norm_minmax(dataset, dataset_const, dataset_time, eps=1e-9):
    num_ch = dataset.shape[1]
    min = np.min(np.transpose(dataset, [1,0,2,3]).reshape([num_ch, -1]), 1)#[:coor_dims]
    max = np.max(np.transpose(dataset, [1,0,2,3]).reshape([num_ch, -1]), 1)#[:coor_dims]
    min = np.reshape(min, [1, num_ch, 1, 1])
    max = np.reshape(max, [1, num_ch, 1, 1])
    dataset = (dataset - min) / (max - min)  # delete eps

    num_ch_const = dataset_const.shape[0]
    min = np.min(dataset_const.reshape([num_ch_const, -1]), 1)#[:coor_dims]
    max = np.max(dataset_const.reshape([num_ch_const, -1]), 1)#[:coor_dims]
    min = np.reshape(min, [num_ch_const, 1, 1])
    max = np.reshape(max, [num_ch_const, 1, 1])
    dataset_const = (dataset_const - min) / (max - min)  # delete eps

    min = np.min(dataset_time, 1, keepdims=True)#[:coor_dims]
    max = np.max(dataset_time, 1, keepdims=True)#[:coor_dims]
    dataset_time = (dataset_time - min) / (max - min)  # delete eps

    return dataset, dataset_const, dataset_time

def norm_meanstd(dataset, dataset_const, dataset_time, eps=1e-9):
    num_ch = dataset.shape[1]
    dataset0 = np.transpose(dataset, [1,0,2,3]).reshape([num_ch, -1])
    mean = np.mean(dataset0, 1)#[:coor_dims]
    mean = np.reshape(mean, [1, num_ch, 1, 1])
    std = np.std(dataset0, 1)
    std = np.reshape(std, [1, num_ch, 1, 1])
    print('mean_std of original data:')
    print(mean.shape)
    print(std.shape)
    dataset = (dataset - mean) / std

    num_ch_const = dataset_const.shape[0]
    mean = np.mean(dataset_const.reshape([num_ch_const, -1]), 1)#[:coor_dims]
    mean = np.reshape(mean, [num_ch_const, 1, 1])
    std = np.std(dataset_const.reshape([num_ch_const, -1]), 1)
    std = np.reshape(std, [num_ch_const, 1, 1])
    print('mean_std of const data:')
    print(mean.shape)
    print(std.shape)
    dataset_const = (dataset_const - mean) / std

    mean = np.mean(dataset_time, 1, keepdims=True)#[:coor_dims]
    std = np.std(dataset_time, 1, keepdims=True)#[:coor_dims]
    print('mean_std of time data:')
    print(mean.shape)
    print(std.shape)
    dataset_time = (dataset_time - mean) / std

    return dataset, dataset_const, dataset_time

def norm_meanstd2(dataset, mean_std, p_dim):
    mean = np.reshape(mean_std[0, :], [1, -1, 1, 1])
    std = np.reshape(mean_std[1, :], [1, -1, 1, 1])

    dataset[:, p_dim:] = (dataset[:, p_dim:] - mean) / std

    return dataset#, dataset_const, dataset_time

class ERA5_dataset_gdal_len1(Dataset):
    def __init__(self, main_path, seg_len, lon_len=64, lat_len=32, bs=32, list_len=8, is_training=False, ch_num_13=4,
                 idx_dim=None, input_time_length=1, in_len_val=5, mode='train'):
        super(ERA5_dataset_gdal_len1, self).__init__()
        self.len_diff = in_len_val - input_time_length
        self.in_len_val = in_len_val
        if mode == 'train':
            print(f"train dataset len is {len(list(range(0, get_days_in_year(*train_year) * 4-seg_len+1, 1)))}")
        elif mode == 'val':
            print(f"val dataset len is {len(list(range(0, get_days_in_year(*val_year) * 4 + self.len_diff -seg_len+1, 1)))}")
        if mode == 'test':
            print(f"test dataset len is {len(list(range(0, get_days_in_year(*test_year) * 4 + self.len_diff -seg_len+1, 1)))}")
        # print(get_days_in_year(*train_year) * 4)
        # print(4 * get_days_in_year(train_year[0], val_year[1]))
        if mode == 'val':
            file_list_num = np.arange(4 * get_days_in_year(train_year[0], train_year[1])-in_len_val, 4 * get_days_in_year(train_year[0], val_year[1]))
            self.file_list = [os.path.join(main_path, str(i)) for i in file_list_num]
        elif mode == 'test':
            file_list_num = np.arange(4 * get_days_in_year(train_year[0], val_year[1])-in_len_val, 4 * get_days_in_year(train_year[0], test_year[1]))
            self.file_list = [os.path.join(main_path, str(i)) for i in file_list_num]

        self.file_list = sorted(self.file_list, key=sort_by_last_digit)
        self.bs, self.seg_len, self.lon_len, self.lat_len = bs, seg_len, lon_len, lat_len
        self.ch_num = 48
        self.mode = mode

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.mode == 'test':
            idx = idx * 2
        # print(idx)
        a = np.array([read_img(os.path.join(self.file_list[i], 'var_data.tif')) for i in range(idx, idx+self.seg_len)])
        b = np.array([read_img(os.path.join(self.file_list[i], 'time_data.tif')) for i in range(idx, idx+self.seg_len)])
        return a, np.transpose(b.squeeze(), axes=[1,0])

    def __len__(self):
        if self.mode != 'test':
            return len(self.file_list) - self.seg_len + 1
        else:
            return len(list(range(0, get_days_in_year(*test_year)*4+self.in_len_val-self.seg_len+1, 2)))

class ERA5_dataset_gdal_4model(Dataset):
    def __init__(self, main_path, seg_len, lon_len=64, lat_len=32, bs=32, time_inte=None, list_len=8, is_training=False, ch_num_13=4, idx_dim=None):
        super(ERA5_dataset_gdal_4model, self).__init__()
        print(f"dataset len is {len(list(range(0, get_days_in_year(*train_year) * 4-seg_len+1, 1)))}")
        # self.file_list = list_subdirectories(main_path)[:get_days_in_year(*train_year)*4]  # 26296<-->5
        file_list_num = np.arange(get_days_in_year(*train_year)*4)
        self.file_list = [os.path.join(main_path, str(i)) for i in file_list_num]
        self.bs, self.seg_len, self.lon_len, self.lat_len = bs, seg_len, lon_len, lat_len
        self.ch_num = 48
        self.time_inte = time_inte
        self.idx = 0
        # print(self.file_list[:10])

    def __getitem__(self, idx):
        rand_idx = (self.idx // self.bs) % len(self.time_inte)
        # rand_idx = np.random.randint(0, len(self.time_inte))
        rand_inte = self.time_inte[rand_idx]
        a = np.array([read_img(os.path.join(self.file_list[i], 'var_data.tif')) for i in range(idx, idx+self.seg_len*rand_inte, rand_inte)])

        b = np.array([read_img(os.path.join(self.file_list[i], 'time_data.tif')) for i in range(idx, idx+self.seg_len*rand_inte, rand_inte)])

        self.idx = self.idx + 1
        return a, np.transpose(b.squeeze(), axes=[1,0]), rand_idx

    def __len__(self):
        # return 200
        return len(self.file_list) - self.seg_len*max(self.time_inte) + 1

def generate_days():
    start = 1
    step = 3
    # 用于存储生成的字符串
    string_list = []
    # 循环生成字符串
    for i in range(start, 31, step):  # 假设我们生成的数字不超过99
        # 计算结束数字
        end = i + step - 1
        # 将数字格式化为两位数的字符串
        start_str = f"{i:02d}"
        end_str = f"{end:02d}"
        # 将起始和结束数字合并为一个字符串，并添加到列表中
        string_list.append(f"{start_str}_{end_str}")
    # print(string_list)
    string_list.append('31_31')
    return string_list

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_input2, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_input2 = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input2 = self.next_input2.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        input2 = self.next_input2
        target = self.next_target
        self.preload()
        return input, input2, target

def load_ERA5_dataset_per(batch_size, val_batch_size, test_batch_size, lon_len, lat_len, save_dir='', num_workers=4,
                           in_shape=[10, 1, 64, 64], distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
                           test=False, args=None):
    image_size = in_shape[-1] if in_shape is not None else 64
    ch_num_13 = 8
    seg_len = args.input_time_length + args.aft_seq_length_train
    if not test:
        time0 = time.time()

        train_set = []
        for i in range(len(args.pred_len)):
            train_set.append(ERA5_dataset_gdal_4model(os.path.join(save_dir, 'all_gdal_len1'),
                                      seg_len=args.input_time_length+args.pred_len[i], lon_len=64, lat_len=32,
                                             bs=args.batch_size, time_inte=args.time_inte))  # **********************  overlap_step)

        # del result_var, time_var
        # gc.collect()
        print(time.time() - time0)

        dataloader_train = []
        sampler_train = []
        for i in range(len(args.pred_len)):
            dataloader_train_tmp, sampler_train_tmp = create_loader(train_set[i],
                                                            batch_size=batch_size,
                                                            shuffle=True, is_training=True,
                                                            pin_memory=args.pin_memory, drop_last=True,
                                                            num_workers=num_workers,
                                                            distributed=distributed, use_prefetcher=use_prefetcher, return_num=3)
            dataloader_train.append(dataloader_train_tmp)
            sampler_train.append(sampler_train_tmp)

    elif test:
        # var_data_train, var_data_val, time_diff_emb_train, time_diff_emb_val = None, None, None, None
        dataloader_train, sampler_train, dataloader_vali = None, None, None
    file_list_train = os.path.join(save_dir, 'all_gdal_len1')
    val_set = ERA5_dataset_gdal_len1(file_list_train, seg_len=args.in_len_val + args.aft_seq_length_val,
                                     lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13, idx_dim=idx_dim,
                                     input_time_length=args.input_time_length, in_len_val=args.in_len_val, mode='val')
    dataloader_vali, _ = create_loader(val_set,
                                       batch_size=val_batch_size,
                                       shuffle=False, is_training=False,
                                       pin_memory=args.pin_memory, drop_last=drop_last,
                                       num_workers=num_workers,
                                       distributed=False, use_prefetcher=use_prefetcher, return_num=2)

    file_list_train = os.path.join(save_dir, 'all_gdal_len1')

    test_set = ERA5_dataset_gdal_len1(file_list_train,
                               seg_len=args.in_len_val + args.aft_seq_length_test,
                               lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13, idx_dim=idx_dim,
                                      input_time_length=args.input_time_length, in_len_val=args.in_len_val, mode='test')  # **********************  overlap_step
    # test_set = ERA5_dataset(np.asarray(result_var_test, dtype='float32'), np.asarray(time_var_val, dtype='float32'))  # **********************  overlap_step

    dataloader_test, _ = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=args.pin_memory, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=use_prefetcher, return_num=2)  # set distributed=False to assign the value to the fuxi framework
    del test_set
    # print('load time: ',time.time() - time0)

    return dataloader_train, sampler_train, dataloader_vali, dataloader_test, #sampler_train

class ERA5_dataset_1440_721_cn(Dataset):
    def __init__(self, file_list, time_embed, seg_len, lon_len=64, lat_len=32,
                 bs=32, time_inte=None, list_len=8, is_training=False, ch_num_13=4, var_num=7, idx_dim=None):
        super(ERA5_dataset_1440_721_cn, self).__init__()
        self.file_list = file_list
        self.time_embed = time_embed
        self.bs, self.seg_len, self.lon_len, self.lat_len = bs, seg_len, lon_len, lat_len
        self.ch_num = 48
        self.time_inte = time_inte
        self.idx = 0
        self.var_num = var_num

    def __getitem__(self, idx):
        # idx_sam = self.index_sample[idx]
        rand_idx = (self.idx // self.bs) % len(self.time_inte)
        rand_inte = self.time_inte[rand_idx]

        # print([[self.file_list[j][i] for j in range(self.var_num)] for i in range(idx, idx+self.seg_len*rand_inte, rand_inte)])
        data = np.array([[read_img(self.file_list[j][i]) for j in range(self.var_num-1)] for i in range(idx, idx+self.seg_len*rand_inte, rand_inte)])
        data_surface = np.array([read_img(self.file_list[-1][i])  for i in
                         range(idx, idx + self.seg_len * rand_inte, rand_inte)])

        time_embedding = np.array(
            [self.time_embed[i] for i in range(idx, idx + self.seg_len * rand_inte, rand_inte)])

        self.idx = self.idx + 1
        return data, data_surface, np.transpose(time_embedding, [1,0]), rand_idx

    def __len__(self):
        # return 100
        return len(self.file_list[0]) - (self.seg_len-1) * max(self.time_inte)# + 1

class ERA5_dataset_len1_1440_721_cn(Dataset):
    def __init__(self,  file_list, time_embed, seg_len, lon_len=64,
                 lat_len=32, bs=32, list_len=8, is_training=False, ch_num_13=4,
                 idx_dim=None, input_time_length=1, in_len_val=5, sample_inte=4, var_num=7, mode='train'):
        super(ERA5_dataset_len1_1440_721_cn, self).__init__()
        self.len_diff = in_len_val - input_time_length
        self.in_len_val = in_len_val
        self.file_list = file_list
        self.time_embed = time_embed
        # print('self.file_list', self.file_list)

        self.bs, self.seg_len, self.lon_len, self.lat_len = bs, seg_len, lon_len, lat_len
        self.ch_num = 48
        self.mode = mode
        self.sample_inte = sample_inte
        self.var_num = var_num

    def __getitem__(self, idx):
        idx = idx * self.sample_inte
        data = np.array([[read_img(self.file_list[j][i]) for j in range(self.var_num-1)] for i in
                         range(idx, idx+self.seg_len)])
        data_surface = np.array([read_img(self.file_list[-1][i]) for i in
                         range(idx, idx+self.seg_len)])
        # a = np.array([[read_img(self.file_list[j][i]) for i in range(idx, idx+self.seg_len)] for j in range(self.var_num)])
        time_embedding = np.array([self.time_embed[i] for i in range(idx, idx + self.seg_len)])
        # print(a.shape)
        # print(time_embedding.shape)
        return data, data_surface, np.transpose(time_embedding, [1,0])

    def __len__(self):
        # print('len(self.file_list) // self.sample_inte', len(self.file_list) // self.sample_inte)
        return (len(self.file_list[0]) - self.seg_len + 1 ) // self.sample_inte

def load_ERA5_dataset_1440_721_cn(batch_size, val_batch_size, test_batch_size, lon_len, lat_len, era5_data_path, num_workers=4,
                           in_shape=[10, 1, 64, 64], distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
                           test=False, root_dir=None, args=None):
    time_embed = np.load(os.path.join(root_dir, "era5_post/_1440_721_cn/time_diff_emb_2021_2023.npy")).transpose([1,0]) #

    time_list = np.arange('2021-01-01', '2024-01-01', dtype='datetime64[1h]')
    image_size = in_shape[-1] if in_shape is not None else 145
    ch_num_13 = 8
    var_name = ['q', 't', 'w', 'z', 'u', 'v', 'surface']

    file_list_era5 = [[os.path.join(era5_data_path, 'era5_asia_' + var_namei + '_' + str(f) + '.tif')
                     for f in time_list] for var_namei in var_name]
    print('len(file_list_era5[0])', len(file_list_era5[0]))
    print('time_embed', time_embed.shape)

    trainset_num = int(len(time_list) * args.trainset_ratio)
    file_list_train = [file_list_era50[:trainset_num] for file_list_era50 in file_list_era5]
    file_list_test = [file_list_era50[trainset_num:] for file_list_era50 in file_list_era5]
    print(file_list_train[0][-1])
    print(file_list_test[0][0])
    time_embed_train = time_embed[:trainset_num]
    time_embed_test = time_embed[trainset_num:]
    if not test:
        time0 = time.time()

        train_set = []
        for i in range(len(args.pred_len)):
            train_set.append(ERA5_dataset_1440_721_cn(file_list_train, time_embed_train,
                                      seg_len=args.input_time_length+args.pred_len[i], lon_len=lon_len, lat_len=lat_len,
                                             bs=args.batch_size, time_inte=args.time_inte, var_num=len(var_name)))  # **********************  overlap_step)

        # del result_var, time_var
        # gc.collect()
        print(time.time() - time0)
        dataloader_train = []
        sampler_train = []
        for i in range(len(args.pred_len)):
            dataloader_train_tmp, sampler_train_tmp = create_loader(train_set[i],
                                                            batch_size=batch_size,
                                                            shuffle=True, is_training=True,
                                                            pin_memory=args.pin_memory, drop_last=True,
                                                            num_workers=num_workers,
                                                            distributed=distributed, use_prefetcher=use_prefetcher
                                                                    , return_num=4)
            dataloader_train.append(dataloader_train_tmp)
            sampler_train.append(sampler_train_tmp)

    elif test:
        # var_data_train, var_data_val, time_diff_emb_train, time_diff_emb_val = None, None, None, None
        dataloader_train, sampler_train, dataloader_vali = None, None, None

    val_set = ERA5_dataset_len1_1440_721_cn(file_list_test, time_embed_test,
                                     seg_len=args.in_len_val + args.aft_seq_length_val,
                                     lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                     input_time_length=args.input_time_length, in_len_val=args.in_len_val,
                                     sample_inte=args.sample_inte_test, var_num=len(var_name), mode='val')
    dataloader_vali, _ = create_loader(val_set,
                                       batch_size=val_batch_size,
                                       shuffle=False, is_training=False,
                                       pin_memory=args.pin_memory, drop_last=drop_last,
                                       num_workers=num_workers,
                                       distributed=False, use_prefetcher=False, return_num=3)

    test_set = ERA5_dataset_len1_1440_721_cn(file_list_test, time_embed_test,
                               seg_len=args.in_len_val + args.aft_seq_length_test,
                               lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                      input_time_length=args.input_time_length, in_len_val=args.in_len_val,
                                      sample_inte=args.sample_inte_test, var_num=len(var_name), mode='test')  # **********************  overlap_step
    # test_set = ERA5_dataset(np.asarray(result_var_test, dtype='float32'), np.asarray(time_var_val, dtype='float32'))  # **********************  overlap_step

    dataloader_test, _ = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=args.pin_memory, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=True, return_num=3)  # set distributed=False to assign the value to the fuxi framework
    print(len(dataloader_test)) #
    del test_set

    return dataloader_train, sampler_train, dataloader_vali, dataloader_test, #sampler_train

if __name__ == '__main__':
    import pandas as pd
