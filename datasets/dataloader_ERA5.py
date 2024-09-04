import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import time
import gc
import h5py
import pickle
from utils.utils import get_days_in_year, create_folder_if_not_exists, sort_by_last_digit
from datasets.utils import create_loader
from datasets.utils import read_img, write_img_gdal
import torchvision.transforms as trans

# name: 10m_u_component_of_wind, shape (92044, 64, 32) 0
# name: 10m_v_component_of_wind, shape (92044, 64, 32) 1
# name: 10m_wind_speed, shape (92044, 64, 32) 2
# name: 2m_temperature, shape (92044, 64, 32) 3
# name: angle_of_sub_gridscale_orography, shape (64, 32)
# name: anisotropy_of_sub_gridscale_orography, shape (64, 32)
# name: geopotential_at_surface, shape (64, 32)
# name: high_vegetation_cover, shape (64, 32)
# name: lake_cover, shape (64, 32)
# name: lake_depth, shape (64, 32)
# name: land_sea_mask, shape (64, 32)
# name: low_vegetation_cover, shape (64, 32)
# name: mean_sea_level_pressure, shape (92044, 64, 32) 4
## name: sea_ice_cover, shape (92044, 64, 32)
## name: sea_surface_temperature, shape (92044, 64, 32)
# name: slope_of_sub_gridscale_orography, shape (64, 32)
# name: soil_type, shape (64, 32)
# name: standard_deviation_of_filtered_subgrid_orography, shape (64, 32)
# name: standard_deviation_of_orography, shape (64, 32)
# name: surface_pressure, shape (92044, 64, 32) 5
# name: toa_incident_solar_radiation, shape (92044, 64, 32) 6
# name: toa_incident_solar_radiation_12hr, shape (92044, 64, 32) 7
## name: toa_incident_solar_radiation_24hr, shape (92044, 64, 32)
# name: toa_incident_solar_radiation_6hr, shape (92044, 64, 32) 8
# name: total_cloud_cover, shape (92044, 64, 32) 9
# name: total_column_water_vapour, shape (92044, 64, 32) 10
# name: total_precipitation_12hr, shape (92044, 64, 32) 11
## name: total_precipitation_24hr, shape (92044, 64, 32)
# name: type_of_high_vegetation, shape (64, 32)
# name: type_of_low_vegetation, shape (64, 32)

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
    # num_ch = dataset.shape[1]
    # dataset0 = np.transpose(dataset, [1,0,2,3]).reshape([num_ch, -1])
    # mean = np.mean(dataset0, 1)#[:coor_dims]
    mean = np.reshape(mean_std[0, :], [1, -1, 1, 1])
    # std = np.std(dataset0, 1)
    std = np.reshape(mean_std[1, :], [1, -1, 1, 1])
    # print(dataset.shape)
    # print(mean.shape)
    dataset[:, p_dim:] = (dataset[:, p_dim:] - mean) / std

    return dataset#, dataset_const, dataset_time

class ERA5_dataset_gdal_len1(Dataset):
    def __init__(self, main_path, seg_len, lon_len=64, lat_len=32, bs=32, list_len=8, is_training=False, ch_num_13=4,
                 idx_dim=None, input_time_length=1, in_len_val=5, mode='train'):
        super(ERA5_dataset_gdal_len1, self).__init__()
        self.file_list = list_subdirectories(main_path)
        self.file_list = sorted(self.file_list, key=sort_by_last_digit)

        self.bs, self.seg_len, self.lon_len, self.lat_len = bs, seg_len, lon_len, lat_len
        self.mode = mode

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.mode == 'test':
            idx = idx * 2
        a = np.array([np.load(os.path.join(self.file_list[i], 'var_data.npy')) for i in range(idx, idx+self.seg_len)])
        # for j in range(ch_num_13):
        #     temp_list.append(np.load(os.path.join(i, f'{j}.npy')))
        # a = np.concatenate(temp_list, 1, dtype='float32')
        # time_var.append(np.load(os.path.join(i, 'time_data.npy')))
        b = np.array([np.load(os.path.join(self.file_list[i], 'time_data.npy')) for i in range(idx, idx+self.seg_len)])
        # print(time.time()-time0)
        # print(a.shape)
        # print(b.shape)
        return a, np.transpose(b.squeeze(), axes=[1,0])

    def __len__(self):
        return (len(self.file_list) - self.seg_len + 1) // 2
            # print(len(list(range(0, get_days_in_year(*test_year)*4-self.seg_len+1, 2))))
            # return len(list(range(0, get_days_in_year(*test_year)*4-self.seg_len+1, 2)))

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

def load_ERA5_dataset_per(batch_size, val_batch_size, test_batch_size, lon_len, lat_len, save_dir='/data02/lisl/ERA5', num_workers=4,
                           in_shape=[10, 1, 64, 64], distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
                           test=False, args=None):
    image_size = in_shape[-1] if in_shape is not None else 64
    ch_num_13 = 8
    seg_len = args.input_time_length + args.aft_seq_length_train
    if not test:
        time0 = time.time()

        # # var_data_train = np.load(os.path.join(base_dir, 'datasets', 'var_data_train.npy'))
        # # var_data_val = np.load(os.path.join(base_dir, 'datasets', 'var_data_val.npy'))
        # # time_diff_emb_train = np.load(os.path.join(base_dir, 'datasets', 'time_diff_emb_train.npy'))
        # # time_diff_emb_val = np.load(os.path.join(base_dir, 'datasets', 'time_diff_emb_val.npy'))
        #
        # seg_len = args.input_time_length + args.aft_seq_length_train
        # file_list_train = list_subdirectories(os.path.join(save_dir, 'train'))[:get_days_in_year(2000, 2018)*4-seg_len+1]
        # result_var = []
        # # result_var = np.empty([args.input_time_length+args.aft_seq_length_train, 0, lon_len, lat_len])
        # time_var = []
        #
        # for idx, i in enumerate(file_list_train):
        #     if idx % 1000 == 0:
        #         print(idx)
        #     temp_list = [np.load(os.path.join(i, f'{j}.npy'))[:, idx_dim[j, 0]:idx_dim[j, 1]] for j in range(ch_num_13)]
        #     # for j in range(ch_num_13):
        #     #     temp_list.append(np.load(os.path.join(i, f'{j}.npy')))
        #
        #     result_var.append(np.concatenate(temp_list, 1, dtype='float32'))
        #     time_var.append(np.load(os.path.join(i, 'time_data.npy')))
        #     # del temp_list
        # # print(np.asarray(result_var).shape)
        # print(time.time()-time0)
        # result_var = np.asarray(result_var)#.reshape([len(file_list_train), args.input_time_length+args.aft_seq_length_train, -1, lon_len, lat_len])
        # time_var = np.asarray(time_var, dtype='float32')
        # np.save(os.path.join(save_dir, 'var_data_train_48.npy'), result_var)
        # np.save(os.path.join(save_dir, 'time_diff_emb_train_48.npy'), time_var)
        # print(jj)
        # result_var = np.load(os.path.join(save_dir, 'var_data_train_48.npy'))#[-(get_days_in_year(2010, 2018)*4-seg_len+1):]
        # time_var = np.load(os.path.join(save_dir, 'time_diff_emb_train_48.npy'))#[-(get_days_in_year(2010, 2018)*4-seg_len+1):]
        # result_var = np.random.random([100, 5, 48, 64, 32])  # debug
        # time_var = np.random.random([100, 11, 5])
        # train_set = ERA5_dataset(result_var, time_var)

        train_set = []
        for i in range(len(args.pred_len)):
            train_set.append(ERA5_dataset_gdal_4model(os.path.join(save_dir, 'all_gdal_len1'),
                                      seg_len=args.input_time_length+args.pred_len[i], lon_len=64, lat_len=32,
                                             bs=args.batch_size, time_inte=args.time_inte))  # **********************  overlap_step)

        # del result_var, time_var
        # gc.collect()
        print(time.time() - time0)

        # file_list_train = os.path.join(save_dir, 'all_gdal_len1')
        # result_var_val = []
        # time_var_val = []
        # for i in file_list_train:
        #     temp_list = []
        #     for j in range(ch_num_13):
        #         temp_list.append(np.load(os.path.join(i, f'{j}.npy')))
        #     result_var_val.append(np.asarray(temp_list).transpose(1,0,2,3,4).reshape([args.input_time_length+args.aft_seq_length_val, -1, lon_len, lat_len]))
        #     time_var_val.append(np.load(os.path.join(i, 'time_data.npy')))

        # val_set = ERA5_dataset_gdal_len1(file_list_train, seg_len=args.in_len_val+args.aft_seq_length_val,
        #                            lon_len=64, lat_len=32, ch_num_13=ch_num_13, idx_dim=idx_dim,
        #                                  input_time_length=args.input_time_length, in_len_val=args.in_len_val, mode='val')  # **********************  overlap_step
        dataloader_train = []
        sampler_train = []
        for i in range(len(args.pred_len)):
            dataloader_train_tmp, sampler_train_tmp = create_loader(train_set[i],
                                                            batch_size=batch_size,
                                                            shuffle=True, is_training=True,
                                                            pin_memory=True, drop_last=True,
                                                            num_workers=num_workers,
                                                            distributed=distributed, use_prefetcher=use_prefetcher)
            dataloader_train.append(dataloader_train_tmp)
            sampler_train.append(sampler_train_tmp)
        # dataloader_vali, _ = create_loader(val_set,
        #                                    batch_size=val_batch_size,
        #                                    shuffle=False, is_training=False,
        #                                    pin_memory=True, drop_last=drop_last,
        #                                    num_workers=num_workers,
        #                                    distributed=distributed, use_prefetcher=use_prefetcher)
        # del train_set, val_set
        # print(len(dataloader_vali))
    elif test:
        # var_data_train, var_data_val, time_diff_emb_train, time_diff_emb_val = None, None, None, None
        dataloader_train, sampler_train, dataloader_vali = None, None, None
    file_list_train = os.path.join(save_dir, 'data_npy')
    val_set = ERA5_dataset_gdal_len1(file_list_train, seg_len=args.in_len_val + args.aft_seq_length_val,
                                     lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                     input_time_length=args.input_time_length, in_len_val=args.in_len_val, mode='val')
    dataloader_vali, _ = create_loader(val_set,
                                       batch_size=val_batch_size,
                                       shuffle=False, is_training=False,
                                       pin_memory=True, drop_last=drop_last,
                                       num_workers=num_workers,
                                       distributed=False, use_prefetcher=False)
    # var_data_test = np.load(os.path.join(base_dir, 'datasets', 'var_data_test.npy'))
    # print(var_data_test.shape)
    # var_const_data = np.load(os.path.join(base_dir, 'datasets', 'var_const_data.npy'))

    # time_diff_emb_test = np.load(os.path.join(base_dir, 'datasets', 'time_diff_emb_test.npy'))
    # print(var_data_val[:, -1:, :, :])
    # print('**************************')
    # eps = 1e-9
    # var_data_train[:, -1:, :, :] = np.log(var_data_train[:, -1:, :, :] + eps) - np.log(eps)
    # var_data_val[:, -1:, :, :] = np.log(var_data_val[:, -1:, :, :] + eps) - np.log(eps)
    # print(var_data_val[:, -1:, :, :])
    # print('**************************')
    # min_val = np.min(var_data_val[:, -1:, :, :])
    # max_val = np.max(var_data_val[:, -1:, :, :])
    # var_data_val[:, -1:, :, :] = (var_data_val[:, -1:, :, :] - min_val) / (max_val - min_val)
    # print(var_data_val[:, -1:, :, :])

    # print(i)
    # print(var_data_test.shape)
    # print(var_const_data.shape)
    # print(time_diff_emb_test.shape)
    # result_var_test = []
    # time_var_val = []
    # for i in file_list_train:
    #     temp_list = []
    #     for j in range(ch_num_13):
    #         temp_list.append(np.load(os.path.join(i, f'{j}.npy')))
    #     result_var_test.append(np.asarray(temp_list).transpose(1, 0, 2, 3, 4).reshape(
    #         [args.input_time_length + args.aft_seq_length_val, -1, lon_len, lat_len]))   # ***************************
    #     time_var_val.append(np.load(os.path.join(i, 'time_data.npy')))

    # val_set = ERA5_dataset(np.asarray(result_var_val), np.asarray(time_var_val))  # **********************  overlap_step
    test_set = ERA5_dataset_gdal_len1(file_list_train,
                               seg_len=args.in_len_val + args.aft_seq_length_test,
                               lon_len=lon_len, lat_len=lat_len, ch_num_13=ch_num_13,
                                      input_time_length=args.input_time_length, in_len_val=args.in_len_val, mode='test')  # **********************  overlap_step
    # test_set = ERA5_dataset(np.asarray(result_var_test, dtype='float32'), np.asarray(time_var_val, dtype='float32'))  # **********************  overlap_step

    dataloader_test, _ = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=False)  # set distributed=False to assign the value to the fuxi framework
    del test_set
    # print('load time: ',time.time() - time0)

    return dataloader_train, sampler_train, dataloader_vali, dataloader_test, #sampler_train


if __name__ == '__main__':
    import pandas as pd
    dataset_dir = '/data01/lisl/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    dataset_dir2 = '/data01/lisl/forcast/fuxi/2020-64x32_equiangular_conservative.zarr/'
    dataset0 = xr.open_zarr(dataset_dir)
    for i in var_name[1:]:   # for testing ************************************************
        data_tmp = dataset0[i].data
        print(f'name: {i}, shape {np.array(data_tmp).shape}')
    exit()
    a = dataset0.sel(time=slice('1979-01-01', '2017-12-31'))
    print(a['time'].data.shape)
    steps_per_day = 4
    train_days = get_days_in_year(1979, 2018) * steps_per_day
    print(train_days)
    # a = np.arange('2020-01-01', '2020-12-17', dtype='datetime64[12h]')
    # b = dataset0.sel(time=slice(str(a[10]) + ':00:00.000000000', str(a[20]) + ':00:00.000000000'))
    # str(a[10]) + ':00:00.000000000'
    print(b['10m_u_component_of_wind'])
    for year in range(1959, 2022):
        timestamps = np.arange('1959-01-01', '2022-01-01', dtype='datetime64[6h]')
        year_mask = (timestamps >= np.datetime64(f'{year}-01-01')) & (timestamps <= np.datetime64(f'{year}-12-31'))
        year_data = np.array(dataset0['total_precipitation_6hr'])[year_mask]

        # print('year: ', year)
        print('min: ', np.min(year_data))
        print('max: ', np.max(year_data))
        print('mean:', np.mean(year_data))
        index = np.where((np.abs(year_data) < 1e-6) & (np.abs(year_data) > 1e-7))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))
        index = np.where((np.abs(year_data) < 1e-5) & (np.abs(year_data) > 1e-6))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))
        index = np.where((np.abs(year_data) < 1e-4) & (np.abs(year_data) > 1e-5))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))
        index = np.where((np.abs(year_data) < 1e-3) & (np.abs(year_data) > 1e-4))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))
        index = np.where((np.abs(year_data) < 1e-2) & (np.abs(year_data) > 1e-3))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))
        index = np.where((np.abs(year_data) < 1e-1) & (np.abs(year_data) > 1e-2))
        print('ratio_0:', index[0].shape[0] / np.prod(year_data.shape))

        index = np.where(year_data < 0)
        print('ratio_nega:', index[0].shape[0] / np.prod(year_data.shape))

        eps = 1e-4  # 尝试调大。如8,7,6
        year_data = np.log(year_data + eps) - np.log(eps)

        print('min_log: ', np.min(year_data))
        print('max_log: ', np.max(year_data))
        print('mean_log:', np.mean(year_data))

    print(jj)
    b = np.random.random((8, 400, 400))
    t0 = time.time()
    np.save('/data01/lisl/forcast/ours/npy.npy', b)
    tmp = np.load('/data01/lisl/forcast/ours/npy.npy')
    print(time.time() - t0)
    ds = xr.Dataset(
        data_vars={
            "a": (("time", "longtitude", "latitude"), b),
            # "b": ("t", np.full(8, 3), {"b atrri": "b value"}),
        },
        coords={
            "longtitude": ("longtitude", np.arange(0, 400, step=1)),
            "latitude": ("latitude", np.arange(0, 400, step=1)),
            "time": ("time",
                     ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07",
                      "2021-01-08"])
        }
    )
    print(ds)
    # print(ds['a'].data)
    # ds2 = xr.Dataset()
    # # total_pre = xr.DataArray(pred_res, dims=)
    # ds2['total_precipitation_6hr'] = np.random.random((400, 300))
    # ds2.coords['lat'] = np.arange(0, 400, step=1)
    # ds2.coords['lon'] = np.arange(0, 300, step=1)
    # ds2.coords['time'] = ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07",
    #                   "2021-01-08"]
    # # xr.to_zarr(ds, )
    # print(ds2)
    import zarr
    t0 = time.time()
    ds.to_zarr("/data01/lisl/forcast/ours/ds1.zarr", mode="w")
    a = xr.open_zarr("/data01/lisl/forcast/ours/ds1.zarr")
    b = a['a']
    print(time.time() - t0)
    # python weatherbench2_main/model2023/datasets/dataloader_ERA5.py
    # dataset_dir = '/data01/lisl/forcast/hres/2016-2022-0012-64x32_equiangular_conservative.zarr/'
    # dataset = xr.open_zarr(dataset_dir)
    # # print(dataset.data_vars)
    # print(np.array(dataset['time']))
    # print(np.array(dataset['total_precipitation_6hr']).shape)

    # load_ERA5_dataset(16,16,'')
