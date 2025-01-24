import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math
import numpy as np
from model2023.modules.vit import VisionTransformer


class Model_x(nn.Module):
    def __init__(self, in_shape, out_ch, hid_S=32, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.1, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, args=None, **kwargs):
        super(Model_x, self).__init__()
        B, T, C, H, W = in_shape  # T is input_time_length

        self.args = args
        self.hid_S = hid_S
        self.out_ch = out_ch

        self.H_d, self.W_d = args.H_d, args.W_d #
        # Encoder, Decoder
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, H, W, self.H_d, self.W_d, act_inplace=act_inplace)        #   C_in, C_hid, N_S, spatio_kernel, act_inplace=True
        self.dec = Decoder(hid_S, out_ch, N_S, spatio_kernel_dec, H_tar=args.tar_size, W_tar=args.tar_size, H_d=self.H_d, W_d=self.W_d,
                           act_inplace=act_inplace)  # 1 means the total_precipitation_6hr var

        self.time_embedding = nn.Sequential(
                nn.Linear(args.input_time_length*args.time_emb_num, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),   #  hid_S  ***
                nn.LeakyReLU(),
                nn.Linear(256, int(hid_S*0.5))
            )
        # prediction model
        scale_fac = args.input_time_length
        num_block = 12
        N_S_pred = 4
        norm_band_num = 4  # in self attention
        patch_size = 4     # in self attention
        step = 2

        self.hid = [nn.Sequential(
            Encoder(int(hid_S * (T+0.5)), scale_fac * hid_S, N_S_pred, spatio_kernel_enc, H=self.H_d, W=self.W_d,
                    H_d=self.H_d, W_d=self.W_d, act_inplace=act_inplace),
            VisionTransformer([self.H_d, self.W_d], patch_size=[4,4], inp_chans=scale_fac*hid_S, out_chans=scale_fac*hid_S,    # from makani
                                     embed_dim=768, depth=8, num_heads=12, mlp_ratio=4., qkv_bias=True, mlp_drop_rate=0.0,
                                        attn_drop_rate=0.0, path_drop_rate=0.0, norm_layer="layer_norm", comm_inp_name="fin",
                                     comm_hidden_name="fout"),
            Decoder(scale_fac * hid_S, hid_S, N_S_pred, spatio_kernel_dec, H_tar=self.H_d, W_tar=self.W_d, H_d=self.H_d,
                     W_d=self.W_d, act_inplace=act_inplace))
            for i in range(len(self.args.time_inte))]  #  args.drop
        self.hid = nn.ModuleList(self.hid)

    def forward(self, x_raw, const_data, time_data, labels, diff_ori=None, aft_seq_length=1, hid_i=0, shrink=1, mode='train', device=None, **kwargs):
        # print(x_raw.shape)
        B, T, C, H, W = x_raw.shape
        x = x_raw.flatten(0,1)

        embed = self.enc(x)
        # print(embed.shape)
        _, C_, H_, W_ = embed.shape

        if mode == 'train':
            embed_label = self.enc(labels.detach().flatten(0,1)).unflatten(0, [B, aft_seq_length])
            label_pred = self.dec(embed).reshape((B, T, -1, H, W))
            embed_diff = None
            embed_label_dec = self.dec(embed_label.view(-1, self.hid_S, self.H_d, self.W_d)).view(B, aft_seq_length, -1, H, W)
            # embed_label, label_pred, embed_diff, embed_tmp, embed_label_dec = None, None, None, None, None
            embed_tmp = embed.view(B, self.args.input_time_length, self.hid_S, self.H_d, self.W_d)
        else:
            embed_label, label_pred, embed_diff, embed_tmp, embed_label_dec = None, None, None, None, None

        z = embed.view(B, -1, H_, W_)#.gather()
        if mode == 'train':
            hid = self._predict(z, time_data, aft_seq_length, hid_i=hid_i, shrink=shrink, mode=mode,
                                device=device)
        else:
            hid = self._predict_pangu(z, time_data, aft_seq_length, shrink=shrink, mode=mode,
                                      device=device)
        # print(hid.shape)
        hid = hid.unflatten(1, [aft_seq_length, self.hid_S]).flatten(0,1)

        Y = self.dec(hid)   # , self.res_conv(skip).reshape(B*self.args.aft_seq_length, -1, H_, W_)
        Y = Y.unflatten(0, [B, aft_seq_length])

        return Y, hid.unflatten(0, [B, aft_seq_length]), embed_label, label_pred, \
               embed_tmp, embed_diff, embed_label_dec

    def _predict_pangu(self, cur_seq, cur_time_data, aft_seq_length, shrink=False, mode='val', device=None, **kwargs):
        # pred_len means the var length that the model would predict
        # fast version
        pred_y = []
        re_time_inte = self.args.time_inte[::-1]

        tmp_arr = np.arange(self.args.time_inte[-1])
        tmp_list = list(tmp_arr+1)
        tmp_list.insert(0, self.args.time_inte[-1])
        map_dict_tmp = dict(zip(tmp_arr, np.array(tmp_list[:-1])))  # *************

        tmp_last_lat_fea = None
        for pred_i in range(1, aft_seq_length+1):
            if pred_i <=4 :
                iter_idx_num = 0
                pred_i_tmp = pred_i
                cur_seq_tmp = cur_seq.clone()
            else:
                iter_idx_num = (pred_i // self.args.time_inte[-1]) * self.args.time_inte[-1] if \
                    pred_i % self.args.time_inte[-1] != 0 else ((pred_i-1) // self.args.time_inte[-1]) * self.args.time_inte[-1]
                pred_i_tmp = map_dict_tmp[pred_i % self.args.time_inte[-1]]
                cur_seq_tmp = tmp_last_lat_fea
            iter_num = []

            for i in reversed(self.args.time_inte):
                iter_num.append(pred_i_tmp // i)
                pred_i_tmp = pred_i_tmp % i
            # print(iter_num)

            for j in range(len(iter_num)):
                if iter_num[j] != 0 and j == 0:
                    # print(f'predict model {len(iter_num)-j-1} iterate {iter_num[j]}')
                    for k in range(iter_num[j]):
                        tmp_last_lat_fea = self.forward_recur(cur_seq_tmp,
                                           cur_time_data[:, :, iter_idx_num:iter_idx_num+self.args.input_time_length]
                                                      , hid_i=len(iter_num)-j-1)
                        cur_seq_tmp = tmp_last_lat_fea.clone()
                        iter_idx_num += self.args.input_time_length * re_time_inte[j]
                elif iter_num[j] != 0 and j != 0:
                    for k in range(iter_num[j]):
                        cur_seq_tmp = self.forward_recur(cur_seq_tmp,
                                                         cur_time_data[:, :, iter_idx_num:iter_idx_num + self.args.input_time_length]
                                                         , hid_i=len(iter_num) - j - 1)
                        iter_idx_num += self.args.input_time_length * re_time_inte[j]
            pred_y.append(cur_seq_tmp)   # **********************************

        pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def _predict(self, cur_seq, cur_time_data, aft_seq_length, hid_i=0, batch_y=None, shrink=0, mode='val', device=None, **kwargs):
        """Forward the model"""
        if aft_seq_length == self.args.pre_seq_length:
            pred_y = self.forward_recur(cur_seq,
                            cur_time_data[:, :, 0*self.args.pre_seq_length:self.args.input_time_length+0*self.args.pre_seq_length], hid_i=hid_i)
        elif aft_seq_length < self.args.pre_seq_length:
            pred_y = self.forward_recur(cur_seq, cur_time_data)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = aft_seq_length // self.args.pre_seq_length
            m = aft_seq_length % self.args.pre_seq_length

            for i in range(d):
                # print(i)
                if shrink:   # means the output length is shorter than the input
                    if mode == 'train':
                        cur_seq = cur_seq + (torch.randn(size=cur_seq.size(), dtype=torch.float32)/100.0).type(torch.float32).to(device)  # ablation
                    temp_out = self.forward_recur(cur_seq,
                                                  cur_time_data[:, :, i*self.args.pre_seq_length:i*self.args.pre_seq_length+self.args.input_time_length], hid_i=hid_i)
                    cur_seq = torch.cat([cur_seq[:, -(cur_seq.shape[1]-temp_out.shape[1]):, ...], temp_out], 1)
                    pred_y.append(temp_out)
                else:
                    if mode == 'train':
                        cur_seq = cur_seq + (torch.randn(size=cur_seq.size(), dtype=torch.float32)/100.0).type(torch.float32).to(device)  # ablation
                    cur_seq = self.forward_recur(cur_seq,
                                                 cur_time_data[:, :, i*self.args.input_time_length:(i+1)*self.args.input_time_length], hid_i=hid_i)      # assume that the length of input and output of the model is same
                    pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.forward_recur(cur_seq, cur_time_data[:, :, -(m+self.args.input_time_length+self.args.pre_seq_length):-m])
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def forward_recur(self, x, time_data, hid_i=0, **kwargs):   # const_emb coube be a const, but the time is changed
        B, C, H, W = x.shape
        # print(x.shape)
        x_res = x[:, -self.hid_S:, ...]

        time_emb = self.time_embedding(time_data.reshape(B, -1))[..., None, None].repeat(1, 1, H, W)#reshape(B, -1, H_, W_)#.contigous().view(B, C_, H_, W_)
        Y = self.hid[hid_i](torch.cat([x, time_emb], 1))
        # print(Y.shape)
        return Y + x_res

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        self.in_channels = in_channels
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation, padding_mode='circular'),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, padding_mode='circular')

        self.norm = nn.GroupNorm(2, out_channels)   # group number: 2
        # self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)  # math.sqrt(2.0 / self.in_channels)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y   # try more conv and resnet


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)
        self.conv2 = BasicConv2d(C_in, C_out, kernel_size=5, stride=stride,
                                upsampling=upsampling, padding=2,
                                act_norm=act_norm, act_inplace=act_inplace)
        self.cat_conv = nn.Conv2d(C_out*2, C_out, 1)

        self.res_conv = nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1,
                                  padding_mode='circular') if C_in != C_out else nn.Identity()

    def forward(self, x):
        # x0 = x.clone()
        y = self.conv(x)
        y2 = self.conv2(x)
        return self.cat_conv(torch.cat([y, y2], 1)) + self.res_conv(x)


def sampling_generator(N, reverse=False):
    samplings = [False, False] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, H=64, W=32, H_d=120, W_d=60, act_inplace=True, tar_dim=None):
        samplings = sampling_generator(N_S)
        # print(samplings)
        super(Encoder, self).__init__()
        self.tar_dim = tar_dim
        mid_ch = (C_hid + C_in) // 2 if ((C_hid + C_in) // 2) % 2 == 0 else (C_hid + C_in) // 2 + 1
        if tar_dim is not None:
            self.corr_linear = nn.Sequential(
                nn.Linear(C_in, C_in//2),
                nn.SiLU(),
                nn.Linear(C_in//2, C_in),
                nn.Sigmoid())
        self.enc = nn.Sequential(
              ConvSC(C_in, mid_ch, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
              ConvSC(mid_ch, C_hid, spatio_kernel, downsampling=samplings[1],
                   act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[2:]]
        )

        self.H, self.W, self.H_d, self.W_d = H, W, H_d, W_d
        print('self.H, self.W, self.H_d, self.W_d', self.H, self.W, self.H_d, self.W_d)
        if H != H_d and W != W_d:
            self.pre_process0 = nn.Conv2d(C_hid, C_hid, kernel_size=3, padding=1, padding_mode='circular', groups=C_hid,
                                          stride=int(H/H_d), dilation=int((H/H_d)/2))
            self.pre_process1 = nn.Conv2d(C_hid, C_hid, kernel_size=3, padding=1, padding_mode='circular', groups=C_hid)
        else:
            self.pre_process0 = nn.Conv2d(C_hid, C_hid, kernel_size=3, padding=1, padding_mode='circular', groups=C_hid,
                                          stride=1, dilation=int((H/H_d)/2))
            self.pre_process1 = nn.Conv2d(C_hid, C_hid, kernel_size=3, padding=1, padding_mode='circular', groups=C_hid)

    def forward(self, x):  # B*4, 3, 128, 128
        # print('x.shape', x.shape)
        B, C, H, W = x.shape
        if self.tar_dim is not None:
            corr_mat = cross_att_matrix(x.view(B, C, -1), tar_dim=self.tar_dim)
            # print(corr_mat.shape)
            x = x * self.corr_linear(corr_mat)[..., None, None]

        latent = self.enc[0](x)
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        # if self.H != self.H_d and self.W != self.W_d:
        # print(latent.shape)
        latent = self.pre_process0(latent)
        # print(latent.shape)
        latent = F.interpolate(latent, size=[self.H_d, self.W_d], mode='bilinear')
        # print(latent.shape)
        latent = self.pre_process1(latent)
        # print(latent.shape)
        return latent    #, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, H_tar=145, W_tar=145, H_d=64, W_d=64, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()

        mid_ch = (C_hid+C_out)//2 if ((C_hid+C_out)//2) % 2 ==0 else (C_hid+C_out)//2 + 1
        self.H, self.W = H_tar, W_tar
        self.H_d, self.W_d = H_d, W_d
        if self.H != self.H_d or self.W != self.W_d:
            self.pre_process0 = nn.Conv2d(C_hid, C_hid, kernel_size=3, padding=1, padding_mode='circular', groups=C_hid)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, mid_ch, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(mid_ch, C_out, 3, padding=1, padding_mode='circular')

    def forward(self, hid, enc1=None):
        if self.H != self.H_d or self.W != self.W_d:
            hid = F.interpolate(hid, size=[self.H, self.W], mode='bilinear')
            hid = self.pre_process0(hid)

        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        # print(hid.shape)
        # print(enc1.shape)
        Y = self.dec[-1](hid)  #  + enc1
        Y = self.readout(Y)
        return Y

class Decoder_recon(nn.Module):
    """Decoder"""
    def __init__(self, C_hid, out_era5_ch, N_S, spatio_kernel, H=64, W=64, out_era5_size=145, act_inplace=True, min_rain=0.1, device=None):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder_recon, self).__init__()
        self.min_rain = min_rain
        self.era5_size = out_era5_size

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )

        self.readout_era5 = nn.Sequential(
            nn.Conv2d(C_hid, (out_era5_ch+C_hid)//2, 3, padding=1, padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Conv2d((out_era5_ch+C_hid)//2, out_era5_ch, 3, padding=1, padding_mode='circular'))

    def forward(self, hid, enc1=None):
        # print(pred_mean.size())
        # print(hid.size())
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid)  #  + enc1

        recon_era5 = F.interpolate(Y, size=[self.era5_size, self.era5_size], mode='bilinear')
        recon_era5 = self.readout_era5(recon_era5)  # pred_class的为降雨发生与否的概率大小值

        # print('self.scale', self.scale)
        # print(f'Decoder', torch.max(-self.scale*Y), torch.min(-self.scale*Y))
        # mask = (1 - (Y >= self.exp_b).type(torch.int))  # no rain=0
        # print(self.exp_b)
        return recon_era5#, recon_fy4b  # -np.log(self.min_rain) - Y - exp_b 考虑到 exp_b的空间变化