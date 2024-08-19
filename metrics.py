import cv2
import numpy as np
import torch
import torch.nn.functional as F

# try:
import lpips
from skimage.metrics import structural_similarity as cal_ssim
# except:
#     lpips = None
#     cal_ssim = None

def diff_div_reg(pred_y, batch_y, tau=0.1, eps=1e-12):
    B, T, C = pred_y.shape[:3]
    if T <= 2:  return 0
    gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
    gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
    softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
    softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
    loss_gap = softmax_gap_p * \
        torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
    return loss_gap.mean()

def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, weight=None, spatial_norm=False):
    # if not spatial_norm:
    #     return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    # else:
    #     norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    #     return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()
    absolute_error = np.abs(pred - true)
    # 使用权重进行加权
    return (absolute_error * weight).mean(0).mean(-1).mean(-1)


def MSE(pred, true, weight=None, spatial_norm=False):
    # if not spatial_norm:
    #     return np.mean((pred-true)**2, axis=(0, 1)).sum()
    # else:
    #     norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    #     return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()
    mse = (pred - true) ** 2
    # 使用权重进行加权
    return np.mean(mse * weight)


def RMSE(pred, true, weight=None, spatial_norm=False):
    # if not spatial_norm:
    #     return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    # else:
    #     norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    #     return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())
    mse = (pred - true) ** 2
    # 使用权重进行加权
    weighted_mse = (mse * weight).mean(0).mean(-1).mean(-1)
    # 计算 RMSE
    return np.sqrt(weighted_mse)


def ACC(pred, true, spatial_norm=False):
    value = np.mean(np.sum(pred * true, axis=(-1, -2)) / np.sqrt(np.sum(pred ** 2, axis=(-1, -2)) * np.sum(true ** 2, axis=(-1, -2))))

    return value


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, weight=None, args=None):
    # eps = args.eps
    eval_res = {}
    # pred = pred.detach().cpu().numpy()
    # true = true.detach().cpu().numpy()
    eval_res['mae'] = MAE(pred, true, weight=weight)
    eval_res['rmse'] = RMSE(pred, true, weight=weight)
    # pred = (np.exp(pred + np.log(eps)) - eps) * args.min_max_array[1] + args.min_max_array[0]
    # true = (np.exp(true + np.log(eps)) - eps) * args.min_max_array[1] + args.min_max_array[0]
    # eval_res['mae_origin'] = MAE(pred, true, weight=weight)
    # eval_res['rmse_origin'] = RMSE(pred, true, weight=weight)
    return eval_res

