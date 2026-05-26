from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
import os
import numpy as np
import cv2
import random
import torch
import numpy as np
import random
import torch.nn.functional as F

# ==========================================

# 优化的退化处理函数 (独立于类外)

# ==========================================



def build_frequency_mask(h, w, device='cpu'):

    """生成适用于 rfft2 的各向异性高斯低通滤波器"""

    fy = torch.fft.fftfreq(h, device=device)

    fx = torch.fft.rfftfreq(w, device=device)

    grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')



    # 随机选择各向异性参数，模拟不同方向的模糊

    sigma_x = random.uniform(0.1, 0.4)

    sigma_y = random.uniform(0.1, 0.4)



    grid = (grid_y ** 2 / (2 * sigma_y ** 2)) + (grid_x ** 2 / (2 * sigma_x ** 2))

    mask = torch.exp(-grid)



    # 随机频率空洞

    if random.random() < 0.3:

        hole_mask = torch.ones_like(mask)

        for _ in range(3):

            ry, rx = random.randint(0, h // 2), random.randint(0, w // 2)

            hole_mask[max(0, ry - 5):ry + 5, max(0, rx - 5):rx + 5] = 0.5

        mask = mask * hole_mask



    return mask.unsqueeze(0).unsqueeze(0)





def frequency_degradation(img, opt):

    """改进的频域退化：加入各向异性滤波与混合噪声"""

    scale = opt.get('scale', 4)

    if img.dtype == np.uint8:

        img = img.astype(np.float32) / 255.0

    h0, w0 = img.shape[:2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


    # 1. 频域变换

    fft = torch.fft.rfft2(img_t, norm='ortho')

    amp = torch.abs(fft)

    phase = torch.angle(fft)

    mask = build_frequency_mask(h0, w0, device=device)


    amp = amp * mask

    # 3. 相位扰动 (模拟轻微结构扭曲)

    phase_shift = (torch.randn_like(phase) * random.uniform(0.01, 0.05))

    phase = phase + phase_shift



    # 4. 频率噪声 (模拟传感器伪影)

    if random.random() > 0.7:

        noise_amp = torch.randn_like(amp) * random.uniform(0, 0.02)

        amp = amp + noise_amp



    # 重构复数并逆变换

    fft_new = torch.polar(amp, phase)

    img_rec = torch.fft.irfft2(fft_new, s=(h0, w0), norm='ortho')



    img_rec = torch.clamp(img_rec, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()



    # 5. 随机插值降采样

    lq_h, lq_w = h0 // scale, w0 // scale

    interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])

    img_lq = cv2.resize(img_rec, (lq_w, lq_h), interpolation=interp)



    # 6. 额外像素噪声

    if random.random() > 0.5:

        noise_level = random.uniform(0, 0.02)

        img_lq += np.random.normal(0, noise_level, img_lq.shape).astype(np.float32)



    return np.clip(img_lq, 0, 1)



def blind_degradation_triplet(img, img_neg, opt):

    """生成对比学习三元组"""

    img1 = frequency_degradation(img, opt)

    img2 = frequency_degradation(img, opt)

    img3 = frequency_degradation(img_neg, opt)

    return img1, img2, img3


# ==========================================
# 数据集类
# ==========================================

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # 路径加载逻辑保持不变
        if self.io_backend_opt['type'] == 'lmdb':
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in opt and opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          opt['meta_info_file'], self.filename_tmpl)
        else:
            from basicsr.data.data_util import paired_paths_from_folder
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # 1. 加载 Anchor GT
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # 1. 定义一个最小安全间距 (例如 100 或者总数的 10%)
        safe_interval = max(10, len(self.paths) // 200)

        # 2. 尝试寻找一个足够远的随机索引
        while True:
            neg_index = random.randint(0, len(self.paths) - 1)
            # 如果随机到的索引与当前索引距离大于安全间距，则退出循环
            if abs(neg_index - index) > safe_interval:
                break
            # 如果数据集实在太小，无法满足间距，则强制给一个偏移量并取模
            if len(self.paths) <= safe_interval:
                neg_index = (index + len(self.paths) // 2) % len(self.paths)
                break

        neg_path = self.paths[neg_index]['gt_path']
        img_bytes_neg = self.file_client.get(neg_path, 'gt')
        img_gt_neg = imfrombytes(img_bytes_neg, float32=True)

        # 统一尺寸以便后续处理
        img_gt_neg = cv2.resize(img_gt_neg, (img_gt.shape[1], img_gt.shape[0]))

        # 3. 生成退化三元组 (LQ1, LQ2, LQ_Neg)
        img_lq1, img_lq2, img_lq3 = blind_degradation_triplet(img_gt, img_gt_neg, self.opt)

        # 4. 训练阶段的裁剪与增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # 注意：paired_random_crop 需要同时对 GT 和生成的 LQ 进行同步裁剪
            img_gts, img_lqs = paired_random_crop(
                [img_gt],
                [img_lq1, img_lq2, img_lq3],
                gt_size, scale, gt_path
            )
            img_gt = img_gts[0]
            img_lq1, img_lq2, img_lq3 = img_lqs

            # 数据增强
            imgs = augment([img_gt, img_lq1, img_lq2, img_lq3], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq1, img_lq2, img_lq3 = [np.ascontiguousarray(img) for img in imgs]

        # 5. 验证阶段尺寸对齐
        else:
            h, w = img_lq1.shape[:2]
            img_gt = img_gt[0:h * scale, 0:w * scale, :]

        # 6. 色彩转换 (BGR to Y)
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq1 = bgr2ycbcr(img_lq1, y_only=True)[..., None]
            img_lq2 = bgr2ycbcr(img_lq2, y_only=True)[..., None]
            img_lq3 = bgr2ycbcr(img_lq3, y_only=True)[..., None]

        # 7. 转 Tensor 并归一化
        imgs_t = img2tensor([img_gt, img_lq1, img_lq2, img_lq3], bgr2rgb=True, float32=True)
        img_gt, img_lq1, img_lq2, img_lq3 = imgs_t

        if self.mean is not None or self.std is not None:
            for t in [img_gt, img_lq1, img_lq2, img_lq3]:
                normalize(t, self.mean, self.std, inplace=True)

        return {
            'lq1': img_lq1,
            'lq2': img_lq2,
            'lq3': img_lq3,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)




@DATASET_REGISTRY.register()
class Dataset_PairedImage(data.Dataset):

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)