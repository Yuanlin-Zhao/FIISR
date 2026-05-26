import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution with Contrastive Learning."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # 【修改点 1】：大幅调低对比损失的绝对权重，不能让它干扰基础重建
        self.contrastive_weight = 0.05
        self.contrastive_margin = 0.1

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define losses
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get(
            'perceptual_opt') else None
        self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device) if train_opt.get('fft_opt') else None

        if train_opt.get('align_opt'):
            self.cri_align = build_loss(train_opt['align_opt']).to(self.device)
            self.cri_align_weight = train_opt['align_opt'].get("loss_weight", 1.0)
        else:
            self.cri_align = None

        self.cri_teacher = build_loss(train_opt['teacher_opt']).to(self.device) if train_opt.get(
            'teacher_opt') else None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_fft is None:
            raise ValueError('Pixel, perceptual and FFT losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        # 【修改点 2】：增加验证集/测试集兼容。验证集通常只有 lq 没有 lq1/2/3
        if 'lq1' in data:
            self.lq1 = data['lq1'].to(self.device)
            self.lq2 = data['lq2'].to(self.device)
            self.lq3 = data['lq3'].to(self.device)
        else:
            self.lq1 = data['lq'].to(self.device)  # 验证阶段使用

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # -------------------------------------------------------
        # 1. 前向传播
        # 【核心修复 1】：切断 LQ3 的梯度！负样本只作为参考锚点，绝不能反向传播破坏网络
        # -------------------------------------------------------
        if self.cri_align or self.cri_teacher:
            out1, attn1, ref1 = self.net_g(self.lq1)
            out2, attn2, ref2 = self.net_g(self.lq2)
            with torch.no_grad():  # 停止负样本梯度
                out3, _, _ = self.net_g(self.lq3)
        else:
            out1 = self.net_g(self.lq1)
            out2 = self.net_g(self.lq2)
            with torch.no_grad():  # 停止负样本梯度
                out3 = self.net_g(self.lq3)
            attn1, ref1 = None, None
            attn2, ref2 = None, None

        self.output = out1
        l_total = 0
        loss_dict = OrderedDict()

        # -------------------------------------------------------
        # 2. 基础重建损失 (保证超分精度不下降)
        # -------------------------------------------------------
        def compute_recon_loss(out, attn, ref, name):
            l_step = 0
            if self.cri_pix:
                l_p = self.cri_pix(out, self.gt)
                l_step += l_p
                loss_dict[f'l_pix_{name}'] = l_p
            if self.cri_fft:
                l_f = self.cri_fft(out, self.gt)
                l_step += l_f
                loss_dict[f'l_fft_{name}'] = l_f
            if self.cri_teacher and attn is not None:
                l_t = self.cri_teacher(attn, self.gt)
                l_step += l_t
                loss_dict[f'l_teacher_{name}'] = l_t
            if self.cri_align and ref is not None:
                l_a = ref * self.cri_align_weight
                l_step += l_a
                loss_dict[f'l_align_{name}'] = ref
            return l_step

        # 正样本 1 全量约束
        l_total += compute_recon_loss(out1, attn1, ref1, 'lq1')
        # 正样本 2 辅助约束
        l_total += compute_recon_loss(out2, attn2, ref2, 'lq2')

        # -------------------------------------------------------
        # 3. 对比损失 (Triplet Loss 纠正)
        # -------------------------------------------------------
        # 【核心修复 2】：正确的 Triplet 逻辑
        # Anchor = 正样本输出 (out1, out2)
        # Positive = 真实高清图 (GT)
        # Negative = 负样本输出 (out3, 已无梯度)
        # 目标：Anchor 靠近 Positive，且 Anchor 远离 Negative

        # d_pos: 正样本距离 (越小越好)
        d_pos1 = F.l1_loss(out1, self.gt)
        d_pos2 = F.l1_loss(out2, self.gt)

        # d_neg: Anchor 与负样本输出的距离 (越大越好)
        d_neg1 = F.l1_loss(out1, out3)
        d_neg2 = F.l1_loss(out2, out3)

        # 公式: max(0, D(anchor, pos) - D(anchor, neg) + margin)
        l_triplet1 = torch.clamp(d_pos1 - d_neg1 + self.contrastive_margin, min=0.0)
        l_triplet2 = torch.clamp(d_pos2 - d_neg2 + self.contrastive_margin, min=0.0)

        l_contrast = (l_triplet1 + l_triplet2) * self.contrastive_weight
        l_total += l_contrast
        loss_dict['l_contrast'] = l_contrast

        # -------------------------------------------------------
        # 4. 全局感知损失 (仅对主路径)
        # -------------------------------------------------------
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(out1, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep

        # 5. 反向传播
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.output = self.net_g_ema(self.lq1)
            else:
                self.output = self.net_g(self.lq1)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq1
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')  # 去掉了 _{self.opt["val"]["suffix"]}
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')  # 去掉了 _{self.opt["name"]}
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq1.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)