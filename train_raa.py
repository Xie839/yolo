"""
train_raa.py — YOLOv5 + RAA Module 完整训练脚本
=================================================
在标准 YOLOv5 训练流程的基础上，集成了区域自适应感知（RAA）模块的
三种辅助损失：注意力监督、多尺度一致性、目标-背景对比。

用法示例：
    python train_raa.py \\
        --img 640 --batch 16 --epochs 100 \\
        --data data/coco128.yaml \\
        --weights yolov5s.pt \\
        --cfg models/yolov5s.yaml \\
        --hyp configs/yolov5_raa.yaml \\
        --raa-enabled

依赖：
    - 标准 YOLOv5 环境（见 requirements.txt）
    - models/raa_module.py
    - utils/raa_loss.py
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

# --------------------------------------------------------------------------
# 将 YOLOv5 根目录加入 sys.path，确保可以 import yolov5 的模块
# --------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# YOLOv5 标准模块（需要安装/克隆官方 yolov5 或把代码放在同目录）
try:
    from models.yolo import Model
    from utils.dataloaders import create_dataloader
    from utils.general import (
        check_img_size, colorstr, increment_path,
        init_seeds, labels_to_class_weights, strip_optimizer,
    )
    from utils.loss import ComputeLoss
    from utils.metrics import fitness
    from utils.torch_utils import (
        EarlyStopping, ModelEMA, de_parallel, select_device,
        smart_optimizer, smart_resume, torch_distributed_zero_first,
    )
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    print("[警告] 未找到 YOLOv5 核心模块，训练脚本将以演示模式运行。")
    print("        请确认 YOLOv5 代码位于同级目录，或已安装 ultralytics。")

from models.raa_module import RegionAdaptiveAwareness
from utils.raa_loss import RAALoss

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK       = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


# ==========================================================================
# RAA-enabled ComputeLoss 包装类
# ==========================================================================

class ComputeLossWithRAA:
    """在原始 YOLOv5 ComputeLoss 基础上叠加 RAA 辅助损失。

    Args:
        base_loss_fn : 已实例化的 YOLOv5 ComputeLoss 对象。
        raa_module   : RegionAdaptiveAwareness 模块实例。
        raa_loss_fn  : RAALoss 实例。
        img_size     : (H, W)，原始输入图像尺寸。
    """

    def __init__(self, base_loss_fn, raa_module, raa_loss_fn, img_size=(640, 640)):
        self.base_loss = base_loss_fn
        self.raa       = raa_module
        self.raa_loss  = raa_loss_fn
        self.img_size  = img_size

    def __call__(self, preds, targets, intermediate_features=None):
        """
        Args:
            preds                 : 检测头输出（与标准 YOLOv5 相同）。
            targets               : [N, 6]  [img_idx, cls, cx, cy, bw, bh]（归一化）。
            intermediate_features : [P3, P4, P5]（来自模型 hook）；若为 None 则跳过 RAA。

        Returns:
            total_loss  : 标量 Tensor。
            loss_items  : [lbox, lobj, lcls, lraa] 用于日志打印。
        """
        # 1. 标准检测损失
        det_loss, det_items = self.base_loss(preds, targets)

        # 2. RAA 辅助损失
        raa_loss_val = torch.tensor(0.0, device=det_loss.device)
        if intermediate_features is not None and self.raa.training:
            enhanced, raa_info = self.raa(
                intermediate_features, targets, img_size=self.img_size
            )
            raa_loss_val, _ = self.raa_loss(raa_info, targets)

        total = det_loss + raa_loss_val
        loss_items = torch.cat([det_items, raa_loss_val.unsqueeze(0).detach()])
        return total, loss_items


# ==========================================================================
# 特征图 Hook 工具
# ==========================================================================

class FeatureHook:
    """注册 forward hook，缓存指定层的输出张量。

    用于在不修改模型结构的情况下获取 Neck 输出的 P3/P4/P5 特征图。

    Args:
        module : nn.Module，要挂载 hook 的层。
    """

    def __init__(self, module: nn.Module):
        self.output = None
        self._hook  = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.output = output

    def remove(self):
        self._hook.remove()


# ==========================================================================
# 主训练函数
# ==========================================================================

def train(opt, device):
    """RAA 集成训练主函数。

    Args:
        opt    : argparse.Namespace，命令行参数。
        device : torch.device。
    """
    if not YOLOV5_AVAILABLE:
        print("[演示] YOLOv5 核心模块未找到，无法启动完整训练。")
        print("[演示] 请先安装 YOLOv5 或将代码复制到对应位置。")
        _demo_raa_forward(device)
        return

    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    # ---- 路径配置 ---------------------------------------------------------
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last    = weights_dir / 'last.pt'
    best    = weights_dir / 'best.pt'

    # ---- 超参数 -----------------------------------------------------------
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    else:
        hyp = opt.hyp

    # ---- 数据配置 ---------------------------------------------------------
    with open(opt.data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    nc     = int(data_dict.get('nc', 80))
    names  = data_dict.get('names', [str(i) for i in range(nc)])

    # ---- 模型 -------------------------------------------------------------
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # ---- RAA 模块 ---------------------------------------------------------
    raa_cfg = hyp.get('raa', {})
    in_ch   = raa_cfg.get('in_channels', [128, 256, 512])
    raa_module = RegionAdaptiveAwareness(
        in_channels_list=in_ch,
        expand_ratio=raa_cfg.get('expand_ratio', 0.5),
        epsilon=raa_cfg.get('epsilon', 1e-6),
    ).to(device)

    loss_cfg = hyp.get('raa_loss', {})
    raa_loss_fn = RAALoss(
        lambda_att=loss_cfg.get('lambda_att', 0.5),
        lambda_consistency=loss_cfg.get('lambda_consistency', 1.0),
        lambda_contrast=loss_cfg.get('lambda_contrast', 1.5),
        margin=loss_cfg.get('margin', 1.0),
    ).to(device)

    print(colorstr('RAA Module: ') + f'{sum(p.numel() for p in raa_module.parameters()):,} parameters')

    # ---- 优化器 -----------------------------------------------------------
    all_params = list(model.parameters()) + list(raa_module.parameters())
    optimizer  = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    optimizer.add_param_group({'params': list(raa_module.parameters()), 'lr': hyp['lr0']})

    # ---- 学习率调度器 -----------------------------------------------------
    lf = lambda x: ((1 - math.cos(x * math.pi / opt.epochs)) / 2) * (hyp['lrf'] - 1) + 1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ---- 断点恢复 ---------------------------------------------------------
    start_epoch, best_fitness = 0, 0.0
    if opt.weights.endswith('.pt'):
        ckpt = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
        print(f'已加载权重: {opt.weights}')

    # ---- EMA --------------------------------------------------------------
    ema = ModelEMA(model)

    # ---- 数据加载器 -------------------------------------------------------
    gs     = max(int(model.stride.max()), 32)
    imgsz  = check_img_size(opt.imgsz, gs, floor=gs * 2)

    train_loader, dataset = create_dataloader(
        data_dict['train'], imgsz, opt.batch_size,
        gs, single_cls=opt.single_cls,
        hyp=hyp, augment=True, cache=opt.cache,
        rect=opt.rect, rank=LOCAL_RANK, workers=opt.workers,
    )
    nb = len(train_loader)

    # ---- 损失函数 ---------------------------------------------------------
    base_loss_fn = ComputeLoss(model)
    compute_loss = ComputeLossWithRAA(
        base_loss_fn, raa_module, raa_loss_fn, img_size=(imgsz, imgsz)
    )

    # ---- Hook：捕获 P3/P4/P5 特征图 ------------------------------------
    # YOLOv5 模型中，Detect 层之前的输出通常在 model.model[-1] 的输入
    # 这里假设最后 3 层 (index -4, -3, -2) 对应 P3/P4/P5 的 upsample 输出
    # 实际索引需根据具体模型结构调整
    detect_layer = model.model[-1]
    hooks = []
    feature_outputs = [None, None, None]

    def make_hook(idx):
        def hook_fn(m, inp, out):
            feature_outputs[idx] = out
        return hook_fn

    # Hook Neck 输出（model.model[-2] 通常是最后一个 concat 之后的 Conv）
    # 此处为示意，具体层索引需与你的模型结构对齐
    # 建议通过 print(model) 查看各层名称后调整
    for i, layer_idx in enumerate([-4, -3, -2]):
        h = model.model[layer_idx].register_forward_hook(make_hook(i))
        hooks.append(h)

    # ---- 混合精度 ---------------------------------------------------------
    scaler = amp.GradScaler(enabled=device.type != 'cpu')

    # ============================================================
    # 训练循环
    # ============================================================
    print(f'\n{"Epoch":>10}{"GPU_mem":>10}{"box":>10}{"obj":>10}{"cls":>10}{"raa":>10}{"labels":>10}{"img_sz":>10}')
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        raa_module.train()

        pbar = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()

        mloss = torch.zeros(4, device=device)  # mean losses [box, obj, cls, raa]

        for i, (imgs, targets, paths, _) in enumerate(pbar):
            ni = i + nb * epoch           # 全局批次序号
            imgs    = imgs.to(device, non_blocking=True).float() / 255

            # ---- 前向传播 ------------------------------------------------
            with amp.autocast(enabled=device.type != 'cpu'):
                preds = model(imgs)

                # 从 hook 获取 P3/P4/P5 特征图
                intermediate = [fo for fo in feature_outputs if fo is not None]
                if len(intermediate) < 3:
                    intermediate = None   # hook 未触发则跳过 RAA 损失

                loss, loss_items = compute_loss(preds, targets.to(device), intermediate)

            # ---- 反向传播 ------------------------------------------------
            scaler.scale(loss).backward()

            if ni % opt.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # ---- 日志 ----------------------------------------------------
            mloss = (mloss * i + loss_items) / (i + 1)
            mem   = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if device.type == 'cuda' else '   -'
            pbar.set_description(
                f'{epoch}/{opt.epochs - 1}  {mem}'
                f'  {mloss[0]:.4g}  {mloss[1]:.4g}  {mloss[2]:.4g}  {mloss[3]:.4g}'
                f'  {targets.shape[0]}  {imgs.shape[-1]}'
            )

        # ---- 学习率步进 --------------------------------------------------
        scheduler.step()

        # ---- 保存模型 ----------------------------------------------------
        ckpt = {
            'epoch':        epoch,
            'best_fitness': best_fitness,
            'model':        deepcopy(de_parallel(model)).half(),
            'raa_module':   deepcopy(raa_module).half(),
            'optimizer':    optimizer.state_dict(),
            'ema':          deepcopy(ema.ema).half(),
        }
        torch.save(ckpt, last)
        torch.save(ckpt, best)   # 简化版：每轮均保存，可加验证后更新
        print(f'[Epoch {epoch}] 已保存模型到 {last}')

    # ---- 清理 hook -------------------------------------------------------
    for h in hooks:
        h.remove()

    print(f'\n训练完成，权重保存于: {weights_dir}')
    strip_optimizer(last)
    strip_optimizer(best)


# ==========================================================================
# 演示函数（无完整 YOLOv5 时运行）
# ==========================================================================

def _demo_raa_forward(device):
    """用随机张量演示 RAA 模块前向传播，验证代码正确性。"""
    print('\n' + '='*60)
    print(' RAA 模块演示（随机输入）')
    print('='*60)

    from models.raa_module import RegionAdaptiveAwareness
    from utils.raa_loss import RAALoss

    B, H, W = 2, 80, 80
    in_ch = [128, 256, 512]

    # 随机特征图 [P3, P4, P5]
    features = [
        torch.randn(B, in_ch[0], H,    W,    device=device),
        torch.randn(B, in_ch[1], H//2, W//2, device=device),
        torch.randn(B, in_ch[2], H//4, W//4, device=device),
    ]

    # 随机目标标注 [img_idx, cls, cx, cy, bw, bh]（归一化）
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.3, 0.3],
        [0, 1, 0.2, 0.3, 0.1, 0.15],
        [1, 0, 0.7, 0.6, 0.2, 0.2],
    ], device=device)

    raa    = RegionAdaptiveAwareness(in_channels_list=in_ch).to(device)
    raa.train()

    enhanced, raa_info = raa(features, targets, img_size=(640, 640))

    print('增强特征图尺寸:')
    for name, feat in zip(('P3', 'P4', 'P5'), enhanced):
        print(f'  {name}: {list(feat.shape)}')

    raa_loss_fn = RAALoss().to(device)
    total_loss, loss_dict = raa_loss_fn(raa_info, targets)

    print('\nRAA 辅助损失:')
    for k, v in loss_dict.items():
        print(f'  {k}: {v.item():.6f}')
    print(f'  总计: {total_loss.item():.6f}')
    print('\n[演示完成] RAA 模块工作正常！')


# ==========================================================================
# 命令行入口
# ==========================================================================

def parse_opt():
    parser = argparse.ArgumentParser(description='YOLOv5 + RAA 训练脚本')

    # 基础参数
    parser.add_argument('--weights',    type=str,   default='yolov5s.pt',            help='预训练权重路径')
    parser.add_argument('--cfg',        type=str,   default='models/yolov5s.yaml',    help='模型配置文件')
    parser.add_argument('--data',       type=str,   default='data/coco128.yaml',      help='数据集配置文件')
    parser.add_argument('--hyp',        type=str,   default='configs/yolov5_raa.yaml',help='超参数配置文件')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch-size', type=int,   default=16,  dest='batch_size')
    parser.add_argument('--imgsz',      type=int,   default=640)
    parser.add_argument('--device',     type=str,   default='',                       help='cuda device 或 cpu')
    parser.add_argument('--workers',    type=int,   default=8)
    parser.add_argument('--project',    type=str,   default='runs/train')
    parser.add_argument('--name',       type=str,   default='exp_raa')
    parser.add_argument('--optimizer',  type=str,   default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--accumulate', type=int,   default=1,                        help='梯度累积步数')
    parser.add_argument('--cache',      type=str,   default='',    nargs='?', const='ram')
    parser.add_argument('--rect',       action='store_true')
    parser.add_argument('--single-cls', action='store_true', dest='single_cls')
    parser.add_argument('--exist-ok',   action='store_true', dest='exist_ok')

    # RAA 参数
    parser.add_argument('--raa-enabled', action='store_true', dest='raa_enabled',
                        help='启用 RAA 模块（默认通过 hyp 文件控制）')
    parser.add_argument('--raa-lambda-att',         type=float, default=None, dest='raa_lambda_att')
    parser.add_argument('--raa-lambda-consistency', type=float, default=None, dest='raa_lambda_cons')
    parser.add_argument('--raa-lambda-contrast',    type=float, default=None, dest='raa_lambda_cont')
    parser.add_argument('--raa-margin',             type=float, default=None, dest='raa_margin')

    parser.add_argument('--demo', action='store_true', help='仅运行演示模式（不需要完整 YOLOv5）')

    return parser.parse_args()


def main(opt):
    if opt.demo:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _demo_raa_forward(device)
        return

    if not YOLOV5_AVAILABLE:
        print("[提示] 未检测到完整 YOLOv5 环境，自动切换到演示模式。")
        print("       如需完整训练，请先克隆并安装 YOLOv5：")
        print("       git clone https://github.com/ultralytics/yolov5")
        print("       pip install -r yolov5/requirements.txt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _demo_raa_forward(device)
        return

    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt, device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
