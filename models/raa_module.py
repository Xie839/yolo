"""
Region Adaptive Awareness (RAA) Module for YOLOv5
==================================================
在Backbone+Neck后对P3、P4、P5三个检测尺度进行目标区域感知：
  - AttentionGenerator  : 轻量卷积生成单通道空间注意力图
  - MaskBuilder         : 基于标注框构建目标掩码与背景掩码
  - MaskedROIPool       : 掩码区域池化（避免零值稀释）
  - RegionAdaptiveAwareness : 整合上述组件的顶层模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class AttentionGenerator(nn.Module):
    """轻量级注意力生成器，输出单通道空间注意力图 A_i ∈ [0,1]。

    网络结构:
        Conv(C→64, 1×1) + ReLU
        Conv(64→32, 3×3, pad=1) + ReLU
        Conv(32→1, 1×1) + Sigmoid

    Args:
        in_channels (int): 输入特征图通道数。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention: [B, 1, H, W]，值域 [0, 1]
        """
        return self.conv(x)


# ---------------------------------------------------------------------------

class MaskedROIPool(nn.Module):
    """掩码区域池化。

    z = Σ(feat ⊙ mask) / (Σ(mask) + ε)

    相比全局平均池化，避免了零值区域对特征向量的稀释。

    Args:
        epsilon (float): 防止分母为零的微小常数。
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.eps = epsilon

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]
            mask: [B, 1, H, W]，值域 [0, 1]
        Returns:
            pooled: [B, C]
        """
        masked = feat * mask                              # [B, C, H, W]
        sum_feat = masked.sum(dim=(2, 3))                 # [B, C]
        sum_mask = mask.sum(dim=(2, 3)) + self.eps        # [B, 1]
        return sum_feat / sum_mask                        # [B, C]


# ---------------------------------------------------------------------------

class MaskBuilder:
    """在特征图尺度上为每张图像中的标注框构建目标掩码与背景掩码。

    目标掩码  M^t : 将标注框映射到特征图并填充为 1。
    背景掩码  M^b : 对目标框按比例 ρ 外扩后去除目标区域及其他目标框区域。

    Args:
        expand_ratio (float): 外扩比例 ρ，默认 0.5。
    """

    def __init__(self, expand_ratio: float = 0.5):
        self.rho = expand_ratio

    # ------------------------------------------------------------------
    def build(
        self,
        targets: torch.Tensor,
        feat_h: int,
        feat_w: int,
        img_h: int,
        img_w: int,
        batch_size: int,
        device: torch.device,
    ):
        """
        Args:
            targets : [N, 6]  每行 = [img_idx, cls, cx, cy, bw, bh]（归一化 YOLO 格式）
            feat_h, feat_w  : 当前尺度特征图的高、宽
            img_h,  img_w   : 原始图像高、宽（通常 640）
            batch_size      : 批次大小
            device          : torch.device

        Returns:
            target_masks : [B, N_max, 1, feat_h, feat_w]
            bg_masks     : [B, N_max, 1, feat_h, feat_w]
            union_masks  : [B, 1, feat_h, feat_w]   —— 所有目标的并集掩码 U_i

        注意：N_max 为该 batch 中目标数量最多的图像所含目标数。
        若某图像目标数不足 N_max，相应掩码填全零（无效目标）。
        """
        # ---- scale factor ------------------------------------------------
        sx = feat_w / img_w
        sy = feat_h / img_h

        # ---- group targets by image index --------------------------------
        groups: list[list] = [[] for _ in range(batch_size)]
        if targets.numel() > 0:
            for t in targets:
                bi = int(t[0].item())
                if 0 <= bi < batch_size:
                    groups[bi].append(t)

        max_objs = max((len(g) for g in groups), default=0)
        if max_objs == 0:
            # 没有目标，返回空掩码
            empty = torch.zeros(batch_size, 1, 1, feat_h, feat_w, device=device)
            union = torch.zeros(batch_size, 1, feat_h, feat_w, device=device)
            return empty, empty, union

        # ---- allocate output tensors -------------------------------------
        t_masks = torch.zeros(batch_size, max_objs, 1, feat_h, feat_w, device=device)
        b_masks = torch.zeros(batch_size, max_objs, 1, feat_h, feat_w, device=device)
        union   = torch.zeros(batch_size, 1, feat_h, feat_w, device=device)

        # ---- fill per-image -----------------------------------------
        for bi, group in enumerate(groups):
            for ji, t in enumerate(group):
                # t = [img_idx, cls, cx, cy, bw, bh] (normalized)
                cx, cy, bw, bh = t[2].item(), t[3].item(), t[4].item(), t[5].item()

                # convert to pixel coords in feature map
                x1 = (cx - bw / 2) * sx
                y1 = (cy - bh / 2) * sy
                x2 = (cx + bw / 2) * sx
                y2 = (cy + bh / 2) * sy

                # clamp to feature map bounds
                # x1i / y1i: inclusive start index → clamp to [0, feat_w-1] / [0, feat_h-1]
                # x2i / y2i: exclusive end index   → clamp to [0, feat_w]   / [0, feat_h]
                x1i = max(0, min(int(x1), feat_w - 1))
                y1i = max(0, min(int(y1), feat_h - 1))
                x2i = max(0, min(math.ceil(x2), feat_w))
                y2i = max(0, min(math.ceil(y2), feat_h))

                if x2i <= x1i or y2i <= y1i:
                    continue  # degenerate box

                # target mask M^t
                t_masks[bi, ji, 0, y1i:y2i, x1i:x2i] = 1.0
                # union mask U_i
                union[bi, 0, y1i:y2i, x1i:x2i] = 1.0

                # expanded box for M^e
                ew = (x2i - x1i) * self.rho / 2
                eh = (y2i - y1i) * self.rho / 2
                ex1 = max(0, int(x1i - ew))
                ey1 = max(0, int(y1i - eh))
                ex2 = min(feat_w, math.ceil(x2i + ew))
                ey2 = min(feat_h, math.ceil(y2i + eh))

                # background mask = M^e - M^t (will subtract U_i later)
                b_masks[bi, ji, 0, ey1:ey2, ex1:ex2] = 1.0
                b_masks[bi, ji, 0, y1i:y2i, x1i:x2i] = 0.0  # remove target

        # Remove all target regions from background masks: M^b = (M^e - M^t) ⊙ (1 - U_i)
        # union: [B, 1, H, W]  →  [B, 1, 1, H, W]  broadcast over max_objs dim
        b_masks = b_masks * (1.0 - union.unsqueeze(1))

        return t_masks, b_masks, union


# ---------------------------------------------------------------------------

class RegionAdaptiveAwareness(nn.Module):
    """区域自适应感知模块（RAA）顶层封装。

    集成 AttentionGenerator、MaskBuilder 和 MaskedROIPool，
    在 P3/P4/P5 三个尺度上完成：
      1. 注意力图生成
      2. 目标/背景掩码构建（训练阶段）
      3. 特征解耦与增强
      4. 掩码区域池化
      5. 增强特征输出（训练和推理均生效）

    由于不同尺度的特征图通道数不同（P3=128, P4=256, P5=512），
    模块内部使用线性投影头将各尺度的池化特征映射到统一的 proj_dim 维度，
    以支持多尺度一致性约束的跨尺度比较。

    Args:
        in_channels_list (list[int]): [P3, P4, P5] 各尺度的输入通道数。
        expand_ratio     (float)    : 背景外扩比例 ρ，默认 0.5。
        epsilon          (float)    : 池化防零常数，默认 1e-6。
        proj_dim         (int)      : 投影后的统一特征维度，默认 128。
    """

    SCALE_NAMES = ('P3', 'P4', 'P5')

    def __init__(
        self,
        in_channels_list: list = None,
        expand_ratio: float = 0.5,
        epsilon: float = 1e-6,
        proj_dim: int = 128,
    ):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [128, 256, 512]

        assert len(in_channels_list) == 3, "需要为 P3/P4/P5 提供 3 个通道数"

        self.attention_generators = nn.ModuleDict({
            name: AttentionGenerator(ch)
            for name, ch in zip(self.SCALE_NAMES, in_channels_list)
        })

        # 线性投影头：将各尺度池化特征投影到统一维度，用于跨尺度比较
        self.proj_heads = nn.ModuleDict({
            name: nn.Linear(ch, proj_dim, bias=False)
            for name, ch in zip(self.SCALE_NAMES, in_channels_list)
        })

        self.mask_builder = MaskBuilder(expand_ratio=expand_ratio)
        self.roi_pool     = MaskedROIPool(epsilon=epsilon)
        self.proj_dim     = proj_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        features: list,
        targets: torch.Tensor = None,
        img_size: tuple = (640, 640),
    ):
        """
        Args:
            features  : [P3, P4, P5] 每项形状 [B, C, H, W]
            targets   : [N, 6]  [img_idx, cls, cx, cy, bw, bh] (归一化)
                        训练阶段传入，推理阶段可省略（传 None）。
            img_size  : (img_h, img_w) 原始图像尺寸，默认 (640, 640)。

        Returns:
            enhanced_features : [P3', P4', P5'] 注意力增强后的特征，形状与输入相同。
            raa_info          : dict，包含以下键（训练阶段）：
                'attentions'       : dict  {scale: [B,1,H,W]}
                'target_pooled'    : dict  {scale: [B,C]}
                'bg_pooled'        : dict  {scale: [B,C]}
                'soft_masks'       : dict  {scale: [B,1,H,W]}  构建的全局软掩码 G_i
        """
        batch_size = features[0].shape[0]
        device     = features[0].device
        img_h, img_w = img_size

        enhanced_features = []
        raa_info = {
            'attentions':    {},
            'target_pooled': {},
            'bg_pooled':     {},
            'soft_masks':    {},
        }

        training = self.training and targets is not None and targets.numel() > 0

        # ---- pre-build masks for all scales if training ------------------
        if training:
            mask_cache = {}
            for name, feat in zip(self.SCALE_NAMES, features):
                fh, fw = feat.shape[2], feat.shape[3]
                t_masks, b_masks, union = self.mask_builder.build(
                    targets, fh, fw, img_h, img_w, batch_size, device
                )
                mask_cache[name] = (t_masks, b_masks, union)

        # ---- per-scale processing ----------------------------------------
        for name, feat in zip(self.SCALE_NAMES, features):
            # 1. 注意力图
            attn = self.attention_generators[name](feat)   # [B, 1, H, W]
            raa_info['attentions'][name] = attn

            if training:
                t_masks, b_masks, union = mask_cache[name]
                num_objs = t_masks.shape[1]                # N_max

                # 2. 软掩码监督信号 G_i = max_j(G_i,j)
                # 这里用目标掩码的逐元素最大值作为 G_i
                # t_masks: [B, N_max, 1, H, W]
                soft_mask = t_masks.max(dim=1).values      # [B, 1, H, W]
                raa_info['soft_masks'][name] = soft_mask

                # 3. 特征解耦
                # P^t_i,j = P_i ⊙ M^t_i,j ⊙ A_i
                # P^b_i,j = P_i ⊙ M^b_i,j
                # feat:    [B, C, H, W]
                # t_masks: [B, N_max, 1, H, W]  →  per-obj掩码池化
                # 聚合每个目标的目标/背景池化向量（取 batch×N_max 视图）

                # flatten batch×N_max 维度便于批量池化
                B, Nm, _, H, W = t_masks.shape
                C = feat.shape[1]

                # expand feat to [B, Nm, C, H, W]
                feat_exp = feat.unsqueeze(1).expand(-1, Nm, -1, -1, -1)

                # target features with attention weighting
                attn_exp = attn.unsqueeze(1).expand(-1, Nm, -1, -1, -1)
                target_feat = feat_exp * t_masks * attn_exp   # [B, Nm, C, H, W]
                bg_feat     = feat_exp * b_masks               # [B, Nm, C, H, W]

                # flatten B×Nm for batch roi pooling: [B*Nm, C, H, W]
                target_feat_flat = target_feat.view(B * Nm, C, H, W)
                bg_feat_flat     = bg_feat.view(B * Nm, C, H, W)
                t_mask_flat      = t_masks.view(B * Nm, 1, H, W)
                b_mask_flat      = b_masks.view(B * Nm, 1, H, W)

                # masked roi pool → [B*Nm, C]
                target_pooled_flat = self.roi_pool(target_feat_flat, t_mask_flat)
                bg_pooled_flat     = self.roi_pool(bg_feat_flat,     b_mask_flat)

                # reshape to [B, Nm, C]
                tp_full = target_pooled_flat.view(B, Nm, C)
                bp_full = bg_pooled_flat.view(B, Nm, C)

                # 投影到统一维度 proj_dim，用于跨尺度比较
                # proj_head: Linear(C → proj_dim)，输入 [B*Nm, C]
                tp_proj = self.proj_heads[name](target_pooled_flat).view(B, Nm, self.proj_dim)
                bp_proj = self.proj_heads[name](bg_pooled_flat).view(B, Nm, self.proj_dim)

                raa_info['target_pooled'][name] = tp_proj
                raa_info['bg_pooled'][name]     = bp_proj

            # 4. 特征增强（推理和训练均生效）: P'_i = P_i ⊙ A_i
            enhanced_features.append(feat * attn)

        return enhanced_features, raa_info
