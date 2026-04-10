"""
RAA Loss Functions for YOLOv5
==============================
包含三种辅助损失：
  1. AttentionSupervisionLoss  —— 注意力监督损失 L_att
  2. MultiScaleConsistencyLoss —— 多尺度一致性损失 L_consistency
  3. TargetBgContrastLoss      —— 目标-背景对比损失 L_tb

以及将三者加权合并的包装类：
  RAALoss —— 总 RAA 辅助损失 = λ_att·L_att + λ_consistency·L_consistency + λ_contrast·L_tb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSupervisionLoss(nn.Module):
    """注意力监督损失。

    G_i = max_j(G_{i,j})   —— 各尺度软掩码逐位置取最大值
    L_att = Σ_{i∈{3,4,5}} BCE(A_i, G_i)

    迫使注意力图聚焦于真实目标区域，防止注意力退化为均匀分布或噪声。

    Args:
        reduction (str): 'mean' 或 'sum'。
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        attentions: dict,   # {scale_name: [B,1,H,W]}
        soft_masks: dict,   # {scale_name: [B,1,H,W]}
    ) -> torch.Tensor:
        """
        Args:
            attentions : RAA 模块输出的注意力图字典。
            soft_masks : RAA 模块生成的软掩码字典（G_i）。

        Returns:
            loss_att (scalar Tensor)
        """
        loss = torch.tensor(0.0, device=next(iter(attentions.values())).device)
        count = 0
        for scale in attentions:
            if scale not in soft_masks:
                continue
            attn = attentions[scale]      # [B, 1, H, W]
            g    = soft_masks[scale]      # [B, 1, H, W]

            # BCE requires inputs in [0, 1]; attn is Sigmoid output so it's fine
            loss = loss + F.binary_cross_entropy(attn, g, reduction=self.reduction)
            count += 1

        return loss / max(count, 1)


class MultiScaleConsistencyLoss(nn.Module):
    """多尺度一致性损失。

    L_consistency = (1/N) Σ_j [ ||z^t_{3,j} - z^t_{4,j}||² + ||z^t_{4,j} - z^t_{5,j}||² ]

    要求同一目标在相邻尺度的语义表征保持一致，
    抑制单一尺度上的偶然高响应区域。

    Args:
        valid_threshold (float): 掩码面积低于此值的目标视为无效（全零掩码），不计入损失。
    """

    def __init__(self, valid_threshold: float = 1e-3):
        super().__init__()
        self.thr = valid_threshold

    def forward(
        self,
        target_pooled: dict,  # {scale_name: [B, N_max, C]}
        targets: torch.Tensor = None,  # [N, 6] 用于判断有效目标数量
    ) -> torch.Tensor:
        """
        Args:
            target_pooled : RAA 模块输出的目标区域池化特征字典。
            targets       : 原始目标标注（用于筛选有效目标，可选）。

        Returns:
            loss_consistency (scalar Tensor)
        """
        scales = ('P3', 'P4', 'P5')
        # 检查所有尺度均有输出
        for s in scales:
            if s not in target_pooled:
                device = next(iter(target_pooled.values())).device
                return torch.tensor(0.0, device=device)

        z3 = target_pooled['P3']   # [B, Nm, C]
        z4 = target_pooled['P4']
        z5 = target_pooled['P5']

        device = z3.device
        loss = torch.tensor(0.0, device=device)

        B, Nm, C = z3.shape
        if Nm == 0:
            return loss

        # 有效性：使用 z3 的范数作为代理（零掩码 → 零向量）
        valid = (z3.norm(dim=-1) > self.thr)   # [B, Nm]

        # 相邻尺度的 MSE
        diff_34 = ((z3 - z4) ** 2).sum(dim=-1)  # [B, Nm]
        diff_45 = ((z4 - z5) ** 2).sum(dim=-1)  # [B, Nm]
        diff = (diff_34 + diff_45) * valid.float()  # mask invalid

        n_valid = valid.float().sum().clamp(min=1.0)
        loss = diff.sum() / n_valid
        return loss


class TargetBgContrastLoss(nn.Module):
    """目标-背景对比损失（基于间隔的特征分离损失）。

    L_tb = (1/(N·S)) Σ_j Σ_{i∈{3,4,5}} max(0, m - d(z^t_{i,j}, z^b_{i,j}))

    其中 d(·,·) 为欧氏距离，m 为分离间隔。
    要求目标区域特征与周边背景特征在表征空间中保持足够距离，
    减少背景相似纹理对目标识别的干扰。

    Args:
        margin            (float): 分离间隔 m，默认 1.0。
        valid_threshold   (float): 掩码面积低于此值的目标视为无效。
    """

    def __init__(self, margin: float = 1.0, valid_threshold: float = 1e-3):
        super().__init__()
        self.margin = margin
        self.thr    = valid_threshold

    def forward(
        self,
        target_pooled: dict,  # {scale_name: [B, Nm, C]}
        bg_pooled: dict,      # {scale_name: [B, Nm, C]}
    ) -> torch.Tensor:
        """
        Args:
            target_pooled : 目标区域池化特征。
            bg_pooled     : 背景区域池化特征。

        Returns:
            loss_contrast (scalar Tensor)
        """
        scales = ('P3', 'P4', 'P5')
        device = next(iter(target_pooled.values())).device
        loss   = torch.tensor(0.0, device=device)
        total_valid = torch.tensor(0.0, device=device)

        for scale in scales:
            if scale not in target_pooled or scale not in bg_pooled:
                continue

            zt = target_pooled[scale]   # [B, Nm, C]
            zb = bg_pooled[scale]       # [B, Nm, C]

            # 有效性判断（目标+背景均非零向量）
            valid = (zt.norm(dim=-1) > self.thr) & (zb.norm(dim=-1) > self.thr)  # [B, Nm]

            # 欧氏距离
            dist = torch.norm(zt - zb, p=2, dim=-1)          # [B, Nm]

            # 间隔损失
            margin_loss = torch.clamp(self.margin - dist, min=0.0)  # [B, Nm]
            margin_loss = margin_loss * valid.float()

            loss        = loss        + margin_loss.sum()
            total_valid = total_valid + valid.float().sum()

        loss = loss / total_valid.clamp(min=1.0)
        return loss


class RAALoss(nn.Module):
    """RAA 辅助损失的加权总和。

    L_raa = λ_att × L_att + λ_consistency × L_consistency + λ_contrast × L_tb

    建议权重:
        λ_att         = 0.5
        λ_consistency = 1.0
        λ_contrast    = 1.5

    Args:
        lambda_att         (float): 注意力监督损失权重。
        lambda_consistency (float): 多尺度一致性损失权重。
        lambda_contrast    (float): 目标-背景对比损失权重。
        margin             (float): 对比损失分离间隔。
    """

    def __init__(
        self,
        lambda_att:         float = 0.5,
        lambda_consistency: float = 1.0,
        lambda_contrast:    float = 1.5,
        margin:             float = 1.0,
    ):
        super().__init__()
        self.lambda_att         = lambda_att
        self.lambda_consistency = lambda_consistency
        self.lambda_contrast    = lambda_contrast

        self.att_loss   = AttentionSupervisionLoss()
        self.cons_loss  = MultiScaleConsistencyLoss()
        self.cont_loss  = TargetBgContrastLoss(margin=margin)

    def forward(self, raa_info: dict, targets: torch.Tensor = None):
        """
        Args:
            raa_info : RegionAdaptiveAwareness.forward() 返回的 raa_info 字典。
                       需包含键: 'attentions', 'soft_masks', 'target_pooled', 'bg_pooled'
            targets  : [N, 6] 原始目标标注（可选，用于 consistency 损失中的有效性判断）。

        Returns:
            total_raa_loss (scalar Tensor)
            loss_dict      (dict) : 各分项损失值（用于日志记录）。
        """
        device = next(iter(raa_info['attentions'].values())).device
        loss_att  = torch.tensor(0.0, device=device)
        loss_cons = torch.tensor(0.0, device=device)
        loss_cont = torch.tensor(0.0, device=device)

        # 注意力监督损失
        if raa_info.get('attentions') and raa_info.get('soft_masks'):
            loss_att = self.att_loss(raa_info['attentions'], raa_info['soft_masks'])

        # 多尺度一致性损失
        if raa_info.get('target_pooled'):
            loss_cons = self.cons_loss(raa_info['target_pooled'], targets)

        # 目标-背景对比损失
        if raa_info.get('target_pooled') and raa_info.get('bg_pooled'):
            loss_cont = self.cont_loss(raa_info['target_pooled'], raa_info['bg_pooled'])

        total = (
            self.lambda_att         * loss_att
            + self.lambda_consistency * loss_cons
            + self.lambda_contrast    * loss_cont
        )

        loss_dict = {
            'loss_att':   loss_att.detach(),
            'loss_cons':  loss_cons.detach(),
            'loss_cont':  loss_cont.detach(),
            'loss_raa':   total.detach(),
        }

        return total, loss_dict
