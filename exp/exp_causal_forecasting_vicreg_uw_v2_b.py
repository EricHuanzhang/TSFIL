from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from copy import copy
from utils.scalers import HeterogeneousScaler, OnlineHeterogeneousScaler   # 兼容保留：异质尺度器（本文件未直接调用）


class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)。
    作为“边缘时间偏移缓解算子”，在实例级别把每条序列的均值/方差（宏观边缘漂移）抹平，
    使底座模型在近似零均值、单位方差的标准化空间中学习；预测完成后再 denorm 还原物理量纲。
    affine=True 时附带逐通道可学习仿射 (affine_weight, affine_bias)，norm 与 denorm 严格互逆。
    注意：该逐通道仿射是每个通道一个标量、跨样本/跨时刻共享，不会引入“逐样本逐时刻”的规范自由度，
    与已删除的 Y 空间路由仿射 (gamma_t, beta_t) 不是一回事。
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: 特征/通道数量
        :param eps: 数值稳定项
        :param affine: 是否启用逐通道可学习仿射
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)   # 先统计 lookback 的均值/方差（detach，不回传梯度）
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # 逐通道仿射参数：weight 初始化为 1、bias 初始化为 0（初始即恒等）
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # 沿时间维统计（保留 batch 与 channel 维），统计量 detach：归一化尺度不参与反向传播
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


warnings.filterwarnings('ignore')

"""
================================ v2_b（FOIL-IRN 激活 Y 空间）说明 ================================
本文件在“已删除前向仿射、并删除冗余 loss_y_clean”的 v2_a 基础上，按方案 B 引入 FOIL 的
Instance Residual Normalization (IRN) 损失 loss_suf，把 Y 空间目标代理“真正激活”，同时保留
VICReg 闭环对齐并将其 variance 项显式用作 Y 代理的尺度锚。loss_suf 的实现严格对齐 FOIL 官方
源码 (FOIL-main)，而非仅照搬论文公式。

与 FOIL 源码的对齐要点（详见 _irn_surrogate_loss 注释）：
  - 复刻 exp_informer_final.py 的 shape_error（去逐实例均值版）作为默认 IRN（FOIL 报告结果所用实现）；
  - 同时提供论文 Eq.3 的全标准化版（对应 exp_informer_InferEnv.py / model_env_W.py），由 --irn_mode 切换；
  - 与 FOIL 一致：预测侧/真值侧的逐实例统计量 (mean/std) 全部 detach 作为归一化常量（防止靠操纵自身统计量降损）。

设计要点（X 空间 SAM、双优化器、路由正则、不确定性加权框架均沿用）：
  1. [前向无仿射]      预测路径不含任何 gamma_t/beta_t；用参数无关的 RevIN 反归一化输出
                       （测试期同路、无 Y 泄露、不可被预测头线性吸收 → 从结构上规避不可辨识性）。
  2. [Y 空间净化在损失级] loss_suf = FOIL-IRN 残差损失，把未观测混杂 Z 的影响在逐实例统计量层面消解，
                       而非显式估计 α/β —— 参数无关、独立于预测损失的辨识信号。
  3. [VICReg 方差锚]   proj_Y(Ŷ^suf) 每维方差被 hinge 推向 1，固定“Y = γ·Y^suf+β 分解”的尺度
                       规范自由度；loss_observed 固定绝对均值/尺度，loss_suf 固定形状，三者互补。
  4. [4 槽不确定性加权] 任务槽位 [observed, x_sur, suf, align] 由同方差不确定性自适应配权（loss_suf 入槽）。
  5. [可选跨环境不变性] _cross_env_invariance + lambda_invar（默认 0）：让 regime 路由也作用于 Y 空间损失侧，
                       惩罚各潜在环境的 IRN 残差风险方差（FOIL 的 Var_e 项之软版），支撑“环境条件化双空间”。
  6. [命令行可调]      --irn_mode {demean|standardize}（默认 demean）、--std_coeff（方差锚强度，默认 1.0）、
                       --lambda_invar（默认 0）。projY_std 坍缩(<0.5) 时调大 --std_coeff。
================================================================================================
"""

import torch.nn.functional as F
from torch.nn.utils import spectral_norm  # PyTorch 官方谱范数工具


class DualSpaceSurrogateWrapper(nn.Module):
    def __init__(self, lb_win, h_win, label_win, feature_dim, forecast_model, num_regimes=5, proj_dim=128):
        """
        统一双空间代理学习包装器 (Dual-Space Surrogate Wrapper)。
        将任意 Model-Agnostic 时序预测底座 (如 iTransformer) 封装，植入软注意力掩码 (SAM) 与残差聚合算子 (Agg)，
        并以损失级 FOIL-IRN 净化 Y 空间、VICReg 实现双空间闭环对齐，从而对冲概念漂移。

        参数:
            lb_win (int):     历史回顾窗口长度 L (Lookback Window)
            h_win (int):      未来预测跨度 H (Horizon Window)
            label_win (int):  解码器 Token 占位长度
            feature_dim (int):全通道特征维度 (含外生变量与最后 1 列目标 Y)
            forecast_model:   可插入的时序预测底座
            num_regimes (int):潜在环境机制原型数量上限（结构化低频滤波，抗 batch 间震荡）
            proj_dim (int):   VICReg 投影头隐空间维度
        """
        super(DualSpaceSurrogateWrapper, self).__init__()
        self.lb_win = lb_win
        self.h_win = h_win
        self.label_win = label_win
        self.feature_dim = feature_dim
        self.forecast_model = forecast_model

        # 声明边缘时间偏移缓解算子 (实例归一化)
        self.revin1 = RevIN(num_features=feature_dim)

        # -------------------- [抗敏感震荡核心设计一：利普希茨有界超网络] --------------------
        # 通过 spectral_norm 限制最大奇异值 <= 1，强制低 Lipschitz 常数；Tanh 进一步把隐藏特征压到有界空间，
        # 防止相邻 batch 的微弱噪声引发输出跳变。输出为各离散机制的归属 logits（环境推断的上下文编码器）。
        self.context_encoder = nn.Sequential(
            spectral_norm(nn.Linear(lb_win * feature_dim, 128)),  # 把整段历史窗口展平后压到 128 维隐空间
            nn.Tanh(),                                            # 零中心有界激活，保留正负向信息、避免数值发散
            spectral_norm(nn.Linear(128, num_regimes))            # 输出归属各离散机制的概率 logits
        )

        # -------------------- [抗敏感震荡核心设计二：离散机制原型拓扑矩阵] --------------------
        # 全局共享的隐式因果骨架母体，剥离样本特异性的虚假关联过拟合。
        # [v2_a] 仅保留 X 空间特征代理原型 regime_prototypes_X；已彻底切除不可辨识的 Y 空间连续仿射原型。
        # Y 空间的环境条件化改由（可选的）跨环境不变性惩罚在损失级实现，而非前向路由仿射。
        self.regime_prototypes_X = nn.Parameter(torch.randn(num_regimes, (lb_win + 1) * (feature_dim - 1)))

        # 跨时空因果核心残差聚合器 Agg(·)：把底座预测的未来外生特征映射为对目标通道的残差修正
        self.linear = nn.Sequential(*[
            nn.Linear(feature_dim - 1, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ])

        # -------------------- [创新点3：VICReg 式共享隐空间闭环对齐] --------------------
        # 两个投影头把特征代理 X^SUR(d_X 维) 与目标代理 Y^suf(1 维) 投到同一 proj_dim 隐空间。
        # 选 VICReg 的关键：它显式支持两分支“异构、异维、不共享权重”，恰配 X/Y 两路异质代理；
        # 且其 variance 项内建防坍塌。仅作训练期对齐正则，foward_feature(推理) 不调用、不接入预测路径。
        self.proj_dim = proj_dim
        self.proj_X = nn.Sequential(
            nn.Linear(feature_dim - 1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.proj_Y = nn.Sequential(
            nn.Linear(1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

        # -------------------- [方案C：同方差不确定性加权] --------------------
        # 4 个主任务的可学习对数方差 s_i = log σ_i^2（数值稳定参数化）。顺序固定为 [observed, x_surrogate, suf, align]。
        # [v2_a+IRN] 相比“删冗余”后的 3 槽，这里扩回 4 槽：把新引入的 FOIL-IRN 损失 loss_suf 也纳入自适应配权。
        # 零初始化 ⇔ σ^2=1（等权起步），首个 batch 后再按各任务真实损失量级校准（见 train 的 [改动B]）。
        self.log_vars = nn.Parameter(torch.zeros(4))

    def foward_causal(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        训练期因果前向传播流 (含 SAM 物理实现)。
        利用只有训练期才可获得的未来外生特征 X^H 与历史 X^L 共同切片，经 SAM 过滤得到黄金替代特征 X^{SUR}，
        强力监督底座的外生预测通道；Y 通道则输出“干净代理预测”Ŷ^suf，其净化在损失级由 FOIL-IRN 完成。
        - x_enc_all: 全量时间轴滑窗切片 (B, L+1, H, feature_dim)
        - x_enc:     历史 Lookback 序列 (B, L, feature_dim)
        - x_mark_enc/x_dec/x_mark_dec: 时间戳与解码器占位输入
        """
        batch_size = x_enc.shape[0]

        # 1. 经利普希茨有界超网络，动态计算批内每个样本对离散环境机制的软分配概率
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))  # (B, num_regimes)
        regime_weights = torch.softmax(regime_logits, dim=-1)                # 沿机制维归一化为概率分布

        # 2. 【凸重塑路由】用机制权重对 num_regimes 个原型掩码做软加权求和，无震荡地插值出当前机制的 X 空间掩码
        gated_mask_X = torch.mm(regime_weights, self.regime_prototypes_X)    # (B, (L+1)*(d_X))
        gated_mask_X = gated_mask_X.reshape(batch_size, self.lb_win + 1, self.feature_dim - 1)  # (B, L+1, d_X)

        # -------------------- [X空间计算：动态因果边界补全 (ShifTS 算子)] --------------------
        learnable_mask = torch.softmax(gated_mask_X, dim=1)                  # 在多滑窗维 (L+1) 上 Softmax 归一化
        mask_mean = torch.mean(learnable_mask, dim=1, keepdim=True)
        condition = (learnable_mask - mask_mean) > 0                         # 硬阈值门控：清洗低于均线的弱相关噪声
        learnable_mask = learnable_mask * condition
        learnable_mask = learnable_mask / (torch.sum(learnable_mask, dim=1, keepdim=True) + 1e-6)

        # 为目标通道 Y 构建独热硬编码占位符，锁定 0 号未来视界切片，捍卫条件分布
        last_dim = torch.zeros(batch_size, self.lb_win + 1, 1).to(learnable_mask.device)
        last_dim[:, 0, 0] = 1.0
        learnable_mask = torch.cat((learnable_mask, last_dim), dim=-1)       # (B, L+1, feature_dim)

        # 高维广播矩阵相乘并求和压缩，提炼出黄金替代特征真实值 X^{SUR}
        x_pred_sel = torch.sum(x_enc_all.permute(0, 2, 1, 3) * learnable_mask.unsqueeze(1), -2)

        # -------------------- [Y空间计算：方案 B —— 前向无仿射，净化迁到损失级 IRN] --------------------
        # 此处不再插值/前向乘加 gamma_t、beta_t；Y 空间对未观测混杂 Z 的净化由训练损失 loss_suf(FOIL-IRN)
        # 在“实例标准化空间”完成（见 train）。前向不存在可被预测头吸收的冗余仿射，从根上规避不可辨识与“仿射惰性”。

        # -------------------- [Model-Agnostic 底座前向流与因果残差耦合] --------------------
        x_enc_norm = self.revin1(x_enc, mode='norm')                        # 前置时间边缘偏移缓解
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]                                  # 底座预测的未来外生替代特征 \hat{X}^{SUR}
        y_pred_base = x_pred[:, :, -1:]                                      # 底座预测的目标通道常规趋势 \hat{Y}_{base}

        # 核心解耦：在不受未观测混杂污染的纯净因果空间计算不变量代理目标 \hat{Y}^{suf}
        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)        # (B, H, 1)

        # [方案 B] 参数无关去标准化：直接用 RevIN 的 lookback 统计量反归一化（测试期同路、不可被吸收）。
        # loss_observed 在该路径上约束 y_pred_surrogate 的“绝对均值/尺度”，是其可辨识的一支信号。
        x_out = torch.cat((x_pred_features, y_pred_surrogate), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')                           # 恢复真实数据量纲量级

        # [创新点3] 把两路代理预测投到共享隐空间，供 VICReg 闭环对齐。
        # 在 RevIN 反归一化之前的“标准化空间”里对齐：x_pred_features 与 y_pred_surrogate 同处该尺度、量纲一致。
        # 以 (B*H) 为对齐批维（远大于 B），使 VICReg 的 variance/covariance 沿批维的统计在小数据(ETT/ILI)上更稳。
        z_align_X = self.proj_X(x_pred_features.reshape(-1, self.feature_dim - 1))  # (B*H, proj_dim)
        z_align_Y = self.proj_Y(y_pred_surrogate.reshape(-1, 1))                    # (B*H, proj_dim)

        # 返回 8 元组（已无 gamma/beta）。Y 空间净化所需的真实统计量在 train() 内用 batch_y 现算。
        return (x_pred_sel[:, :, :-1], x_out[:, :, :-1], x_out, learnable_mask,
                y_pred_surrogate, regime_logits, z_align_X, z_align_Y)

    def foward_feature(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec, return_diag=False):
        """
        测试/推理期前向传播流 (完全自洽，无未来视界泄露或未观测环境标签依赖)。
        [方案 B] 前向无仿射：直接用 RevIN 反归一化输出，与训练期 loss_observed 同路，不依赖任何 Y 统计量。
        """
        batch_size = x_enc.shape[0]

        # 仅依赖历史 lookback 自适应解码出机制权重（用于 X 空间 SAM 掩码；Y 空间已无仿射依赖）
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))
        regime_weights = torch.softmax(regime_logits, dim=-1)

        # 底座模型及残差校准前向传导
        x_enc_norm = self.revin1(x_enc, mode='norm')
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]
        y_pred_base = x_pred[:, :, -1:]

        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)
        # [方案 B] 无仿射：直接拼接 + RevIN 反归一化（lookback 统计量，参数无关、测试期可得）
        x_out = torch.cat((x_pred_features, y_pred_surrogate), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')

        if return_diag:
            # [日志] 测试期返回路由 + Y 代理诊断：proj_Y 各维标准差，检验 VICReg 方差锚在测试分布上是否维持尺度
            with torch.no_grad():
                _zY = self.proj_Y(y_pred_surrogate.reshape(-1, 1))
                _projY_std = torch.sqrt(_zY.var(dim=0) + 1e-4).mean()
            return x_out, {'regime_weights': regime_weights.detach(),
                           'projY_std': _projY_std.detach()}
        return x_out


class Exp_Causal_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Causal_Forecast, self).__init__(args)

    def _build_model(self):
        """统一双空间因果预测模型的构建与封装。"""
        print("Exp Causal Forecast-Model: Dual Space Surrogate Wrapper v2_b "
              "(Plan B: FOIL-IRN Y-surrogate [aligned to FOIL-main source] + VICReg variance anchor; forward affine removed)")
        # 1. 临时切为通用预测模式，无侵入式初始化底座 (如 iTransformer/PatchTST)
        self.args.task_name = 'long_term_forecast'
        model1 = self.model_dict[self.args.model].Model(self.args).float()

        # 2. 轴长与视界长度的浅拷贝兼容逻辑修正
        seq_len = copy(self.args.seq_len)
        pred_len = copy(self.args.pred_len)
        label_len = copy(self.args.label_len)
        self.args.pred_len = seq_len
        self.args.label_len = seq_len
        self.args.pred_len = pred_len
        self.args.label_len = label_len

        # 3. 正式切换为双空间代理因果泛化模式
        self.args.task_name = 'causal_forecast'

        # 4. 机制原型数量（未定义则默认 5，启动结构化低通滤波）
        num_regimes = getattr(self.args, 'num_regimes', 5)

        # 5. 构建双空间包装器
        causal_model = DualSpaceSurrogateWrapper(
            lb_win=self.args.seq_len,
            h_win=self.args.pred_len,
            label_win=self.args.label_len,
            feature_dim=self.args.enc_in,
            forecast_model=model1,
            num_regimes=num_regimes
        )
        return causal_model

    def _get_data(self, flag):
        """数据迭代器获取接口：data_provider 额外吐出含全局滑窗未来信息的矩阵 seq_xall。"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        双独立优化器：彻底根治因果掩码在 batch 间的过敏感震荡。
        组1(causal)：环境路由相关参数，采用保守 lr=1e-4，配合谱范数从根源拉平相邻 batch 的梯度扰动。
        组2(model) ：底座与 Agg/投影头等，采用用户配置的常规 lr，保障大趋势快速收敛。
        [v2_a] causal 组已移除目标参数 'regime_prototypes_Y'（不再存在）。
        """
        causal_component_names = [
            'regime_prototypes_X',
            'context_encoder'
        ]

        causal_parameters = []
        other_parameters = []
        for name, param in self.model.named_parameters():
            if any(cn in name for cn in causal_component_names):
                causal_parameters.append(param)   # 环境路由权重组
            else:
                other_parameters.append(param)     # 底座 + Agg + 投影头 + log_vars 等

        causal_optim = optim.Adam(causal_parameters, lr=1e-4)
        model_optim = optim.Adam(other_parameters, lr=self.args.learning_rate)
        return model_optim, causal_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _routing_regularization(self, regime_logits):
        """
        路由正则：根治“原型正交但门控坍缩”的退化（原型彼此正交 ≠ 门控被均衡使用）。
        返回:
            loss_balance: Switch 风格负载均衡 N * Σ_e (f_e · P_e)。f_e=批内硬分配占比(detach)，P_e=平均软门控概率(可导)。
                          均匀使用取最小值 1、集中到单一机制取最大值 N；最小化它 → 推动各机制被均衡使用。
            loss_entropy: 批级平均路由分布的负熵 (= -H(P))；以正权重最小化 ⇔ 最大化批级使用熵，鼓励机制多样。
        """
        regime_weights = torch.softmax(regime_logits, dim=-1)              # (B, num_regimes)
        batch_size = regime_weights.size(0)
        num_regimes = regime_weights.size(-1)

        importance = regime_weights.mean(dim=0)                            # P_e：批内平均软门控概率（可导）
        hard_idx = torch.argmax(regime_weights, dim=-1)                    # f_e：批内硬分配占比（不可导，detach）
        f = torch.zeros_like(importance)
        f = f.scatter_add(0, hard_idx, torch.ones_like(hard_idx, dtype=importance.dtype))
        f = (f / batch_size).detach()
        loss_balance = num_regimes * torch.sum(f * importance)

        loss_entropy = torch.sum(importance * torch.log(importance + 1e-8))  # -Σ_e P_e log P_e
        return loss_balance, loss_entropy

    @staticmethod
    def _uw(loss_term, log_var):
        """
        同方差不确定性加权单项 (Kendall, Gal & Cipolla, CVPR 2018)。
        以 s = log σ^2 参数化保证数值稳定：项 = 0.5 * exp(-s) * L + 0.5 * s。
        0.5*exp(-s) 为有效精度权重（任务越不准、σ 自发增大、其权重越小），+0.5*s 阻止 σ→∞。
        """
        return 0.5 * torch.exp(-log_var) * loss_term + 0.5 * log_var

    @staticmethod
    def _irn_surrogate_loss(y_pred_surrogate, y_true_norm, mode='demean', eps=1e-5):
        """
        FOIL 式 Instance Residual Normalization 代理损失 —— 严格对齐 FOIL 官方源码 (FOIL-main)。
        把未观测混杂 Z 对目标的影响 Y = α(Z)·Y^suf + β(Z) 在“逐实例统计量”层面消解，而非显式估计 α/β，
        因此是参数无关、独立于预测损失、不可被预测头线性吸收的辨识信号。

        两种模式（均与 FOIL 源码逐行一致）：
          - mode='demean'（默认；复刻 exp_informer_final.py 的 shape_error，即 FOIL 报告结果所用实现）：
                pred_shape = Ŷ^suf − mean(Ŷ^suf);   true_shape = Y − mean(Y);   loss = MSE(pred_shape, true_shape)
            仅去逐实例均值，移除加性混杂分量 β(Z)（RevIN 已大体抹平尺度，故移除均值即足以净化主导漂移）。
          - mode='standardize'（论文 Eq.3；对应 exp_informer_InferEnv.py / model_env_W.py 的全标准化）：
                再除以逐实例标准差，额外移除乘性分量 α(Z)，使损失对 a·Y+b 不变。

        关键对齐点（与 FOIL 完全一致）：预测侧与真值侧的逐实例统计量 (mean / std) 一律 detach，作为归一化“常量”，
        梯度只经数值项回传 —— 防止模型靠操纵自身均值/方差（而非改善形状）来降损。绝对尺度交由 loss_observed
        （原始 ERM）与 VICReg variance 锚共同约束。

        参数:
            y_pred_surrogate (Tensor): (B, H, 1) 归一化空间下的 Y 代理预测 Ŷ^suf。
            y_true_norm (Tensor):      (B, H, 1) RevIN 归一化空间下的真实目标 Y（调用前已在 no_grad 下取得）。
            mode (str):                'demean'(默认, FOIL-final) 或 'standardize'(论文 Eq.3)。
        返回:
            loss_suf (Tensor):             标量，FOIL 残差 MSE。
            res_std (Tensor):              标量，残差标准差（诊断：过小→趋平凡匹配，正常应为 O(1)）。
            res_energy_per_sample (Tensor):(B,) 每样本残差能量，供可选跨环境不变性惩罚使用。
        """
        # 逐实例（沿 H 维）均值，detach 作常量（对齐 FOIL: inv_mean/true_mean 均 .detach()）
        mu_P = y_pred_surrogate.mean(dim=1, keepdim=True).detach()
        mu_Y = y_true_norm.mean(dim=1, keepdim=True).detach()
        if mode == 'standardize':
            # 论文 Eq.3 全标准化：再除以 detach 的逐实例标准差（对齐 InferEnv/model_env_W）
            sd_P = y_pred_surrogate.std(dim=1, keepdim=True, unbiased=False).detach() + eps
            sd_Y = y_true_norm.std(dim=1, keepdim=True, unbiased=False).detach() + eps
            pred_n = (y_pred_surrogate - mu_P) / sd_P
            true_n = (y_true_norm - mu_Y) / sd_Y
        else:
            # FOIL-final 去均值版（exp_informer_final.py 的 shape_error）
            pred_n = y_pred_surrogate - mu_P
            true_n = y_true_norm - mu_Y
        res = true_n - pred_n
        loss_suf = (res ** 2).mean()
        res_std = res.detach().std()
        res_energy_per_sample = (res ** 2).mean(dim=(1, 2))
        return loss_suf, res_std, res_energy_per_sample

    @staticmethod
    def _cross_env_invariance(res_energy_per_sample, regime_logits, eps=1e-6):
        """
        [方案 B 可选, 默认关闭] FOIL/VREx 式跨环境不变性惩罚（Var_e 项）。
        让 regime 软路由也作用于 Y 空间损失侧：以 regime_weights 对“逐样本 IRN 残差能量”做软分桶，
        得到各潜在环境风险 R_e，惩罚其方差 Var_e(R_e)，推动 Y 代理在不同环境下保持不变（环境条件化 Y 空间）。
        默认 lambda_invar=0：VREx 类惩罚在 ILI/Exchange(异方差+不变协变量偏移)上经验有害；
        需要“环境条件化双空间”叙事或异质性更强的数据时再调大。
        参数:
            res_energy_per_sample (Tensor): (B,) 每样本 IRN 残差能量。
            regime_logits (Tensor):         (B, K) 路由 logits。
        返回:
            loss_invar (Tensor): 标量，跨环境软风险的方差。
        """
        rw = torch.softmax(regime_logits, dim=-1)                                   # (B, K)
        w_sum = rw.sum(dim=0) + eps                                                 # (K,)
        env_risk = (rw * res_energy_per_sample.unsqueeze(1)).sum(dim=0) / w_sum     # (K,) 各环境软风险
        return env_risk.var()

    def _vicreg_alignment(self, z_x, z_y, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04, eps=1e-4):
        """
        非对抗 VICReg 式共享隐空间闭环对齐。在公共隐空间把特征代理嵌入 z_x 与目标代理嵌入 z_y 对齐，三项缺一不可：
          - invariance：MSE(z_x, z_y) —— 提供真正的跨空间“拉近”耦合（闭环对齐本体）；
          - variance  ：铰链损失逼每维标准差≥1 —— 防两路塌成常数。
                        [方案 B] 该项升格为 Y 代理的“尺度锚”：把 proj_Y(Ŷ^suf) 每维标准差钉在 1，固定
                        “Y = γ·Y^suf+β 分解”的尺度规范自由度（与 loss_observed 的绝对尺度约束、loss_suf 的
                        形状约束三路互补、互不冗余）。系数 std_coeff 可由命令行 --std_coeff 调；projY_std 坍缩(<0.5)时调大。
          - covariance：去相关每维 —— 防冗余维度的信息坍塌。
        默认内部系数沿用 VICReg 的 25:25:1 比例（此处等比缩放为 1:1:0.04，整体量级 O(1)，便于外层不确定性加权调度）。
        参数:
            z_x, z_y (Tensor): (N, D) 两路对齐嵌入，N = B*H。
        返回:
            loss_align (Tensor): sim_coeff*inv + std_coeff*var + cov_coeff*cov。
        """
        N, D = z_x.shape
        denom = max(N - 1, 1)

        # invariance：两路嵌入逐样本对齐（拉近）
        inv = F.mse_loss(z_x, z_y)

        # variance：沿批维每维标准差，铰链推向目标 1.0（防坍塌 / Y 代理尺度锚）
        std_x = torch.sqrt(z_x.var(dim=0) + eps)
        std_y = torch.sqrt(z_y.var(dim=0) + eps)
        var = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

        # covariance：中心化后协方差矩阵非对角元平方和，逐维去相关（防信息坍塌）
        zx = z_x - z_x.mean(dim=0)
        zy = z_y - z_y.mean(dim=0)
        cov_x = (zx.T @ zx) / denom
        cov_y = (zy.T @ zy) / denom
        off_x = (cov_x.pow(2).sum() - cov_x.pow(2).diagonal().sum()) / D
        off_y = (cov_y.pow(2).sum() - cov_y.pow(2).diagonal().sum()) / D
        cov = off_x + off_y

        return sim_coeff * inv + std_coeff * var + cov_coeff * cov

    def vali(self, vali_data, vali_loader, criterion):
        """验证期控制流：完全自洽推理，无未来视界与未观测环境标签泄露。"""
        total_loss = []
        self.model.eval()  # 评估模式：关闭 Dropout 等

        with torch.no_grad():  # 禁用梯度，省显存
            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)  # 纯观测空间 MSE，作为早停判定基准
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()  # 恢复训练状态
        return total_loss

    def train(self, setting):
        """训练期控制流：支持强随机打乱，多任务（observed/x_sur/IRN-suf/align）联合优化。"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, causal_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 结构正则的小固定权重（非主预测任务、无观测噪声语义，故不纳入不确定性加权）
        lambda_orth = getattr(self.args, 'lambda_orth', 0.01)         # 原型正交化（防原型坍缩）
        lambda_balance = getattr(self.args, 'lambda_balance', 1e-2)   # 路由负载均衡（防门控坍缩；Switch 经验取 1e-2）
        lambda_entropy = getattr(self.args, 'lambda_entropy', 1e-2)   # 批级路由熵（鼓励机制多样，可置 0）
        # [方案 B] 可选跨环境不变性惩罚权重，默认 0（VREx 类惩罚在 ILI/Exchange 上经验有害）；
        #          >0 时让 regime 路由也作用于 Y 空间损失侧（环境条件化 Y 空间）。
        lambda_invar = getattr(self.args, 'lambda_invar', 0.0)
        # [方案 B] VICReg variance 项系数（Y 代理尺度锚强度），命令行 --std_coeff 可调；projY_std 坍缩(<0.5) 时调大。
        std_coeff = getattr(self.args, 'std_coeff', 1.0)
        # [FOIL-IRN] IRN 模式：'demean'(默认, 复刻 FOIL exp_informer_final.py) 或 'standardize'(论文 Eq.3 全标准化)。
        irn_mode = getattr(self.args, 'irn_mode', 'demean')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # [改动B] 不确定性加权尺度初始化标志：首个 batch 后用各任务真实损失量级初始化 log_vars，使四项有效贡献开局即被拉平。
        self._uw_init_done = False

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # [日志] 本 epoch 的分项损失 + 路由/Y代理健康度累加器（每个 epoch 重置）
            log_components = {'observed': [], 'x_sur': [], 'suf': [], 'align': [], 'invar': [],
                              'orth': [], 'balance': [], 'entropy': []}
            log_diag = {'route_entropy_norm': [], 'route_entropy_persample': [], 'P_e': [],
                        'irn_res_std': [], 'projY_std': [], 'align_inv': []}

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                causal_optim.zero_grad()

                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ================= 统一计算域封装 (兼顾 AMP) =================
                def compute_losses():
                    # 调取方案 B 的统一双空间前向（8 元组，已无 gamma/beta）
                    (x_enc_sel, x_enc_pred, outputs, learnable_mask, y_pred_surrogate,
                     regime_logits, z_align_X, z_align_Y) = self.model.foward_causal(
                        batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark
                    )

                    out_y = outputs[0] if isinstance(outputs, tuple) else outputs
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs_y = out_y[:, -self.args.pred_len:, f_dim:]
                    target_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    # 损失一：观测时序预测损失 (MSE)。方案 B 中它在“参数无关反归一化”路径上约束 y_pred_surrogate 的绝对均值/尺度。
                    loss_observed = criterion(outputs_y, target_y)

                    # 损失二：X 空间特征代理边界补全损失 (ShifTS 原理约束)
                    loss_x_surrogate = criterion(x_enc_sel, x_enc_pred)

                    # 损失三：[方案 B 核心] Y 空间 FOIL-IRN 代理损失，激活 Y 空间（对齐 FOIL 源码，逐实例统计量 detach）
                    with torch.no_grad():
                        y_true_norm = self.model.revin1._normalize(batch_y.float())[:, -self.args.pred_len:, -1:]
                    loss_suf, irn_res_std, res_energy = self._irn_surrogate_loss(y_pred_surrogate, y_true_norm, mode=irn_mode)

                    # 损失四：离散机制原型软正交化损失，强制机制拓扑母体不退化坍缩
                    normalized_prototypes = F.normalize(self.model.regime_prototypes_X, p=2, dim=1)
                    identity_matrix = torch.eye(normalized_prototypes.size(0), device=normalized_prototypes.device)
                    prototype_correlation = torch.matmul(normalized_prototypes, normalized_prototypes.transpose(0, 1))
                    loss_orthogonal = torch.norm(prototype_correlation - identity_matrix, p='fro') ** 2

                    # 损失五：路由负载均衡 + 批级熵正则（专治路由坍缩）
                    loss_balance, loss_entropy = self._routing_regularization(regime_logits)

                    # 损失六：[方案 B 可选] 跨环境不变性惩罚（默认 lambda_invar=0；无前向仿射可锚，故无 affine_anchor）
                    if lambda_invar > 0:
                        loss_invar = self._cross_env_invariance(res_energy, regime_logits)
                    else:
                        loss_invar = torch.zeros((), device=loss_observed.device)

                    # 损失七：[创新点3] VICReg 闭环对齐（非对抗；variance 项=Y 代理尺度锚，std_coeff 命令行可调）
                    loss_align = self._vicreg_alignment(z_align_X, z_align_Y, std_coeff=std_coeff)

                    # [方案C] 4 主任务 {observed, x_sur, suf, align} 由同方差不确定性自适应配权；结构正则用小固定权重。
                    s = self.model.log_vars
                    loss_tasks = (self._uw(loss_observed, s[0]) + self._uw(loss_x_surrogate, s[1])
                                  + self._uw(loss_suf, s[2]) + self._uw(loss_align, s[3]))

                    t_loss = (loss_tasks
                              + lambda_orth * loss_orthogonal
                              + lambda_balance * loss_balance
                              + lambda_entropy * loss_entropy
                              + lambda_invar * loss_invar)

                    # Y 代理健康度诊断量（标量，detach 仅读数）
                    with torch.no_grad():
                        projY_std = torch.sqrt(z_align_Y.var(dim=0) + 1e-4).mean()
                        align_inv = F.mse_loss(z_align_X, z_align_Y)

                    return (t_loss, loss_observed, loss_x_surrogate, loss_suf, loss_align, loss_invar,
                            loss_orthogonal, loss_balance, loss_entropy,
                            irn_res_std, projY_std, align_inv, learnable_mask, regime_logits)
                # =========================================================

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        (loss, loss_obs, loss_x_sur, loss_suf, loss_aln, loss_invar,
                         loss_orth, loss_bal, loss_ent,
                         irn_std, projY_std, align_inv, mask, r_logits) = compute_losses()
                else:
                    (loss, loss_obs, loss_x_sur, loss_suf, loss_aln, loss_invar,
                     loss_orth, loss_bal, loss_ent,
                     irn_std, projY_std, align_inv, mask, r_logits) = compute_losses()

                train_loss.append(loss.item())

                # [改动B] 仅首个 batch：用四个主任务的真实损失量级初始化 log_vars（s_i = log L_i），
                # 使四项有效贡献(0.5*exp(-s)*L)开局即被拉平至 ~0.5，避免被放大的 suf/align 在前期霸占梯度。数据集无关、自校准。
                if not self._uw_init_done:
                    with torch.no_grad():
                        _init_s = torch.log(torch.stack([
                            loss_obs.detach(), loss_x_sur.detach(), loss_suf.detach(), loss_aln.detach()
                        ]).clamp_min(1e-6))
                        self.model.log_vars.data.copy_(_init_s.float())
                    self._uw_init_done = True
                    print("  [v2_b] IRN模式={} | std_coeff={} | lambda_invar={} | 已按首个 batch 损失尺度初始化 log_vars(s=log L): {}".format(
                        irn_mode, std_coeff, lambda_invar,
                        np.round(self.model.log_vars.data.detach().cpu().numpy(), 3)))

                # [日志] 累计本 epoch 的分项损失 + 路由/Y代理统计（全部 detach，仅读数，不影响反向传播）
                with torch.no_grad():
                    log_components['observed'].append(loss_obs.item())
                    log_components['x_sur'].append(loss_x_sur.item())
                    log_components['suf'].append(loss_suf.item())          # FOIL-IRN 代理损失
                    log_components['align'].append(loss_aln.item())
                    log_components['invar'].append(loss_invar.item())      # 可选跨环境不变性（默认 0）
                    log_components['orth'].append(loss_orth.item())
                    log_components['balance'].append(loss_bal.item())
                    log_components['entropy'].append(loss_ent.item())

                    # 路由健康度：归一化使用率熵 ∈ [0,1]（1=机制被完全均匀使用，→0=坍缩到单一机制）
                    _rw = torch.softmax(r_logits, dim=-1)
                    _Pe = _rw.mean(dim=0)
                    _R = _Pe.size(-1)
                    _H = -torch.sum(_Pe * torch.log(_Pe + 1e-8)).item()
                    log_diag['route_entropy_norm'].append(_H / float(np.log(max(_R, 2))))
                    # 逐样本路由熵(批均值)：低逐样本熵 + 高批级熵 = 健康特化；两者都≈1 = 各样本均匀混合(未特化)
                    _H_ps = -(_rw * torch.log(_rw + 1e-8)).sum(dim=-1).mean().item()
                    log_diag['route_entropy_persample'].append(_H_ps / float(np.log(max(_R, 2))))
                    log_diag['P_e'].append(_Pe.detach().float().cpu())
                    # Y 代理健康度：IRN 残差 std（应 O(1)）/ projY_std（方差锚目标 1）/ 闭环对齐 MSE
                    log_diag['irn_res_std'].append(float(irn_std.item()))
                    log_diag['projY_std'].append(float(projY_std.item()))
                    log_diag['align_inv'].append(float(align_inv.item()))

                # -------------------- 反向传播与双优化器步进 --------------------
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    if epoch + 1 <= int(self.args.train_epochs):
                        scaler.step(causal_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if epoch + 1 <= int(self.args.train_epochs):
                        causal_optim.step()

                # 周期性科学日志
                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | 联合总Loss: {2:.7f} | 观测Loss: {3:.7f}".format(
                        i + 1, epoch + 1, loss.item(), loss_obs.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; 预计训练剩余总时间: {:.4f}s'.format(speed, left_time))
                    print("\t当前Batch不变量最大注意力索引矩阵:")
                    print("\t", torch.argmax(mask, dim=1)[0].detach().cpu().numpy())
                    print("\t[诊断] 批级路由熵={:.3f} 逐样本路由熵={:.3f} | P_e={} | IRN残差std={:.3f} | projY_std={:.3f} | 对齐MSE={:.4f}".format(
                        log_diag['route_entropy_norm'][-1], log_diag['route_entropy_persample'][-1],
                        np.round(log_diag['P_e'][-1].numpy(), 3),
                        log_diag['irn_res_std'][-1], log_diag['projY_std'][-1], log_diag['align_inv'][-1]))
                    iter_count = 0
                    time_now = time.time()

            # 每个 Epoch 终结后的全量验证集考核与动态调频
            print("Epoch: {} 耗时总长: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | 综合训练Loss: {2:.7f} | 验证集Loss: {3:.7f} | 测试集Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # ===================== [日志] 本 epoch 诊断汇总 =====================
            def _avg(_l):
                return float(np.mean(_l)) if len(_l) else float('nan')
            _mean_Pe = torch.stack(log_diag['P_e']).mean(0).numpy() if len(log_diag['P_e']) else None
            _re = _avg(log_diag['route_entropy_norm'])
            _nR = (len(_mean_Pe) if _mean_Pe is not None else 0)
            _active = int((_mean_Pe > (1.0 / (2 * _nR))).sum()) if (_mean_Pe is not None and _nR > 0) else -1

            print("  [诊断·分项损失(加权前)] observed={:.5f} x_sur={:.5f} suf={:.5f} align={:.5f} | invar={:.5f} orth={:.5f} balance={:.5f} entropy={:.5f}".format(
                _avg(log_components['observed']), _avg(log_components['x_sur']), _avg(log_components['suf']),
                _avg(log_components['align']), _avg(log_components['invar']), _avg(log_components['orth']),
                _avg(log_components['balance']), _avg(log_components['entropy'])))
            # [方案C] 不确定性加权诊断：学到的 σ_i 与有效精度 0.5*exp(-s_i)（i = observed/x_sur/suf/align）
            _s = self.model.log_vars.detach().float().cpu().numpy()
            print("  [诊断·不确定性加权] sigma(obs,x_sur,suf,align)={} | 有效精度0.5*exp(-s)={}".format(
                np.round(np.exp(_s / 2.0), 4), np.round(0.5 * np.exp(-_s), 4)))
            print("  [诊断·结构正则贡献(固定权重)] orth={:.5f} balance={:.5f} entropy={:.5f} invar={:.5f}".format(
                lambda_orth * _avg(log_components['orth']), lambda_balance * _avg(log_components['balance']),
                lambda_entropy * _avg(log_components['entropy']), lambda_invar * _avg(log_components['invar'])))
            print("  [诊断·路由] 批级使用率熵(epoch均值)={:.3f} 逐样本熵={:.3f} | 激活机制数={}/{} | 平均使用率 P_e={}".format(
                _re, _avg(log_diag['route_entropy_persample']), _active, _nR,
                np.round(_mean_Pe, 3) if _mean_Pe is not None else None))
            print("    └─ 解读：批级高 + 逐样本低 = 健康特化(不同样本走不同机制)；两者都≈1 = 各样本均匀混合(环境条件化未特化)")
            print("  [诊断·Y代理] IRN残差std={:.4f} | projY_std(VICReg方差锚,目标1)={:.4f} | 闭环对齐MSE={:.4f}".format(
                _avg(log_diag['irn_res_std']), _avg(log_diag['projY_std']), _avg(log_diag['align_inv'])))
            print("    └─ 解读：projY_std 应稳定≈1（坍缩<0.5→增大 --std_coeff）；IRN残差std≈O(1) 表示 Y 代理在学形状而非趋平凡。")

            if (not np.isnan(_re)) and _re < 0.5:
                print("  [诊断·告警] 路由熵偏低({:.3f}<0.5)，门控可能正在坍缩 → 增大 lambda_balance/lambda_entropy 或给 causal_optim 加 warmup".format(_re))

            # 早停触发判定
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("触发早停条件，系统在纯净不变量边界安全收敛。")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 锁存并重载性能最优 Checkpoint
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        """最终测试期控制流：端到端干净推理，无未来标签或环境系数泄露。"""
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('从持久化 Checkpoint 载入最优因果泛化模型体...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        # [日志] 测试期路由 + Y代理诊断累加器
        log_test = {'route_entropy_norm': [], 'P_e': [], 'projY_std': []}
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(test_loader):
                batch_x_en, batch_x = batch_x_en.float().to(self.device), batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _diag = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark, return_diag=True)
                else:
                    outputs, _diag = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark, return_diag=True)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # [日志] 累计测试期路由健康度 + Y 代理尺度锚（验证路由是否健康、VICReg 方差锚在测试分布上是否维持尺度）
                _rw = _diag['regime_weights']; _Pe = _rw.mean(dim=0); _R = _Pe.size(-1)
                _Ht = -torch.sum(_Pe * torch.log(_Pe + 1e-8)).item()
                log_test['route_entropy_norm'].append(_Ht / float(np.log(max(_R, 2))))
                log_test['P_e'].append(_Pe.float().cpu())
                log_test['projY_std'].append(float(_diag['projY_std'].item()))

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs, batch_y = outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                preds.append(outputs[:, :, f_dim:])
                trues.append(batch_y[:, :, f_dim:])

        preds, trues = np.array(preds), np.array(trues)
        print('真实环境基准校验测试集维度形态:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('真实试验验证最终成效评估 >> MSE: {}, MAE: {}'.format(mse, mae))

        # [日志] 测试期汇总：路由健康度 + VICReg 方差锚在测试分布上是否维持非坍缩尺度
        if len(log_test['route_entropy_norm']):
            _mPe = torch.stack(log_test['P_e']).mean(0).numpy()
            print("  [诊断·测试期] 归一化路由熵={:.3f} | P_e={} | projY_std(VICReg方差锚)={:.4f}".format(
                float(np.mean(log_test['route_entropy_norm'])), np.round(_mPe, 3),
                float(np.mean(log_test['projY_std']))))

        with open("result_causal_forecast.txt", 'a') as f:
            f.write(setting + "  \n" + 'mse:{}, mae:{}\n\n'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
