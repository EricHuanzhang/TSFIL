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
from utils.scalers import HeterogeneousScaler, OnlineHeterogeneousScaler   # 新增导入

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
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
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


warnings.filterwarnings('ignore')
"""
将任意一个底座预测模型（如 `iTransformer`、`PatchTST` 等）进行封装，并植入**软注意力掩码机制（SAM）
以及多层感知机聚合算子（Agg），从而实现对概念漂移的对冲。
"""
import torch.nn.functional as F
from torch.nn.utils import spectral_norm  # 导入 PyTorch 官方谱范数工具
class DualSpaceSurrogateWrapper(nn.Module):
    def __init__(self, lb_win, h_win, label_win, feature_dim, forecast_model, num_regimes=5, proj_dim=128):
        """
        统一双空间代理学习包装器 (Dual-Space Surrogate Wrapper)
        参数:
            lb_win (int): 历史回顾窗口长度 L (Lookback Window)
            h_win (int): 未来预测跨度 H (Horizon Window)
            label_win (int): 解码器 Token 占位长度
            feature_dim (int): 全通道特征维度 (包含外生变量和最后的1列目标 Y)
            forecast_model (nn.Module): 任意可插入的 Model-Agnostic 时序预测底座 (如 iTransformer)
            num_regimes (int): 潜在环境机制原型的离散数量上限，用作抗震荡的结构化低频滤波器
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
        # 通过 spectral_norm 限制最大奇异值 <= 1，强制使网络具备低 Lipschitz 常数。
        # Tanh 激活函数将连续的隐藏特征进一步向有界空间压缩，防止相邻 Batch 的随机微弱噪声引发输出跳变。
        self.context_encoder = nn.Sequential(
            # (batch, lb_win, feature_dim) → 展平成 (batch, lb_win * feature_dim)，将整个历史窗口视为一个长向量。
            # 将高维输入压缩到 128 维隐空间，捕捉全局时序模式。
            spectral_norm(nn.Linear(lb_win * feature_dim, 128)),
            nn.Tanh(),  #将输出值限制在 [-1, 1] 之间，使特征保持有界，避免数值发散。相比 ReLU，Tanh 具有零中心对称性，能更好保留正负向信息，适合作为“权重基底”的预处理。
            spectral_norm(nn.Linear(128, num_regimes))  # 输出归属于各离散机制的概率 Logits 权重，代表预设的离散机制数量（如“上升期”“震荡期”“下降期”等）
        )
        """
        Note：利普希茨有界超网络指：这个超网络的每一层都通过谱归一化等方式控制了利普希茨常数，
        从而整个网络对输入变化的响应是平滑、有上限的。这在处理非平稳时间序列或需要稳定状态切换的场景中尤为重要。
        """
        # -------------------- [抗敏感震荡核心设计二：离散机制原型拓扑矩阵] --------------------
        # 声明有限个全局共享的隐式因果骨架母体，剥离样本特异性的虚假关联过拟合。
        # regime_prototypes_X 负责 X 空间的滑窗不变量分布 (num_regimes, (L+1) * d_X_exogenous)
        self.regime_prototypes_X = nn.Parameter(torch.randn(num_regimes, (lb_win + 1) * (feature_dim - 1)))
        # regime_prototypes_Y 负责 Y 空间的逐时间步变系数仿射 (num_regimes, 2 * H)
        # [改动A] 近恒等初始化：经 reshape (H,2) 后 [...,0]=gamma_raw、[...,1]=beta。
        # 令 gamma_raw = softplus^{-1}(0.5) ≈ -0.4328 → softplus(-0.4328)+0.5 = 1.0；beta = 0。
        # 这样仿射在 init 即为恒等(gamma≈1, beta≈0)，配合锚定可在弱漂移(ETT)下稳定退化为 ShifTS；
        # 叠加 0.02 微扰打破各机制对称性。注意：foward_* 中的 gamma=softplus(gamma_raw)+0.5 公式保持不变。
        _protoY = torch.zeros(num_regimes, h_win, 2)
        _protoY[:, :, 0] = -0.4328   # gamma 通道 → 初始 gamma = 1.0
        _protoY[:, :, 1] = 0.0       # beta 通道 → 初始 beta = 0.0
        _protoY = _protoY + 0.02 * torch.randn(num_regimes, h_win, 2)
        self.regime_prototypes_Y = nn.Parameter(_protoY.reshape(num_regimes, 2 * h_win))

        # 跨时空因果核心残差聚合器 Agg(·)
        self.linear = nn.Sequential(*[
            nn.Linear(feature_dim - 1, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ])

        # -------------------- [创新点3：VICReg 式共享隐空间闭环对齐] --------------------
        # 两个投影头把特征代理 X^SUR(d_X 维) 与目标代理 Y^suf(1 维) 投到同一 proj_dim 隐空间。
        # 选 VICReg 的关键：它显式支持两分支“异构、异维、不共享权重”，恰配 X/Y 两路异质代理；
        # 且其 variance 项内建防坍塌（纯拉近必塌成常数的根因防护）。仅作训练期对齐正则，
        # foward_feature(推理) 不调用，不接入预测路径，故不影响测试期输出。
        self.proj_dim = proj_dim
        self.proj_X = nn.Sequential(
            nn.Linear(feature_dim - 1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.proj_Y = nn.Sequential(
            nn.Linear(1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

        # -------------------- [方案C：同方差不确定性加权] --------------------
        # 4 个主预测/对齐任务的可学习对数方差 s_i = log σ_i^2（数值稳定参数化，避免对 σ 直接求导的除零）。
        # 顺序固定为 [observed, x_surrogate, y_clean, align]；零初始化 ⇔ σ^2=1（等权起步）。
        # 经 _select_optimizer 的名称匹配，self.log_vars / self.proj_* 均自动归入 model_optim(常规 lr)，
        # 不进入慢速 causal_optim（那只收 regime_prototypes_* 与 context_encoder）。
        self.log_vars = nn.Parameter(torch.zeros(4))

    """
    训练前向传播方法 `foward_causal` (含 SAM 物理实现)
    该方法在【训练阶段】被调用。其精妙之处在于：利用只有训练期才可获得的未来外生特征 X^H 与历史 X^L 共同切片，
    并通过 SAM 过滤得到黄金替代特征 X^{SUR}，用于强力监督底座预测模型的外生预测通道。
    - `x_enc_all`：全量时间轴滑窗切片，形状为 `(B, L + 1, H, feature_dim)`（包含历史和未来拼接后切片出的 $L+1$ 个候选局部特征窗口）。
    - `x_enc`：历史 Lookback 序列，形状为 `(B, L, feature_dim)`。
    - `x_mark_enc`：Lookback 时间戳特征。
    - `x_dec`/`x_mark_dec`：解码器占位输入及对应时间戳特征。
    """
    def foward_causal(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        训练期因果前向传播流 (利用未来全量视界提取 X 代理，并结合 IRN 净化 Y 污染)
        """
        batch_size = x_enc.shape[0]

        # 1. 经过利普希茨有界超网络，动态计算当前批次内每个样本对离散环境机制的分配概率
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1)) #(batch_size, num_regimes)
        regime_weights = torch.softmax(regime_logits, dim=-1)  # 沿着最后一维（dim=-1，即机制维度）进行归一化，将 logits 转换为概率分布。

        # 2. 【凸重塑路由】通过软路由组合，无震荡地平滑插值出当前机制下的特异性 X 空间掩码
        #对每个样本，将其机制权重作为系数，对 num_regimes 个原型掩码向量进行加权求和。
        # 每个样本得到了一个融合后的展平掩码向量，它综合了所有机制的贡献，权重由样本自身的机制概率决定。
        gated_mask_X = torch.mm(regime_weights, self.regime_prototypes_X)  # 维度: (B, (L+1)*(feature_dim-1))
        # 将展平的掩码向量恢复为三维矩阵形状 (B, L+1, d_X)，即软注意力掩码，用于后续对历史-未来切片进行加权聚合，它决定了在每个时间切片、每个外生特征通道上的注意力强度。
        gated_mask_X = gated_mask_X.reshape(batch_size, self.lb_win + 1, self.feature_dim - 1)

        # -------------------- [X空间计算：动态因果边界补全 (ShifTS 算子)] --------------------
        # 在多滑窗维度 (L+1) 进行 Softmax 归一化
        learnable_mask = torch.softmax(gated_mask_X, dim=1)
        mask_mean = torch.mean(learnable_mask, dim=1, keepdim=True)
        # 执行硬阈值门控，清洗低于平均线贡献的环境虚假局部弱相关噪声
        condition = (learnable_mask - mask_mean) > 0
        learnable_mask = learnable_mask * condition
        learnable_mask = learnable_mask / (torch.sum(learnable_mask, dim=1, keepdim=True) + 1e-6)

        # 为目标通道 Y 构建一列独热硬编码占位符，锁定 0 号未来视界切片，捍卫条件分布
        last_dim = torch.zeros(batch_size, self.lb_win + 1, 1).to(learnable_mask.device)
        last_dim[:, 0, 0] = 1.0
        # 在任何一个特定的时段环境下，针对每一个独立的外生特征通道，赋予其在所有候选滑动窗口（Slices）上的注意力权重之和必须严格等于 1。
        learnable_mask = torch.cat((learnable_mask, last_dim), dim=-1)  # 维度: (B, L+1, feature_dim)

        # 高维广播矩阵相乘并求和压缩，提炼出黄金替代特征真实值 X^{SUR}
        # x_enc_all 变换轴序后形状为 (B, H, L+1, feature_dim)
        x_pred_sel = torch.sum(x_enc_all.permute(0, 2, 1, 3) * learnable_mask.unsqueeze(1), -2)

        # -------------------- [Y空间计算：精细化时变 IRN 净化 (完全对齐 FOIL 源码)] --------------------
        # 动态插值提取当前环境机制对应的未来 H 个时间步特异性仿射变换系数
        gated_mask_Y = torch.mm(regime_weights, self.regime_prototypes_Y)  # 维度: (B, 2*H)
        irn_params = gated_mask_Y.reshape(batch_size, self.h_win, 2)  # 维度: (B, H, 2)

        gamma_raw = irn_params[:, :, 0:1]  # 提取每个未来时刻独立的乘性分量 (B, H, 1)
        beta = irn_params[:, :, 1:2]  # 提取每个未来时刻独立的加性分量 (B, H, 1)

        # Softplus 运算保证仿射尺度缩放的安全正定边界，防止梯度爆炸
        gamma = F.softplus(gamma_raw) + 0.5

        # -------------------- [Model-Agnostic 底座数据前向流与因果残差耦合] --------------------
        x_enc_norm = self.revin1(x_enc, mode='norm')  # 前置时间边缘偏移缓解
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]  # 底座预测的未来外生替代特征估计 \hat{X}^{SUR}
        y_pred_base = x_pred[:, :, -1:]  # 底座预测的目标通道常规趋势预测 \hat{Y}_{base}

        # 核心解耦：在不受未观测混杂变量污染的纯净因果空间中计算不变量代理目标 \hat{Y}^{SUR}
        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)  # 维度: (B, H, 1)

        # 逆转混杂污染：将当前时段对应的时变乘加效应重新注入，还原为真实观测空间下的物理估计
        y_pred_observed = y_pred_surrogate * gamma + beta  # 执行精细的逐时间步点对点仿射

        # 全通道拼接并执行反向实例归一化，恢复真实数据量纲量级
        x_out = torch.cat((x_pred_features, y_pred_observed), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')

        # [创新点3] 把两路代理预测投到共享隐空间，供 VICReg 闭环对齐。
        # 注意：在 RevIN 反归一化之前的“标准化空间”里对齐——x_pred_features 与 y_pred_surrogate 同处该尺度，量纲一致。
        # 以 (B*H) 为对齐批维（远大于 B），使 VICReg 的 variance/covariance 沿批维的统计在小数据(ETT/ILI)上更稳。
        z_align_X = self.proj_X(x_pred_features.reshape(-1, self.feature_dim - 1))  # (B*H, proj_dim)
        z_align_Y = self.proj_Y(y_pred_surrogate.reshape(-1, 1))                    # (B*H, proj_dim)

        # 返回核心变量：含正交损失用的 regime_logits，以及创新点3的两路对齐嵌入 z_align_X / z_align_Y
        return (x_pred_sel[:, :, :-1], x_out[:, :, :-1], x_out, learnable_mask,
                y_pred_surrogate, gamma, beta, regime_logits, z_align_X, z_align_Y)

    def foward_feature(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec, return_diag=False):
        """
        测试/推理期前向传播流 (完全自洽，无任何未来视界泄露或未来混杂领域标签依赖)
        """
        batch_size = x_enc.shape[0]

        # 仅依赖历史 lookback 输入自适应解码出当前的机制权重
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))
        regime_weights = torch.softmax(regime_logits, dim=-1)

        # 动态平滑组装当前时刻未来 H 个步长的 IRN 时变净化系数
        gated_mask_Y = torch.mm(regime_weights, self.regime_prototypes_Y)
        irn_params = gated_mask_Y.reshape(batch_size, self.h_win, 2)
        gamma = (F.softplus(irn_params[:, :, 0:1]) + 0.5)
        beta = irn_params[:, :, 1:2]

        # 底座模型及残差校准前向传导
        x_enc_norm = self.revin1(x_enc, mode='norm')
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]
        y_pred_base = x_pred[:, :, -1:]

        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)
        y_pred_observed = y_pred_surrogate * gamma + beta  # 混杂还原

        x_out = torch.cat((x_pred_features, y_pred_observed), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')
        if return_diag:  # [日志] 测试期可选返回路由/仿射诊断量；vali 不传该参数，返回行为不变
            return x_out, {'regime_weights': regime_weights.detach(),
                           'gamma': gamma.detach(), 'beta': beta.detach()}
        return x_out

class Exp_Causal_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Causal_Forecast, self).__init__(args)

    def _build_model(self):
        """
        统一双空间因果预测模型的构建与封装函数
        """
        print("Exp Causal Forecast-Model: Dual Space Surrogate Wrapper")
        # 1. 临时切为通用预测模式，无侵入式地初始化底层多变量时序预测底座 (如 iTransformer/PatchTST)
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

        # 4. 从配置项中调取机制原型的聚类数量边界 (若参数未定义则默认卡死为 5，启动结构化低pass滤波)
        num_regimes = getattr(self.args, 'num_regimes', 5)

        # 5. 调用最新的升级版 DualSpaceSurrogateWrapper 替换原版僵硬的 Warpper 模型体
        causal_model = DualSpaceSurrogateWrapper(
            lb_win=self.args.seq_len,
            h_win=self.args.pred_len,
            label_win=self.args.label_len,
            feature_dim=self.args.enc_in,
            forecast_model=model1,
            num_regimes=num_regimes
        )

        return causal_model

    """
    - 根据传入的标识（`train`/`val`/`test`）调用 `data_factory.py` 的 `data_provider`。
    - 注意：此时配置项为 `customc` 或 `ETT*c`，Dataloader 内部重写的 `__getitem__` 将额外返回利用 Lookback 和 Horizon 全局滑窗切好的模式矩阵 `seq_xall`。
    """
    def _get_data(self, flag):
        """
        数据迭代器获取接口：内部配合 data_provider 吐出含有全局滑窗全量未来信息的矩阵
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    """
    - 原理：双独立梯度传导设计。
    - 作用：
        - 将整个复杂网络的参数拆解为两个正交的组别。
        - 组 1：`mask_parameters`，仅仅包含软注意力不变量掩码矩阵 `self.model.mask`。
        - 组 2：`other_parameters`，包含底座神经网络的所有参数以及残差 Agg MLP 映射层的参数。
    """

    def _select_optimizer(self):
        """
        双独立优化器精细化分配算子：彻底根治因果掩码在 Batch 间的过敏感震荡
        """
        # 显式声明所有属于“因果机制/隐藏环境推断”范畴内的网络组件名称
        # 包括：离散机制原型X/Y、利普希茨连续上下文编码器、仿射变系数超网络
        causal_component_names = [
            'regime_prototypes_X',
            'regime_prototypes_Y',
            'context_encoder'
        ]

        causal_parameters = []
        other_parameters = []

        # 遍历全网拓扑，通过名称字符串匹配执行权重的正交切分
        for name, param in self.model.named_parameters():
            if any(cn in name for cn in causal_component_names):
                causal_parameters.append(param)  # 归属于因果环境路由的权重组
            else:
                other_parameters.append(param)  # 归属于基础时序预测底座与常规 Agg MLP 聚合层的权重组

        # 【因果核心防御】：因果环境超网络统一采用 1e-4 的稳健、保守学习率
        # 这配合模型内部的谱范数限制，从根源上拉平了相邻Batch的梯度扰动，捍卫了跨领域的因果一致性
        causal_optim = optim.Adam(causal_parameters, lr=1e-4)

        # 基础预测通道采用常规用户配置的大学习率，保障基础时序大趋势的快速收敛拟合
        model_optim = optim.Adam(other_parameters, lr=self.args.learning_rate)

        return model_optim, causal_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _routing_regularization(self, regime_logits):
        """
        [改动1] 路由正则：根治“原型正交但门控坍缩”的退化。
        原有的 loss_orthogonal 只约束 regime_prototypes_X（原型彼此正交），
        但完全管不住门控分布——可能出现“原型互相正交，可 softmax 永远只挑某一个原型”的坍缩，
        使 num_regimes 实际退化为 1，模型只剩一张全局静态掩码（≈ShifTS）却背着多余开销。
        本正则直接作用在路由权重上，与 loss_orthogonal 互补，二者合起来才真正提供额外容量。

        参数:
            regime_logits (Tensor): foward_causal 返回的机制 logits，形状 (B, num_regimes)。
        返回:
            loss_balance (Tensor): Switch 风格负载均衡 N * Σ_e (f_e · P_e)。
                f_e = 批内被硬分配(argmax)到机制 e 的样本占比 (按 Switch 设计 detach, 不可导)；
                P_e = 机制 e 的平均软门控概率 (可导)。均匀使用时取最小值 1，集中到单一机制时取最大值 N。
                最小化它 → 推动各机制被均衡使用，梯度经 P_e 回传至 context_encoder 路由器。
            loss_entropy (Tensor): 批级平均路由分布的“负熵” (= -H(P)，取值 ∈ [-log N, 0])。
                以正权重加入总损失并最小化，等价于最大化批级使用熵 → 鼓励不同样本路由到不同机制。
        """
        regime_weights = torch.softmax(regime_logits, dim=-1)              # (B, num_regimes)
        batch_size = regime_weights.size(0)
        num_regimes = regime_weights.size(-1)

        # P_e：批内平均软门控概率（可导）
        importance = regime_weights.mean(dim=0)                            # (num_regimes,)

        # f_e：批内被硬分配到各机制的样本占比（不可导，detach）
        hard_idx = torch.argmax(regime_weights, dim=-1)                    # (B,)
        f = torch.zeros_like(importance)
        f = f.scatter_add(0, hard_idx, torch.ones_like(hard_idx, dtype=importance.dtype))
        f = (f / batch_size).detach()
        loss_balance = num_regimes * torch.sum(f * importance)

        # 批级平均分布负熵：-Σ_e P_e log P_e
        loss_entropy = torch.sum(importance * torch.log(importance + 1e-8))

        return loss_balance, loss_entropy

    @staticmethod
    def _uw(loss_term, log_var):
        """
        [方案C] 同方差不确定性加权单项（Kendall, Gal & Cipolla, CVPR 2018）。
        以 s = log σ^2 的对数方差参数化保证数值稳定：项 = 0.5 * exp(-s) * L + 0.5 * s。
        其中 0.5*exp(-s) 是该任务的有效精度权重（任务越不准，网络自发增大 σ 调小其权重），
        而 +0.5*s 作为数学天平阻止 σ→∞。四个主任务共享该形式，权重全程自适应、免手调。
        """
        return 0.5 * torch.exp(-log_var) * loss_term + 0.5 * log_var

    def _vicreg_alignment(self, z_x, z_y, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04, eps=1e-4):
        """
        [创新点3] 非对抗 VICReg 式共享隐空间闭环对齐（取代原“对抗监督”的含糊表述）。
        在公共隐空间把特征代理嵌入 z_x 与目标代理嵌入 z_y 对齐，三项缺一不可：
          - invariance：MSE(z_x, z_y) —— 提供真正的跨空间“拉近”耦合（闭环对齐本体）；
          - variance：铰链损失逼每维标准差≥1 —— 防两路塌成常数（纯拉近必坍缩的根因防护）；
          - covariance：去相关每维 —— 防冗余维度的信息坍塌。
        默认内部系数沿用 VICReg 的 25:25:1 比例（此处等比缩放为 1:1:0.04，使整体量级 O(1)，
        便于外层方案C的不确定性加权自适应调度）。
        参数:
            z_x, z_y (Tensor): 形状 (N, D) 的两路对齐嵌入，N = B*H。
        返回:
            loss_align (Tensor): 标量 = sim_coeff*inv + std_coeff*var + cov_coeff*cov。
        """
        N, D = z_x.shape
        denom = max(N - 1, 1)

        # invariance：两路嵌入逐样本对齐（拉近）
        inv = F.mse_loss(z_x, z_y)

        # variance：沿批维的每维标准差，铰链推向目标 1.0（防坍塌）
        std_x = torch.sqrt(z_x.var(dim=0) + eps)
        std_y = torch.sqrt(z_y.var(dim=0) + eps)
        var = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

        # covariance：中心化后协方差矩阵的非对角元平方和，逐维去相关（防信息坍塌）
        zx = z_x - z_x.mean(dim=0)
        zy = z_y - z_y.mean(dim=0)
        cov_x = (zx.T @ zx) / denom
        cov_y = (zy.T @ zy) / denom
        off_x = (cov_x.pow(2).sum() - cov_x.pow(2).diagonal().sum()) / D
        off_y = (cov_y.pow(2).sum() - cov_y.pow(2).diagonal().sum()) / D
        cov = off_x + off_y

        return sim_coeff * inv + std_coeff * var + cov_coeff * cov

    # ==============================================================================
    # 以下为 Exp_Causal_Forecast 类中 train、vali、test 函数的完整无损重构代码
    # ==============================================================================
    def vali(self, vali_data, vali_loader, criterion):
        """
        完整的验证期控制流：完全自恰推理，无未来视界与未观测混杂环境标签泄露
        """
        total_loss = []
        self.model.eval()  # 开启评估模式，关闭 Dropout 和 BatchNorm 的动态更新

        with torch.no_grad():  # 禁用全计算图梯度流，节省显存
            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                # 精度与计算卡设备强行拉齐
                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构建常规解码器占位输入 (未来预测区域用0填充)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 执行自洽环境推理（内部谱范数超网络与离散机制路由平滑工作）
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                # 健壮性自适应修复：若底座模型在开启output_attention时强行截断返回了元组，则自动提取首位预测Tensor
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # 截取未来预测跨度内的目标通道
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # 计算纯粹观测空间下的时序 MSE 损失，作为早停机制（Early Stopping）的判定基准
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()  # 恢复训练状态
        return total_loss

    def train(self, setting):
        """
        完整的训练期控制流：支持强随机打乱 (shuffle=True)，多任务高精密空间对齐
        """
        # 获取三套相互独立的数据迭代器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建权重Checkpoints专属保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # 初始化早停追踪器
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 调取正交拆分后的双独立优化器（模型参数与静态机制原型各自独立传导梯度）
        model_optim, causal_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # [方案C + 改动1/3] 损失权重策略：
        #   四个主预测/对齐任务{observed, x_sur, y_clean, align}的相对权重不再手调，
        #   改由同方差不确定性 self.model.log_vars 自适应配权（取代原固定的 lambda_x / lambda_y）。
        #   下列为“结构正则项”的小固定权重（它们非预测任务、无观测噪声语义，故不纳入不确定性加权）：
        lambda_orth = getattr(self.args, 'lambda_orth', 0.01)         # 原型正交化（防原型坍缩）
        lambda_balance = getattr(self.args, 'lambda_balance', 1e-2)   # [改动1] 路由负载均衡（防门控坍缩；Switch 经验取 1e-2）
        lambda_entropy = getattr(self.args, 'lambda_entropy', 1e-2)   # [改动1] 批级路由熵（鼓励机制多样，与负载均衡互补，可置 0）
        lambda_affine = getattr(self.args, 'lambda_affine', 0.05)     # [改动A] 上调至 0.05：配合近恒等初始化，确保弱漂移下仿射稳在恒等(退化为 ShifTS)；ILI/Exchange 上数据仍可驱动其偏离

        # 混合精度加速算子声明
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # [改动B] 不确定性加权尺度初始化标志：在首个 batch 后用各任务真实损失量级初始化 log_vars，
        # 使四项有效贡献开局即被拉平（避免被放大的 y_clean/align 在前若干 epoch 霸占梯度）。数据集无关、自校准。
        self._uw_init_done = False

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # [日志] 本 epoch 的分项损失 + 路由/仿射健康度累加器（每个 epoch 重置）
            log_components = {'observed': [], 'x_sur': [], 'y_clean': [], 'align': [], 'orth': [],
                              'balance': [], 'entropy': [], 'affine': []}
            log_diag = {'route_entropy_norm': [], 'route_entropy_persample': [], 'P_e': [],
                        'gamma_mean': [], 'gamma_dev': [], 'beta_mean': [], 'beta_abs': []}
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                iter_count += 1

                # 双优化器梯度同时双向置零，彻底避免多任务计算图中的梯度残留污染
                model_optim.zero_grad()
                causal_optim.zero_grad()

                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构建常规解码器输入占位符
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # -------------------- 前向传播与多任务空间损失联合计算 --------------------
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 调取升级版的统一双空间前向传播接口
                        (x_enc_sel, x_enc_pred, outputs, learnable_mask, y_pred_surrogate,
                         gamma, beta, regime_logits, z_align_X, z_align_Y) = self.model.foward_causal(
                            batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark
                        )
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        # 1. 提取未来视界内的观测预测与真实切片
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs_y = outputs[:, -self.args.pred_len:, f_dim:]
                        target_y = batch_y[:, -self.args.pred_len:, f_dim:]

                        # 损失项一：常规物理观测时序预测损失 (MSE)
                        loss_observed = criterion(outputs_y, target_y)

                        # 损失项二：X空间特征代理边界补全损失 (ShifTS 原理约束)
                        loss_x_surrogate = criterion(x_enc_sel, x_enc_pred)

                        # 损失项三：Y空间精细化时变混杂清洗损失 (FOIL 变系数原理重构)
                        with torch.no_grad():
                            #y_true_norm = self.model.revin1(batch_y.float(), mode='norm')[:, -self.args.pred_len:, -1:]
                            y_true_norm = self.model.revin1._normalize(batch_y.float())[:, -self.args.pred_len:, -1:]
                        # [改动2] stop-gradient：对 gamma/beta 施加 detach，将净化目标固定为常量。
                        # 原实现中 y_true_surrogate 依赖可学习的 gamma/beta（移动靶），且与 loss_observed 近乎循环
                        # （y_pred_observed = y_pred_surrogate*gamma+beta，一旦观测损失收敛此项几乎被蕴含），
                        # detach 后仿射只经 loss_observed 学习，本项纯粹监督代理预测器 y_pred_surrogate。
                        y_true_surrogate = (y_true_norm - beta.detach()) / (gamma.detach() + 1e-6)
                        loss_y_clean = criterion(y_pred_surrogate, y_true_surrogate)

                        # 损失项四：离散机制原型软正交化损失，强制机制拓扑母体不退化坍缩
                        normalized_prototypes = F.normalize(self.model.regime_prototypes_X, p=2, dim=1)
                        identity_matrix = torch.eye(normalized_prototypes.size(0), device=normalized_prototypes.device)
                        prototype_correlation = torch.matmul(normalized_prototypes,
                                                             normalized_prototypes.transpose(0, 1))
                        loss_orthogonal = torch.norm(prototype_correlation - identity_matrix, p='fro') ** 2

                        # [改动1] 损失项五：路由负载均衡 + 批级熵正则（原型正交≠门控均衡，专治“路由坍缩”）
                        loss_balance, loss_entropy = self._routing_regularization(regime_logits)

                        # [改动3] 损失项六：仿射趋同恒等先验，确立“不退化下界”。
                        # 弱漂移（如 ETT）时 gamma→1、beta→0，双空间平滑退化为 ShifTS，从而至少不劣于 ShifTS；
                        # 强漂移（如 ILI/Exchange）时数据自会驱动仿射偏离恒等以发挥作用。
                        loss_affine_anchor = torch.mean((gamma - 1.0) ** 2) + torch.mean(beta ** 2)

                        # [创新点3] 损失项七：VICReg 式共享隐空间闭环对齐（非对抗）
                        loss_align = self._vicreg_alignment(z_align_X, z_align_Y)

                        # [方案C] 四个主任务{observed, x_sur, y_clean, align}由同方差不确定性自适应配权：
                        #   uw(L, s) = 0.5*exp(-s)*L + 0.5*s,  s = log σ^2（log 项阻止 σ→∞，权重全程自适应）。
                        # 结构正则{orth, balance, entropy, affine}非预测任务、无观测噪声语义，保留小固定权重。
                        s = self.model.log_vars
                        loss_tasks = (self._uw(loss_observed, s[0]) + self._uw(loss_x_surrogate, s[1])
                                      + self._uw(loss_y_clean, s[2]) + self._uw(loss_align, s[3]))
                        loss = (loss_tasks
                                + lambda_orth * loss_orthogonal
                                + lambda_balance * loss_balance
                                + lambda_entropy * loss_entropy
                                + lambda_affine * loss_affine_anchor)
                        train_loss.append(loss.item())
                else:
                    (x_enc_sel, x_enc_pred, outputs, learnable_mask, y_pred_surrogate,
                     gamma, beta, regime_logits, z_align_X, z_align_Y) = self.model.foward_causal(
                        batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark
                    )
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs_y = outputs[:, -self.args.pred_len:, f_dim:]
                    target_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss_observed = criterion(outputs_y, target_y)
                    loss_x_surrogate = criterion(x_enc_sel, x_enc_pred)

                    with torch.no_grad():
                        #y_true_norm = self.model.revin1(batch_y.float(), mode='norm')[:, -self.args.pred_len:, -1:]
                        y_true_norm = self.model.revin1._normalize(batch_y.float())[:, -self.args.pred_len:, -1:]
                    # [改动2] stop-gradient：固定净化目标，仿射只经 loss_observed 学习（详见 AMP 分支注释）
                    y_true_surrogate = (y_true_norm - beta.detach()) / (gamma.detach() + 1e-6)
                    loss_y_clean = criterion(y_pred_surrogate, y_true_surrogate)

                    normalized_prototypes = F.normalize(self.model.regime_prototypes_X, p=2, dim=1)
                    identity_matrix = torch.eye(normalized_prototypes.size(0), device=normalized_prototypes.device)
                    prototype_correlation = torch.matmul(normalized_prototypes, normalized_prototypes.transpose(0, 1))
                    loss_orthogonal = torch.norm(prototype_correlation - identity_matrix, p='fro') ** 2

                    # [改动1] 路由负载均衡 + 批级熵正则（专治路由坍缩）
                    loss_balance, loss_entropy = self._routing_regularization(regime_logits)
                    # [改动3] 仿射趋同恒等先验（不退化下界：弱漂移时退化为 ShifTS）
                    loss_affine_anchor = torch.mean((gamma - 1.0) ** 2) + torch.mean(beta ** 2)
                    # [创新点3] VICReg 式共享隐空间闭环对齐（非对抗）
                    loss_align = self._vicreg_alignment(z_align_X, z_align_Y)

                    # [方案C] 四主任务由同方差不确定性自适应配权；结构正则保留小固定权重（详见 AMP 分支注释）
                    s = self.model.log_vars
                    loss_tasks = (self._uw(loss_observed, s[0]) + self._uw(loss_x_surrogate, s[1])
                                  + self._uw(loss_y_clean, s[2]) + self._uw(loss_align, s[3]))
                    loss = (loss_tasks
                            + lambda_orth * loss_orthogonal
                            + lambda_balance * loss_balance
                            + lambda_entropy * loss_entropy
                            + lambda_affine * loss_affine_anchor)
                    train_loss.append(loss.item())

                # [改动B] 仅在首个 batch：用四个主任务的真实损失量级初始化 log_vars（s_i = log L_i），
                # 使四项有效贡献(0.5*exp(-s)*L)开局即被拉平至 ~0.5，避免被放大的 y_clean/align 在前期霸占梯度。
                # 本数据集无关（ETT/ILI/Exchange 皆自校准）。首个 batch 的总损失已用 log_vars=0 反传一次，无碍。
                if not self._uw_init_done:
                    with torch.no_grad():
                        _init_s = torch.log(torch.stack([
                            loss_observed.detach(), loss_x_surrogate.detach(),
                            loss_y_clean.detach(), loss_align.detach()
                        ]).clamp_min(1e-6))
                        self.model.log_vars.data.copy_(_init_s.float())
                    self._uw_init_done = True
                    print("  [改动B] 已按首个 batch 损失尺度初始化 log_vars(s=log L): {}".format(
                        np.round(self.model.log_vars.data.detach().cpu().numpy(), 3)))

                # [日志] 累计本 epoch 的分项损失 + 路由/仿射统计（全部 detach，仅读数，不影响反向传播）
                with torch.no_grad():
                    log_components['observed'].append(loss_observed.item())
                    log_components['x_sur'].append(loss_x_surrogate.item())
                    log_components['y_clean'].append(loss_y_clean.item())
                    log_components['align'].append(loss_align.item())
                    log_components['orth'].append(loss_orthogonal.item())
                    log_components['balance'].append(loss_balance.item())
                    log_components['entropy'].append(loss_entropy.item())
                    log_components['affine'].append(loss_affine_anchor.item())
                    # 路由健康度：归一化使用率熵 ∈ [0,1]（1=机制被完全均匀使用，→0=坍缩到单一机制）
                    _rw = torch.softmax(regime_logits, dim=-1)
                    _Pe = _rw.mean(dim=0)
                    _R = _Pe.size(-1)
                    _H = -torch.sum(_Pe * torch.log(_Pe + 1e-8)).item()
                    log_diag['route_entropy_norm'].append(_H / float(np.log(max(_R, 2))))
                    # [改动D] 逐样本路由熵(批均值)：区分"每样本均匀混合(未特化)" vs "特化但批级均衡"。
                    # 低逐样本熵 + 高批级熵 = 健康特化；逐样本熵也≈1 = 各样本均匀混合(环境条件化未特化)。
                    _H_ps = -(_rw * torch.log(_rw + 1e-8)).sum(dim=-1).mean().item()
                    log_diag['route_entropy_persample'].append(_H_ps / float(np.log(max(_R, 2))))
                    log_diag['P_e'].append(_Pe.detach().float().cpu())
                    # 仿射偏离恒等程度：弱漂移应 gamma≈1 / beta≈0，强漂移（ILI/Exchange）才偏离
                    log_diag['gamma_mean'].append(gamma.mean().item())
                    log_diag['gamma_dev'].append((gamma - 1.0).abs().mean().item())
                    log_diag['beta_mean'].append(beta.mean().item())
                    log_diag['beta_abs'].append(beta.abs().mean().item())

                # -------------------- 混合精度与常规精度的反向传播控制 --------------------
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

                # 周期性科学日志记录
                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | 联合总Loss: {2:.7f} | 观测Loss: {3:.7f}".format(
                        i + 1, epoch + 1, loss.item(), loss_observed.item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; 预计训练剩余总时间: {:.4f}s'.format(speed, left_time))

                    # 实时在线监测当前样本级不变量过滤掩码的收敛走向分布
                    print("\t当前Batch不变量最大注意力索引矩阵:")
                    print("\t", torch.argmax(learnable_mask, dim=1)[0].detach().cpu().numpy())
                    # [日志] 路由/仿射实时快照（本 batch 刚累计的值）
                    print("\t[诊断] 批级路由熵={:.3f} 逐样本路由熵={:.3f} | P_e={} | |gamma-1|={:.4f}(gamma_bar={:.3f}) | |beta|={:.4f}(beta_bar={:.3f})".format(
                        log_diag['route_entropy_norm'][-1], log_diag['route_entropy_persample'][-1],
                        np.round(log_diag['P_e'][-1].numpy(), 3),
                        log_diag['gamma_dev'][-1], log_diag['gamma_mean'][-1],
                        log_diag['beta_abs'][-1], log_diag['beta_mean'][-1]))
                    iter_count = 0
                    time_now = time.time()

            # 每个 Epoch 终结后的全量验证集考核与自适应动态调频
            print("Epoch: {} 耗时总长: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | 综合训练Loss: {2:.7f} | 验证集验证Loss: {3:.7f} | 测试集测试Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # ===================== [日志] 本 epoch 诊断汇总 =====================
            def _avg(_l):
                return float(np.mean(_l)) if len(_l) else float('nan')
            _mean_Pe = torch.stack(log_diag['P_e']).mean(0).numpy() if len(log_diag['P_e']) else None
            _re = _avg(log_diag['route_entropy_norm'])
            _nR = (len(_mean_Pe) if _mean_Pe is not None else 0)
            _active = int((_mean_Pe > (1.0 / (2 * _nR))).sum()) if (_mean_Pe is not None and _nR > 0) else -1
            print("  [诊断·分项损失(加权前)] observed={:.5f} x_sur={:.5f} y_clean={:.5f} align={:.5f} | orth={:.5f} balance={:.5f} entropy={:.5f} affine={:.5f}".format(
                _avg(log_components['observed']), _avg(log_components['x_sur']), _avg(log_components['y_clean']),
                _avg(log_components['align']), _avg(log_components['orth']), _avg(log_components['balance']),
                _avg(log_components['entropy']), _avg(log_components['affine'])))
            # [方案C] 不确定性加权诊断：学到的 σ_i 与有效精度 0.5*exp(-s_i)（i = observed/x_sur/y_clean/align）。
            # σ 越大 → 该任务权重越小（网络判定其噪声大/难学）；据此可看四任务的自适应权重走向。
            _s = self.model.log_vars.detach().float().cpu().numpy()
            print("  [诊断·不确定性加权] sigma(obs,x_sur,y_clean,align)={} | 有效精度0.5*exp(-s)={}".format(
                np.round(np.exp(_s / 2.0), 4), np.round(0.5 * np.exp(-_s), 4)))
            print("  [诊断·结构正则贡献(固定权重)] orth={:.5f} balance={:.5f} entropy={:.5f} affine={:.5f}".format(
                lambda_orth * _avg(log_components['orth']), lambda_balance * _avg(log_components['balance']),
                lambda_entropy * _avg(log_components['entropy']), lambda_affine * _avg(log_components['affine'])))
            _re_ps = _avg(log_diag['route_entropy_persample'])
            print("  [诊断·路由] 批级使用率熵(epoch均值)={:.3f} 逐样本熵={:.3f} | 激活机制数={}/{} | 平均使用率 P_e={}".format(
                _re, _re_ps, _active, _nR, np.round(_mean_Pe, 3) if _mean_Pe is not None else None))
            print("    └─ 解读：批级高 + 逐样本低 = 健康特化(不同样本走不同机制)；两者都≈1 = 各样本均匀混合(环境条件化未特化)")
            print("  [诊断·仿射] gamma_bar={:.3f} 平均|gamma-1|={:.4f} | beta_bar={:.3f} 平均|beta|={:.4f}".format(
                _avg(log_diag['gamma_mean']), _avg(log_diag['gamma_dev']),
                _avg(log_diag['beta_mean']), _avg(log_diag['beta_abs'])))
            if (not np.isnan(_re)) and _re < 0.5:
                print("  [诊断·告警] 路由熵偏低({:.3f}<0.5)，门控可能正在坍缩 → 增大 lambda_balance/lambda_entropy 或给 causal_optim 加 warmup".format(_re))

            # 早停触发判定
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("触发早停条件，系统在纯净不变量边界安全收敛。")
                break

            # 动态学习率周期性衰减调整
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 锁存并重新加载性能最优版本的参数 Checkpoint 权重
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        完整的最终测试期控制流：端到端干净推理，无任何未来标签或环境系数泄露风险
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('从持久化 Checkpoint 载入最优因果泛化模型体...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # [日志] 测试期路由/仿射诊断累加器
        log_test = {'route_entropy_norm': [], 'P_e': [], 'gamma_dev': [], 'beta_abs': []}
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(test_loader):
                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 解码输入占位构建
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 执行自洽推理
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _diag = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark, return_diag=True)
                else:
                    outputs, _diag = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark, return_diag=True)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # [日志] 累计测试期路由/仿射诊断（验证 Cut1 路由是否健康、Cut3 仿射在测试分布上是否贴近恒等）
                _rw = _diag['regime_weights']; _Pe = _rw.mean(dim=0); _R = _Pe.size(-1)
                _Ht = -torch.sum(_Pe * torch.log(_Pe + 1e-8)).item()
                log_test['route_entropy_norm'].append(_Ht / float(np.log(max(_R, 2))))
                log_test['P_e'].append(_Pe.float().cpu())
                log_test['gamma_dev'].append((_diag['gamma'] - 1.0).abs().mean().item())
                log_test['beta_abs'].append(_diag['beta'].abs().mean().item())

                # 反实例归一化尺度维度还原并抽离目标变量
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

                # 周期性单轨迹物理波形抽样（每 20 个样本抽取首个单通道样本波形）
                if i % 20 == 0:
                    input = batch_x_en.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    # 可选择性开启可视化函数：visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('真实环境基准校验测试集维度形态:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # 创建并输出 OOD 指标评测报告
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 统计常规核心多阶预测精度指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('真实试验验证最终成效评估 >> MSE 均方误差: {}, MAE 平均绝对误差: {}'.format(mse, mae))

        # [日志] 测试期路由/仿射诊断汇总（Cut1 路由健康度 + Cut3 仿射在测试分布上是否贴近恒等）
        if len(log_test['route_entropy_norm']):
            _mPe = torch.stack(log_test['P_e']).mean(0).numpy()
            print("  [诊断·测试期] 归一化路由熵={:.3f} | 平均使用率 P_e={} | 平均|gamma-1|={:.4f} | 平均|beta|={:.4f}".format(
                float(np.mean(log_test['route_entropy_norm'])), np.round(_mPe, 3),
                float(np.mean(log_test['gamma_dev'])), float(np.mean(log_test['beta_abs']))))

        # 结果本地固化落盘保存
        f = open("result_causal_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 保存为 CSV 可读格式
        with open(folder_path + 'metrics.csv', 'w') as f:
            f.write('metric,value\n')
            f.write(f'MAE,{mae}\n')
            f.write(f'MSE,{mse}\n')
            f.write(f'RMSE,{rmse}\n')
            f.write(f'MAPE,{mape}\n')
            f.write(f'MSPE,{mspe}\n')

        return

