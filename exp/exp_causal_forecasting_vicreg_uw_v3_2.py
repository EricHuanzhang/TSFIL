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
    def __init__(self, lb_win, h_win, label_win, feature_dim, forecast_model, num_regimes=5, proj_dim=128,
                 ablate_routing=False, ablate_align=False, ablate_uw=False, ablate_affine=False,
                 capacity_dropout=0.0, use_irn=False, use_vrex=False, vrex_pooled=False):
        """
        统一双空间代理学习包装器 (Dual-Space Surrogate Wrapper)
        参数:
            lb_win (int): 历史回顾窗口长度 L (Lookback Window)
            h_win (int): 未来预测跨度 H (Horizon Window)
            label_win (int): 解码器 Token 占位长度
            feature_dim (int): 全通道特征维度 (包含外生变量和最后的1列目标 Y)
            forecast_model (nn.Module): 任意可插入的 Model-Agnostic 时序预测底座 (如 iTransformer)
            num_regimes (int): 潜在环境机制原型的离散数量上限，用作抗震荡的结构化低频滤波器

        [消融开关] 以下四个布尔开关用于组件归因消融，默认全部 False（=完整 TSFIL，与改造前行为完全一致）：
            ablate_routing (bool): 关闭"逐样本环境路由"。开启后 regime_weights 强制取均匀分布(1/K)，
                X 掩码退化为全局共享静态掩码、Y 仿射退化为全局静态，等价于剥离创新点2的样本自适应性。
            ablate_align   (bool): 关闭创新点3的 VICReg 共享隐空间对齐（loss_align 置 0 且不计入主任务）。
            ablate_uw      (bool): 关闭方案C不确定性加权，四个主任务改为等权(系数1.0)相加。
            ablate_affine  (bool): 关闭 FOIL 式时变仿射（前向强制 gamma=1/beta=0；loss_y_clean 与
                loss_affine_anchor 置 0 且不计入）。注：实验已显示仿射全程惰性，此开关用于"关掉无损"验证。
            说明：四者同时为 True ≈ 退化为 ShifTS（全局静态掩码 + 等权 L_TS+L_SUR，无仿射/对齐/UW），
                  可作为"框架内复现 ShifTS"的健全性配置（其余休眠参数不参与计算，仅占显存）。
        [容量控制] capacity_dropout (float, 默认0.0): 针对短步长过拟合的容量正则。
            >0 时在上下文编码器与两个投影头的非线性后插入 Dropout(p)；=0 时不插入任何层，
            网络结构与改造前逐字节一致（零行为差异、零随机数消耗）。
        """
        super(DualSpaceSurrogateWrapper, self).__init__()
        self.lb_win = lb_win
        self.h_win = h_win
        self.label_win = label_win
        self.feature_dim = feature_dim
        self.forecast_model = forecast_model

        # [消融开关] 落库为模块属性，前向(foward_*)与训练损失装配(_assemble_total_loss)统一读取，单一真源。
        self.ablate_routing = bool(ablate_routing)
        self.ablate_align = bool(ablate_align)
        self.ablate_uw = bool(ablate_uw)
        self.ablate_affine = bool(ablate_affine)
        self.num_regimes = num_regimes  # 供前向构造均匀权重时读取 K
        # [FOIL 复刻支路开关] 默认 False：IRN 代理损失 / VREx 跨环境不变性均不参与（行为同改造前）。
        self.use_irn = bool(use_irn)
        self.use_vrex = bool(use_vrex)
        self.vrex_pooled = bool(vrex_pooled)  # VREx 方差形式：False=逐环境均值方差(默认), True=FOIL汇集逐样本方差(量级大)

        # 声明边缘时间偏移缓解算子 (实例归一化)
        self.revin1 = RevIN(num_features=feature_dim)

        # -------------------- [抗敏感震荡核心设计一：利普希茨有界超网络] --------------------
        # 通过 spectral_norm 限制最大奇异值 <= 1，强制使网络具备低 Lipschitz 常数。
        # Tanh 激活函数将连续的隐藏特征进一步向有界空间压缩，防止相邻 Batch 的随机微弱噪声引发输出跳变。
        # [容量控制] 按需插入 Dropout：capacity_dropout=0 时 _enc_layers 不含 Dropout，
        # nn.Sequential 结构与改造前完全一致（[Linear, Tanh, Linear]），无任何行为/随机数差异。
        _enc_layers = [
            # (batch, lb_win, feature_dim) → 展平成 (batch, lb_win * feature_dim)，将整个历史窗口视为一个长向量。
            # 将高维输入压缩到 128 维隐空间，捕捉全局时序模式。
            spectral_norm(nn.Linear(lb_win * feature_dim, 128)),
            nn.Tanh(),  #将输出值限制在 [-1, 1] 之间，使特征保持有界，避免数值发散。相比 ReLU，Tanh 具有零中心对称性，能更好保留正负向信息，适合作为“权重基底”的预处理。
        ]
        if capacity_dropout > 0:
            _enc_layers.append(nn.Dropout(capacity_dropout))
        _enc_layers.append(
            spectral_norm(nn.Linear(128, num_regimes))  # 输出归属于各离散机制的概率 Logits 权重，代表预设的离散机制数量（如“上升期”“震荡期”“下降期”等）
        )
        self.context_encoder = nn.Sequential(*_enc_layers)
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
        # [容量控制] 同理按需插入 Dropout：p=0 时结构为 [Linear, ReLU, Linear]，与改造前完全一致。
        _projX_layers = [nn.Linear(feature_dim - 1, proj_dim), nn.ReLU()]
        _projY_layers = [nn.Linear(1, proj_dim), nn.ReLU()]
        if capacity_dropout > 0:
            _projX_layers.append(nn.Dropout(capacity_dropout))
            _projY_layers.append(nn.Dropout(capacity_dropout))
        _projX_layers.append(nn.Linear(proj_dim, proj_dim))
        _projY_layers.append(nn.Linear(proj_dim, proj_dim))
        self.proj_X = nn.Sequential(*_projX_layers)
        self.proj_Y = nn.Sequential(*_projY_layers)

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
        #    [消融] regime_logits 始终计算（供返回与路由正则使用）；ablate_routing 时改用均匀权重 1/K，
        #    使 X 掩码与 Y 仿射对所有样本相同（退化为全局静态），剥离逐样本路由的自适应性。
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1)) #(batch_size, num_regimes)
        if self.ablate_routing:
            regime_weights = torch.ones_like(regime_logits) / regime_logits.size(-1)
        else:
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
        # [消融] 关闭 FOIL 时变仿射：强制恒等(gamma=1, beta=0)，y_pred_observed 即等于 y_pred_surrogate。
        if self.ablate_affine:
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)

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
        #    [消融] 与 foward_causal 保持一致：ablate_routing 时均匀化，确保"训练关路由→测试也关路由"。
        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))
        if self.ablate_routing:
            regime_weights = torch.ones_like(regime_logits) / regime_logits.size(-1)
        else:
            regime_weights = torch.softmax(regime_logits, dim=-1)

        # 动态平滑组装当前时刻未来 H 个步长的 IRN 时变净化系数
        gated_mask_Y = torch.mm(regime_weights, self.regime_prototypes_Y)
        irn_params = gated_mask_Y.reshape(batch_size, self.h_win, 2)
        gamma = (F.softplus(irn_params[:, :, 0:1]) + 0.5)
        beta = irn_params[:, :, 1:2]
        # [消融] 与 foward_causal 一致：关闭仿射则测试期也用恒等，避免训练/测试净化系数错配。
        if self.ablate_affine:
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)

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
        print("Exp Causal Forecast-Model: Dual Space Surrogate Wrapper v3-2")
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
        proj_dim = getattr(self.args, 'proj_dim', 128)            # [容量控制] VICReg 投影维度，默认128

        # [消融开关·一键集成] 从命令行/配置 args 读取四个消融开关；未定义则默认 False(=完整 TSFIL)。
        #   主开关 ablate_to_shifts=True 时一键置全部为 True(框架内复现 ShifTS 健全性配置)。
        master = getattr(self.args, 'ablate_to_shifts', False)
        ablate_routing = master or getattr(self.args, 'ablate_routing', False)
        ablate_align = master or getattr(self.args, 'ablate_align', False)
        ablate_uw = master or getattr(self.args, 'ablate_uw', False)
        ablate_affine = master or getattr(self.args, 'ablate_affine', False)
        capacity_dropout = getattr(self.args, 'capacity_dropout', 0.0)  # [容量控制] 默认0=无Dropout=原结构
        # [FOIL 复刻支路] 默认关闭；开启后作为固定权重正则（不进 UW）。
        use_irn = getattr(self.args, 'use_irn', False)
        use_vrex = getattr(self.args, 'use_vrex', False)
        vrex_pooled = getattr(self.args, 'vrex_pooled', False)  # VREx 方差形式开关(默认False=逐环境均值方差)
        # [日志] 打印生效的消融/容量配置，便于核对每次实验条件
        print("  [消融/容量配置] ablate_to_shifts={} | routing={} align={} uw={} affine={} | "
              "num_regimes={} proj_dim={} capacity_dropout={}".format(
                  master, ablate_routing, ablate_align, ablate_uw, ablate_affine,
                  num_regimes, proj_dim, capacity_dropout))
        print("  [FOIL支路配置] use_irn={} use_vrex={} vrex_pooled={}".format(use_irn, use_vrex, vrex_pooled))

        # 5. 调用最新的升级版 DualSpaceSurrogateWrapper 替换原版僵硬的 Warpper 模型体
        causal_model = DualSpaceSurrogateWrapper(
            lb_win=self.args.seq_len,
            h_win=self.args.pred_len,
            label_win=self.args.label_len,
            feature_dim=self.args.enc_in,
            forecast_model=model1,
            num_regimes=num_regimes,
            proj_dim=proj_dim,
            ablate_routing=ablate_routing,
            ablate_align=ablate_align,
            ablate_uw=ablate_uw,
            ablate_affine=ablate_affine,
            capacity_dropout=capacity_dropout,
            use_irn=use_irn,
            use_vrex=use_vrex,
            vrex_pooled=vrex_pooled
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
        # [容量控制] causal_weight_decay：仅对路由相关参数(原型/上下文编码器)施加 L2 权重衰减，
        #   default 0.0 时 Adam 的 weight_decay=0，与改造前完全一致；短步长过拟合时可设 1e-4~1e-3。
        causal_weight_decay = getattr(self.args, 'causal_weight_decay', 0.0)
        causal_optim = optim.Adam(causal_parameters, lr=1e-4, weight_decay=causal_weight_decay)

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

    def _assemble_total_loss(self, fwd, batch_y, criterion,
                             lambda_orth, lambda_balance, lambda_entropy, lambda_affine,
                             lambda_irn=0.0, lambda_vrex=0.0, vrex_min_env_weight=1.0):
        """
        [集中式损失装配 + 消融单一真源] 把原先在 AMP / 非AMP 两个分支里重复的七项损失计算与加权聚合
        统一到此处，便于消融开关在唯一位置生效，杜绝两分支漂移。
        关键不变性：当四个消融开关全部为 False 时，本函数返回的 total loss 与改造前 v2 逐项一致
        （已由独立数值单测验证 all-off ≡ 原实现）。

        参数:
            fwd: foward_causal 返回的 10 元组。
            batch_y: 真实未来序列 (B, label+pred, feature_dim)。
            criterion: nn.MSELoss。
            lambda_*: 四个结构正则的固定权重（由 train 一次性读取并传入，保证与日志侧同源）。
        返回:
            loss (Tensor 标量): 反传用的总损失。
            comp (dict): 各分项损失张量 + 前向中间量(outputs/gamma/beta/regime_logits/learnable_mask)，供日志与UW初始化读取。
        """
        (x_enc_sel, x_enc_pred, outputs, learnable_mask, y_pred_surrogate,
         gamma, beta, regime_logits, z_align_X, z_align_Y) = fwd
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        dev = outputs.device
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs_y = outputs[:, -self.args.pred_len:, f_dim:]
        target_y = batch_y[:, -self.args.pred_len:, f_dim:]

        # 损失项一：常规物理观测时序预测损失 (MSE)
        loss_observed = criterion(outputs_y, target_y)
        # 损失项二：X空间特征代理边界补全损失 (ShifTS 原理约束)
        loss_x_surrogate = criterion(x_enc_sel, x_enc_pred)

        # 损失项三：Y空间精细化时变混杂清洗损失 (FOIL 变系数原理重构)
        # [消融] ablate_affine 时仿射恒等、此项无意义 → 置 0 且不计入主任务。
        if self.model.ablate_affine:
            loss_y_clean = torch.zeros((), device=dev)
        else:
            with torch.no_grad():
                y_true_norm = self.model.revin1._normalize(batch_y.float())[:, -self.args.pred_len:, -1:]
            # [改动2] stop-gradient：对 gamma/beta 施加 detach，将净化目标固定为常量（详见 v2 注释）。
            y_true_surrogate = (y_true_norm - beta.detach()) / (gamma.detach() + 1e-6)
            loss_y_clean = criterion(y_pred_surrogate, y_true_surrogate)

        # 损失项四：离散机制原型软正交化损失（保留；ablate_routing 下原型仅取均值作单一掩码，正交项影响可忽略）
        normalized_prototypes = F.normalize(self.model.regime_prototypes_X, p=2, dim=1)
        identity_matrix = torch.eye(normalized_prototypes.size(0), device=normalized_prototypes.device)
        prototype_correlation = torch.matmul(normalized_prototypes, normalized_prototypes.transpose(0, 1))
        loss_orthogonal = torch.norm(prototype_correlation - identity_matrix, p='fro') ** 2

        # [改动1] 损失项五：路由负载均衡 + 批级熵正则。[消融] ablate_routing 时路由被均匀化、此正则无意义 → 置 0。
        if self.model.ablate_routing:
            loss_balance = torch.zeros((), device=dev)
            loss_entropy = torch.zeros((), device=dev)
        else:
            loss_balance, loss_entropy = self._routing_regularization(regime_logits)

        # [改动3] 损失项六：仿射趋同恒等先验。[消融] ablate_affine 时仿射已恒等 → 置 0。
        if self.model.ablate_affine:
            loss_affine_anchor = torch.zeros((), device=dev)
        else:
            loss_affine_anchor = torch.mean((gamma - 1.0) ** 2) + torch.mean(beta ** 2)

        # [创新点3] 损失项七：VICReg 式共享隐空间闭环对齐。[消融] ablate_align 时置 0 且不计入主任务。
        if self.model.ablate_align:
            loss_align = torch.zeros((), device=dev)
        else:
            loss_align = self._vicreg_alignment(z_align_X, z_align_Y)

        # [方案C] 四主任务{observed, x_sur, y_clean, align}加权聚合。
        #   [消融] ablate_uw → 四主任务等权(系数1.0)相加；否则按同方差不确定性自适应配权。
        #   y_clean / align 在被各自消融时不计入（保证与"删去该任务"语义一致，且 all-off 时恰为原四项和）。
        s = self.model.log_vars
        if self.model.ablate_uw:
            loss_tasks = loss_observed + loss_x_surrogate
            if not self.model.ablate_affine:
                loss_tasks = loss_tasks + loss_y_clean
            if not self.model.ablate_align:
                loss_tasks = loss_tasks + loss_align
        else:
            loss_tasks = self._uw(loss_observed, s[0]) + self._uw(loss_x_surrogate, s[1])
            if not self.model.ablate_affine:
                loss_tasks = loss_tasks + self._uw(loss_y_clean, s[2])
            if not self.model.ablate_align:
                loss_tasks = loss_tasks + self._uw(loss_align, s[3])

        # -------------------- [FOIL 复刻支路：IRN 代理损失 + VREx 跨环境不变性] (默认关闭) --------------------
        # 二者均为"不变性/结构正则"，用固定权重相加（刻意不进 UW，避免像 y_clean 那样被自动降权关闭）。
        # 默认 use_irn=use_vrex=False 时，loss_irn/loss_vrex 恒为 0，本段对总损失零贡献，与改造前逐位一致。
        # IRN（忠实复刻 FOIL 源码版：仅"去均值"的形状误差，未做论文 Eq.3 的"除标准差"；去均值参考量 detach）：
        pred_shape = outputs_y - outputs_y.mean(dim=1, keepdim=True).detach()
        true_shape = target_y - target_y.mean(dim=1, keepdim=True).detach()
        shape_err_ps = ((pred_shape - true_shape) ** 2).mean(dim=tuple(range(1, outputs_y.dim())))  # (B,) 逐样本去均值残差MSE
        if self.model.use_irn:
            loss_irn = shape_err_ps.mean()                                  # 代理充分性损失 ℓ_suf 的均值项
        else:
            loss_irn = torch.zeros((), device=dev)
        # VREx（用我们现成的"软路由权重"当环境，惩罚跨环境风险方差）：
        #   两种方差形式，由 self.model.vrex_pooled 选择（默认 False=当前行为）：
        #   (a) 默认/Krueger式：有效环境"平均风险"的方差——量级小(各环境均值挤在一起)，FOIL var系数≈0.2~0.3 时易被淹没；
        #   (b) vrex_pooled=True / FOIL源码忠实版：汇集逐样本形状误差的方差(torch.var(cat(valid_loss)))——量级大~100x。
        #   两模式都对环境分配 detach（FOIL"冻结环境标签"思想：方差惩罚只塑造预测器、不让路由为最小化方差而坍缩）。
        vrex_nvalid = 0
        vrex_spread = 0.0
        if self.model.use_vrex:
            rw_v = torch.softmax(regime_logits, dim=-1).detach()           # (B,K) 软环境分配(冻结，不回传路由)
            Pe_v = rw_v.sum(dim=0)                                          # (K,) 各环境的软样本量
            R_e = (rw_v * shape_err_ps.unsqueeze(1)).sum(dim=0) / (Pe_v + 1e-6)  # (K,) 各环境加权平均风险(诊断/Krueger式损失共用)
            if self.model.vrex_pooled:
                # (b) FOIL 源码忠实：按硬 argmax 环境做 min_samples 过滤，对汇集的逐样本形状误差求方差
                env_id = rw_v.argmax(dim=1)                                # (B,) 硬环境分配
                counts = torch.bincount(env_id, minlength=rw_v.size(1)).float()
                keep_envs = counts >= vrex_min_env_weight                  # 样本数≥阈值的环境才计入
                keep_mask = keep_envs[env_id]                              # (B,) 样本是否落在有效环境
                kept = shape_err_ps[keep_mask]
                n_valid = int(keep_envs.sum().item())
                if kept.numel() >= 2 and n_valid >= 2:
                    loss_vrex = kept.var()                                 # 汇集逐样本方差(量级大)
                    vrex_nvalid = n_valid
                else:
                    loss_vrex = torch.zeros((), device=dev)
            else:
                # (a) 默认/Krueger式：有效环境间"平均风险"的方差(量级小)
                valid = Pe_v >= vrex_min_env_weight                        # 丢弃近乎空的环境（类比 FOIL 的 min_samples）
                if int(valid.sum().item()) >= 2:
                    loss_vrex = R_e[valid].var()
                    vrex_nvalid = int(valid.sum().item())
                else:
                    # 有效环境 < 2（典型为路由坍缩到单一机制）→ VREx 本批失效，置 0 并记 nvalid 供诊断
                    loss_vrex = torch.zeros((), device=dev)
            # 诊断：环境风险极差(两模式通用，反映跨环境异质强度)
            with torch.no_grad():
                _vd = Pe_v >= vrex_min_env_weight
                if int(_vd.sum().item()) >= 2:
                    _rv = R_e[_vd]
                    vrex_spread = float((_rv.max() - _rv.min()).item())
        else:
            loss_vrex = torch.zeros((), device=dev)

        loss = (loss_tasks
                + lambda_orth * loss_orthogonal
                + lambda_balance * loss_balance
                + lambda_entropy * loss_entropy
                + lambda_affine * loss_affine_anchor
                + lambda_irn * loss_irn
                + lambda_vrex * loss_vrex)

        comp = dict(loss_observed=loss_observed, loss_x_surrogate=loss_x_surrogate,
                    loss_y_clean=loss_y_clean, loss_align=loss_align, loss_orthogonal=loss_orthogonal,
                    loss_balance=loss_balance, loss_entropy=loss_entropy, loss_affine_anchor=loss_affine_anchor,
                    loss_irn=loss_irn, loss_vrex=loss_vrex, vrex_nvalid=vrex_nvalid, vrex_spread=vrex_spread,
                    outputs=outputs, gamma=gamma, beta=beta, regime_logits=regime_logits,
                    learnable_mask=learnable_mask)
        return loss, comp

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
        # [FOIL 复刻支路·固定权重] 仅当 use_irn/use_vrex 开启时生效；关闭时对应损失项恒为 0，权重取值无影响。
        #   FOIL 源码中 VREx(var) 系数约 0.2~0.3、ERM(avg) 约 0.7~0.8，故 lambda_vrex 默认取 0.2 量级起步；
        #   IRN 作为辅助充分性损失，lambda_irn 默认 0.5。二者均需按数据集微调。
        lambda_irn = getattr(self.args, 'lambda_irn', 0.5)
        lambda_vrex = getattr(self.args, 'lambda_vrex', 0.2)
        vrex_min_env_weight = getattr(self.args, 'vrex_min_env_weight', 1.0)  # VREx 计入环境的最小软样本量阈值

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
                              'balance': [], 'entropy': [], 'affine': [], 'irn': [], 'vrex': []}
            log_diag = {'route_entropy_norm': [], 'route_entropy_persample': [], 'P_e': [],
                        'gamma_mean': [], 'gamma_dev': [], 'beta_mean': [], 'beta_abs': [],
                        'vrex_nvalid': [], 'vrex_spread': []}
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

                # -------------------- 前向传播 + 集中式多任务损失装配(含消融) --------------------
                # 前向在 AMP/非AMP 下分别(autocast 内/外)调用；损失装配统一走 _assemble_total_loss，
                # 四个消融开关只在该辅助函数(及前向)内生效，两分支不再各写一遍，杜绝逻辑漂移。
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        fwd = self.model.foward_causal(
                            batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                        loss, comp = self._assemble_total_loss(
                            fwd, batch_y, criterion,
                            lambda_orth, lambda_balance, lambda_entropy, lambda_affine,
                            lambda_irn, lambda_vrex, vrex_min_env_weight)
                else:
                    fwd = self.model.foward_causal(
                        batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                    loss, comp = self._assemble_total_loss(
                        fwd, batch_y, criterion,
                        lambda_orth, lambda_balance, lambda_entropy, lambda_affine,
                        lambda_irn, lambda_vrex, vrex_min_env_weight)
                train_loss.append(loss.item())

                # 从装配结果中取出分项损失与前向中间量(供 UW 初始化与日志读取)
                loss_observed = comp['loss_observed']
                loss_x_surrogate = comp['loss_x_surrogate']
                loss_y_clean = comp['loss_y_clean']
                loss_align = comp['loss_align']
                regime_logits = comp['regime_logits']
                gamma = comp['gamma']; beta = comp['beta']
                learnable_mask = comp['learnable_mask']

                # [改动B] 仅在首个 batch：用四个主任务的真实损失量级初始化 log_vars（s_i = log L_i），
                # 使四项有效贡献(0.5*exp(-s)*L)开局即被拉平至 ~0.5，避免被放大的 y_clean/align 在前期霸占梯度。
                # 本数据集无关（ETT/ILI/Exchange 皆自校准）。首个 batch 的总损失已用 log_vars=0 反传一次，无碍。
                # [消融] ablate_uw 时不使用不确定性加权，跳过此初始化。
                if (not self.model.ablate_uw) and (not self._uw_init_done):
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
                    log_components['orth'].append(comp['loss_orthogonal'].item())
                    log_components['balance'].append(comp['loss_balance'].item())
                    log_components['entropy'].append(comp['loss_entropy'].item())
                    log_components['affine'].append(comp['loss_affine_anchor'].item())
                    # 路由健康度：归一化使用率熵 ∈ [0,1]（1=机制被完全均匀使用，→0=坍缩到单一机制）
                    # 注：ablate_routing 下实际使用均匀权重，此处仍报告 context_encoder 原始 logits 的熵（仅诊断参考）。
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
                    # [FOIL 复刻支路] 累计 IRN/VREx 损失与 VREx 环境健康度（关闭时这些值恒为 0，无碍）
                    log_components['irn'].append(comp['loss_irn'].item())
                    log_components['vrex'].append(comp['loss_vrex'].item())
                    log_diag['vrex_nvalid'].append(comp['vrex_nvalid'])
                    log_diag['vrex_spread'].append(comp['vrex_spread'])

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
            # [FOIL 复刻支路诊断] 仅在开启 IRN/VREx 时打印（关闭时不输出，保持基线日志格式不变）
            if self.model.use_irn or self.model.use_vrex:
                _vn_mean = _avg(log_diag['vrex_nvalid'])
                _vmode = 'pooled汇集逐样本' if self.model.vrex_pooled else '逐环境均值'
                print("  [诊断·FOIL支路] IRN(加权前)={:.5f}(贡献={:.5f}) | VREx[{}](加权前)={:.6f}(贡献={:.6f}) | "
                      "VREx有效环境数(epoch均值)={:.2f}/{} | 环境风险极差(epoch均值)={:.5f}".format(
                          _avg(log_components['irn']), lambda_irn * _avg(log_components['irn']),
                          _vmode, _avg(log_components['vrex']), lambda_vrex * _avg(log_components['vrex']),
                          _vn_mean, (_nR if _nR else self.model.num_regimes), _avg(log_diag['vrex_spread'])))
                print("    └─ 解读：VREx有效环境数<2 表示路由坍缩、VREx 本批被置0而失效；环境风险极差越大说明跨环境异质越强、VREx 越有发力空间")
                if self.model.use_vrex and (not np.isnan(_vn_mean)) and _vn_mean < 2.0:
                    print("  [诊断·FOIL支路·告警] VREx 平均有效环境数<2 → VREx 实际常被置0(路由坍缩)；"
                          "考虑增大 lambda_balance/lambda_entropy 或下调 vrex_min_env_weight")

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

