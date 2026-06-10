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
class Warpper(nn.Module):
    """
    - `lb_win`：历史 Lookback 窗口长度 $L$（输入时间步长）。
    - `h_win`：未来预测 Horizon 窗口长度 $H$。
    - `label_win`：解码器起始 Token 长度。
    - `feature_dim`：特征的总通道数（包括所有外生变量和 1 列目标变量）。
    - `forecast_model`：底座多变量时间序列预测网络。
    """
    def __init__(self, lb_win, h_win, label_win, feature_dim, forecast_model):
        super(Warpper, self).__init__()
        # 对应论文中的 软注意力掩码矩阵M,维度为 `(L + 1, d_X)`,
        # 其中 L+1 是滑窗切片数，`feature_dim - 1` 是外生变量个数（去掉了最后的 Y 通道）。
        self.mask = nn.Parameter(torch.rand(lb_win+1, feature_dim-1), requires_grad=True)

        self.revin1 = RevIN(num_features=feature_dim)
        self.revin2 = RevIN(num_features=1)

        # 声明两个可学习的权重标量，默认初始化为 0.5。它们可用于后续多任务 Loss
        # （如预测目标 Loss 与不变量预测 Loss）的自适应动态加权。
        self.w1 = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)

        self.lb_win = lb_win
        self.label_win = label_win

        self.forecast_model = forecast_model    #将传入的方法无关（Model-Agnostic）的底座预测网络（如 `iTransformer`）绑定至包装器

        """
        对应论文中的非线性特征因果聚合算子 Agg()
        - 构建一个三层 MLP：输入维度为外生变量数 `feature_dim - 1`，经过隐藏层 `512` 节点并由 `ReLU` 激活，最终输出维度为 `1`。
        - 它的物理职责是将隐式预测得到的黄金替代特征映射为“因果环境漂移修正项”，以便残差注入到目标变量Y的预测通道中。
        """
        self.linear = nn.Sequential(*[
            nn.Linear(feature_dim-1, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        ])

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
        zz = torch.ones(self.mask.shape).to(x_enc.device)

        # mask维度：(L + 1, feature_dim - 1)
        # 既保留了self.mask 的可训练梯度，又生成了一个安全的“临时替身” learnable_mask，
        # 以便随意执行后续的稀疏门控置零、归一化与通道拼接，是保证 ShifTS 因果不变量过滤机制能够稳定运转的重要底层防线。
        learnable_mask = self.mask * zz

        # 让矩阵沿着第 0 维（纵向/行维度）进行归一化，使得每一个通道（每一列）上的所有元素值在经过 Softmax 激活后，
        # 范围都处于 [0, 1] 之间，并且它们的累加之和等于 1。
        learnable_mask = torch.softmax(learnable_mask, dim=0)

        # - 计算各特征通道在所有L+1个滑窗上的平均权重，判断哪些切片位置的权重超越了平均线，生成一个布尔张量（True/False）。
        # - 硬阈值门控作用：用来筛除权重过低、仅在局部偶然相关的瞬态虚假噪声规律。
        condition = (learnable_mask - torch.mean(learnable_mask, dim=0)) > 0
        # condition = (learnable_mask - torch.amax(learnable_mask, dim=0) + 1e-6) > 0

        #通过与布尔张量进行点乘（True 保持原权重，False 权重直接清零），实现稀疏化（Sparsity）过滤
        learnable_mask = learnable_mask * condition
        # 由于部分弱不变量被清零，剩余活跃权重的累加和不再为 1。这里对过滤后的掩码重新进行一范数归一化（Normalize）。
        learnable_mask = learnable_mask / torch.sum(learnable_mask, dim=0)

        # 构建目标变量 $Y$ 专属的注意力权重向量，长度为 $L+1$，除了第 `0` 个元素设为 1 外，其余全设为 0。
        # - len(self.mask) —— 获取行数即L+1（切片总数）（对一个二维张量执行 len()，默认会返回其第 0 维（行维度）的长度）
        # - torch.zeros(len(self.mask))创建一个形状为 (L + 1,) 的一维全 0 向量。
        # - .reshape(-1, 1) —— 升维变成“列向量” (张量维度重塑（Reshape）):它将原本形状为 (L + 1,)
        #                      的一维普通张量，转变成了形状为 (L + 1, 1) 的二维矩阵（物理上表现为一列纵向排列的列向量）。
        last_dim = torch.zeros(len(self.mask)).reshape(-1,1).to(learnable_mask.device) #1 的含义：明确指定重塑后的新张量，其最后一维（第 1 维）的长度必须是 1（即让它只有 1 列）。
        last_dim[0] = 1

        # 将包含 `feature_dim - 1` 列的外生不变量掩码与这一列独热的目标变量掩码进行通道拼接，合成为一个形状为 `(L+1, feature_dim)` 的全通道掩码矩阵。
        learnable_mask = torch.cat((learnable_mask, last_dim), dim=-1)

        # x_pred_sel: 前feature_dim-1列即为X^{SUR} 黄金替代特征
        # $X^{SUR} = \sum_{k=1}^{L+1} \mathcal{M}_k \cdot Slice([X^L, X^H])_k$
        # x_enc_all: [B, L+1, H, feature_dim]
        # learnable_mask: [L+1, feature_dim]
        # x_pred_sel: [B, H, feature_dim], 前 `feature_dim-1` 个通道代表了训练期根据全量未来信息过滤提炼出的黄金替代特征X^{SUR}；最后一个通道是未来的真实目标
        # torch.sum(..., -2): 在倒数第 2 维（L+1 切片维度）执行加权求和，将多源局部不变量加权压缩。
        # 每一个通道的外生变量去跨时空过滤、整合历史因果不变量时，所使用的“滤镜权重”（即 `learnable_mask`）是完全共享、绝对恒定的
        # 这种机制强行逼迫【可学习掩码】去挖掘那些“超越具体未来相对时刻”的恒久定律
        # x_pred_sel: 训练期间通过未来真实数据计算获得的真实X^{SUR}，作为未来预测值的外生变量，注意是真实值。
        # [B, H, L+1, feature_dim] * [L+1, feature_dim] ——> [B, H, feature_dim]
        x_pred_sel = torch.sum(x_enc_all.permute(0,2,1,3) * learnable_mask, -2)

        x_enc = self.revin1(x_enc, mode='norm')

        # 调用底座多变量预测网络，进行前向传播，输出形状为 `(B, H, feature_dim)` 的未来多通道预测
        # 送入 MLP 因果聚合算子 `self.linear` 计算对冲因果漂移的环境偏移修正量
        # 与模型常规目标通道输出的基础预测x_pred[:,:,-1:]进行残差相加，实现因果对冲，计算出标准空间下的完美目标预测x_pred_last
        """
        - `x_enc`：历史 Lookback 序列，形状为 `(B, L, feature_dim)`。
        - `x_mark_enc`：Lookback 时间戳特征。
        - `x_dec`/`x_mark_dec`：解码器占位输入及对应时间戳特征。
        """
        x_pred = self.forecast_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # x_pred[:,:,:-1]: 代表模型【预测出的】未来替代特征表达 \hat{X}_{Norm}^{SUR}，注意不是真实值，是预测出的
        # x_pred[:, :, :-1]	取前feature_dim-1个，特征维度变为[B, H, feature_dim-1]
        # x_pred[:, :, -1:]	取feature_dim的最后一个维度[B, H, 1]
        x_pred_last = self.linear(x_pred[:,:,:-1]) + x_pred[:,:,-1:] #因果解耦，x_pred[:,:,:-1]（即\hat{X}^{SUR}）为因，x_pred[:,:,-1:]为果
        x_out = torch.cat((x_pred[:,:,:-1], x_pred_last[:,:,-1:]), dim=-1) #沿着最后一个维度（即特征维）将两个张量拼接在一起。
        x_out = self.revin1(x_out, mode='denorm')

        """
        - x_pred_sel:训练期间通过未来真实数据计算获得的真实X^{SUR}
        - x_pred_sel[:,:,:-1]:
        - x_out[:,:,:-1]: 
        """
        return x_pred_sel[:,:,:-1], x_out[:,:,:-1], x_out, learnable_mask

    """
    推理前向传播方法 `foward_feature`
    - x_enc_all：全量时间轴滑窗切片，形状为 (B, L + 1, H, feature_dim)（包含历史和未来拼接后切片出的 $L+1$ 个候选局部特征窗口）。
    - x_enc：历史 Lookback 序列，形状为 (B, L, feature_dim)。
    - x_mark_enc：Lookback 时间戳特征。
    - x_dec/x_mark_dec：解码器占位输入及对应时间戳特征。
    """
    def foward_feature(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.revin1(x_enc, mode='norm')

        # 一阶段：底座模型基于已经经过梯度约束的外生预测通道，隐式预测出泛化不变量外生表达。
        x_pred = self.forecast_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 二阶段：将隐式外生不变量通过 Agg MLP 算子残差融合成因果偏移量，校准目标:\hatY_{NORM}
        x_pred_last = self.linear(x_pred[:,:,:-1]) + x_pred[:,:,-1:]

        x_out = torch.cat((x_pred[:,:,:-1], x_pred_last[:,:,-1:]), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')
        return x_out

    def reset_model(self, forecast_model, causal_model):
        self.forecast_model = forecast_model.to(self.mask.device)
        self.translation_model = causal_model.to(self.mask.device)


class Exp_Causal_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Causal_Forecast, self).__init__(args)

    def _build_model(self):
        """
        - 将任务名临时设为 `long_term_forecast`，从而让底座预测网络的初始化流程走标准的“长期预测”模式（不带有任何外界包裹结构）。
        - 实例化一个标准的多变量时序底座模型（如 `iTransformer`）。
        """
        self.args.task_name = 'long_term_forecast'
        model1 = self.model_dict[self.args.model].Model(self.args).float()
        """
        lb_win：历史窗口长度 seq_len
        h_win：预测窗口长度 pred_len
        label_win：解码器起始标记长度 label_len
        feature_dim：输入特征维度 enc_in
        forecast_model：刚刚构建的底座模型 model1
        """
        seq_len = copy(self.args.seq_len)
        pred_len = copy(self.args.pred_len)
        label_len = copy(self.args.label_len)
        self.args.pred_len = seq_len
        self.args.label_len = seq_len
        self.args.pred_len = pred_len
        self.args.label_len = label_len

        self.args.task_name = 'causal_forecast' #将实验任务名正式切回 `causal_forecast`（时空偏移解耦预测）

        # 将标准底座 model1装入我们前面拆解的Warpper类中，生成最终的 ShifTS 架构体causal_model。
        causal_model = Warpper(lb_win=self.args.seq_len, h_win=self.args.pred_len, label_win=self.args.label_len, feature_dim=self.args.enc_in, forecast_model=model1)

        return causal_model

    """
    - 根据传入的标识（`train`/`val`/`test`）调用 `data_factory.py` 的 `data_provider`。
    - 注意：此时配置项为 `customc` 或 `ETT*c`，Dataloader 内部重写的 `__getitem__` 将额外返回利用 Lookback 和 Horizon 全局滑窗切好的模式矩阵 `seq_xall`。
    """
    def _get_data(self, flag):
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
        mask_parameters = [self.model.mask]
        other_parameters = [param for name, param in self.model.named_parameters() if name != 'mask']

        # 掩码的优化需要极为平滑和保守，防止过快地坍缩到局部极值（导致不变量规律提炼不全），因此其学习率被硬编码锁死在 `1e-4`。
        # 其余庞大的深度学习权重采用用户配置的学习率 `args.learning_rate`。
        # 两者在训练循环中交替梯度清零、反向传播与单步更新。
        causal_optim = optim.Adam(mask_parameters, lr=1e-4)
        model_optim = optim.Adam(other_parameters, lr=self.args.learning_rate)

        return model_optim, causal_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            """
            Dataloader 返回变量说明:
            - `batch_x`：训练期全量滑窗不变量特征 `seq_xall`。
            - `batch_x_en`：历史外生和目标的 Lookback 序列 `seq_x`。
            - `batch_y`：未来的真实目标与外生序列 $Y^H$。
            - `batch_x_mark` / `batch_y_mark`：对应的时间戳。
            """
            for i, (batch_x, batch_x_en, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                batch_x_en = batch_x_en.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 解码器输入构造
                """
                构造解码器输入 dec_inp，其包含两部分：
                - 已知的历史序列（作为“起始令牌”）
                - 待预测的位置（初始填充为 0）
                - 通常用于自回归解码或一次性预测（如 Transformer 中的解码器输入格式）。
                """
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() #创建一个形状为 (B, pred_len, D) 的全零张量，用于存放待预测的未来时间步（初始时未知，通常用 0 或其它占位符填充）。

                #沿时间维（第二维） 拼接，结果形状为 (B, label_len + pred_len, D)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                """
                self.args.use_amp
                -含义
                AMP 是 Automatic Mixed Precision（自动混合精度）的缩写;该参数为布尔值（True / False），表示是否在训练和验证过程中启用混合精度训练。
                -作用
                混合精度训练同时使用 float16（半精度）和 float32（单精度）数据类型：前向传播和梯度计算使用 float16，可显著减少显存占用并加速计算。
                关键操作（如损失缩放、参数更新）保留 float32 以防止数值下溢。
                在代码中，当 self.args.use_amp == True 时，会使用 torch.cuda.amp.autocast() 上下文管理器包裹前向计算，并使用 torch.cuda.amp.GradScaler 对梯度进行缩放（scaler.scale(loss).backward()、scaler.step(optimizer)、scaler.update()）。
                """
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 调用 foward_feature,依据历史进行推理，得到预测值 outputs
                        # : [B, pred_len, feature_dim]
                        if self.args.output_attention:  #该参数为布尔值，表示模型的前向传播是否返回注意力权重,用于可视化分析。
                            outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                # 当 features 为 'MS'（多变量预测单变量）时，只取最后一维（目标变量）；否则取所有特征。
                f_dim = -1 if self.args.features == 'MS' else 0

                # outputs: 取f_dim及之后的所有变量，以及最后pred_len个时间步
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs ):
            iter_count = 0
            train_loss = []

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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #当 output_attention == True 时，底座预测模型（如 iTransformer）的前向函数会返回一个元组 (output, attentions)，其中 attentions 是各注意力层的权重矩阵（列表形式）
                        if self.args.output_attention:
                            """
                            x_enc_sel：从未来数据中加权聚合得到的真实黄金替代特征（形状 (B, pred_len, feature_dim-1)）
                            x_enc_pred：底座模型预测出的外生特征部分（形状 (B, pred_len, feature_dim-1)）
                            outputs：最终预测（形状 (B, label_len+pred_len, feature_dim)）
                            learnable_mask：经过 softmax、硬阈值门控、归一化后的掩码矩阵
                            """
                            x_enc_sel, x_enc_pred, outputs_, learnable_mask = self.model.foward_causal(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = outputs_[0]   # 取出第一个返回值（预测值），忽略注意力
                        else:
                            x_enc_sel, x_enc_pred, outputs, learnable_mask = self.model.foward_causal(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        #预测损失：模型最终预测值 outputs 与真实目标 batch_y 的 MSE
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]

                        #因果一致性损失：从未来数据中提取的“真实”黄金替代特征 x_enc_sel 与底座模型预测出的外生特征 x_enc_pred 之间的 MSE。
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) + criterion(x_enc_sel, x_enc_pred)
                        # loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        x_enc_sel, x_enc_pred, outputs_, learnable_mask = self.model.foward_causal(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs_[0]
                    else:
                        x_enc_sel, x_enc_pred, outputs, learnable_mask = self.model.foward_causal(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + criterion(x_enc_sel, x_enc_pred)
                    # loss = criterion(x_enc_sel, x_enc_pred)
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # print(learnable_mask)
                    print(torch.argmax(learnable_mask, dim=0))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    if epoch+1 <= int(self.args.train_epochs):
                        scaler.step(causal_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if epoch+1 <= int(self.args.train_epochs):
                        causal_optim.step()


            if epoch+1 > 0:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            # 调整学习率（仅针对 model_optim，掩码优化器学习率固定）。
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # adjust_learning_rate(causal_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model.foward_feature(batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                #构造解码器输入 dec_inp：先创建一个全零张量，形状为 (B, pred_len, feature_dim)，然后
                #batch_y 的前 label_len 个时间步在时间维拼接，得到 (B, label_len+pred_len, feature_dim)
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

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x_en.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_causal_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
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
