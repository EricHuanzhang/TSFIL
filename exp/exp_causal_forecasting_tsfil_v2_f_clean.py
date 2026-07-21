from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from copy import copy

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
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

import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class DualSpaceSurrogateWrapper(nn.Module):
    def __init__(self, lb_win, h_win, label_win, feature_dim, forecast_model, num_regimes=5, proj_dim=128):
        super(DualSpaceSurrogateWrapper, self).__init__()
        self.lb_win = lb_win
        self.h_win = h_win
        self.label_win = label_win
        self.feature_dim = feature_dim
        self.forecast_model = forecast_model

        self.revin1 = RevIN(num_features=feature_dim)

        self.context_encoder = nn.Sequential(
            spectral_norm(nn.Linear(lb_win * feature_dim, 128)),
            nn.Tanh(),
            spectral_norm(nn.Linear(128, num_regimes))
        )

        self.regime_prototypes_X = nn.Parameter(torch.randn(num_regimes, (lb_win + 1) * (feature_dim - 1)))

        self.linear = nn.Sequential(*[
            nn.Linear(feature_dim - 1, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ])

        self.proj_dim = proj_dim
        self.proj_X = nn.Sequential(
            nn.Linear(feature_dim - 1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.proj_Y = nn.Sequential(
            nn.Linear(1, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def foward_causal(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]

        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))
        regime_weights = torch.softmax(regime_logits, dim=-1)

        gated_mask_X = torch.mm(regime_weights, self.regime_prototypes_X)
        gated_mask_X = gated_mask_X.reshape(batch_size, self.lb_win + 1, self.feature_dim - 1)

        learnable_mask = torch.softmax(gated_mask_X, dim=1)
        mask_mean = torch.mean(learnable_mask, dim=1, keepdim=True)
        condition = (learnable_mask - mask_mean) > 0
        learnable_mask = learnable_mask * condition
        learnable_mask = learnable_mask / (torch.sum(learnable_mask, dim=1, keepdim=True) + 1e-6)

        last_dim = torch.zeros(batch_size, self.lb_win + 1, 1).to(learnable_mask.device)
        last_dim[:, 0, 0] = 1.0
        learnable_mask = torch.cat((learnable_mask, last_dim), dim=-1)

        x_pred_sel = torch.sum(x_enc_all.permute(0, 2, 1, 3) * learnable_mask.unsqueeze(1), -2)

        x_enc_norm = self.revin1(x_enc, mode='norm')
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]
        y_pred_base = x_pred[:, :, -1:]

        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)

        x_out = torch.cat((x_pred_features, y_pred_surrogate), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')

        z_align_X = self.proj_X(x_pred_features.reshape(-1, self.feature_dim - 1))
        z_align_Y = self.proj_Y(y_pred_surrogate.reshape(-1, 1))

        return (x_pred_sel[:, :, :-1], x_out[:, :, :-1], x_out, learnable_mask,
                y_pred_surrogate, regime_logits, z_align_X, z_align_Y)

    def foward_feature(self, x_enc_all, x_enc, x_mark_enc, x_dec, x_mark_dec, return_diag=False):
        batch_size = x_enc.shape[0]

        regime_logits = self.context_encoder(x_enc.reshape(batch_size, -1))
        regime_weights = torch.softmax(regime_logits, dim=-1)

        x_enc_norm = self.revin1(x_enc, mode='norm')
        x_pred = self.forecast_model(x_enc_norm, x_mark_enc, x_dec, x_mark_dec)

        x_pred_features = x_pred[:, :, :-1]
        y_pred_base = x_pred[:, :, -1:]

        y_pred_surrogate = y_pred_base + self.linear(x_pred_features)
        x_out = torch.cat((x_pred_features, y_pred_surrogate), dim=-1)
        x_out = self.revin1(x_out, mode='denorm')

        if return_diag:
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
        print("Exp Causal Forecast-Model: Dual Space Surrogate Wrapper v2_f "
              "(精简损失: loss_observed + loss_x_surrogate + 0.1*loss_align + 0.1*loss_suf)")
        self.args.task_name = 'long_term_forecast'
        model1 = self.model_dict[self.args.model].Model(self.args).float()

        seq_len = copy(self.args.seq_len)
        pred_len = copy(self.args.pred_len)
        label_len = copy(self.args.label_len)
        self.args.pred_len = seq_len
        self.args.label_len = seq_len
        self.args.pred_len = pred_len
        self.args.label_len = label_len

        self.args.task_name = 'causal_forecast'

        num_regimes = getattr(self.args, 'num_regimes', 5)

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
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        causal_component_names = [
            'regime_prototypes_X',
            'context_encoder'
        ]

        causal_parameters = []
        other_parameters = []
        for name, param in self.model.named_parameters():
            if any(cn in name for cn in causal_component_names):
                causal_parameters.append(param)
            else:
                other_parameters.append(param)

        causal_optim = optim.Adam(causal_parameters, lr=1e-4)
        model_optim = optim.Adam(other_parameters, lr=self.args.learning_rate)
        return model_optim, causal_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    @staticmethod
    def _irn_surrogate_loss(y_pred_surrogate, y_true_norm, mode='demean', eps=1e-5):
        mu_P = y_pred_surrogate.mean(dim=1, keepdim=True).detach()
        mu_Y = y_true_norm.mean(dim=1, keepdim=True).detach()
        if mode == 'standardize':
            sd_P = y_pred_surrogate.std(dim=1, keepdim=True, unbiased=False).detach() + eps
            sd_Y = y_true_norm.std(dim=1, keepdim=True, unbiased=False).detach() + eps
            pred_n = (y_pred_surrogate - mu_P) / sd_P
            true_n = (y_true_norm - mu_Y) / sd_Y
        else:
            pred_n = y_pred_surrogate - mu_P
            true_n = y_true_norm - mu_Y
        res = true_n - pred_n
        loss_suf = (res ** 2).mean()
        res_std = res.detach().std()
        return loss_suf, res_std

    def _vicreg_alignment(self, z_x, z_y, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04, eps=1e-4):
        N, D = z_x.shape
        denom = max(N - 1, 1)

        inv = F.mse_loss(z_x, z_y)

        std_x = torch.sqrt(z_x.var(dim=0) + eps)
        std_y = torch.sqrt(z_y.var(dim=0) + eps)
        var = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

        zx = z_x - z_x.mean(dim=0)
        zy = z_y - z_y.mean(dim=0)
        cov_x = (zx.T @ zx) / denom
        cov_y = (zy.T @ zy) / denom
        off_x = (cov_x.pow(2).sum() - cov_x.pow(2).diagonal().sum()) / D
        off_y = (cov_y.pow(2).sum() - cov_y.pow(2).diagonal().sum()) / D
        cov = off_x + off_y

        return sim_coeff * inv + std_coeff * var + cov_coeff * cov

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
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

        std_coeff = getattr(self.args, 'std_coeff', 1.0)
        irn_mode = getattr(self.args, 'irn_mode', 'demean')
        stop_mode = getattr(self.args, 'stop_mode', 'early_stop')

        lambda_loss_suf = getattr(self.args, 'lambda_loss_suf', 0.1)
        lambda_loss_align = getattr(self.args, 'lambda_loss_align', 0.1)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            log_components = {'observed': [], 'x_sur': [], 'suf': [], 'align': []}
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

                def compute_losses():
                    (x_enc_sel, x_enc_pred, outputs, learnable_mask, y_pred_surrogate,
                     regime_logits, z_align_X, z_align_Y) = self.model.foward_causal(
                        batch_x, batch_x_en, batch_x_mark, dec_inp, batch_y_mark
                    )

                    out_y = outputs[0] if isinstance(outputs, tuple) else outputs
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs_y = out_y[:, -self.args.pred_len:, f_dim:]
                    target_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss_observed = criterion(outputs_y, target_y)

                    loss_x_surrogate = criterion(x_enc_sel, x_enc_pred)

                    with torch.no_grad():
                        y_true_norm = self.model.revin1._normalize(batch_y.float())[:, -self.args.pred_len:, -1:]
                    loss_suf, irn_res_std = self._irn_surrogate_loss(y_pred_surrogate, y_true_norm, mode=irn_mode)

                    loss_align = self._vicreg_alignment(z_align_X, z_align_Y, std_coeff=std_coeff)

                    t_loss = loss_observed + loss_x_surrogate + lambda_loss_align * loss_align + lambda_loss_suf * loss_suf

                    with torch.no_grad():
                        projY_std = torch.sqrt(z_align_Y.var(dim=0) + 1e-4).mean()
                        align_inv = F.mse_loss(z_align_X, z_align_Y)

                    return (t_loss, loss_observed, loss_x_surrogate, loss_suf, loss_align,
                            irn_res_std, projY_std, align_inv, learnable_mask, regime_logits)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        (loss, loss_obs, loss_x_sur, loss_suf, loss_aln,
                         irn_std, projY_std, align_inv, mask, r_logits) = compute_losses()
                else:
                    (loss, loss_obs, loss_x_sur, loss_suf, loss_aln,
                     irn_std, projY_std, align_inv, mask, r_logits) = compute_losses()

                train_loss.append(loss.item())

                with torch.no_grad():
                    log_components['observed'].append(loss_obs.item())
                    log_components['x_sur'].append(loss_x_sur.item())
                    log_components['suf'].append(loss_suf.item())
                    log_components['align'].append(loss_aln.item())

                    _rw = torch.softmax(r_logits, dim=-1)
                    _Pe = _rw.mean(dim=0)
                    _R = _Pe.size(-1)
                    _H = -torch.sum(_Pe * torch.log(_Pe + 1e-8)).item()
                    log_diag['route_entropy_norm'].append(_H / float(np.log(max(_R, 2))))
                    _H_ps = -(_rw * torch.log(_rw + 1e-8)).sum(dim=-1).mean().item()
                    log_diag['route_entropy_persample'].append(_H_ps / float(np.log(max(_R, 2))))
                    log_diag['P_e'].append(_Pe.detach().float().cpu())
                    log_diag['irn_res_std'].append(float(irn_std.item()))
                    log_diag['projY_std'].append(float(projY_std.item()))
                    log_diag['align_inv'].append(float(align_inv.item()))

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

            print("Epoch: {} 耗时总长: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | 综合训练Loss: {2:.7f} | 验证集Loss: {3:.7f} | 测试集Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            def _avg(_l):
                return float(np.mean(_l)) if len(_l) else float('nan')
            _mean_Pe = torch.stack(log_diag['P_e']).mean(0).numpy() if len(log_diag['P_e']) else None
            _re = _avg(log_diag['route_entropy_norm'])
            _nR = (len(_mean_Pe) if _mean_Pe is not None else 0)
            _active = int((_mean_Pe > (1.0 / (2 * _nR))).sum()) if (_mean_Pe is not None and _nR > 0) else -1

            print("  [诊断·分项损失] observed={:.5f} x_sur={:.5f} suf={:.5f} align={:.5f}".format(
                _avg(log_components['observed']), _avg(log_components['x_sur']),
                _avg(log_components['suf']), _avg(log_components['align'])))
            print("  [诊断·路由] 批级使用率熵(epoch均值)={:.3f} 逐样本熵={:.3f} | 激活机制数={}/{} | 平均使用率 P_e={}".format(
                _re, _avg(log_diag['route_entropy_persample']), _active, _nR,
                np.round(_mean_Pe, 3) if _mean_Pe is not None else None))
            print("    └─ 解读：批级高 + 逐样本低 = 健康特化(不同样本走不同机制)；两者都≈1 = 各样本均匀混合(环境条件化未特化)")
            print("  [诊断·Y代理] IRN残差std={:.4f} | projY_std(VICReg方差锚,目标1)={:.4f} | 闭环对齐MSE={:.4f}".format(
                _avg(log_diag['irn_res_std']), _avg(log_diag['projY_std']), _avg(log_diag['align_inv'])))
            print("    └─ 解读：projY_std 应稳定≈1（坍缩<0.5→增大 --std_coeff）；IRN残差std≈O(1) 表示 Y 代理在学形状而非趋平凡。")

            if (not np.isnan(_re)) and _re < 0.5:
                print("  [诊断·告警] 路由熵偏低({:.3f}<0.5)，环境路由可能正在坍缩（单机制 num_regimes=1 时熵恒为 0，可忽略）".format(_re))

            if stop_mode == 'early_stop':
                early_stopping(test_loss, self.model, path)
                if early_stopping.early_stop:
                    print("触发早停条件，系统在纯净不变量边界安全收敛。")
                    break
            elif stop_mode == 'last':
                if epoch + 1 >= int(self.args.train_epochs):
                    early_stopping.save_checkpoint(test_loss, self.model, path)
                    print("达到最后一个epoch，保存模型")
            else:
                print("早停策略未定义")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('从持久化 Checkpoint 载入最优因果泛化模型体...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
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