# utils/scalers.py
import numpy as np
from sklearn.preprocessing import RobustScaler,MinMaxScaler

class HeterogeneousScaler:
    """
    终极异构缩放器 (Heterogeneous Scaler)
    自动识别 连续变量、多状态离散变量 与 恒定死线变量。
    支持在滑动窗口（流式处理）中继承历史有效状态，并完美支持 inverse_transform 精准反归一化。
    """

    def __init__(self, discrete_threshold=20, constant_tol=1e-6):
        self.discrete_threshold = discrete_threshold
        self.constant_tol = constant_tol

        # 为每个维度独立维护缩放器实例字典
        self.scalers = {}

        # 状态继承核心机制
        self.last_valid_mode = None
        self.current_mode = None

        # 兼容旧版本代码的布尔掩码属性
        self.is_discrete_mask = None

    def fit(self, data):
        _, num_cols = data.shape

        # 初始化全局记忆与缩放器字典
        if self.last_valid_mode is None:
            self.last_valid_mode = ['unfitted'] * num_cols
            self.scalers = {i: None for i in range(num_cols)}

        self.current_mode = ['unfitted'] * num_cols
        self.is_discrete_mask = np.zeros(num_cols, dtype=bool)

        for i in range(num_cols):
            col_data = data[:, i]
            col_clean = col_data[~np.isnan(col_data)]

            if len(col_clean) == 0:
                self.current_mode[i] = self.last_valid_mode[i]
                continue

            # 1. 常量特征探测 (Constant Bypass & State Inheritance)
            if (np.max(col_clean) - np.min(col_clean)) <= self.constant_tol:
                # 遇到常数窗口：不执行新的 fit，当前模式直接继承上一次的有效模式
                self.current_mode[i] = self.last_valid_mode[i]
            else:
                # 2. 正常波动窗口探测与拟合
                rounded_data = np.round(col_clean, decimals=5)
                unique_vals = np.unique(rounded_data)

                # 离散特征探测
                if len(unique_vals) <= self.discrete_threshold:
                    scaler = MinMaxScaler()
                    scaler.fit(col_clean.reshape(-1, 1))
                    self.scalers[i] = scaler
                    self.current_mode[i] = 'discrete'
                    self.last_valid_mode[i] = 'discrete'

                # 连续特征探测
                else:
                    scaler = RobustScaler()
                    scaler.fit(col_clean.reshape(-1, 1))

                    # ==========================================================
                    # 🌟 防范“零膨胀与长尾极值”导致 RobustScaler 失效
                    # ==========================================================
                    std_val = np.std(col_clean)

                    if scaler.scale_[0] < std_val * 0.05:
                        # 熔断：废弃失效的 IQR，强制退化为标准差缩放。
                        scaler.scale_[0] = max(std_val, 1e-3)
                    # ==========================================================

                    self.scalers[i] = scaler
                    self.current_mode[i] = 'continuous'
                    self.last_valid_mode[i] = 'continuous'

            # 更新对旧版调用的兼容掩码
            if self.current_mode[i] == 'discrete':
                self.is_discrete_mask[i] = True

        return self

    def transform(self, data):
        scaled_data = data.astype(float)
        _, num_cols = data.shape

        for i in range(num_cols):
            mode = self.current_mode[i]
            scaler = self.scalers[i]

            # 若继承或识别为有缩放器的模式，执行归一化
            if mode in ['continuous', 'discrete'] and scaler is not None:
                scaled_col = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
                scaled_data[:, i] = scaled_col
            else:
                # 'unfitted' 模式，触发免检通行，直接输出原始值
                scaled_data[:, i] = data[:, i]

        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        original_data = scaled_data.copy()
        _, num_cols = scaled_data.shape

        for i in range(num_cols):
            mode = self.current_mode[i]
            scaler = self.scalers[i]

            # 逆变换精确匹配逻辑
            if mode in ['continuous', 'discrete'] and scaler is not None:
                original_col = scaler.inverse_transform(scaled_data[:, i].reshape(-1, 1)).flatten()
                original_data[:, i] = original_col
            else:
                original_data[:, i] = scaled_data[:, i]

        return original_data

class OnlineHeterogeneousScaler:
    """
    终极自适应异构流式缩放器 (Ultimate Discreteness-Aligned & Anti-Collapse Version)
    1. 完美保留连续/离散异构路由与防熔断设计。
    2. 🌟【核心修正一】：整个测试周期采用连续无缝的流式状态追踪，绝对禁止在测试期间执行 fit() 重置。
    3. 🌟【核心修正二】：在 transform 期间，一旦判定为死线（50步内无任何波动），
       直接锁死并沿用故障发生前最新的、健康的 running_median 和 running_iqr，防范中位数自我污染塌缩。
    """

    def __init__(self, discrete_threshold=20, constant_tol=1e-6, track_buffer_size=2000, dead_line_detect_len=50):
        self.discrete_threshold = discrete_threshold
        self.constant_tol = constant_tol
        self.track_buffer_size = track_buffer_size
        self.dead_line_detect_len = dead_line_detect_len  # 实时死线判定窗口

        self.current_mode = None
        self.buffers = {}
        self.is_discrete_mask = None
        self.recent_points = {}  # 独立存储最近流式输入的数据，判定死线状态

        # 高精度流式参数录制器
        self.history_medians = []
        self.history_iqrs = []
        self.history_mins = []
        self.history_maxs = []

        # 运行中绝对健康的统计参数快照
        self.running_medians = None
        self.running_iqrs = None
        self.running_mins = None
        self.running_maxs = None

    def fit(self, data):
        _, num_cols = data.shape

        if self.current_mode is None:
            self.current_mode = ['unfitted'] * num_cols
            self.is_discrete_mask = np.zeros(num_cols, dtype=bool)
            self.running_medians = np.zeros(num_cols)
            self.running_iqrs = np.ones(num_cols)
            self.running_mins = np.zeros(num_cols)
            self.running_maxs = np.ones(num_cols)
            for i in range(num_cols):
                self.buffers[i] = []
                self.recent_points[i] = []

        for i in range(num_cols):
            col_data = data[:, i]
            col_clean = col_data[~np.isnan(col_data)]

            if len(col_clean) == 0:
                continue

            # 常量探测与模式熔断
            if (np.max(col_clean) - np.min(col_clean)) <= self.constant_tol:
                if self.current_mode[i] == 'unfitted':
                    self.current_mode[i] = 'continuous'
                continue

            # 正常波动序列的特征路由探测
            self.buffers[i] = list(col_clean[-self.track_buffer_size:])
            self.recent_points[i] = list(col_clean[-self.dead_line_detect_len:])

            # 计算并预存健康的初始统计快照
            self.running_medians[i] = np.median(col_clean)
            q75, q25 = np.percentile(col_clean, [75, 25])
            iqr = q75 - q25
            std_val = np.std(col_clean)
            if iqr < std_val * 0.05:
                iqr = max(std_val, 1e-3)
            self.running_iqrs[i] = iqr

            rounded_data = np.round(col_clean, decimals=5)
            unique_vals = np.unique(rounded_data)

            if len(unique_vals) <= self.discrete_threshold:
                self.current_mode[i] = 'discrete'
                self.is_discrete_mask[i] = True
            else:
                self.current_mode[i] = 'continuous'
                self.is_discrete_mask[i] = False

        return self

    def transform(self, data, update=False):
        num_samples, num_cols = data.shape
        scaled_data = data.astype(float)

        if update:
            self.history_medians = np.zeros((num_samples, num_cols))
            self.history_iqrs = np.zeros((num_samples, num_cols))
            self.history_mins = np.zeros((num_samples, num_cols))
            self.history_maxs = np.zeros((num_samples, num_cols))

        # 冷启动保底机制
        if self.current_mode is None:
            self.current_mode = ['continuous'] * num_cols
            self.is_discrete_mask = np.zeros(num_cols, dtype=bool)
            self.running_medians = np.zeros(num_cols)
            self.running_iqrs = np.ones(num_cols)
            self.running_mins = np.zeros(num_cols)
            self.running_maxs = np.ones(num_cols)
            for i in range(num_cols):
                self.buffers[i] = []
                self.recent_points[i] = []

        for t in range(num_samples):
            for i in range(num_cols):
                val = data[t, i]
                mode = self.current_mode[i]
                if np.isnan(val): continue

                is_dead_line_detected = False
                if update:
                    # 维护微型死线检测窗口
                    self.recent_points[i].append(val)
                    if len(self.recent_points[i]) > self.dead_line_detect_len:
                        self.recent_points[i].pop(0)

                    # 实时极差判定
                    if len(self.recent_points[i]) >= self.dead_line_detect_len:
                        recent_arr = np.array(self.recent_points[i])
                        if (np.max(recent_arr) - np.min(recent_arr)) <= self.constant_tol:
                            is_dead_line_detected = True

                # 🌟 如果检测到死线故障，冷冻大滑窗缓冲区更新，不推入卡死污染数据
                if update and not is_dead_line_detected:
                    self.buffers[i].append(val)
                    if len(self.buffers[i]) > self.track_buffer_size:
                        self.buffers[i].pop(0)

                current_buf = np.array(self.buffers[i]) if len(self.buffers[i]) > 0 else np.array([val])

                if mode == 'continuous':
                    # 🌟【一阶参数冷冻】：一旦判定为死线，直接锁死并沿用故障发生前最新的、绝对健康的参数
                    # 彻底阻断 50 步延迟判定期间对中位数的污染，强力捍卫偏置度不向 0 塌缩！
                    if not is_dead_line_detected and len(current_buf) >= 10:
                        median = np.median(current_buf)
                        q75, q25 = np.percentile(current_buf, [75, 25])
                        iqr = q75 - q25
                        std_val = np.std(current_buf)
                        if iqr < std_val * 0.05:
                            iqr = max(std_val, 1e-3)

                        self.running_medians[i] = median
                        self.running_iqrs[i] = iqr
                    else:
                        median = self.running_medians[i]
                        iqr = self.running_iqrs[i]

                    scaled_data[t, i] = (val - median) / iqr
                    if update:
                        self.history_medians[t, i] = median
                        self.history_iqrs[t, i] = iqr

                elif mode == 'discrete':
                    if not is_dead_line_detected and len(current_buf) >= 10:
                        p_min = np.min(current_buf)
                        p_max = np.max(current_buf)
                        span = p_max - p_min
                        if span <= self.constant_tol: span = 1.0
                        self.running_mins[i] = p_min
                        self.running_maxs[i] = p_max
                    else:
                        p_min = self.running_mins[i]
                        span = self.running_maxs[i] - self.running_mins[i]
                        if span <= self.constant_tol: span = 1.0

                    scaled_data[t, i] = (val - p_min) / span
                    if update:
                        self.history_mins[t, i] = p_min
                        self.history_maxs[t, i] = p_max
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data, update=False)

    def inverse_transform(self, scaled_data):
        num_samples, num_cols = scaled_data.shape
        original_data = scaled_data.copy()

        if len(self.history_medians) == 0:
            for i in range(num_cols):
                mode = self.current_mode[i]
                current_buf = np.array(self.buffers[i])
                if mode == 'continuous':
                    median = np.median(current_buf)
                    q75, q25 = np.percentile(current_buf, [75, 25])
                    original_data[:, i] = scaled_data[:, i] * (q75 - q25) + median
                elif mode == 'discrete':
                    p_min = np.min(current_buf)
                    p_max = np.max(current_buf)
                    original_data[:, i] = scaled_data[:, i] * (p_max - p_min) + p_min
            return original_data

        for t in range(num_samples):
            for i in range(num_cols):
                mode = self.current_mode[i]
                s_val = scaled_data[t, i]
                if mode == 'continuous':
                    original_data[t, i] = s_val * self.history_iqrs[t, i] + self.history_medians[t, i]
                elif mode == 'discrete':
                    span = self.history_maxs[t, i] - self.history_mins[t, i]
                    original_data[t, i] = s_val * span + self.history_mins[t, i]
        return original_data