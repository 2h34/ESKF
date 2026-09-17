# 基于 IMU + FAST-LIO 的 ESKF 姿态解算

## 1. 项目目标

本项目使用 IMU 角速度进行姿态递推，并使用 FAST-LIO 姿态作为观测，逐步实现一个仅包含姿态与陀螺仪零偏的 6D Error-State Kalman Filter（误差状态卡尔曼滤波器，ESKF）。当前阶段强调数据语义、坐标约定和时间边界可审计，不扩展位置、速度或加速度计零偏状态。

## 2. 当前实现进度

已经完成：

- 原始 IMU 与 FAST-LIO CSV 校验和预处理；
- 加速度从 `g` 转换为 `m/s^2`；
- 初始静止区间统计；
- 单调、一对一的时间戳匹配；
- `xyzw` 四元数与旋转工具；
- processed `.npz` 的强校验加载器；
- FAST-LIO 四元数方向的源码与真实数据验证；
- 6D ESKF Initialization（初始化）：`q0`、`bg0`、`P0`、`Qc`；
- 6D ESKF Prediction：名义状态、`Fc/Gc/Phi/Qd` 与 `P` 预测。

尚未实现：Observation Update、Kalman Gain、Joseph Update、Injection、Reset、主滤波循环、RPY 结果绘图和任何 15D/位置/速度状态。

## 3. 数据说明

正式运行代码通过 [processed_io.py](processed_io.py) 读取：

- `data/processed/imu_processed.npz`
- `data/processed/pose_processed.npz`
- `data/processed/init_stats.npz`
- `data/processed/match_table.npz`

IMU 角速度单位为 `rad/s`，加速度单位为 `m/s^2`。预处理数据保持原始测量含义，不预先扣除 `bg0`，也不执行重力补偿。`dt[0]` 为 `NaN`，因为第一个样本没有前驱；它不是人为补造的采样间隔。

默认无效行策略只丢弃无法解析为完整有限数值记录的行，并在诊断中记录原始 CSV 行号。若需要遇错即停，可在 `ProjectConfig` 中设置 `invalid_sample_policy="raise"`。时间戳重复或逆序、无效四元数和负协方差始终报错。

## 4. 数学与坐标约定

- 四元数存储顺序：`[qx, qy, qz, qw]`，即 `xyzw`。
- 姿态方向：`R_WB` 将 Body Frame 向量转换到 World Frame，`v_W = R_WB @ v_B`。
- 误差约定：右乘误差，`R_true = R_hat Exp(delta_theta^)`。
- 世界系重力：`g_W = [0, 0, -9.81] m/s^2`。
- 匹配误差：`pose_timestamp[j] - imu_timestamp[k]`。

当前 Nominal State（名义状态）为 `q_hat` 与 `b_g_hat`。Error State（误差状态）为：

```text
delta_x = [delta_theta, delta_b_g]
```

维度为 6。工程中不长期保存 `delta_x = zeros(6)`；后续 injection/reset 后将误差均值按零处理。

## 5. 第一阶段：数据预处理与时间对齐

预处理入口：

```powershell
python scripts/run_preprocess.py
```

该命令生成四个 processed `.npz`，以及 JSON 和文本诊断。时间匹配采用由数据采样周期确定的容差，保持 pose 不重复使用，并用 `-1` 标记没有有效 pose 的 IMU 样本。

## 6. FAST-LIO 姿态方向验证

本项目正式把 FAST-LIO CSV 四元数解释为 `q_WB`，但区分以下两类证据：

1. 标准 FAST-LIO 源码支持 `R_WB`：[`pointBodyToWorld`](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L165-L169) 使用 `state_point.rot` 将 IMU/body 点变换到全局；过程模型通过 [`s.rot * (in.acc - s.ba)`](https://github.com/hku-mars/FAST_LIO/blob/main/include/use-ikfom.hpp#L42-L53) 得到惯性系加速度；源码把 [`state_point.rot` 直接复制到 `geoQuat`](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L887-L897)，并发布 `camera_init -> body` 的 [odometry/TF](https://github.com/hku-mars/FAST_LIO/blob/main/src/laserMapping.cpp#L545-L574)。
2. 当前仓库没有生成所给 CSV 的导出代码，因此仅凭标准源码无法证明导出程序没有额外求逆或坐标变换。

真实静止数据在排除 20 个非正姿态协方差帧后，共使用 347 组匹配：

| 四元数假设 | 平均角误差 | 中位角误差 | 最大角误差 | 平均向量残差 |
| --- | ---: | ---: | ---: | ---: |
| `q = R_WB` | 0.736329 deg | 0.656430 deg | 1.699284 deg | 0.146254 m/s^2 |
| `q = R_BW` | 1.335364 deg | 1.293349 deg | 2.516289 deg | 0.240803 m/s^2 |

`R_WB` 在 86.17% 的样本上角误差更小；代表性平均姿态的误差分别为 0.400813 deg 与 1.231878 deg。在本项目设定的启发式判据下，真实数据支持 `R_WB`；结合标准 FAST-LIO 的源码和 frame 语义，本项目采用 `R_WB` 约定。重力只能约束 roll/pitch 方向，不能独立验证 yaw 语义。

复验命令：

```powershell
python scripts/check_pose_convention.py
```

## 7. Observation Covariance 使用说明

前 20 个 pose 帧的协方差对角线全零，时间范围为 `1786181234.225484610` 至 `1786181234.319545984`；静止初始化区间结束时间为 `1786181236.055275679`，因此当前数据从初始化点向后不会使用这些观测。

当前策略是：任一姿态方差 `<= 0` 时拒绝初始化或跳过后续观测。零方差会把观测解释为完全确定，可能造成奇异或过强更新；在没有数据依据时添加 covariance floor 会引入隐藏调参，因此当前不设置 `R_min`。

题面给出的 `cov_33/cov_44/cov_55` 是 roll/pitch/yaw 方差，而右乘 ESKF 使用局部旋转向量 `delta_theta`。两者严格来说不是同一坐标表达，映射依赖当前姿态、欧拉角顺序、误差乘法侧和坐标系。当前作业采用小误差工程近似：

```text
P_theta ≈ diag(cov_33, cov_44, cov_55)
```

当前不额外引入题目未要求的 Euler-to-tangent covariance Jacobian。

## 8. FAST-LIO 与 IMU 相关性假设

[FAST-LIO 是紧耦合 LiDAR-inertial estimator](https://github.com/hku-mars/FAST_LIO#fast-lio)，其姿态输出与本项目预测所使用的同一 IMU 数据并非严格统计独立。忽略交叉相关性可能重复利用信息、低估不确定性，并使名义 Kalman filter 不一致。

题目没有要求相关噪声或交叉协方差建模，因此本项目采用“过程噪声与观测噪声独立”的课程简化假设。这是明确的工程近似，不是严格独立性的声明。

## 9. 第二阶段：6D ESKF Initialization

[initialization.py](initialization.py) 只负责建立 `q0`、`bg0`、`P0` 和连续时间 Process Noise（过程噪声）`Qc`，不包含 Prediction 或 Update。

### 9.1 初始化时刻

静止区间为 `[static_start_idx, static_end_idx)`。当前数据为 `[0, 396)`，因此：

```text
static_end_idx = 396
k0 = static_end_idx - 1 = 395
t0 = imu.timestamp[k0]
j0 = alignment.pose_index_for_imu[k0]
```

`j0` 必须有效；若为 `-1`，直接报错，不前后搜索、不插值，也不改变 `k0`。`q0`、`bg0` 和 `P0` 都表示 `t0` 时刻的初始化状态。Prediction 的第一步为 `395 -> 396`，使用 `gyro[395]` 和 `imu.dt[396]`。

### 9.2 名义状态初值

```text
q0 = normalize_quaternion(pose.quaternion_xyzw[j0])
bg0 = init_stats.gyro_mean
```

`q0` 保持 `q_WB` 方向，不求逆。`bg0` 是 Gyro Bias（陀螺仪零偏）初值，不会写回或修改 processed IMU 数据。

### 9.3 初始协方差 P0

```text
N_static = static_end_idx - static_start_idx
P_theta0 = diag(pose.cov_diag[j0, 3:6])
P_bg0 = diag(init_stats.gyro_std**2 / N_static)
P0 = block_diag(P_theta0, P_bg0)
```

姿态方差必须有限且严格大于零。姿态与 bias 的交叉块保持为零，不人为加入 cross-correlation，也不增加 uncertainty floor。

### 9.4 连续时间过程噪声 Qc

Gyro Noise Density（陀螺仪噪声密度）由当前静止数据估计：

```text
gyro_noise_density = init_stats.gyro_std * sqrt(init_stats.median_dt)
Qg_diag = gyro_noise_density**2
```

这里使用真实 `median_dt`，不写死 200 Hz 或 `dt=0.005`。

约 2 秒的静止数据不足以可靠估计 Gyro Bias Random Walk（陀螺仪零偏随机游走）。配置项 `gyro_bias_random_walk_density = 1e-4` 是第一版工程初值，不是题面参数或传感器标定结果。后续需要结合 bias trajectory、innovation 和最终姿态结果检查并调参。

```text
Qbg_diag = [density**2, density**2, density**2]
Qc = diag(Qg_x, Qg_y, Qg_z, Qbg_x, Qbg_y, Qbg_z)
```

Initialization 模块只构造连续时间 `Qc`；离散化和协方差预测由 `eskf6d.py` 负责。

## 10. 第二阶段：6D ESKF Prediction

[eskf6d.py](eskf6d.py) 维护 `q_WB`、`b_g`、`P`、固定的 `Qc` 以及当前 IMU 索引和时间戳。每次 `predict()` 使用 Zero-Order Hold（零阶保持）的左端点约定：

```text
state[k] + gyro[k] + dt[k+1] -> state[k+1]
```

不使用 `gyro[k+1]`、相邻均值或 midpoint integration。名义状态预测为：

```text
omega_hat = gyro[k] - bg
delta_theta = omega_hat * dt[k+1]
q[k+1] = normalize(q[k] ⊗ Exp(delta_theta))
bg[k+1] = bg[k]
```

右乘四元数增量与项目的 right-multiplicative error convention 一致。连续误差模型和噪声映射为：

```text
Fc = [-hat(omega_hat)  -I]
     [       0          0]

Gc = [-I  0]
     [ 0  I]
```

第一版只采用一阶离散近似：

```text
Phi ≈ I + Fc * dt
Qd ≈ Gc @ Qc @ Gc.T * dt
P_new = Phi @ P_old @ Phi.T + Qd
```

`Qd` 使用每一步真实 `dt[k+1]`，不使用静止段 `median_dt` 或写死的 200 Hz。预测后对 `P` 做 `0.5 * (P + P.T)` 数值对称化，并检查有限性、对称性和对角线非负性；不把明显负值偷偷 clamp 为零。

当前 Prediction 不读取 CSV、不修复时间戳、不插值 IMU，也不包含 FAST-LIO residual、`H/S/K`、Measurement Update、Injection 或 Reset。

## 11. 运行方式

使用安装了 NumPy 的 Python 环境：

```powershell
python scripts/check_pose_convention.py
python scripts/check_initialization.py
python scripts/check_prediction.py
python -m unittest discover -s tests -v
```

`check_initialization.py` 只输出初始化量；`check_prediction.py` 只检查第一个真实 Prediction 和后续 20 步 Prediction-only 短序列。二者都不是正式滤波入口。

## 12. 开发期测试与最终提交整理

`tests/`、`scripts/check_pose_convention.py`、`scripts/check_initialization.py` 和 `scripts/check_prediction.py` 都属于开发期验证资产，与正式算法模块隔离。当前保留这些文件用于人工审核和回归检查；项目完成后将单独执行 submission cleanup，只保留题目要求和程序正常运行所需的正式代码。
