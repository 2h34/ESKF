# 基于 EKF 的 IMU 姿态解算（含位置估计选做）

本题为[传感器组考核题目二](exam.pdf)。使用六轴 IMU 做 EKF 预测，使用 FAST-LIO 位姿与协方差做观测更新，输出姿态（Roll/Pitch/Yaw）与位置（x/y/z）数据及六条时间曲线。

状态设计采用题目给出的 15 维误差状态形式，因此同时覆盖必做的姿态估计与选做的位置估计：

```text
x_hat   = (p_W, v_W, q_WB, bg, ba)
delta_x = [delta_p, delta_v, delta_theta, delta_bg, delta_ba]     # 15 维
```

必做部分对应 `delta_x` 中的姿态与陀螺零偏两块（6 维），其余 9 维为位置选做扩展。四元数顺序为 xyzw、`q_WB` 表示 Body → World；位置与速度在 World frame，两个零偏在 Body/IMU frame；姿态误差用旋转向量表示，采用右乘局部扰动 `R_true = R_hat Exp(hat(delta_theta))`。

## 1. 数据读取与预处理

对应题目任务要求（1）：

- **数据读取**：按列名读取 `data/raw/imu.csv` 与 `data/raw/pose_cov.csv`，丢弃非数值行。
- **单位转换**：加速度由 g 乘以 9.81 转为 m/s²，角速度保持 rad/s。静止段实测 |acc| ≈ 0.994 g，与题目提示一致。
- **时间戳处理**：校验严格递增；相邻两帧时间间隔 `dt` 由相邻 timestamp 差分得到（`dt[0] = NaN`）。
- **陀螺零偏初值（必做）**：取初始静止段均值 `bg0 = mean(gyro)`。
- **加计零偏初值（选做）**：静止时加计只感受重力，`ba0 = mean(acc) + R_WB(q0)^T @ g_W`。
- **异常值处理**：静止判据校验（陀螺均值与标准差、加计模长与标准差）；观测协方差非正或四元数非单位时拒绝该观测。

初始静止段为 `[0, 396)`，初始化时刻 `k0 = 395`，位置与姿态取该时刻对齐的 FAST-LIO 观测，`v0 = 0`。两份数据 timestamp 为同一时间基准，用 O(N+M) 最近邻匹配对齐，每个位姿最多使用一次。

## 2. EKF 预测模型

IMU 在相邻两帧间积分，采用左端点 ZOH（所有传播项使用旧状态）。`omega_hat = gyro - bg`，`f_B = acc - ba`，重力 `g_W = [0, 0, -9.81]` m/s²：

```text
a_W   = R_WB(q_old) @ f_B + g_W
p_new = p_old + v_old * dt + 0.5 * a_W * dt^2          # 选做
v_new = v_old + a_W * dt                               # 选做
q_new = normalize(q_old ⊗ Exp(omega_hat * dt))         # 必做
bg_new = bg_old
ba_new = ba_old
```

四元数更新后归一化；名义零偏保持不变，随机游走通过过程噪声进入协方差。

连续误差模型为误差状态 `delta_x` 的线性化模型，令 `R = R_WB(q_old)`，每个块为 3×3：

```text
Fc = [0 I       0           0  0]
     [0 0 -R@hat(f_B)       0 -R]
     [0 0 -hat(omega_hat)  -I  0]
     [0 0       0           0  0]
     [0 0       0           0  0]

w  = [n_a, n_g, n_bg, n_ba]
Gc = [ 0  0  0  0]
     [-R  0  0  0]
     [ 0 -I  0  0]
     [ 0  0  I  0]
     [ 0  0  0  I]

Phi = I15 + Fc * dt
Qd  = Gc @ Qc @ Gc.T * dt
P^- = Phi @ P @ Phi.T + Qd
```

采用一阶离散，忽略高阶 dt²/dt³ 噪声耦合。

## 3. EKF 观测更新

对应题目任务要求（3）。观测为 `pose_cov.csv` 的位姿，观测四元数先归一化，若与预测四元数点积为负则取反，实现 q / -q sign alignment：

```text
r_p     = p_obs - p_pred                               # 选做
r_theta = Log(inverse(q_pred) ⊗ q_obs)                 # 必做
r       = [r_p, r_theta]
H       = [I 0 0 0 0]
          [0 0 I 0 0]
R_pose  = diag(cov_00, cov_11, cov_22, cov_33, cov_44, cov_55)
S       = H @ P^- @ H.T + R_pose
K       = P^- @ H.T @ S^-1
delta_x = K @ r
```

观测噪声取自 `pose_cov.csv` 的协方差对角项：姿态取 `cov_33 / cov_44 / cov_55`，位置取 `cov_00 / cov_11 / cov_22`。S 的逆仅为数学表示，代码使用线性求解。H 直接观测位置与姿态，速度与两个零偏通过 P 的交叉项间接修正。姿态观测使 Roll、Pitch、Yaw 全部可观，其中 Yaw 是加速度计重力观测无法约束的。无匹配或观测方差非正时保留 Prediction，不做更新。

Joseph 形式协方差更新、名义状态注入与 Reset：

```text
A        = I15 - K @ H
P_joseph = A @ P^- @ A.T + K @ R_pose @ K.T
p  <- p_pred  + delta_p
v  <- v_pred  + delta_v
q  <- normalize(q_pred ⊗ Exp(delta_theta))
bg <- bg_pred + delta_bg
ba <- ba_pred + delta_ba
G_reset = I15
G_reset[6:9, 6:9] = I3 - 0.5 * hat(delta_theta)
P <- G_reset @ P_joseph @ G_reset.T
```

Reset 后误差均值归零，P 不清零。

## 4. P、Q、R 的作用

| 矩阵 | 维数 | 作用 |
| --- | --- | --- |
| P | 15×15 | 位置、速度、姿态及两个零偏的误差状态协方差；决定增益分配与各状态的可观程度 |
| Qc / Qd | 12×12 / 15×15 | Qc 为加计噪声、陀螺噪声及两个零偏随机游走的连续强度；Qd 为对应离散过程噪声协方差，决定预测阶段不确定性的增长 |
| R_pose | 6×6 | 来自 FAST-LIO 方差对角项的观测不确定性，决定观测被信任的程度 |

陀螺与加计的噪声密度由静止段标准差换算，两个零偏随机游走密度与初始速度不确定性取配置中的工程先验。P0 包含初始姿态与加计零偏因重力补偿产生的交叉协方差，以及由静止段样本数决定的零偏初值不确定性。

## 5. 输出与实验结果

### 5.1 输出

对应题目任务要求（4）与输出要求。欧拉角由最终四元数按 ZYX 顺序转换得到。

[正式结果 CSV](results/eskf15d_result.csv) 每行对应一个 IMU 时刻的最终 posterior，含 px_m/py_m/pz_m、roll_rad/pitch_rad/yaw_rad，以及速度、xyzw 四元数、两个零偏、timestamp、IMU index 与 Update 标志。

以下来自[运行摘要](results/eskf15d_summary.json)：

| 项目 | 结果 |
| --- | ---: |
| 起止 IMU index | 395–25613 |
| Duration | 126.094735 s |
| Output frames / Prediction steps | 25219 / 25218 |
| Successful updates | 25212 |
| 最大 quaternion norm error | 2.220446049250313e-16 |
| 最大 P symmetry error | 0.0 |
| 最小 P eigenvalue | 2.187804287285118e-09 |
| 正式结果全部有限 | true |

### 5.2 六条曲线

横轴为 `timestamp - timestamp[0]`，单位 s；位置单位 m，姿态单位 rad。曲线直接使用正式 CSV，未平滑或重采样，Yaw 保留原始主值、不 unwrap。

![Position X](results/figures15d/x_time.png)

![Position Y](results/figures15d/y_time.png)

![Position Z](results/figures15d/z_time.png)

![Roll](results/figures15d/roll_time.png)

![Pitch](results/figures15d/pitch_time.png)

![Yaw](results/figures15d/yaw_time.png)

## 6. 误差分析与局限

[离线诊断摘要](results/analysis15d/error_diagnosis_summary.json)：position residual norm 均值为 0.002531094 m，attitude residual norm 均值为 0.000777045 rad；NIS mean / median / p95 为 23.834862 / 5.531808 / 113.084843。经验 p95 以上高 NIS 帧集中于 60–110 s 动态时段，位置与姿态 residual 同时增大。Yaw 在 IMU 16608、17009、17504 处跨越 ±π，但相邻四元数实际旋转仅约 0.189735°、0.101484°、0.082441°，属于欧拉角表示边界。

FAST-LIO 是参与 Update 的观测来源，本身也使用 IMU，不是独立 ground truth。Prediction-to-observation residual 不等于真实定位误差；本实验验证了完整流程与数值稳定性，但没有独立验证真实位置/姿态精度，也没有证明 Q/R/P0 已正确标定。一阶离散、姿态协方差的小角度近似与观测相关性都可能影响统计一致性；现有日志不足以确定唯一根因。本项目未进行自动调参或按 NIS 剔除观测。

## 附录：代码结构与运行方式

| 文件 / 目录 | 职责 |
| --- | --- |
| [config.py](config.py)、[data_types.py](data_types.py) | 参数常量、4 个具名数据结构 |
| [rotation_utils.py](rotation_utils.py) | 四元数、旋转矩阵与 SO(3) 工具 |
| [preprocess.py](preprocess.py) | CSV / NPZ 读写与校验、单位转换、静止统计、时间匹配 |
| [eskf15d.py](eskf15d.py) | 初始化、ESKF15D 状态、Prediction、Pose Update、Injection、Reset |
| [run_preprocess.py](scripts/run_preprocess.py) | 原始 CSV → processed 数据 |
| [run_eskf15d.py](scripts/run_eskf15d.py) | 加载 → 初始化 → 预测/更新循环 → CSV 和摘要保存 |
| [plot_eskf15d_results.py](scripts/plot_eskf15d_results.py) | 从正式结果 CSV 绘制六张曲线 |
| `data/raw/` | [imu.csv](data/raw/imu.csv)、[pose_cov.csv](data/raw/pose_cov.csv) |
| `data/processed/` | [IMU](data/processed/imu_processed.npz)、[pose](data/processed/pose_processed.npz)、[静止统计](data/processed/init_stats.npz)、[匹配表](data/processed/match_table.npz) |
| `results/` | 正式结果、六张曲线与诊断摘要 |

在仓库根目录执行，需先准备 Python 与 uv。uv 按 [requirements.txt](requirements.txt) 准备依赖：

```powershell
# 已有 data/processed 四个 NPZ 时可跳过此步
uv run --with-requirements requirements.txt python scripts/run_preprocess.py --raw-dir data/raw --output-dir data/processed

uv run --with-requirements requirements.txt python scripts/run_eskf15d.py --processed-dir data/processed --output-dir results
uv run --with-requirements requirements.txt python scripts/plot_eskf15d_results.py --result-csv results/eskf15d_result.csv --output-dir results/figures15d
```

也可先执行 `python -m pip install -r requirements.txt`，再使用相同命令中的 `python scripts/...` 部分。
