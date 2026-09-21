# IMU + FAST-LIO 15D ESKF 位姿估计

## 1. 项目与作业完成情况

本项目完成[传感器组考核题目二](exam.pdf)的姿态估计必做内容，以及位置估计选做扩展。使用 IMU 进行 Prediction，使用 FAST-LIO 位置、四元数及协方差进行 Update，输出位置和 Roll/Pitch/Yaw 数据及六张时间曲线。

当前 `15D-final` 是最终作业展示分支，保留可独立运行的 15D 程序、原始输入、预处理数据和正式实验结果。算法与参数沿用已验收版本。

## 2. 文件结构与运行方式

| 文件 / 目录 | 职责 |
| --- | --- |
| [config.py](config.py)、[data_types.py](data_types.py) | 分组参数常量、4 个具名数据结构 |
| [rotation_utils.py](rotation_utils.py) | 四元数、旋转矩阵和 SO(3) 工具 |
| [preprocess.py](preprocess.py) | CSV / NPZ 读写与校验、单位转换、静止统计、时间匹配 |
| [eskf15d.py](eskf15d.py) | 初始化、ESKF15D 状态、Prediction、Pose Update、Injection、Reset |
| [run_preprocess.py](scripts/run_preprocess.py) | 原始 CSV → processed 数据 |
| [run_eskf15d.py](scripts/run_eskf15d.py) | 加载 → 初始化 → 预测/更新循环 → CSV 和摘要保存 |
| [plot_eskf15d_results.py](scripts/plot_eskf15d_results.py) | 从正式结果 CSV 绘制六张曲线 |
| `data/raw/` | [imu.csv](data/raw/imu.csv)、[pose_cov.csv](data/raw/pose_cov.csv) |
| `data/processed/` | [IMU](data/processed/imu_processed.npz)、[pose](data/processed/pose_processed.npz)、[静止统计](data/processed/init_stats.npz)、[匹配表](data/processed/match_table.npz) |
| `results/` | 正式结果、六张曲线和简洁诊断摘要 |

在仓库根目录执行，需先准备 Python 和 uv。uv 按 [requirements.txt](requirements.txt) 准备依赖：

```powershell
# 已有 data/processed 四个 NPZ 时可跳过此步
uv run --with-requirements requirements.txt python scripts/run_preprocess.py --raw-dir data/raw --output-dir data/processed

uv run --with-requirements requirements.txt python scripts/run_eskf15d.py --processed-dir data/processed --output-dir results
uv run --with-requirements requirements.txt python scripts/plot_eskf15d_results.py --result-csv results/eskf15d_result.csv --output-dir results/figures15d
```

也可先执行 `python -m pip install -r requirements.txt`，再使用相同命令中的 `python scripts/...` 部分。不能假定所有机器均已预装依赖。

命令会写入指定目录；要保留当前正式结果，请使用其他 `--output-dir`。预处理生成四个兼容原格式的 NPZ 和简短 `diagnostic_report.json`，保留丢弃行、静止区间及时间匹配统计。滤波生成正式 CSV 与运行摘要，不再生成逐帧 debug CSV 或长文本报告。绘图接受至少两行的有效结果，不限制为随附数据的 25219 行。

程序共 8 个 Python 文件；唯一有运行行为的类是 `ESKF15D`，另有 4 个仅描述数据字段的 dataclass。维护时按以下顺序阅读即可：

1. 改阈值、噪声和初值先看 `config.py`，无需逐层查找配置对象。
2. 看数据处理先读 `scripts/run_preprocess.py`，具体规则在 `preprocess.py`。
3. 看运行时序先读 `scripts/run_eskf15d.py` 的 `run()`；初始化返回 `(eskf, k0)`，随后循环按顺序预测、条件更新并记录标量结果。
4. 改公式看 `eskf15d.py` 的初始化函数、`predict()` 和 `update_pose()`；预测不返回调试对象，更新只返回两个残差范数和 NIS。

读取文件时检查结构与时间轴，初始化时检查静止条件和初值，滤波时检查计算产生的数值问题。数据对象不再经过初始化上下文、状态快照、Debug 和运行结果等中转层。

## 3. 状态、初始化与 Prediction

名义状态包含 16 个存储标量，15D 指误差状态维数：

```text
x_hat   = (p_W, v_W, q_WB, bg, ba)
delta_x = [delta_p, delta_v, delta_theta, delta_bg, delta_ba]
```

四元数顺序为 xyzw，q_WB 表示 Body → World。位置、速度在 World frame，两个 bias 在 Body/IMU frame。姿态误差采用右乘局部扰动 `R_true = R_hat Exp(hat(delta_theta))`。

初始静止段为 [0,396)，初始化时刻 k0=395。位置、姿态来自该时刻匹配的 FAST-LIO 观测，v0=0，bg0=mean(gyro)，ba0=mean(acc)+R0.T@g_W。P0 包含初始姿态与加速度计零偏因重力补偿产生的交叉协方差，速度初始不确定性来自配置。

IMU 原始加速度单位为 g，预处理乘以 9.81 转为 m/s²，gyro 单位 rad/s。重力 g_W=[0,0,-9.81] m/s²。左端点 ZOH 名义传播使用旧状态：

```text
omega_hat = gyro - bg
f_B       = acc - ba
a_W       = R_WB(q_old) @ f_B + g_W
p_new     = p_old + v_old * dt + 0.5 * a_W * dt^2
v_new     = v_old + a_W * dt
q_new     = normalize(q_old ⊗ Exp(omega_hat * dt))
bg_new    = bg_old
ba_new    = ba_old
```

这里 Exp 将旋转向量转为四元数。右乘来自 Body-frame gyro 与 R_WB 的定义。名义 bias 保持不变，随机游走通过过程噪声进入协方差。

连续误差模型令 R=R_WB(q_old)，每个块为 3×3：

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

Phi ≈ I15 + Fc * dt
Qd  ≈ Gc @ Qc @ Gc.T * dt
P^- = Phi @ P @ Phi.T + Qd
```

采用一阶离散，忽略高阶 dt²/dt³ 噪声耦合。当前 Qd 的位置对角块为零，不代表加速度噪声不会通过动力学影响位置。

主循环使用 gyro[k]、acc[k]、dt[k+1] 传播至 k+1，再按已有 pose_index_for_imu[k+1] 选择 Update。首步为 395 → 396，初始化观测不重复使用。dt 来自实际相邻 IMU timestamp，输出时间直接取 IMU timestamp，不累计重建时间、不重新匹配或插值。

## 4. FAST-LIO 更新与 P / Q / R

FAST-LIO 提供 position xyz、quaternion xyzw。观测四元数先归一化；若与预测四元数点积为负则取反，实现 q / -q sign alignment：

```text
r_p     = p_obs - p_pred
r_theta = Log(inverse(q_pred) ⊗ q_obs)
r       = [r_p, r_theta]
H       = [I 0 0 0 0]
          [0 0 I 0 0]
R_pose  = diag(cov_00, cov_11, cov_22, cov_33, cov_44, cov_55)
S       = H @ P^- @ H.T + R_pose
K       = P^- @ H.T @ S^-1
delta_x = K @ r
```

S 的逆仅为数学表示，代码使用线性求解，不显式求逆。H 直接观察位置和姿态，速度与两个 bias 通过 P 的交叉项间接修正。

| 矩阵 | 作用 |
| --- | --- |
| P，15×15 | 位置、速度、姿态及两个 bias 的误差状态协方差 |
| Qc，12×12；Qd，15×15 | 加计噪声、gyro 噪声及两个 bias 随机游走的连续强度和离散过程噪声协方差 |
| R_pose，6×6 | 来自 FAST-LIO CSV 方差对角项的位姿观测不确定性 |

把 RPY 方差直接用于局部 rotation-vector residual 是小角度近似，不是严格的协方差坐标转换。无匹配或观测方差非正时保留 Prediction；covariance 含 NaN/Inf 时报错，不使用 floor 修复。

Joseph covariance update、名义状态 Injection 和 Reset 为：

```text
A        = I15 - K @ H
P_joseph = A @ P^- @ A.T + K @ R_pose @ K.T
p  <- p_pred  + delta_p
v  <- v_pred  + delta_v
q  <- normalize(q_pred ⊗ Exp(delta_theta))
bg <- bg_pred + delta_bg
ba <- ba_pred + delta_ba
G_reset = I15
G_reset[6:9,6:9] = I3 - 0.5 * hat(delta_theta)
P <- G_reset @ P_joseph @ G_reset.T
```

Reset 后误差均值归零，P 不清零。代码仅在非对称误差通过容差检查后做数值对称化；主循环检查 P 有限、对称及最小特征值，不做特征值裁剪。

## 5. 实验结果与六张曲线

[正式结果 CSV](results/eskf15d_result.csv) 每行对应一个 IMU 时刻的最终 posterior，包含 px_m/py_m/pz_m、roll_rad/pitch_rad/yaw_rad，以及速度、xyzw 四元数、两个 bias、timestamp、IMU index 和 Update 标志。

以下来自[保存的运行摘要](results/eskf15d_summary.json)（phase=3.5 是 Runner 实现阶段标识）：

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

横轴为 timestamp-timestamp[0]，单位 s；位置单位 m，姿态单位 rad。曲线直接使用正式 CSV，未平滑或重采样，Yaw 保留原始主值、不 unwrap。

![Position X](results/figures15d/x_time.png)

![Position Y](results/figures15d/y_time.png)

![Position Z](results/figures15d/z_time.png)

![Roll](results/figures15d/roll_time.png)

![Pitch](results/figures15d/pitch_time.png)

![Yaw](results/figures15d/yaw_time.png)

## 6. 误差分析与局限

[离线诊断摘要](results/analysis15d/error_diagnosis_summary.json)显示：position residual norm 均值为 0.002531094 m，attitude residual norm 均值为 0.000777045 rad；NIS mean / median / p95 为 23.834862 / 5.531808 / 113.084843。

经验 p95 以上高 NIS 帧集中于 60–110 s 动态时段，位置、姿态 residual 同时增大。Yaw 在 IMU 16608、17009、17504 处跨越 ±π；相邻四元数实际旋转仅约 0.189735°、0.101484°、0.082441°，属于欧拉角表示边界。

FAST-LIO 是参与 Update 的观测来源，本身也使用 IMU，不是独立 ground truth。Prediction-to-observation residual 不等于 posterior 与观测的差异，更不等于真实定位误差。当前实验验证了完整流程和数值稳定性，没有独立验证真实位置/姿态精度，也没有证明 Q/R/P0 已正确标定。

一阶离散、姿态 covariance 近似和观测相关性可能影响统计一致性。现有证据不能确定唯一根因；日志缺少完整 S，不能精确分解位置与姿态的 NIS 贡献。本项目未进行自动调参或按 NIS 剔除观测。

## 7. 完整开发版本

[完整 15D 开发分支](https://github.com/2h34/ESKF/tree/15D)保留独立 6D 实现、单元测试、逐帧 Debug、离线误差诊断和完整学习开发过程。该分支 README 包含原有 6D 版本与验证内容。

当前 `15D-final` 用于最终作业展示，已移除 6D 遗留初始化和开发诊断包装。保留具名数据结构、独立旋转函数与直接对应公式的滤波实现；内部接口不兼容旧开发版，但三个命令行入口、四个 NPZ、正式结果 CSV 和运行摘要保持兼容。学习笔记记录历史学习过程，旧模块名不代表当前文件结构。

重构验证应在临时目录执行完整预处理、滤波和绘图：逐字段比较四个 NPZ，逐行比较正式 CSV，比较全部摘要指标，并检查六张图片可以解码。仅通过导入或 `--help` 不构成结果验证。
