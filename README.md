# IMU + FAST-LIO 15D ESKF 位姿估计

本项目用六轴 IMU 做预测，用 FAST-LIO 的位置、四元数和协方差做位姿更新，输出位置与 Roll/Pitch/Yaw 六条曲线。状态为：

```text
x_hat   = (p_W, v_W, q_WB, bg, ba)
delta_x = [delta_p, delta_v, delta_theta, delta_bg, delta_ba]
```

四元数为 xyzw，`q_WB` 表示 Body 到 World 的旋转；位置、速度在 World 系，零偏在 IMU 系。姿态误差使用右乘扰动。

## 运行

在仓库根目录执行：

```powershell
uv run --with-requirements requirements.txt python scripts/run_preprocess.py --raw-dir data/raw --output-dir data/processed
uv run --with-requirements requirements.txt python scripts/run_eskf15d.py --processed-dir data/processed --output-dir results
uv run --with-requirements requirements.txt python scripts/plot_eskf15d_results.py --result-csv results/eskf15d_result.csv --output-dir results/figures15d
```

三个入口依次完成：原始 CSV 转为 SI 制数据和匹配表；初始化后执行 Prediction/Update；从正式结果 CSV 绘制六张图。

## 代码位置

| 文件 | 内容 |
| --- | --- |
| `config.py` | 坐标约定、阈值和噪声参数 |
| `data_types.py` | IMU、pose、静止统计和匹配表的数据字段 |
| `preprocess.py` | CSV/NPZ 读写、静止段统计和时间匹配 |
| `rotation_utils.py` | 四元数和 SO(3) 运算 |
| `eskf15d.py` | 初始化、预测、观测更新、注入和 Reset |
| `scripts/run_eskf15d.py` | 加载、主循环、CSV 和摘要保存 |

## 模型要点

预处理将加速度从 g 转为 m/s²，并选取开头两秒静止段得到 `bg0`、`ba0`、初始协方差和过程噪声。每个 IMU 时间戳最多匹配一条 FAST-LIO 位姿；没有匹配或观测方差非正时只做预测。

预测使用左端点零阶保持：

```text
omega = gyro - bg
f_B   = acc - ba
a_W   = R_WB(q) @ f_B + g_W
p     = p + v * dt + 0.5 * a_W * dt^2
v     = v + a_W * dt
q     = q ⊗ Exp(omega * dt)
```

观测残差为位置差与 `Log(q_pred^-1 ⊗ q_obs)`。协方差采用 Joseph 更新，之后把误差注入名义状态并执行姿态误差 Reset。代码使用线性求解，不显式求逆。

FAST-LIO 参与更新，本身不是独立真值。结果说明流程和数值计算是否正常，不能单独证明真实定位精度或参数已经最优。
