# 2026-09-12 传感器组考核 Day 3 学习记录
## 主题：Probability / Gaussian / Covariance 与 Kalman Filter

> 学习目标：从“已经能够利用 IMU 做姿态 Prediction”过渡到“能够描述 Prediction 与 Observation 的不确定性，并理解 Kalman Filter 如何依据这些不确定性进行融合”。

---

# 1. 前情与 Day 3 要解决的问题

五天理论路线：Day 1 三维 Rotation 与 Quaternion；Day 2 SO(3) / so(3) / Exp / Log + IMU Quaternion Propagation；Day 3 Probability / Gaussian / Covariance + Kalman Filter；Day 4 EKF + Error-State EKF + 6D Attitude ESKF；Day 5 15D Position Extension + 题目二完整数学模型。

Day 2 已经解决从角速度到四元数的预测链：

$\displaystyle \omega_m \rightarrow \hat{\omega} \rightarrow \Delta\theta \rightarrow \Delta q \rightarrow q_{k+1}$

即 $IMU \rightarrow\mathrm{Quaternion\ Prediction}$。Day 3 不再重新学习姿态表示，也不提前推 Quaternion Error-State，而是补上“Prediction 和 Observation 各自有多可信、如何据此融合”这一环：

$\displaystyle \boxed{ Prediction + Observation + Uncertainty \rightarrow\mathrm{Kalman\ Fusion}}$

---

# 2. Day 2 回顾：IMU 预测链及其不确定性

## 2.1 陀螺仪输出与测量模型

Gyroscope（陀螺仪）直接输出 **Angular Velocity（角速度）**：

$\displaystyle \omega_m= \begin{bmatrix} \omega_x\\ \omega_y\\ \omega_z \end{bmatrix}$

单位是 $rad/s$。它**不是直接输出姿态**；姿态需要通过 $\mathrm{Angular\ Velocity}\rightarrow Integration \rightarrow Attitude$ 得到。

陀螺仪测量模型为：

$\displaystyle \omega_m = \omega_{true} + b_g + n_g$

其中 $b_g$ 是 Gyroscope Bias（陀螺仪零偏），$n_g$ 是 Gyroscope Noise（陀螺仪噪声）。进行 Bias Compensation（零偏补偿）后得到 $\hat{\omega} = \omega_m-\hat b_g$。

## 2.2 为什么纯陀螺积分会 Drift

即使已经补偿，也可能仍然存在 residual bias $\delta b_g=b_g-\hat b_g$ 以及测量噪声。这些 Angular Velocity Error 经过时间积分：

$\displaystyle \delta \theta(t) \approx \int_0^t \delta \omega(\tau)d\tau$

会逐渐累积成 Attitude Error，最终形成 Drift（漂移）。核心因果链：

$\displaystyle \boxed{\mathrm{Gyro\ Error}\rightarrow Integration \rightarrow\mathrm{Attitude\ Error\ Accumulation}\rightarrow Drift }$

## 2.3 增量旋转向量 $\Delta\theta$ 的物理意义

$\displaystyle \Delta\theta = (\omega_m-\hat b_g)\Delta t$

这里的 $\Delta\theta$ 是 **Rotation Vector（旋转向量）**，更准确地说，是在这一小段时间 $\Delta t$ 内发生的增量旋转对应的旋转向量。可写成：

$\displaystyle \Delta\theta=\theta n$

其中 $n$ 是旋转轴方向，$\theta$ 是这段时间内转过的角度。然后 $\Delta\theta \rightarrow \Delta q$，再用于 Quaternion Prediction。

## 2.4 有具体 Quaternion 不等于姿态绝对准确

例如算法得到：

$\displaystyle \hat q= [0.01,0.02,0.10,0.994]$

这里必须区分两个不同层级的问题。归一化要求：

$\displaystyle \|q\|=1$

它解决的是“当前 Quaternion 表示是否合法”。但即使一个 Quaternion 完全满足单位约束，也不代表它就是真实姿态。State uncertainty（状态不确定性）还来自 Gyroscope noise、residual bias、数值积分误差、模型近似、初始姿态误差等。所以 $\hat q$ 只是当前的 Estimated State（状态估计值），而不是“绝对真实值”。

---

# 3. 为什么状态估计需要 Probability

两个算法都输出 $Yaw=10^\circ$，但一个是 $10^\circ \pm 0.1^\circ$，另一个是 $10^\circ \pm 20^\circ$。Estimated State 相同，confidence（可信程度）却完全不同。所以：

$\displaystyle \boxed{\mathrm{State\ Estimation}=\mathrm{State\ Estimate}+\mathrm{State\ Uncertainty}}$

状态估计算法必须同时回答两个问题：我认为状态是多少？我对这个判断有多确定？

## 3.1 Random Variable：被随机建模的是我们对真实状态的认识

一个重要边界：机器人某一时刻真实的 Yaw、Position、Velocity 实际上具有一个确定值 $Yaw_{true}=10.3^\circ$，它不是物理上“随机在不同角度之间跳”。真正被随机建模的是我们对未知真实状态的认识。因此在状态估计里，用 Random Variable $X$ 描述：

$\displaystyle \boxed{ \text{真实状态可能位于哪些值，以及这些值分别有多可能} }$

需要区分三个量：True State $x_{true}$ 客观存在但通常未知；Estimated State $\hat x$ 是算法当前给出的具体估计结果；Random Variable $X$ 用于描述我们对未知真实状态的 Probability Distribution（概率分布）。

## 3.2 Mean / Expected Value：均值 / 期望

$\displaystyle \mu=E[X]$

例如测量 $9.9,\ 10.1,\ 10.0,\ 10.2,\ 9.8$ 整体围绕 $10$ 分布，所以 Mean 约为 10。工程直觉：

$\displaystyle \boxed{ Mean = \mathrm{Probability\ Distribution}\ \text{的中心位置} }$

在 Kalman Filter 中 $\hat x$ 可以理解为当前状态分布的中心或最佳估计。注意 $\hat x$ 是最佳估计，不代表 $\hat x=x_{true}$ 一定成立。

## 3.3 Variance：方差，以及与 Standard Deviation 的区别

$\displaystyle \boxed{ \sigma^2 = E[(X-\mu)^2] }$

它描述随机变量在 Mean 周围分散得有多厉害。例如两个传感器 Mean 都是 10，但 Sensor A 输出 $9.99,\ 10.01,\ 10.00,\ldots$，Sensor B 输出 $8,\ 12,\ 9,\ 11,\ldots$；虽然 Mean 一样，但 B 更不稳定，因此 B 的 Variance 更大。关系是：$\sigma^2$ 小 $\Rightarrow$ Distribution 更集中 $\Rightarrow$ Uncertainty 更小；$\sigma^2$ 大则相反。

Variance 记为 $\sigma^2$，Standard Deviation（标准差）记为 $\sigma$。例如 $\sigma=2^\circ$ 时 $\sigma^2=4\mathrm{\ deg}^2$。姿态滤波中一般采用弧度，因此姿态 Variance 常见单位是 $rad^2$。

边界：题目二中 `cov_33`、`cov_44`、`cov_55` 是姿态误差的 **Variance / Covariance diagonal elements**，不是 Standard Deviation。

## 3.4 Gaussian Distribution：Mean + Variance

一维形式：

$\displaystyle X\sim\mathcal N(\mu,\sigma^2)$

其中 $\mu$ 是 Mean，决定 Distribution 中心；$\sigma^2$ 是 Variance，决定 Distribution 宽度。工程上可先记：

$\displaystyle \boxed{ Gaussian = Mean + Variance }$

例如 $Yaw\sim\mathcal N(10^\circ,0.01\mathrm{\ deg}^2)$ 表示当前最可能在 $10^\circ$ 附近且 uncertainty 很小；而 $Yaw\sim\mathcal N(10^\circ,25\mathrm{\ deg}^2)$ 虽然中心仍是 $10^\circ$，uncertainty 大得多。

多维形式：

$\displaystyle x\sim \mathcal N(\mu,\Sigma)$

其中 $\mu$ 是 Mean Vector（均值向量），$\Sigma$ 是 Covariance Matrix（协方差矩阵）。一维状态用一个 Variance 就能描述 uncertainty，多维状态则需要 Covariance Matrix。

## 3.5 Covariance：两变量误差是否一起变化

设 $x=\begin{bmatrix} p\\ v \end{bmatrix}$。不仅要知道 position 自身、velocity 自身多不确定，还要知道 position error 和 velocity error 是否存在统计关联。Covariance 定义：

$\displaystyle Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$

它描述的是：

$\displaystyle \boxed{ \text{两个变量相对各自} Mean \text{的偏差是否具有一起变化的趋势} }$

如果 velocity 估计偏大时 position 通常也偏大，则 $Cov(p,v)>0$；若一个偏大时另一个往往偏小，则 $Cov(p,v)<0$；若没有明显线性关联，则 $Cov(p,v)\approx0$。

## 3.6 Covariance Matrix：协方差矩阵

对于 $x=\begin{bmatrix} p\\ v \end{bmatrix}$，可写：

$\displaystyle P= \begin{bmatrix} \sigma_p^2 & Cov(p,v)\\ Cov(v,p) & \sigma_v^2 \end{bmatrix}$

核心结构是：

$\displaystyle \begin{aligned} &\boxed{\mathrm{Diagonal\ Terms}= \text{每个} State \text{自己的} Variance }\\ &\boxed{ Off\text{-}\mathrm{Diagonal\ Terms} = \text{不同} State Error \text{之间的} Covariance } \end{aligned}$

例如

$\displaystyle P= \begin{bmatrix} 4&1.5\\ 1.5&9 \end{bmatrix}$

的含义是 $4=Var(p)$、$9=Var(v)$、$1.5=Cov(p,v)=Cov(v,p)$。Covariance Matrix 满足：

$\displaystyle P=P^T$

即它是对称矩阵。

## 3.7 重要边界：Covariance ≠ Error

定义 Estimation Error（估计误差）：

$\displaystyle e=x-\hat x$

但因为真实状态 $x$ 通常未知，所以当前 $e$ 到底是多少通常也不知道。而：

$\displaystyle \boxed{ P = E[ee^T] }$

描述的是 **State Estimate Error Covariance（状态估计误差协方差）**，也就是：

$\displaystyle \boxed{ \text{我们对} Error \text{的统计不确定性} }$

因此 $P_{pp}=100$ 不能理解为“当前 position 一定错了 100 m”。如果 $P_{pp}=\mathrm{100\ m}^2$，它表示的是 Position Error Variance，对应 Standard Deviation：

$\displaystyle \sigma_p=10m$

所以：

$\displaystyle \boxed{ P\text{ 大} \neq \text{当前实际} Error \text{一定大} }$

更准确的理解是 $P$ 大 $\Rightarrow$ State Estimate uncertainty 大。

---

# 4. 三个最重要的量：P / Q / R

这是 Day 3 第一个核心验收点，也是最容易混淆的地方。

## 4.1 P：State Estimate Error Covariance

$\displaystyle \boxed{ P = \text{当前} State Estimate \text{的} uncertainty }$

它回答“我现在对当前 State Estimate 有多确定”。$P$ 小 $\Rightarrow$ confidence 高；$P$ 大 $\Rightarrow$ uncertainty 高。但不能说 $P$ 大 $\Rightarrow$ 实际 State 一定错得多。

## 4.2 Q：Process Noise Covariance

Process Noise（过程噪声）记为 $w_k$，状态模型：

$\displaystyle x_k = Fx_{k-1}+w_k$

假设：

$\displaystyle w_k\sim \mathcal N(0,Q)$

其中：

$\displaystyle \boxed{ Q =\mathrm{Process\ Noise\ Covariance}}$

含义是：

$\displaystyle \boxed{ Q = \text{这一次} Prediction \text{过程中新增的} uncertainty }$

IMU 题目中 Q 的来源包括 Gyroscope Noise、Gyroscope Bias Random Walk、模型近似、其他未建模扰动。如果 IMU 很差，$Q\uparrow$，表示每次 Prediction 都应该引入更多 uncertainty。

## 4.3 R：Measurement Noise Covariance

Measurement Noise（测量噪声）记为 $v_k$，观测模型：

$\displaystyle z_k = Hx_k+v_k$

假设：

$\displaystyle v_k\sim\mathcal N(0,R)$

其中：

$\displaystyle \boxed{ R =\mathrm{Measurement\ Noise\ Covariance}}$

含义是：

$\displaystyle \boxed{ R = \text{这一次} Observation \text{自身的} uncertainty }$

如果 FAST-LIO 某一帧 covariance 很大，则 $R\uparrow$，意味着这一帧 Observation 应该更少相信。

## 4.4 P / Q / R 的最终区分

$\displaystyle \begin{aligned} &\boxed{ P=\text{我现在有多不确定} }\\ &\boxed{ Q=\text{我预测这一步又增加多少不确定性} }\\ &\boxed{ R=\text{外部观测本身有多不确定} } \end{aligned}$

---

# 5. State-Space Model 与 F / H

## 5.1 State-Space Model：状态空间模型

Linear System（线性系统）中，状态模型用 $F_k$ 把上一时刻 State 传播到当前时刻：

$\displaystyle \boxed{ x_k = F_kx_{k-1} + B_ku_k + w_k }$

Observation Model 把 State 映射到 Sensor 能观测的空间：

$\displaystyle \boxed{ z_k = H_kx_k + v_k }$

其中 $x_k$ 是 State（状态）；$u_k$ 是 Input / Control（输入 / 控制）；$F_k$ 是 State Transition Matrix；$B_k$ 是 Input Matrix；$w_k$ 是 Process Noise；$z_k$ 是 Measurement / Observation；$H_k$ 是 Observation Matrix；$v_k$ 是 Measurement Noise。

## 5.2 F：State Transition Matrix

$F$ 描述：

$\displaystyle \boxed{ \text{上一时刻} State \text{怎样传播到下一时刻} State }$

以 $x=\begin{bmatrix} p\\ v \end{bmatrix}$ 的恒速模型为例，$p_k=p_{k-1}+v_{k-1}\Delta t$ 且 $v_k=v_{k-1}$，对应：

$\displaystyle \boxed{ F= \begin{bmatrix} 1&\Delta t\\ 0&1 \end{bmatrix} }$

逐项理解：$F_{11}=1$ 表示旧 position 保留到新 position；$F_{12}=\Delta t$ 表示 velocity 通过积分影响下一时刻 position（右上角 $\Delta t$ 存在，正是因为表达了 velocity 对下一时刻 position 的积分作用）；$F_{21}=0$ 表示恒速模型中 position 不影响下一时刻 velocity；$F_{22}=1$ 表示 velocity 保持不变。所以：

$\displaystyle \boxed{ F =\mathrm{State\ propagation\ relationship}}$

## 5.3 H：Observation Matrix

$H$ 的核心作用是：

$\displaystyle \boxed{ \text{把完整} State \text{映射到} Sensor \text{能观测的空间} }$

若 $x=\begin{bmatrix} position\\ velocity \end{bmatrix}$ 而 Sensor 只测 Position，则 $H=\begin{bmatrix} 1&0 \end{bmatrix}$；若 Sensor 只测 Velocity，则 $H=\begin{bmatrix} 0&1 \end{bmatrix}$。

这里出现过一次理解问题：一开始把“只测 velocity”的 H 写成了 $2\times2$ 的 $\begin{bmatrix} 0&0\\ 0&1 \end{bmatrix}$。后续明确：如果 $x\in\mathbb R^n$ 而 $z\in\mathbb R^m$，那么：

$\displaystyle \boxed{ H\in\mathbb R^{m\times n} }$

也就是：

$\displaystyle \boxed{ H:\mathrm{State\ Space}\rightarrow\mathrm{Measurement\ Space}}$

如果 State 是 2D 但 Measurement 只有 1D，则 $H\in\mathbb R^{1\times2}$，正确写法是 $H=\begin{bmatrix} 0&1 \end{bmatrix}$。

## 5.4 F 与 H 的区别

这是后面 EKF 必须清楚的边界：$F$ 管“State 怎样随时间传播”，$H$ 管“Sensor 能从 State 中看到什么”。

$\displaystyle \begin{aligned} &\boxed{ F: State\ \text{怎样随时间传播} }\\ &\boxed{ H: Sensor\ \text{能从} State \text{中看到什么} } \end{aligned}$

即 $x_{k-1} \xrightarrow{F} x_k$，而 $x_k \xrightarrow{H}\mathrm{Predicted\ Measurement}$。

---

# 6. Prior / Posterior 与 Kalman 的两条主线

**Prior Estimate（先验估计）** 记为 $\hat x_k^-$，表示第 $k$ 帧已经完成 Prediction，但还没有融合当前 Observation。**Posterior Estimate（后验估计）** 记为 $\hat x_k^+$，表示第 $k$ 帧已经完成 Observation Correction。

注意上标 $-$ 和 $+$ 不是数学上的“负”和“正”，而是 Kalman Filter 流程阶段标记。状态流程：

$\displaystyle \boxed{ \hat x_{k-1}^+ \rightarrow Prediction \rightarrow \hat x_k^- \rightarrow Correction \rightarrow \hat x_k^+ }$

Covariance 同理：

$\displaystyle P_{k-1}^+ \rightarrow P_k^- \rightarrow P_k^+$

其中 $P_k^-$ 表示第 $k$ 帧 Observation 融合之前的状态 uncertainty。

## 6.1 Prediction —— State

$\displaystyle \boxed{ \hat x_k^- = F_k\hat x_{k-1}^+ + B_ku_k }$

它表示使用 Motion Model，把上一帧 corrected state 推到当前时刻。例如 $\hat x_{k-1}^+=\begin{bmatrix} 10\\ 2 \end{bmatrix}$、$\Delta t=0.5s$、$F=\begin{bmatrix} 1&0.5\\ 0&1 \end{bmatrix}$，得到：

$\displaystyle \hat x_k^-= \begin{bmatrix} 11\\ 2 \end{bmatrix}$

也就是 $p_k^-=11m$、$v_k^-=2m/s$。

## 6.2 Prediction —— Covariance

状态需要 Prediction，uncertainty 也必须 Prediction。核心公式：

$\displaystyle \boxed{ P_k^- = F_kP_{k-1}^+F_k^T+Q_k }$

**为什么是 $FPF^T$**：先从一维开始。若 $y=ax$ 且 $Var(x)=\sigma_x^2$，由于 $\delta y=a\delta x$，所以 $Var(y)=a^2Var(x)$。推广到多维 $y=Fx$，误差满足 $\delta y=F\delta x$，Covariance：

$\displaystyle P_y = E[\delta y\delta y^T]$

代入 $\delta y=F\delta x$ 得到 $P_y = E[(F\delta x)(F\delta x)^T]$，最终：

$\displaystyle \boxed{ P_y = FP_xF^T }$

所以：

$\displaystyle \boxed{ FPF^T = \text{旧} uncertainty \text{根据} State propagation \text{关系进行传播} }$

**为什么还要 +Q**：$FPF^T$ 只考虑上一帧已经存在的 uncertainty 如何传播，而本轮 Prediction 本身还会新引入 Gyroscope noise、bias random walk、model uncertainty 等。所以 $+Q$ 表示：

$\displaystyle \boxed{ \text{本次} Prediction \text{新加入的} Process uncertainty }$

最终：

$\displaystyle \boxed{ P_k^- = \text{旧} uncertainty \text{的传播} + \text{新的} Process uncertainty }$

一个重要判断：即使 $Q=0$，也不等于 $P_k^-=0$，因为旧 uncertainty 仍会通过 $FPF^T$ 继续传播。

---

# 7. Observation 到来后：Residual / Innovation 与 S

## 7.1 Residual / Innovation

Prediction 完成后得到 $\hat x_k^-$，Sensor 提供 $z_k$，而 Prediction 认为 Sensor 理论上应该看到 $H\hat x_k^-$，所以：

$\displaystyle \boxed{ r_k = z_k-H\hat x_k^- }$

其中 **Residual = 残差**，**Innovation = 新息 / 创新量**。例如 $z=5.0$、$H\hat x^-=4.8$，则 $r=0.2$。

边界：Residual ≠ True Error。$r=0.2$ 只能说明 Observation 和 Prediction 相差 0.2，不能说明 Prediction 相对真实状态一定错了 0.2，因为 $z$ 自己也存在 Measurement Noise。也就是说，不能直接把 residual 当真实误差，因为观测值本身也可能不准确。

## 7.2 Innovation Covariance S：新息协方差

$\displaystyle \boxed{ S = HP^-H^T+R }$

它描述：

$\displaystyle \boxed{ Residual / Innovation \text{自身的不确定性} }$

其中 $HP^-H^T$ 表示 Prediction uncertainty 映射到 Measurement Space 后的 uncertainty，而 $R$ 表示 Measurement 自己的 uncertainty。所以：

$\displaystyle \boxed{ S =\mathrm{Prediction\ uncertainty\ in\ measurement\ space}+\mathrm{Measurement\ uncertainty}}$

S 不是没有物理意义的“中间矩阵”。

---

# 8. Kalman Gain 与 Update

## 8.1 Kalman Gain：卡尔曼增益

$\displaystyle \boxed{ K = P^-H^TS^{-1} }$

即：

$\displaystyle K = P^-H^T (HP^-H^T+R)^{-1}$

在一维并且 $H=1$ 时：

$\displaystyle \boxed{ K= \frac{P^-}{P^-+R} }$

核心直觉：如果 $P^-\gg R$，说明 Prediction 很不确定、Measurement 比较可靠，则 $K\rightarrow1$，所以：

$\displaystyle \boxed{ \text{更相信} Measurement }$

如果 $P^-\ll R$，说明 Prediction 很可靠、Measurement 很不可靠，则 $K\rightarrow0$，所以：

$\displaystyle \boxed{ \text{更相信} Prediction }$

因此：

$\displaystyle \boxed{\mathrm{Kalman\ Gain}= \text{根据} Prediction uncertainty \text{和} Measurement uncertainty \text{自动计算的动态融合权重} }$

它不是人为固定设置的常数。

## 8.2 State Update：状态更新

$\displaystyle \boxed{ \hat x^+ = \hat x^-+Kr }$

也就是：

$\displaystyle \hat x^+ = \hat x^- + K(z-H\hat x^-)$

直觉是：

$\displaystyle \boxed{ \text{从} Prediction \text{出发，沿着} Observation \text{指出的方向修正一部分} }$

修正多少由 $K$ 决定：若 $K\approx1$ 则结果更靠近 Measurement，若 $K\approx0$ 则结果更靠近 Prediction。

## 8.3 Covariance Update：协方差更新

融合 Observation 后，不仅 State 要更新，uncertainty 也要更新。基础形式：

$\displaystyle \boxed{ P^+ = (I-KH)P^- }$

一维、$H=1$ 时：

$\displaystyle P^+=(1-K)P^-$

如果 Measurement 很可靠（$K$ 大），通常 $P^+\ll P^-$，说明融合新信息后我们对 State 更确定。如果 Measurement 很差，$R\text{ 大} \Rightarrow K\approx0$，则 $P^+\approx P^-$，也就是说一个很不可靠的 Observation 不会让滤波器凭空变得非常有信心。

## 8.4 Joseph Form：Joseph 形式

$\displaystyle \boxed{ P^+ = (I-KH)P^-(I-KH)^T + KRK^T }$

称为 **Joseph Form = Joseph 形式**。目前只需要知道：它和基础 covariance update 目标相同，工程实现中数值稳定性通常更好，后续实际实现 EKF 时可再深入。

---

# 9. 完整 Kalman Filter 循环（八步）

上一帧 Correction 后得到 $\hat x_{k-1}^+,\quad P_{k-1}^+$，然后依次执行八步。

| 步骤 | 公式 |
|---|---|
| Step 1 State Prediction | $\hat x_k^- = F\hat x_{k-1}^+ + Bu$ |
| Step 2 Covariance Prediction | $P_k^- = FP_{k-1}^+F^T+Q$ |
| Step 3 Measurement | $z_k$ |
| Step 4 Residual / Innovation | $r_k = z_k-H\hat x_k^-$ |
| Step 5 Innovation Covariance | $S_k = HP_k^-H^T+R$ |
| Step 6 Kalman Gain | $K_k = P_k^-H^TS_k^{-1}$ |
| Step 7 State Correction | $\hat x_k^+ = \hat x_k^-+K_kr_k$ |
| Step 8 Covariance Correction | $P_k^+ = (I-K_kH)P_k^-$ |

然后 $\hat x_k^+,P_k^+$ 进入下一帧。

---

# 10. 一维 Kalman Filter 手算例子 1

给定 $x^-=4.8$、$P^-=0.4$、$z=5.0$、$R=0.1$、$H=1$。

Kalman Gain：$K = \frac{0.4}{0.4+0.1} = 0.8$。

Residual：$r = 5.0-4.8 = 0.2$。

State Update：$x^+ = 4.8+0.8\times0.2$，得到

$\displaystyle \boxed{ x^+=4.96 }$

结果明显更靠近 Measurement $5.0$，原因是 $P^->R$，所以 Measurement 相对更可靠。

Covariance Update：$P^+ = (1-0.8)\times0.4$，得到

$\displaystyle \boxed{ P^+=0.08 }$

uncertainty 从 $0.4$ 降低到 $0.08$；融合有效 Observation 后，滤波器对 State 更有信心。

当时的算术错误：最初把 $0.2\times0.4$ 算成了 $0.04$，后修正为 $0.08$。这里 $(1-0.8)\times0.4$ 的正确结果就是 $0.08$，是单纯算术错误，不影响 Kalman Filter 概念理解。

---

# 11. 一维 Kalman Filter 手算例子 2

第二组：$x^-=4.8$、$z=5.0$、$P^-=0.1$、$R=10$。

Kalman Gain：$K = \frac{0.1}{0.1+10} \approx0.0099$，非常接近 $0$，所以更相信 Prediction。

Residual：$r=0.2$，和第一组相同。这说明：

$\displaystyle \boxed{ Residual \text{只描述} Prediction \text{与} Measurement \text{差多少} }$

而：

$\displaystyle \boxed{ \text{这个差应该用于修正多少，由} K \text{决定} }$

State Update：$x^+ = 4.8+0.0099\times0.2$，得到

$\displaystyle \boxed{ x^+\approx4.802 }$

几乎仍在 Prediction $4.8$ 附近。

Covariance Update：$P^+ = (1-0.0099)\times0.1$，得到

$\displaystyle \boxed{ P^+\approx0.099 }$

几乎没有下降。原因是 $R\gg P^-$，这一帧 Measurement 太不可靠，没有提供多少有效信息。

## 11.1 一维 KF 与矩阵 KF 的关系

一维里 $P,Q,R,K$ 是数字；多维中 $P,Q,R,K,F,H$ 变成 Matrix（矩阵），但思想没有改变。多维只是同时估计多个 State、State 之间可能存在 covariance、Sensor 不一定能观测全部 State、需要 H 完成 State Space → Measurement Space 映射、uncertainty 需要 Covariance Matrix 表达。所以：

$\displaystyle \boxed{\mathrm{Matrix\ KF}\neq \text{另一套算法} }$

而是：

$\displaystyle \boxed{ \text{一维} Kalman Filter \text{在多维相互关联} State \text{上的推广} }$

---

# 12. 与题目二：IMU 姿态 EKF 的连接

Day 2 的 Gyroscope Integration 链条 $\omega_m \rightarrow \hat{\omega} \rightarrow \Delta\theta \rightarrow \Delta q \rightarrow q_{k+1}$，在 EKF 框架中属于：

$\displaystyle \boxed{ Prediction }$

题目中的 `pose_cov.csv` 提供 FAST-LIO 位姿结果，它在 EKF 中属于：

$\displaystyle \boxed{ Observation }$

因此：

$\displaystyle \begin{aligned} &\boxed{ IMU \rightarrow Prediction }\\ &\boxed{ FAST\text{-}LIO \rightarrow Observation } \end{aligned}$

题目规定 position covariance 为 `cov_00`、`cov_11`、`cov_22`，attitude covariance 为 `cov_33`、`cov_44`、`cov_55`。所以姿态 Observation Noise Covariance 可概念上写成：

$\displaystyle \boxed{ R_\theta = diag( cov_{33}, cov_{44}, cov_{55} ) }$

位置选做：

$\displaystyle \boxed{ R_p = diag( cov_{00}, cov_{11}, cov_{22} ) }$

这些 covariance 属于：

$\displaystyle \boxed{ R }$

因为它们描述的是 FAST-LIO Observation 本身的 uncertainty。

---

# 13. Day 3 的核心心智模型

所有公式背后的真正核心是：

$\displaystyle \boxed{\mathrm{Kalman\ Filter}\text{不是只维护} State Estimate\text{，} \text{还同时维护} State Estimate \text{的} Uncertainty }$

每一帧有两条并行主线。State Estimate 主线：

$\displaystyle \hat x_{k-1}^+ \rightarrow \hat x_k^- \rightarrow \hat x_k^+$

State Uncertainty 主线：

$\displaystyle P_{k-1}^+ \rightarrow P_k^- \rightarrow P_k^+$

Prediction 做两件事：

$\displaystyle \boxed{ \text{传播} State + \text{传播} / \text{增加} Uncertainty }$

Observation 也携带两类信息：

$\displaystyle \boxed{ Measurement +\mathrm{Measurement\ Uncertainty}}$

Kalman Filter 再根据 $P^-$ 和 $R$ 判断 Prediction 与 Measurement 各应该相信多少。

---

# 14. 关键认知修正与必须保持清楚的概念边界

Day 3 学习过程中有四处需要修正的理解。

1. **Quaternion 归一化和 State Accuracy 混淆**：最初的说法是“已经算出 Quaternion 后还需要归一化，所以不能认为绝对准确”，这混了两个层级。归一化解决的是 $\|q\|=1$，即 Quaternion 表示是否合法；Probability / Covariance 解决的是“这个合法 Quaternion 到底有多可信”。最终明确 $Normalization \neq\mathrm{Uncertainty\ Estimation}$。

2. **Observation Matrix H 的维度**：最初在“State 为 [position, velocity]，Sensor 只测 velocity”时写成了 $2\times2$ 矩阵。后来明确 $H\in\mathbb R^{m\times n}$，其中 $n$ 是 State Dimension，$m$ 是 Measurement Dimension。这里只有一个 measurement，所以 $H=\begin{bmatrix} 0&1 \end{bmatrix}$。

3. **P 的物理含义容易和 Error 混淆**：最终明确 $P$ 不是当前实际 Error，它描述：

$\displaystyle \boxed{ Error \text{的统计} uncertainty }$

所以不能说“P 大就说明当前一定错得很多”。

4. **Covariance Update 中一次算术错误**：$(1-0.8)\times0.4$ 正确结果是 $0.08$，概念判断没有问题。

此外，以下概念边界必须保持清楚。

True State 与 Estimated State 不同：

$\displaystyle x_{true} \neq \hat x$

一般情况下不能认为两者严格相同。

Error 与 Covariance 不同：$e=x-\hat x$ 是实际 Estimation Error，而 $P=E[ee^T]$ 描述的是 Error 的统计 uncertainty。

Variance 与 Standard Deviation 不同：

$\displaystyle \begin{aligned} &Variance=\sigma^2\\ &\mathrm{Standard\ Deviation}=\sigma \end{aligned}$

Covariance 与 Correlation Coefficient 不同：Covariance 能反映两个变量误差之间的联合变化关系，但不能直接和标准化后的 Correlation Coefficient（相关系数）等同。

P ≠ Q ≠ R：

$\displaystyle \begin{aligned} &P: \text{当前} State Estimate uncertainty\\ &Q: Prediction \text{新增} process uncertainty\\ &R: Observation uncertainty \end{aligned}$

F ≠ H：

$\displaystyle \begin{aligned} &F:\mathrm{State\ propagation}\\ &H: State\rightarrow Measurement \end{aligned}$

Residual ≠ True Error：$r=z-H\hat x^-$ 只是 Observation 与 Prediction 的差。因为 Observation 也有 Noise，所以它不是实际 State Error。

Kalman Gain 不是人为固定权重：$K$ 由 $P^-,H,R$ 动态计算。

---

# 15. 主线总结与 Day 4 衔接

Day 1 解决“三维姿态怎样表示”，学习 Rotation Matrix、Rotation Vector、Euler Angle、Quaternion：

$\displaystyle \boxed{ \text{三维姿态怎样表示} }$

Day 2 解决“IMU 怎样预测姿态”，学习 SO(3)、so(3)、Exp / Log、Gyroscope Measurement Model、Bias Compensation、$\Delta\theta$、$\Delta q$、Quaternion Propagation、Drift：

$\displaystyle \boxed{ IMU \text{怎样预测姿态} }$

Day 3 解决“Prediction 和 Observation 怎样根据 uncertainty 进行融合”，学习 Probability、Gaussian、Mean、Variance、Covariance、P / Q / R、F / H、Prediction、Residual、S、Kalman Gain、State Update、Covariance Update、完整 Linear Kalman Filter：

$\displaystyle \boxed{ Prediction \text{和} Observation \text{怎样根据} uncertainty \text{进行融合} }$

Day 3 学习的是：

$\displaystyle \boxed{\mathrm{Linear\ Kalman\ Filter}}$

但题目二真正的姿态状态包含 Quaternion，Quaternion Rotation 是 Nonlinear（非线性）的，而且：

$\displaystyle Quaternion: \mathrm{4\ parameters}$

但 Rotation 实际只有：

$\displaystyle \mathrm{3\ DOF}$

因此 Day 4 要回答：普通 KF 的线性 State 模型无法直接完整处理 Quaternion Rotation，那么应该如何设计姿态 EKF？最终目标是：

$\displaystyle \boxed{\mathrm{Quaternion\ Prediction}+\mathrm{Error\ State}+ EKF + FAST\text{-}LIO\mathrm{\ Attitude\ Observation}}$

得到：

$\displaystyle \boxed{ \mathrm{6D\ Attitude}\ Error\text{-}\mathrm{State\ EKF} }$

最终一句话总结：

$\displaystyle \boxed{ Day\ 1\text{：姿态怎么表示} \rightarrow Day\ 2\text{：}IMU \text{怎么预测姿态} \rightarrow Day\ 3\text{：}Prediction \text{和} Observation \text{怎么依据} Uncertainty \text{融合} }$

Day 4 才会进一步解决：

$\displaystyle \boxed{ \text{如何把} Kalman Filter \text{的思想真正落到非线性的} Quaternion \text{姿态估计上} }$
