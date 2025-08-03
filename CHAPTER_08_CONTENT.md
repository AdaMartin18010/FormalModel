# 第八章：工程科学模型 / Chapter 8: Engineering Science Models

## 8.1 优化模型 / Optimization Models

工程科学中的优化模型是解决复杂工程问题的核心工具。优化模型通过数学方法寻找最优解，广泛应用于工程设计、生产调度、资源分配等领域。

### 8.1.1 线性规划模型 / Linear Programming Model

线性规划是优化理论的基础，它处理目标函数和约束条件都是线性的优化问题。

#### 标准形式

**数学表示**：
线性规划的标准形式为：
$$\begin{align}
\text{最大化} \quad & c^T x \\
\text{约束条件} \quad & Ax \leq b \\
& x \geq 0
\end{align}$$

其中：
- $x \in \mathbb{R}^n$ 是决策变量向量
- $c \in \mathbb{R}^n$ 是目标函数系数向量
- $A \in \mathbb{R}^{m \times n}$ 是约束系数矩阵
- $b \in \mathbb{R}^m$ 是约束右端项向量

**可行域**：
可行域定义为满足所有约束条件的点的集合：
$$\mathcal{F} = \{x \in \mathbb{R}^n : Ax \leq b, x \geq 0\}$$

**最优解**：
最优解 $x^*$ 满足：
$$c^T x^* = \max\{c^T x : x \in \mathcal{F}\}$$

#### 对偶理论

**对偶问题**：
原问题的对偶形式为：
$$\begin{align}
\text{最小化} \quad & b^T y \\
\text{约束条件} \quad & A^T y \geq c \\
& y \geq 0
\end{align}$$

**弱对偶性**：
对于任意可行解 $x$ 和 $y$，有：
$$c^T x \leq b^T y$$

**强对偶性**：
如果原问题和对偶问题都有有限的最优解，则：
$$c^T x^* = b^T y^*$$

**互补松弛条件**：
最优解满足：
$$y_i^*(Ax^* - b)_i = 0, \quad i = 1, 2, ..., m$$
$$x_j^*(A^T y^* - c)_j = 0, \quad j = 1, 2, ..., n$$

#### 单纯形法

**基本可行解**：
设 $B$ 是 $A$ 的基矩阵，基本可行解为：
$$x_B = B^{-1}b, \quad x_N = 0$$

**进基变量选择**：
选择具有最大正检验数的非基变量：
$$\Delta_j = c_j - c_B^T B^{-1} A_j$$

**出基变量选择**：
使用最小比值规则：
$$\theta = \min\left\{\frac{(B^{-1}b)_i}{(B^{-1}A_j)_i} : (B^{-1}A_j)_i > 0\right\}$$

**最优性检验**：
当所有检验数 $\Delta_j \leq 0$ 时，当前解为最优解。

#### 灵敏度分析

**目标函数系数变化**：
设 $c_j$ 变化 $\Delta c_j$，新的检验数为：
$$\Delta_j' = \Delta_j + \Delta c_j$$

**约束右端项变化**：
设 $b_i$ 变化 $\Delta b_i$，新的基本解为：
$$x_B' = x_B + B^{-1}e_i \Delta b_i$$

**影子价格**：
影子价格 $y_i^*$ 表示约束 $i$ 右端项增加一个单位时目标函数的变化量。

### 8.1.2 非线性规划模型 / Nonlinear Programming Model

非线性规划处理目标函数或约束条件中至少有一个是非线性的优化问题。

#### 无约束优化

**梯度法**：
迭代公式：
$$x^{k+1} = x^k - \alpha_k \nabla f(x^k)$$

其中 $\alpha_k$ 是步长，$\nabla f(x^k)$ 是梯度。

**牛顿法**：
迭代公式：
$$x^{k+1} = x^k - [\nabla^2 f(x^k)]^{-1} \nabla f(x^k)$$

其中 $\nabla^2 f(x^k)$ 是Hessian矩阵。

**拟牛顿法**：
使用近似Hessian矩阵 $H_k$：
$$x^{k+1} = x^k - H_k^{-1} \nabla f(x^k)$$

BFGS更新公式：
$$H_{k+1} = H_k + \frac{s_k s_k^T}{s_k^T y_k} - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k}$$

其中 $s_k = x^{k+1} - x^k$，$y_k = \nabla f(x^{k+1}) - \nabla f(x^k)$。

**共轭梯度法**：
搜索方向：
$$d^{k+1} = -\nabla f(x^{k+1}) + \beta_k d^k$$

其中 $\beta_k = \frac{\|\nabla f(x^{k+1})\|^2}{\|\nabla f(x^k)\|^2}$。

#### 约束优化

**拉格朗日乘数法**：
对于等式约束问题：
$$\begin{align}
\text{最小化} \quad & f(x) \\
\text{约束条件} \quad & h_i(x) = 0, \quad i = 1, 2, ..., m
\end{align}$$

拉格朗日函数：
$$L(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i h_i(x)$$

KKT条件：
$$\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla h_i(x^*) = 0$$
$$h_i(x^*) = 0, \quad i = 1, 2, ..., m$$

**惩罚函数法**：
将约束转化为惩罚项：
$$P(x, \mu) = f(x) + \frac{\mu}{2} \sum_{i=1}^m [h_i(x)]^2$$

**障碍函数法**：
对于不等式约束 $g_i(x) \leq 0$：
$$B(x, \mu) = f(x) - \mu \sum_{i=1}^p \ln(-g_i(x))$$

#### 全局优化

**遗传算法**：
1. 初始化种群
2. 计算适应度
3. 选择操作
4. 交叉操作
5. 变异操作
6. 重复直到收敛

**模拟退火**：
接受概率：
$$P(接受) = \min\left\{1, \exp\left(-\frac{\Delta E}{T}\right)\right\}$$

其中 $\Delta E$ 是能量差，$T$ 是温度。

**粒子群优化**：
速度和位置更新：
$$v_i^{t+1} = w v_i^t + c_1 r_1(p_i - x_i^t) + c_2 r_2(g - x_i^t)$$
$$x_i^{t+1} = x_i^t + v_i^{t+1}$$

其中 $w$ 是惯性权重，$c_1, c_2$ 是学习因子。

#### 凸优化

**凸函数**：
函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是凸的，如果：
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

对所有 $x, y \in \mathbb{R}^n$ 和 $\lambda \in [0,1]$ 成立。

**凸集**：
集合 $C \subseteq \mathbb{R}^n$ 是凸的，如果：
$$\lambda x + (1-\lambda)y \in C$$

对所有 $x, y \in C$ 和 $\lambda \in [0,1]$ 成立。

**凸规划**：
$$\begin{align}
\text{最小化} \quad & f(x) \\
\text{约束条件} \quad & g_i(x) \leq 0, \quad i = 1, 2, ..., m \\
& h_i(x) = 0, \quad i = 1, 2, ..., p
\end{align}$$

其中 $f$ 和 $g_i$ 是凸函数，$h_i$ 是仿射函数。

### 8.1.3 整数规划模型 / Integer Programming Model

整数规划要求部分或全部决策变量取整数值。

#### 纯整数规划

**数学形式**：
$$\begin{align}
\text{最大化} \quad & c^T x \\
\text{约束条件} \quad & Ax \leq b \\
& x \geq 0 \\
& x_j \in \mathbb{Z}, \quad j = 1, 2, ..., n
\end{align}$$

**分支定界法**：
1. 求解线性松弛问题
2. 如果最优解不满足整数约束，选择分支变量
3. 添加分支约束 $x_j \leq \lfloor x_j^* \rfloor$ 和 $x_j \geq \lceil x_j^* \rceil$
4. 递归求解子问题

**割平面法**：
添加有效不等式：
$$\sum_{j \in N} a_{ij} x_j \leq b_i$$

其中 $N$ 是非基变量集合。

#### 混合整数规划

**数学形式**：
$$\begin{align}
\text{最大化} \quad & c^T x + d^T y \\
\text{约束条件} \quad & Ax + By \leq b \\
& x \geq 0, y \geq 0 \\
& x_j \in \mathbb{Z}, \quad j \in I
\end{align}$$

其中 $I$ 是整数变量集合。

**分支定价法**：
结合列生成和分支定界：
1. 主问题：选择列（变量）
2. 子问题：生成新的列
3. 分支：添加整数约束

#### 0-1规划

**数学形式**：
$$\begin{align}
\text{最大化} \quad & c^T x \\
\text{约束条件} \quad & Ax \leq b \\
& x_j \in \{0,1\}, \quad j = 1, 2, ..., n
\end{align}$$

**集合覆盖问题**：
$$\begin{align}
\text{最小化} \quad & \sum_{j=1}^n c_j x_j \\
\text{约束条件} \quad & \sum_{j: i \in S_j} x_j \geq 1, \quad i = 1, 2, ..., m \\
& x_j \in \{0,1\}, \quad j = 1, 2, ..., n
\end{align}$$

**背包问题**：
$$\begin{align}
\text{最大化} \quad & \sum_{j=1}^n v_j x_j \\
\text{约束条件} \quad & \sum_{j=1}^n w_j x_j \leq W \\
& x_j \in \{0,1\}, \quad j = 1, 2, ..., n
\end{align}$$

其中 $v_j$ 是价值，$w_j$ 是重量，$W$ 是容量。

#### 网络优化

**最短路径问题**：
使用Dijkstra算法：
$$d[v] = \min\{d[v], d[u] + w(u,v)\}$$

其中 $d[v]$ 是从起点到节点 $v$ 的最短距离。

**最大流问题**：
Ford-Fulkerson算法：
1. 找到增广路径
2. 计算路径上的最小容量
3. 更新网络流量

**最小费用流问题**：
$$\begin{align}
\text{最小化} \quad & \sum_{(i,j)} c_{ij} x_{ij} \\
\text{约束条件} \quad & \sum_j x_{ij} - \sum_j x_{ji} = b_i \\
& 0 \leq x_{ij} \leq u_{ij}
\end{align}$$

其中 $c_{ij}$ 是单位费用，$u_{ij}$ 是容量，$b_i$ 是供需。

### 8.1.4 多目标优化模型 / Multi-objective Optimization Model

多目标优化处理多个相互冲突的目标函数。

#### 帕累托最优

**帕累托支配**：
解 $x$ 支配解 $y$，如果：
$$f_i(x) \leq f_i(y), \quad \forall i$$
$$f_j(x) < f_j(y), \quad \text{至少一个} j$$

**帕累托前沿**：
非支配解的集合：
$$\mathcal{P} = \{x \in \mathcal{F} : \text{不存在} y \in \mathcal{F} \text{支配} x\}$$

**加权和方法**：
$$\min \sum_{i=1}^k w_i f_i(x)$$

其中 $w_i \geq 0$ 是权重，$\sum_{i=1}^k w_i = 1$。

#### 多目标算法

**NSGA-II算法**：
1. 非支配排序
2. 拥挤度距离计算
3. 精英策略
4. 遗传操作

**MOEA/D算法**：
1. 分解多目标问题
2. 邻域更新
3. 遗传操作

**多目标粒子群优化**：
1. 非支配解存储
2. 全局最优选择
3. 速度和位置更新

## 8.2 控制论模型 / Control Theory Models

### 8.2.1 线性控制系统 / Linear Control Systems

#### 状态空间模型

**连续时间系统**：
$$\begin{align}
\dot{x}(t) &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t)
\end{align}$$

其中：
- $x(t) \in \mathbb{R}^n$ 是状态向量
- $u(t) \in \mathbb{R}^m$ 是输入向量
- $y(t) \in \mathbb{R}^p$ 是输出向量
- $A, B, C, D$ 是系统矩阵

**离散时间系统**：
$$\begin{align}
x(k+1) &= Ax(k) + Bu(k) \\
y(k) &= Cx(k) + Du(k)
\end{align}$$

#### 传递函数模型

**单输入单输出系统**：
$$G(s) = \frac{Y(s)}{U(s)} = \frac{b_n s^n + b_{n-1} s^{n-1} + ... + b_0}{a_n s^n + a_{n-1} s^{n-1} + ... + a_0}$$

**零极点形式**：
$$G(s) = K \frac{(s - z_1)(s - z_2)...(s - z_m)}{(s - p_1)(s - p_2)...(s - p_n)}$$

其中 $z_i$ 是零点，$p_i$ 是极点。

#### 稳定性分析

**Lyapunov稳定性**：
系统在平衡点 $x_e$ 处稳定，如果对任意 $\epsilon > 0$，存在 $\delta > 0$，使得：
$$\|x(0) - x_e\| < \delta \Rightarrow \|x(t) - x_e\| < \epsilon$$

**Routh-Hurwitz判据**：
特征方程 $a_n s^n + a_{n-1} s^{n-1} + ... + a_0 = 0$ 的所有根具有负实部的充分必要条件是Routh表的第一列元素都为正。

**Nyquist判据**：
闭环系统稳定的充分必要条件是Nyquist图绕点 $(-1,0)$ 的圈数等于开环系统在右半平面的极点数。

### 8.2.2 非线性控制系统 / Nonlinear Control Systems

#### 非线性系统模型

**一般形式**：
$$\begin{align}
\dot{x} &= f(x, u, t) \\
y &= h(x, u, t)
\end{align}$$

其中 $f$ 和 $h$ 是非线性函数。

**Lyapunov函数**：
正定函数 $V(x)$ 满足：
$$\dot{V}(x) = \frac{\partial V}{\partial x} f(x) < 0$$

#### 反馈线性化

**输入-输出线性化**：
通过状态反馈和坐标变换，将非线性系统转化为线性系统。

**相对度**：
输出 $y$ 对输入 $u$ 的相对度是使 $\frac{\partial}{\partial u} L_f^r h(x) \neq 0$ 的最小整数 $r$。

**反馈控制律**：
$$u = \frac{v - L_f^r h(x)}{L_g L_f^{r-1} h(x)}$$

其中 $v$ 是新的控制输入。

### 8.2.3 自适应控制系统 / Adaptive Control Systems

#### 模型参考自适应控制

**参考模型**：
$$\dot{x}_m = A_m x_m + B_m r$$

其中 $x_m$ 是参考状态，$r$ 是参考输入。

**控制律**：
$$u = K_x(t) x + K_r(t) r$$

**参数更新律**：
$$\dot{K}_x = -\gamma e x^T$$
$$\dot{K}_r = -\gamma e r^T$$

其中 $e = x - x_m$ 是跟踪误差。

#### 自校正控制

**参数估计**：
使用递归最小二乘法：
$$\hat{\theta}(k) = \hat{\theta}(k-1) + K(k)[y(k) - \phi^T(k)\hat{\theta}(k-1)]$$

其中 $K(k)$ 是增益矩阵。

**控制律设计**：
基于估计参数设计控制器：
$$u(k) = -\frac{\hat{a}_1 y(k) + ... + \hat{a}_n y(k-n) + \hat{b}_1 u(k-1) + ... + \hat{b}_m u(k-m)}{\hat{b}_0}$$

### 8.2.4 鲁棒控制系统 / Robust Control Systems

#### H∞控制

**性能指标**：
$$\|T_{zw}\|_\infty = \sup_{\omega} \sigma_{\max}[T_{zw}(j\omega)] < \gamma$$

其中 $T_{zw}$ 是从干扰 $w$ 到输出 $z$ 的传递函数。

**Riccati方程**：
$$A^T X + XA + X(\gamma^{-2} B_1 B_1^T - B_2 B_2^T)X + C_1^T C_1 = 0$$

**控制器**：
$$u = -B_2^T X x$$

#### μ综合

**结构奇异值**：
$$\mu(M) = \frac{1}{\min\{\sigma(\Delta) : \det(I - M\Delta) = 0\}}$$

其中 $\Delta$ 是结构化的不确定性。

**D-K迭代**：
1. 固定 $D$，优化 $K$
2. 固定 $K$，优化 $D$
3. 重复直到收敛

## 8.3 信号处理模型 / Signal Processing Models

### 8.3.1 数字信号处理 / Digital Signal Processing

#### 离散时间信号

**采样定理**：
如果信号 $x(t)$ 的频谱 $X(f)$ 在 $|f| > f_m$ 时为零，则采样频率 $f_s$ 必须满足：
$$f_s > 2f_m$$

**离散时间傅里叶变换**：
$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}$$

**逆变换**：
$$x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n} d\omega$$

#### 数字滤波器

**FIR滤波器**：
$$y[n] = \sum_{k=0}^{N-1} h[k] x[n-k]$$

其中 $h[k]$ 是滤波器系数。

**IIR滤波器**：
$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

**频率响应**：
$$H(e^{j\omega}) = \frac{\sum_{k=0}^{M} b_k e^{-j\omega k}}{1 + \sum_{k=1}^{N} a_k e^{-j\omega k}}$$

#### 快速傅里叶变换

**FFT算法**：
$$X[k] = \sum_{n=0}^{N-1} x[n] W_N^{kn}$$

其中 $W_N = e^{-j2\pi/N}$。

**蝶形运算**：
$$X[k] = X_e[k] + W_N^k X_o[k]$$
$$X[k+N/2] = X_e[k] - W_N^k X_o[k]$$

其中 $X_e[k]$ 和 $X_o[k]$ 分别是偶数和奇数点的DFT。

### 8.3.2 自适应信号处理 / Adaptive Signal Processing

#### LMS算法

**更新公式**：
$$w(n+1) = w(n) + \mu e(n) x(n)$$

其中：
- $w(n)$ 是滤波器系数
- $\mu$ 是步长参数
- $e(n) = d(n) - y(n)$ 是误差信号
- $x(n)$ 是输入信号

**收敛条件**：
$$0 < \mu < \frac{2}{\lambda_{\max}}$$

其中 $\lambda_{\max}$ 是输入信号自相关矩阵的最大特征值。

#### RLS算法

**更新公式**：
$$w(n) = w(n-1) + k(n) e(n)$$

其中 $k(n)$ 是卡尔曼增益：
$$k(n) = P(n-1) x(n) [\lambda + x^T(n) P(n-1) x(n)]^{-1}$$

**逆相关矩阵更新**：
$$P(n) = \lambda^{-1} [P(n-1) - k(n) x^T(n) P(n-1)]$$

### 8.3.3 多速率信号处理 / Multirate Signal Processing

#### 采样率转换

**插值**：
在信号样本之间插入零值，然后进行低通滤波：
$$y[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-Lk]$$

其中 $L$ 是插值因子，$h[n]$ 是插值滤波器。

**抽取**：
先进行低通滤波，然后每隔 $M$ 个样本取一个：
$$y[n] = \sum_{k=-\infty}^{\infty} x[k] h[Mn-k]$$

其中 $M$ 是抽取因子。

#### 滤波器组

**分析滤波器组**：
$$X_k[m] = \sum_{n=-\infty}^{\infty} x[n] h_k[mN-n]$$

其中 $h_k[n]$ 是第 $k$ 个分析滤波器。

**综合滤波器组**：
$$y[n] = \sum_{k=0}^{K-1} \sum_{m=-\infty}^{\infty} X_k[m] f_k[n-mN]$$

其中 $f_k[n]$ 是第 $k$ 个综合滤波器。

**完美重构条件**：
$$\sum_{k=0}^{K-1} H_k(e^{j\omega}) F_k(e^{j\omega}) = K$$

## 8.4 材料科学模型 / Materials Science Models

### 8.4.1 晶体结构模型 / Crystal Structure Models

#### 布拉格定律

**衍射条件**：
$$2d \sin \theta = n\lambda$$

其中：
- $d$ 是晶面间距
- $\theta$ 是入射角
- $\lambda$ 是波长
- $n$ 是衍射级数

**倒易空间**：
倒易点阵向量：
$$\mathbf{G} = h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*$$

其中 $\mathbf{a}^*, \mathbf{b}^*, \mathbf{c}^*$ 是倒易基向量。

#### 电子结构

**布洛赫定理**：
电子波函数：
$$\psi_{n\mathbf{k}}(\mathbf{r}) = u_{n\mathbf{k}}(\mathbf{r}) e^{i\mathbf{k} \cdot \mathbf{r}}$$

其中 $u_{n\mathbf{k}}(\mathbf{r})$ 具有晶格周期性。

**能带结构**：
电子能量：
$$E_n(\mathbf{k}) = \frac{\hbar^2 k^2}{2m} + V_{eff}(\mathbf{k})$$

其中 $V_{eff}(\mathbf{k})$ 是有效势能。

### 8.4.2 相变模型 / Phase Transition Models

#### 热力学相变

**Gibbs自由能**：
$$G = H - TS$$

其中 $H$ 是焓，$S$ 是熵，$T$ 是温度。

**相平衡条件**：
$$\mu_1 = \mu_2 = ... = \mu_n$$

其中 $\mu_i$ 是第 $i$ 相的化学势。

**Clausius-Clapeyron方程**：
$$\frac{dP}{dT} = \frac{\Delta S}{\Delta V}$$

其中 $\Delta S$ 和 $\Delta V$ 是相变时的熵变和体积变化。

#### 动力学相变

**Avrami方程**：
转变分数：
$$f(t) = 1 - \exp(-kt^n)$$

其中 $k$ 是速率常数，$n$ 是Avrami指数。

**Johnson-Mehl-Avrami-Kolmogorov理论**：
$$f(t) = 1 - \exp\left(-\int_0^t I(\tau) \left[\int_\tau^t G(s) ds\right]^{d-1} d\tau\right)$$

其中 $I(\tau)$ 是成核率，$G(s)$ 是生长速率，$d$ 是维度。

### 8.4.3 力学性能模型 / Mechanical Properties Models

#### 弹性理论

**胡克定律**：
$$\sigma_{ij} = C_{ijkl} \epsilon_{kl}$$

其中 $\sigma_{ij}$ 是应力张量，$\epsilon_{kl}$ 是应变张量，$C_{ijkl}$ 是弹性常数张量。

**杨氏模量**：
$$E = \frac{\sigma}{\epsilon}$$

**泊松比**：
$$\nu = -\frac{\epsilon_{transverse}}{\epsilon_{axial}}$$

#### 塑性理论

**屈服准则**：
von Mises准则：
$$\sqrt{\frac{1}{2}[(\sigma_1 - \sigma_2)^2 + (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2]} = \sigma_y$$

其中 $\sigma_y$ 是屈服强度。

**流动法则**：
$$d\epsilon_{ij}^p = d\lambda \frac{\partial f}{\partial \sigma_{ij}}$$

其中 $f$ 是屈服函数，$d\lambda$ 是塑性乘子。

### 8.4.4 扩散模型 / Diffusion Models

#### Fick定律

**第一定律**：
$$J = -D \nabla c$$

其中 $J$ 是扩散通量，$D$ 是扩散系数，$c$ 是浓度。

**第二定律**：
$$\frac{\partial c}{\partial t} = D \nabla^2 c$$

**扩散系数**：
$$D = D_0 \exp\left(-\frac{Q}{RT}\right)$$

其中 $D_0$ 是频率因子，$Q$ 是激活能，$R$ 是气体常数。

#### 界面扩散

**Fisher模型**：
$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - v \frac{\partial c}{\partial x}$$

其中 $v$ 是界面移动速度。

**Kirkendall效应**：
由于不同组元的扩散系数不同，导致标记面移动。

## 8.5 机械工程模型 / Mechanical Engineering Models

### 8.5.1 动力学模型 / Dynamics Models

#### 刚体动力学

**牛顿-欧拉方程**：
$$\begin{align}
m \ddot{\mathbf{r}}_c &= \mathbf{F} \\
\mathbf{I}_c \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times \mathbf{I}_c \boldsymbol{\omega} &= \mathbf{M}_c
\end{align}$$

其中：
- $m$ 是质量
- $\mathbf{r}_c$ 是质心位置
- $\mathbf{I}_c$ 是转动惯量张量
- $\boldsymbol{\omega}$ 是角速度
- $\mathbf{F}$ 是外力
- $\mathbf{M}_c$ 是外力矩

**拉格朗日方程**：
$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = Q_i$$

其中 $L = T - V$ 是拉格朗日函数，$q_i$ 是广义坐标。

#### 振动分析

**单自由度系统**：
$$m \ddot{x} + c \dot{x} + kx = F(t)$$

其中 $m$ 是质量，$c$ 是阻尼系数，$k$ 是刚度系数。

**自然频率**：
$$\omega_n = \sqrt{\frac{k}{m}}$$

**阻尼比**：
$$\zeta = \frac{c}{2\sqrt{mk}}$$

**强迫振动响应**：
$$x(t) = A \cos(\omega t - \phi)$$

其中：
$$A = \frac{F_0/k}{\sqrt{(1-r^2)^2 + (2\zeta r)^2}}$$
$$\phi = \tan^{-1}\left(\frac{2\zeta r}{1-r^2}\right)$$

其中 $r = \omega/\omega_n$ 是频率比。

### 8.5.2 流体力学模型 / Fluid Mechanics Models

#### 连续方程

**质量守恒**：
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

其中 $\rho$ 是密度，$\mathbf{v}$ 是速度向量。

**不可压缩流体**：
$$\nabla \cdot \mathbf{v} = 0$$

#### 动量方程

**Navier-Stokes方程**：
$$\rho \left(\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v}\right) = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{f}$$

其中：
- $p$ 是压力
- $\mu$ 是动力粘度
- $\mathbf{f}$ 是体积力

**雷诺数**：
$$Re = \frac{\rho VL}{\mu}$$

其中 $V$ 是特征速度，$L$ 是特征长度。

#### 边界层理论

**边界层厚度**：
$$\delta(x) = 5.0 \sqrt{\frac{\nu x}{U_\infty}}$$

其中 $\nu$ 是运动粘度，$U_\infty$ 是自由流速度。

**摩擦系数**：
$$C_f = \frac{0.664}{\sqrt{Re_x}}$$

其中 $Re_x = \frac{U_\infty x}{\nu}$。

### 8.5.3 热传导模型 / Heat Conduction Models

#### 傅里叶热传导

**热传导方程**：
$$\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + \dot{q}$$

其中：
- $\rho$ 是密度
- $c_p$ 是比热容
- $k$ 是热导率
- $\dot{q}$ 是内热源

**热流密度**：
$$\mathbf{q} = -k \nabla T$$

#### 边界条件

**第一类边界条件**：
$$T = T_s$$

**第二类边界条件**：
$$-k \frac{\partial T}{\partial n} = q_s$$

**第三类边界条件**：
$$-k \frac{\partial T}{\partial n} = h(T - T_\infty)$$

其中 $h$ 是对流换热系数。

#### 瞬态热传导

**一维热传导**：
$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

其中 $\alpha = \frac{k}{\rho c_p}$ 是热扩散率。

**解析解**：
对于半无限大平板：
$$T(x,t) = T_0 + (T_s - T_0) \text{erfc}\left(\frac{x}{2\sqrt{\alpha t}}\right)$$

其中 $\text{erfc}$ 是互补误差函数。

### 8.5.4 疲劳分析模型 / Fatigue Analysis Models

#### S-N曲线

**疲劳寿命**：
$$N = C S^{-m}$$

其中：
- $N$ 是疲劳寿命
- $S$ 是应力幅值
- $C$ 和 $m$ 是材料常数

**Goodman关系**：
$$\frac{\sigma_a}{\sigma_{-1}} + \frac{\sigma_m}{\sigma_u} = 1$$

其中：
- $\sigma_a$ 是应力幅值
- $\sigma_m$ 是平均应力
- $\sigma_{-1}$ 是疲劳极限
- $\sigma_u$ 是抗拉强度

#### 断裂力学

**应力强度因子**：
$$K_I = Y \sigma \sqrt{\pi a}$$

其中：
- $Y$ 是几何因子
- $\sigma$ 是远场应力
- $a$ 是裂纹长度

**Paris定律**：
$$\frac{da}{dN} = C (\Delta K)^m$$

其中 $\Delta K$ 是应力强度因子范围。

## 8.6 电子工程模型 / Electronic Engineering Models

### 8.6.1 电路分析模型 / Circuit Analysis Models

#### 基尔霍夫定律

**电流定律（KCL）**：
$$\sum_{k=1}^n i_k = 0$$

**电压定律（KVL）**：
$$\sum_{k=1}^n v_k = 0$$

#### 电路元件模型

**电阻**：
$$v = Ri$$

**电感**：
$$v = L \frac{di}{dt}$$

**电容**：
$$i = C \frac{dv}{dt}$$

#### 网络分析

**节点分析法**：
$$\mathbf{Y} \mathbf{V} = \mathbf{I}$$

其中：
- $\mathbf{Y}$ 是节点导纳矩阵
- $\mathbf{V}$ 是节点电压向量
- $\mathbf{I}$ 是节点电流向量

**网孔分析法**：
$$\mathbf{Z} \mathbf{I} = \mathbf{V}$$

其中：
- $\mathbf{Z}$ 是网孔阻抗矩阵
- $\mathbf{I}$ 是网孔电流向量
- $\mathbf{V}$ 是网孔电压向量

### 8.6.2 半导体器件模型 / Semiconductor Device Models

#### PN结模型

**理想二极管方程**：
$$I = I_s \left(e^{V/V_T} - 1\right)$$

其中：
- $I_s$ 是反向饱和电流
- $V_T = kT/q$ 是热电压

**扩散电流**：
$$I_n = qA D_n \frac{dn}{dx}$$
$$I_p = -qA D_p \frac{dp}{dx}$$

其中 $D_n$ 和 $D_p$ 是扩散系数。

#### 晶体管模型

**Ebers-Moll模型**：
$$\begin{align}
I_E &= I_{ES} (e^{V_{BE}/V_T} - 1) - \alpha_R I_{CS} (e^{V_{BC}/V_T} - 1) \\
I_C &= \alpha_F I_{ES} (e^{V_{BE}/V_T} - 1) - I_{CS} (e^{V_{BC}/V_T} - 1)
\end{align}$$

其中 $\alpha_F$ 和 $\alpha_R$ 是正向和反向电流增益。

**小信号模型**：
$$\begin{align}
i_b &= \frac{v_{be}}{r_\pi} + \frac{v_{ce}}{r_\mu} \\
i_c &= g_m v_{be} + \frac{v_{ce}}{r_o}
\end{align}$$

其中：
- $r_\pi$ 是输入电阻
- $g_m$ 是跨导
- $r_o$ 是输出电阻

### 8.6.3 电磁场模型 / Electromagnetic Field Models

#### 麦克斯韦方程

**微分形式**：
$$\begin{align}
\nabla \cdot \mathbf{D} &= \rho \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{H} &= \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}
\end{align}$$

其中：
- $\mathbf{D}$ 是电位移向量
- $\mathbf{B}$ 是磁感应强度
- $\mathbf{E}$ 是电场强度
- $\mathbf{H}$ 是磁场强度
- $\rho$ 是电荷密度
- $\mathbf{J}$ 是电流密度

#### 波动方程

**电磁波方程**：
$$\nabla^2 \mathbf{E} - \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2} = 0$$

**平面波解**：
$$\mathbf{E} = \mathbf{E}_0 e^{j(\omega t - \mathbf{k} \cdot \mathbf{r})}$$

其中 $\mathbf{k}$ 是波向量。

#### 传输线理论

**电报方程**：
$$\begin{align}
\frac{\partial V}{\partial x} &= -L \frac{\partial I}{\partial t} - RI \\
\frac{\partial I}{\partial x} &= -C \frac{\partial V}{\partial t} - GV
\end{align}$$

其中：
- $L$ 是单位长度电感
- $C$ 是单位长度电容
- $R$ 是单位长度电阻
- $G$ 是单位长度电导

**特性阻抗**：
$$Z_0 = \sqrt{\frac{R + j\omega L}{G + j\omega C}}$$

**传播常数**：
$$\gamma = \sqrt{(R + j\omega L)(G + j\omega C)}$$

### 8.6.4 通信系统模型 / Communication System Models

#### 调制理论

**幅度调制（AM）**：
$$s(t) = A_c [1 + m(t)] \cos(\omega_c t)$$

其中：
- $A_c$ 是载波幅度
- $m(t)$ 是调制信号
- $\omega_c$ 是载波频率

**频率调制（FM）**：
$$s(t) = A_c \cos\left(\omega_c t + k_f \int_{-\infty}^t m(\tau) d\tau\right)$$

其中 $k_f$ 是频率调制常数。

**相位调制（PM）**：
$$s(t) = A_c \cos(\omega_c t + k_p m(t))$$

其中 $k_p$ 是相位调制常数。

#### 数字调制

**相移键控（PSK）**：
$$s(t) = A_c \cos(\omega_c t + \phi_i)$$

其中 $\phi_i = \frac{2\pi i}{M}$ 是相位。

**正交幅度调制（QAM）**：
$$s(t) = A_i \cos(\omega_c t) + B_i \sin(\omega_c t)$$

其中 $A_i$ 和 $B_i$ 是幅度。

#### 信道模型

**加性高斯白噪声（AWGN）**：
$$r(t) = s(t) + n(t)$$

其中 $n(t)$ 是高斯白噪声。

**瑞利衰落**：
$$h(t) = \alpha(t) e^{j\phi(t)}$$

其中 $\alpha(t)$ 是瑞利分布的幅度，$\phi(t)$ 是均匀分布的相位。

**误码率**：
对于BPSK：
$$P_e = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$$

其中 $Q(x)$ 是Q函数，$E_b$ 是每比特能量，$N_0$ 是噪声功率谱密度。

---

*编写日期: 2025-08-01*  
*版本: 1.0.0* 