# 第七章：社会科学模型 / Chapter 7: Social Science Models

## 7.1 社会网络模型 / Social Network Models

### 7.1.1 网络结构模型 / Network Structure Model

社会网络是社会科学中最重要的模型之一，它通过图论的方法来描述和分析社会关系。网络结构模型为我们提供了理解复杂社会系统的数学工具。

#### 网络表示方法

**图论基础**：
社会网络可以用图 $G = (V, E)$ 来表示，其中 $V$ 是节点集合（代表个体），$E$ 是边集合（代表关系）。对于无向网络，邻接矩阵 $A$ 定义为：

$$A_{ij} = \begin{cases}
1 & \text{如果节点 } i \text{ 和 } j \text{ 之间有连接} \\
0 & \text{否则}
\end{cases}$$

**网络拓扑特征**：
- **度分布**：节点度数的分布函数 $P(k)$，描述网络中不同连接数的节点分布
- **聚类系数**：节点 $i$ 的聚类系数定义为：
  $$C_i = \frac{2E_i}{k_i(k_i-1)}$$
  其中 $E_i$ 是节点 $i$ 的邻居之间的边数，$k_i$ 是节点 $i$ 的度数
- **平均路径长度**：网络中任意两节点间最短路径的平均长度
- **网络直径**：网络中任意两节点间最短路径的最大长度

#### 网络类型

**随机网络（Erdős-Rényi模型）**：
在 $n$ 个节点的网络中，每条边以概率 $p$ 独立存在。度分布近似为泊松分布：
$$P(k) = \frac{\langle k \rangle^k e^{-\langle k \rangle}}{k!}$$

**小世界网络**：
具有高聚类系数和短平均路径长度的网络。Watts-Strogatz模型通过重连规则生成：
1. 从规则环开始
2. 以概率 $p$ 重连每条边
3. 当 $p$ 很小时，保持高聚类系数但减少平均路径长度

**无标度网络**：
度分布遵循幂律的网络：$P(k) \sim k^{-\gamma}$。Barabási-Albert模型通过优先连接生成：
- 新节点以概率 $\Pi(k_i) = \frac{k_i}{\sum_j k_j}$ 连接到现有节点
- 度分布为 $P(k) \sim k^{-3}$

### 7.1.2 传播模型 / Diffusion Model

传播模型描述了信息、疾病或影响在网络中的扩散过程。

#### 传染病传播模型

**SIR模型**：
将人群分为三类：
- 易感者（Susceptible）：$S(t)$
- 感染者（Infected）：$I(t)$
- 恢复者（Recovered）：$R(t)$

动力学方程：
$$\frac{dS}{dt} = -\beta SI$$
$$\frac{dI}{dt} = \beta SI - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

其中 $\beta$ 是传播率，$\gamma$ 是恢复率。基本再生数 $R_0 = \frac{\beta}{\gamma}$ 决定了传播是否会发生。

**SIS模型**：
感染者可以重新变为易感者：
$$\frac{dS}{dt} = -\beta SI + \gamma I$$
$$\frac{dI}{dt} = \beta SI - \gamma I$$

**传播阈值**：
在异质网络中，传播阈值取决于度分布。对于无标度网络，当 $\gamma > 2$ 时，传播阈值为零，意味着即使很小的传播率也能导致大规模传播。

#### 信息传播模型

**线性阈值模型**：
节点 $i$ 的激活条件：
$$\sum_{j \in N_i} w_{ji} \geq \theta_i$$
其中 $N_i$ 是节点 $i$ 的邻居，$w_{ji}$ 是影响权重，$\theta_i$ 是阈值。

**独立级联模型**：
每条边 $(i,j)$ 有传播概率 $p_{ij}$。如果节点 $i$ 被激活，它以概率 $p_{ij}$ 激活邻居 $j$。

### 7.1.3 社区发现模型 / Community Detection Model

社区发现旨在识别网络中的紧密连接群体。

#### 模块度优化

**模块度定义**：
$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

其中 $m$ 是总边数，$c_i$ 是节点 $i$ 的社区标签，$\delta$ 是克罗内克函数。

**Louvain算法**：
1. 每个节点初始化为独立社区
2. 对每个节点，计算将其移动到邻居社区的模块度增益
3. 选择最大增益的移动
4. 重复直到收敛
5. 将社区收缩为超级节点，重复过程

#### 谱聚类方法

基于拉普拉斯矩阵的谱聚类：
1. 计算拉普拉斯矩阵 $L = D - A$
2. 计算前 $k$ 个最小特征值对应的特征向量
3. 对特征向量进行聚类

### 7.1.4 网络动力学模型 / Network Dynamics Model

#### 同步模型

**Kuramoto模型**：
描述耦合振子的相位同步：
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)$$

其中 $\theta_i$ 是振子 $i$ 的相位，$\omega_i$ 是自然频率，$K$ 是耦合强度。

**同步阶参数**：
$$r = \left|\frac{1}{N} \sum_{j=1}^N e^{i\theta_j}\right|$$

当 $r = 1$ 时完全同步，当 $r = 0$ 时完全异步。

#### 博弈论模型

**囚徒困境**：
在网络中，每个节点可以选择合作（C）或背叛（D）。收益矩阵：
$$\begin{pmatrix}
R & S \\
T & P
\end{pmatrix}$$

其中 $T > R > P > S$，且 $2R > T + S$。

**演化博弈**：
策略更新规则：
$$p_i^{t+1} = \frac{\sum_{j \in N_i} A_{ij} p_j^t \pi_j^t}{\sum_{j \in N_i} A_{ij} \pi_j^t}$$

其中 $p_i^t$ 是节点 $i$ 在时间 $t$ 采用策略的概率，$\pi_i^t$ 是适应度。

#### 意见动力学

**Deffuant模型**：
连续意见的演化：
$$\theta_i(t+1) = \theta_i(t) + \mu [\theta_j(t) - \theta_i(t)]$$
$$\theta_j(t+1) = \theta_j(t) + \mu [\theta_i(t) - \theta_j(t)]$$

其中 $\mu \in [0, 0.5]$ 是收敛参数，只有当 $|\theta_i - \theta_j| < \epsilon$ 时才发生交互。

**Voter模型**：
离散意见的演化：
$$P(\sigma_i \rightarrow \sigma_j) = \frac{1}{k_i} \sum_{j \in N_i} \delta(\sigma_i, \sigma_j)$$

其中 $\sigma_i$ 是节点 $i$ 的意见状态。

## 7.2 经济学模型 / Economics Models

### 7.2.1 微观经济学模型 / Microeconomics Model

微观经济学模型关注个体经济行为和市场机制。

#### 消费者理论

**效用函数**：
消费者效用函数 $U(x_1, x_2, ..., x_n)$ 满足：
- 单调性：$\frac{\partial U}{\partial x_i} > 0$
- 凸性：效用函数的二阶导数为负

**预算约束**：
$$\sum_{i=1}^n p_i x_i \leq I$$
其中 $p_i$ 是商品 $i$ 的价格，$I$ 是收入。

**需求函数**：
通过效用最大化得到的需求函数：
$$x_i^* = x_i(p_1, p_2, ..., p_n, I)$$

**消费者剩余**：
$$CS = \int_0^{q^*} [p(q) - p^*] dq$$
其中 $p(q)$ 是需求函数，$p^*$ 是市场价格。

#### 生产者理论

**生产函数**：
$$Q = f(K, L)$$
其中 $K$ 是资本，$L$ 是劳动。

**成本函数**：
$$C(Q) = \min_{K,L} \{rK + wL : f(K,L) = Q\}$$
其中 $r$ 是资本价格，$w$ 是工资率。

**供给函数**：
通过利润最大化得到的供给函数：
$$Q^s = Q^s(p, w, r)$$

#### 市场均衡

**供需平衡**：
$$Q^d(p) = Q^s(p)$$
解出均衡价格 $p^*$ 和均衡数量 $Q^*$。

**市场效率**：
帕累托最优条件：
$$\frac{MU_1}{p_1} = \frac{MU_2}{p_2} = ... = \frac{MU_n}{p_n}$$

### 7.2.2 宏观经济学模型 / Macroeconomics Model

#### 经济增长模型

**Solow增长模型**：
生产函数：$Y = K^\alpha (AL)^{1-\alpha}$
资本积累：$\dot{K} = sY - \delta K$
技术增长：$\dot{A} = gA$
人口增长：$\dot{L} = nL$

稳态条件：
$$\frac{K}{AL} = \left(\frac{s}{n + g + \delta}\right)^{\frac{1}{1-\alpha}}$$

**内生增长模型**：
AK模型：$Y = AK$
资本积累：$\dot{K} = sY - \delta K = sAK - \delta K$
增长率：$g = sA - \delta$

#### 经济周期模型

**IS-LM模型**：
IS曲线（商品市场均衡）：
$$Y = C(Y-T) + I(r) + G$$

LM曲线（货币市场均衡）：
$$\frac{M}{P} = L(r, Y)$$

**AD-AS模型**：
总需求曲线：
$$Y = C(Y-T) + I(r) + G + NX$$

总供给曲线：
$$Y = Y^* + \alpha(P - P^e)$$

### 7.2.3 金融经济学模型 / Financial Economics Model

#### 资产定价模型

**CAPM模型**：
期望收益：
$$E[R_i] = R_f + \beta_i(E[R_m] - R_f)$$

其中 $\beta_i = \frac{Cov(R_i, R_m)}{Var(R_m)}$ 是系统性风险。

**Black-Scholes期权定价**：
$$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$

其中：
$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

#### 投资组合理论

**Markowitz模型**：
投资组合方差：
$$\sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}$$

最优权重：
$$w_i = \frac{\sum_{j=1}^n \sigma^{ij}(\mu_j - R_f)}{\sum_{k=1}^n \sum_{l=1}^n \sigma^{kl}(\mu_k - R_f)}$$

其中 $\sigma^{ij}$ 是协方差矩阵的逆矩阵元素。

### 7.2.4 计量经济学模型 / Econometrics Model

#### 回归分析

**线性回归模型**：
$$Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + ... + \beta_k X_{ki} + \epsilon_i$$

OLS估计：
$$\hat{\beta} = (X'X)^{-1}X'y$$

**回归诊断**：
- 残差分析：$\epsilon_i = Y_i - \hat{Y}_i$
- 多重共线性：$VIF_j = \frac{1}{1-R_j^2}$
- 异方差性检验：White检验、Breusch-Pagan检验

#### 时间序列分析

**ARIMA模型**：
$$(1-\phi_1 B - ... - \phi_p B^p)(1-B)^d Y_t = (1-\theta_1 B - ... - \theta_q B^q)\epsilon_t$$

其中 $B$ 是滞后算子，$d$ 是差分次数。

**VAR模型**：
$$Y_t = c + \Phi_1 Y_{t-1} + \Phi_2 Y_{t-2} + ... + \Phi_p Y_{t-p} + \epsilon_t$$

其中 $Y_t$ 是向量，$\Phi_i$ 是系数矩阵。

## 7.3 心理学模型 / Psychology Models

### 7.3.1 认知心理学模型 / Cognitive Psychology Model

#### 信息加工模型

**Atkinson-Shiffrin模型**：
记忆系统的三个组成部分：
1. 感觉记忆：容量大，持续时间短
2. 短时记忆：容量有限（7±2），持续时间约30秒
3. 长时记忆：容量无限，持续时间长

**工作记忆模型（Baddeley）**：
- 中央执行系统：控制注意和认知资源
- 语音环路：处理语音信息
- 视觉空间画板：处理视觉空间信息
- 情节缓冲器：整合不同来源的信息

#### 学习理论

**经典条件反射**：
$$CS + US \rightarrow UR$$
$$CS \rightarrow CR$$

其中CS是条件刺激，US是无条件刺激，UR是无条件反应，CR是条件反应。

**操作性条件反射**：
强化概率：
$$P(R) = \frac{1}{1 + e^{-(\alpha + \beta X)}}$$

其中 $\alpha$ 是基线概率，$\beta$ 是强化效应，$X$ 是强化历史。

#### 记忆模型

**多重存储模型**：
记忆强度：
$$S(t) = S_0 e^{-\lambda t}$$

其中 $S_0$ 是初始强度，$\lambda$ 是遗忘率。

**工作记忆容量**：
$$C = \frac{1}{N} \sum_{i=1}^N \frac{1}{d_i}$$

其中 $d_i$ 是项目 $i$ 的难度。

### 7.3.2 发展心理学模型 / Developmental Psychology Model

#### 认知发展理论

**Piaget阶段理论**：
1. 感知运动期（0-2岁）：对象永久性
2. 前运算期（2-7岁）：符号思维
3. 具体运算期（7-11岁）：逻辑思维
4. 形式运算期（11岁以上）：抽象思维

**Vygotsky社会文化理论**：
最近发展区：
$$ZPD = ZAR - ZAR'$$

其中ZAR是实际发展水平，ZAR'是潜在发展水平。

#### 社会发展理论

**Erikson心理社会发展**：
八个发展阶段，每个阶段都有特定的心理社会危机：
1. 信任 vs 不信任（0-1岁）
2. 自主 vs 羞怯（1-3岁）
3. 主动 vs 内疚（3-6岁）
4. 勤奋 vs 自卑（6-12岁）
5. 同一性 vs 角色混乱（12-18岁）
6. 亲密 vs 孤独（18-40岁）
7. 繁衍 vs 停滞（40-65岁）
8. 完整 vs 绝望（65岁以上）

### 7.3.3 社会心理学模型 / Social Psychology Model

#### 态度模型

**态度结构**：
态度 $A$ 由认知 $C$、情感 $E$、行为 $B$ 组成：
$$A = w_1 C + w_2 E + w_3 B$$

其中 $w_i$ 是权重。

**态度改变**：
说服概率：
$$P(说服) = \frac{1}{1 + e^{-(\alpha + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n)}}$$

其中 $X_i$ 是说服因素。

#### 群体行为模型

**从众行为**：
从众概率：
$$P(从众) = f(群体规模, 群体一致性, 任务难度, 个体特征)$$

**群体极化**：
群体决策倾向：
$$D_g = D_i + \alpha \sum_{j \neq i} (D_j - D_i)$$

其中 $D_i$ 是个人初始倾向，$\alpha$ 是影响系数。

### 7.3.4 临床心理学模型 / Clinical Psychology Model

#### 心理障碍模型

**生物医学模型**：
心理障碍的生物学基础：
- 神经递质失衡
- 脑结构异常
- 遗传因素

**认知行为模型**：
认知三角：
- 自动思维
- 核心信念
- 中间信念

#### 治疗模型

**认知行为治疗**：
认知重构过程：
1. 识别自动思维
2. 评估思维合理性
3. 生成替代思维
4. 行为实验验证

**治疗效果评估**：
治疗效果：
$$Effect = \frac{X_{post} - X_{pre}}{SD_{pre}}$$

其中 $X$ 是测量分数，$SD$ 是标准差。

## 7.4 认知科学模型 / Cognitive Science Models

### 7.4.1 认知架构模型 / Cognitive Architecture Model

#### ACT-R模型

**模块化结构**：
- 感知模块：处理视觉、听觉信息
- 运动模块：控制身体动作
- 记忆模块：存储和检索信息
- 目标模块：管理当前目标

**产生式系统**：
产生式规则：
$$IF \text{ 条件 } THEN \text{ 动作 }$$

匹配过程：
$$P(i) = \frac{A_i}{\sum_j A_j}$$

其中 $A_i$ 是激活强度。

#### SOAR模型

**问题空间**：
状态表示：$S = \{s_1, s_2, ..., s_n\}$
算子：$O = \{o_1, o_2, ..., o_m\}$
目标：$G = \{g_1, g_2, ..., g_k\}$

**决策过程**：
偏好函数：
$$P(a) = \sum_i w_i f_i(a)$$

其中 $w_i$ 是权重，$f_i$ 是评价函数。

### 7.4.2 感知模型 / Perception Model

#### 视觉感知

**特征检测**：
Gabor滤波器：
$$G(x,y) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(\frac{2\pi x'}{\lambda}\right)$$

其中 $x' = x\cos\theta + y\sin\theta$，$y' = -x\sin\theta + y\cos\theta$。

**模式识别**：
模板匹配：
$$S = \sum_{i,j} T(i,j) I(i,j)$$

其中 $T$ 是模板，$I$ 是输入图像。

#### 听觉感知

**声音定位**：
双耳时间差：
$$\Delta t = \frac{d \sin \theta}{c}$$

其中 $d$ 是双耳距离，$\theta$ 是声源角度，$c$ 是声速。

**语音识别**：
隐马尔可夫模型：
$$P(O|λ) = \sum_Q P(O,Q|λ)$$

其中 $O$ 是观察序列，$Q$ 是状态序列，$λ$ 是模型参数。

### 7.4.3 语言认知模型 / Language Cognition Model

#### 语言理解

**词汇识别**：
激活扩散模型：
$$A_i(t+1) = A_i(t) + \sum_j w_{ij} A_j(t) - \alpha A_i(t)$$

其中 $A_i$ 是词汇 $i$ 的激活，$w_{ij}$ 是连接权重。

**句法分析**：
概率上下文无关文法：
$$P(T) = \prod_{r \in T} P(r)$$

其中 $T$ 是句法树，$r$ 是语法规则。

#### 语言产生

**概念化**：
概念激活：
$$A(c) = \sum_i w_i f_i(c)$$

其中 $f_i$ 是概念特征，$w_i$ 是特征权重。

**公式化**：
词汇选择：
$$P(w|c) = \frac{\exp(\beta \cos(w,c))}{\sum_{w'} \exp(\beta \cos(w',c))}$$

其中 $\cos(w,c)$ 是词汇和概念的相似度。

### 7.4.4 决策认知模型 / Decision Cognition Model

#### 理性决策模型

**期望效用理论**：
期望效用：
$$EU(A) = \sum_i p_i u(x_i)$$

其中 $p_i$ 是概率，$u(x_i)$ 是效用。

**主观期望效用**：
主观概率：
$$P(A) = \frac{1}{1 + e^{-\alpha \log \frac{O(A)}{O(\neg A)}}}$$

其中 $O(A)$ 是事件 $A$ 的赔率。

#### 启发式决策

**可用性启发式**：
判断概率：
$$P(A) \propto \text{记忆可得性}(A)$$

**代表性启发式**：
相似性判断：
$$P(A|B) \propto \text{相似性}(A,B)$$

**锚定启发式**：
调整过程：
$$估计值 = 锚定值 + \alpha \times 调整$$

其中 $\alpha$ 是调整系数，通常小于1。

## 7.5 语言学模型 / Linguistics Models

### 7.5.1 语音学模型 / Phonetics Model

#### 发音语音学

**发音器官模型**：
声道传输函数：
$$H(f) = \prod_{i=1}^n \frac{1 - z_i e^{-j2\pi f}}{1 - p_i e^{-j2\pi f}}$$

其中 $z_i$ 是零点，$p_i$ 是极点。

**音素识别**：
声学特征向量：
$$\mathbf{x} = [f_1, f_2, f_3, ..., f_n]^T$$

其中 $f_i$ 是第 $i$ 个声学特征。

#### 声学语音学

**频谱分析**：
短时傅里叶变换：
$$X(k) = \sum_{n=0}^{N-1} x(n) e^{-j2\pi kn/N}$$

**共振峰提取**：
线性预测系数：
$$x(n) = \sum_{k=1}^p a_k x(n-k) + e(n)$$

其中 $a_k$ 是预测系数，$e(n)$ 是预测误差。

### 7.5.2 形态学模型 / Morphology Model

#### 词素分析

**词素识别**：
词素概率：
$$P(m|w) = \frac{P(w|m) P(m)}{P(w)}$$

其中 $m$ 是词素，$w$ 是词。

**形态规则**：
规则应用概率：
$$P(r|w) = \frac{\exp(\beta \text{匹配度}(r,w))}{\sum_{r'} \exp(\beta \text{匹配度}(r',w))}$$

其中 $\beta$ 是温度参数。

#### 构词法

**派生规则**：
新词概率：
$$P(w_{new}) = P(w_{base}) \times P(affix|w_{base})$$

**复合规则**：
复合词概率：
$$P(w_{comp}) = P(w_1) \times P(w_2|w_1) \times P(\text{复合规则})$$

### 7.5.3 句法学模型 / Syntax Model

#### 句法结构

**短语结构语法**：
句法树概率：
$$P(T) = \prod_{r \in T} P(r)$$

其中 $r$ 是语法规则。

**依存语法**：
依存关系概率：
$$P(dep|head) = \frac{\exp(\mathbf{w}^T \phi(head, dep))}{\sum_{dep'} \exp(\mathbf{w}^T \phi(head, dep'))}$$

其中 $\phi$ 是特征函数，$\mathbf{w}$ 是权重向量。

#### 句法分析

**概率上下文无关文法**：
句法分析概率：
$$P(T|S) = \frac{P(S|T) P(T)}{P(S)}$$

其中 $S$ 是句子，$T$ 是句法树。

**句法消歧**：
最优句法树：
$$T^* = \arg\max_T P(T|S)$$

### 7.5.4 语义学模型 / Semantics Model

#### 词汇语义

**词义表示**：
词向量：
$$\mathbf{v}_w = \frac{1}{|C_w|} \sum_{c \in C_w} \mathbf{v}_c$$

其中 $C_w$ 是词 $w$ 的上下文集合。

**语义相似度**：
余弦相似度：
$$\text{sim}(w_1, w_2) = \frac{\mathbf{v}_{w_1} \cdot \mathbf{v}_{w_2}}{|\mathbf{v}_{w_1}| |\mathbf{v}_{w_2}|}$$

#### 句法语义

**语义角色标注**：
角色概率：
$$P(r|w, s) = \frac{\exp(\mathbf{w}_r^T \phi(w, s))}{\sum_{r'} \exp(\mathbf{w}_{r'}^T \phi(w, s))}$$

其中 $r$ 是语义角色，$w$ 是词，$s$ 是句子。

**语义框架**：
框架激活：
$$P(f|w) = \frac{\exp(\beta \text{相似度}(w, f))}{\sum_{f'} \exp(\beta \text{相似度}(w, f'))}$$

#### 语用语义

**指称解析**：
指称概率：
$$P(ref|ante) = \frac{\exp(\mathbf{w}^T \phi(ref, ante))}{\sum_{ref'} \exp(\mathbf{w}^T \phi(ref', ante))}$$

其中 $ref$ 是指称，$ante$ 是先行词。

**会话含义**：
含义推理：
$$P(impl|utt) = \sum_{ctx} P(impl|utt, ctx) P(ctx|utt)$$

其中 $impl$ 是隐含含义，$utt$ 是话语，$ctx$ 是上下文。

---

*编写日期: 2025-08-01*  
*版本: 1.0.0*
