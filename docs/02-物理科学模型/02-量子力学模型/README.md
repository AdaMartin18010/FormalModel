# 2.2 量子力学模型 / Quantum Mechanics Models

## 目录 / Table of Contents

- [2.2 量子力学模型 / Quantum Mechanics Models](#22-量子力学模型--quantum-mechanics-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [2.2.1 量子力学基础 / Quantum Mechanics Fundamentals](#221-量子力学基础--quantum-mechanics-fundamentals)
    - [量子态 / Quantum States](#量子态--quantum-states)
    - [算符与可观测量 / Operators and Observables](#算符与可观测量--operators-and-observables)
    - [测量与坍缩 / Measurement and Collapse](#测量与坍缩--measurement-and-collapse)
  - [2.2.2 薛定谔方程 / Schrödinger Equation](#222-薛定谔方程--schrödinger-equation)
    - [时间相关薛定谔方程 / Time-dependent Schrödinger Equation](#时间相关薛定谔方程--time-dependent-schrödinger-equation)
    - [定态薛定谔方程 / Time-independent Schrödinger Equation](#定态薛定谔方程--time-independent-schrödinger-equation)
    - [本征值问题 / Eigenvalue Problems](#本征值问题--eigenvalue-problems)
  - [2.2.3 海森堡不确定性原理 / Heisenberg Uncertainty Principle](#223-海森堡不确定性原理--heisenberg-uncertainty-principle)
    - [位置-动量不确定性 / Position-Momentum Uncertainty](#位置-动量不确定性--position-momentum-uncertainty)
    - [时间-能量不确定性 / Time-Energy Uncertainty](#时间-能量不确定性--time-energy-uncertainty)
    - [一般不确定性关系 / General Uncertainty Relations](#一般不确定性关系--general-uncertainty-relations)
  - [2.2.4 量子叠加与纠缠 / Quantum Superposition and Entanglement](#224-量子叠加与纠缠--quantum-superposition-and-entanglement)
    - [量子叠加原理 / Quantum Superposition Principle](#量子叠加原理--quantum-superposition-principle)
    - [量子纠缠 / Quantum Entanglement](#量子纠缠--quantum-entanglement)
    - [贝尔不等式 / Bell Inequalities](#贝尔不等式--bell-inequalities)
  - [2.2.5 量子力学形式化 / Quantum Mechanics Formalization](#225-量子力学形式化--quantum-mechanics-formalization)
    - [希尔伯特空间 / Hilbert Space](#希尔伯特空间--hilbert-space)
    - [狄拉克符号 / Dirac Notation](#狄拉克符号--dirac-notation)
    - [密度矩阵 / Density Matrix](#密度矩阵--density-matrix)
  - [2.2.6 量子力学应用 / Quantum Mechanics Applications](#226-量子力学应用--quantum-mechanics-applications)
    - [原子物理 / Atomic Physics](#原子物理--atomic-physics)
    - [分子物理 / Molecular Physics](#分子物理--molecular-physics)
    - [量子化学 / Quantum Chemistry](#量子化学--quantum-chemistry)
    - [量子计算 / Quantum Computing](#量子计算--quantum-computing)
  - [2.2.7 量子场论 / Quantum Field Theory](#227-量子场论--quantum-field-theory)
    - [量子电动力学 / Quantum Electrodynamics](#量子电动力学--quantum-electrodynamics)
    - [量子色动力学 / Quantum Chromodynamics](#量子色动力学--quantum-chromodynamics)
    - [标准模型 / Standard Model](#标准模型--standard-model)
  - [参考文献 / References](#参考文献--references)

---

## 2.2.1 量子力学基础 / Quantum Mechanics Fundamentals

### 量子态 / Quantum States

**量子态** 是描述量子系统完整信息的数学对象，通常用希尔伯特空间中的向量表示：

$$|\psi\rangle = \sum_n c_n |n\rangle$$

其中：

- $|\psi\rangle$: 量子态向量
- $c_n$: 复数系数
- $|n\rangle$: 正交基向量

**归一化条件**:
$$\langle\psi|\psi\rangle = \sum_n |c_n|^2 = 1$$

### 算符与可观测量 / Operators and Observables

**厄米算符** 对应物理可观测量：
$$\hat{A} = \hat{A}^\dagger$$

**本征值方程**:
$$\hat{A}|\psi_n\rangle = a_n|\psi_n\rangle$$

其中：

- $\hat{A}$: 厄米算符
- $a_n$: 本征值
- $|\psi_n\rangle$: 本征态

### 测量与坍缩 / Measurement and Collapse

**测量公设**: 测量可观测量 $\hat{A}$ 得到本征值 $a_n$ 的概率为：
$$P(a_n) = |\langle\psi_n|\psi\rangle|^2$$

**测量后坍缩**: 测量后系统坍缩到对应的本征态：
$$|\psi\rangle \xrightarrow{\text{measurement}} |\psi_n\rangle$$

---

## 2.2.2 薛定谔方程 / Schrödinger Equation

### 时间相关薛定谔方程 / Time-dependent Schrödinger Equation

**一般形式**:
$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

其中：

- $\hbar$: 约化普朗克常数
- $\hat{H}$: 哈密顿算符
- $|\psi(t)\rangle$: 时间相关的量子态

**一维势场中的形式**:
$$i\hbar\frac{\partial}{\partial t}\psi(x,t) = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)$$

### 定态薛定谔方程 / Time-independent Schrödinger Equation

**定态解**:
$$\psi(x,t) = \psi(x)e^{-iEt/\hbar}$$

**定态方程**:
$$\hat{H}\psi(x) = E\psi(x)$$

**一维形式**:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi(x) = E\psi(x)$$

### 本征值问题 / Eigenvalue Problems

**无限深势阱**:
$$
V(x) = \begin{cases}
0, & 0 < x < a \\
\infty, & \text{otherwise}
\end{cases}
$$

**本征函数**:
$$\psi_n(x) = \sqrt{\frac{2}{a}}\sin\left(\frac{n\pi x}{a}\right)$$

**本征值**:
$$E_n = \frac{n^2\pi^2\hbar^2}{2ma^2}$$

---

## 2.2.3 海森堡不确定性原理 / Heisenberg Uncertainty Principle

### 位置-动量不确定性 / Position-Momentum Uncertainty

**不确定性关系**:
$$\Delta x \Delta p \geq \frac{\hbar}{2}$$

其中：

- $\Delta x$: 位置的标准差
- $\Delta p$: 动量的标准差

**高斯波包的最小不确定性**:
$$\Delta x \Delta p = \frac{\hbar}{2}$$

### 时间-能量不确定性 / Time-Energy Uncertainty

**时间-能量不确定性**:
$$\Delta t \Delta E \geq \frac{\hbar}{2}$$

**应用**: 解释粒子寿命与能量宽度的关系。

### 一般不确定性关系 / General Uncertainty Relations

**任意两个可观测量**:
$$\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$$

其中 $[\hat{A},\hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$ 是交换子。

---

## 2.2.4 量子叠加与纠缠 / Quantum Superposition and Entanglement

### 量子叠加原理 / Quantum Superposition Principle

**叠加态**:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

其中 $|\alpha|^2 + |\beta|^2 = 1$。

**双缝实验**: 粒子同时通过两个狭缝，产生干涉图案。

### 量子纠缠 / Quantum Entanglement

**贝尔态**:
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

**纠缠度量**: 冯·诺依曼熵
$$S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A)$$

### 贝尔不等式 / Bell Inequalities

**CHSH不等式**:
$$|E(a,b) - E(a,b') + E(a',b) + E(a',b')| \leq 2$$

其中 $E(a,b)$ 是关联函数。

**量子力学违反**: 最大违反为 $2\sqrt{2}$。

---

## 2.2.5 量子力学形式化 / Quantum Mechanics Formalization

### 希尔伯特空间 / Hilbert Space

**定义**: 完备的内积空间。

**内积**: $\langle\psi|\phi\rangle$ 满足：

- 线性性: $\langle\psi|a\phi_1 + b\phi_2\rangle = a\langle\psi|\phi_1\rangle + b\langle\psi|\phi_2\rangle$
- 共轭对称性: $\langle\psi|\phi\rangle = \langle\phi|\psi\rangle^*$
- 正定性: $\langle\psi|\psi\rangle \geq 0$

### 狄拉克符号 / Dirac Notation

**右矢 (Ket)**: $|\psi\rangle$
**左矢 (Bra)**: $\langle\psi|$
**内积**: $\langle\psi|\phi\rangle$
**外积**: $|\psi\rangle\langle\phi|$

**投影算符**:
$$\hat{P}_n = |n\rangle\langle n|$$

### 密度矩阵 / Density Matrix

**纯态密度矩阵**:
$$\rho = |\psi\rangle\langle\psi|$$

**混合态密度矩阵**:
$$\rho = \sum_i p_i|\psi_i\rangle\langle\psi_i|$$

其中 $\sum_i p_i = 1$。

**冯·诺依曼方程**:
$$i\hbar\frac{d\rho}{dt} = [\hat{H},\rho]$$

---

## 2.2.6 量子力学应用 / Quantum Mechanics Applications

### 原子物理 / Atomic Physics

**氢原子**:
$$\hat{H} = -\frac{\hbar^2}{2\mu}\nabla^2 - \frac{e^2}{4\pi\epsilon_0 r}$$

**本征值**:
$$E_n = -\frac{13.6\text{ eV}}{n^2}$$

**波函数**:
$$\psi_{nlm}(r,\theta,\phi) = R_{nl}(r)Y_l^m(\theta,\phi)$$

### 分子物理 / Molecular Physics

**玻恩-奥本海默近似**:
$$\hat{H} = \hat{T}_e + \hat{T}_N + V_{ee} + V_{NN} + V_{eN}$$

**分子轨道理论**: 线性组合原子轨道 (LCAO)。

### 量子化学 / Quantum Chemistry

**哈特里-福克方法**:
$$\hat{F}\psi_i = \epsilon_i\psi_i$$

其中 $\hat{F}$ 是福克算符。

**密度泛函理论 (DFT)**:
$$E[\rho] = T[\rho] + V_{ext}[\rho] + J[\rho] + E_{xc}[\rho]$$

### 量子计算 / Quantum Computing

**量子比特**:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

**量子门**:

- **Hadamard门**: $H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$
- **Pauli门**: $X = \begin{pmatrix}0&1\\1&0\end{pmatrix}$, $Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$

**量子算法**:

- **Shor算法**: 大数分解
- **Grover算法**: 量子搜索

---

## 2.2.7 量子场论 / Quantum Field Theory

### 量子电动力学 / Quantum Electrodynamics

**拉格朗日密度**:
$$\mathcal{L} = \bar{\psi}(i\gamma^\mu\partial_\mu - m)\psi - \frac{1}{4}F_{\mu\nu}F^{\mu\nu} - e\bar{\psi}\gamma^\mu\psi A_\mu$$

**费曼图**: 描述粒子相互作用。

### 量子色动力学 / Quantum Chromodynamics

**拉格朗日密度**:
$$\mathcal{L} = \bar{\psi}(i\gamma^\mu D_\mu - m)\psi - \frac{1}{4}G^a_{\mu\nu}G^{a\mu\nu}$$

其中 $D_\mu = \partial_\mu + igA_\mu^a T^a$。

### 标准模型 / Standard Model

**规范群**: $SU(3)_C \times SU(2)_L \times U(1)_Y$

**希格斯机制**: 自发对称性破缺。

**粒子分类**:

- **费米子**: 夸克、轻子
- **玻色子**: 光子、W/Z玻色子、胶子、希格斯玻色子

---

## 参考文献 / References

1. Sakurai, J. J. (1994). Modern Quantum Mechanics. Addison-Wesley.
2. Griffiths, D. J. (2005). Introduction to Quantum Mechanics. Pearson.
3. Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information. Cambridge University Press.
4. Peskin, M. E., & Schroeder, D. V. (1995). An Introduction to Quantum Field Theory. Addison-Wesley.
5. Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox. Physics Physique Физика.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
