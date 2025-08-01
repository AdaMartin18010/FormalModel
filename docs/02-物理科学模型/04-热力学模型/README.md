# 2.4 热力学模型 / Thermodynamics Models

## 目录 / Table of Contents

- [2.4 热力学模型 / Thermodynamics Models](#24-热力学模型--thermodynamics-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [2.4.1 热力学定律 / Thermodynamic Laws](#241-热力学定律--thermodynamic-laws)
    - [热力学第零定律 / Zeroth Law](#热力学第零定律--zeroth-law)
    - [热力学第一定律 / First Law](#热力学第一定律--first-law)
    - [热力学第二定律 / Second Law](#热力学第二定律--second-law)
    - [热力学第三定律 / Third Law](#热力学第三定律--third-law)
  - [2.4.2 热力学势 / Thermodynamic Potentials](#242-热力学势--thermodynamic-potentials)
    - [内能 / Internal Energy](#内能--internal-energy)
    - [焓 / Enthalpy](#焓--enthalpy)
    - [亥姆霍兹自由能 / Helmholtz Free Energy](#亥姆霍兹自由能--helmholtz-free-energy)
    - [吉布斯自由能 / Gibbs Free Energy](#吉布斯自由能--gibbs-free-energy)
  - [2.4.3 统计力学 / Statistical Mechanics](#243-统计力学--statistical-mechanics)
    - [玻尔兹曼分布 / Boltzmann Distribution](#玻尔兹曼分布--boltzmann-distribution)
    - [配分函数 / Partition Function](#配分函数--partition-function)
    - [系综理论 / Ensemble Theory](#系综理论--ensemble-theory)
  - [2.4.4 相变理论 / Phase Transition Theory](#244-相变理论--phase-transition-theory)
    - [一级相变 / First-Order Phase Transitions](#一级相变--first-order-phase-transitions)
    - [二级相变 / Second-Order Phase Transitions](#二级相变--second-order-phase-transitions)
    - [临界现象 / Critical Phenomena](#临界现象--critical-phenomena)
  - [2.4.5 非平衡热力学 / Non-Equilibrium Thermodynamics](#245-非平衡热力学--non-equilibrium-thermodynamics)
    - [熵产生 / Entropy Production](#熵产生--entropy-production)
    - [昂萨格关系 / Onsager Relations](#昂萨格关系--onsager-relations)
    - [耗散结构 / Dissipative Structures](#耗散结构--dissipative-structures)
  - [2.4.6 量子统计 / Quantum Statistics](#246-量子统计--quantum-statistics)
    - [费米-狄拉克统计 / Fermi-Dirac Statistics](#费米-狄拉克统计--fermi-dirac-statistics)
    - [玻色-爱因斯坦统计 / Bose-Einstein Statistics](#玻色-爱因斯坦统计--bose-einstein-statistics)
    - [量子气体 / Quantum Gases](#量子气体--quantum-gases)
  - [2.4.7 热力学应用 / Thermodynamic Applications](#247-热力学应用--thermodynamic-applications)
    - [热机效率 / Heat Engine Efficiency](#热机效率--heat-engine-efficiency)
    - [制冷循环 / Refrigeration Cycles](#制冷循环--refrigeration-cycles)
    - [化学平衡 / Chemical Equilibrium](#化学平衡--chemical-equilibrium)
  - [参考文献 / References](#参考文献--references)

---

## 2.4.1 热力学定律 / Thermodynamic Laws

### 热力学第零定律 / Zeroth Law

**温度定义**: 如果两个系统都与第三个系统处于热平衡，则它们彼此也处于热平衡。

**温度测量**: 通过热平衡建立温度标度。

### 热力学第一定律 / First Law

**能量守恒**: 
$$\Delta U = Q - W$$

其中：
- $\Delta U$: 内能变化
- $Q$: 吸收的热量
- $W$: 对外做功

**微分形式**:
$$dU = \delta Q - \delta W$$

### 热力学第二定律 / Second Law

**克劳修斯表述**: 热量不能自发地从低温物体传递到高温物体。

**开尔文表述**: 不可能从单一热源吸收热量，使其完全变为功而不产生其他影响。

**熵增原理**:
$$\Delta S \geq 0$$

**卡诺效率**:
$$\eta = 1 - \frac{T_c}{T_h}$$

### 热力学第三定律 / Third Law

**能斯特定理**: 当温度趋近于绝对零度时，熵的变化趋近于零。

**绝对零度**: $T = 0$ K时，$S = 0$。

---

## 2.4.2 热力学势 / Thermodynamic Potentials

### 内能 / Internal Energy

**定义**: $U(S, V, N)$

**微分关系**:
$$dU = TdS - pdV + \mu dN$$

**麦克斯韦关系**:
$$\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial p}{\partial S}\right)_V$$

### 焓 / Enthalpy

**定义**: $H = U + pV$

**微分关系**:
$$dH = TdS + Vdp + \mu dN$$

**等压过程**: $\Delta H = Q_p$

### 亥姆霍兹自由能 / Helmholtz Free Energy

**定义**: $F = U - TS$

**微分关系**:
$$dF = -SdT - pdV + \mu dN$$

**等温过程**: $\Delta F = W_{rev}$

### 吉布斯自由能 / Gibbs Free Energy

**定义**: $G = H - TS$

**微分关系**:
$$dG = -SdT + Vdp + \mu dN$$

**化学势**: $\mu = \left(\frac{\partial G}{\partial N}\right)_{T,p}$

---

## 2.4.3 统计力学 / Statistical Mechanics

### 玻尔兹曼分布 / Boltzmann Distribution

**分布函数**:
$$P(E_i) = \frac{e^{-\beta E_i}}{Z}$$

其中：
- $\beta = \frac{1}{k_B T}$
- $Z$: 配分函数

**平均能量**:
$$\langle E \rangle = -\frac{\partial \ln Z}{\partial \beta}$$

### 配分函数 / Partition Function

**正则配分函数**:
$$Z = \sum_i e^{-\beta E_i}$$

**巨正则配分函数**:
$$\mathcal{Z} = \sum_{N,i} e^{-\beta(E_i - \mu N)}$$

**热力学量**:
- 内能: $U = -\frac{\partial \ln Z}{\partial \beta}$
- 熵: $S = k_B(\ln Z + \beta U)$
- 自由能: $F = -k_B T \ln Z$

### 系综理论 / Ensemble Theory

**微正则系综**: 固定$N, V, E$

**正则系综**: 固定$N, V, T$

**巨正则系综**: 固定$\mu, V, T$

---

## 2.4.4 相变理论 / Phase Transition Theory

### 一级相变 / First-Order Phase Transitions

**特征**: 体积和熵的突变

**克拉珀龙方程**:
$$\frac{dp}{dT} = \frac{\Delta S}{\Delta V}$$

**潜热**: $L = T\Delta S$

### 二级相变 / Second-Order Phase Transitions

**特征**: 比热容发散

**临界指数**:
- $\alpha$: 比热容指数
- $\beta$: 序参量指数
- $\gamma$: 磁化率指数
- $\delta$: 临界等温线指数

### 临界现象 / Critical Phenomena

**标度律**:
$$\xi \sim |T - T_c|^{-\nu}$$

**普适性**: 不同系统具有相同的临界指数。

**重整化群**: 研究临界现象的理论工具。

---

## 2.4.5 非平衡热力学 / Non-Equilibrium Thermodynamics

### 熵产生 / Entropy Production

**局部熵平衡**:
$$\frac{\partial s}{\partial t} + \nabla \cdot \vec{J}_s = \sigma$$

其中：
- $s$: 熵密度
- $\vec{J}_s$: 熵流
- $\sigma$: 熵产生率

### 昂萨格关系 / Onsager Relations

**线性响应**:
$$J_i = \sum_j L_{ij} X_j$$

**昂萨格关系**:
$$L_{ij} = L_{ji}$$

### 耗散结构 / Dissipative Structures

**远离平衡**: 系统在远离平衡态时出现的有序结构。

**自组织**: 系统自发形成的有序结构。

---

## 2.4.6 量子统计 / Quantum Statistics

### 费米-狄拉克统计 / Fermi-Dirac Statistics

**分布函数**:
$$f_{FD}(E) = \frac{1}{e^{\beta(E-\mu)} + 1}$$

**费米能级**: $E_F = \mu(T=0)$

**简并费米气体**: 低温下的量子效应。

### 玻色-爱因斯坦统计 / Bose-Einstein Statistics

**分布函数**:
$$f_{BE}(E) = \frac{1}{e^{\beta(E-\mu)} - 1}$$

**玻色-爱因斯坦凝聚**: 宏观量子态。

### 量子气体 / Quantum Gases

**理想费米气体**:
$$E = \frac{3}{5}N E_F$$

**理想玻色气体**:
$$N = \sum_i \frac{1}{e^{\beta(E_i-\mu)} - 1}$$

---

## 2.4.7 热力学应用 / Thermodynamic Applications

### 热机效率 / Heat Engine Efficiency

**卡诺循环**:
$$\eta = 1 - \frac{T_c}{T_h}$$

**斯特林循环**: 等温压缩和膨胀。

**奥托循环**: 内燃机循环。

### 制冷循环 / Refrigeration Cycles

**卡诺制冷机**:
$$\text{COP} = \frac{T_c}{T_h - T_c}$$

**蒸汽压缩循环**: 实际制冷系统。

### 化学平衡 / Chemical Equilibrium

**反应商**:
$$Q = \prod_i a_i^{\nu_i}$$

**平衡常数**:
$$K = \prod_i a_i^{\nu_i}$$

**范特霍夫方程**:
$$\frac{d\ln K}{dT} = \frac{\Delta H^\circ}{RT^2}$$

---

## 参考文献 / References

1. Callen, H. B. (1985). Thermodynamics and an Introduction to Thermostatistics. Wiley.
2. Pathria, R. K., & Beale, P. D. (2011). Statistical Mechanics. Academic Press.
3. Huang, K. (1987). Statistical Mechanics. Wiley.
4. Landau, L. D., & Lifshitz, E. M. (1980). Statistical Physics. Pergamon Press.
5. Prigogine, I. (1967). Introduction to Thermodynamics of Irreversible Processes. Interscience.

---

*最后更新: 2025-08-01*
*版本: 1.0.0* 