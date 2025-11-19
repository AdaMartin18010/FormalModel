# 2.2 量子力学模型 / Quantum Mechanics Models

## 目录 / Table of Contents

- [2.2 量子力学模型 / Quantum Mechanics Models](#22-量子力学模型--quantum-mechanics-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [2.2.1 量子力学基础 / Quantum Mechanics Fundamentals](#221-量子力学基础--quantum-mechanics-fundamentals)
    - [量子力学基础框架图 / Framework Diagram of Quantum Mechanics Fundamentals](#量子力学基础框架图--framework-diagram-of-quantum-mechanics-fundamentals)
    - [量子态演化流程图 / Flowchart of Quantum State Evolution](#量子态演化流程图--flowchart-of-quantum-state-evolution)
    - [量子态 / Quantum States](#量子态--quantum-states)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition)
      - [等价定义 / Equivalent Definitions](#等价定义--equivalent-definitions)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation)
    - [算符与可观测量 / Operators and Observables](#算符与可观测量--operators-and-observables)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-1)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-1)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-1)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-1)
    - [测量与坍缩 / Measurement and Collapse](#测量与坍缩--measurement-and-collapse)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-2)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-2)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-2)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-2)
  - [2.2.2 薛定谔方程 / Schrödinger Equation](#222-薛定谔方程--schrödinger-equation)
    - [薛定谔方程框架图 / Framework Diagram of Schrödinger Equation](#薛定谔方程框架图--framework-diagram-of-schrödinger-equation)
    - [薛定谔方程求解流程图 / Flowchart of Schrödinger Equation Solution](#薛定谔方程求解流程图--flowchart-of-schrödinger-equation-solution)
    - [时间相关薛定谔方程 / Time-dependent Schrödinger Equation](#时间相关薛定谔方程--time-dependent-schrödinger-equation)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-3)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-3)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-3)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-3)
    - [定态薛定谔方程 / Time-independent Schrödinger Equation](#定态薛定谔方程--time-independent-schrödinger-equation)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-4)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-4)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-4)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-4)
    - [本征值问题 / Eigenvalue Problems](#本征值问题--eigenvalue-problems)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-5)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-5)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-5)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-5)
  - [2.2.3 海森堡不确定性原理 / Heisenberg Uncertainty Principle](#223-海森堡不确定性原理--heisenberg-uncertainty-principle)
    - [位置-动量不确定性 / Position-Momentum Uncertainty](#位置-动量不确定性--position-momentum-uncertainty)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-6)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-6)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-6)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-6)
    - [时间-能量不确定性 / Time-Energy Uncertainty](#时间-能量不确定性--time-energy-uncertainty)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-7)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-7)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-7)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-7)
    - [一般不确定性关系 / General Uncertainty Relations](#一般不确定性关系--general-uncertainty-relations)
      - [形式化定义 / Formal Definition](#形式化定义--formal-definition-8)
      - [公理化定义 / Axiomatic Definition](#公理化定义--axiomatic-definition-8)
      - [形式化定理 / Formal Theorems](#形式化定理--formal-theorems-8)
      - [算法实现 / Algorithmic Implementation](#算法实现--algorithmic-implementation-8)
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
  - [2.2.8 实现与应用 / Implementation and Applications](#228-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [Julia实现示例 / Julia Implementation Example](#julia实现示例--julia-implementation-example)
  - [相关模型 / Related Models](#相关模型--related-models)
    - [物理科学模型 / Physical Science Models](#物理科学模型--physical-science-models)
    - [基础理论 / Basic Theory](#基础理论--basic-theory)
  - [参考文献 / References](#参考文献--references)

---

## 2.2.1 量子力学基础 / Quantum Mechanics Fundamentals

### 量子力学基础框架图 / Framework Diagram of Quantum Mechanics Fundamentals

```mermaid
graph TB
    A[量子力学基础] --> B[量子态]
    A --> C[算符与可观测量]
    A --> D[测量与坍缩]

    B --> E[希尔伯特空间]
    B --> F[态向量 |ψ⟩]
    B --> G[叠加原理]

    C --> H[厄米算符]
    C --> I[本征值问题]
    C --> J[对易关系]

    D --> K[测量公设]
    D --> L[波函数坍缩]
    D --> M[概率解释]

    E --> N[薛定谔方程]
    H --> N
    K --> N

    N --> O[量子动力学]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#fff4e1
    style N fill:#e8f5e9
    style O fill:#e8f5e9
```

### 量子态演化流程图 / Flowchart of Quantum State Evolution

```mermaid
flowchart TD
    Start([开始]) --> InitState[初始化量子态 |ψ0⟩]
    InitState --> ConstructH[构造哈密顿算符 H]
    ConstructH --> SolveSE[求解薛定谔方程]
    SolveSE --> TimeEvolve[时间演化 |ψt⟩ = U|ψ0⟩]
    TimeEvolve --> Measure{进行测量?}
    Measure -->|否| ContinueEvolve[继续演化]
    Measure -->|是| ChooseObs[选择可观测量 O]
    ChooseObs --> CalcProb[计算概率分布]
    CalcProb --> Collapse[波函数坍缩]
    Collapse --> NewState[新量子态 |ψ'⟩]
    NewState --> Measure
    ContinueEvolve --> TimeEvolve
    Measure --> End([结束])

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style ConstructH fill:#fff4e1
    style SolveSE fill:#fff4e1
    style TimeEvolve fill:#e8f5e9
    style Collapse fill:#ffe1f5
```

### 量子态 / Quantum States

#### 形式化定义 / Formal Definition

**量子态** 是描述量子系统完整信息的数学对象，通常用希尔伯特空间中的向量表示：

$$|\psi\rangle = \sum_n c_n |n\rangle$$

其中：

- $|\psi\rangle$: 量子态向量
- $c_n$: 复数系数
- $|n\rangle$: 正交基向量

**归一化条件**:
$$\langle\psi|\psi\rangle = \sum_n |c_n|^2 = 1$$

#### 公理化定义 / Axiomatic Definition

**量子态公理系统** $\mathcal{QS} = \langle \mathcal{H}, \mathcal{S}, \mathcal{N}, \mathcal{U} \rangle$：

**QS1 (希尔伯特空间公理)**: 量子态存在于完备的内积空间 $\mathcal{H}$ 中
**QS2 (归一化公理)**: 物理量子态满足 $\langle\psi|\psi\rangle = 1$
**QS3 (线性叠加公理)**: 若 $|\psi_1\rangle, |\psi_2\rangle \in \mathcal{S}$，则 $|\psi\rangle = \alpha|\psi_1\rangle + \beta|\psi_2\rangle \in \mathcal{S}$
**QS4 (测量坍缩公理)**: 测量后量子态坍缩到本征态
**QS5 (时间演化公理)**: 量子态的时间演化由薛定谔方程决定

#### 等价定义 / Equivalent Definitions

**定义1 (向量表示)**: 量子态是希尔伯特空间中的归一化向量
**定义2 (密度矩阵表示)**: 量子态是迹为1的厄米正定算符
**定义3 (概率幅表示)**: 量子态是概率幅的集合，满足概率守恒
**定义4 (波函数表示)**: 量子态是位置空间的复值函数

#### 形式化定理 / Formal Theorems

**定理2.2.1 (量子态唯一性定理)**: 在相位因子范围内，量子态表示是唯一的
**证明**: 设 $|\psi_1\rangle = e^{i\phi}|\psi_2\rangle$，则所有物理量期望值相同，故表示等价。

**定理2.2.2 (量子态完备性定理)**: 任意量子态可表示为正交基的线性组合
**证明**: 由希尔伯特空间的完备性，任意向量可展开为正交基的线性组合。

**定理2.2.3 (量子态演化定理)**: 量子态的幺正演化保持内积不变
**证明**: $\langle\psi(t)|\phi(t)\rangle = \langle\psi(0)|U^\dagger(t)U(t)|\phi(0)\rangle = \langle\psi(0)|\phi(0)\rangle$

#### 算法实现 / Algorithmic Implementation

**算法2.2.1 (量子态归一化算法)**:

```python
def normalize_quantum_state(state_vector):
    """
    归一化量子态向量

    参数:
        state_vector: 复数向量

    返回:
        归一化后的量子态向量
    """
    norm = np.sqrt(np.sum(np.abs(state_vector)**2))
    if norm == 0:
        raise ValueError("零向量无法归一化")
    return state_vector / norm

def check_normalization(state_vector, tolerance=1e-10):
    """
    检查量子态是否归一化

    参数:
        state_vector: 量子态向量
        tolerance: 容差

    返回:
        是否归一化
    """
    norm = np.sum(np.abs(state_vector)**2)
    return abs(norm - 1.0) < tolerance
```

**算法2.2.2 (量子态叠加算法)**:

```python
def quantum_superposition(states, coefficients):
    """
    构造量子叠加态

    参数:
        states: 量子态列表
        coefficients: 复数系数列表

    返回:
        叠加态向量
    """
    if len(states) != len(coefficients):
        raise ValueError("态和系数数量不匹配")

    # 确保所有态具有相同维度
    dim = len(states[0])
    for state in states:
        if len(state) != dim:
            raise ValueError("所有态必须具有相同维度")

    # 构造叠加态
    superposition = np.zeros(dim, dtype=complex)
    for state, coeff in zip(states, coefficients):
        superposition += coeff * state

    # 归一化
    return normalize_quantum_state(superposition)
```

### 算符与可观测量 / Operators and Observables

#### 形式化定义 / Formal Definition

**厄米算符** 对应物理可观测量：
$$\hat{A} = \hat{A}^\dagger$$

**本征值方程**:
$$\hat{A}|\psi_n\rangle = a_n|\psi_n\rangle$$

其中：

- $\hat{A}$: 厄米算符
- $a_n$: 本征值
- $|\psi_n\rangle$: 本征态

#### 公理化定义 / Axiomatic Definition

**可观测量公理系统** $\mathcal{OB} = \langle \mathcal{O}, \mathcal{E}, \mathcal{M}, \mathcal{P} \rangle$：

**OB1 (厄米性公理)**: 可观测量对应厄米算符 $\hat{A} = \hat{A}^\dagger$
**OB2 (本征值公理)**: 厄米算符的本征值为实数
**OB3 (本征态正交公理)**: 不同本征值对应的本征态正交
**OB4 (完备性公理)**: 本征态构成完备基
**OB5 (测量公理)**: 测量结果为本征值，概率由玻恩规则决定

#### 形式化定理 / Formal Theorems

**定理2.2.4 (厄米算符本征值定理)**: 厄米算符的本征值都是实数
**证明**: 设 $\hat{A}|\psi\rangle = a|\psi\rangle$，则 $a = \langle\psi|\hat{A}|\psi\rangle = \langle\psi|\hat{A}^\dagger|\psi\rangle = a^*$，故 $a$ 为实数。

**定理2.2.5 (本征态正交定理)**: 厄米算符不同本征值对应的本征态正交
**证明**: 设 $\hat{A}|\psi_1\rangle = a_1|\psi_1\rangle$, $\hat{A}|\psi_2\rangle = a_2|\psi_2\rangle$，则 $(a_1-a_2)\langle\psi_1|\psi_2\rangle = 0$，故 $a_1 \neq a_2$ 时 $\langle\psi_1|\psi_2\rangle = 0$。

**定理2.2.6 (谱分解定理)**: 厄米算符可表示为 $\hat{A} = \sum_n a_n|n\rangle\langle n|$
**证明**: 由本征态的完备性，任意态可展开为 $|\psi\rangle = \sum_n c_n|n\rangle$，故 $\hat{A}|\psi\rangle = \sum_n a_n c_n|n\rangle = \sum_n a_n|n\rangle\langle n|\psi\rangle$。

#### 算法实现 / Algorithmic Implementation

**算法2.2.3 (厄米算符验证算法)**:

```python
def is_hermitian(operator):
    """
    检查算符是否为厄米算符

    参数:
        operator: 复数矩阵

    返回:
        是否为厄米算符
    """
    return np.allclose(operator, operator.conj().T)

def find_eigenvalues_eigenvectors(operator):
    """
    计算厄米算符的本征值和本征向量

    参数:
        operator: 厄米算符矩阵

    返回:
        本征值数组, 本征向量矩阵
    """
    if not is_hermitian(operator):
        raise ValueError("算符必须是厄米的")

    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    return eigenvalues, eigenvectors
```

**算法2.2.4 (可观测量期望值算法)**:

```python
def expectation_value(operator, state):
    """
    计算可观测量在给定态下的期望值

    参数:
        operator: 厄米算符
        state: 量子态向量

    返回:
        期望值
    """
    return np.real(np.dot(state.conj(), np.dot(operator, state)))

def variance(operator, state):
    """
    计算可观测量在给定态下的方差

    参数:
        operator: 厄米算符
        state: 量子态向量

    返回:
        方差
    """
    exp_val = expectation_value(operator, state)
    exp_val_sq = expectation_value(operator @ operator, state)
    return exp_val_sq - exp_val**2
```

### 测量与坍缩 / Measurement and Collapse

#### 形式化定义 / Formal Definition

**测量公设**: 测量可观测量 $\hat{A}$ 得到本征值 $a_n$ 的概率为：
$$P(a_n) = |\langle\psi_n|\psi\rangle|^2$$

**测量后坍缩**: 测量后系统坍缩到对应的本征态：
$$|\psi\rangle \xrightarrow{\text{measurement}} |\psi_n\rangle$$

#### 公理化定义 / Axiomatic Definition

**测量公理系统** $\mathcal{MS} = \langle \mathcal{P}, \mathcal{C}, \mathcal{U}, \mathcal{R} \rangle$：

**MS1 (玻恩规则公理)**: 测量概率由玻恩规则 $P(a_n) = |\langle\psi_n|\psi\rangle|^2$ 决定
**MS2 (坍缩公理)**: 测量后量子态坍缩到对应本征态
**MS3 (期望值公理)**: 期望值 $\langle A \rangle = \sum_n a_n P(a_n)$
**MS4 (不确定性公理)**: 测量引入不确定性，满足海森堡不确定性原理
**MS5 (投影公理)**: 测量可用投影算符 $\hat{P}_n = |n\rangle\langle n|$ 描述

#### 形式化定理 / Formal Theorems

**定理2.2.7 (测量概率归一化定理)**: 测量概率满足 $\sum_n P(a_n) = 1$
**证明**: $\sum_n P(a_n) = \sum_n |\langle\psi_n|\psi\rangle|^2 = \langle\psi|\psi\rangle = 1$，其中使用了本征态的完备性。

**定理2.2.8 (期望值定理)**: $\langle A \rangle = \langle\psi|\hat{A}|\psi\rangle$
**证明**: $\langle A \rangle = \sum_n a_n P(a_n) = \sum_n a_n |\langle\psi_n|\psi\rangle|^2 = \langle\psi|\hat{A}|\psi\rangle$。

**定理2.2.9 (测量坍缩定理)**: 测量后密度矩阵变为 $\rho' = \sum_n P_n \rho P_n$
**证明**: 由投影测量公理，测量后系统以概率 $P(a_n)$ 处于态 $|n\rangle$，故密度矩阵为 $\rho' = \sum_n P(a_n)|n\rangle\langle n| = \sum_n P_n \rho P_n$。

#### 算法实现 / Algorithmic Implementation

**算法2.2.5 (量子测量算法)**:

```python
def quantum_measurement(operator, state, num_measurements=1000):
    """
    模拟量子测量过程

    参数:
        operator: 厄米算符
        state: 量子态向量
        num_measurements: 测量次数

    返回:
        测量结果统计
    """
    eigenvalues, eigenvectors = find_eigenvalues_eigenvectors(operator)

    # 计算测量概率
    probabilities = []
    for i in range(len(eigenvalues)):
        prob = abs(np.dot(eigenvectors[:, i].conj(), state))**2
        probabilities.append(prob)

    # 模拟测量
    results = np.random.choice(eigenvalues, size=num_measurements, p=probabilities)

    return {
        'eigenvalues': eigenvalues,
        'probabilities': probabilities,
        'measurement_results': results,
        'expectation_value': np.mean(results),
        'variance': np.var(results)
    }

def projective_measurement(projector, state):
    """
    投影测量

    参数:
        projector: 投影算符
        state: 量子态向量

    返回:
        测量概率, 坍缩后的态
    """
    # 计算测量概率
    probability = np.real(np.dot(state.conj(), np.dot(projector, state)))

    # 计算坍缩后的态
    collapsed_state = np.dot(projector, state)
    if probability > 0:
        collapsed_state = collapsed_state / np.sqrt(probability)

    return probability, collapsed_state
```

**算法2.2.6 (测量不确定性算法)**:

```python
def heisenberg_uncertainty(operator_A, operator_B, state):
    """
    计算两个可观测量之间的不确定性关系

    参数:
        operator_A: 可观测量A
        operator_B: 可观测量B
        state: 量子态向量

    返回:
        不确定性关系
    """
    # 计算标准差
    delta_A = np.sqrt(variance(operator_A, state))
    delta_B = np.sqrt(variance(operator_B, state))

    # 计算交换子
    commutator = operator_A @ operator_B - operator_B @ operator_A
    commutator_expectation = expectation_value(commutator, state)

    # 不确定性关系
    uncertainty_product = delta_A * delta_B
    commutator_bound = abs(commutator_expectation) / 2

    return {
        'delta_A': delta_A,
        'delta_B': delta_B,
        'uncertainty_product': uncertainty_product,
        'commutator_bound': commutator_bound,
        'satisfies_uncertainty': uncertainty_product >= commutator_bound
    }
```

---

## 2.2.2 薛定谔方程 / Schrödinger Equation

### 薛定谔方程框架图 / Framework Diagram of Schrödinger Equation

```mermaid
graph TB
    A[薛定谔方程] --> B[时间相关薛定谔方程]
    A --> C[定态薛定谔方程]
    A --> D[本征值问题]

    B --> E[iℏ∂|ψ⟩/∂t = H|ψ⟩]
    B --> F[时间演化算符 U]
    B --> G[幺正演化]

    C --> H[H|ψ⟩ = E|ψ⟩]
    C --> I[能量本征态]
    C --> J[能级结构]

    D --> K[本征值 En]
    D --> L[本征函数 |n⟩]
    D --> M[谱分解]

    E --> N[量子动力学]
    H --> N
    K --> N

    N --> O[物理预测]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#fff4e1
    style N fill:#e8f5e9
    style O fill:#e8f5e9
```

### 薛定谔方程求解流程图 / Flowchart of Schrödinger Equation Solution

```mermaid
flowchart TD
    Start([开始]) --> Input[输入: 哈密顿算符 H, 初始态 |ψ0⟩]
    Input --> CheckType{方程类型?}
    CheckType -->|时间相关| TimeDep[时间相关薛定谔方程]
    CheckType -->|定态| TimeIndep[定态薛定谔方程]

    TimeDep --> ConstructU[构造时间演化算符 U]
    ConstructU --> TimeEvolve[时间演化 |ψt⟩ = U|ψ0⟩]
    TimeEvolve --> Output1[输出: 时间演化态]

    TimeIndep --> SolveEigen[求解本征值问题 H|n⟩ = En|n⟩]
    SolveEigen --> FindEigenval[找到本征值 En]
    SolveEigen --> FindEigenvec[找到本征态 |n⟩]
    FindEigenval --> ExpandState[展开初始态 |ψ0⟩ = Σcn|n⟩]
    FindEigenvec --> ExpandState
    ExpandState --> TimeEvolve2[时间演化 |ψt⟩ = Σcn e^(-iEnt/ℏ)|n⟩]
    TimeEvolve2 --> Output2[输出: 时间演化态]

    Output1 --> End([结束])
    Output2 --> End

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style ConstructU fill:#fff4e1
    style SolveEigen fill:#fff4e1
    style TimeEvolve fill:#e8f5e9
    style TimeEvolve2 fill:#e8f5e9
```

### 时间相关薛定谔方程 / Time-dependent Schrödinger Equation

#### 形式化定义 / Formal Definition

**一般形式**:
$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

其中：

- $\hbar$: 约化普朗克常数
- $\hat{H}$: 哈密顿算符
- $|\psi(t)\rangle$: 时间相关的量子态

**一维势场中的形式**:
$$i\hbar\frac{\partial}{\partial t}\psi(x,t) = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)$$

#### 公理化定义 / Axiomatic Definition

**薛定谔方程公理系统** $\mathcal{SE} = \langle \mathcal{H}, \mathcal{T}, \mathcal{U}, \mathcal{E} \rangle$：

**SE1 (时间演化公理)**: 量子态的时间演化由薛定谔方程决定
**SE2 (幺正性公理)**: 时间演化算符是幺正的 $U^\dagger(t)U(t) = I$
**SE3 (线性性公理)**: 薛定谔方程是线性的
**SE4 (确定性公理)**: 给定初始态和哈密顿量，演化是确定的
**SE5 (能量守恒公理)**: 在保守系统中，能量期望值守恒

#### 形式化定理 / Formal Theorems

**定理2.2.10 (时间演化幺正性定理)**: 薛定谔方程的解保持内积不变
**证明**: $\frac{d}{dt}\langle\psi(t)|\phi(t)\rangle = \frac{i}{\hbar}\langle\psi(t)|[\hat{H}^\dagger - \hat{H}]|\phi(t)\rangle = 0$，因为 $\hat{H}$ 是厄米的。

**定理2.2.11 (能量期望值守恒定理)**: 在保守系统中 $\frac{d}{dt}\langle H \rangle = 0$
**证明**: $\frac{d}{dt}\langle H \rangle = \frac{i}{\hbar}\langle[\hat{H},\hat{H}]\rangle = 0$。

**定理2.2.12 (时间演化算符定理)**: $|\psi(t)\rangle = U(t)|\psi(0)\rangle$，其中 $U(t) = e^{-i\hat{H}t/\hbar}$
**证明**: 直接代入薛定谔方程验证。

#### 算法实现 / Algorithmic Implementation

**算法2.2.7 (时间演化算法)**:

```python
def time_evolution(hamiltonian, initial_state, time_points):
    """
    计算量子态的时间演化

    参数:
        hamiltonian: 哈密顿算符
        initial_state: 初始量子态
        time_points: 时间点数组

    返回:
        时间演化后的量子态列表
    """
    eigenvalues, eigenvectors = find_eigenvalues_eigenvectors(hamiltonian)

    evolved_states = []
    for t in time_points:
        # 计算演化算符
        evolution_operator = np.zeros_like(hamiltonian, dtype=complex)
        for i, eigenvalue in enumerate(eigenvalues):
            phase = np.exp(-1j * eigenvalue * t / hbar)
            eigenvector = eigenvectors[:, i].reshape(-1, 1)
            evolution_operator += phase * (eigenvector @ eigenvector.conj().T)

        # 计算演化后的态
        evolved_state = evolution_operator @ initial_state
        evolved_states.append(evolved_state)

    return evolved_states

def energy_expectation(hamiltonian, state):
    """
    计算能量期望值

    参数:
        hamiltonian: 哈密顿算符
        state: 量子态

    返回:
        能量期望值
    """
    return expectation_value(hamiltonian, state)
```

### 定态薛定谔方程 / Time-independent Schrödinger Equation

#### 形式化定义 / Formal Definition

**定态解**:
$$\psi(x,t) = \psi(x)e^{-iEt/\hbar}$$

**定态方程**:
$$\hat{H}\psi(x) = E\psi(x)$$

**一维形式**:
$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi(x) = E\psi(x)$$

#### 公理化定义 / Axiomatic Definition

**定态公理系统** $\mathcal{SS} = \langle \mathcal{E}, \mathcal{P}, \mathcal{O}, \mathcal{N} \rangle$：

**SS1 (定态公理)**: 定态是哈密顿算符的本征态
**SS2 (能量本征值公理)**: 定态对应的能量是本征值
**SS3 (时间无关公理)**: 定态的概率密度与时间无关
**SS4 (正交性公理)**: 不同能量的定态正交
**SS5 (完备性公理)**: 定态构成完备基

#### 形式化定理 / Formal Theorems

**定理2.2.13 (定态能量定理)**: 定态的能量是确定的
**证明**: 定态是哈密顿算符的本征态，故能量为确定的本征值。

**定理2.2.14 (定态正交定理)**: 不同能量的定态正交
**证明**: 由厄米算符本征态的正交性。

**定理2.2.15 (定态完备性定理)**: 任意态可展开为定态的线性组合
**证明**: 由哈密顿算符本征态的完备性。

#### 算法实现 / Algorithmic Implementation

**算法2.2.8 (定态求解算法)**:

```python
def solve_stationary_states(hamiltonian):
    """
    求解定态薛定谔方程

    参数:
        hamiltonian: 哈密顿算符

    返回:
        能量本征值和本征态
    """
    eigenvalues, eigenvectors = find_eigenvalues_eigenvectors(hamiltonian)

    # 按能量排序
    sorted_indices = np.argsort(eigenvalues)
    sorted_energies = eigenvalues[sorted_indices]
    sorted_states = eigenvectors[:, sorted_indices]

    return sorted_energies, sorted_states

def ground_state_energy(hamiltonian):
    """
    计算基态能量

    参数:
        hamiltonian: 哈密顿算符

    返回:
        基态能量
    """
    eigenvalues, _ = find_eigenvalues_eigenvectors(hamiltonian)
    return np.min(eigenvalues)
```

### 本征值问题 / Eigenvalue Problems

#### 形式化定义 / Formal Definition

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

#### 公理化定义 / Axiomatic Definition

**势阱公理系统** $\mathcal{PW} = \langle \mathcal{V}, \mathcal{B}, \mathcal{E}, \mathcal{N} \rangle$：

**PW1 (边界条件公理)**: 波函数在势阱边界为零
**PW2 (归一化公理)**: 本征函数归一化
**PW3 (正交性公理)**: 不同能级的本征函数正交
**PW4 (完备性公理)**: 本征函数构成完备基
**PW5 (节点定理)**: 第n个本征函数有n-1个节点

#### 形式化定理 / Formal Theorems

**定理2.2.16 (节点定理)**: 第n个本征函数在势阱内有n-1个节点
**证明**: 由Sturm-Liouville理论，本征函数按节点数递增排列。

**定理2.2.17 (能级间距定理)**: 相邻能级间距随n增加而增加
**证明**: $E_{n+1} - E_n = \frac{(2n+1)\pi^2\hbar^2}{2ma^2}$ 随n增加。

**定理2.2.18 (基态无节点定理)**: 基态本征函数在势阱内无节点
**证明**: 基态对应最低能量，由变分原理，无节点的函数能量最低。

#### 算法实现 / Algorithmic Implementation

**算法2.2.9 (无限深势阱求解算法)**:

```python
def infinite_potential_well(a, n_max, mass=1.0):
    """
    求解无限深势阱的本征值和本征函数

    参数:
        a: 势阱宽度
        n_max: 最大能级数
        mass: 粒子质量

    返回:
        能级和本征函数
    """
    energies = []
    wavefunctions = []

    for n in range(1, n_max + 1):
        # 计算能级
        energy = (n**2 * np.pi**2 * hbar**2) / (2 * mass * a**2)
        energies.append(energy)

        # 计算本征函数（在离散点上）
        x = np.linspace(0, a, 1000)
        wavefunction = np.sqrt(2/a) * np.sin(n * np.pi * x / a)
        wavefunctions.append(wavefunction)

    return np.array(energies), np.array(wavefunctions)

def probability_density(wavefunction, x):
    """
    计算概率密度

    参数:
        wavefunction: 波函数
        x: 位置坐标

    返回:
        概率密度
    """
    return np.abs(wavefunction)**2
```

---

## 2.2.3 海森堡不确定性原理 / Heisenberg Uncertainty Principle

### 位置-动量不确定性 / Position-Momentum Uncertainty

#### 形式化定义 / Formal Definition

**不确定性关系**:
$$\Delta x \Delta p \geq \frac{\hbar}{2}$$

其中：

- $\Delta x$: 位置的标准差
- $\Delta p$: 动量的标准差

**高斯波包的最小不确定性**:
$$\Delta x \Delta p = \frac{\hbar}{2}$$

#### 公理化定义 / Axiomatic Definition

**不确定性公理系统** $\mathcal{UP} = \langle \mathcal{O}, \mathcal{S}, \mathcal{M}, \mathcal{B} \rangle$：

**UP1 (交换子公理)**: 位置和动量算符满足 $[\hat{x},\hat{p}] = i\hbar$
**UP2 (标准差公理)**: 标准差定义为 $\Delta A = \sqrt{\langle A^2 \rangle - \langle A \rangle^2}$
**UP3 (最小不确定性公理)**: 高斯波包达到最小不确定性
**UP4 (测量干扰公理)**: 测量一个量会影响另一个量的精度
**UP5 (量子本质公理)**: 不确定性是量子力学的本质特征

#### 形式化定理 / Formal Theorems

**定理2.2.19 (海森堡不确定性定理)**: $\Delta x \Delta p \geq \frac{\hbar}{2}$
**证明**: 使用施瓦茨不等式和交换子关系 $[\hat{x},\hat{p}] = i\hbar$。

**定理2.2.20 (最小不确定性定理)**: 高斯波包达到最小不确定性
**证明**: 高斯波包 $\psi(x) = \frac{1}{\sqrt{\sigma\sqrt{\pi}}}e^{-x^2/2\sigma^2}$ 满足 $\Delta x \Delta p = \frac{\hbar}{2}$。

**定理2.2.21 (不确定性传播定理)**: 不确定性关系在时间演化下保持不变
**证明**: 幺正演化保持内积，故不确定性关系不变。

#### 算法实现 / Algorithmic Implementation

**算法2.2.10 (位置-动量不确定性算法)**:

```python
def position_momentum_uncertainty(wavefunction, x_grid, hbar=1.0):
    """
    计算位置-动量不确定性关系

    参数:
        wavefunction: 波函数
        x_grid: 位置网格
        hbar: 约化普朗克常数

    返回:
        位置和动量的标准差
    """
    # 归一化波函数
    norm = np.sqrt(np.trapz(np.abs(wavefunction)**2, x_grid))
    wavefunction = wavefunction / norm

    # 计算位置期望值和方差
    x_expectation = np.trapz(x_grid * np.abs(wavefunction)**2, x_grid)
    x_variance = np.trapz((x_grid - x_expectation)**2 * np.abs(wavefunction)**2, x_grid)

    # 计算动量期望值和方差（通过傅里叶变换）
    dx = x_grid[1] - x_grid[0]
    k_grid = np.fft.fftfreq(len(x_grid), dx) * 2 * np.pi
    momentum_wavefunction = np.fft.fft(wavefunction) * dx / np.sqrt(2 * np.pi)

    p_expectation = np.trapz(k_grid * hbar * np.abs(momentum_wavefunction)**2, k_grid)
    p_variance = np.trapz((k_grid * hbar - p_expectation)**2 * np.abs(momentum_wavefunction)**2, k_grid)

    delta_x = np.sqrt(x_variance)
    delta_p = np.sqrt(p_variance)

    return delta_x, delta_p, delta_x * delta_p

def gaussian_wave_packet(x, x0, sigma, k0):
    """
    构造高斯波包

    参数:
        x: 位置坐标
        x0: 中心位置
        sigma: 宽度参数
        k0: 中心动量

    返回:
        高斯波包
    """
    return (1 / (sigma * np.sqrt(np.pi)))**0.5 * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
```

### 时间-能量不确定性 / Time-Energy Uncertainty

#### 形式化定义 / Formal Definition

**时间-能量不确定性**:
$$\Delta t \Delta E \geq \frac{\hbar}{2}$$

**应用**: 解释粒子寿命与能量宽度的关系。

#### 公理化定义 / Axiomatic Definition

**时间-能量不确定性公理系统** $\mathcal{TE} = \langle \mathcal{T}, \mathcal{E}, \mathcal{L}, \mathcal{W} \rangle$：

**TE1 (寿命-宽度公理)**: 粒子寿命与能量宽度满足 $\Delta t \Delta E \geq \frac{\hbar}{2}$
**TE2 (衰变公理)**: 不稳定粒子的能量分布有宽度
**TE3 (测量时间公理)**: 测量时间影响能量精度
**TE4 (共振公理)**: 共振态的能量宽度与寿命相关
**TE5 (量子隧道公理)**: 隧道效应中的时间-能量关系

#### 形式化定理 / Formal Theorems

**定理2.2.22 (时间-能量不确定性定理)**: $\Delta t \Delta E \geq \frac{\hbar}{2}$
**证明**: 通过傅里叶变换和时间-频率对偶性证明。

**定理2.2.23 (粒子寿命定理)**: 不稳定粒子的能量宽度 $\Gamma$ 与寿命 $\tau$ 满足 $\Gamma \tau \approx \hbar$
**证明**: 由指数衰变和傅里叶变换得到洛伦兹分布。

**定理2.2.24 (测量时间定理)**: 测量时间 $\Delta t$ 与能量精度 $\Delta E$ 满足不确定性关系
**证明**: 测量过程的时间分辨率和能量分辨率相互制约。

#### 算法实现 / Algorithmic Implementation

**算法2.2.11 (时间-能量不确定性算法)**:

```python
def time_energy_uncertainty(time_signal, time_grid, hbar=1.0):
    """
    计算时间-能量不确定性关系

    参数:
        time_signal: 时间信号
        time_grid: 时间网格
        hbar: 约化普朗克常数

    返回:
        时间和能量的标准差
    """
    # 归一化信号
    norm = np.sqrt(np.trapz(np.abs(time_signal)**2, time_grid))
    signal = time_signal / norm

    # 计算时间期望值和方差
    t_expectation = np.trapz(time_grid * np.abs(signal)**2, time_grid)
    t_variance = np.trapz((time_grid - t_expectation)**2 * np.abs(signal)**2, time_grid)

    # 计算能量期望值和方差（通过傅里叶变换）
    dt = time_grid[1] - time_grid[0]
    omega_grid = np.fft.fftfreq(len(time_grid), dt) * 2 * np.pi
    energy_signal = np.fft.fft(signal) * dt / np.sqrt(2 * np.pi)

    E_expectation = np.trapz(omega_grid * hbar * np.abs(energy_signal)**2, omega_grid)
    E_variance = np.trapz((omega_grid * hbar - E_expectation)**2 * np.abs(energy_signal)**2, omega_grid)

    delta_t = np.sqrt(t_variance)
    delta_E = np.sqrt(E_variance)

    return delta_t, delta_E, delta_t * delta_E

def exponential_decay(t, tau, E0):
    """
    构造指数衰变信号

    参数:
        t: 时间
        tau: 寿命
        E0: 初始能量

    返回:
        衰变信号
    """
    return np.exp(-t / tau) * np.exp(-1j * E0 * t / hbar)
```

### 一般不确定性关系 / General Uncertainty Relations

#### 形式化定义 / Formal Definition

**任意两个可观测量**:
$$\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$$

其中 $[\hat{A},\hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$ 是交换子。

#### 公理化定义 / Axiomatic Definition

**一般不确定性公理系统** $\mathcal{GU} = \langle \mathcal{O}, \mathcal{C}, \mathcal{I}, \mathcal{B} \rangle$：

**GU1 (交换子公理)**: 任意两个算符的交换子定义不确定性关系
**GU2 (施瓦茨不等式公理)**: 不确定性关系基于施瓦茨不等式
**GU3 (对易算符公理)**: 对易算符可同时精确测量
**GU4 (互补性公理)**: 不对易的算符具有互补性
**GU5 (测量精度公理)**: 测量精度受不确定性关系限制

#### 形式化定理 / Formal Theorems

**定理2.2.25 (一般不确定性定理)**: $\Delta A \Delta B \geq \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$
**证明**: 使用施瓦茨不等式 $|\langle f|g\rangle|^2 \leq \langle f|f\rangle\langle g|g\rangle$。

**定理2.2.26 (对易算符定理)**: 若 $[\hat{A},\hat{B}] = 0$，则 $\Delta A \Delta B \geq 0$
**证明**: 对易算符有共同本征态，可同时精确测量。

**定理2.2.27 (最小不确定性态定理)**: 最小不确定性态满足 $(\hat{A} - \langle A \rangle)|\psi\rangle = i\lambda(\hat{B} - \langle B \rangle)|\psi\rangle$
**证明**: 当施瓦茨不等式取等号时，$|f\rangle$ 和 $|g\rangle$ 线性相关。

#### 算法实现 / Algorithmic Implementation

**算法2.2.12 (一般不确定性算法)**:

```python
def general_uncertainty(operator_A, operator_B, state):
    """
    计算两个可观测量的一般不确定性关系

    参数:
        operator_A: 可观测量A
        operator_B: 可观测量B
        state: 量子态

    返回:
        不确定性关系
    """
    # 计算标准差
    delta_A = np.sqrt(variance(operator_A, state))
    delta_B = np.sqrt(variance(operator_B, state))

    # 计算交换子
    commutator = operator_A @ operator_B - operator_B @ operator_A
    commutator_expectation = expectation_value(commutator, state)

    # 不确定性关系
    uncertainty_product = delta_A * delta_B
    commutator_bound = abs(commutator_expectation) / 2

    return {
        'delta_A': delta_A,
        'delta_B': delta_B,
        'uncertainty_product': uncertainty_product,
        'commutator_bound': commutator_bound,
        'satisfies_uncertainty': uncertainty_product >= commutator_bound,
        'commutator': commutator_expectation
    }

def check_commutation(operator_A, operator_B, tolerance=1e-10):
    """
    检查两个算符是否对易

    参数:
        operator_A: 算符A
        operator_B: 算符B
        tolerance: 容差

    返回:
        是否对易
    """
    commutator = operator_A @ operator_B - operator_B @ operator_A
    return np.allclose(commutator, 0, atol=tolerance)
```

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

## 2.2.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use num_complex::Complex64;
use nalgebra::{DVector, DMatrix};

// 量子态结构
pub struct QuantumState {
    amplitudes: DVector<Complex64>,
}

impl QuantumState {
    pub fn new(amplitudes: Vec<Complex64>) -> Self {
        let state = QuantumState {
            amplitudes: DVector::from_vec(amplitudes),
        };
        state.normalize()
    }

    pub fn normalize(mut self) -> Self {
        let norm = self.amplitudes.norm();
        if norm > 1e-10 {
            self.amplitudes /= norm;
        }
        self
    }

    pub fn is_normalized(&self, tolerance: f64) -> bool {
        (self.amplitudes.norm() - 1.0).abs() < tolerance
    }

    pub fn expectation_value(&self, operator: &DMatrix<Complex64>) -> f64 {
        let result = operator * &self.amplitudes;
        self.amplitudes.dot(&result).re
    }

    pub fn measure(&self, operator: &DMatrix<Complex64>) -> (usize, Complex64) {
        // 计算本征值和本征向量
        let eigen = operator.clone().try_eigen().unwrap();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // 计算测量概率
        let mut probabilities = Vec::new();
        for i in 0..eigenvectors.ncols() {
            let eigenvector = eigenvectors.column(i);
            let prob = self.amplitudes.dot(&eigenvector).norm().powi(2);
            probabilities.push(prob);
        }

        // 根据概率分布选择测量结果
        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return (i, eigenvalues[i]);
            }
        }

        (probabilities.len() - 1, eigenvalues[probabilities.len() - 1])
    }
}

// 量子算符
pub struct QuantumOperator {
    matrix: DMatrix<Complex64>,
}

impl QuantumOperator {
    pub fn pauli_x() -> Self {
        QuantumOperator {
            matrix: DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ]),
        }
    }

    pub fn pauli_y() -> Self {
        QuantumOperator {
            matrix: DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ]),
        }
    }

    pub fn pauli_z() -> Self {
        QuantumOperator {
            matrix: DMatrix::from_row_slice(2, 2, &[
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ]),
        }
    }

    pub fn hadamard() -> Self {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        QuantumOperator {
            matrix: DMatrix::from_row_slice(2, 2, &[
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(-sqrt2_inv, 0.0),
            ]),
        }
    }

    pub fn apply(&self, state: &QuantumState) -> QuantumState {
        QuantumState {
            amplitudes: &self.matrix * &state.amplitudes,
        }.normalize()
    }
}

// 薛定谔方程求解器
pub struct SchrodingerSolver {
    hamiltonian: DMatrix<Complex64>,
    dt: f64,
}

impl SchrodingerSolver {
    pub fn new(hamiltonian: DMatrix<Complex64>, dt: f64) -> Self {
        SchrodingerSolver { hamiltonian, dt }
    }

    pub fn evolve(&self, state: &QuantumState, time: f64) -> QuantumState {
        // 简化的时间演化：U(t) = exp(-iHt/hbar)
        // 这里使用近似：U ≈ I - iH*dt (对于小dt)
        let hbar = 1.0; // 自然单位制
        let evolution_op = DMatrix::identity(self.hamiltonian.nrows(), self.hamiltonian.ncols())
            - Complex64::new(0.0, self.dt / hbar) * &self.hamiltonian;

        QuantumState {
            amplitudes: &evolution_op * &state.amplitudes,
        }.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_normalization() {
        let state = QuantumState::new(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_pauli_operators() {
        let x = QuantumOperator::pauli_x();
        let state = QuantumState::new(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let result = x.apply(&state);
        assert!((result.amplitudes[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((result.amplitudes[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module QuantumMechanics where

import Data.Complex
import Data.Vector (Vector)
import qualified Data.Vector as V
import Numeric.LinearAlgebra

-- 量子态类型
type QuantumState = Vector (Complex Double)

-- 归一化量子态
normalize :: QuantumState -> QuantumState
normalize state = V.map (/ norm) state
    where
        norm = sqrt $ sum $ map (\x -> magnitude x ** 2) $ V.toList state

-- 检查是否归一化
isNormalized :: QuantumState -> Double -> Bool
isNormalized state tolerance = abs (norm - 1.0) < tolerance
    where
        norm = sqrt $ sum $ map (\x -> magnitude x ** 2) $ V.toList state

-- 量子算符类型
type QuantumOperator = Matrix (Complex Double)

-- Pauli-X算符
pauliX :: QuantumOperator
pauliX = (2><2) [0, 1, 1, 0]

-- Pauli-Y算符
pauliY :: QuantumOperator
pauliY = (2><2) [0 :+ 0, 0 :+ (-1), 0 :+ 1, 0 :+ 0]

-- Pauli-Z算符
pauliZ :: QuantumOperator
pauliZ = (2><2) [1, 0, 0, -1]

-- Hadamard门
hadamard :: QuantumOperator
hadamard = (2><2) [sqrt2_inv, sqrt2_inv, sqrt2_inv, -sqrt2_inv]
    where
        sqrt2_inv = 1.0 / sqrt 2.0

-- 应用算符到量子态
applyOperator :: QuantumOperator -> QuantumState -> QuantumState
applyOperator op state = normalize $ V.fromList $ toList $ op #> (fromList $ V.toList state)

-- 期望值
expectationValue :: QuantumOperator -> QuantumState -> Double
expectationValue op state = realPart $ sum $ zipWith (*) (V.toList state) (V.toList result)
    where
        result = V.fromList $ toList $ op #> (fromList $ V.toList state)

-- 量子叠加
superposition :: [QuantumState] -> [Complex Double] -> QuantumState
superposition states coefficients = normalize $ V.generate (V.length $ head states) $ \i ->
    sum $ zipWith (*) coefficients $ map (V.! i) states

-- 示例使用
example :: IO ()
example = do
    -- 创建量子态 |0⟩
    let state0 = V.fromList [1 :+ 0, 0 :+ 0]

    -- 应用Hadamard门
    let state_super = applyOperator hadamard state0
    putStrLn $ "Hadamard(|0⟩) = " ++ show state_super

    -- 计算期望值
    let exp_val = expectationValue pauliZ state_super
    putStrLn $ "期望值 ⟨Z⟩ = " ++ show exp_val
```

### Julia实现示例 / Julia Implementation Example

```julia
using LinearAlgebra
using Random

# 量子态结构
struct QuantumState
    amplitudes::Vector{ComplexF64}

    function QuantumState(amplitudes::Vector{ComplexF64})
        new(normalize(amplitudes))
    end
end

# 归一化量子态
function normalize(amplitudes::Vector{ComplexF64})::Vector{ComplexF64}
    norm = sqrt(sum(abs2, amplitudes))
    return norm > 1e-10 ? amplitudes / norm : amplitudes
end

# 检查是否归一化
function is_normalized(state::QuantumState, tolerance::Float64=1e-10)::Bool
    norm = sqrt(sum(abs2, state.amplitudes))
    return abs(norm - 1.0) < tolerance
end

# 期望值
function expectation_value(state::QuantumState, operator::Matrix{ComplexF64})::Float64
    result = operator * state.amplitudes
    return real(dot(state.amplitudes, result))
end

# 量子测量
function measure(state::QuantumState, operator::Matrix{ComplexF64})::Tuple{Int, ComplexF64}
    # 计算本征值和本征向量
    eigenvals, eigenvecs = eigen(operator)

    # 计算测量概率
    probabilities = Float64[]
    for i in 1:length(eigenvals)
        prob = abs2(dot(state.amplitudes, eigenvecs[:, i]))
        push!(probabilities, prob)
    end

    # 归一化概率
    probabilities ./= sum(probabilities)

    # 根据概率分布选择测量结果
    random_val = rand()
    cumulative = 0.0
    for (i, prob) in enumerate(probabilities)
        cumulative += prob
        if random_val <= cumulative
            return (i, eigenvals[i])
        end
    end

    return (length(eigenvals), eigenvals[end])
end

# 量子算符
struct QuantumOperator
    matrix::Matrix{ComplexF64}
end

# Pauli-X算符
function pauli_x()::QuantumOperator
    QuantumOperator([0 1; 1 0])
end

# Pauli-Y算符
function pauli_y()::QuantumOperator
    QuantumOperator([0 -im; im 0])
end

# Pauli-Z算符
function pauli_z()::QuantumOperator
    QuantumOperator([1 0; 0 -1])
end

# Hadamard门
function hadamard()::QuantumOperator
    sqrt2_inv = 1.0 / sqrt(2.0)
    QuantumOperator([sqrt2_inv sqrt2_inv; sqrt2_inv -sqrt2_inv])
end

# 应用算符到量子态
function apply(op::QuantumOperator, state::QuantumState)::QuantumState
    QuantumState(op.matrix * state.amplitudes)
end

# 薛定谔方程求解器
struct SchrodingerSolver
    hamiltonian::Matrix{ComplexF64}
    dt::Float64
end

function evolve(solver::SchrodingerSolver, state::QuantumState, time::Float64)::QuantumState
    # 简化的时间演化：U(t) = exp(-iHt/hbar)
    hbar = 1.0  # 自然单位制
    evolution_op = exp(-im * solver.hamiltonian * time / hbar)
    QuantumState(evolution_op * state.amplitudes)
end

# 量子叠加
function superposition(states::Vector{QuantumState}, coefficients::Vector{ComplexF64})::QuantumState
    result = zeros(ComplexF64, length(states[1].amplitudes))
    for (state, coeff) in zip(states, coefficients)
        result .+= coeff .* state.amplitudes
    end
    QuantumState(result)
end

# 使用示例
Random.seed!(42)

# 创建量子态 |0⟩
state0 = QuantumState([1.0 + 0.0im, 0.0 + 0.0im])
println("初始态 |0⟩: ", state0.amplitudes)
println("是否归一化: ", is_normalized(state0))

# 应用Hadamard门
had = hadamard()
state_super = apply(had, state0)
println("\nHadamard(|0⟩): ", state_super.amplitudes)

# 计算期望值
pauli_z_op = pauli_z()
exp_val = expectation_value(state_super, pauli_z_op.matrix)
println("期望值 ⟨Z⟩: ", exp_val)

# 量子测量
idx, eigenvalue = measure(state_super, pauli_z_op.matrix)
println("\n测量结果: 本征值索引=", idx, ", 本征值=", eigenvalue)
```

---

## 相关模型 / Related Models

### 物理科学模型 / Physical Science Models

- [经典力学模型](../01-经典力学模型/README.md) - 量子力学的经典极限，对应原理
- [相对论模型](../03-相对论模型/README.md) - 相对论量子力学，狄拉克方程
- [热力学模型](../04-热力学模型/README.md) - 量子统计力学，费米-狄拉克统计
- [光学模型](../06-光学模型/README.md) - 量子光学，光子与光子的相互作用
- [电磁学模型](../05-电磁学模型/README.md) - 量子电动力学，电磁场的量子化

### 基础理论 / Basic Theory

- [模型分类学](../../01-基础理论/01-模型分类学/README.md) - 量子力学模型的分类
- [形式化方法论](../../01-基础理论/02-形式化方法论/README.md) - 量子力学的形式化方法
- [科学模型论](../../01-基础理论/03-科学模型论/README.md) - 量子力学作为科学模型的理论基础

## 参考文献 / References

1. Sakurai, J. J. (1994). Modern Quantum Mechanics. Addison-Wesley.
2. Griffiths, D. J. (2005). Introduction to Quantum Mechanics. Pearson.
3. Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information. Cambridge University Press.
4. Peskin, M. E., & Schroeder, D. V. (1995). An Introduction to Quantum Field Theory. Addison-Wesley.
5. Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox. Physics Physique Физика.

---

*最后更新: 2025-01-XX*
*版本: 1.2.0*
*状态: 核心功能已完成 / Status: Core Features Completed*
