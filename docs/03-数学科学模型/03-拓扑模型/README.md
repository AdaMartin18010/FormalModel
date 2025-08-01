# 3.3 拓扑模型 / Topological Models

## 目录 / Table of Contents

- [3.3 拓扑模型 / Topological Models](#33-拓扑模型--topological-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [3.3.1 点集拓扑 / Point-Set Topology](#331-点集拓扑--point-set-topology)
    - [拓扑空间 / Topological Spaces](#拓扑空间--topological-spaces)
    - [连续映射 / Continuous Maps](#连续映射--continuous-maps)
    - [紧致性 / Compactness](#紧致性--compactness)
  - [3.3.2 代数拓扑 / Algebraic Topology](#332-代数拓扑--algebraic-topology)
    - [同伦论 / Homotopy Theory](#同伦论--homotopy-theory)
    - [同调论 / Homology Theory](#同调论--homology-theory)
    - [上同调论 / Cohomology Theory](#上同调论--cohomology-theory)
  - [3.3.3 微分拓扑 / Differential Topology](#333-微分拓扑--differential-topology)
    - [流形 / Manifolds](#流形--manifolds)
    - [切丛 / Tangent Bundles](#切丛--tangent-bundles)
    - [向量场 / Vector Fields](#向量场--vector-fields)
  - [3.3.4 纤维丛 / Fiber Bundles](#334-纤维丛--fiber-bundles)
    - [主丛 / Principal Bundles](#主丛--principal-bundles)
    - [向量丛 / Vector Bundles](#向量丛--vector-bundles)
    - [联络 / Connections](#联络--connections)
  - [3.3.5 示性类 / Characteristic Classes](#335-示性类--characteristic-classes)
    - [陈类 / Chern Classes](#陈类--chern-classes)
    - [庞特里亚金类 / Pontryagin Classes](#庞特里亚金类--pontryagin-classes)
    - [施蒂费尔-惠特尼类 / Stiefel-Whitney Classes](#施蒂费尔-惠特尼类--stiefel-whitney-classes)
  - [3.3.6 低维拓扑 / Low-Dimensional Topology](#336-低维拓扑--low-dimensional-topology)
    - [曲面 / Surfaces](#曲面--surfaces)
    - [三维流形 / 3-Manifolds](#三维流形--3-manifolds)
    - [纽结理论 / Knot Theory](#纽结理论--knot-theory)
  - [3.3.7 拓扑应用 / Topological Applications](#337-拓扑应用--topological-applications)
    - [拓扑数据分析 / Topological Data Analysis](#拓扑数据分析--topological-data-analysis)
    - [拓扑量子场论 / Topological Quantum Field Theory](#拓扑量子场论--topological-quantum-field-theory)
    - [拓扑绝缘体 / Topological Insulators](#拓扑绝缘体--topological-insulators)
  - [参考文献 / References](#参考文献--references)

---

## 3.3.1 点集拓扑 / Point-Set Topology

### 拓扑空间 / Topological Spaces

**定义**: 拓扑空间 $(X, \tau)$ 包含：

- 集合 $X$
- 拓扑 $\tau$ (开集族)

**开集公理**:

1. $\emptyset, X \in \tau$
2. 有限交: $U_1, U_2 \in \tau \Rightarrow U_1 \cap U_2 \in \tau$
3. 任意并: $\{U_i\} \subset \tau \Rightarrow \bigcup U_i \in \tau$

**闭集**: $A$ 是闭集当且仅当 $X \setminus A$ 是开集。

### 连续映射 / Continuous Maps

**定义**: $f: X \to Y$ 连续当且仅当对任意开集 $V \subset Y$，$f^{-1}(V)$ 是开集。

**同胚**: 双射连续映射，其逆也连续。

**拓扑不变量**: 在同胚下保持的性质。

### 紧致性 / Compactness

**定义**: 空间 $X$ 紧致当且仅当每个开覆盖都有有限子覆盖。

**海涅-博雷尔定理**: $\mathbb{R}^n$ 的子集紧致当且仅当它是有界闭集。

**紧致性性质**:

- 紧致空间的连续像是紧致的
- 紧致空间在豪斯多夫空间中的像是闭的

---

## 3.3.2 代数拓扑 / Algebraic Topology

### 同伦论 / Homotopy Theory

**同伦**: 两个连续映射 $f, g: X \to Y$ 之间的连续变形。

**同伦等价**: 存在映射 $f: X \to Y$ 和 $g: Y \to X$ 使得 $f \circ g \simeq \text{id}_Y$，$g \circ f \simeq \text{id}_X$。

**基本群**: $\pi_1(X, x_0)$ 是点 $x_0$ 处的环路同伦类群。

**高阶同伦群**: $\pi_n(X, x_0)$ 是 $n$ 维球面到 $X$ 的映射同伦类群。

### 同调论 / Homology Theory

**奇异同调**: $H_n(X)$ 是 $n$ 维同调群。

**胞腔同调**: 基于胞腔分解的同调理论。

**同调序列**: 长正合序列
$$\cdots \to H_n(A) \to H_n(X) \to H_n(X,A) \to H_{n-1}(A) \to \cdots$$

### 上同调论 / Cohomology Theory

**奇异上同调**: $H^n(X; G)$ 是系数在 $G$ 中的上同调群。

**上同调环**: $H^*(X)$ 具有环结构。

**上同调运算**: 斯廷罗德运算等。

---

## 3.3.3 微分拓扑 / Differential Topology

### 流形 / Manifolds

**定义**: $n$ 维流形是局部同胚于 $\mathbb{R}^n$ 的豪斯多夫空间。

**坐标卡**: $(U, \phi)$ 其中 $U$ 是开集，$\phi: U \to \mathbb{R}^n$ 是同胚。

**光滑流形**: 坐标变换是光滑的。

**切空间**: $T_p M$ 是点 $p$ 处的切向量空间。

### 切丛 / Tangent Bundles

**切丛**: $TM = \bigcup_{p \in M} T_p M$

**切向量**: 满足莱布尼茨规则的线性算子。

**切空间基**: $\{\frac{\partial}{\partial x^i}\}$

**余切丛**: $T^*M$ 是切丛的对偶。

### 向量场 / Vector Fields

**定义**: 向量场是切丛的截面。

**李括号**: $[X, Y] = XY - YX$

**积分曲线**: 向量场的积分曲线满足 $\frac{dx}{dt} = X(x)$。

---

## 3.3.4 纤维丛 / Fiber Bundles

### 主丛 / Principal Bundles

**定义**: 主 $G$-丛 $(P, M, \pi, G)$ 包含：

- 全空间 $P$
- 底空间 $M$
- 投影 $\pi: P \to M$
- 结构群 $G$

**局部平凡化**: $(U, \phi)$ 其中 $\phi: \pi^{-1}(U) \to U \times G$。

**主丛上的联络**: 水平分布。

### 向量丛 / Vector Bundles

**定义**: 向量丛是纤维为向量空间的纤维丛。

**截面**: 向量丛的截面是底空间到全空间的映射。

**协变导数**: 向量丛上的联络。

### 联络 / Connections

**仿射联络**: 切丛上的联络。

**曲率**: $R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$

**挠率**: $T(X, Y) = \nabla_X Y - \nabla_Y X - [X, Y]$

---

## 3.3.5 示性类 / Characteristic Classes

### 陈类 / Chern Classes

**陈类**: 复向量丛的示性类。

**陈-韦尔理论**: 陈类是曲率形式的闭形式的上同调类。

**陈数**: 陈类的积分。

### 庞特里亚金类 / Pontryagin Classes

**庞特里亚金类**: 实向量丛的示性类。

**庞特里亚金数**: 庞特里亚金类的积分。

### 施蒂费尔-惠特尼类 / Stiefel-Whitney Classes

**施蒂费尔-惠特尼类**: 实向量丛的 $\mathbb{Z}_2$ 示性类。

**施蒂费尔-惠特尼数**: 施蒂费尔-惠特尼类的积分。

---

## 3.3.6 低维拓扑 / Low-Dimensional Topology

### 曲面 / Surfaces

**分类定理**: 紧致连通曲面同胚于：

- 球面 $S^2$
- 环面 $T^2$
- 连通和 $\#_g T^2$

**亏格**: 曲面的"洞"的数量。

**欧拉示性数**: $\chi = 2 - 2g$。

### 三维流形 / 3-Manifolds

**素分解**: 每个三维流形可以分解为素流形的连通和。

**几何化猜想**: 每个三维流形都有几何结构。

**瑟斯顿几何**: 八种三维几何。

### 纽结理论 / Knot Theory

**纽结**: $S^1$ 到 $S^3$ 的嵌入。

**纽结不变量**: 琼斯多项式、亚历山大多项式等。

**纽结群**: 纽结的补集的基本群。

---

## 3.3.7 拓扑应用 / Topological Applications

### 拓扑数据分析 / Topological Data Analysis

**持续同调**: 研究数据集的拓扑特征。

**Morse理论**: 研究流形的拓扑与临界点。

**离散Morse理论**: 应用于组合对象。

### 拓扑量子场论 / Topological Quantum Field Theory

**Chern-Simons理论**: 三维拓扑量子场论。

**Donaldson理论**: 四维流形的拓扑不变量。

**Seiberg-Witten理论**: 四维流形的规范理论。

### 拓扑绝缘体 / Topological Insulators

**拓扑绝缘体**: 具有非平凡拓扑序的材料。

**量子霍尔效应**: 二维拓扑绝缘体。

**拓扑不变量**: 陈数、$Z_2$ 不变量等。

---

## 参考文献 / References

1. Munkres, J. R. (2000). Topology. Prentice Hall.
2. Hatcher, A. (2002). Algebraic Topology. Cambridge University Press.
3. Milnor, J. W., & Stasheff, J. D. (1974). Characteristic Classes. Princeton University Press.
4. Guillemin, V., & Pollack, A. (2010). Differential Topology. AMS Chelsea Publishing.
5. Adams, J. F. (1974). Stable Homotopy and Generalised Homology. University of Chicago Press.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
