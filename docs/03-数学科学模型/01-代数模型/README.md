# 3.1 代数模型 / Algebraic Models

## 目录 / Table of Contents

- [3.1 代数模型 / Algebraic Models](#31-代数模型--algebraic-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [3.1.1 群论模型 / Group Theory Models](#311-群论模型--group-theory-models)
    - [群的定义 / Group Definition](#群的定义--group-definition)
    - [子群 / Subgroups](#子群--subgroups)
    - [同态与同构 / Homomorphisms and Isomorphisms](#同态与同构--homomorphisms-and-isomorphisms)
    - [重要群类 / Important Group Classes](#重要群类--important-group-classes)
  - [3.1.2 环论模型 / Ring Theory Models](#312-环论模型--ring-theory-models)
    - [环的定义 / Ring Definition](#环的定义--ring-definition)
    - [理想 / Ideals](#理想--ideals)
    - [商环 / Quotient Rings](#商环--quotient-rings)
  - [3.1.3 域论模型 / Field Theory Models](#313-域论模型--field-theory-models)
    - [域的定义 / Field Definition](#域的定义--field-definition)
    - [域扩张 / Field Extensions](#域扩张--field-extensions)
    - [伽罗瓦理论 / Galois Theory](#伽罗瓦理论--galois-theory)
    - [算法实现：`GF(p)[x]`多项式与不可约性 / Algorithms: Polynomials over GF(p)](#算法实现gfpx多项式与不可约性--algorithms-polynomials-over-gfp)
    - [算法实现：多项式扩展欧几里得与GF(p^m) / Algorithms: Poly xgcd and GF(p^m)](#算法实现多项式扩展欧几里得与gfpm--algorithms-poly-xgcd-and-gfpm)
  - [3.1.4 线性代数模型 / Linear Algebra Models](#314-线性代数模型--linear-algebra-models)
    - [向量空间 / Vector Spaces](#向量空间--vector-spaces)
    - [线性变换 / Linear Transformations](#线性变换--linear-transformations)
    - [特征值与特征向量 / Eigenvalues and Eigenvectors](#特征值与特征向量--eigenvalues-and-eigenvectors)
    - [算法实现：秩与零空间 / Algorithms: Rank and Null Space](#算法实现秩与零空间--algorithms-rank-and-null-space)
  - [3.1.5 同调代数模型 / Homological Algebra Models](#315-同调代数模型--homological-algebra-models)
    - [链复形 / Chain Complexes](#链复形--chain-complexes)
    - [上链复形 / Cochain Complexes](#上链复形--cochain-complexes)
    - [导出函子 / Derived Functors](#导出函子--derived-functors)
  - [3.1.6 表示论模型 / Representation Theory Models](#316-表示论模型--representation-theory-models)
    - [群表示 / Group Representations](#群表示--group-representations)
    - [特征理论 / Character Theory](#特征理论--character-theory)
  - [3.1.7 代数几何模型 / Algebraic Geometry Models](#317-代数几何模型--algebraic-geometry-models)
    - [仿射代数集 / Affine Algebraic Sets](#仿射代数集--affine-algebraic-sets)
    - [射影代数集 / Projective Algebraic Sets](#射影代数集--projective-algebraic-sets)
  - [3.1.8 实现与应用 / Implementation and Applications](#318-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [Python实现示例 / Python Implementation Example](#python实现示例--python-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [密码学 / Cryptography](#密码学--cryptography)
      - [编码理论 / Coding Theory](#编码理论--coding-theory)
      - [量子计算 / Quantum Computing](#量子计算--quantum-computing)
  - [参考文献 / References](#参考文献--references)

---

## 3.1.1 群论模型 / Group Theory Models

### 群的定义 / Group Definition

**群** 是一个四元组 $(G, \cdot, e, ^{-1})$，其中：

- $G$: 非空集合
- $\cdot$: 二元运算 $G \times G \to G$
- $e$: 单位元
- $^{-1}$: 逆元运算 $G \to G$

**群公理**:

1. **结合律**: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
2. **单位元**: $e \cdot a = a \cdot e = a$
3. **逆元**: $a \cdot a^{-1} = a^{-1} \cdot a = e$

### 子群 / Subgroups

**子群**: $H \subseteq G$ 是群，如果：

- $e \in H$
- $a, b \in H \Rightarrow a \cdot b \in H$
- $a \in H \Rightarrow a^{-1} \in H$

**拉格朗日定理**: $|H|$ 整除 $|G|$

### 同态与同构 / Homomorphisms and Isomorphisms

**群同态**: $\phi: G \to H$ 满足 $\phi(a \cdot b) = \phi(a) \cdot \phi(b)$

**核**: $\ker(\phi) = \{g \in G : \phi(g) = e_H\}$

**像**: $\text{im}(\phi) = \{\phi(g) : g \in G\}$

**第一同构定理**: $G/\ker(\phi) \cong \text{im}(\phi)$

### 重要群类 / Important Group Classes

**循环群**: $G = \langle g \rangle = \{g^n : n \in \mathbb{Z}\}$

**对称群**: $S_n$ 是 $n$ 个元素的置换群

**阿贝尔群**: 满足交换律 $a \cdot b = b \cdot a$

---

## 3.1.2 环论模型 / Ring Theory Models

### 环的定义 / Ring Definition

**环** 是一个六元组 $(R, +, \cdot, 0, 1, -)$，其中：

- $(R, +, 0, -)$ 是阿贝尔群
- $(R, \cdot, 1)$ 是幺半群
- **分配律**: $a \cdot (b + c) = a \cdot b + a \cdot c$

### 理想 / Ideals

**左理想**: $I \subseteq R$ 满足：

- $I$ 是加法子群
- $r \in R, a \in I \Rightarrow ra \in I$

**主理想**: $(a) = \{ra : r \in R\}$

**极大理想**: 没有真包含它的理想

### 商环 / Quotient Rings

**商环**: $R/I = \{r + I : r \in R\}$

**运算**: $(r + I) + (s + I) = (r + s) + I$

**第二同构定理**: $(R/I)/(J/I) \cong R/J$

---

## 3.1.3 域论模型 / Field Theory Models

### 域的定义 / Field Definition

**域** 是一个环 $(F, +, \cdot, 0, 1, -)$，其中：

- $(F \setminus \{0\}, \cdot, 1, ^{-1})$ 是阿贝尔群
- $0 \neq 1$

### 域扩张 / Field Extensions

**域扩张**: $K/F$ 表示 $F \subseteq K$

**代数扩张**: $\alpha \in K$ 是 $F$ 上的代数元

**超越扩张**: $\alpha \in K$ 是 $F$ 上的超越元

**有限扩张**: $[K:F] < \infty$

### 伽罗瓦理论 / Galois Theory

**伽罗瓦群**: $\text{Gal}(K/F) = \{\sigma: K \to K : \sigma|_F = \text{id}\}$

**基本定理**: 在有限伽罗瓦扩张中，子群与中间域一一对应

---

### 算法实现：`GF(p)[x]`多项式与不可约性 / Algorithms: Polynomials over GF(p)

```python
from typing import List, Optional

def poly_trim(a: List[int], p: int) -> List[int]:
    while len(a) > 1 and a[-1] % p == 0:
        a.pop()
    return [x % p for x in a]

def poly_add(a: List[int], b: List[int], p: int) -> List[int]:
    n = max(len(a), len(b))
    c = [(0) for _ in range(n)]
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        c[i] = (ai + bi) % p
    return poly_trim(c, p)

def poly_sub(a: List[int], b: List[int], p: int) -> List[int]:
    n = max(len(a), len(b))
    c = [(0) for _ in range(n)]
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        c[i] = (ai - bi) % p
    return poly_trim(c, p)

def poly_mul(a: List[int], b: List[int], p: int) -> List[int]:
    c = [0]*(len(a)+len(b)-1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            c[i+j] = (c[i+j] + ai*bj) % p
    return poly_trim(c, p)

def modinv_int(a: int, p: int) -> Optional[int]:
    # 扩展欧几里得
    def eg(a: int, b: int):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = eg(b, a % b)
        return g, y1, x1 - (a//b)*y1
    g, x, _ = eg(a % p, p)
    if g != 1:
        return None
    return x % p

def poly_divmod(a: List[int], b: List[int], p: int):
    a = a[:]
    b = poly_trim(b[:], p)
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError
    inv_lead = modinv_int(b[-1], p)
    q = [0]*(max(1, len(a)-len(b)+1))
    r = a
    while len(r) >= len(b) and not (len(r) == 1 and r[0] == 0):
        k = len(r) - len(b)
        coef = (r[-1] * inv_lead) % p
        q[k] = coef
        sub = [0]*k + [(coef * bj) % p for bj in b]
        r = poly_sub(r, sub, p)
    return poly_trim(q, p), poly_trim(r, p)

def poly_gcd(a: List[int], b: List[int], p: int) -> List[int]:
    a = poly_trim(a[:], p)
    b = poly_trim(b[:], p)
    while not (len(b) == 1 and b[0] == 0):
        _, r = poly_divmod(a, b, p)
        a, b = b, r
    # 归一化首项为1
    if not (len(a) == 1 and a[0] == 0):
        inv = modinv_int(a[-1], p)
        a = [(ai*inv) % p for ai in a]
    return poly_trim(a, p)

def poly_pow_mod(x: List[int], e: int, mod_poly: List[int], p: int) -> List[int]:
    res = [1]
    base = x[:]
    e0 = e
    while e0 > 0:
        if e0 & 1:
            res = poly_mul(res, base, p)
            _, res = poly_divmod(res, mod_poly, p)
        base = poly_mul(base, base, p)
        _, base = poly_divmod(base, mod_poly, p)
        e0 >>= 1
    return res

def is_irreducible(f: List[int], p: int) -> bool:
    """Rabin测试：GF(p)上不可约性（p为素数），f为首一多项式系数列表（低位在前）。"""
    f = poly_trim(f[:], p)
    n = len(f) - 1
    if n <= 0:
        return False
    # x^(p^k) mod f - x 的gcd检查
    x = [0, 1]
    power = x[:]
    for _ in range(1, n//2 + 1):
        # 计算 power = x^{p} (Frobenius) 可用重复平方实现：x -> x^p mod f
        # 这里简化：逐次做p次平方与乘法（对小p可行）
        for _ in range(p-1):
            power = poly_mul(power, x, p)
            _, power = poly_divmod(power, f, p)
        g = poly_gcd(poly_sub(power, x, p), f, p)
        if len(g) > 1:
            return False
    return True
```

### 算法实现：多项式扩展欧几里得与GF(p^m) / Algorithms: Poly xgcd and GF(p^m)

```python
from typing import Tuple, List

def poly_scalar_mul(a: List[int], c: int, p: int) -> List[int]:
    return [(ai * c) % p for ai in a]

def poly_xgcd(a: List[int], b: List[int], p: int) -> Tuple[List[int], List[int], List[int]]:
    """返回(g, s, t) 使得 g = gcd(a,b), s*a + t*b = g（首一或零）"""
    a = poly_trim(a[:], p)
    b = poly_trim(b[:], p)
    if len(b) == 1 and b[0] == 0:
        if not (len(a) == 1 and a[0] == 0):
            inv = modinv_int(a[-1], p)
            a = poly_scalar_mul(a, inv, p)
        return a, [1], [0]
    s0, s1 = [1], [0]
    t0, t1 = [0], [1]
    r0, r1 = a, b
    while not (len(r1) == 1 and r1[0] == 0):
        q, r2 = poly_divmod(r0, r1, p)
        r0, r1 = r1, r2
        s0, s1 = s1, poly_sub(s0, poly_mul(q, s1, p), p)
        t0, t1 = t1, poly_sub(t0, poly_mul(q, t1, p), p)
    if not (len(r0) == 1 and r0[0] == 0):
        inv = modinv_int(r0[-1], p)
        r0 = poly_scalar_mul(r0, inv, p)
        s0 = poly_scalar_mul(s0, inv, p)
        t0 = poly_scalar_mul(t0, inv, p)
    return poly_trim(r0, p), poly_trim(s0, p), poly_trim(t0, p)

class GFpm:
    """GF(p^m) 有限域，表示为GF(p)[x]/(modulus)。元素为低位在前的系数列表。"""
    def __init__(self, p: int, modulus: List[int]):
        if p <= 1:
            raise ValueError("p must be prime (not checked)")
        f = poly_trim(modulus[:], p)
        if len(f) <= 1:
            raise ValueError("modulus degree must be >= 1")
        inv_lead = modinv_int(f[-1], p)
        self.modulus = poly_scalar_mul(f, inv_lead, p)
        self.p = p
        self.m = len(self.modulus) - 1

    def add(self, a: List[int], b: List[int]) -> List[int]:
        return poly_trim(poly_add(a, b, self.p), self.p)

    def neg(self, a: List[int]) -> List[int]:
        return [(self.p - (ai % self.p)) % self.p for ai in a]

    def sub(self, a: List[int], b: List[int]) -> List[int]:
        return poly_trim(poly_sub(a, b, self.p), self.p)

    def mul(self, a: List[int], b: List[int]) -> List[int]:
        prod = poly_mul(a, b, self.p)
        _, r = poly_divmod(prod, self.modulus, self.p)
        return r

    def inv(self, a: List[int]) -> List[int]:
        g, s, _ = poly_xgcd(poly_trim(a, self.p), self.modulus, self.p)
        if not (len(g) == 1 and g[0] == 1):
            raise ZeroDivisionError("non-invertible element")
        _, r = poly_divmod(s, self.modulus, self.p)
        return r

    def pow(self, a: List[int], e: int) -> List[int]:
        res = [1]
        base = poly_trim(a, self.p)
        ee = e
        while ee > 0:
            if ee & 1:
                res = self.mul(res, base)
            base = self.mul(base, base)
            ee >>= 1
        return res

    def element(self, coeffs: List[int]) -> List[int]:
        _, r = poly_divmod(poly_trim(coeffs, self.p), self.modulus, self.p)
        return r
```

## 3.1.4 线性代数模型 / Linear Algebra Models

### 向量空间 / Vector Spaces

**向量空间**: $(V, +, \cdot)$ 满足：

- $(V, +)$ 是阿贝尔群
- **标量乘法**: $\mathbb{F} \times V \to V$
- **分配律**: $a(v + w) = av + aw$
- **结合律**: $(ab)v = a(bv)$

### 线性变换 / Linear Transformations

**线性变换**: $T: V \to W$ 满足：

- $T(v + w) = T(v) + T(w)$
- $T(av) = aT(v)$

**矩阵表示**: $[T]_\beta^\gamma$ 是 $T$ 在基 $\beta, \gamma$ 下的矩阵

### 特征值与特征向量 / Eigenvalues and Eigenvectors

**特征值**: $\lambda$ 满足 $T(v) = \lambda v$

**特征多项式**: $p_T(\lambda) = \det(T - \lambda I)$

**对角化**: $A = PDP^{-1}$ 其中 $D$ 是对角矩阵

---

### 算法实现：秩与零空间 / Algorithms: Rank and Null Space

```python
import numpy as np
from typing import Tuple

def matrix_rank(A: np.ndarray, tol: float = 1e-12) -> int:
    """使用奇异值分解估计矩阵秩"""
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

def null_space(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """计算零空间的一个正交基（列向量构成）"""
    U, s, Vt = np.linalg.svd(A)
    rank = np.sum(s > tol)
    return Vt[rank:].T

def rank_nullity_check(A: np.ndarray) -> Tuple[int, int]:
    r = matrix_rank(A)
    n = A.shape[1]
    ns_dim = n - r
    return r, ns_dim
```

## 3.1.5 同调代数模型 / Homological Algebra Models

### 链复形 / Chain Complexes

**链复形**: $\cdots \to C_{n+1} \xrightarrow{d_{n+1}} C_n \xrightarrow{d_n} C_{n-1} \to \cdots$

**边界条件**: $d_n \circ d_{n+1} = 0$

**同调群**: $H_n(C) = \ker(d_n)/\text{im}(d_{n+1})$

### 上链复形 / Cochain Complexes

**上链复形**: $\cdots \to C^{n-1} \xrightarrow{d^{n-1}} C^n \xrightarrow{d^n} C^{n+1} \to \cdots$

**上同调群**: $H^n(C) = \ker(d^n)/\text{im}(d^{n-1})$

### 导出函子 / Derived Functors

**左导出函子**: $L_nF(A) = H_n(F(P))$ 其中 $P$ 是 $A$ 的投射分解

**右导出函子**: $R^nF(A) = H^n(F(I))$ 其中 $I$ 是 $A$ 的内射分解

---

## 3.1.6 表示论模型 / Representation Theory Models

### 群表示 / Group Representations

**表示**: $\rho: G \to GL(V)$ 是群同态

**不可约表示**: 没有非平凡不变子空间

**特征**: $\chi_\rho(g) = \text{tr}(\rho(g))$

### 特征理论 / Character Theory

**正交关系**: $\langle \chi_i, \chi_j \rangle = \frac{1}{|G|} \sum_{g \in G} \chi_i(g) \overline{\chi_j(g)}$

**不可约特征**: 完全正交系

**特征表**: 不可约特征值表

---

## 3.1.7 代数几何模型 / Algebraic Geometry Models

### 仿射代数集 / Affine Algebraic Sets

**仿射空间**: $\mathbb{A}^n = \{(a_1, \ldots, a_n) : a_i \in k\}$

**代数集**: $V(I) = \{P \in \mathbb{A}^n : f(P) = 0 \text{ for all } f \in I\}$

**理想**: $I(V) = \{f \in k[x_1, \ldots, x_n] : f(P) = 0 \text{ for all } P \in V\}$

### 射影代数集 / Projective Algebraic Sets

**射影空间**: $\mathbb{P}^n = \{(a_0 : \cdots : a_n) : a_i \in k, \text{ not all zero}\}$

**齐次理想**: $I$ 由齐次多项式生成

**射影代数集**: $V(I) = \{P \in \mathbb{P}^n : f(P) = 0 \text{ for all } f \in I\}$

---

## 3.1.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Group {
    pub elements: Vec<String>,
    pub operation: HashMap<(String, String), String>,
    pub identity: String,
    pub inverses: HashMap<String, String>,
}

impl Group {
    pub fn new(elements: Vec<String>, identity: String) -> Self {
        Self {
            elements,
            operation: HashMap::new(),
            identity,
            inverses: HashMap::new(),
        }
    }
    
    pub fn set_operation(&mut self, a: String, b: String, result: String) {
        self.operation.insert((a, b), result);
    }
    
    pub fn set_inverse(&mut self, element: String, inverse: String) {
        self.inverses.insert(element, inverse);
    }
    
    pub fn multiply(&self, a: &str, b: &str) -> Option<&str> {
        self.operation.get(&(a.to_string(), b.to_string())).map(|s| s.as_str())
    }
    
    pub fn inverse(&self, element: &str) -> Option<&str> {
        self.inverses.get(element).map(|s| s.as_str())
    }
    
    pub fn is_group(&self) -> bool {
        // 检查结合律
        for a in &self.elements {
            for b in &self.elements {
                for c in &self.elements {
                    let ab = self.multiply(a, b);
                    let bc = self.multiply(b, c);
                    
                    if let (Some(ab), Some(bc)) = (ab, bc) {
                        let (ab)c = self.multiply(ab, c);
                        let a(bc) = self.multiply(a, bc);
                        
                        if ab_c != a_bc {
                            return false;
                        }
                    }
                }
            }
        }
        
        // 检查单位元
        for a in &self.elements {
            if self.multiply(&self.identity, a) != Some(a.as_str()) ||
               self.multiply(a, &self.identity) != Some(a.as_str()) {
                return false;
            }
        }
        
        // 检查逆元
        for a in &self.elements {
            if let Some(inv) = self.inverse(a) {
                if self.multiply(a, inv) != Some(&self.identity) ||
                   self.multiply(inv, a) != Some(&self.identity) {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
}

#[derive(Debug)]
pub struct Ring {
    pub additive_group: Group,
    pub multiplication: HashMap<(String, String), String>,
    pub multiplicative_identity: String,
}

impl Ring {
    pub fn new(elements: Vec<String>, additive_identity: String, multiplicative_identity: String) -> Self {
        let mut additive_group = Group::new(elements.clone(), additive_identity.clone());
        
        // 设置加法运算
        for i in 0..elements.len() {
            for j in 0..elements.len() {
                let result = format!("sum_{}_{}", i, j);
                additive_group.set_operation(elements[i].clone(), elements[j].clone(), result);
            }
        }
        
        Self {
            additive_group,
            multiplication: HashMap::new(),
            multiplicative_identity,
        }
    }
    
    pub fn set_multiplication(&mut self, a: String, b: String, result: String) {
        self.multiplication.insert((a, b), result);
    }
    
    pub fn multiply(&self, a: &str, b: &str) -> Option<&str> {
        self.multiplication.get(&(a.to_string(), b.to_string())).map(|s| s.as_str())
    }
    
    pub fn is_ring(&self) -> bool {
        // 检查加法群
        if !self.additive_group.is_group() {
            return false;
        }
        
        // 检查乘法结合律
        for a in &self.additive_group.elements {
            for b in &self.additive_group.elements {
                for c in &self.additive_group.elements {
                    let ab = self.multiply(a, b);
                    let bc = self.multiply(b, c);
                    
                    if let (Some(ab), Some(bc)) = (ab, bc) {
                        let (ab)c = self.multiply(ab, c);
                        let a(bc) = self.multiply(a, bc);
                        
                        if ab_c != a_bc {
                            return false;
                        }
                    }
                }
            }
        }
        
        // 检查分配律
        for a in &self.additive_group.elements {
            for b in &self.additive_group.elements {
                for c in &self.additive_group.elements {
                    let b_plus_c = self.additive_group.multiply(b, c);
                    let a_times_b_plus_c = self.multiply(a, b_plus_c.unwrap_or(""));
                    
                    let a_times_b = self.multiply(a, b);
                    let a_times_c = self.multiply(a, c);
                    let a_times_b_plus_a_times_c = self.additive_group.multiply(
                        a_times_b.unwrap_or(""), 
                        a_times_c.unwrap_or("")
                    );
                    
                    if a_times_b_plus_c != a_times_b_plus_a_times_c {
                        return false;
                    }
                }
            }
        }
        
        true
    }
}

// 使用示例：创建循环群
fn main() {
    let mut cyclic_group = Group::new(
        vec!["e".to_string(), "a".to_string(), "a²".to_string()],
        "e".to_string()
    );
    
    // 设置运算表 (Z₃)
    cyclic_group.set_operation("e".to_string(), "e".to_string(), "e".to_string());
    cyclic_group.set_operation("e".to_string(), "a".to_string(), "a".to_string());
    cyclic_group.set_operation("e".to_string(), "a²".to_string(), "a²".to_string());
    cyclic_group.set_operation("a".to_string(), "e".to_string(), "a".to_string());
    cyclic_group.set_operation("a".to_string(), "a".to_string(), "a²".to_string());
    cyclic_group.set_operation("a".to_string(), "a²".to_string(), "e".to_string());
    cyclic_group.set_operation("a²".to_string(), "e".to_string(), "a²".to_string());
    cyclic_group.set_operation("a²".to_string(), "a".to_string(), "e".to_string());
    cyclic_group.set_operation("a²".to_string(), "a²".to_string(), "a".to_string());
    
    // 设置逆元
    cyclic_group.set_inverse("e".to_string(), "e".to_string());
    cyclic_group.set_inverse("a".to_string(), "a²".to_string());
    cyclic_group.set_inverse("a²".to_string(), "a".to_string());
    
    println!("Is cyclic group: {}", cyclic_group.is_group());
    
    // 测试运算
    if let Some(result) = cyclic_group.multiply("a", "a²") {
        println!("a * a² = {}", result);
    }
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module AlgebraicModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (all, any)

-- 群的数据类型
data Group = Group {
    elements :: [String],
    operation :: Map (String, String) String,
    identity :: String,
    inverses :: Map String String
} deriving (Show)

-- 创建群
newGroup :: [String] -> String -> Group
newGroup elems ident = Group {
    elements = elems,
    operation = Map.empty,
    identity = ident,
    inverses = Map.empty
}

-- 设置运算
setOperation :: Group -> String -> String -> String -> Group
setOperation g a b result = g { operation = Map.insert (a, b) result (operation g) }

-- 设置逆元
setInverse :: Group -> String -> String -> Group
setInverse g element inv = g { inverses = Map.insert element inv (inverses g) }

-- 群运算
multiply :: Group -> String -> String -> Maybe String
multiply g a b = Map.lookup (a, b) (operation g)

-- 获取逆元
inverse :: Group -> String -> Maybe String
inverse g element = Map.lookup element (inverses g)

-- 检查群公理
isGroup :: Group -> Bool
isGroup g = associativity && identity_law && inverse_law
  where
    associativity = all (\a -> all (\b -> all (\c -> 
        case (multiply g a b, multiply g b c) of
            (Just ab, Just bc) -> 
                case (multiply g ab c, multiply g a bc) of
                    (Just ab_c, Just a_bc) -> ab_c == a_bc
                    _ -> False
            _ -> True) (elements g)) (elements g)) (elements g)
    
    identity_law = all (\a -> 
        multiply g (identity g) a == Just a && 
        multiply g a (identity g) == Just a) (elements g)
    
    inverse_law = all (\a -> 
        case inverse g a of
            Just inv -> multiply g a inv == Just (identity g) && 
                       multiply g inv a == Just (identity g)
            Nothing -> False) (elements g)

-- 环的数据类型
data Ring = Ring {
    additiveGroup :: Group,
    multiplication :: Map (String, String) String,
    multiplicativeIdentity :: String
} deriving (Show)

-- 创建环
newRing :: [String] -> String -> String -> Ring
newRing elems addIdent multIdent = Ring {
    additiveGroup = newGroup elems addIdent,
    multiplication = Map.empty,
    multiplicativeIdentity = multIdent
}

-- 设置乘法
setMultiplication :: Ring -> String -> String -> String -> Ring
setMultiplication r a b result = r { multiplication = Map.insert (a, b) result (multiplication r) }

-- 环乘法
ringMultiply :: Ring -> String -> String -> Maybe String
ringMultiply r a b = Map.lookup (a, b) (multiplication r)

-- 检查环公理
isRing :: Ring -> Bool
isRing r = isGroup (additiveGroup r) && associativity && distributivity
  where
    associativity = all (\a -> all (\b -> all (\c -> 
        case (ringMultiply r a b, ringMultiply r b c) of
            (Just ab, Just bc) -> 
                case (ringMultiply r ab c, ringMultiply r a bc) of
                    (Just ab_c, Just a_bc) -> ab_c == a_bc
                    _ -> False
            _ -> True) (elements (additiveGroup r))) (elements (additiveGroup r))) (elements (additiveGroup r))
    
    distributivity = all (\a -> all (\b -> all (\c -> 
        case multiply (additiveGroup r) b c of
            Just b_plus_c -> 
                case (ringMultiply r a b_plus_c, ringMultiply r a b, ringMultiply r a c) of
                    (Just a_times_b_plus_c, Just a_times_b, Just a_times_c) ->
                        case multiply (additiveGroup r) a_times_b a_times_c of
                            Just sum_result -> a_times_b_plus_c == sum_result
                            Nothing -> False
                    _ -> False
            Nothing -> True) (elements (additiveGroup r))) (elements (additiveGroup r))) (elements (additiveGroup r))

-- 示例：创建循环群 Z₃
example :: IO ()
example = do
    let g = newGroup ["e", "a", "a²"] "e"
        
        -- 设置运算表
        g1 = setOperation g "e" "e" "e"
        g2 = setOperation g1 "e" "a" "a"
        g3 = setOperation g2 "e" "a²" "a²"
        g4 = setOperation g3 "a" "e" "a"
        g5 = setOperation g4 "a" "a" "a²"
        g6 = setOperation g5 "a" "a²" "e"
        g7 = setOperation g6 "a²" "e" "a²"
        g8 = setOperation g7 "a²" "a" "e"
        g9 = setOperation g8 "a²" "a²" "a"
        
        -- 设置逆元
        g10 = setInverse g9 "e" "e"
        g11 = setInverse g10 "a" "a²"
        g12 = setInverse g11 "a²" "a"
        
        finalGroup = g12
    
    putStrLn $ "Is group: " ++ show (isGroup finalGroup)
    
    case multiply finalGroup "a" "a²" of
        Just result -> putStrLn $ "a * a² = " ++ result
        Nothing -> putStrLn "Operation not defined"
```

### Python实现示例 / Python Implementation Example

```python
from typing import Dict, Tuple, Optional, List

class FiniteGroup:
    def __init__(self, elements: List[str], identity: str,
                 op: Dict[Tuple[str, str], str], inv: Dict[str, str]):
        self.elements = elements
        self.identity = identity
        self.op = op
        self.inv = inv

    def multiply(self, a: str, b: str) -> Optional[str]:
        return self.op.get((a, b))

    def inverse(self, a: str) -> Optional[str]:
        return self.inv.get(a)

    def is_group(self) -> bool:
        # 闭包与运算定义完整性
        for a in self.elements:
            for b in self.elements:
                if (a, b) not in self.op:
                    return False
                if self.op[(a, b)] not in self.elements:
                    return False
        # 结合律
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    ab = self.multiply(a, b)
                    bc = self.multiply(b, c)
                    if ab is None or bc is None:
                        return False
                    ab_c = self.multiply(ab, c)
                    a_bc = self.multiply(a, bc)
                    if ab_c != a_bc:
                        return False
        # 单位元
        e = self.identity
        if e not in self.elements:
            return False
        for a in self.elements:
            if self.multiply(e, a) != a or self.multiply(a, e) != a:
                return False
        # 逆元
        for a in self.elements:
            inv_a = self.inverse(a)
            if inv_a is None:
                return False
            if self.multiply(a, inv_a) != e or self.multiply(inv_a, a) != e:
                return False
        return True

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """扩展欧几里得算法: 返回(g, x, y) 使得 ax + by = g = gcd(a, b)"""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a: int, p: int) -> Optional[int]:
    """模逆 a^{-1} mod p（p为素数或当gcd(a,p)=1时存在）"""
    g, x, _ = egcd(a % p, p)
    if g != 1:
        return None
    return x % p

class FiniteField:
    def __init__(self, p: int):
        if p <= 1:
            raise ValueError("p must be prime (not checked here)")
        self.p = p

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def inv(self, a: int) -> Optional[int]:
        return modinv(a, self.p)

    def div(self, a: int, b: int) -> Optional[int]:
        inv_b = self.inv(b)
        if inv_b is None:
            return None
        return self.mul(a, inv_b)

# 示例：Z_3循环群与有限域GF(7)
def demo_algebra():
    # Z_3群
    elems = ["0", "1", "2"]
    op = {}
    inv = {"0": "0", "1": "2", "2": "1"}
    for a in elems:
        for b in elems:
            op[(a, b)] = str((int(a) + int(b)) % 3)
    G = FiniteGroup(elems, "0", op, inv)

    # 检查群公理
    ok_group = G.is_group()

    # GF(7)
    F7 = FiniteField(7)
    inv3 = F7.inv(3)
    seven_demo = {
        "sum_3_5": F7.add(3, 5),
        "prod_3_5": F7.mul(3, 5),
        "inv_3": inv3,
        "div_4_by_3": F7.div(4, 3)
    }
    return {"is_Z3_group": ok_group, "GF7_demo": seven_demo}
```

### 应用领域 / Application Domains

#### 密码学 / Cryptography

- **RSA算法**: 基于大整数分解
- **椭圆曲线密码**: 基于椭圆曲线群
- **格密码**: 基于格理论

#### 编码理论 / Coding Theory

- **线性码**: 向量空间的子空间
- **循环码**: 基于多项式环
- **BCH码**: 基于有限域

#### 量子计算 / Quantum Computing

- **量子群**: 非交换代数结构
- **量子表示论**: 量子力学中的对称性
- **拓扑量子场论**: 代数拓扑应用

---

## 参考文献 / References

1. Dummit, D. S., & Foote, R. M. (2004). Abstract Algebra. Wiley.
2. Lang, S. (2002). Algebra. Springer.
3. Hungerford, T. W. (1974). Algebra. Springer.
4. Artin, M. (1991). Algebra. Prentice Hall.

---

*最后更新: 2025-08-26*
*版本: 1.3.0*
