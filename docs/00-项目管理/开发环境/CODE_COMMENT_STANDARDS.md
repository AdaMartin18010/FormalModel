# 代码注释和文档标准 / Code Comment and Documentation Standards

## 概述 / Overview

本文档建立了FormalModel项目的代码注释和文档标准，确保所有代码示例具有清晰的注释和完整的文档，提升代码的可读性和可维护性。

## 一、Python代码注释标准 / Python Code Comment Standards

### 1.1 函数文档字符串 / Function Docstrings

所有函数都应包含完整的文档字符串，使用Google风格或NumPy风格：

**Google风格示例**:

```python
def newton_second_law(mass: float, acceleration: np.ndarray) -> np.ndarray:
    """
    计算牛顿第二定律的力。

    根据牛顿第二定律 F = ma，计算作用在物体上的力。

    参数:
        mass: 物体质量 (kg)
        acceleration: 加速度向量 (m/s²)

    返回:
        力向量 (N)

    示例:
        >>> mass = 2.0
        >>> acceleration = np.array([3.0, 4.0, 0.0])
        >>> force = newton_second_law(mass, acceleration)
        >>> np.allclose(force, np.array([6.0, 8.0, 0.0]))
        True

    注意:
        - 质量必须为正数
        - 加速度向量应为3维
    """
    if mass <= 0:
        raise ValueError("质量必须为正数")
    if len(acceleration) != 3:
        raise ValueError("加速度向量应为3维")
    return mass * acceleration
```

**NumPy风格示例**:

```python
def calculate_gravitational_force(mass1: float, mass2: float,
                                  distance: float) -> float:
    """
    计算两个物体之间的万有引力。

    参数
    ----------
    mass1 : float
        第一个物体的质量 (kg)
    mass2 : float
        第二个物体的质量 (kg)
    distance : float
        两个物体之间的距离 (m)

    返回
    -------
    float
        万有引力大小 (N)

    示例
    -------
    >>> G = 6.674e-11
    >>> force = calculate_gravitational_force(1e3, 1e3, 1.0)
    >>> abs(force - G * 1e6) < 1e-10
    True

    注意
    -------
    使用标准引力常数 G = 6.674 × 10⁻¹¹ N⋅m²/kg²
    """
    G = 6.674e-11  # 引力常数 (N⋅m²/kg²)
    return G * mass1 * mass2 / (distance ** 2)
```

### 1.2 类文档字符串 / Class Docstrings

所有类都应包含完整的文档字符串：

```python
class ClassicalParticle:
    """
    经典粒子类，用于表示经典力学中的粒子。

    该类实现了经典力学中粒子的基本属性（质量、位置、速度）和
    基本操作（运动、能量计算等）。

    属性
    ----------
    mass : float
        粒子质量 (kg)
    position : np.ndarray
        位置向量 (m)
    velocity : np.ndarray
        速度向量 (m/s)

    示例
    -------
    >>> particle = ClassicalParticle(mass=1.0, position=np.array([0, 0, 0]))
    >>> particle.position
    array([0., 0., 0.])
    >>> particle.kinetic_energy()
    0.0
    """

    def __init__(self, mass: float, position: np.ndarray,
                 velocity: np.ndarray = None):
        """
        初始化经典粒子。

        参数
        ----------
        mass : float
            粒子质量 (kg)，必须为正数
        position : np.ndarray
            初始位置向量 (m)，应为3维向量
        velocity : np.ndarray, optional
            初始速度向量 (m/s)，默认为零向量
        """
        if mass <= 0:
            raise ValueError("质量必须为正数")
        if len(position) != 3:
            raise ValueError("位置向量应为3维")

        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros(3)
```

### 1.3 行内注释 / Inline Comments

**好的注释示例**:

```python
# 计算动能: E_k = (1/2)mv²
kinetic_energy = 0.5 * mass * np.dot(velocity, velocity)

# 使用Verlet算法进行时间积分
# 位置更新: r(t+Δt) = 2r(t) - r(t-Δt) + a(t)Δt²
new_position = 2 * current_position - old_position + acceleration * dt_squared

# 检查能量守恒: |E(t) - E₀| < ε
energy_conserved = abs(current_energy - initial_energy) < tolerance
```

**避免的注释**:

```python
# 不好的注释：只是重复代码
mass = 10.0  # 设置质量为10.0

# 不好的注释：没有提供额外信息
x = x + 1  # 增加x
```

### 1.4 复杂算法注释 / Complex Algorithm Comments

对于复杂算法，应提供详细的步骤说明：

```python
def solve_kepler_equation(eccentricity: float, mean_anomaly: float,
                          tolerance: float = 1e-10) -> float:
    """
    使用牛顿法求解开普勒方程: M = E - e sin(E)

    算法步骤:
    1. 初始化偏近点角 E₀ = M
    2. 迭代计算: E_{n+1} = E_n - (E_n - e sin(E_n) - M) / (1 - e cos(E_n))
    3. 当 |E_{n+1} - E_n| < tolerance 时停止

    参数
    ----------
    eccentricity : float
        轨道离心率 (0 ≤ e < 1)
    mean_anomaly : float
        平近点角 (rad)
    tolerance : float, optional
        收敛容差，默认为 1e-10

    返回
    -------
    float
        偏近点角 E (rad)

    注意
    -------
    该方法仅适用于椭圆轨道 (e < 1)
    """
    if not (0 <= eccentricity < 1):
        raise ValueError("离心率必须在 [0, 1) 范围内")

    # 步骤1: 初始化
    E = mean_anomaly

    # 步骤2: 迭代求解
    for iteration in range(100):  # 最大迭代次数
        # 计算函数值和导数值
        f = E - eccentricity * np.sin(E) - mean_anomaly
        f_prime = 1 - eccentricity * np.cos(E)

        # 牛顿法更新
        E_new = E - f / f_prime

        # 步骤3: 检查收敛
        if abs(E_new - E) < tolerance:
            return E_new

        E = E_new

    raise RuntimeError("牛顿法未收敛")
```

## 二、代码示例文档标准 / Code Example Documentation Standards

### 2.1 代码块格式 / Code Block Format

所有代码块应包含：

1. 语言标识符
2. 简要说明
3. 完整的注释

```markdown
#### 算法实现 / Algorithm Implementation

```python
# 牛顿第二定律实现
# 根据 F = ma 计算力

import numpy as np

def newton_second_law(mass: float, acceleration: np.ndarray) -> np.ndarray:
    """
    计算牛顿第二定律的力。

    参数:
        mass: 物体质量 (kg)
        acceleration: 加速度向量 (m/s²)

    返回:
        力向量 (N)
    """
    return mass * acceleration

# 示例使用
if __name__ == "__main__":
    mass = 2.0  # kg
    acceleration = np.array([3.0, 4.0, 0.0])  # m/s²
    force = newton_second_law(mass, acceleration)
    print(f"力向量: {force} N")
```

```

### 2.2 参数说明 / Parameter Documentation

所有参数应包含：
- 类型说明
- 单位（如适用）
- 取值范围或约束条件

```python
def calculate_orbit_period(semi_major_axis: float, central_mass: float) -> float:
    """
    计算轨道周期（开普勒第三定律）。

    参数:
        semi_major_axis: 轨道半长轴 (m)，必须为正数
        central_mass: 中心天体质量 (kg)，必须为正数

    返回:
        轨道周期 (s)

    公式:
        T = 2π √(a³ / GM)

    其中:
        T: 轨道周期
        a: 半长轴
        G: 引力常数
        M: 中心天体质量
    """
    G = 6.674e-11  # 引力常数 (N⋅m²/kg²)
    return 2 * np.pi * np.sqrt(semi_major_axis ** 3 / (G * central_mass))
```

### 2.3 示例代码 / Example Code

每个算法实现都应包含使用示例：

```python
def example_newton_second_law():
    """
    牛顿第二定律使用示例。

    演示如何计算作用在物体上的力。
    """
    # 创建测试数据
    mass = 5.0  # kg
    acceleration = np.array([2.0, 0.0, -9.8])  # m/s² (包含重力)

    # 计算力
    force = newton_second_law(mass, acceleration)

    # 输出结果
    print(f"质量: {mass} kg")
    print(f"加速度: {acceleration} m/s²")
    print(f"力: {force} N")

    # 验证: F = ma
    expected_force = mass * acceleration
    assert np.allclose(force, expected_force), "计算结果不正确"

    return {
        "mass": mass,
        "acceleration": acceleration,
        "force": force
    }
```

## 三、数学公式注释标准 / Mathematical Formula Comment Standards

### 3.1 公式对应关系 / Formula Correspondence

代码中的数学公式应明确标注对应的数学表达式：

```python
def lagrangian_mechanics(kinetic_energy: float, potential_energy: float) -> float:
    """
    计算拉格朗日量。

    数学公式:
        L = T - V

    其中:
        L: 拉格朗日量
        T: 动能
        V: 势能

    参数:
        kinetic_energy: 动能 T (J)
        potential_energy: 势能 V (J)

    返回:
        拉格朗日量 L (J)
    """
    # L = T - V
    return kinetic_energy - potential_energy
```

### 3.2 物理量单位 / Physical Quantity Units

所有物理量应明确标注单位：

```python
def calculate_gravitational_potential_energy(mass1: float, mass2: float,
                                           distance: float) -> float:
    """
    计算引力势能。

    公式: V = -G m₁ m₂ / r

    参数:
        mass1: 第一个物体质量 (kg)
        mass2: 第二个物体质量 (kg)
        distance: 两物体间距离 (m)

    返回:
        引力势能 (J)

    注意:
        引力势能为负值，表示束缚态
    """
    G = 6.674e-11  # 引力常数 (N⋅m²/kg²)
    # V = -G m₁ m₂ / r
    return -G * mass1 * mass2 / distance
```

## 四、错误处理注释标准 / Error Handling Comment Standards

### 4.1 异常处理注释 / Exception Handling Comments

所有异常处理应包含清晰的注释：

```python
def calculate_orbital_velocity(central_mass: float, distance: float) -> float:
    """
    计算轨道速度。

    参数:
        central_mass: 中心天体质量 (kg)
        distance: 轨道半径 (m)

    返回:
        轨道速度 (m/s)

    异常:
        ValueError: 如果质量或距离非正数
    """
    # 检查输入有效性
    if central_mass <= 0:
        raise ValueError(f"中心天体质量必须为正数，得到: {central_mass}")
    if distance <= 0:
        raise ValueError(f"轨道半径必须为正数，得到: {distance}")

    G = 6.674e-11  # 引力常数 (N⋅m²/kg²)
    # v = √(GM / r)
    return np.sqrt(G * central_mass / distance)
```

## 五、代码组织标准 / Code Organization Standards

### 5.1 导入语句组织 / Import Organization

导入语句应按以下顺序组织：

1. 标准库
2. 第三方库
3. 本地模块

```python
# 标准库导入
import math
from typing import Tuple, Optional

# 第三方库导入
import numpy as np
from scipy.constants import c, G

# 本地模块导入（如果有）
# from .utils import helper_function
```

### 5.2 常量定义 / Constant Definitions

所有物理常量应在文件顶部定义：

```python
"""
经典力学模型实现

包含牛顿力学、拉格朗日力学、哈密顿力学的实现。
"""

import numpy as np
from typing import Tuple

# 物理常量
GRAVITATIONAL_CONSTANT = 6.674e-11  # 引力常数 (N⋅m²/kg²)
SPEED_OF_LIGHT = 2.998e8  # 光速 (m/s)
PLANCK_CONSTANT = 6.626e-34  # 普朗克常数 (J⋅s)
```

## 六、文档字符串模板 / Docstring Templates

### 6.1 函数模板 / Function Template

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    函数简要描述（一行）。

    函数详细描述（多行，可选）。
    可以包含算法的说明、使用场景等。

    参数
    ----------
    param1 : type1
        参数1的说明（包含单位和约束）
    param2 : type2
        参数2的说明（包含单位和约束）

    返回
    -------
    return_type
        返回值的说明（包含单位和含义）

    示例
    -------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0

    注意
    -------
    特殊注意事项或限制条件

    参考
    -------
    相关函数或文档的引用
    """
    pass
```

### 6.2 类模板 / Class Template

```python
class ClassName:
    """
    类的简要描述。

    类的详细描述，包括：
    - 类的用途
    - 主要功能
    - 使用场景

    属性
    ----------
    attribute1 : type1
        属性1的说明
    attribute2 : type2
        属性2的说明

    示例
    -------
    >>> obj = ClassName(param1, param2)
    >>> obj.method()
    result
    """

    def __init__(self, param1: type1, param2: type2):
        """
        初始化类实例。

        参数
        ----------
        param1 : type1
            参数1的说明
        param2 : type2
            参数2的说明
        """
        self.attribute1 = param1
        self.attribute2 = param2
```

## 七、检查清单 / Checklist

在提交代码前，请确保：

- [ ] 所有函数都有完整的文档字符串
- [ ] 所有参数都有类型注解和说明
- [ ] 所有返回值都有说明
- [ ] 复杂算法有步骤说明
- [ ] 数学公式有对应的注释
- [ ] 物理量有单位标注
- [ ] 异常处理有注释说明
- [ ] 代码示例可以运行
- [ ] 示例代码有输出结果

---

*最后更新: 2025-12-XX*
*版本: 1.0.0*
