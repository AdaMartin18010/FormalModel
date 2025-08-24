# 2.5 电磁学模型 / Electromagnetism Models

## 目录 / Table of Contents

- [2.5 电磁学模型 / Electromagnetism Models](#25-电磁学模型--electromagnetism-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [2.5.1 静电学 / Electrostatics](#251-静电学--electrostatics)
    - [库仑定律 / Coulomb's Law](#库仑定律--coulombs-law)
    - [电场 / Electric Field](#电场--electric-field)
    - [电势 / Electric Potential](#电势--electric-potential)
    - [高斯定律 / Gauss's Law](#高斯定律--gausss-law)
    - [磁场 / Magnetic Field](#磁场--magnetic-field)
    - [磁矢势 / Magnetic Vector Potential](#磁矢势--magnetic-vector-potential)
    - [毕奥-萨伐尔定律 / Biot-Savart Law](#毕奥-萨伐尔定律--biot-savart-law)
    - [楞次定律 / Lenz's Law](#楞次定律--lenzs-law)
    - [自感和互感 / Self and Mutual Inductance](#自感和互感--self-and-mutual-inductance)
  - [2.5.4 麦克斯韦方程组 / Maxwell's Equations](#254-麦克斯韦方程组--maxwells-equations)
    - [积分形式 / Integral Form](#积分形式--integral-form)
  - [2.5.5 电磁波 / Electromagnetic Waves](#255-电磁波--electromagnetic-waves)
    - [波动方程 / Wave Equation](#波动方程--wave-equation)
    - [平面波 / Plane Waves](#平面波--plane-waves)
    - [偏振 / Polarization](#偏振--polarization)
    - [多普勒效应 / Doppler Effect](#多普勒效应--doppler-effect)
  - [2.5.6 电磁辐射 / Electromagnetic Radiation](#256-电磁辐射--electromagnetic-radiation)
    - [偶极辐射 / Dipole Radiation](#偶极辐射--dipole-radiation)
    - [天线理论 / Antenna Theory](#天线理论--antenna-theory)
    - [辐射功率 / Radiated Power](#辐射功率--radiated-power)

---

## 2.5.1 静电学 / Electrostatics

### 库仑定律 / Coulomb's Law

**形式化定义**: 库仑定律描述了静止点电荷之间相互作用的定量关系，是静电学的基础定律。

**公理化定义**:
设 $\mathcal{CL} = \langle \mathcal{Q}, \mathcal{R}, \mathcal{F}, \mathcal{K} \rangle$ 为库仑定律系统，其中：

1. **电荷集合**: $\mathcal{Q}$ 为点电荷集合
2. **位置集合**: $\mathcal{R}$ 为三维空间位置集合
3. **力函数**: $\mathcal{F}: \mathcal{Q} \times \mathcal{Q} \times \mathcal{R} \times \mathcal{R} \rightarrow \mathbb{R}^3$ 为库仑力函数
4. **库仑常数**: $\mathcal{K} = \frac{1}{4\pi\epsilon_0}$ 为真空介电常数

**等价定义**:

1. **矢量形式**: $\vec{F} = k_e \frac{q_1 q_2}{r^2} \hat{r}$
2. **分量形式**: $F_i = k_e \frac{q_1 q_2}{r^3} r_i$
3. **势能形式**: $U = k_e \frac{q_1 q_2}{r}$

**形式化定理**:

**定理2.5.1.1 (库仑力性质)**: 库仑力满足牛顿第三定律
$$\vec{F}_{12} = -\vec{F}_{21}$$

**定理2.5.1.2 (库仑力叠加性)**: 多个电荷的库仑力满足叠加原理
$$\vec{F}_i = \sum_{j \neq i} \vec{F}_{ij}$$

**定理2.5.1.3 (库仑力保守性)**: 库仑力是保守力
$$\oint \vec{F} \cdot d\vec{r} = 0$$

**Python算法实现**:

```python
import numpy as np
from typing import List, Tuple, Optional

class PointCharge:
    """点电荷类"""
    def __init__(self, charge: float, position: np.ndarray):
        self.charge = charge
        self.position = np.array(position)
    
    def get_charge(self) -> float:
        """获取电荷量"""
        return self.charge
    
    def get_position(self) -> np.ndarray:
        """获取位置"""
        return self.position.copy()

def coulomb_constant() -> float:
    """
    计算库仑常数
    
    返回:
        库仑常数 (N·m²/C²)
    """
    epsilon_0 = 8.854e-12  # F/m
    return 1.0 / (4 * np.pi * epsilon_0)

def coulomb_force(charge1: float, charge2: float, 
                 position1: np.ndarray, position2: np.ndarray) -> np.ndarray:
    """
    计算库仑力
    
    参数:
        charge1, charge2: 电荷量 (C)
        position1, position2: 位置向量 (m)
    
    返回:
        库仑力向量 (N)
    """
    k_e = coulomb_constant()
    
    # 计算相对位置向量
    r_vec = position2 - position1
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算库仑力
    force_mag = k_e * charge1 * charge2 / (r_mag**2)
    force_vec = force_mag * r_vec / r_mag
    
    return force_vec

def coulomb_potential_energy(charge1: float, charge2: float, 
                           position1: np.ndarray, position2: np.ndarray) -> float:
    """
    计算库仑势能
    
    参数:
        charge1, charge2: 电荷量 (C)
        position1, position2: 位置向量 (m)
    
    返回:
        势能 (J)
    """
    k_e = coulomb_constant()
    
    # 计算距离
    r_vec = position2 - position1
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return float('inf')
    
    # 计算势能
    potential_energy = k_e * charge1 * charge2 / r_mag
    
    return potential_energy

def total_coulomb_force(charges: List[PointCharge], 
                       target_index: int) -> np.ndarray:
    """
    计算作用在目标电荷上的总库仑力
    
    参数:
        charges: 电荷列表
        target_index: 目标电荷索引
    
    返回:
        总库仑力向量 (N)
    """
    if target_index >= len(charges):
        raise ValueError("目标索引超出范围")
    
    total_force = np.zeros(3)
    target_charge = charges[target_index]
    
    for i, charge in enumerate(charges):
        if i != target_index:
            force = coulomb_force(
                target_charge.get_charge(), charge.get_charge(),
                target_charge.get_position(), charge.get_position()
            )
            total_force += force
    
    return total_force

def coulomb_force_verification(charge1: float, charge2: float,
                             position1: np.ndarray, position2: np.ndarray,
                             tolerance: float = 1e-10) -> bool:
    """
    验证库仑力是否满足牛顿第三定律
    
    参数:
        charge1, charge2: 电荷量
        position1, position2: 位置向量
        tolerance: 容差
    
    返回:
        是否满足牛顿第三定律
    """
    force_12 = coulomb_force(charge1, charge2, position1, position2)
    force_21 = coulomb_force(charge2, charge1, position2, position1)
    
    # 检查是否满足牛顿第三定律
    return np.allclose(force_12, -force_21, atol=tolerance)

def coulomb_energy_conservation(charges: List[PointCharge],
                              initial_positions: List[np.ndarray],
                              final_positions: List[np.ndarray],
                              tolerance: float = 1e-6) -> bool:
    """
    验证库仑势能守恒
    
    参数:
        charges: 电荷列表
        initial_positions: 初始位置列表
        final_positions: 最终位置列表
        tolerance: 容差
    
    返回:
        势能是否守恒
    """
    # 计算初始总势能
    initial_energy = 0.0
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            energy = coulomb_potential_energy(
                charges[i].get_charge(), charges[j].get_charge(),
                initial_positions[i], initial_positions[j]
            )
            initial_energy += energy
    
    # 计算最终总势能
    final_energy = 0.0
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            energy = coulomb_potential_energy(
                charges[i].get_charge(), charges[j].get_charge(),
                final_positions[i], final_positions[j]
            )
            final_energy += energy
    
    # 检查能量守恒
    return abs(final_energy - initial_energy) < tolerance

# 示例使用
def coulomb_law_example():
    """库仑定律示例"""
    # 创建点电荷
    charge1 = PointCharge(1e-6, np.array([0.0, 0.0, 0.0]))  # 1μC at origin
    charge2 = PointCharge(-2e-6, np.array([1.0, 0.0, 0.0]))  # -2μC at x=1m
    
    # 计算库仑力
    force = coulomb_force(
        charge1.get_charge(), charge2.get_charge(),
        charge1.get_position(), charge2.get_position()
    )
    
    # 计算势能
    potential_energy = coulomb_potential_energy(
        charge1.get_charge(), charge2.get_charge(),
        charge1.get_position(), charge2.get_position()
    )
    
    # 验证牛顿第三定律
    newton_third_law = coulomb_force_verification(
        charge1.get_charge(), charge2.get_charge(),
        charge1.get_position(), charge2.get_position()
    )
    
    print(f"库仑力: {force} N")
    print(f"势能: {potential_energy:.6e} J")
    print(f"满足牛顿第三定律: {newton_third_law}")
    
    return force, potential_energy, newton_third_law
```

### 电场 / Electric Field

**形式化定义**: 电场是描述电荷周围空间电学性质的矢量场，表示单位正电荷在该点所受的力。

**公理化定义**:
设 $\mathcal{EF} = \langle \mathcal{Q}, \mathcal{R}, \mathcal{E}, \mathcal{F} \rangle$ 为电场系统，其中：

1. **电荷集合**: $\mathcal{Q}$ 为源电荷集合
2. **位置集合**: $\mathcal{R}$ 为空间位置集合
3. **电场函数**: $\mathcal{E}: \mathcal{R} \rightarrow \mathbb{R}^3$ 为电场强度函数
4. **力关系**: $\mathcal{F}: \mathcal{Q} \times \mathcal{E} \rightarrow \mathbb{R}^3$ 为电场力函数

**等价定义**:

1. **点电荷电场**: $\vec{E} = k_e \frac{q}{r^2} \hat{r}$
2. **电场力**: $\vec{F} = q\vec{E}$
3. **电势梯度**: $\vec{E} = -\nabla V$

**形式化定理**:

**定理2.5.1.4 (电场叠加性)**: 多个电荷的电场满足叠加原理
$$\vec{E} = \sum_i \vec{E}_i$$

**定理2.5.1.5 (电场无旋性)**: 静电场是无旋场
$$\nabla \times \vec{E} = 0$$

**定理2.5.1.6 (电场高斯定理)**: 电场通量与电荷关系
$$\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$$

**Python算法实现**:

```python
def electric_field_point_charge(charge: float, 
                              source_position: np.ndarray,
                              field_position: np.ndarray) -> np.ndarray:
    """
    计算点电荷产生的电场
    
    参数:
        charge: 源电荷量 (C)
        source_position: 源电荷位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        电场强度向量 (N/C)
    """
    k_e = coulomb_constant()
    
    # 计算相对位置向量
    r_vec = field_position - source_position
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算电场强度
    field_mag = k_e * charge / (r_mag**2)
    field_vec = field_mag * r_vec / r_mag
    
    return field_vec

def total_electric_field(charges: List[PointCharge], 
                       field_position: np.ndarray) -> np.ndarray:
    """
    计算多个电荷产生的总电场
    
    参数:
        charges: 电荷列表
        field_position: 场点位置 (m)
    
    返回:
        总电场强度向量 (N/C)
    """
    total_field = np.zeros(3)
    
    for charge in charges:
        field = electric_field_point_charge(
            charge.get_charge(),
            charge.get_position(),
            field_position
        )
        total_field += field
    
    return total_field

def electric_field_line_charge(linear_density: float,
                             line_start: np.ndarray,
                             line_end: np.ndarray,
                             field_position: np.ndarray) -> np.ndarray:
    """
    计算线电荷产生的电场
    
    参数:
        linear_density: 线电荷密度 (C/m)
        line_start: 线电荷起点 (m)
        line_end: 线电荷终点 (m)
        field_position: 场点位置 (m)
    
    返回:
        电场强度向量 (N/C)
    """
    k_e = coulomb_constant()
    
    # 计算线电荷方向向量
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        return np.zeros(3)
    
    line_unit = line_vec / line_length
    
    # 计算场点到线电荷起点的向量
    r_vec = field_position - line_start
    
    # 计算垂直距离
    r_perp = r_vec - np.dot(r_vec, line_unit) * line_unit
    r_perp_mag = np.linalg.norm(r_perp)
    
    if r_perp_mag == 0:
        return np.zeros(3)
    
    # 计算电场强度（使用积分结果）
    field_mag = k_e * linear_density / r_perp_mag
    field_vec = field_mag * r_perp / r_perp_mag
    
    return field_vec

def electric_field_surface_charge(surface_density: float,
                                surface_normal: np.ndarray,
                                field_position: np.ndarray) -> np.ndarray:
    """
    计算面电荷产生的电场
    
    参数:
        surface_density: 面电荷密度 (C/m²)
        surface_normal: 面法向量
        field_position: 场点位置 (m)
    
    返回:
        电场强度向量 (N/C)
    """
    epsilon_0 = 8.854e-12  # F/m
    
    # 面电荷产生的电场垂直于表面
    field_mag = surface_density / (2 * epsilon_0)
    field_vec = field_mag * surface_normal / np.linalg.norm(surface_normal)
    
    return field_vec

def electric_field_verification(charges: List[PointCharge],
                              test_positions: List[np.ndarray],
                              tolerance: float = 1e-6) -> bool:
    """
    验证电场的无旋性
    
    参数:
        charges: 电荷列表
        test_positions: 测试位置列表
        tolerance: 容差
    
    返回:
        电场是否无旋
    """
    for position in test_positions:
        # 计算电场
        field = total_electric_field(charges, position)
        
        # 计算电场的旋度（简化计算）
        # 对于静电场，旋度应该为零
        # 这里使用数值方法验证
        
        # 在三个方向上计算偏导数
        dx = 1e-6
        dy = 1e-6
        dz = 1e-6
        
        # 计算旋度的z分量
        field_x_plus = total_electric_field(charges, position + np.array([dx, 0, 0]))
        field_x_minus = total_electric_field(charges, position - np.array([dx, 0, 0]))
        field_y_plus = total_electric_field(charges, position + np.array([0, dy, 0]))
        field_y_minus = total_electric_field(charges, position - np.array([0, dy, 0]))
        
        curl_z = (field_y_plus[0] - field_y_minus[0]) / (2 * dy) - \
                (field_x_plus[1] - field_x_minus[1]) / (2 * dx)
        
        if abs(curl_z) > tolerance:
            return False
    
    return True

# 示例使用
def electric_field_example():
    """电场示例"""
    # 创建电荷系统
    charges = [
        PointCharge(1e-6, np.array([0.0, 0.0, 0.0])),
        PointCharge(-1e-6, np.array([1.0, 0.0, 0.0]))
    ]
    
    # 计算电场
    field_position = np.array([0.5, 0.5, 0.0])
    electric_field = total_electric_field(charges, field_position)
    
    # 验证电场无旋性
    test_positions = [
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.5, 0.1])
    ]
    is_irrotational = electric_field_verification(charges, test_positions)
    
    print(f"电场强度: {electric_field} N/C")
    print(f"电场无旋: {is_irrotational}")
    
    return electric_field, is_irrotational
```

### 电势 / Electric Potential

**形式化定义**: 电势是描述电场中某点电学势能的标量场，是电场的势函数。

**公理化定义**:
设 $\mathcal{EP} = \langle \mathcal{Q}, \mathcal{R}, \mathcal{V}, \mathcal{E} \rangle$ 为电势系统，其中：

1. **电荷集合**: $\mathcal{Q}$ 为源电荷集合
2. **位置集合**: $\mathcal{R}$ 为空间位置集合
3. **电势函数**: $\mathcal{V}: \mathcal{R} \rightarrow \mathbb{R}$ 为电势函数
4. **电场关系**: $\mathcal{E}: \mathcal{V} \rightarrow \mathbb{R}^3$ 为电场与电势关系

**等价定义**:

1. **点电荷电势**: $V = k_e \frac{q}{r}$
2. **电势差**: $\Delta V = -\int \vec{E} \cdot d\vec{r}$
3. **电场梯度**: $\vec{E} = -\nabla V$

**形式化定理**:

**定理2.5.1.7 (电势叠加性)**: 多个电荷的电势满足叠加原理
$$V = \sum_i V_i$$

**定理2.5.1.8 (电势唯一性)**: 在给定边界条件下，电势解是唯一的
$$\nabla^2 V = -\frac{\rho}{\epsilon_0}$$

**定理2.5.1.9 (电势连续性)**: 电势在电荷分布处连续
$$\lim_{r \to 0} V(r) = V(0)$$

**Python算法实现**:

```python
def electric_potential_point_charge(charge: float,
                                  source_position: np.ndarray,
                                  field_position: np.ndarray) -> float:
    """
    计算点电荷产生的电势
    
    参数:
        charge: 源电荷量 (C)
        source_position: 源电荷位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        电势 (V)
    """
    k_e = coulomb_constant()
    
    # 计算距离
    r_vec = field_position - source_position
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return float('inf')
    
    # 计算电势
    potential = k_e * charge / r_mag
    
    return potential

def total_electric_potential(charges: List[PointCharge],
                           field_position: np.ndarray) -> float:
    """
    计算多个电荷产生的总电势
    
    参数:
        charges: 电荷列表
        field_position: 场点位置 (m)
    
    返回:
        总电势 (V)
    """
    total_potential = 0.0
    
    for charge in charges:
        potential = electric_potential_point_charge(
            charge.get_charge(),
            charge.get_position(),
            field_position
        )
        total_potential += potential
    
    return total_potential

def electric_field_from_potential(potential_func, position: np.ndarray,
                                step_size: float = 1e-6) -> np.ndarray:
    """
    从电势计算电场（数值梯度）
    
    参数:
        potential_func: 电势函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        电场强度向量 (N/C)
    """
    # 数值计算梯度
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dV_dx = (potential_func(position + dx) - potential_func(position - dx)) / (2 * step_size)
    dV_dy = (potential_func(position + dy) - potential_func(position - dy)) / (2 * step_size)
    dV_dz = (potential_func(position + dz) - potential_func(position - dz)) / (2 * step_size)
    
    # 电场 = -梯度电势
    electric_field = -np.array([dV_dx, dV_dy, dV_dz])
    
    return electric_field

def potential_difference(electric_field_func, start_position: np.ndarray,
                        end_position: np.ndarray, num_points: int = 100) -> float:
    """
    计算电势差
    
    参数:
        electric_field_func: 电场函数
        start_position: 起始位置 (m)
        end_position: 终止位置 (m)
        num_points: 积分点数
    
    返回:
        电势差 (V)
    """
    # 路径积分
    path = np.linspace(0, 1, num_points)
    potential_diff = 0.0
    
    for i in range(num_points - 1):
        # 路径中点
        mid_point = start_position + path[i] * (end_position - start_position)
        # 路径方向
        direction = (end_position - start_position) / num_points
        
        # 电场在路径方向的分量
        field = electric_field_func(mid_point)
        field_component = np.dot(field, direction)
        
        # 累加电势差
        potential_diff -= field_component
    
    return potential_diff

def equipotential_surfaces(charges: List[PointCharge], potential_value: float,
                          x_range: Tuple[float, float], y_range: Tuple[float, float],
                          z_value: float, grid_size: int = 50) -> List[np.ndarray]:
    """
    计算等势面
    
    参数:
        charges: 电荷列表
        potential_value: 电势值 (V)
        x_range: x坐标范围
        y_range: y坐标范围
        z_value: z坐标值
        grid_size: 网格大小
    
    返回:
        等势面上的点列表
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    equipotential_points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            position = np.array([X[i, j], Y[i, j], z_value])
            potential = total_electric_potential(charges, position)
            
            # 检查是否在等势面上（允许一定误差）
            if abs(potential - potential_value) < 1e-3:
                equipotential_points.append(position)
    
    return equipotential_points

def potential_energy_system(charges: List[PointCharge]) -> float:
    """
    计算电荷系统的总电势能
    
    参数:
        charges: 电荷列表
    
    返回:
        总电势能 (J)
    """
    total_energy = 0.0
    
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            energy = coulomb_potential_energy(
                charges[i].get_charge(), charges[j].get_charge(),
                charges[i].get_position(), charges[j].get_position()
            )
            total_energy += energy
    
    return total_energy

# 示例使用
def electric_potential_example():
    """电势示例"""
    # 创建电荷系统
    charges = [
        PointCharge(1e-6, np.array([0.0, 0.0, 0.0])),
        PointCharge(-1e-6, np.array([1.0, 0.0, 0.0]))
    ]
    
    # 计算电势
    field_position = np.array([0.5, 0.5, 0.0])
    potential = total_electric_potential(charges, field_position)
    
    # 计算电势差
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([1.0, 0.0, 0.0])
    
    def electric_field_func(pos):
        return total_electric_field(charges, pos)
    
    potential_diff = potential_difference(electric_field_func, start_pos, end_pos)
    
    # 计算系统电势能
    total_energy = potential_energy_system(charges)
    
    print(f"电势: {potential:.6f} V")
    print(f"电势差: {potential_diff:.6f} V")
    print(f"系统电势能: {total_energy:.6e} J")
    
    return potential, potential_diff, total_energy
```

### 高斯定律 / Gauss's Law

**形式化定义**: 高斯定律描述了电场通量与封闭曲面内电荷的关系，是静电学的基本定律之一。

**公理化定义**:
设 $\mathcal{GL} = \langle \mathcal{S}, \mathcal{Q}, \mathcal{E}, \mathcal{F} \rangle$ 为高斯定律系统，其中：

1. **曲面集合**: $\mathcal{S}$ 为封闭曲面集合
2. **电荷集合**: $\mathcal{Q}$ 为曲面内电荷集合
3. **电场函数**: $\mathcal{E}: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ 为电场函数
4. **通量函数**: $\mathcal{F}: \mathcal{S} \times \mathcal{E} \rightarrow \mathbb{R}$ 为电场通量函数

**等价定义**:

1. **积分形式**: $\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$
2. **微分形式**: $\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$
3. **对称性应用**: 利用对称性简化电场计算

**形式化定理**:

**定理2.5.1.10 (高斯定律)**: 电场通量等于封闭曲面内电荷除以介电常数
$$\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$$

**定理2.5.1.11 (高斯定律微分形式)**: 电场散度等于电荷密度除以介电常数
$$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$$

**定理2.5.1.12 (高斯定律对称性)**: 在对称电荷分布下，电场具有相同的对称性

**Python算法实现**:

```python
def gauss_law_integral(electric_field_func, surface_points: List[np.ndarray],
                      surface_normals: List[np.ndarray]) -> float:
    """
    计算高斯定律积分
    
    参数:
        electric_field_func: 电场函数
        surface_points: 表面点列表
        surface_normals: 表面法向量列表
    
    返回:
        电场通量 (N·m²/C)
    """
    flux = 0.0
    
    for point, normal in zip(surface_points, surface_normals):
        # 计算电场
        field = electric_field_func(point)
        
        # 计算通量
        flux += np.dot(field, normal)
    
    return flux

def enclosed_charge(charges: List[PointCharge], surface_center: np.ndarray,
                   surface_radius: float) -> float:
    """
    计算封闭曲面内的电荷
    
    参数:
        charges: 电荷列表
        surface_center: 曲面中心
        surface_radius: 曲面半径
    
    返回:
        封闭电荷量 (C)
    """
    enclosed_charge = 0.0
    
    for charge in charges:
        # 计算电荷到曲面中心的距离
        distance = np.linalg.norm(charge.get_position() - surface_center)
        
        # 如果电荷在曲面内
        if distance <= surface_radius:
            enclosed_charge += charge.get_charge()
    
    return enclosed_charge

def gauss_law_verification(charges: List[PointCharge], surface_center: np.ndarray,
                          surface_radius: float, num_points: int = 100,
                          tolerance: float = 1e-6) -> bool:
    """
    验证高斯定律
    
    参数:
        charges: 电荷列表
        surface_center: 曲面中心
        surface_radius: 曲面半径
        num_points: 积分点数
        tolerance: 容差
    
    返回:
        是否满足高斯定律
    """
    # 计算封闭电荷
    Q_enc = enclosed_charge(charges, surface_center, surface_radius)
    
    # 生成球面上的点
    surface_points = []
    surface_normals = []
    
    for i in range(num_points):
        # 球坐标
        theta = 2 * np.pi * i / num_points
        phi = np.pi * (i % (num_points // 2)) / (num_points // 2)
        
        # 转换为笛卡尔坐标
        x = surface_center[0] + surface_radius * np.sin(phi) * np.cos(theta)
        y = surface_center[1] + surface_radius * np.sin(phi) * np.sin(theta)
        z = surface_center[2] + surface_radius * np.cos(phi)
        
        point = np.array([x, y, z])
        normal = (point - surface_center) / surface_radius
        
        surface_points.append(point)
        surface_normals.append(normal)
    
    # 计算电场通量
    def electric_field_func(pos):
        return total_electric_field(charges, pos)
    
    flux = gauss_law_integral(electric_field_func, surface_points, surface_normals)
    
    # 高斯定律：通量 = 封闭电荷 / ε₀
    epsilon_0 = 8.854e-12  # F/m
    expected_flux = Q_enc / epsilon_0
    
    # 验证
    return abs(flux - expected_flux) < tolerance

def electric_field_from_gauss(charge_distribution, symmetry_type: str,
                            distance: float) -> np.ndarray:
    """
    利用高斯定律计算电场
    
    参数:
        charge_distribution: 电荷分布
        symmetry_type: 对称类型 ('spherical', 'cylindrical', 'planar')
        distance: 距离
    
    返回:
        电场强度向量 (N/C)
    """
    epsilon_0 = 8.854e-12  # F/m
    
    if symmetry_type == 'spherical':
        # 球对称
        total_charge = charge_distribution['total_charge']
        field_mag = total_charge / (4 * np.pi * epsilon_0 * distance**2)
        field_vec = field_mag * np.array([1, 0, 0])  # 径向方向
    
    elif symmetry_type == 'cylindrical':
        # 柱对称
        linear_density = charge_distribution['linear_density']
        field_mag = linear_density / (2 * np.pi * epsilon_0 * distance)
        field_vec = field_mag * np.array([1, 0, 0])  # 径向方向
    
    elif symmetry_type == 'planar':
        # 平面对称
        surface_density = charge_distribution['surface_density']
        field_mag = surface_density / (2 * epsilon_0)
        field_vec = field_mag * np.array([0, 0, 1])  # 法向方向
    
    else:
        raise ValueError("不支持的对称类型")
    
    return field_vec

def charge_density_from_field(electric_field_func, position: np.ndarray,
                            step_size: float = 1e-6) -> float:
    """
    从电场计算电荷密度（散度）
    
    参数:
        electric_field_func: 电场函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        电荷密度 (C/m³)
    """
    epsilon_0 = 8.854e-12  # F/m
    
    # 计算电场散度
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dEx_dx = (electric_field_func(position + dx)[0] - 
              electric_field_func(position - dx)[0]) / (2 * step_size)
    dEy_dy = (electric_field_func(position + dy)[1] - 
              electric_field_func(position - dy)[1]) / (2 * step_size)
    dEz_dz = (electric_field_func(position + dz)[2] - 
              electric_field_func(position - dz)[2]) / (2 * step_size)
    
    # 散度
    divergence = dEx_dx + dEy_dy + dEz_dz
    
    # 电荷密度 = ε₀ × 散度
    charge_density = epsilon_0 * divergence
    
    return charge_density

# 示例使用
def gauss_law_example():
    """高斯定律示例"""
    # 创建电荷系统
    charges = [
        PointCharge(1e-6, np.array([0.0, 0.0, 0.0])),
        PointCharge(2e-6, np.array([0.5, 0.0, 0.0]))
    ]
    
    # 验证高斯定律
    surface_center = np.array([0.0, 0.0, 0.0])
    surface_radius = 2.0
    gauss_satisfied = gauss_law_verification(charges, surface_center, surface_radius)
    
    # 利用高斯定律计算电场
    charge_distribution = {'total_charge': 3e-6}  # 总电荷
    field = electric_field_from_gauss(charge_distribution, 'spherical', 1.0)
    
    # 计算电荷密度
    def electric_field_func(pos):
        return total_electric_field(charges, pos)
    
    charge_density = charge_density_from_field(electric_field_func, np.array([0.1, 0.0, 0.0]))
    
    print(f"高斯定律满足: {gauss_satisfied}")
    print(f"电场强度: {field} N/C")
    print(f"电荷密度: {charge_density:.6e} C/m³")
    
    return gauss_satisfied, field, charge_density

---

## 2.5.2 静磁学 / Magnetostatics

### 安培定律 / Ampère's Law

**形式化定义**: 安培定律描述了稳恒电流与磁场之间的定量关系，是静磁学的基础定律。

**公理化定义**:
设 $\mathcal{AL} = \langle \mathcal{I}, \mathcal{C}, \mathcal{B}, \mathcal{M} \rangle$ 为安培定律系统，其中：

1. **电流集合**: $\mathcal{I}$ 为稳恒电流集合
2. **闭合路径集合**: $\mathcal{C}$ 为空间闭合曲线集合
3. **磁场函数**: $\mathcal{B}: \mathcal{C} \rightarrow \mathbb{R}^3$ 为磁场函数
4. **磁导率常数**: $\mathcal{M} = \mu_0$ 为真空磁导率

**等价定义**:

1. **积分形式**: $\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}$
2. **微分形式**: $\nabla \times \vec{B} = \mu_0 \vec{J}$
3. **斯托克斯形式**: $\int_S (\nabla \times \vec{B}) \cdot d\vec{S} = \mu_0 \int_S \vec{J} \cdot d\vec{S}$

**形式化定理**:

**定理2.5.2.1 (安培定律对称性)**: 安培定律在电流方向反转时保持形式不变
$$\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I_{enc} = -\mu_0 (-I_{enc})$$

**定理2.5.2.2 (安培定律叠加性)**: 多个电流的磁场满足叠加原理
$$\oint_C \vec{B} \cdot d\vec{l} = \mu_0 \sum_i I_{i,enc}$$

**定理2.5.2.3 (安培定律路径无关性)**: 对于同一包围电流的路径，安培定律积分值相同
$$\oint_{C_1} \vec{B} \cdot d\vec{l} = \oint_{C_2} \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}$$

**Python算法实现**:

```python
class CurrentElement:
    """电流元类"""
    def __init__(self, current: float, direction: np.ndarray, position: np.ndarray):
        self.current = current
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.position = np.array(position)
    
    def get_current(self) -> float:
        """获取电流强度"""
        return self.current
    
    def get_direction(self) -> np.ndarray:
        """获取电流方向"""
        return self.direction.copy()
    
    def get_position(self) -> np.ndarray:
        """获取位置"""
        return self.position.copy()

def magnetic_constant() -> float:
    """
    计算真空磁导率
    
    返回:
        真空磁导率 (H/m)
    """
    return 4 * np.pi * 1e-7

def ampere_law_integral(magnetic_field_func, path_points: List[np.ndarray]) -> float:
    """
    计算安培定律积分
    
    参数:
        magnetic_field_func: 磁场函数
        path_points: 闭合路径点列表
    
    返回:
        安培定律积分值 (T·m)
    """
    mu_0 = magnetic_constant()
    integral_value = 0.0
    
    # 计算路径积分
    for i in range(len(path_points)):
        current_point = path_points[i]
        next_point = path_points[(i + 1) % len(path_points)]
        
        # 计算路径元
        dl = next_point - current_point
        
        # 计算中点处的磁场
        mid_point = (current_point + next_point) / 2
        B = magnetic_field_func(mid_point)
        
        # 计算积分贡献
        integral_value += np.dot(B, dl)
    
    return integral_value

def enclosed_current(current_elements: List[CurrentElement], 
                    path_points: List[np.ndarray]) -> float:
    """
    计算被路径包围的电流
    
    参数:
        current_elements: 电流元列表
        path_points: 闭合路径点列表
    
    返回:
        包围电流 (A)
    """
    total_current = 0.0
    
    for element in current_elements:
        if is_point_inside_polygon(element.position, path_points):
            # 计算电流在路径法向的投影
            path_normal = calculate_path_normal(path_points)
            current_component = element.current * np.dot(element.direction, path_normal)
            total_current += current_component
    
    return total_current

def ampere_law_verification(current_elements: List[CurrentElement], 
                          path_points: List[np.ndarray], 
                          tolerance: float = 1e-6) -> bool:
    """
    验证安培定律
    
    参数:
        current_elements: 电流元列表
        path_points: 闭合路径点列表
        tolerance: 容差
    
    返回:
        是否满足安培定律
    """
    mu_0 = magnetic_constant()
    
    # 计算安培定律积分
    def magnetic_field_func(pos):
        return total_magnetic_field(current_elements, pos)
    
    integral_value = ampere_law_integral(magnetic_field_func, path_points)
    
    # 计算包围电流
    enclosed_current_value = enclosed_current(current_elements, path_points)
    
    # 验证安培定律
    expected_value = mu_0 * enclosed_current_value
    return abs(integral_value - expected_value) < tolerance

def magnetic_field_from_ampere(current_density_func, symmetry_type: str, 
                             distance: float) -> np.ndarray:
    """
    利用安培定律计算磁场
    
    参数:
        current_density_func: 电流密度函数
        symmetry_type: 对称类型 ('cylindrical', 'planar')
        distance: 距离 (m)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    
    if symmetry_type == 'cylindrical':
        # 柱对称电流
        current_per_unit_length = current_density_func(distance)
        field_mag = mu_0 * current_per_unit_length / (2 * np.pi * distance)
        field_vec = field_mag * np.array([0, 1, 0])  # 切向方向
    
    elif symmetry_type == 'planar':
        # 平面对称电流
        surface_current_density = current_density_func(distance)
        field_mag = mu_0 * surface_current_density / 2
        field_vec = field_mag * np.array([0, 1, 0])  # 切向方向
    
    else:
        raise ValueError("不支持的对称类型")
    
    return field_vec

def current_density_from_field(magnetic_field_func, position: np.ndarray,
                             step_size: float = 1e-6) -> np.ndarray:
    """
    从磁场计算电流密度（旋度）
    
    参数:
        magnetic_field_func: 磁场函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        电流密度向量 (A/m²)
    """
    mu_0 = magnetic_constant()
    
    # 计算磁场旋度
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dBz_dy = (magnetic_field_func(position + dy)[2] - 
              magnetic_field_func(position - dy)[2]) / (2 * step_size)
    dBy_dz = (magnetic_field_func(position + dz)[1] - 
              magnetic_field_func(position - dz)[1]) / (2 * step_size)
    
    dBx_dz = (magnetic_field_func(position + dz)[0] - 
              magnetic_field_func(position - dz)[0]) / (2 * step_size)
    dBz_dx = (magnetic_field_func(position + dx)[2] - 
              magnetic_field_func(position - dx)[2]) / (2 * step_size)
    
    dBy_dx = (magnetic_field_func(position + dx)[1] - 
              magnetic_field_func(position - dx)[1]) / (2 * step_size)
    dBx_dy = (magnetic_field_func(position + dy)[0] - 
              magnetic_field_func(position - dy)[0]) / (2 * step_size)
    
    # 旋度
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy
    
    # 电流密度 = 旋度 / μ₀
    current_density = np.array([curl_x, curl_y, curl_z]) / mu_0
    
    return current_density

# 辅助函数
def is_point_inside_polygon(point: np.ndarray, polygon_points: List[np.ndarray]) -> bool:
    """判断点是否在多边形内部"""
    x, y = point[0], point[1]
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0][0], polygon_points[0][1]
    for i in range(n + 1):
        p2x, p2y = polygon_points[i % n][0], polygon_points[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def calculate_path_normal(path_points: List[np.ndarray]) -> np.ndarray:
    """计算路径法向量"""
    # 简化为z方向
    return np.array([0, 0, 1])

# 示例使用
def ampere_law_example():
    """安培定律示例"""
    # 创建电流系统
    current_elements = [
        CurrentElement(1.0, np.array([0, 0, 1]), np.array([0.0, 0.0, 0.0])),
        CurrentElement(2.0, np.array([0, 0, 1]), np.array([0.5, 0.0, 0.0]))
    ]
    
    # 创建闭合路径
    path_points = [
        np.array([-1.0, -1.0, 0.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([-1.0, 1.0, 0.0])
    ]
    
    # 验证安培定律
    ampere_satisfied = ampere_law_verification(current_elements, path_points)
    
    # 利用安培定律计算磁场
    def current_density_func(r):
        return 1.0  # 单位电流密度
    
    field = magnetic_field_from_ampere(current_density_func, 'cylindrical', 1.0)
    
    # 计算电流密度
    def magnetic_field_func(pos):
        return total_magnetic_field(current_elements, pos)
    
    current_density = current_density_from_field(magnetic_field_func, np.array([0.1, 0.0, 0.0]))
    
    print(f"安培定律满足: {ampere_satisfied}")
    print(f"磁场强度: {field} T")
    print(f"电流密度: {current_density} A/m²")
    
    return ampere_satisfied, field, current_density
```

### 磁场 / Magnetic Field

**形式化定义**: 磁场是描述磁力作用的空间矢量场，由电流或磁偶极子产生。

**公理化定义**:
设 $\mathcal{MF} = \langle \mathcal{I}, \mathcal{R}, \mathcal{B}, \mathcal{F} \rangle$ 为磁场系统，其中：

1. **电流集合**: $\mathcal{I}$ 为电流元集合
2. **位置集合**: $\mathcal{R}$ 为三维空间位置集合
3. **磁场函数**: $\mathcal{B}: \mathcal{I} \times \mathcal{R} \rightarrow \mathbb{R}^3$ 为磁场函数
4. **力函数**: $\mathcal{F}: \mathcal{B} \times \mathcal{Q} \times \mathcal{V} \rightarrow \mathbb{R}^3$ 为洛伦兹力函数

**等价定义**:

1. **毕奥-萨伐尔形式**: $\vec{B} = \frac{\mu_0}{4\pi} \int \frac{I d\vec{l} \times \hat{r}}{r^2}$
2. **安培定律形式**: $\nabla \times \vec{B} = \mu_0 \vec{J}$
3. **磁矢势形式**: $\vec{B} = \nabla \times \vec{A}$

**形式化定理**:

**定理2.5.2.4 (磁场叠加性)**: 多个电流产生的磁场满足叠加原理
$$\vec{B} = \sum_i \vec{B}_i$$

**定理2.5.2.5 (磁场无散性)**: 磁场是无散场
$$\nabla \cdot \vec{B} = 0$$

**定理2.5.2.6 (磁场有旋性)**: 磁场是有旋场
$$\nabla \times \vec{B} = \mu_0 \vec{J}$$

**Python算法实现**:

```python
def magnetic_field_current_element(current: float, direction: np.ndarray,
                                 source_position: np.ndarray, 
                                 field_position: np.ndarray) -> np.ndarray:
    """
    计算电流元产生的磁场
    
    参数:
        current: 电流强度 (A)
        direction: 电流方向
        source_position: 电流元位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    
    # 计算相对位置向量
    r_vec = field_position - source_position
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算磁场（毕奥-萨伐尔定律）
    dl = direction * 1e-6  # 电流元长度
    r_hat = r_vec / r_mag
    
    # 叉积: dl × r̂
    cross_product = np.cross(dl, r_hat)
    
    # 磁场强度
    B_mag = mu_0 * current / (4 * np.pi * r_mag**2)
    B_vec = B_mag * cross_product
    
    return B_vec

def total_magnetic_field(current_elements: List[CurrentElement], 
                        field_position: np.ndarray) -> np.ndarray:
    """
    计算总磁场
    
    参数:
        current_elements: 电流元列表
        field_position: 场点位置 (m)
    
    返回:
        总磁场强度向量 (T)
    """
    total_field = np.zeros(3)
    
    for element in current_elements:
        field = magnetic_field_current_element(
            element.current, element.direction, 
            element.position, field_position
        )
        total_field += field
    
    return total_field

def magnetic_field_line_current(current: float, line_start: np.ndarray,
                               line_end: np.ndarray, 
                               field_position: np.ndarray) -> np.ndarray:
    """
    计算直线电流产生的磁场
    
    参数:
        current: 电流强度 (A)
        line_start: 线段起点 (m)
        line_end: 线段终点 (m)
        field_position: 场点位置 (m)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    
    # 计算线段方向
    line_direction = line_end - line_start
    line_length = np.linalg.norm(line_direction)
    
    if line_length == 0:
        return np.zeros(3)
    
    line_unit = line_direction / line_length
    
    # 计算场点到线段的垂直距离
    r_vec = field_position - line_start
    r_parallel = np.dot(r_vec, line_unit) * line_unit
    r_perp = r_vec - r_parallel
    r_perp_mag = np.linalg.norm(r_perp)
    
    if r_perp_mag == 0:
        return np.zeros(3)
    
    # 计算角度
    theta1 = np.arctan2(np.dot(field_position - line_start, line_unit), r_perp_mag)
    theta2 = np.arctan2(np.dot(field_position - line_end, line_unit), r_perp_mag)
    
    # 磁场强度
    B_mag = mu_0 * current / (4 * np.pi * r_perp_mag) * (np.sin(theta2) - np.sin(theta1))
    B_vec = B_mag * np.cross(line_unit, r_perp / r_perp_mag)
    
    return B_vec

def magnetic_field_circular_loop(current: float, radius: float,
                                center: np.ndarray, normal: np.ndarray,
                                field_position: np.ndarray) -> np.ndarray:
    """
    计算圆形电流环产生的磁场
    
    参数:
        current: 电流强度 (A)
        radius: 圆环半径 (m)
        center: 圆环中心 (m)
        normal: 圆环法向量
        field_position: 场点位置 (m)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    
    # 计算场点到圆环中心的距离
    r_vec = field_position - center
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算圆环轴上的磁场
    z = np.dot(r_vec, normal)
    r_perp = r_vec - z * normal
    r_perp_mag = np.linalg.norm(r_perp)
    
    # 磁场强度（轴上）
    if r_perp_mag == 0:
        # 在轴上
        B_mag = mu_0 * current * radius**2 / (2 * (radius**2 + z**2)**(3/2))
        B_vec = B_mag * normal
    else:
        # 不在轴上，近似计算
        B_mag = mu_0 * current * radius**2 / (4 * (radius**2 + r_mag**2)**(3/2))
        B_vec = B_mag * normal
    
    return B_vec

def magnetic_field_verification(current_elements: List[CurrentElement], 
                              test_positions: List[np.ndarray], 
                              tolerance: float = 1e-6) -> bool:
    """
    验证磁场计算
    
    参数:
        current_elements: 电流元列表
        test_positions: 测试位置列表
        tolerance: 容差
    
    返回:
        计算是否正确
    """
    for pos in test_positions:
        # 计算磁场
        field = total_magnetic_field(current_elements, pos)
        
        # 验证磁场无散性（近似）
        div_B = magnetic_field_divergence(current_elements, pos)
        if abs(div_B) > tolerance:
            return False
    
    return True

def magnetic_field_divergence(current_elements: List[CurrentElement], 
                            position: np.ndarray, step_size: float = 1e-6) -> float:
    """
    计算磁场散度
    
    参数:
        current_elements: 电流元列表
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        磁场散度
    """
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dBx_dx = (total_magnetic_field(current_elements, position + dx)[0] - 
              total_magnetic_field(current_elements, position - dx)[0]) / (2 * step_size)
    dBy_dy = (total_magnetic_field(current_elements, position + dy)[1] - 
              total_magnetic_field(current_elements, position - dy)[1]) / (2 * step_size)
    dBz_dz = (total_magnetic_field(current_elements, position + dz)[2] - 
              total_magnetic_field(current_elements, position - dz)[2]) / (2 * step_size)
    
    # 散度
    divergence = dBx_dx + dBy_dy + dBz_dz
    
    return divergence

# 示例使用
def magnetic_field_example():
    """磁场示例"""
    # 创建电流系统
    current_elements = [
        CurrentElement(1.0, np.array([0, 0, 1]), np.array([0.0, 0.0, 0.0])),
        CurrentElement(2.0, np.array([0, 0, 1]), np.array([0.5, 0.0, 0.0]))
    ]
    
    # 计算磁场
    field_position = np.array([1.0, 0.0, 0.0])
    field = total_magnetic_field(current_elements, field_position)
    
    # 验证磁场无散性
    divergence = magnetic_field_divergence(current_elements, field_position)
    
    # 计算直线电流磁场
    line_field = magnetic_field_line_current(
        1.0, np.array([0, -1, 0]), np.array([0, 1, 0]), field_position
    )
    
    # 计算圆形电流环磁场
    loop_field = magnetic_field_circular_loop(
        1.0, 0.5, np.array([0, 0, 0]), np.array([0, 0, 1]), field_position
    )
    
    print(f"总磁场: {field} T")
    print(f"磁场散度: {divergence:.6e}")
    print(f"直线电流磁场: {line_field} T")
    print(f"圆形电流环磁场: {loop_field} T")
    
    return field, divergence, line_field, loop_field
```

### 磁矢势 / Magnetic Vector Potential

**形式化定义**: 磁矢势是描述磁场的辅助矢量场，满足 $\vec{B} = \nabla \times \vec{A}$。

**公理化定义**:
设 $\mathcal{MVP} = \langle \mathcal{I}, \mathcal{R}, \mathcal{A}, \mathcal{B} \rangle$ 为磁矢势系统，其中：

1. **电流集合**: $\mathcal{I}$ 为电流元集合
2. **位置集合**: $\mathcal{R}$ 为三维空间位置集合
3. **磁矢势函数**: $\mathcal{A}: \mathcal{I} \times \mathcal{R} \rightarrow \mathbb{R}^3$ 为磁矢势函数
4. **磁场函数**: $\mathcal{B}: \mathcal{A} \rightarrow \mathbb{R}^3$ 为磁场函数

**等价定义**:

1. **积分形式**: $\vec{A} = \frac{\mu_0}{4\pi} \int \frac{\vec{J}}{r} dV$
2. **微分形式**: $\nabla^2 \vec{A} = -\mu_0 \vec{J}$
3. **库仑规范**: $\nabla \cdot \vec{A} = 0$

**形式化定理**:

**定理2.5.2.7 (磁矢势存在性)**: 对于无散磁场，存在磁矢势
$$\nabla \cdot \vec{B} = 0 \Rightarrow \exists \vec{A}: \vec{B} = \nabla \times \vec{A}$$

**定理2.5.2.8 (磁矢势规范不变性)**: 磁矢势在规范变换下磁场不变
$$\vec{A}' = \vec{A} + \nabla \chi \Rightarrow \nabla \times \vec{A}' = \nabla \times \vec{A}$$

**定理2.5.2.9 (磁矢势唯一性)**: 在库仑规范下，磁矢势唯一确定
$$\nabla \cdot \vec{A} = 0 \Rightarrow \vec{A} \text{ 唯一}$$

**Python算法实现**:

```python
def magnetic_vector_potential_current_element(current: float, direction: np.ndarray,
                                            source_position: np.ndarray, 
                                            field_position: np.ndarray) -> np.ndarray:
    """
    计算电流元产生的磁矢势
    
    参数:
        current: 电流强度 (A)
        direction: 电流方向
        source_position: 电流元位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        磁矢势向量 (T·m)
    """
    mu_0 = magnetic_constant()
    
    # 计算相对位置向量
    r_vec = field_position - source_position
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算磁矢势
    dl = direction * 1e-6  # 电流元长度
    A_mag = mu_0 * current / (4 * np.pi * r_mag)
    A_vec = A_mag * dl
    
    return A_vec

def total_magnetic_vector_potential(current_elements: List[CurrentElement], 
                                  field_position: np.ndarray) -> np.ndarray:
    """
    计算总磁矢势
    
    参数:
        current_elements: 电流元列表
        field_position: 场点位置 (m)
    
    返回:
        总磁矢势向量 (T·m)
    """
    total_potential = np.zeros(3)
    
    for element in current_elements:
        potential = magnetic_vector_potential_current_element(
            element.current, element.direction, 
            element.position, field_position
        )
        total_potential += potential
    
    return total_potential

def magnetic_field_from_vector_potential(vector_potential_func, 
                                       position: np.ndarray,
                                       step_size: float = 1e-6) -> np.ndarray:
    """
    从磁矢势计算磁场
    
    参数:
        vector_potential_func: 磁矢势函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        磁场强度向量 (T)
    """
    # 计算磁矢势旋度
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dAz_dy = (vector_potential_func(position + dy)[2] - 
              vector_potential_func(position - dy)[2]) / (2 * step_size)
    dAy_dz = (vector_potential_func(position + dz)[1] - 
              vector_potential_func(position - dz)[1]) / (2 * step_size)
    
    dAx_dz = (vector_potential_func(position + dz)[0] - 
              vector_potential_func(position - dz)[0]) / (2 * step_size)
    dAz_dx = (vector_potential_func(position + dx)[2] - 
              vector_potential_func(position - dx)[2]) / (2 * step_size)
    
    dAy_dx = (vector_potential_func(position + dx)[1] - 
              vector_potential_func(position - dx)[1]) / (2 * step_size)
    dAx_dy = (vector_potential_func(position + dy)[0] - 
              vector_potential_func(position - dy)[0]) / (2 * step_size)
    
    # 旋度
    curl_x = dAz_dy - dAy_dz
    curl_y = dAx_dz - dAz_dx
    curl_z = dAy_dx - dAx_dy
    
    return np.array([curl_x, curl_y, curl_z])

def vector_potential_divergence(vector_potential_func, position: np.ndarray,
                              step_size: float = 1e-6) -> float:
    """
    计算磁矢势散度
    
    参数:
        vector_potential_func: 磁矢势函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        磁矢势散度
    """
    dx = np.array([step_size, 0, 0])
    dy = np.array([0, step_size, 0])
    dz = np.array([0, 0, step_size])
    
    # 计算偏导数
    dAx_dx = (vector_potential_func(position + dx)[0] - 
              vector_potential_func(position - dx)[0]) / (2 * step_size)
    dAy_dy = (vector_potential_func(position + dy)[1] - 
              vector_potential_func(position - dy)[1]) / (2 * step_size)
    dAz_dz = (vector_potential_func(position + dz)[2] - 
              vector_potential_func(position - dz)[2]) / (2 * step_size)
    
    # 散度
    divergence = dAx_dx + dAy_dy + dAz_dz
    
    return divergence

def coulomb_gauge_condition(vector_potential_func, position: np.ndarray,
                          step_size: float = 1e-6) -> bool:
    """
    验证库仑规范条件
    
    参数:
        vector_potential_func: 磁矢势函数
        position: 位置 (m)
        step_size: 步长 (m)
    
    返回:
        是否满足库仑规范
    """
    divergence = vector_potential_divergence(vector_potential_func, position, step_size)
    return abs(divergence) < 1e-10

def vector_potential_verification(current_elements: List[CurrentElement], 
                                test_positions: List[np.ndarray], 
                                tolerance: float = 1e-6) -> bool:
    """
    验证磁矢势计算
    
    参数:
        current_elements: 电流元列表
        test_positions: 测试位置列表
        tolerance: 容差
    
    返回:
        计算是否正确
    """
    for pos in test_positions:
        # 计算磁矢势
        def vector_potential_func(position):
            return total_magnetic_vector_potential(current_elements, position)
        
        # 从磁矢势计算磁场
        field_from_potential = magnetic_field_from_vector_potential(vector_potential_func, pos)
        
        # 直接计算磁场
        field_direct = total_magnetic_field(current_elements, pos)
        
        # 验证一致性
        if np.linalg.norm(field_from_potential - field_direct) > tolerance:
            return False
        
        # 验证库仑规范
        if not coulomb_gauge_condition(vector_potential_func, pos):
            return False
    
    return True

# 示例使用
def magnetic_vector_potential_example():
    """磁矢势示例"""
    # 创建电流系统
    current_elements = [
        CurrentElement(1.0, np.array([0, 0, 1]), np.array([0.0, 0.0, 0.0])),
        CurrentElement(2.0, np.array([0, 0, 1]), np.array([0.5, 0.0, 0.0]))
    ]
    
    # 计算磁矢势
    field_position = np.array([1.0, 0.0, 0.0])
    vector_potential = total_magnetic_vector_potential(current_elements, field_position)
    
    # 从磁矢势计算磁场
    def vector_potential_func(position):
        return total_magnetic_vector_potential(current_elements, position)
    
    field_from_potential = magnetic_field_from_vector_potential(vector_potential_func, field_position)
    
    # 直接计算磁场
    field_direct = total_magnetic_field(current_elements, field_position)
    
    # 验证库仑规范
    coulomb_satisfied = coulomb_gauge_condition(vector_potential_func, field_position)
    
    print(f"磁矢势: {vector_potential} T·m")
    print(f"从磁矢势计算的磁场: {field_from_potential} T")
    print(f"直接计算的磁场: {field_direct} T")
    print(f"库仑规范满足: {coulomb_satisfied}")
    
    return vector_potential, field_from_potential, field_direct, coulomb_satisfied
```

### 毕奥-萨伐尔定律 / Biot-Savart Law

**形式化定义**: 毕奥-萨伐尔定律描述了电流元产生磁场的定量关系，是计算磁场的基本定律。

**公理化定义**:
设 $\mathcal{BSL} = \langle \mathcal{I}, \mathcal{R}, \mathcal{B}, \mathcal{K} \rangle$ 为毕奥-萨伐尔定律系统，其中：

1. **电流元集合**: $\mathcal{I}$ 为电流元集合
2. **位置集合**: $\mathcal{R}$ 为三维空间位置集合
3. **磁场函数**: $\mathcal{B}: \mathcal{I} \times \mathcal{R} \rightarrow \mathbb{R}^3$ 为磁场函数
4. **毕奥-萨伐尔常数**: $\mathcal{K} = \frac{\mu_0}{4\pi}$ 为磁常数

**等价定义**:

1. **微分形式**: $d\vec{B} = \frac{\mu_0}{4\pi} \frac{I d\vec{l} \times \hat{r}}{r^2}$
2. **积分形式**: $\vec{B} = \frac{\mu_0}{4\pi} \int \frac{I d\vec{l} \times \hat{r}}{r^2}$
3. **电流密度形式**: $\vec{B} = \frac{\mu_0}{4\pi} \int \frac{\vec{J} \times \hat{r}}{r^2} dV$

**形式化定理**:

**定理2.5.2.10 (毕奥-萨伐尔定律叠加性)**: 多个电流元的磁场满足叠加原理
$$d\vec{B} = \sum_i d\vec{B}_i$$

**定理2.5.2.11 (毕奥-萨伐尔定律对称性)**: 磁场在电流方向反转时改变符号
$$d\vec{B}(-I) = -d\vec{B}(I)$$

**定理2.5.2.12 (毕奥-萨伐尔定律距离依赖性)**: 磁场强度与距离平方成反比
$$|d\vec{B}| \propto \frac{1}{r^2}$$

**Python算法实现**:

```python
def biot_savart_law(current: float, dl: np.ndarray, source_position: np.ndarray,
                   field_position: np.ndarray) -> np.ndarray:
    """
    毕奥-萨伐尔定律计算磁场
    
    参数:
        current: 电流强度 (A)
        dl: 电流元向量 (m)
        source_position: 电流元位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    
    # 计算相对位置向量
    r_vec = field_position - source_position
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag == 0:
        return np.zeros(3)
    
    # 计算单位向量
    r_hat = r_vec / r_mag
    
    # 计算叉积: dl × r̂
    cross_product = np.cross(dl, r_hat)
    
    # 计算磁场
    B_mag = mu_0 * current / (4 * np.pi * r_mag**2)
    B_vec = B_mag * cross_product
    
    return B_vec

def biot_savart_integral(current: float, path_points: List[np.ndarray],
                        field_position: np.ndarray) -> np.ndarray:
    """
    毕奥-萨伐尔定律积分计算磁场
    
    参数:
        current: 电流强度 (A)
        path_points: 电流路径点列表
        field_position: 场点位置 (m)
    
    返回:
        磁场强度向量 (T)
    """
    total_field = np.zeros(3)
    
    for i in range(len(path_points) - 1):
        # 计算电流元
        dl = path_points[i + 1] - path_points[i]
        source_position = (path_points[i] + path_points[i + 1]) / 2
        
        # 计算磁场贡献
        field_contribution = biot_savart_law(current, dl, source_position, field_position)
        total_field += field_contribution
    
    return total_field

def biot_savart_current_density(current_density_func, volume_points: List[np.ndarray],
                               field_position: np.ndarray, volume_element: float) -> np.ndarray:
    """
    毕奥-萨伐尔定律计算电流密度产生的磁场
    
    参数:
        current_density_func: 电流密度函数
        volume_points: 体积元位置列表
        field_position: 场点位置 (m)
        volume_element: 体积元大小 (m³)
    
    返回:
        磁场强度向量 (T)
    """
    mu_0 = magnetic_constant()
    total_field = np.zeros(3)
    
    for point in volume_points:
        # 计算电流密度
        J = current_density_func(point)
        
        # 计算相对位置向量
        r_vec = field_position - point
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            continue
        
        # 计算单位向量
        r_hat = r_vec / r_mag
        
        # 计算叉积: J × r̂
        cross_product = np.cross(J, r_hat)
        
        # 计算磁场贡献
        B_contribution = mu_0 / (4 * np.pi * r_mag**2) * cross_product * volume_element
        total_field += B_contribution
    
    return total_field

def biot_savart_verification(current_elements: List[CurrentElement], 
                           test_positions: List[np.ndarray], 
                           tolerance: float = 1e-6) -> bool:
    """
    验证毕奥-萨伐尔定律
    
    参数:
        current_elements: 电流元列表
        test_positions: 测试位置列表
        tolerance: 容差
    
    返回:
        计算是否正确
    """
    for pos in test_positions:
        # 使用毕奥-萨伐尔定律计算
        biot_savart_field = np.zeros(3)
        for element in current_elements:
            dl = element.direction * 1e-6
            field_contribution = biot_savart_law(
                element.current, dl, element.position, pos
            )
            biot_savart_field += field_contribution
        
        # 使用直接方法计算
        direct_field = total_magnetic_field(current_elements, pos)
        
        # 验证一致性
        if np.linalg.norm(biot_savart_field - direct_field) > tolerance:
            return False
    
    return True

def biot_savart_symmetry_test(current: float, dl: np.ndarray, 
                            source_position: np.ndarray,
                            field_position: np.ndarray) -> bool:
    """
    测试毕奥-萨伐尔定律对称性
    
    参数:
        current: 电流强度 (A)
        dl: 电流元向量 (m)
        source_position: 电流元位置 (m)
        field_position: 场点位置 (m)
    
    返回:
        对称性是否满足
    """
    # 正向电流磁场
    B_positive = biot_savart_law(current, dl, source_position, field_position)
    
    # 反向电流磁场
    B_negative = biot_savart_law(-current, dl, source_position, field_position)
    
    # 验证对称性
    return np.allclose(B_positive, -B_negative)

def biot_savart_distance_test(current: float, dl: np.ndarray,
                            source_position: np.ndarray,
                            field_positions: List[np.ndarray]) -> bool:
    """
    测试毕奥-萨伐尔定律距离依赖性
    
    参数:
        current: 电流强度 (A)
        dl: 电流元向量 (m)
        source_position: 电流元位置 (m)
        field_positions: 不同距离的场点位置列表
    
    返回:
        距离依赖性是否正确
    """
    distances = []
    field_magnitudes = []
    
    for pos in field_positions:
        r_vec = pos - source_position
        distance = np.linalg.norm(r_vec)
        distances.append(distance)
        
        field = biot_savart_law(current, dl, source_position, pos)
        field_magnitude = np.linalg.norm(field)
        field_magnitudes.append(field_magnitude)
    
    # 验证 1/r² 关系
    for i in range(len(distances)):
        expected_magnitude = field_magnitudes[0] * (distances[0] / distances[i])**2
        if abs(field_magnitudes[i] - expected_magnitude) > 1e-6:
            return False
    
    return True

# 示例使用
def biot_savart_law_example():
    """毕奥-萨伐尔定律示例"""
    # 创建电流元
    current = 1.0
    dl = np.array([0, 0, 1e-6])  # 1μm 电流元
    source_position = np.array([0.0, 0.0, 0.0])
    field_position = np.array([1.0, 0.0, 0.0])
    
    # 计算磁场
    field = biot_savart_law(current, dl, source_position, field_position)
    
    # 验证对称性
    symmetry_satisfied = biot_savart_symmetry_test(current, dl, source_position, field_position)
    
    # 验证距离依赖性
    test_positions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0])
    ]
    distance_test_satisfied = biot_savart_distance_test(current, dl, source_position, test_positions)
    
    # 计算直线电流磁场
    path_points = [
        np.array([0, -1, 0]),
        np.array([0, 1, 0])
    ]
    line_field = biot_savart_integral(current, path_points, field_position)
    
    print(f"毕奥-萨伐尔磁场: {field} T")
    print(f"对称性满足: {symmetry_satisfied}")
    print(f"距离依赖性满足: {distance_test_satisfied}")
    print(f"直线电流磁场: {line_field} T")
    
    return field, symmetry_satisfied, distance_test_satisfied, line_field

---

## 2.5.3 电磁感应 / Electromagnetic Induction

### 法拉第定律 / Faraday's Law

**形式化定义**: 法拉第定律描述了变化的磁通量产生感应电动势的定量关系，是电磁感应的基础定律。

**公理化定义**:
设 $\mathcal{FL} = \langle \mathcal{B}, \mathcal{S}, \mathcal{E}, \mathcal{T} \rangle$ 为法拉第定律系统，其中：

1. **磁场集合**: $\mathcal{B}$ 为磁场函数集合
2. **曲面集合**: $\mathcal{S}$ 为空间曲面集合
3. **电动势函数**: $\mathcal{E}: \mathcal{S} \times \mathcal{T} \rightarrow \mathbb{R}$ 为感应电动势函数
4. **时间集合**: $\mathcal{T}$ 为时间集合

**等价定义**:

1. **积分形式**: $\mathcal{E} = -\frac{d\Phi_B}{dt}$
2. **微分形式**: $\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$
3. **斯托克斯形式**: $\oint_C \vec{E} \cdot d\vec{l} = -\frac{d}{dt}\int_S \vec{B} \cdot d\vec{S}$

**形式化定理**:

**定理2.5.3.1 (法拉第定律线性性)**: 感应电动势与磁通量变化率成正比
$$\mathcal{E} = -k \frac{d\Phi_B}{dt}$$

**定理2.5.3.2 (法拉第定律叠加性)**: 多个磁场的感应电动势满足叠加原理
$$\mathcal{E} = \sum_i \mathcal{E}_i = -\sum_i \frac{d\Phi_{B,i}}{dt}$$

**定理2.5.3.3 (法拉第定律时间反演性)**: 法拉第定律在时间反演下改变符号
$$\mathcal{E}(-t) = -\mathcal{E}(t)$$

**Python算法实现**:

```python
import numpy as np
from typing import List, Tuple, Callable
from scipy.integrate import quad

class MagneticField:
    """磁场类"""
    def __init__(self, field_function: Callable, time_function: Callable = None):
        self.field_function = field_function
        self.time_function = time_function or (lambda t: 1.0)
    
    def get_field(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """获取磁场"""
        return self.field_function(position) * self.time_function(time)
    
    def get_field_derivative(self, position: np.ndarray, time: float = 0.0, 
                           dt: float = 1e-6) -> np.ndarray:
        """获取磁场时间导数"""
        field_t = self.get_field(position, time)
        field_t_dt = self.get_field(position, time + dt)
        return (field_t_dt - field_t) / dt

def magnetic_flux(magnetic_field: MagneticField, surface_points: List[np.ndarray],
                 surface_normals: List[np.ndarray], time: float = 0.0) -> float:
    """
    计算磁通量
    
    参数:
        magnetic_field: 磁场对象
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time: 时间 (s)
    
    返回:
        磁通量 (Wb)
    """
    flux = 0.0
    
    for i, point in enumerate(surface_points):
        # 获取磁场
        field = magnetic_field.get_field(point, time)
        
        # 计算通量
        normal = surface_normals[i]
        area_element = 1.0  # 假设单位面积
        flux += np.dot(field, normal) * area_element
    
    return flux

def faraday_emf(magnetic_field: MagneticField, surface_points: List[np.ndarray],
                surface_normals: List[np.ndarray], time: float = 0.0,
                dt: float = 1e-6) -> float:
    """
    计算法拉第感应电动势
    
    参数:
        magnetic_field: 磁场对象
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        感应电动势 (V)
    """
    # 计算磁通量
    flux_t = magnetic_flux(magnetic_field, surface_points, surface_normals, time)
    flux_t_dt = magnetic_flux(magnetic_field, surface_points, surface_normals, time + dt)
    
    # 磁通量变化率
    dflux_dt = (flux_t_dt - flux_t) / dt
    
    # 法拉第电动势
    emf = -dflux_dt
    
    return emf

def faraday_emf_verification(magnetic_field: MagneticField,
                           surface_points: List[np.ndarray],
                           surface_normals: List[np.ndarray],
                           time_range: Tuple[float, float],
                           tolerance: float = 1e-6) -> bool:
    """
    验证法拉第定律
    
    参数:
        magnetic_field: 磁场对象
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time_range: 时间范围
        tolerance: 容差
    
    返回:
        法拉第定律是否满足
    """
    t_start, t_end = time_range
    num_tests = 10
    times = np.linspace(t_start, t_end, num_tests)
    
    for t in times:
        # 计算电动势
        emf = faraday_emf(magnetic_field, surface_points, surface_normals, t)
        
        # 计算磁通量变化率
        flux_t = magnetic_flux(magnetic_field, surface_points, surface_normals, t)
        flux_t_dt = magnetic_flux(magnetic_field, surface_points, surface_normals, t + 1e-6)
        dflux_dt = (flux_t_dt - flux_t) / 1e-6
        
        # 验证关系
        if abs(emf + dflux_dt) > tolerance:
            return False
    
    return True

def induced_electric_field(magnetic_field: MagneticField, position: np.ndarray,
                          time: float = 0.0, dt: float = 1e-6) -> np.ndarray:
    """
    计算感应电场
    
    参数:
        magnetic_field: 磁场对象
        position: 位置 (m)
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        感应电场 (N/C)
    """
    # 磁场时间导数
    dB_dt = magnetic_field.get_field_derivative(position, time, dt)
    
    # 感应电场（简化计算，假设均匀磁场）
    # 实际应用中需要求解麦克斯韦方程组
    induced_field = -0.5 * np.cross(np.array([0, 0, 1]), dB_dt)
    
    return induced_field

def faraday_emf_example():
    """法拉第定律示例"""
    # 定义磁场函数
    def uniform_magnetic_field(position: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 1.0])  # 均匀磁场，z方向
    
    def time_varying_field(position: np.ndarray, time: float) -> np.ndarray:
        return np.array([0, 0, np.sin(time)])  # 正弦变化的磁场
    
    # 创建磁场对象
    static_field = MagneticField(uniform_magnetic_field)
    varying_field = MagneticField(lambda pos: np.array([0, 0, 1.0]), 
                                lambda t: np.sin(t))
    
    # 定义曲面
    surface_points = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0])
    ]
    surface_normals = [np.array([0, 0, 1])] * 4
    
    # 计算磁通量
    flux_static = magnetic_flux(static_field, surface_points, surface_normals)
    flux_varying = magnetic_flux(varying_field, surface_points, surface_normals, time=1.0)
    
    # 计算感应电动势
    emf = faraday_emf(varying_field, surface_points, surface_normals, time=1.0)
    
    # 验证法拉第定律
    time_range = (0.0, 2.0)
    faraday_satisfied = faraday_emf_verification(varying_field, surface_points, 
                                               surface_normals, time_range)
    
    print(f"静态磁通量: {flux_static:.6f} Wb")
    print(f"变化磁通量: {flux_varying:.6f} Wb")
    print(f"感应电动势: {emf:.6f} V")
    print(f"法拉第定律满足: {faraday_satisfied}")
    
    return flux_static, flux_varying, emf, faraday_satisfied
```

### 楞次定律 / Lenz's Law

**形式化定义**: 楞次定律描述了感应电流的方向，感应电流产生的磁场总是阻碍引起感应的磁通量变化。

**公理化定义**:
设 $\mathcal{LL} = \langle \mathcal{I}, \mathcal{B}, \mathcal{F}, \mathcal{D} \rangle$ 为楞次定律系统，其中：

1. **感应电流集合**: $\mathcal{I}$ 为感应电流集合
2. **磁场集合**: $\mathcal{B}$ 为磁场集合
3. **磁通量函数**: $\mathcal{F}: \mathcal{B} \times \mathcal{S} \rightarrow \mathbb{R}$ 为磁通量函数
4. **方向函数**: $\mathcal{D}: \mathcal{F} \rightarrow \mathcal{I}$ 为感应电流方向函数

**等价定义**:

1. **定性表述**: 感应电流方向总是阻碍磁通量变化
2. **定量表述**: $\vec{B}_{induced} \cdot \frac{d\vec{B}_{external}}{dt} < 0$
3. **能量表述**: 感应电流总是消耗能量来阻碍变化

**形式化定理**:

**定理2.5.3.4 (楞次定律方向性)**: 感应电流方向与磁通量变化方向相反
$$\vec{I}_{induced} \propto -\frac{d\Phi_B}{dt}$$

**定理2.5.3.5 (楞次定律能量守恒)**: 楞次定律保证了电磁感应的能量守恒
$$P_{induced} = -I_{induced} \cdot \mathcal{E} > 0$$

**定理2.5.3.6 (楞次定律稳定性)**: 楞次定律确保了电磁系统的稳定性
$$\frac{d^2\Phi_B}{dt^2} \cdot \frac{d\Phi_B}{dt} < 0$$

**Python算法实现**:

```python
def lenz_law_direction(magnetic_flux_change: float, 
                      current_direction: np.ndarray) -> np.ndarray:
    """
    确定楞次定律感应电流方向
    
    参数:
        magnetic_flux_change: 磁通量变化率 (Wb/s)
        current_direction: 参考电流方向
    
    返回:
        感应电流方向
    """
    # 楞次定律：感应电流方向与磁通量变化方向相反
    if magnetic_flux_change > 0:
        # 磁通量增加，感应电流产生反向磁场
        induced_direction = -current_direction
    else:
        # 磁通量减少，感应电流产生同向磁场
        induced_direction = current_direction
    
    return induced_direction / np.linalg.norm(induced_direction)

def induced_magnetic_field(induced_current: float, current_loop: List[np.ndarray],
                          field_position: np.ndarray) -> np.ndarray:
    """
    计算感应电流产生的磁场
    
    参数:
        induced_current: 感应电流强度 (A)
        current_loop: 电流环路点列表
        field_position: 场点位置 (m)
    
    返回:
        感应磁场 (T)
    """
    # 使用毕奥-萨伐尔定律计算感应磁场
    induced_field = np.zeros(3)
    
    for i in range(len(current_loop) - 1):
        dl = current_loop[i + 1] - current_loop[i]
        source_position = (current_loop[i] + current_loop[i + 1]) / 2
        
        # 计算电流元产生的磁场
        field_element = biot_savart_law(induced_current, dl, source_position, field_position)
        induced_field += field_element
    
    return induced_field

def lenz_law_verification(external_magnetic_field: MagneticField,
                         induced_current: float, current_loop: List[np.ndarray],
                         surface_points: List[np.ndarray], surface_normals: List[np.ndarray],
                         time: float = 0.0, tolerance: float = 1e-6) -> bool:
    """
    验证楞次定律
    
    参数:
        external_magnetic_field: 外磁场
        induced_current: 感应电流
        current_loop: 电流环路
        surface_points: 曲面点
        surface_normals: 曲面法向量
        time: 时间
        tolerance: 容差
    
    返回:
        楞次定律是否满足
    """
    # 计算外磁场变化率
    flux_t = magnetic_flux(external_magnetic_field, surface_points, surface_normals, time)
    flux_t_dt = magnetic_flux(external_magnetic_field, surface_points, surface_normals, time + 1e-6)
    dflux_dt = (flux_t_dt - flux_t) / 1e-6
    
    # 计算感应磁场
    field_position = np.array([0, 0, 0])  # 参考点
    induced_field = induced_magnetic_field(induced_current, current_loop, field_position)
    
    # 计算外磁场
    external_field = external_magnetic_field.get_field(field_position, time)
    
    # 验证楞次定律：感应磁场与外磁场变化方向相反
    dot_product = np.dot(induced_field, external_field)
    
    if dflux_dt > 0:
        # 磁通量增加，感应磁场应该与外磁场方向相反
        return dot_product < -tolerance
    else:
        # 磁通量减少，感应磁场应该与外磁场方向相同
        return dot_product > tolerance

def energy_dissipation(induced_current: float, resistance: float) -> float:
    """
    计算感应电流的能量耗散
    
    参数:
        induced_current: 感应电流 (A)
        resistance: 电阻 (Ω)
    
    返回:
        功率耗散 (W)
    """
    return induced_current**2 * resistance

def lenz_law_example():
    """楞次定律示例"""
    # 定义变化的磁场
    def varying_magnetic_field(position: np.ndarray, time: float) -> np.ndarray:
        return np.array([0, 0, np.sin(time)])
    
    magnetic_field = MagneticField(lambda pos: np.array([0, 0, 1.0]), 
                                 lambda t: np.sin(t))
    
    # 定义电流环路
    current_loop = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 0])
    ]
    
    # 定义曲面
    surface_points = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0])
    ]
    surface_normals = [np.array([0, 0, 1])] * 4
    
    # 计算磁通量变化
    time = 1.0
    flux_change = faraday_emf(magnetic_field, surface_points, surface_normals, time)
    
    # 确定感应电流方向
    reference_direction = np.array([0, 0, 1])
    induced_direction = lenz_law_direction(flux_change, reference_direction)
    
    # 计算感应电流（假设电阻为1Ω）
    resistance = 1.0
    induced_current = abs(flux_change) / resistance
    
    # 验证楞次定律
    lenz_satisfied = lenz_law_verification(magnetic_field, induced_current, 
                                         current_loop, surface_points, surface_normals, time)
    
    # 计算能量耗散
    power_dissipation = energy_dissipation(induced_current, resistance)
    
    print(f"磁通量变化率: {flux_change:.6f} Wb/s")
    print(f"感应电流方向: {induced_direction}")
    print(f"感应电流强度: {induced_current:.6f} A")
    print(f"楞次定律满足: {lenz_satisfied}")
    print(f"功率耗散: {power_dissipation:.6f} W")
    
    return flux_change, induced_direction, induced_current, lenz_satisfied, power_dissipation
```

### 自感和互感 / Self and Mutual Inductance

**形式化定义**: 自感是导体中电流变化产生的感应电动势与电流变化率的比值，互感是两个导体之间的磁耦合系数。

**公理化定义**:
设 $\mathcal{LI} = \langle \mathcal{C}, \mathcal{I}, \mathcal{E}, \mathcal{L} \rangle$ 为电感系统，其中：

1. **导体集合**: $\mathcal{C}$ 为导体集合
2. **电流集合**: $\mathcal{I}$ 为电流集合
3. **电动势集合**: $\mathcal{E}$ 为感应电动势集合
4. **电感函数**: $\mathcal{L}: \mathcal{C} \times \mathcal{C} \rightarrow \mathbb{R}$ 为电感函数

**等价定义**:

1. **自感定义**: $L = \frac{\Phi_B}{I}$
2. **互感定义**: $M_{12} = \frac{\Phi_{12}}{I_1}$
3. **电动势关系**: $\mathcal{E} = -L\frac{dI}{dt}$

**形式化定理**:

**定理2.5.3.7 (自感正定性)**: 自感系数总是正值
$$L = \frac{\Phi_B}{I} > 0$$

**定理2.5.3.8 (互感对称性)**: 互感系数满足对称性
$$M_{12} = M_{21}$$

**定理2.5.3.9 (电感叠加性)**: 串联电感的等效电感为各电感之和
$$L_{eq} = \sum_i L_i$$

**Python算法实现**:

```python
class Inductor:
    """电感器类"""
    def __init__(self, inductance: float, resistance: float = 0.0):
        self.inductance = inductance
        self.resistance = resistance
        self.current = 0.0
        self.voltage = 0.0
    
    def update_current(self, voltage: float, dt: float):
        """更新电流"""
        # 电感方程：V = L * dI/dt + R * I
        di_dt = (voltage - self.resistance * self.current) / self.inductance
        self.current += di_dt * dt
        self.voltage = voltage
    
    def get_induced_voltage(self, di_dt: float) -> float:
        """计算感应电压"""
        return -self.inductance * di_dt

def self_inductance_calculation(current_loop: List[np.ndarray], 
                              current: float) -> float:
    """
    计算自感系数
    
    参数:
        current_loop: 电流环路
        current: 电流强度 (A)
    
    返回:
        自感系数 (H)
    """
    # 简化计算：假设均匀磁场
    # 实际应用中需要数值积分
    
    # 计算环路面积
    area = 0.0
    for i in range(len(current_loop) - 1):
        p1 = current_loop[i]
        p2 = current_loop[i + 1]
        area += 0.5 * np.cross(p1, p2)[2]  # z分量
    
    # 假设磁场与面积成正比
    magnetic_flux = area * current
    
    # 自感系数
    inductance = magnetic_flux / current
    
    return inductance

def mutual_inductance_calculation(loop1: List[np.ndarray], loop2: List[np.ndarray],
                                current1: float) -> float:
    """
    计算互感系数
    
    参数:
        loop1: 第一个环路
        loop2: 第二个环路
        current1: 第一个环路的电流 (A)
    
    返回:
        互感系数 (H)
    """
    # 简化计算：基于几何关系
    # 实际应用中需要复杂的数值积分
    
    # 计算两个环路的中心距离
    center1 = np.mean(loop1[:-1], axis=0)
    center2 = np.mean(loop2[:-1], axis=0)
    distance = np.linalg.norm(center2 - center1)
    
    # 计算环路面积
    area1 = 0.0
    for i in range(len(loop1) - 1):
        p1 = loop1[i]
        p2 = loop1[i + 1]
        area1 += 0.5 * np.cross(p1, p2)[2]
    
    area2 = 0.0
    for i in range(len(loop2) - 1):
        p1 = loop2[i]
        p2 = loop2[i + 1]
        area2 += 0.5 * np.cross(p1, p2)[2]
    
    # 互感系数（简化公式）
    mutual_inductance = (area1 * area2) / (4 * np.pi * distance**3)
    
    return mutual_inductance

def inductor_voltage_response(inductor: Inductor, input_voltage: Callable,
                            time_range: Tuple[float, float], dt: float = 1e-3) -> Tuple[List[float], List[float]]:
    """
    计算电感器的电压响应
    
    参数:
        inductor: 电感器对象
        input_voltage: 输入电压函数
        time_range: 时间范围
        dt: 时间步长
    
    返回:
        时间列表和电流列表
    """
    t_start, t_end = time_range
    times = np.arange(t_start, t_end, dt)
    currents = []
    
    for t in times:
        voltage = input_voltage(t)
        inductor.update_current(voltage, dt)
        currents.append(inductor.current)
    
    return times.tolist(), currents

def coupled_inductors_analysis(inductor1: Inductor, inductor2: Inductor,
                             mutual_inductance: float, input_voltage: float,
                             dt: float = 1e-3) -> Tuple[float, float]:
    """
    分析耦合电感器
    
    参数:
        inductor1, inductor2: 电感器对象
        mutual_inductance: 互感系数 (H)
        input_voltage: 输入电压 (V)
        dt: 时间步长 (s)
    
    返回:
        两个电感器的电流
    """
    # 耦合电感器方程
    # V1 = L1 * dI1/dt + M * dI2/dt + R1 * I1
    # V2 = L2 * dI2/dt + M * dI1/dt + R2 * I2
    
    # 简化计算：假设I2初始为0
    di1_dt = input_voltage / inductor1.inductance
    inductor1.current += di1_dt * dt
    
    # 互感效应
    di2_dt = -mutual_inductance * di1_dt / inductor2.inductance
    inductor2.current += di2_dt * dt
    
    return inductor1.current, inductor2.current

def inductance_example():
    """电感示例"""
    # 创建电感器
    inductor1 = Inductor(inductance=1e-3, resistance=1.0)  # 1mH, 1Ω
    inductor2 = Inductor(inductance=2e-3, resistance=2.0)  # 2mH, 2Ω
    
    # 定义电流环路
    loop1 = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 0])
    ]
    
    loop2 = [
        np.array([2, 0, 0]),
        np.array([3, 0, 0]),
        np.array([3, 1, 0]),
        np.array([2, 1, 0]),
        np.array([2, 0, 0])
    ]
    
    # 计算自感系数
    self_inductance1 = self_inductance_calculation(loop1, 1.0)
    self_inductance2 = self_inductance_calculation(loop2, 1.0)
    
    # 计算互感系数
    mutual_inductance = mutual_inductance_calculation(loop1, loop2, 1.0)
    
    # 分析耦合电感器
    input_voltage = 5.0  # 5V
    current1, current2 = coupled_inductors_analysis(inductor1, inductor2, 
                                                   mutual_inductance, input_voltage)
    
    # 计算感应电压
    di_dt = 1.0  # 假设电流变化率
    induced_voltage1 = inductor1.get_induced_voltage(di_dt)
    induced_voltage2 = inductor2.get_induced_voltage(di_dt)
    
    print(f"自感系数1: {self_inductance1:.6e} H")
    print(f"自感系数2: {self_inductance2:.6e} H")
    print(f"互感系数: {mutual_inductance:.6e} H")
    print(f"电流1: {current1:.6f} A")
    print(f"电流2: {current2:.6f} A")
    print(f"感应电压1: {induced_voltage1:.6f} V")
    print(f"感应电压2: {induced_voltage2:.6f} V")
    
    return (self_inductance1, self_inductance2, mutual_inductance, 
            current1, current2, induced_voltage1, induced_voltage2)
```

---

## 2.5.4 麦克斯韦方程组 / Maxwell's Equations

### 积分形式 / Integral Form

**形式化定义**: 麦克斯韦方程组是电磁学的基本定律，描述了电场和磁场的基本性质及其相互关系。

**公理化定义**:
设 $\mathcal{ME} = \langle \mathcal{E}, \mathcal{B}, \mathcal{J}, \mathcal{Q}, \mathcal{S}, \mathcal{V} \rangle$ 为麦克斯韦方程组系统，其中：

1. **电场集合**: $\mathcal{E}$ 为电场向量场集合
2. **磁场集合**: $\mathcal{B}$ 为磁场向量场集合
3. **电流密度集合**: $\mathcal{J}$ 为电流密度向量场集合
4. **电荷密度集合**: $\mathcal{Q}$ 为电荷密度标量场集合
5. **曲面集合**: $\mathcal{S}$ 为封闭曲面集合
6. **体积集合**: $\mathcal{V}$ 为封闭体积集合

**等价定义**:

1. **高斯电场定律**: $\oint_S \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$
2. **高斯磁场定律**: $\oint_S \vec{B} \cdot d\vec{A} = 0$
3. **法拉第电磁感应定律**: $\oint_C \vec{E} \cdot d\vec{l} = -\frac{d\Phi_B}{dt}$
4. **安培-麦克斯韦定律**: $\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I_{enc} + \mu_0 \epsilon_0 \frac{d\Phi_E}{dt}$

**形式化定理**:

**定理2.5.4.1 (电荷守恒)**: 麦克斯韦方程组隐含电荷守恒定律
$$\oint_S \vec{J} \cdot d\vec{A} = -\frac{dQ_{enc}}{dt}$$

**定理2.5.4.2 (电磁场能量守恒)**: 电磁场能量密度满足守恒定律
$$\frac{\partial u}{\partial t} + \nabla \cdot \vec{S} = -\vec{J} \cdot \vec{E}$$

**定理2.5.4.3 (电磁场动量守恒)**: 电磁场动量密度满足守恒定律
$$\frac{\partial \vec{g}}{\partial t} + \nabla \cdot \mathbf{T} = -\rho \vec{E} - \vec{J} \times \vec{B}$$

**Python算法实现**:

```python
import numpy as np
from typing import List, Tuple, Callable
from scipy.integrate import quad, dblquad
from scipy.spatial.distance import cdist

class ElectromagneticField:
    """电磁场类"""
    def __init__(self, electric_field: Callable, magnetic_field: Callable,
                 current_density: Callable = None, charge_density: Callable = None):
        self.electric_field = electric_field
        self.magnetic_field = magnetic_field
        self.current_density = current_density or (lambda r, t: np.zeros(3))
        self.charge_density = charge_density or (lambda r, t: 0.0)
    
    def get_electric_field(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """获取电场"""
        return self.electric_field(position, time)
    
    def get_magnetic_field(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """获取磁场"""
        return self.magnetic_field(position, time)
    
    def get_current_density(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """获取电流密度"""
        return self.current_density(position, time)
    
    def get_charge_density(self, position: np.ndarray, time: float = 0.0) -> float:
        """获取电荷密度"""
        return self.charge_density(position, time)

def gauss_electric_law_integral(electric_field: Callable, surface_points: List[np.ndarray],
                               surface_normals: List[np.ndarray], time: float = 0.0) -> float:
    """
    计算高斯电场定律积分
    
    参数:
        electric_field: 电场函数
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time: 时间 (s)
    
    返回:
        电场通量 (V·m)
    """
    epsilon_0 = 8.854e-12  # F/m
    
    total_flux = 0.0
    for i in range(len(surface_points) - 1):
        # 计算相邻两点间的面积元
        point1 = surface_points[i]
        point2 = surface_points[i + 1]
        
        # 简化：假设每个线段贡献相等的通量
        mid_point = (point1 + point2) / 2
        normal = surface_normals[i]
        
        # 计算电场在法向量方向的投影
        E = electric_field(mid_point, time)
        flux_element = np.dot(E, normal)
        
        # 计算面积元大小（简化）
        area_element = np.linalg.norm(point2 - point1)
        total_flux += flux_element * area_element
    
    return total_flux

def gauss_magnetic_law_integral(magnetic_field: Callable, surface_points: List[np.ndarray],
                               surface_normals: List[np.ndarray], time: float = 0.0) -> float:
    """
    计算高斯磁场定律积分
    
    参数:
        magnetic_field: 磁场函数
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time: 时间 (s)
    
    返回:
        磁场通量 (Wb)
    """
    total_flux = 0.0
    for i in range(len(surface_points) - 1):
        # 计算相邻两点间的面积元
        point1 = surface_points[i]
        point2 = surface_points[i + 1]
        
        # 简化：假设每个线段贡献相等的通量
        mid_point = (point1 + point2) / 2
        normal = surface_normals[i]
        
        # 计算磁场在法向量方向的投影
        B = magnetic_field(mid_point, time)
        flux_element = np.dot(B, normal)
        
        # 计算面积元大小（简化）
        area_element = np.linalg.norm(point2 - point1)
        total_flux += flux_element * area_element
    
    return total_flux

def faraday_law_integral(electric_field: Callable, contour_points: List[np.ndarray],
                        magnetic_field: Callable, surface_points: List[np.ndarray],
                        surface_normals: List[np.ndarray], time: float = 0.0,
                        dt: float = 1e-6) -> float:
    """
    计算法拉第电磁感应定律积分
    
    参数:
        electric_field: 电场函数
        contour_points: 闭合路径上的点列表
        magnetic_field: 磁场函数
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        电动势 (V)
    """
    # 计算电场沿闭合路径的线积分
    emf_line = 0.0
    for i in range(len(contour_points) - 1):
        point1 = contour_points[i]
        point2 = contour_points[i + 1]
        
        # 计算路径元素
        dl = point2 - point1
        mid_point = (point1 + point2) / 2
        
        # 计算电场在路径方向的投影
        E = electric_field(mid_point, time)
        emf_element = np.dot(E, dl)
        emf_line += emf_element
    
    # 计算磁场通量的时间导数
    flux_t = gauss_magnetic_law_integral(magnetic_field, surface_points, surface_normals, time)
    flux_t_dt = gauss_magnetic_law_integral(magnetic_field, surface_points, surface_normals, time + dt)
    dflux_dt = (flux_t_dt - flux_t) / dt
    
    # 验证法拉第定律
    emf_faraday = -dflux_dt
    
    return emf_line, emf_faraday

def ampere_maxwell_law_integral(magnetic_field: Callable, contour_points: List[np.ndarray],
                              current_density: Callable, surface_points: List[np.ndarray],
                              surface_normals: List[np.ndarray], electric_field: Callable,
                              time: float = 0.0, dt: float = 1e-6) -> Tuple[float, float, float]:
    """
    计算安培-麦克斯韦定律积分
    
    参数:
        magnetic_field: 磁场函数
        contour_points: 闭合路径上的点列表
        current_density: 电流密度函数
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        electric_field: 电场函数
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        磁场线积分、传导电流、位移电流 (A)
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m
    epsilon_0 = 8.854e-12  # F/m
    
    # 计算磁场沿闭合路径的线积分
    b_line_integral = 0.0
    for i in range(len(contour_points) - 1):
        point1 = contour_points[i]
        point2 = contour_points[i + 1]
        
        # 计算路径元素
        dl = point2 - point1
        mid_point = (point1 + point2) / 2
        
        # 计算磁场在路径方向的投影
        B = magnetic_field(mid_point, time)
        b_element = np.dot(B, dl)
        b_line_integral += b_element
    
    # 计算传导电流
    conduction_current = 0.0
    for i in range(len(surface_points) - 1):
        point1 = surface_points[i]
        point2 = surface_points[i + 1]
        
        mid_point = (point1 + point2) / 2
        normal = surface_normals[i]
        
        # 计算电流密度在法向量方向的投影
        J = current_density(mid_point, time)
        current_element = np.dot(J, normal)
        
        # 计算面积元大小（简化）
        area_element = np.linalg.norm(point2 - point1)
        conduction_current += current_element * area_element
    
    # 计算位移电流
    flux_e_t = gauss_electric_law_integral(electric_field, surface_points, surface_normals, time)
    flux_e_t_dt = gauss_electric_law_integral(electric_field, surface_points, surface_normals, time + dt)
    dflux_e_dt = (flux_e_t_dt - flux_e_t) / dt
    displacement_current = epsilon_0 * dflux_e_dt
    
    return b_line_integral, conduction_current, displacement_current

def maxwell_equations_verification(em_field: ElectromagneticField,
                                 surface_points: List[np.ndarray],
                                 surface_normals: List[np.ndarray],
                                 contour_points: List[np.ndarray],
                                 time: float = 0.0, tolerance: float = 1e-6) -> bool:
    """
    验证麦克斯韦方程组
    
    参数:
        em_field: 电磁场对象
        surface_points: 曲面上的点列表
        surface_normals: 对应的法向量列表
        contour_points: 闭合路径上的点列表
        time: 时间 (s)
        tolerance: 容差
    
    返回:
        验证是否通过
    """
    # 验证高斯电场定律
    electric_flux = gauss_electric_law_integral(em_field.get_electric_field, 
                                               surface_points, surface_normals, time)
    
    # 计算封闭电荷（简化）
    total_charge = 0.0
    for point in surface_points:
        charge_density = em_field.get_charge_density(point, time)
        # 假设每个点代表一个体积元
        total_charge += charge_density
    
    epsilon_0 = 8.854e-12
    expected_flux = total_charge / epsilon_0
    
    if abs(electric_flux - expected_flux) > tolerance:
        return False
    
    # 验证高斯磁场定律
    magnetic_flux = gauss_magnetic_law_integral(em_field.get_magnetic_field,
                                               surface_points, surface_normals, time)
    
    if abs(magnetic_flux) > tolerance:
        return False
    
    # 验证法拉第定律
    emf_line, emf_faraday = faraday_law_integral(em_field.get_electric_field, contour_points,
                                                em_field.get_magnetic_field, surface_points,
                                                surface_normals, time)
    
    if abs(emf_line - emf_faraday) > tolerance:
        return False
    
    # 验证安培-麦克斯韦定律
    b_integral, conduction_current, displacement_current = ampere_maxwell_law_integral(
        em_field.get_magnetic_field, contour_points, em_field.get_current_density,
        surface_points, surface_normals, em_field.get_electric_field, time)
    
    mu_0 = 4 * np.pi * 1e-7
    expected_integral = mu_0 * (conduction_current + displacement_current)
    
    if abs(b_integral - expected_integral) > tolerance:
        return False
    
    return True

def maxwell_equations_example():
    """麦克斯韦方程组示例"""
    # 定义简单的电磁场
    def electric_field(r, t):
        # 简单的径向电场
        r_mag = np.linalg.norm(r)
        if r_mag == 0:
            return np.zeros(3)
        return r / (r_mag**3)
    
    def magnetic_field(r, t):
        # 简单的环形磁场
        x, y, z = r
        r_xy = np.sqrt(x**2 + y**2)
        if r_xy == 0:
            return np.zeros(3)
        return np.array([-y/r_xy, x/r_xy, 0])
    
    def current_density(r, t):
        # 简单的电流密度
        return np.array([0, 0, 1.0])
    
    def charge_density(r, t):
        # 简单的电荷密度
        return 1.0
    
    # 创建电磁场对象
    em_field = ElectromagneticField(electric_field, magnetic_field, current_density, charge_density)
    
    # 定义曲面和路径
    surface_points = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 0])
    ]
    
    surface_normals = [
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1])
    ]
    
    contour_points = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 0])
    ]
    
    # 计算各种积分
    electric_flux = gauss_electric_law_integral(electric_field, surface_points, surface_normals)
    magnetic_flux = gauss_magnetic_law_integral(magnetic_field, surface_points, surface_normals)
    emf_line, emf_faraday = faraday_law_integral(electric_field, contour_points,
                                                magnetic_field, surface_points, surface_normals)
    b_integral, conduction_current, displacement_current = ampere_maxwell_law_integral(
        magnetic_field, contour_points, current_density, surface_points, surface_normals, electric_field)
    
    # 验证麦克斯韦方程组
    verification_result = maxwell_equations_verification(em_field, surface_points, 
                                                        surface_normals, contour_points)
    
    print(f"电场通量: {electric_flux:.6e} V·m")
    print(f"磁场通量: {magnetic_flux:.6e} Wb")
    print(f"电动势线积分: {emf_line:.6e} V")
    print(f"电动势法拉第: {emf_faraday:.6e} V")
    print(f"磁场线积分: {b_integral:.6e} T·m")
    print(f"传导电流: {conduction_current:.6e} A")
    print(f"位移电流: {displacement_current:.6e} A")
    print(f"麦克斯韦方程组验证: {'通过' if verification_result else '失败'}")
    
    return (electric_flux, magnetic_flux, emf_line, emf_faraday, 
            b_integral, conduction_current, displacement_current, verification_result)

### 微分形式 / Differential Form

**形式化定义**: 麦克斯韦方程组的微分形式描述了电磁场在空间和时间上的局部变化规律。

**公理化定义**:
设 $\mathcal{ME}_D = \langle \mathcal{E}, \mathcal{B}, \mathcal{J}, \mathcal{Q}, \nabla, \partial_t \rangle$ 为麦克斯韦方程组微分系统，其中：

1. **电场集合**: $\mathcal{E}$ 为电场向量场集合
2. **磁场集合**: $\mathcal{B}$ 为磁场向量场集合
3. **电流密度集合**: $\mathcal{J}$ 为电流密度向量场集合
4. **电荷密度集合**: $\mathcal{Q}$ 为电荷密度标量场集合
5. **梯度算子**: $\nabla$ 为空间微分算子
6. **时间导数**: $\partial_t$ 为时间微分算子

**等价定义**:

1. **高斯电场定律**: $\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$
2. **高斯磁场定律**: $\nabla \cdot \vec{B} = 0$
3. **法拉第电磁感应定律**: $\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$
4. **安培-麦克斯韦定律**: $\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}$

**形式化定理**:

**定理2.5.4.4 (电荷连续性方程)**: 麦克斯韦方程组隐含电荷连续性方程
$$\nabla \cdot \vec{J} + \frac{\partial \rho}{\partial t} = 0$$

**定理2.5.4.5 (电磁波方程)**: 在真空中，电场和磁场满足波动方程
$$\nabla^2 \vec{E} - \frac{1}{c^2} \frac{\partial^2 \vec{E}}{\partial t^2} = 0$$
$$\nabla^2 \vec{B} - \frac{1}{c^2} \frac{\partial^2 \vec{B}}{\partial t^2} = 0$$

**定理2.5.4.6 (电磁场能量密度)**: 电磁场能量密度为
$$u = \frac{1}{2} \epsilon_0 E^2 + \frac{1}{2\mu_0} B^2$$

**Python算法实现**:

```python
import numpy as np
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist

def divergence(field_function: Callable, position: np.ndarray, 
               time: float = 0.0, dx: float = 1e-6) -> float:
    """
    计算向量场的散度
    
    参数:
        field_function: 向量场函数
        position: 位置向量
        time: 时间 (s)
        dx: 空间步长 (m)
    
    返回:
        散度值
    """
    x, y, z = position
    
    # 计算x方向偏导数
    field_x_plus = field_function(np.array([x + dx, y, z]), time)
    field_x_minus = field_function(np.array([x - dx, y, z]), time)
    dEx_dx = (field_x_plus[0] - field_x_minus[0]) / (2 * dx)
    
    # 计算y方向偏导数
    field_y_plus = field_function(np.array([x, y + dx, z]), time)
    field_y_minus = field_function(np.array([x, y - dx, z]), time)
    dEy_dy = (field_y_plus[1] - field_y_minus[1]) / (2 * dx)
    
    # 计算z方向偏导数
    field_z_plus = field_function(np.array([x, y, z + dx]), time)
    field_z_minus = field_function(np.array([x, y, z - dx]), time)
    dEz_dz = (field_z_plus[2] - field_z_minus[2]) / (2 * dx)
    
    return dEx_dx + dEy_dy + dEz_dz

def curl(field_function: Callable, position: np.ndarray, 
         time: float = 0.0, dx: float = 1e-6) -> np.ndarray:
    """
    计算向量场的旋度
    
    参数:
        field_function: 向量场函数
        position: 位置向量
        time: 时间 (s)
        dx: 空间步长 (m)
    
    返回:
        旋度向量
    """
    x, y, z = position
    
    # 计算各方向的偏导数
    # dEz/dy - dEy/dz
    field_y_plus = field_function(np.array([x, y + dx, z]), time)
    field_y_minus = field_function(np.array([x, y - dx, z]), time)
    field_z_plus = field_function(np.array([x, y, z + dx]), time)
    field_z_minus = field_function(np.array([x, y, z - dx]), time)
    
    curl_x = (field_z_plus[1] - field_z_minus[1]) / (2 * dx) - \
             (field_y_plus[2] - field_y_minus[2]) / (2 * dx)
    
    # dEx/dz - dEz/dx
    field_x_plus = field_function(np.array([x + dx, y, z]), time)
    field_x_minus = field_function(np.array([x - dx, y, z]), time)
    
    curl_y = (field_x_plus[2] - field_x_minus[2]) / (2 * dx) - \
             (field_z_plus[0] - field_z_minus[0]) / (2 * dx)
    
    # dEy/dx - dEx/dy
    curl_z = (field_y_plus[0] - field_y_minus[0]) / (2 * dx) - \
             (field_x_plus[1] - field_x_minus[1]) / (2 * dx)
    
    return np.array([curl_x, curl_y, curl_z])

def gauss_electric_law_differential(electric_field: Callable, charge_density: Callable,
                                   position: np.ndarray, time: float = 0.0) -> Tuple[float, float]:
    """
    验证高斯电场定律微分形式
    
    参数:
        electric_field: 电场函数
        charge_density: 电荷密度函数
        position: 位置向量
        time: 时间 (s)
    
    返回:
        电场散度、期望散度
    """
    epsilon_0 = 8.854e-12  # F/m
    
    # 计算电场散度
    div_E = divergence(electric_field, position, time)
    
    # 计算电荷密度
    rho = charge_density(position, time)
    
    # 期望的散度
    expected_div = rho / epsilon_0
    
    return div_E, expected_div

def gauss_magnetic_law_differential(magnetic_field: Callable, position: np.ndarray,
                                   time: float = 0.0) -> float:
    """
    验证高斯磁场定律微分形式
    
    参数:
        magnetic_field: 磁场函数
        position: 位置向量
        time: 时间 (s)
    
    返回:
        磁场散度
    """
    # 计算磁场散度
    div_B = divergence(magnetic_field, position, time)
    
    return div_B

def faraday_law_differential(electric_field: Callable, magnetic_field: Callable,
                            position: np.ndarray, time: float = 0.0,
                            dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    验证法拉第电磁感应定律微分形式
    
    参数:
        electric_field: 电场函数
        magnetic_field: 磁场函数
        position: 位置向量
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        电场旋度、磁场时间导数的负值
    """
    # 计算电场旋度
    curl_E = curl(electric_field, position, time)
    
    # 计算磁场时间导数
    B_t = magnetic_field(position, time)
    B_t_dt = magnetic_field(position, time + dt)
    dB_dt = (B_t_dt - B_t) / dt
    
    return curl_E, -dB_dt

def ampere_maxwell_law_differential(magnetic_field: Callable, electric_field: Callable,
                                   current_density: Callable, position: np.ndarray,
                                   time: float = 0.0, dt: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    验证安培-麦克斯韦定律微分形式
    
    参数:
        magnetic_field: 磁场函数
        electric_field: 电场函数
        current_density: 电流密度函数
        position: 位置向量
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        磁场旋度、期望旋度
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m
    epsilon_0 = 8.854e-12  # F/m
    
    # 计算磁场旋度
    curl_B = curl(magnetic_field, position, time)
    
    # 计算传导电流密度
    J = current_density(position, time)
    
    # 计算位移电流密度
    E_t = electric_field(position, time)
    E_t_dt = electric_field(position, time + dt)
    dE_dt = (E_t_dt - E_t) / dt
    
    # 期望的旋度
    expected_curl = mu_0 * J + mu_0 * epsilon_0 * dE_dt
    
    return curl_B, expected_curl

def charge_continuity_equation(current_density: Callable, charge_density: Callable,
                              position: np.ndarray, time: float = 0.0,
                              dt: float = 1e-6) -> Tuple[float, float]:
    """
    验证电荷连续性方程
    
    参数:
        current_density: 电流密度函数
        charge_density: 电荷密度函数
        position: 位置向量
        time: 时间 (s)
        dt: 时间步长 (s)
    
    返回:
        电流密度散度、电荷密度时间导数的负值
    """
    # 计算电流密度散度
    div_J = divergence(current_density, position, time)
    
    # 计算电荷密度时间导数
    rho_t = charge_density(position, time)
    rho_t_dt = charge_density(position, time + dt)
    drho_dt = (rho_t_dt - rho_t) / dt
    
    return div_J, -drho_dt

def electromagnetic_wave_equation(field_function: Callable, position: np.ndarray,
                                 time: float = 0.0, dx: float = 1e-6, dt: float = 1e-6) -> np.ndarray:
    """
    计算电磁波方程
    
    参数:
        field_function: 场函数（电场或磁场）
        position: 位置向量
        time: 时间 (s)
        dx: 空间步长 (m)
        dt: 时间步长 (s)
    
    返回:
        波动方程左侧值
    """
    c = 3e8  # m/s
    
    x, y, z = position
    
    # 计算拉普拉斯算子
    field_center = field_function(position, time)
    field_x_plus = field_function(np.array([x + dx, y, z]), time)
    field_x_minus = field_function(np.array([x - dx, y, z]), time)
    field_y_plus = field_function(np.array([x, y + dx, z]), time)
    field_y_minus = field_function(np.array([x, y - dx, z]), time)
    field_z_plus = field_function(np.array([x, y, z + dx]), time)
    field_z_minus = field_function(np.array([x, y, z - dx]), time)
    
    laplacian = (field_x_plus + field_x_minus + field_y_plus + field_y_minus + 
                 field_z_plus + field_z_minus - 6 * field_center) / (dx**2)
    
    # 计算时间二阶导数
    field_t_plus = field_function(position, time + dt)
    field_t_minus = field_function(position, time - dt)
    d2field_dt2 = (field_t_plus + field_t_minus - 2 * field_center) / (dt**2)
    
    # 波动方程
    wave_equation = laplacian - (1 / c**2) * d2field_dt2
    
    return wave_equation

def electromagnetic_energy_density(electric_field: Callable, magnetic_field: Callable,
                                  position: np.ndarray, time: float = 0.0) -> float:
    """
    计算电磁场能量密度
    
    参数:
        electric_field: 电场函数
        magnetic_field: 磁场函数
        position: 位置向量
        time: 时间 (s)
    
    返回:
        能量密度 (J/m³)
    """
    epsilon_0 = 8.854e-12  # F/m
    mu_0 = 4 * np.pi * 1e-7  # H/m
    
    # 获取电场和磁场
    E = electric_field(position, time)
    B = magnetic_field(position, time)
    
    # 计算能量密度
    E_squared = np.dot(E, E)
    B_squared = np.dot(B, B)
    
    energy_density = 0.5 * epsilon_0 * E_squared + 0.5 / mu_0 * B_squared
    
    return energy_density

def maxwell_differential_verification(em_field: ElectromagneticField, position: np.ndarray,
                                     time: float = 0.0, tolerance: float = 1e-6) -> bool:
    """
    验证麦克斯韦方程组微分形式
    
    参数:
        em_field: 电磁场对象
        position: 位置向量
        time: 时间 (s)
        tolerance: 容差
    
    返回:
        验证是否通过
    """
    # 验证高斯电场定律
    div_E, expected_div_E = gauss_electric_law_differential(
        em_field.get_electric_field, em_field.get_charge_density, position, time)
    
    if abs(div_E - expected_div_E) > tolerance:
        return False
    
    # 验证高斯磁场定律
    div_B = gauss_magnetic_law_differential(em_field.get_magnetic_field, position, time)
    
    if abs(div_B) > tolerance:
        return False
    
    # 验证法拉第定律
    curl_E, neg_dB_dt = faraday_law_differential(
        em_field.get_electric_field, em_field.get_magnetic_field, position, time)
    
    if np.any(np.abs(curl_E - neg_dB_dt) > tolerance):
        return False
    
    # 验证安培-麦克斯韦定律
    curl_B, expected_curl_B = ampere_maxwell_law_differential(
        em_field.get_magnetic_field, em_field.get_electric_field, 
        em_field.get_current_density, position, time)
    
    if np.any(np.abs(curl_B - expected_curl_B) > tolerance):
        return False
    
    # 验证电荷连续性方程
    div_J, neg_drho_dt = charge_continuity_equation(
        em_field.get_current_density, em_field.get_charge_density, position, time)
    
    if abs(div_J - neg_drho_dt) > tolerance:
        return False
    
    return True

def maxwell_differential_example():
    """麦克斯韦方程组微分形式示例"""
    # 定义简单的电磁场
    def electric_field(r, t):
        # 简单的平面波电场
        x, y, z = r
        k = 2 * np.pi / 1e-6  # 波数
        omega = 2 * np.pi * 3e14  # 角频率
        return np.array([np.cos(k * x - omega * t), 0, 0])
    
    def magnetic_field(r, t):
        # 对应的磁场
        x, y, z = r
        k = 2 * np.pi / 1e-6
        omega = 2 * np.pi * 3e14
        c = 3e8
        return np.array([0, np.cos(k * x - omega * t) / c, 0])
    
    def current_density(r, t):
        # 零电流密度
        return np.zeros(3)
    
    def charge_density(r, t):
        # 零电荷密度
        return 0.0
    
    # 创建电磁场对象
    em_field = ElectromagneticField(electric_field, magnetic_field, current_density, charge_density)
    
    # 选择测试点
    test_position = np.array([1e-6, 0, 0])  # 1微米处
    test_time = 0.0
    
    # 验证麦克斯韦方程组
    verification_result = maxwell_differential_verification(em_field, test_position, test_time)
    
    # 计算各种微分量
    div_E, expected_div_E = gauss_electric_law_differential(
        electric_field, charge_density, test_position, test_time)
    div_B = gauss_magnetic_law_differential(magnetic_field, test_position, test_time)
    curl_E, neg_dB_dt = faraday_law_differential(electric_field, magnetic_field, 
                                                test_position, test_time)
    curl_B, expected_curl_B = ampere_maxwell_law_differential(magnetic_field, electric_field,
                                                             current_density, test_position, test_time)
    
    # 计算能量密度
    energy_density = electromagnetic_energy_density(electric_field, magnetic_field,
                                                   test_position, test_time)
    
    print(f"电场散度: {div_E:.6e}")
    print(f"期望电场散度: {expected_div_E:.6e}")
    print(f"磁场散度: {div_B:.6e}")
    print(f"电场旋度: {curl_E}")
    print(f"磁场时间导数负值: {neg_dB_dt}")
    print(f"磁场旋度: {curl_B}")
    print(f"期望磁场旋度: {expected_curl_B}")
    print(f"电磁场能量密度: {energy_density:.6e} J/m³")
    print(f"麦克斯韦方程组微分形式验证: {'通过' if verification_result else '失败'}")
    
    return (div_E, div_B, curl_E, curl_B, energy_density, verification_result)

### 边界条件 / Boundary Conditions

**形式化定义**: 麦克斯韦方程组的边界条件描述了电磁场在不同介质界面上的连续性关系。

**公理化定义**:
设 $\mathcal{BC} = \langle \mathcal{S}, \mathcal{E}, \mathcal{B}, \mathcal{D}, \mathcal{H}, \mathcal{J}_s, \mathcal{Q}_s \rangle$ 为边界条件系统，其中：

1. **界面集合**: $\mathcal{S}$ 为介质界面集合
2. **电场集合**: $\mathcal{E}$ 为电场向量场集合
3. **磁场集合**: $\mathcal{B}$ 为磁场向量场集合
4. **电位移场集合**: $\mathcal{D}$ 为电位移场向量场集合
5. **磁场强度集合**: $\mathcal{H}$ 为磁场强度向量场集合
6. **面电流密度集合**: $\mathcal{J}_s$ 为面电流密度向量场集合
7. **面电荷密度集合**: $\mathcal{Q}_s$ 为面电荷密度标量场集合

**等价定义**:

1. **电场切向分量连续性**: $\hat{n} \times (\vec{E}_2 - \vec{E}_1) = 0$
2. **磁场切向分量连续性**: $\hat{n} \times (\vec{H}_2 - \vec{H}_1) = \vec{J}_s$
3. **电位移场法向分量连续性**: $\hat{n} \cdot (\vec{D}_2 - \vec{D}_1) = \sigma_s$
4. **磁场法向分量连续性**: $\hat{n} \cdot (\vec{B}_2 - \vec{B}_1) = 0$

**形式化定理**:

**定理2.5.4.7 (边界条件一致性)**: 边界条件与麦克斯韦方程组保持一致
$$\nabla \cdot \vec{J}_s + \frac{\partial \sigma_s}{\partial t} = 0$$

**定理2.5.4.8 (反射折射定律)**: 电磁波在界面上的反射和折射满足斯涅尔定律
$$n_1 \sin \theta_1 = n_2 \sin \theta_2$$

**定理2.5.4.9 (菲涅尔公式)**: 电磁波的反射和透射系数满足菲涅尔公式
$$r_s = \frac{n_1 \cos \theta_1 - n_2 \cos \theta_2}{n_1 \cos \theta_1 + n_2 \cos \theta_2}$$

**Python算法实现**:

```python
import numpy as np
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist

class BoundaryInterface:
    """边界界面类"""
    def __init__(self, normal_vector: np.ndarray, position: np.ndarray,
                 medium1_properties: dict, medium2_properties: dict):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.position = np.array(position)
        self.medium1_properties = medium1_properties
        self.medium2_properties = medium2_properties
    
    def get_normal(self) -> np.ndarray:
        """获取法向量"""
        return self.normal_vector.copy()
    
    def get_position(self) -> np.ndarray:
        """获取位置"""
        return self.position.copy()

def electric_field_tangential_continuity(electric_field1: np.ndarray, electric_field2: np.ndarray,
                                        normal_vector: np.ndarray) -> np.ndarray:
    """
    验证电场切向分量连续性
    
    参数:
        electric_field1, electric_field2: 两侧电场
        normal_vector: 法向量
    
    返回:
        切向分量差值
    """
    # 计算切向分量
    tangential1 = electric_field1 - np.dot(electric_field1, normal_vector) * normal_vector
    tangential2 = electric_field2 - np.dot(electric_field2, normal_vector) * normal_vector
    
    # 计算差值
    difference = tangential2 - tangential1
    
    return difference

def magnetic_field_tangential_continuity(magnetic_field1: np.ndarray, magnetic_field2: np.ndarray,
                                        normal_vector: np.ndarray, surface_current: np.ndarray = None) -> np.ndarray:
    """
    验证磁场切向分量连续性
    
    参数:
        magnetic_field1, magnetic_field2: 两侧磁场
        normal_vector: 法向量
        surface_current: 面电流密度
    
    返回:
        切向分量差值
    """
    # 计算切向分量
    tangential1 = magnetic_field1 - np.dot(magnetic_field1, normal_vector) * normal_vector
    tangential2 = magnetic_field2 - np.dot(magnetic_field2, normal_vector) * normal_vector
    
    # 计算差值
    difference = tangential2 - tangential1
    
    # 如果有面电流，需要考虑其贡献
    if surface_current is not None:
        # 面电流产生的磁场切向分量
        current_contribution = np.cross(normal_vector, surface_current)
        difference -= current_contribution
    
    return difference

def electric_displacement_normal_continuity(displacement1: np.ndarray, displacement2: np.ndarray,
                                          normal_vector: np.ndarray, surface_charge: float = 0.0) -> float:
    """
    验证电位移场法向分量连续性
    
    参数:
        displacement1, displacement2: 两侧电位移场
        normal_vector: 法向量
        surface_charge: 面电荷密度
    
    返回:
        法向分量差值
    """
    # 计算法向分量
    normal1 = np.dot(displacement1, normal_vector)
    normal2 = np.dot(displacement2, normal_vector)
    
    # 计算差值
    difference = normal2 - normal1 - surface_charge
    
    return difference

def magnetic_field_normal_continuity(magnetic_field1: np.ndarray, magnetic_field2: np.ndarray,
                                    normal_vector: np.ndarray) -> float:
    """
    验证磁场法向分量连续性
    
    参数:
        magnetic_field1, magnetic_field2: 两侧磁场
        normal_vector: 法向量
    
    返回:
        法向分量差值
    """
    # 计算法向分量
    normal1 = np.dot(magnetic_field1, normal_vector)
    normal2 = np.dot(magnetic_field2, normal_vector)
    
    # 计算差值
    difference = normal2 - normal1
    
    return difference

def snell_law(incident_angle: float, refractive_index1: float, refractive_index2: float) -> float:
    """
    计算折射角（斯涅尔定律）
    
    参数:
        incident_angle: 入射角 (弧度)
        refractive_index1, refractive_index2: 折射率
    
    返回:
        折射角 (弧度)
    """
    sin_refracted = (refractive_index1 / refractive_index2) * np.sin(incident_angle)
    
    # 检查是否发生全反射
    if abs(sin_refracted) > 1:
        return np.nan  # 全反射
    
    return np.arcsin(sin_refracted)

def fresnel_reflection_coefficient(incident_angle: float, refractive_index1: float, 
                                  refractive_index2: float, polarization: str = 's') -> complex:
    """
    计算菲涅尔反射系数
    
    参数:
        incident_angle: 入射角 (弧度)
        refractive_index1, refractive_index2: 折射率
        polarization: 偏振方向 ('s' 或 'p')
    
    返回:
        反射系数（复数）
    """
    # 计算折射角
    refracted_angle = snell_law(incident_angle, refractive_index1, refractive_index2)
    
    if np.isnan(refracted_angle):
        return 1.0  # 全反射
    
    if polarization == 's':
        # s偏振（电场垂直于入射面）
        numerator = refractive_index1 * np.cos(incident_angle) - refractive_index2 * np.cos(refracted_angle)
        denominator = refractive_index1 * np.cos(incident_angle) + refractive_index2 * np.cos(refracted_angle)
    else:
        # p偏振（电场平行于入射面）
        numerator = refractive_index2 * np.cos(incident_angle) - refractive_index1 * np.cos(refracted_angle)
        denominator = refractive_index2 * np.cos(incident_angle) + refractive_index1 * np.cos(refracted_angle)
    
    return numerator / denominator

def fresnel_transmission_coefficient(incident_angle: float, refractive_index1: float,
                                    refractive_index2: float, polarization: str = 's') -> complex:
    """
    计算菲涅尔透射系数
    
    参数:
        incident_angle: 入射角 (弧度)
        refractive_index1, refractive_index2: 折射率
        polarization: 偏振方向 ('s' 或 'p')
    
    返回:
        透射系数（复数）
    """
    # 计算折射角
    refracted_angle = snell_law(incident_angle, refractive_index1, refractive_index2)
    
    if np.isnan(refracted_angle):
        return 0.0  # 全反射
    
    if polarization == 's':
        # s偏振
        numerator = 2 * refractive_index1 * np.cos(incident_angle)
        denominator = refractive_index1 * np.cos(incident_angle) + refractive_index2 * np.cos(refracted_angle)
    else:
        # p偏振
        numerator = 2 * refractive_index1 * np.cos(incident_angle)
        denominator = refractive_index2 * np.cos(incident_angle) + refractive_index1 * np.cos(refracted_angle)
    
    return numerator / denominator

def boundary_conditions_verification(interface: BoundaryInterface,
                                   fields_medium1: dict, fields_medium2: dict,
                                   surface_current: np.ndarray = None,
                                   surface_charge: float = 0.0,
                                   tolerance: float = 1e-6) -> bool:
    """
    验证边界条件
    
    参数:
        interface: 边界界面对象
        fields_medium1, fields_medium2: 两侧场量
        surface_current: 面电流密度
        surface_charge: 面电荷密度
        tolerance: 容差
    
    返回:
        验证是否通过
    """
    normal = interface.get_normal()
    
    # 验证电场切向分量连续性
    e_tangential_diff = electric_field_tangential_continuity(
        fields_medium1['electric_field'], fields_medium2['electric_field'], normal)
    
    if np.any(np.abs(e_tangential_diff) > tolerance):
        return False
    
    # 验证磁场切向分量连续性
    h_tangential_diff = magnetic_field_tangential_continuity(
        fields_medium1['magnetic_field'], fields_medium2['magnetic_field'], 
        normal, surface_current)
    
    if np.any(np.abs(h_tangential_diff) > tolerance):
        return False
    
    # 验证电位移场法向分量连续性
    d_normal_diff = electric_displacement_normal_continuity(
        fields_medium1['displacement'], fields_medium2['displacement'], 
        normal, surface_charge)
    
    if abs(d_normal_diff) > tolerance:
        return False
    
    # 验证磁场法向分量连续性
    b_normal_diff = magnetic_field_normal_continuity(
        fields_medium1['magnetic_field'], fields_medium2['magnetic_field'], normal)
    
    if abs(b_normal_diff) > tolerance:
        return False
    
    return True

def total_internal_reflection_angle(refractive_index1: float, refractive_index2: float) -> float:
    """
    计算全反射临界角
    
    参数:
        refractive_index1, refractive_index2: 折射率
    
    返回:
        临界角 (弧度)
    """
    if refractive_index1 <= refractive_index2:
        return np.nan  # 不会发生全反射
    
    return np.arcsin(refractive_index2 / refractive_index1)

def boundary_conditions_example():
    """边界条件示例"""
    # 定义界面
    normal_vector = np.array([0, 0, 1])  # z方向法向量
    position = np.array([0, 0, 0])
    
    # 介质属性
    medium1_properties = {
        'permittivity': 8.854e-12,  # 真空
        'permeability': 4 * np.pi * 1e-7
    }
    
    medium2_properties = {
        'permittivity': 2.2 * 8.854e-12,  # 玻璃
        'permeability': 4 * np.pi * 1e-7
    }
    
    interface = BoundaryInterface(normal_vector, position, medium1_properties, medium2_properties)
    
    # 定义两侧场量
    fields_medium1 = {
        'electric_field': np.array([1.0, 0, 0]),
        'magnetic_field': np.array([0, 1.0, 0]),
        'displacement': np.array([1.0, 0, 0])
    }
    
    fields_medium2 = {
        'electric_field': np.array([1.0, 0, 0]),
        'magnetic_field': np.array([0, 1.0, 0]),
        'displacement': np.array([2.2, 0, 0])
    }
    
    # 验证边界条件
    verification_result = boundary_conditions_verification(interface, fields_medium1, fields_medium2)
    
    # 计算反射和折射
    incident_angle = np.pi / 6  # 30度
    refractive_index1 = 1.0  # 空气
    refractive_index2 = 1.5  # 玻璃
    
    refracted_angle = snell_law(incident_angle, refractive_index1, refractive_index2)
    reflection_coefficient_s = fresnel_reflection_coefficient(incident_angle, refractive_index1, refractive_index2, 's')
    reflection_coefficient_p = fresnel_reflection_coefficient(incident_angle, refractive_index1, refractive_index2, 'p')
    transmission_coefficient_s = fresnel_transmission_coefficient(incident_angle, refractive_index1, refractive_index2, 's')
    transmission_coefficient_p = fresnel_transmission_coefficient(incident_angle, refractive_index1, refractive_index2, 'p')
    
    # 计算全反射临界角
    critical_angle = total_internal_reflection_angle(refractive_index2, refractive_index1)
    
    print(f"边界条件验证: {'通过' if verification_result else '失败'}")
    print(f"入射角: {np.degrees(incident_angle):.2f}°")
    print(f"折射角: {np.degrees(refracted_angle):.2f}°")
    print(f"s偏振反射系数: {reflection_coefficient_s:.6f}")
    print(f"p偏振反射系数: {reflection_coefficient_p:.6f}")
    print(f"s偏振透射系数: {transmission_coefficient_s:.6f}")
    print(f"p偏振透射系数: {transmission_coefficient_p:.6f}")
    print(f"全反射临界角: {np.degrees(critical_angle):.2f}°")
    
    return (verification_result, refracted_angle, reflection_coefficient_s, 
            reflection_coefficient_p, transmission_coefficient_s, transmission_coefficient_p, critical_angle)
```

---

## 2.5.5 电磁波 / Electromagnetic Waves

### 波动方程 / Wave Equation

**形式化定义**: 电磁波方程描述了电磁场在空间和时间中的传播规律，是麦克斯韦方程组的直接推论。

**公理化定义**:
设 $\mathcal{EW} = \langle \mathcal{E}, \mathcal{B}, \mathcal{V}, \mathcal{T}, \mathcal{S}, \mathcal{W} \rangle$ 为电磁波系统，其中：

1. **电场函数**: $\mathcal{E}: \mathbb{R}^3 \times \mathbb{R} \rightarrow \mathbb{R}^3$ 为电场函数
2. **磁场函数**: $\mathcal{B}: \mathbb{R}^3 \times \mathbb{R} \rightarrow \mathbb{R}^3$ 为磁场函数
3. **传播速度**: $\mathcal{V} = c = \frac{1}{\sqrt{\mu_0 \epsilon_0}}$ 为真空光速
4. **时间变量**: $\mathcal{T} = \mathbb{R}$ 为时间集合
5. **空间变量**: $\mathcal{S} = \mathbb{R}^3$ 为空间集合
6. **波动算子**: $\mathcal{W} = \nabla^2 - \frac{1}{c^2} \frac{\partial^2}{\partial t^2}$ 为达朗贝尔算子

**等价定义**:

1. **电场波动方程**: $\nabla^2 \vec{E} - \frac{1}{c^2} \frac{\partial^2 \vec{E}}{\partial t^2} = 0$
2. **磁场波动方程**: $\nabla^2 \vec{B} - \frac{1}{c^2} \frac{\partial^2 \vec{B}}{\partial t^2} = 0$
3. **矢量势波动方程**: $\nabla^2 \vec{A} - \frac{1}{c^2} \frac{\partial^2 \vec{A}}{\partial t^2} = 0$
4. **标量势波动方程**: $\nabla^2 \phi - \frac{1}{c^2} \frac{\partial^2 \phi}{\partial t^2} = 0$

**形式化定理**:

**定理2.5.5.1 (电磁波传播速度)**: 电磁波在真空中的传播速度为光速
$$c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} \approx 3 \times 10^8 \text{ m/s}$$

**定理2.5.5.2 (电磁波横波性)**: 电磁波是横波，电场和磁场垂直于传播方向
$$\vec{k} \cdot \vec{E} = 0, \quad \vec{k} \cdot \vec{B} = 0$$

**定理2.5.5.3 (电磁波正交性)**: 电场、磁场和传播方向三者正交
$$\vec{E} \times \vec{B} = \frac{1}{c} \vec{k}$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple
from scipy.spatial.distance import cdist

class ElectromagneticWave:
    """电磁波类"""
    def __init__(self, frequency: float, wavelength: float = None, 
                 amplitude: float = 1.0, direction: np.ndarray = None):
        self.frequency = frequency
        self.wavelength = wavelength or (3e8 / frequency)
        self.amplitude = amplitude
        self.direction = direction or np.array([1, 0, 0])
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.wavenumber = 2 * np.pi / self.wavelength
        self.angular_frequency = 2 * np.pi * frequency
    
    def get_frequency(self) -> float:
        """获取频率"""
        return self.frequency
    
    def get_wavelength(self) -> float:
        """获取波长"""
        return self.wavelength
    
    def get_wavenumber(self) -> float:
        """获取波数"""
        return self.wavenumber
    
    def get_angular_frequency(self) -> float:
        """获取角频率"""
        return self.angular_frequency

def wave_equation_operator(field_function: Callable, position: np.ndarray, 
                          time: float, dx: float = 1e-6, dt: float = 1e-6) -> np.ndarray:
    """
    计算波动方程算子
    
    参数:
        field_function: 场函数
        position: 位置向量
        time: 时间
        dx: 空间步长
        dt: 时间步长
    
    返回:
        波动方程算子值
    """
    # 计算拉普拉斯算子
    laplacian = np.zeros(3)
    for i in range(3):
        pos_plus = position.copy()
        pos_minus = position.copy()
        pos_plus[i] += dx
        pos_minus[i] -= dx
        
        laplacian[i] = (field_function(pos_plus, time) - 2 * field_function(position, time) + 
                       field_function(pos_minus, time)) / (dx**2)
    
    # 计算时间二阶导数
    time_plus = time + dt
    time_minus = time - dt
    time_derivative = (field_function(position, time_plus) - 2 * field_function(position, time) + 
                      field_function(position, time_minus)) / (dt**2)
    
    # 波动方程算子
    wave_operator = laplacian - (1 / (3e8**2)) * time_derivative
    
    return wave_operator

def electromagnetic_wave_speed(permittivity: float = 8.854e-12, 
                             permeability: float = 4 * np.pi * 1e-7) -> float:
    """
    计算电磁波传播速度
    
    参数:
        permittivity: 介电常数
        permeability: 磁导率
    
    返回:
        传播速度 (m/s)
    """
    return 1.0 / np.sqrt(permittivity * permeability)

def wave_equation_verification(wave: ElectromagneticWave, position: np.ndarray, 
                             time: float, tolerance: float = 1e-6) -> bool:
    """
    验证波动方程
    
    参数:
        wave: 电磁波对象
        position: 位置向量
        time: 时间
        tolerance: 容差
    
    返回:
        是否满足波动方程
    """
    def electric_field_func(pos, t):
        k_dot_r = np.dot(wave.wavenumber * wave.direction, pos)
        omega_t = wave.angular_frequency * t
        phase = k_dot_r - omega_t
        return wave.amplitude * np.array([0, np.cos(phase), 0])
    
    def magnetic_field_func(pos, t):
        k_dot_r = np.dot(wave.wavenumber * wave.direction, pos)
        omega_t = wave.angular_frequency * t
        phase = k_dot_r - omega_t
        return (wave.amplitude / 3e8) * np.array([0, 0, np.cos(phase)])
    
    # 验证电场波动方程
    e_wave_operator = wave_equation_operator(electric_field_func, position, time)
    e_satisfied = np.all(np.abs(e_wave_operator) < tolerance)
    
    # 验证磁场波动方程
    b_wave_operator = wave_equation_operator(magnetic_field_func, position, time)
    b_satisfied = np.all(np.abs(b_wave_operator) < tolerance)
    
    return e_satisfied and b_satisfied

def wave_equation_example():
    """波动方程示例"""
    # 创建电磁波
    frequency = 1e9  # 1 GHz
    wave = ElectromagneticWave(frequency)
    
    # 计算传播速度
    speed = electromagnetic_wave_speed()
    
    # 验证波动方程
    position = np.array([1.0, 2.0, 3.0])
    time = 1e-9
    verification_result = wave_equation_verification(wave, position, time)
    
    print(f"频率: {wave.frequency:.2e} Hz")
    print(f"波长: {wave.wavelength:.2e} m")
    print(f"波数: {wave.wavenumber:.2e} m⁻¹")
    print(f"角频率: {wave.angular_frequency:.2e} rad/s")
    print(f"传播速度: {speed:.2e} m/s")
    print(f"波动方程验证: {'通过' if verification_result else '失败'}")
    
    return wave, speed, verification_result
```

### 平面波 / Plane Waves

**形式化定义**: 平面波是电磁波的一种基本形式，其波前为平面，在传播过程中保持恒定相位。

**公理化定义**:
设 $\mathcal{PW} = \langle \mathcal{A}, \mathcal{K}, \mathcal{\Omega}, \mathcal{\Phi}, \mathcal{P} \rangle$ 为平面波系统，其中：

1. **振幅向量**: $\mathcal{A} \in \mathbb{R}^3$ 为电场振幅向量
2. **波矢**: $\mathcal{K} \in \mathbb{R}^3$ 为传播方向波矢
3. **角频率**: $\mathcal{\Omega} \in \mathbb{R}^+$ 为角频率
4. **相位函数**: $\mathcal{\Phi}: \mathbb{R}^3 \times \mathbb{R} \rightarrow \mathbb{R}$ 为相位函数
5. **偏振向量**: $\mathcal{P} \in \mathbb{R}^3$ 为偏振方向向量

**等价定义**:

1. **电场表达式**: $\vec{E}(\vec{r}, t) = \vec{A} \cos(\vec{k} \cdot \vec{r} - \omega t + \phi_0)$
2. **磁场表达式**: $\vec{B}(\vec{r}, t) = \frac{1}{c} \hat{k} \times \vec{E}(\vec{r}, t)$
3. **复数形式**: $\vec{E}(\vec{r}, t) = \vec{A} e^{i(\vec{k} \cdot \vec{r} - \omega t + \phi_0)}$
4. **相位函数**: $\phi(\vec{r}, t) = \vec{k} \cdot \vec{r} - \omega t + \phi_0$

**形式化定理**:

**定理2.5.5.4 (平面波色散关系)**: 平面波满足色散关系
$$\omega = c|\vec{k}|$$

**定理2.5.5.5 (平面波能量密度)**: 平面波的能量密度为常数
$$u = \frac{1}{2}(\epsilon_0 E^2 + \frac{B^2}{\mu_0}) = \text{常数}$$

**定理2.5.5.6 (平面波动量密度)**: 平面波的动量密度与能量密度关系
$$\vec{g} = \frac{1}{c^2} \vec{S} = \frac{u}{c} \hat{k}$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple
import cmath

class PlaneWave:
    """平面波类"""
    def __init__(self, amplitude: np.ndarray, wavevector: np.ndarray, 
                 angular_frequency: float, phase_shift: float = 0.0):
        self.amplitude = np.array(amplitude)
        self.wavevector = np.array(wavevector)
        self.angular_frequency = angular_frequency
        self.phase_shift = phase_shift
        self.wavenumber = np.linalg.norm(wavevector)
        self.direction = wavevector / self.wavenumber if self.wavenumber > 0 else np.array([1, 0, 0])
    
    def get_amplitude(self) -> np.ndarray:
        """获取振幅"""
        return self.amplitude.copy()
    
    def get_wavevector(self) -> np.ndarray:
        """获取波矢"""
        return self.wavevector.copy()
    
    def get_direction(self) -> np.ndarray:
        """获取传播方向"""
        return self.direction.copy()

def plane_wave_electric_field(wave: PlaneWave, position: np.ndarray, time: float) -> np.ndarray:
    """
    计算平面波电场
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
    
    返回:
        电场向量
    """
    phase = np.dot(wave.wavevector, position) - wave.angular_frequency * time + wave.phase_shift
    return wave.amplitude * np.cos(phase)

def plane_wave_magnetic_field(wave: PlaneWave, position: np.ndarray, time: float) -> np.ndarray:
    """
    计算平面波磁场
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
    
    返回:
        磁场向量
    """
    electric_field = plane_wave_electric_field(wave, position, time)
    return np.cross(wave.direction, electric_field) / 3e8

def plane_wave_phase(wave: PlaneWave, position: np.ndarray, time: float) -> float:
    """
    计算平面波相位
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
    
    返回:
        相位 (弧度)
    """
    return np.dot(wave.wavevector, position) - wave.angular_frequency * time + wave.phase_shift

def dispersion_relation(wavenumber: float, speed: float = 3e8) -> float:
    """
    计算色散关系
    
    参数:
        wavenumber: 波数
        speed: 传播速度
    
    返回:
        角频率
    """
    return speed * wavenumber

def plane_wave_energy_density(wave: PlaneWave, position: np.ndarray, time: float) -> float:
    """
    计算平面波能量密度
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
    
    返回:
        能量密度 (J/m³)
    """
    electric_field = plane_wave_electric_field(wave, position, time)
    magnetic_field = plane_wave_magnetic_field(wave, position, time)
    
    epsilon_0 = 8.854e-12
    mu_0 = 4 * np.pi * 1e-7
    
    energy_density = 0.5 * (epsilon_0 * np.dot(electric_field, electric_field) + 
                           np.dot(magnetic_field, magnetic_field) / mu_0)
    
    return energy_density

def plane_wave_momentum_density(wave: PlaneWave, position: np.ndarray, time: float) -> np.ndarray:
    """
    计算平面波动量密度
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
    
    返回:
        动量密度向量 (kg/(m²·s))
    """
    energy_density = plane_wave_energy_density(wave, position, time)
    return (energy_density / 3e8) * wave.direction

def plane_wave_verification(wave: PlaneWave, position: np.ndarray, time: float, 
                           tolerance: float = 1e-6) -> bool:
    """
    验证平面波性质
    
    参数:
        wave: 平面波对象
        position: 位置向量
        time: 时间
        tolerance: 容差
    
    返回:
        是否满足平面波性质
    """
    # 验证横波性
    electric_field = plane_wave_electric_field(wave, position, time)
    magnetic_field = plane_wave_magnetic_field(wave, position, time)
    
    e_transverse = abs(np.dot(wave.direction, electric_field)) < tolerance
    b_transverse = abs(np.dot(wave.direction, magnetic_field)) < tolerance
    
    # 验证正交性
    e_b_orthogonal = abs(np.dot(electric_field, magnetic_field)) < tolerance
    
    # 验证色散关系
    calculated_frequency = dispersion_relation(wave.wavenumber)
    frequency_match = abs(calculated_frequency - wave.angular_frequency) < tolerance
    
    return e_transverse and b_transverse and e_b_orthogonal and frequency_match

def plane_wave_example():
    """平面波示例"""
    # 创建平面波
    amplitude = np.array([0, 1.0, 0])  # y方向偏振
    wavevector = np.array([2 * np.pi / 0.3, 0, 0])  # x方向传播，波长0.3m
    angular_frequency = 2 * np.pi * 1e9  # 1 GHz
    
    wave = PlaneWave(amplitude, wavevector, angular_frequency)
    
    # 计算场量
    position = np.array([0.1, 0.2, 0.3])
    time = 1e-9
    
    electric_field = plane_wave_electric_field(wave, position, time)
    magnetic_field = plane_wave_magnetic_field(wave, position, time)
    phase = plane_wave_phase(wave, position, time)
    energy_density = plane_wave_energy_density(wave, position, time)
    momentum_density = plane_wave_momentum_density(wave, position, time)
    
    # 验证性质
    verification_result = plane_wave_verification(wave, position, time)
    
    print(f"波数: {wave.wavenumber:.2e} m⁻¹")
    print(f"角频率: {wave.angular_frequency:.2e} rad/s")
    print(f"传播方向: {wave.direction}")
    print(f"电场: {electric_field}")
    print(f"磁场: {magnetic_field}")
    print(f"相位: {phase:.6f} rad")
    print(f"能量密度: {energy_density:.2e} J/m³")
    print(f"动量密度: {momentum_density}")
    print(f"平面波性质验证: {'通过' if verification_result else '失败'}")
    
    return wave, electric_field, magnetic_field, energy_density, verification_result
```

### 偏振 / Polarization

**形式化定义**: 偏振描述了电磁波电场矢量在垂直于传播方向的平面内的振动方式。

**公理化定义**:
设 $\mathcal{PL} = \langle \mathcal{E}_x, \mathcal{E}_y, \mathcal{\Delta}, \mathcal{\Theta}, \mathcal{E} \rangle$ 为偏振系统，其中：

1. **x分量**: $\mathcal{E}_x: \mathbb{R} \rightarrow \mathbb{R}$ 为电场x分量
2. **y分量**: $\mathcal{E}_y: \mathbb{R} \rightarrow \mathbb{R}$ 为电场y分量
3. **相位差**: $\mathcal{\Delta} \in [0, 2\pi)$ 为x、y分量间的相位差
4. **偏振角**: $\mathcal{\Theta} \in [0, \pi)$ 为偏振方向角
5. **椭圆参数**: $\mathcal{E} = \langle a, b, \psi \rangle$ 为椭圆偏振参数

**等价定义**:

1. **线偏振**: $\vec{E}(t) = E_0 \cos(\omega t) \hat{n}$
2. **圆偏振**: $\vec{E}(t) = E_0 [\cos(\omega t) \hat{x} \pm \sin(\omega t) \hat{y}]$
3. **椭圆偏振**: $\vec{E}(t) = E_x \cos(\omega t) \hat{x} + E_y \cos(\omega t + \delta) \hat{y}$
4. **琼斯矢量**: $\vec{J} = \begin{pmatrix} E_x \\ E_y e^{i\delta} \end{pmatrix}$

**形式化定理**:

**定理2.5.5.7 (偏振椭圆方程)**: 椭圆偏振满足椭圆方程
$$\frac{E_x^2}{a^2} + \frac{E_y^2}{b^2} = 1$$

**定理2.5.5.8 (偏振度)**: 偏振度定义为
$$P = \frac{I_{max} - I_{min}}{I_{max} + I_{min}}$$

**定理2.5.5.9 (斯托克斯参数)**: 任意偏振态可用斯托克斯参数表示
$$S_0 = I, \quad S_1 = I_p - I_m, \quad S_2 = I_{+45°} - I_{-45°}, \quad S_3 = I_R - I_L$$

**Python算法实现**:

```python
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class Polarization:
    """偏振类"""
    def __init__(self, polarization_type: str, amplitude: float = 1.0, 
                 phase_difference: float = 0.0, angle: float = 0.0):
        self.polarization_type = polarization_type
        self.amplitude = amplitude
        self.phase_difference = phase_difference
        self.angle = angle
    
    def get_type(self) -> str:
        """获取偏振类型"""
        return self.polarization_type
    
    def get_amplitude(self) -> float:
        """获取振幅"""
        return self.amplitude
    
    def get_phase_difference(self) -> float:
        """获取相位差"""
        return self.phase_difference

def linear_polarization(amplitude: float, angle: float, time: float, 
                       frequency: float) -> np.ndarray:
    """
    计算线偏振电场
    
    参数:
        amplitude: 振幅
        angle: 偏振角
        time: 时间
        frequency: 频率
    
    返回:
        电场向量
    """
    omega = 2 * np.pi * frequency
    field_magnitude = amplitude * np.cos(omega * time)
    
    # 计算偏振方向
    direction = np.array([np.cos(angle), np.sin(angle), 0])
    
    return field_magnitude * direction

def circular_polarization(amplitude: float, handedness: str, time: float, 
                         frequency: float) -> np.ndarray:
    """
    计算圆偏振电场
    
    参数:
        amplitude: 振幅
        handedness: 手性 ('right' 或 'left')
        time: 时间
        frequency: 频率
    
    返回:
        电场向量
    """
    omega = 2 * np.pi * frequency
    
    if handedness.lower() == 'right':
        # 右旋圆偏振
        e_x = amplitude * np.cos(omega * time)
        e_y = amplitude * np.sin(omega * time)
    else:
        # 左旋圆偏振
        e_x = amplitude * np.cos(omega * time)
        e_y = -amplitude * np.sin(omega * time)
    
    return np.array([e_x, e_y, 0])

def elliptical_polarization(amplitude_x: float, amplitude_y: float, 
                          phase_difference: float, time: float, 
                          frequency: float) -> np.ndarray:
    """
    计算椭圆偏振电场
    
    参数:
        amplitude_x, amplitude_y: x、y方向振幅
        phase_difference: 相位差
        time: 时间
        frequency: 频率
    
    返回:
        电场向量
    """
    omega = 2 * np.pi * frequency
    
    e_x = amplitude_x * np.cos(omega * time)
    e_y = amplitude_y * np.cos(omega * time + phase_difference)
    
    return np.array([e_x, e_y, 0])

def jones_vector(amplitude_x: float, amplitude_y: float, 
                phase_difference: float) -> np.ndarray:
    """
    计算琼斯矢量
    
    参数:
        amplitude_x, amplitude_y: x、y方向振幅
        phase_difference: 相位差
    
    返回:
        琼斯矢量
    """
    return np.array([amplitude_x, amplitude_y * np.exp(1j * phase_difference)])

def stokes_parameters(electric_field_x: np.ndarray, electric_field_y: np.ndarray) -> np.ndarray:
    """
    计算斯托克斯参数
    
    参数:
        electric_field_x, electric_field_y: x、y方向电场时间序列
    
    返回:
        斯托克斯参数 [S0, S1, S2, S3]
    """
    # 计算强度
    i_x = np.mean(np.abs(electric_field_x)**2)
    i_y = np.mean(np.abs(electric_field_y)**2)
    
    # 计算交叉项
    cross_term = np.mean(electric_field_x * np.conj(electric_field_y))
    
    # 斯托克斯参数
    s0 = i_x + i_y
    s1 = i_x - i_y
    s2 = 2 * np.real(cross_term)
    s3 = 2 * np.imag(cross_term)
    
    return np.array([s0, s1, s2, s3])

def polarization_degree(stokes_params: np.ndarray) -> float:
    """
    计算偏振度
    
    参数:
        stokes_params: 斯托克斯参数
    
    返回:
        偏振度
    """
    s0, s1, s2, s3 = stokes_params
    
    if s0 == 0:
        return 0.0
    
    return np.sqrt(s1**2 + s2**2 + s3**2) / s0

def polarization_ellipse_parameters(amplitude_x: float, amplitude_y: float, 
                                  phase_difference: float) -> Tuple[float, float, float]:
    """
    计算偏振椭圆参数
    
    参数:
        amplitude_x, amplitude_y: x、y方向振幅
        phase_difference: 相位差
    
    返回:
        椭圆参数 (长轴, 短轴, 倾角)
    """
    # 计算椭圆参数
    a_squared = amplitude_x**2 + amplitude_y**2
    b_squared = 2 * amplitude_x * amplitude_y * np.cos(phase_difference)
    c_squared = amplitude_x**2 - amplitude_y**2
    
    # 长轴和短轴
    major_axis = np.sqrt(0.5 * (a_squared + np.sqrt(b_squared**2 + c_squared**2)))
    minor_axis = np.sqrt(0.5 * (a_squared - np.sqrt(b_squared**2 + c_squared**2)))
    
    # 倾角
    if abs(amplitude_x) > 1e-10:
        tilt_angle = 0.5 * np.arctan2(b_squared, c_squared)
    else:
        tilt_angle = 0.0
    
    return major_axis, minor_axis, tilt_angle

def polarization_verification(polarization_type: str, amplitude_x: float, 
                            amplitude_y: float, phase_difference: float, 
                            tolerance: float = 1e-6) -> bool:
    """
    验证偏振性质
    
    参数:
        polarization_type: 偏振类型
        amplitude_x, amplitude_y: x、y方向振幅
        phase_difference: 相位差
        tolerance: 容差
    
    返回:
        是否满足偏振性质
    """
    if polarization_type.lower() == 'linear':
        # 线偏振：相位差为0或π
        return abs(np.sin(phase_difference)) < tolerance
    
    elif polarization_type.lower() == 'circular':
        # 圆偏振：振幅相等，相位差为±π/2
        amplitude_equal = abs(amplitude_x - amplitude_y) < tolerance
        phase_correct = abs(abs(phase_difference) - np.pi/2) < tolerance
        return amplitude_equal and phase_correct
    
    elif polarization_type.lower() == 'elliptical':
        # 椭圆偏振：其他情况
        return True
    
    return False

def polarization_example():
    """偏振示例"""
    frequency = 1e9
    time_points = np.linspace(0, 2*np.pi/(2*np.pi*frequency), 100)
    
    # 线偏振
    linear_angle = np.pi/4  # 45度
    linear_field = np.array([linear_polarization(1.0, linear_angle, t, frequency) 
                            for t in time_points])
    
    # 圆偏振
    circular_field = np.array([circular_polarization(1.0, 'right', t, frequency) 
                              for t in time_points])
    
    # 椭圆偏振
    elliptical_field = np.array([elliptical_polarization(1.0, 0.5, np.pi/4, t, frequency) 
                                for t in time_points])
    
    # 计算斯托克斯参数
    stokes_linear = stokes_parameters(linear_field[:, 0], linear_field[:, 1])
    stokes_circular = stokes_parameters(circular_field[:, 0], circular_field[:, 1])
    stokes_elliptical = stokes_parameters(elliptical_field[:, 0], elliptical_field[:, 1])
    
    # 计算偏振度
    p_linear = polarization_degree(stokes_linear)
    p_circular = polarization_degree(stokes_circular)
    p_elliptical = polarization_degree(stokes_elliptical)
    
    # 计算椭圆参数
    major, minor, tilt = polarization_ellipse_parameters(1.0, 0.5, np.pi/4)
    
    # 验证性质
    linear_verified = polarization_verification('linear', 1.0, 1.0, 0.0)
    circular_verified = polarization_verification('circular', 1.0, 1.0, np.pi/2)
    elliptical_verified = polarization_verification('elliptical', 1.0, 0.5, np.pi/4)
    
    print(f"线偏振度: {p_linear:.6f}")
    print(f"圆偏振度: {p_circular:.6f}")
    print(f"椭圆偏振度: {p_elliptical:.6f}")
    print(f"椭圆长轴: {major:.6f}")
    print(f"椭圆短轴: {minor:.6f}")
    print(f"椭圆倾角: {np.degrees(tilt):.2f}°")
    print(f"线偏振验证: {'通过' if linear_verified else '失败'}")
    print(f"圆偏振验证: {'通过' if circular_verified else '失败'}")
    print(f"椭圆偏振验证: {'通过' if elliptical_verified else '失败'}")
    
    return (linear_field, circular_field, elliptical_field, 
            stokes_linear, stokes_circular, stokes_elliptical,
            p_linear, p_circular, p_elliptical)
```

### 多普勒效应 / Doppler Effect

**形式化定义**: 多普勒效应描述了电磁波在相对运动观察者和源之间传播时频率的变化现象。

**公理化定义**:
设 $\mathcal{DE} = \langle \mathcal{F}_s, \mathcal{F}_o, \mathcal{V}, \mathcal{C}, \mathcal{\Theta} \rangle$ 为多普勒效应系统，其中：

1. **源频率**: $\mathcal{F}_s \in \mathbb{R}^+$ 为电磁波源频率
2. **观察频率**: $\mathcal{F}_o \in \mathbb{R}^+$ 为观察者接收频率
3. **相对速度**: $\mathcal{V} \in \mathbb{R}$ 为相对运动速度
4. **光速**: $\mathcal{C} = c$ 为真空光速
5. **运动角度**: $\mathcal{\Theta} \in [0, \pi]$ 为运动方向与传播方向夹角

**等价定义**:

1. **经典多普勒公式**: $f_o = f_s \frac{c \pm v}{c \mp v}$
2. **相对论多普勒公式**: $f_o = f_s \sqrt{\frac{1 + \beta}{1 - \beta}}$
3. **角度依赖公式**: $f_o = f_s \frac{1 - \beta \cos\theta}{\sqrt{1 - \beta^2}}$
4. **红移公式**: $z = \frac{\lambda_o - \lambda_s}{\lambda_s} = \frac{f_s - f_o}{f_o}$

**形式化定理**:

**定理2.5.5.10 (多普勒频移)**: 相对论多普勒频移为
$$\frac{f_o}{f_s} = \sqrt{\frac{1 + \beta \cos\theta}{1 - \beta \cos\theta}}$$

**定理2.5.5.11 (横向多普勒效应)**: 横向多普勒效应为
$$\frac{f_o}{f_s} = \frac{1}{\sqrt{1 - \beta^2}}$$

**定理2.5.5.12 (多普勒展宽)**: 多普勒展宽为
$$\Delta f = f_0 \sqrt{\frac{2kT}{mc^2}}$$

**Python算法实现**:

```python
import numpy as np
from typing import Tuple, List

class DopplerEffect:
    """多普勒效应类"""
    def __init__(self, source_frequency: float, relative_velocity: float = 0.0, 
                 angle: float = 0.0, relativistic: bool = True):
        self.source_frequency = source_frequency
        self.relative_velocity = relative_velocity
        self.angle = angle
        self.relativistic = relativistic
        self.speed_of_light = 3e8
        self.beta = relative_velocity / self.speed_of_light
    
    def get_source_frequency(self) -> float:
        """获取源频率"""
        return self.source_frequency
    
    def get_relative_velocity(self) -> float:
        """获取相对速度"""
        return self.relative_velocity
    
    def get_beta(self) -> float:
        """获取相对论参数β"""
        return self.beta

def classical_doppler_shift(source_frequency: float, relative_velocity: float, 
                          approach: bool = True) -> float:
    """
    计算经典多普勒频移
    
    参数:
        source_frequency: 源频率
        relative_velocity: 相对速度
        approach: 是否接近 (True为接近，False为远离)
    
    返回:
        观察频率
    """
    speed_of_light = 3e8
    
    if approach:
        # 接近时频率增加
        return source_frequency * (speed_of_light + relative_velocity) / (speed_of_light - relative_velocity)
    else:
        # 远离时频率减少
        return source_frequency * (speed_of_light - relative_velocity) / (speed_of_light + relative_velocity)

def relativistic_doppler_shift(source_frequency: float, relative_velocity: float, 
                             angle: float = 0.0) -> float:
    """
    计算相对论多普勒频移
    
    参数:
        source_frequency: 源频率
        relative_velocity: 相对速度
        angle: 运动方向与传播方向夹角
    
    返回:
        观察频率
    """
    speed_of_light = 3e8
    beta = relative_velocity / speed_of_light
    
    if abs(beta) >= 1:
        raise ValueError("相对速度不能超过光速")
    
    # 相对论多普勒公式
    frequency_ratio = np.sqrt((1 + beta * np.cos(angle)) / (1 - beta * np.cos(angle)))
    
    return source_frequency * frequency_ratio

def transverse_doppler_effect(source_frequency: float, relative_velocity: float) -> float:
    """
    计算横向多普勒效应
    
    参数:
        source_frequency: 源频率
        relative_velocity: 相对速度
    
    返回:
        观察频率
    """
    speed_of_light = 3e8
    beta = relative_velocity / speed_of_light
    
    if abs(beta) >= 1:
        raise ValueError("相对速度不能超过光速")
    
    # 横向多普勒效应 (θ = π/2)
    frequency_ratio = 1.0 / np.sqrt(1 - beta**2)
    
    return source_frequency * frequency_ratio

def doppler_broadening(temperature: float, molecular_mass: float, 
                      center_frequency: float) -> float:
    """
    计算多普勒展宽
    
    参数:
        temperature: 温度 (K)
        molecular_mass: 分子质量 (kg)
        center_frequency: 中心频率 (Hz)
    
    返回:
        多普勒展宽 (Hz)
    """
    boltzmann_constant = 1.381e-23
    speed_of_light = 3e8
    
    # 多普勒展宽公式
    doppler_width = center_frequency * np.sqrt(2 * boltzmann_constant * temperature / 
                                              (molecular_mass * speed_of_light**2))
    
    return doppler_width

def redshift(source_frequency: float, observed_frequency: float) -> float:
    """
    计算红移
    
    参数:
        source_frequency: 源频率
        observed_frequency: 观察频率
    
    返回:
        红移参数
    """
    return (source_frequency - observed_frequency) / observed_frequency

def blueshift(source_frequency: float, observed_frequency: float) -> float:
    """
    计算蓝移
    
    参数:
        source_frequency: 源频率
        observed_frequency: 观察频率
    
    返回:
        蓝移参数
    """
    return (observed_frequency - source_frequency) / source_frequency

def doppler_velocity_from_redshift(redshift: float) -> float:
    """
    从红移计算相对速度
    
    参数:
        redshift: 红移参数
    
    返回:
        相对速度 (m/s)
    """
    speed_of_light = 3e8
    
    # 相对论红移公式
    if redshift >= 0:
        # 红移 (远离)
        velocity = speed_of_light * ((1 + redshift)**2 - 1) / ((1 + redshift)**2 + 1)
    else:
        # 蓝移 (接近)
        velocity = -speed_of_light * ((1 - redshift)**2 - 1) / ((1 - redshift)**2 + 1)
    
    return velocity

def doppler_effect_verification(source_frequency: float, relative_velocity: float, 
                              angle: float, tolerance: float = 1e-6) -> bool:
    """
    验证多普勒效应
    
    参数:
        source_frequency: 源频率
        relative_velocity: 相对速度
        angle: 运动角度
        tolerance: 容差
    
    返回:
        是否满足多普勒效应
    """
    speed_of_light = 3e8
    beta = relative_velocity / speed_of_light
    
    # 检查相对论限制
    if abs(beta) >= 1:
        return False
    
    # 计算相对论频移
    relativistic_frequency = relativistic_doppler_shift(source_frequency, relative_velocity, angle)
    
    # 验证频率比
    frequency_ratio = relativistic_frequency / source_frequency
    expected_ratio = np.sqrt((1 + beta * np.cos(angle)) / (1 - beta * np.cos(angle)))
    
    return abs(frequency_ratio - expected_ratio) < tolerance

def doppler_effect_example():
    """多普勒效应示例"""
    # 创建多普勒效应对象
    source_frequency = 1e9  # 1 GHz
    relative_velocity = 0.1 * 3e8  # 0.1c
    angle = np.pi/4  # 45度
    
    doppler = DopplerEffect(source_frequency, relative_velocity, angle)
    
    # 计算各种多普勒效应
    classical_approach = classical_doppler_shift(source_frequency, relative_velocity, True)
    classical_recede = classical_doppler_shift(source_frequency, relative_velocity, False)
    relativistic_frequency = relativistic_doppler_shift(source_frequency, relative_velocity, angle)
    transverse_frequency = transverse_doppler_effect(source_frequency, relative_velocity)
    
    # 计算多普勒展宽
    temperature = 300  # 室温
    molecular_mass = 1.67e-27  # 氢原子质量
    doppler_width = doppler_broadening(temperature, molecular_mass, source_frequency)
    
    # 计算红移
    z = redshift(source_frequency, relativistic_frequency)
    
    # 从红移计算速度
    calculated_velocity = doppler_velocity_from_redshift(z)
    
    # 验证多普勒效应
    verification_result = doppler_effect_verification(source_frequency, relative_velocity, angle)
    
    print(f"源频率: {source_frequency:.2e} Hz")
    print(f"相对速度: {relative_velocity:.2e} m/s (β = {doppler.beta:.3f})")
    print(f"经典多普勒 (接近): {classical_approach:.2e} Hz")
    print(f"经典多普勒 (远离): {classical_recede:.2e} Hz")
    print(f"相对论多普勒: {relativistic_frequency:.2e} Hz")
    print(f"横向多普勒: {transverse_frequency:.2e} Hz")
    print(f"多普勒展宽: {doppler_width:.2e} Hz")
    print(f"红移参数: {z:.6f}")
    print(f"计算速度: {calculated_velocity:.2e} m/s")
    print(f"多普勒效应验证: {'通过' if verification_result else '失败'}")
    
    return (doppler, classical_approach, classical_recede, relativistic_frequency,
            transverse_frequency, doppler_width, z, calculated_velocity, verification_result)
```

---

## 2.5.6 电磁辐射 / Electromagnetic Radiation

### 偶极辐射 / Dipole Radiation

**形式化定义**: 偶极辐射描述了振荡电偶极子产生的电磁辐射场，是电磁辐射理论的基础模型。

**公理化定义**:
设 $\mathcal{DR} = \langle \mathcal{P}, \mathcal{R}, \mathcal{E}, \mathcal{B}, \mathcal{S} \rangle$ 为偶极辐射系统，其中：

1. **偶极矩集合**: $\mathcal{P}$ 为电偶极矩集合
2. **辐射场集合**: $\mathcal{R}$ 为辐射电磁场集合
3. **电场函数**: $\mathcal{E}: \mathcal{P} \times \mathbb{R}^3 \times \mathbb{R} \rightarrow \mathbb{R}^3$ 为辐射电场函数
4. **磁场函数**: $\mathcal{B}: \mathcal{P} \times \mathbb{R}^3 \times \mathbb{R} \rightarrow \mathbb{R}^3$ 为辐射磁场函数
5. **坡印廷矢量**: $\mathcal{S}: \mathcal{E} \times \mathcal{B} \rightarrow \mathbb{R}^3$ 为能流密度函数

**等价定义**:

1. **远场近似**: $\vec{E} = \frac{\mu_0 \ddot{\vec{p}}}{4\pi r} \sin\theta \hat{\theta}$
2. **磁场关系**: $\vec{B} = \frac{1}{c} \hat{r} \times \vec{E}$
3. **功率密度**: $S = \frac{\mu_0 \ddot{p}^2}{16\pi^2 c r^2} \sin^2\theta$

**形式化定理**:

**定理2.5.6.1 (偶极辐射远场性)**: 偶极辐射在远场区域为横波
$$\vec{E} \cdot \hat{r} = 0, \quad \vec{B} \cdot \hat{r} = 0$$

**定理2.5.6.2 (偶极辐射方向性)**: 偶极辐射具有方向性，最大辐射方向垂直于偶极矩
$$S_{max} = \frac{\mu_0 \ddot{p}^2}{16\pi^2 c r^2}$$

**定理2.5.6.3 (偶极辐射总功率)**: 偶极辐射的总功率为
$$P = \frac{\mu_0 \ddot{p}^2}{6\pi c}$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple
from scipy.constants import mu_0, c

class DipoleRadiation:
    """偶极辐射类"""
    def __init__(self, dipole_moment: np.ndarray, frequency: float, 
                 amplitude: float = 1.0):
        self.dipole_moment = np.array(dipole_moment)
        self.frequency = frequency
        self.amplitude = amplitude
        self.angular_frequency = 2 * np.pi * frequency
        self.wavenumber = self.angular_frequency / c
    
    def get_dipole_moment(self) -> np.ndarray:
        """获取偶极矩"""
        return self.dipole_moment.copy()
    
    def get_frequency(self) -> float:
        """获取频率"""
        return self.frequency
    
    def get_angular_frequency(self) -> float:
        """获取角频率"""
        return self.angular_frequency
    
    def get_wavenumber(self) -> float:
        """获取波数"""
        return self.wavenumber

def dipole_electric_field(dipole_moment: np.ndarray, position: np.ndarray, 
                         time: float, frequency: float) -> np.ndarray:
    """
    计算偶极辐射电场
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        position: 观察位置 (m)
        time: 时间 (s)
        frequency: 频率 (Hz)
    
    返回:
        电场向量 (V/m)
    """
    r = np.linalg.norm(position)
    if r == 0:
        return np.zeros(3)
    
    r_hat = position / r
    k = 2 * np.pi * frequency / c
    phase = k * r - 2 * np.pi * frequency * time
    
    # 远场近似
    if k * r >> 1:
        # 横向电场
        p_parallel = np.dot(dipole_moment, r_hat) * r_hat
        p_perp = dipole_moment - p_parallel
        
        E_magnitude = mu_0 * (2 * np.pi * frequency)**2 * np.linalg.norm(p_perp) / (4 * np.pi * r)
        E_direction = p_perp / np.linalg.norm(p_perp) if np.linalg.norm(p_perp) > 0 else np.array([0, 1, 0])
        
        return E_magnitude * np.cos(phase) * E_direction
    else:
        # 近场计算（简化）
        return np.zeros(3)

def dipole_magnetic_field(dipole_moment: np.ndarray, position: np.ndarray, 
                         time: float, frequency: float) -> np.ndarray:
    """
    计算偶极辐射磁场
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        position: 观察位置 (m)
        time: 时间 (s)
        frequency: 频率 (Hz)
    
    返回:
        磁场向量 (T)
    """
    E_field = dipole_electric_field(dipole_moment, position, time, frequency)
    r = np.linalg.norm(position)
    
    if r == 0:
        return np.zeros(3)
    
    r_hat = position / r
    
    # 磁场与电场的关系
    return np.cross(r_hat, E_field) / c

def dipole_power_density(dipole_moment: np.ndarray, position: np.ndarray, 
                        frequency: float) -> float:
    """
    计算偶极辐射功率密度
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        position: 观察位置 (m)
        frequency: 频率 (Hz)
    
    返回:
        功率密度 (W/m²)
    """
    r = np.linalg.norm(position)
    if r == 0:
        return 0.0
    
    r_hat = position / r
    p_parallel = np.dot(dipole_moment, r_hat) * r_hat
    p_perp = dipole_moment - p_parallel
    
    # 功率密度公式
    S = mu_0 * (2 * np.pi * frequency)**2 * np.linalg.norm(p_perp)**2 / (16 * np.pi**2 * c * r**2)
    
    return S

def dipole_total_power(dipole_moment: np.ndarray, frequency: float) -> float:
    """
    计算偶极辐射总功率
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        frequency: 频率 (Hz)
    
    返回:
        总功率 (W)
    """
    # 总功率公式
    P = mu_0 * (2 * np.pi * frequency)**2 * np.linalg.norm(dipole_moment)**2 / (6 * np.pi * c)
    
    return P

def dipole_radiation_pattern(dipole_moment: np.ndarray, theta_values: np.ndarray, 
                           phi_values: np.ndarray, frequency: float) -> np.ndarray:
    """
    计算偶极辐射方向图
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        theta_values: 极角数组 (rad)
        phi_values: 方位角数组 (rad)
        frequency: 频率 (Hz)
    
    返回:
        辐射强度数组 (W/sr)
    """
    pattern = np.zeros((len(theta_values), len(phi_values)))
    
    for i, theta in enumerate(theta_values):
        for j, phi in enumerate(phi_values):
            # 球坐标到直角坐标
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            position = np.array([x, y, z])
            
            # 计算功率密度
            S = dipole_power_density(dipole_moment, position, frequency)
            pattern[i, j] = S * 1.0  # 假设单位距离
    
    return pattern

def dipole_radiation_verification(dipole_moment: np.ndarray, position: np.ndarray, 
                                time: float, frequency: float, tolerance: float = 1e-6) -> bool:
    """
    验证偶极辐射
    
    参数:
        dipole_moment: 偶极矩向量 (C·m)
        position: 观察位置 (m)
        time: 时间 (s)
        frequency: 频率 (Hz)
        tolerance: 容差
    
    返回:
        是否满足偶极辐射条件
    """
    E_field = dipole_electric_field(dipole_moment, position, time, frequency)
    B_field = dipole_magnetic_field(dipole_moment, position, time, frequency)
    
    r = np.linalg.norm(position)
    if r == 0:
        return True
    
    r_hat = position / r
    
    # 验证横波性
    E_transverse = abs(np.dot(E_field, r_hat))
    B_transverse = abs(np.dot(B_field, r_hat))
    
    # 验证电场和磁场正交
    E_B_orthogonal = abs(np.dot(E_field, B_field))
    
    return (E_transverse < tolerance and B_transverse < tolerance and 
            E_B_orthogonal < tolerance)

def dipole_radiation_example():
    """偶极辐射示例"""
    # 创建偶极辐射对象
    dipole_moment = np.array([0, 0, 1e-12])  # 1 pC·m
    frequency = 1e9  # 1 GHz
    radiation = DipoleRadiation(dipole_moment, frequency)
    
    # 计算远场电场和磁场
    position = np.array([1, 0, 0])  # 1米距离
    time = 0.0
    E_field = dipole_electric_field(dipole_moment, position, time, frequency)
    B_field = dipole_magnetic_field(dipole_moment, position, time, frequency)
    
    # 计算功率密度和总功率
    power_density = dipole_power_density(dipole_moment, position, frequency)
    total_power = dipole_total_power(dipole_moment, frequency)
    
    # 计算辐射方向图
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    pattern = dipole_radiation_pattern(dipole_moment, theta, phi, frequency)
    
    # 验证偶极辐射
    verification = dipole_radiation_verification(dipole_moment, position, time, frequency)
    
    print(f"偶极矩: {dipole_moment} C·m")
    print(f"频率: {frequency:.2e} Hz")
    print(f"电场: {E_field} V/m")
    print(f"磁场: {B_field} T")
    print(f"功率密度: {power_density:.2e} W/m²")
    print(f"总功率: {total_power:.2e} W")
    print(f"偶极辐射验证: {'通过' if verification else '失败'}")
    
    return (radiation, E_field, B_field, power_density, total_power, pattern, verification)
```

### 天线理论 / Antenna Theory

**形式化定义**: 天线理论描述了电磁辐射器的辐射特性，包括方向性、增益、阻抗等参数。

**公理化定义**:
设 $\mathcal{AT} = \langle \mathcal{A}, \mathcal{D}, \mathcal{G}, \mathcal{Z}, \mathcal{E} \rangle$ 为天线理论系统，其中：

1. **天线集合**: $\mathcal{A}$ 为天线结构集合
2. **方向性函数**: $\mathcal{D}: \mathcal{A} \times \mathbb{R}^3 \rightarrow \mathbb{R}$ 为方向性函数
3. **增益函数**: $\mathcal{G}: \mathcal{A} \times \mathbb{R}^3 \rightarrow \mathbb{R}$ 为增益函数
4. **阻抗函数**: $\mathcal{Z}: \mathcal{A} \times \mathbb{R} \rightarrow \mathbb{C}$ 为阻抗函数
5. **效率函数**: $\mathcal{E}: \mathcal{A} \rightarrow [0,1]$ 为辐射效率函数

**等价定义**:

1. **方向性**: $D(\theta,\phi) = \frac{4\pi U(\theta,\phi)}{P_{rad}}$
2. **增益**: $G(\theta,\phi) = \eta D(\theta,\phi)$
3. **阻抗**: $Z = R + jX = \frac{V}{I}$

**形式化定理**:

**定理2.5.6.4 (天线方向性积分)**: 天线方向性满足积分约束
$$\int_0^{2\pi} \int_0^{\pi} D(\theta,\phi) \sin\theta d\theta d\phi = 4\pi$$

**定理2.5.6.5 (天线增益上限)**: 天线增益受物理尺寸限制
$$G_{max} \leq \frac{4\pi A_{eff}}{\lambda^2}$$

**定理2.5.6.6 (天线阻抗匹配)**: 最大功率传输条件为阻抗共轭匹配
$$Z_L = Z_a^*$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple, Complex
from scipy.constants import mu_0, c

class Antenna:
    """天线类"""
    def __init__(self, length: float, frequency: float, 
                 efficiency: float = 1.0, impedance: complex = 50+0j):
        self.length = length
        self.frequency = frequency
        self.efficiency = efficiency
        self.impedance = impedance
        self.wavelength = c / frequency
        self.wavenumber = 2 * np.pi / self.wavelength
    
    def get_length(self) -> float:
        """获取天线长度"""
        return self.length
    
    def get_frequency(self) -> float:
        """获取工作频率"""
        return self.frequency
    
    def get_wavelength(self) -> float:
        """获取波长"""
        return self.wavelength
    
    def get_impedance(self) -> complex:
        """获取阻抗"""
        return self.impedance

def dipole_directivity(theta: float, phi: float) -> float:
    """
    计算偶极天线方向性
    
    参数:
        theta: 极角 (rad)
        phi: 方位角 (rad)
    
    返回:
        方向性
    """
    # 偶极天线方向性函数
    D = 1.5 * np.sin(theta)**2
    return D

def dipole_gain(theta: float, phi: float, efficiency: float = 1.0) -> float:
    """
    计算偶极天线增益
    
    参数:
        theta: 极角 (rad)
        phi: 方位角 (rad)
        efficiency: 效率
    
    返回:
        增益
    """
    directivity = dipole_directivity(theta, phi)
    gain = efficiency * directivity
    return gain

def antenna_impedance(length: float, frequency: float, 
                     wire_radius: float = 1e-3) -> complex:
    """
    计算天线阻抗
    
    参数:
        length: 天线长度 (m)
        frequency: 频率 (Hz)
        wire_radius: 导线半径 (m)
    
    返回:
        阻抗 (Ω)
    """
    wavelength = c / frequency
    k = 2 * np.pi / wavelength
    
    # 半波偶极子阻抗近似
    if abs(length - wavelength/2) < wavelength/20:
        # 半波偶极子
        R_rad = 73.0  # 辐射电阻
        X_react = 42.5  # 电抗
    else:
        # 一般情况
        R_rad = 20 * (np.pi * length / wavelength)**2
        X_react = 120 * (np.log(length / (2 * wire_radius)) - 1) * np.tan(np.pi * length / wavelength)
    
    return complex(R_rad, X_react)

def antenna_efficiency(radiation_resistance: float, loss_resistance: float) -> float:
    """
    计算天线效率
    
    参数:
        radiation_resistance: 辐射电阻 (Ω)
        loss_resistance: 损耗电阻 (Ω)
    
    返回:
        效率
    """
    total_resistance = radiation_resistance + loss_resistance
    efficiency = radiation_resistance / total_resistance
    return efficiency

def antenna_beamwidth(directivity_pattern: np.ndarray, 
                     theta_values: np.ndarray, phi_values: np.ndarray) -> Tuple[float, float]:
    """
    计算天线波束宽度
    
    参数:
        directivity_pattern: 方向性图
        theta_values: 极角数组 (rad)
        phi_values: 方位角数组 (rad)
    
    返回:
        (E面波束宽度, H面波束宽度) (度)
    """
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(directivity_pattern), directivity_pattern.shape)
    max_theta = theta_values[max_idx[0]]
    max_phi = phi_values[max_idx[1]]
    
    # 计算半功率点
    max_value = directivity_pattern[max_idx]
    half_power = max_value / 2
    
    # E面波束宽度 (theta方向)
    e_plane_width = 0
    for i, theta in enumerate(theta_values):
        if directivity_pattern[i, max_idx[1]] >= half_power:
            e_plane_width = max(e_plane_width, abs(theta - max_theta))
    
    # H面波束宽度 (phi方向)
    h_plane_width = 0
    for j, phi in enumerate(phi_values):
        if directivity_pattern[max_idx[0], j] >= half_power:
            h_plane_width = max(h_plane_width, abs(phi - max_phi))
    
    return (np.degrees(2 * e_plane_width), np.degrees(2 * h_plane_width))

def antenna_array_factor(positions: np.ndarray, phases: np.ndarray, 
                        theta: float, phi: float, frequency: float) -> complex:
    """
    计算天线阵列因子
    
    参数:
        positions: 天线位置数组 (m)
        phases: 相位数组 (rad)
        theta: 极角 (rad)
        phi: 方位角 (rad)
        frequency: 频率 (Hz)
    
    返回:
        阵列因子
    """
    k = 2 * np.pi * frequency / c
    
    # 观察方向单位向量
    r_hat = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
    
    array_factor = 0j
    for i, pos in enumerate(positions):
        # 相位差
        phase_diff = k * np.dot(pos, r_hat) + phases[i]
        array_factor += np.exp(1j * phase_diff)
    
    return array_factor

def antenna_theory_verification(antenna: Antenna, theta: float, phi: float, 
                              tolerance: float = 1e-6) -> bool:
    """
    验证天线理论
    
    参数:
        antenna: 天线对象
        theta: 极角 (rad)
        phi: 方位角 (rad)
        tolerance: 容差
    
    返回:
        是否满足天线理论
    """
    # 计算方向性和增益
    directivity = dipole_directivity(theta, phi)
    gain = dipole_gain(theta, phi, antenna.efficiency)
    
    # 验证增益与方向性关系
    gain_directivity_ratio = gain / directivity
    efficiency_check = abs(gain_directivity_ratio - antenna.efficiency) < tolerance
    
    # 验证方向性积分
    theta_values = np.linspace(0, np.pi, 100)
    phi_values = np.linspace(0, 2*np.pi, 100)
    integral = 0
    
    for t in theta_values:
        for p in phi_values:
            D = dipole_directivity(t, p)
            integral += D * np.sin(t) * (np.pi/100) * (2*np.pi/100)
    
    integral_check = abs(integral - 4*np.pi) < tolerance
    
    return efficiency_check and integral_check

def antenna_theory_example():
    """天线理论示例"""
    # 创建天线对象
    length = 0.15  # 半波长天线
    frequency = 1e9  # 1 GHz
    antenna = Antenna(length, frequency)
    
    # 计算阻抗
    impedance = antenna_impedance(length, frequency)
    
    # 计算方向性和增益
    theta = np.pi/2  # 水平方向
    phi = 0
    directivity = dipole_directivity(theta, phi)
    gain = dipole_gain(theta, phi, antenna.efficiency)
    
    # 计算效率
    radiation_resistance = impedance.real
    loss_resistance = 1.0  # 假设损耗电阻
    efficiency = antenna_efficiency(radiation_resistance, loss_resistance)
    
    # 计算波束宽度
    theta_values = np.linspace(0, np.pi, 50)
    phi_values = np.linspace(0, 2*np.pi, 50)
    pattern = np.zeros((len(theta_values), len(phi_values)))
    
    for i, t in enumerate(theta_values):
        for j, p in enumerate(phi_values):
            pattern[i, j] = dipole_directivity(t, p)
    
    beamwidth_e, beamwidth_h = antenna_beamwidth(pattern, theta_values, phi_values)
    
    # 计算阵列因子
    positions = np.array([[0, 0, 0], [0.1, 0, 0]])  # 两个天线
    phases = np.array([0, np.pi/2])  # 90度相位差
    array_factor = antenna_array_factor(positions, phases, theta, phi, frequency)
    
    # 验证天线理论
    verification = antenna_theory_verification(antenna, theta, phi)
    
    print(f"天线长度: {length:.3f} m")
    print(f"工作频率: {frequency:.2e} Hz")
    print(f"阻抗: {impedance:.1f} Ω")
    print(f"方向性: {directivity:.2f}")
    print(f"增益: {gain:.2f}")
    print(f"效率: {efficiency:.3f}")
    print(f"E面波束宽度: {beamwidth_e:.1f}°")
    print(f"H面波束宽度: {beamwidth_h:.1f}°")
    print(f"阵列因子幅度: {abs(array_factor):.2f}")
    print(f"天线理论验证: {'通过' if verification else '失败'}")
    
    return (antenna, impedance, directivity, gain, efficiency, 
            beamwidth_e, beamwidth_h, array_factor, verification)
```

### 辐射功率 / Radiated Power

**形式化定义**: 辐射功率描述了电磁辐射器向空间辐射的总功率，是天线性能的重要指标。

**公理化定义**:
设 $\mathcal{RP} = \langle \mathcal{P}, \mathcal{S}, \mathcal{U}, \mathcal{I} \rangle$ 为辐射功率系统，其中：

1. **功率集合**: $\mathcal{P}$ 为辐射功率集合
2. **功率密度**: $\mathcal{S}: \mathbb{R}^3 \rightarrow \mathbb{R}$ 为功率密度函数
3. **辐射强度**: $\mathcal{U}: \mathbb{S}^2 \rightarrow \mathbb{R}$ 为辐射强度函数
4. **功率积分**: $\mathcal{I}: \mathcal{S} \times \mathbb{S}^2 \rightarrow \mathbb{R}$ 为功率积分函数

**等价定义**:

1. **总功率**: $P = \oint \vec{S} \cdot d\vec{A}$
2. **辐射强度**: $U = r^2 S$
3. **功率密度**: $S = \frac{1}{2} \text{Re}(\vec{E} \times \vec{H}^*)$

**形式化定理**:

**定理2.5.6.7 (功率守恒)**: 辐射功率满足能量守恒定律
$$\frac{dP}{dt} = -\oint \vec{S} \cdot d\vec{A}$$

**定理2.5.6.8 (功率方向性)**: 辐射功率与方向性相关
$$P = \frac{1}{4\pi} \int_0^{2\pi} \int_0^{\pi} D(\theta,\phi) P_0 \sin\theta d\theta d\phi$$

**定理2.5.6.9 (功率增益关系)**: 辐射功率与增益成正比
$$P = \frac{G(\theta,\phi) P_0}{4\pi r^2}$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple
from scipy.constants import mu_0, c
from scipy.integrate import dblquad

class RadiatedPower:
    """辐射功率类"""
    def __init__(self, power_density: Callable, frequency: float):
        self.power_density = power_density
        self.frequency = frequency
        self.wavelength = c / frequency
    
    def get_power_density(self) -> Callable:
        """获取功率密度函数"""
        return self.power_density
    
    def get_frequency(self) -> float:
        """获取频率"""
        return self.frequency
    
    def get_wavelength(self) -> float:
        """获取波长"""
        return self.wavelength

def total_radiated_power(power_density_func: Callable, radius: float = 1.0) -> float:
    """
    计算总辐射功率
    
    参数:
        power_density_func: 功率密度函数
        radius: 积分半径 (m)
    
    返回:
        总功率 (W)
    """
    def integrand(theta, phi):
        # 球坐标到直角坐标
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        position = np.array([x, y, z])
        
        # 功率密度
        S = power_density_func(position)
        
        # 球面面积元
        dA = radius**2 * np.sin(theta)
        
        return S * dA
    
    # 球面积分
    power, error = dblquad(integrand, 0, 2*np.pi, lambda phi: 0, lambda phi: np.pi)
    
    return power

def radiation_intensity(power_density_func: Callable, direction: np.ndarray, 
                       radius: float = 1.0) -> float:
    """
    计算辐射强度
    
    参数:
        power_density_func: 功率密度函数
        direction: 方向向量
        radius: 距离 (m)
    
    返回:
        辐射强度 (W/sr)
    """
    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)
    position = radius * direction
    
    # 功率密度
    S = power_density_func(position)
    
    # 辐射强度
    U = radius**2 * S
    
    return U

def power_density_from_fields(E_field: np.ndarray, H_field: np.ndarray) -> float:
    """
    从电磁场计算功率密度
    
    参数:
        E_field: 电场向量 (V/m)
        H_field: 磁场向量 (A/m)
    
    返回:
        功率密度 (W/m²)
    """
    # 坡印廷矢量
    S_vector = np.cross(E_field, np.conj(H_field))
    S = 0.5 * np.real(S_vector)
    
    return np.linalg.norm(S)

def power_efficiency(input_power: float, radiated_power: float) -> float:
    """
    计算功率效率
    
    参数:
        input_power: 输入功率 (W)
        radiated_power: 辐射功率 (W)
    
    返回:
        效率
    """
    efficiency = radiated_power / input_power
    return max(0.0, min(1.0, efficiency))

def power_density_pattern(power_density_func: Callable, 
                         theta_values: np.ndarray, phi_values: np.ndarray, 
                         radius: float = 1.0) -> np.ndarray:
    """
    计算功率密度方向图
    
    参数:
        power_density_func: 功率密度函数
        theta_values: 极角数组 (rad)
        phi_values: 方位角数组 (rad)
        radius: 距离 (m)
    
    返回:
        功率密度图 (W/m²)
    """
    pattern = np.zeros((len(theta_values), len(phi_values)))
    
    for i, theta in enumerate(theta_values):
        for j, phi in enumerate(phi_values):
            # 球坐标到直角坐标
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            position = np.array([x, y, z])
            
            # 功率密度
            pattern[i, j] = power_density_func(position)
    
    return pattern

def power_beamwidth(power_pattern: np.ndarray, 
                   theta_values: np.ndarray, phi_values: np.ndarray) -> Tuple[float, float]:
    """
    计算功率波束宽度
    
    参数:
        power_pattern: 功率方向图
        theta_values: 极角数组 (rad)
        phi_values: 方位角数组 (rad)
    
    返回:
        (E面波束宽度, H面波束宽度) (度)
    """
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(power_pattern), power_pattern.shape)
    max_theta = theta_values[max_idx[0]]
    max_phi = phi_values[max_idx[1]]
    
    # 计算半功率点
    max_value = power_pattern[max_idx]
    half_power = max_value / 2
    
    # E面波束宽度 (theta方向)
    e_plane_width = 0
    for i, theta in enumerate(theta_values):
        if power_pattern[i, max_idx[1]] >= half_power:
            e_plane_width = max(e_plane_width, abs(theta - max_theta))
    
    # H面波束宽度 (phi方向)
    h_plane_width = 0
    for j, phi in enumerate(phi_values):
        if power_pattern[max_idx[0], j] >= half_power:
            h_plane_width = max(h_plane_width, abs(phi - max_phi))
    
    return (np.degrees(2 * e_plane_width), np.degrees(2 * h_plane_width))

def power_verification(power_density_func: Callable, tolerance: float = 1e-6) -> bool:
    """
    验证辐射功率
    
    参数:
        power_density_func: 功率密度函数
        tolerance: 容差
    
    返回:
        是否满足功率守恒
    """
    # 计算不同半径的总功率
    radius1 = 1.0
    radius2 = 2.0
    
    power1 = total_radiated_power(power_density_func, radius1)
    power2 = total_radiated_power(power_density_func, radius2)
    
    # 功率应该与半径无关（在远场）
    power_ratio = power1 / power2
    expected_ratio = 1.0  # 理想情况下应该相等
    
    return abs(power_ratio - expected_ratio) < tolerance

def radiated_power_example():
    """辐射功率示例"""
    # 定义功率密度函数（偶极子辐射）
    def dipole_power_density(position):
        r = np.linalg.norm(position)
        if r == 0:
            return 0.0
        
        r_hat = position / r
        dipole_moment = np.array([0, 0, 1e-12])  # 1 pC·m
        frequency = 1e9  # 1 GHz
        
        p_parallel = np.dot(dipole_moment, r_hat) * r_hat
        p_perp = dipole_moment - p_parallel
        
        S = mu_0 * (2 * np.pi * frequency)**2 * np.linalg.norm(p_perp)**2 / (16 * np.pi**2 * c * r**2)
        return S
    
    # 创建辐射功率对象
    frequency = 1e9
    power_obj = RadiatedPower(dipole_power_density, frequency)
    
    # 计算总辐射功率
    total_power = total_radiated_power(dipole_power_density)
    
    # 计算辐射强度
    direction = np.array([1, 0, 0])
    intensity = radiation_intensity(dipole_power_density, direction)
    
    # 计算功率效率
    input_power = 1e-6  # 1 μW输入功率
    efficiency = power_efficiency(input_power, total_power)
    
    # 计算功率方向图
    theta_values = np.linspace(0, np.pi, 50)
    phi_values = np.linspace(0, 2*np.pi, 50)
    power_pattern = power_density_pattern(dipole_power_density, theta_values, phi_values)
    
    # 计算波束宽度
    beamwidth_e, beamwidth_h = power_beamwidth(power_pattern, theta_values, phi_values)
    
    # 验证功率守恒
    verification = power_verification(dipole_power_density)
    
    print(f"总辐射功率: {total_power:.2e} W")
    print(f"辐射强度: {intensity:.2e} W/sr")
    print(f"功率效率: {efficiency:.3f}")
    print(f"E面波束宽度: {beamwidth_e:.1f}°")
    print(f"H面波束宽度: {beamwidth_h:.1f}°")
    print(f"功率守恒验证: {'通过' if verification else '失败'}")
    
    return (power_obj, total_power, intensity, efficiency, 
            beamwidth_e, beamwidth_h, verification)
```

---

## 2.5.7 电磁介质 / Electromagnetic Media

### 电介质 / Dielectrics

**形式化定义**: 电介质是能够在外电场作用下产生极化现象的介质，其内部电荷分布发生重新排列，形成电偶极矩。

**公理化定义**:
设 $\mathcal{D} = \langle \mathcal{E}, \mathcal{P}, \mathcal{D}, \mathcal{\chi}, \mathcal{\epsilon} \rangle$ 为电介质系统，其中：

1. **外电场**: $\mathcal{E}$ 为外加电场集合
2. **极化矢量**: $\mathcal{P}$ 为电极化强度集合
3. **电位移**: $\mathcal{D}$ 为电位移矢量集合
4. **电极化率**: $\mathcal{\chi}$ 为电极化率张量集合
5. **介电常数**: $\mathcal{\epsilon}$ 为相对介电常数集合

**等价定义**:

1. **极化关系**: $\vec{P} = \epsilon_0 \chi_e \vec{E}$
2. **电位移关系**: $\vec{D} = \epsilon_0 \vec{E} + \vec{P} = \epsilon_0 \epsilon_r \vec{E}$
3. **介电常数关系**: $\epsilon_r = 1 + \chi_e$

**形式化定理**:

**定理2.5.7.1 (电介质高斯定律)**: 电介质中的高斯定律
$$\oint_S \vec{D} \cdot d\vec{S} = Q_{free}$$

**定理2.5.7.2 (电介质边界条件)**: 电介质界面上的边界条件
$$\vec{D}_{1n} - \vec{D}_{2n} = \sigma_{free}, \quad \vec{E}_{1t} = \vec{E}_{2t}$$

**定理2.5.7.3 (电介质能量密度)**: 电介质中的电场能量密度
$$u_e = \frac{1}{2} \vec{D} \cdot \vec{E} = \frac{1}{2} \epsilon_0 \epsilon_r E^2$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple, Optional
from scipy.constants import epsilon_0

class Dielectric:
    """电介质类"""
    def __init__(self, relative_permittivity: float, 
                 susceptibility: Optional[float] = None):
        self.epsilon_r = relative_permittivity
        if susceptibility is None:
            self.chi_e = self.epsilon_r - 1
        else:
            self.chi_e = susceptibility
            self.epsilon_r = 1 + self.chi_e
    
    def get_permittivity(self) -> float:
        """获取相对介电常数"""
        return self.epsilon_r
    
    def get_susceptibility(self) -> float:
        """获取电极化率"""
        return self.chi_e
    
    def get_absolute_permittivity(self) -> float:
        """获取绝对介电常数"""
        return epsilon_0 * self.epsilon_r

def polarization_vector(electric_field: np.ndarray, 
                       susceptibility: float) -> np.ndarray:
    """
    计算电极化矢量
    
    参数:
        electric_field: 电场矢量 (V/m)
        susceptibility: 电极化率
    
    返回:
        电极化矢量 (C/m²)
    """
    return epsilon_0 * susceptibility * electric_field

def displacement_vector(electric_field: np.ndarray, 
                       relative_permittivity: float) -> np.ndarray:
    """
    计算电位移矢量
    
    参数:
        electric_field: 电场矢量 (V/m)
        relative_permittivity: 相对介电常数
    
    返回:
        电位移矢量 (C/m²)
    """
    return epsilon_0 * relative_permittivity * electric_field

def electric_field_from_displacement(displacement: np.ndarray, 
                                   relative_permittivity: float) -> np.ndarray:
    """
    从电位移计算电场
    
    参数:
        displacement: 电位移矢量 (C/m²)
        relative_permittivity: 相对介电常数
    
    返回:
        电场矢量 (V/m)
    """
    return displacement / (epsilon_0 * relative_permittivity)

def dielectric_energy_density(electric_field: np.ndarray, 
                            relative_permittivity: float) -> float:
    """
    计算电介质中的电场能量密度
    
    参数:
        electric_field: 电场矢量 (V/m)
        relative_permittivity: 相对介电常数
    
    返回:
        能量密度 (J/m³)
    """
    return 0.5 * epsilon_0 * relative_permittivity * np.dot(electric_field, electric_field)

def dielectric_boundary_conditions(e_field1: np.ndarray, e_field2: np.ndarray,
                                 d_field1: np.ndarray, d_field2: np.ndarray,
                                 normal: np.ndarray, surface_charge: float = 0.0) -> Tuple[bool, bool]:
    """
    验证电介质边界条件
    
    参数:
        e_field1, e_field2: 两侧电场 (V/m)
        d_field1, d_field2: 两侧电位移 (C/m²)
        normal: 法向量
        surface_charge: 自由面电荷密度 (C/m²)
    
    返回:
        (法向条件满足, 切向条件满足)
    """
    # 法向条件: D1n - D2n = σ_free
    d1n = np.dot(d_field1, normal)
    d2n = np.dot(d_field2, normal)
    normal_condition = abs(d1n - d2n - surface_charge) < 1e-12
    
    # 切向条件: E1t = E2t
    e1_tangential = e_field1 - np.dot(e_field1, normal) * normal
    e2_tangential = e_field2 - np.dot(e_field2, normal) * normal
    tangential_condition = np.allclose(e1_tangential, e2_tangential)
    
    return normal_condition, tangential_condition

def dielectric_gauss_law(displacement_field: Callable, surface: np.ndarray,
                        free_charge: float) -> bool:
    """
    验证电介质中的高斯定律
    
    参数:
        displacement_field: 电位移场函数
        surface: 闭合曲面点集
        free_charge: 自由电荷 (C)
    
    返回:
        是否满足高斯定律
    """
    # 简化验证：计算电位移通量
    total_flux = 0.0
    for i in range(len(surface) - 1):
        point = surface[i]
        next_point = surface[i + 1]
        d_field = displacement_field(point)
        
        # 计算面元
        ds = next_point - point
        flux = np.dot(d_field, ds)
        total_flux += flux
    
    # 闭合曲面
    if len(surface) > 0:
        d_field = displacement_field(surface[-1])
        ds = surface[0] - surface[-1]
        flux = np.dot(d_field, ds)
        total_flux += flux
    
    return abs(total_flux - free_charge) < 1e-12

def dielectric_example():
    """电介质示例"""
    # 创建电介质
    dielectric = Dielectric(relative_permittivity=2.5)
    
    # 定义电场
    electric_field = np.array([1000, 0, 0])  # 1 kV/m
    
    # 计算电极化矢量
    polarization = polarization_vector(electric_field, dielectric.get_susceptibility())
    
    # 计算电位移矢量
    displacement = displacement_vector(electric_field, dielectric.get_permittivity())
    
    # 计算能量密度
    energy_density = dielectric_energy_density(electric_field, dielectric.get_permittivity())
    
    # 验证边界条件
    e_field1 = np.array([1000, 500, 0])
    e_field2 = np.array([400, 500, 0])  # 切向分量相同
    d_field1 = displacement_vector(e_field1, 2.5)
    d_field2 = displacement_vector(e_field2, 1.0)
    normal = np.array([1, 0, 0])
    
    normal_condition, tangential_condition = dielectric_boundary_conditions(
        e_field1, e_field2, d_field1, d_field2, normal)
    
    print(f"相对介电常数: {dielectric.get_permittivity()}")
    print(f"电极化率: {dielectric.get_susceptibility()}")
    print(f"电极化矢量: {polarization} C/m²")
    print(f"电位移矢量: {displacement} C/m²")
    print(f"能量密度: {energy_density:.2e} J/m³")
    print(f"法向边界条件: {'满足' if normal_condition else '不满足'}")
    print(f"切向边界条件: {'满足' if tangential_condition else '不满足'}")
    
    return (dielectric, polarization, displacement, energy_density, 
            normal_condition, tangential_condition)
```

### 磁介质 / Magnetic Media

**形式化定义**: 磁介质是能够在外磁场作用下产生磁化现象的介质，其内部磁矩发生重新排列，形成磁化强度。

**公理化定义**:
设 $\mathcal{M} = \langle \mathcal{H}, \mathcal{M}, \mathcal{B}, \mathcal{\chi_m}, \mathcal{\mu} \rangle$ 为磁介质系统，其中：

1. **外磁场**: $\mathcal{H}$ 为外加磁场集合
2. **磁化矢量**: $\mathcal{M}$ 为磁化强度集合
3. **磁感应强度**: $\mathcal{B}$ 为磁感应强度集合
4. **磁化率**: $\mathcal{\chi_m}$ 为磁化率张量集合
5. **磁导率**: $\mathcal{\mu}$ 为相对磁导率集合

**等价定义**:

1. **磁化关系**: $\vec{M} = \chi_m \vec{H}$
2. **磁感应关系**: $\vec{B} = \mu_0(\vec{H} + \vec{M}) = \mu_0 \mu_r \vec{H}$
3. **磁导率关系**: $\mu_r = 1 + \chi_m$

**形式化定理**:

**定理2.5.7.4 (磁介质安培定律)**: 磁介质中的安培定律
$$\oint_C \vec{H} \cdot d\vec{l} = I_{free}$$

**定理2.5.7.5 (磁介质边界条件)**: 磁介质界面上的边界条件
$$\vec{B}_{1n} = \vec{B}_{2n}, \quad \vec{H}_{1t} - \vec{H}_{2t} = \vec{K}_{free}$$

**定理2.5.7.6 (磁介质能量密度)**: 磁介质中的磁场能量密度
$$u_m = \frac{1}{2} \vec{B} \cdot \vec{H} = \frac{1}{2} \mu_0 \mu_r H^2$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple, Optional
from scipy.constants import mu_0

class MagneticMedia:
    """磁介质类"""
    def __init__(self, relative_permeability: float, 
                 susceptibility: Optional[float] = None):
        self.mu_r = relative_permeability
        if susceptibility is None:
            self.chi_m = self.mu_r - 1
        else:
            self.chi_m = susceptibility
            self.mu_r = 1 + self.chi_m
    
    def get_permeability(self) -> float:
        """获取相对磁导率"""
        return self.mu_r
    
    def get_susceptibility(self) -> float:
        """获取磁化率"""
        return self.chi_m
    
    def get_absolute_permeability(self) -> float:
        """获取绝对磁导率"""
        return mu_0 * self.mu_r

def magnetization_vector(magnetic_field: np.ndarray, 
                        susceptibility: float) -> np.ndarray:
    """
    计算磁化矢量
    
    参数:
        magnetic_field: 磁场矢量 (A/m)
        susceptibility: 磁化率
    
    返回:
        磁化矢量 (A/m)
    """
    return susceptibility * magnetic_field

def magnetic_induction(magnetic_field: np.ndarray, 
                      relative_permeability: float) -> np.ndarray:
    """
    计算磁感应强度
    
    参数:
        magnetic_field: 磁场矢量 (A/m)
        relative_permeability: 相对磁导率
    
    返回:
        磁感应强度 (T)
    """
    return mu_0 * relative_permeability * magnetic_field

def magnetic_field_from_induction(induction: np.ndarray, 
                                relative_permeability: float) -> np.ndarray:
    """
    从磁感应强度计算磁场
    
    参数:
        induction: 磁感应强度 (T)
        relative_permeability: 相对磁导率
    
    返回:
        磁场矢量 (A/m)
    """
    return induction / (mu_0 * relative_permeability)

def magnetic_energy_density(magnetic_field: np.ndarray, 
                          relative_permeability: float) -> float:
    """
    计算磁介质中的磁场能量密度
    
    参数:
        magnetic_field: 磁场矢量 (A/m)
        relative_permeability: 相对磁导率
    
    返回:
        能量密度 (J/m³)
    """
    return 0.5 * mu_0 * relative_permeability * np.dot(magnetic_field, magnetic_field)

def magnetic_boundary_conditions(h_field1: np.ndarray, h_field2: np.ndarray,
                               b_field1: np.ndarray, b_field2: np.ndarray,
                               normal: np.ndarray, surface_current: np.ndarray = np.zeros(3)) -> Tuple[bool, bool]:
    """
    验证磁介质边界条件
    
    参数:
        h_field1, h_field2: 两侧磁场 (A/m)
        b_field1, b_field2: 两侧磁感应强度 (T)
        normal: 法向量
        surface_current: 自由面电流密度 (A/m)
    
    返回:
        (法向条件满足, 切向条件满足)
    """
    # 法向条件: B1n = B2n
    b1n = np.dot(b_field1, normal)
    b2n = np.dot(b_field2, normal)
    normal_condition = abs(b1n - b2n) < 1e-12
    
    # 切向条件: H1t - H2t = K_free
    h1_tangential = h_field1 - np.dot(h_field1, normal) * normal
    h2_tangential = h_field2 - np.dot(h_field2, normal) * normal
    tangential_difference = h1_tangential - h2_tangential
    tangential_condition = np.allclose(tangential_difference, surface_current)
    
    return normal_condition, tangential_condition

def magnetic_ampere_law(magnetic_field: Callable, path: np.ndarray,
                       free_current: float) -> bool:
    """
    验证磁介质中的安培定律
    
    参数:
        magnetic_field: 磁场函数
        path: 闭合路径点集
        free_current: 自由电流 (A)
    
    返回:
        是否满足安培定律
    """
    # 简化验证：计算磁场环流
    total_circulation = 0.0
    for i in range(len(path) - 1):
        point = path[i]
        next_point = path[i + 1]
        h_field = magnetic_field(point)
        
        # 计算路径元
        dl = next_point - point
        circulation = np.dot(h_field, dl)
        total_circulation += circulation
    
    # 闭合路径
    if len(path) > 0:
        h_field = magnetic_field(path[-1])
        dl = path[0] - path[-1]
        circulation = np.dot(h_field, dl)
        total_circulation += circulation
    
    return abs(total_circulation - free_current) < 1e-12

def magnetic_media_example():
    """磁介质示例"""
    # 创建磁介质
    magnetic_media = MagneticMedia(relative_permeability=1000)  # 铁磁材料
    
    # 定义磁场
    magnetic_field = np.array([100, 0, 0])  # 100 A/m
    
    # 计算磁化矢量
    magnetization = magnetization_vector(magnetic_field, magnetic_media.get_susceptibility())
    
    # 计算磁感应强度
    induction = magnetic_induction(magnetic_field, magnetic_media.get_permeability())
    
    # 计算能量密度
    energy_density = magnetic_energy_density(magnetic_field, magnetic_media.get_permeability())
    
    # 验证边界条件
    h_field1 = np.array([100, 50, 0])
    h_field2 = np.array([0.1, 50, 0])  # 切向分量相同
    b_field1 = magnetic_induction(h_field1, 1000)
    b_field2 = magnetic_induction(h_field2, 1.0)
    normal = np.array([1, 0, 0])
    
    normal_condition, tangential_condition = magnetic_boundary_conditions(
        h_field1, h_field2, b_field1, b_field2, normal)
    
    print(f"相对磁导率: {magnetic_media.get_permeability()}")
    print(f"磁化率: {magnetic_media.get_susceptibility()}")
    print(f"磁化矢量: {magnetization} A/m")
    print(f"磁感应强度: {induction} T")
    print(f"能量密度: {energy_density:.2e} J/m³")
    print(f"法向边界条件: {'满足' if normal_condition else '不满足'}")
    print(f"切向边界条件: {'满足' if tangential_condition else '不满足'}")
    
    return (magnetic_media, magnetization, induction, energy_density, 
            normal_condition, tangential_condition)
```

### 色散关系 / Dispersion Relations

**形式化定义**: 色散关系描述了电磁波在介质中传播时，频率与波数之间的函数关系，反映了介质的频率响应特性。

**公理化定义**:
设 $\mathcal{DR} = \langle \mathcal{\omega}, \mathcal{k}, \mathcal{n}, \mathcal{v}, \mathcal{\alpha} \rangle$ 为色散关系系统，其中：

1. **角频率**: $\mathcal{\omega}$ 为角频率集合
2. **波数**: $\mathcal{k}$ 为波数集合
3. **折射率**: $\mathcal{n}$ 为复折射率集合
4. **相速度**: $\mathcal{v}$ 为相速度集合
5. **衰减系数**: $\mathcal{\alpha}$ 为衰减系数集合

**等价定义**:

1. **色散关系**: $\omega = \omega(k)$ 或 $k = k(\omega)$
2. **折射率关系**: $n(\omega) = \sqrt{\epsilon_r(\omega) \mu_r(\omega)}$
3. **相速度关系**: $v_p(\omega) = \frac{c}{n(\omega)}$

**形式化定理**:

**定理2.5.7.7 (色散关系因果性)**: 色散关系满足因果性条件
$$\text{Im}[n(\omega)] \geq 0 \quad \text{for} \quad \text{Im}[\omega] > 0$$

**定理2.5.7.8 (克拉默-克朗尼关系)**: 折射率的实部和虚部满足克拉默-克朗尼关系
$$\text{Re}[n(\omega)] = 1 + \frac{2}{\pi} P \int_0^{\infty} \frac{\omega' \text{Im}[n(\omega')]}{\omega'^2 - \omega^2} d\omega'$$

**定理2.5.7.9 (群速度关系)**: 群速度与相速度的关系
$$v_g = \frac{d\omega}{dk} = v_p + k \frac{dv_p}{dk}$$

**Python算法实现**:

```python
import numpy as np
from typing import Callable, Tuple, Complex
from scipy.constants import c
from scipy.integrate import quad

class DispersionRelation:
    """色散关系类"""
    def __init__(self, epsilon_func: Callable, mu_func: Callable):
        self.epsilon_func = epsilon_func
        self.mu_func = mu_func
    
    def refractive_index(self, frequency: float) -> complex:
        """
        计算复折射率
        
        参数:
            frequency: 频率 (Hz)
        
        返回:
            复折射率
        """
        omega = 2 * np.pi * frequency
        epsilon_r = self.epsilon_func(omega)
        mu_r = self.mu_func(omega)
        return np.sqrt(epsilon_r * mu_r)
    
    def phase_velocity(self, frequency: float) -> complex:
        """
        计算相速度
        
        参数:
            frequency: 频率 (Hz)
        
        返回:
            相速度 (m/s)
        """
        n = self.refractive_index(frequency)
        return c / n
    
    def group_velocity(self, frequency: float, delta_f: float = 1e6) -> complex:
        """
        计算群速度
        
        参数:
            frequency: 频率 (Hz)
            delta_f: 频率差 (Hz)
        
        返回:
            群速度 (m/s)
        """
        omega = 2 * np.pi * frequency
        delta_omega = 2 * np.pi * delta_f
        
        # 数值微分
        k1 = omega * self.refractive_index(frequency) / c
        k2 = (omega + delta_omega) * self.refractive_index(frequency + delta_f) / c
        
        dk_domega = (k2 - k1) / delta_omega
        return 1 / dk_domega

def dispersion_relation_omega_k(omega: float, medium_params: dict) -> float:
    """
    计算色散关系 ω(k)
    
    参数:
        omega: 角频率 (rad/s)
        medium_params: 介质参数字典
    
    返回:
        波数 k (rad/m)
    """
    # 简化的色散关系：k = ω/v
    phase_velocity = medium_params.get('phase_velocity', c)
    return omega / phase_velocity

def dispersion_relation_k_omega(k: float, medium_params: dict) -> float:
    """
    计算色散关系 k(ω)
    
    参数:
        k: 波数 (rad/m)
        medium_params: 介质参数字典
    
    返回:
        角频率 ω (rad/s)
    """
    # 简化的色散关系：ω = kv
    phase_velocity = medium_params.get('phase_velocity', c)
    return k * phase_velocity

def attenuation_coefficient(refractive_index: complex, frequency: float) -> float:
    """
    计算衰减系数
    
    参数:
        refractive_index: 复折射率
        frequency: 频率 (Hz)
    
    返回:
        衰减系数 (m⁻¹)
    """
    omega = 2 * np.pi * frequency
    k_imag = omega * np.imag(refractive_index) / c
    return k_imag

def propagation_constant(refractive_index: complex, frequency: float) -> complex:
    """
    计算传播常数
    
    参数:
        refractive_index: 复折射率
        frequency: 频率 (Hz)
    
    返回:
        传播常数 (m⁻¹)
    """
    omega = 2 * np.pi * frequency
    return omega * refractive_index / c

def kramers_kronig_relation(imaginary_part: Callable, omega: float, 
                           omega_max: float = 1e15) -> float:
    """
    计算克拉默-克朗尼关系
    
    参数:
        imaginary_part: 虚部函数
        omega: 角频率 (rad/s)
        omega_max: 积分上限
    
    返回:
        实部值
    """
    def integrand(omega_prime):
        if abs(omega_prime**2 - omega**2) < 1e-12:
            return 0.0
        return omega_prime * imaginary_part(omega_prime) / (omega_prime**2 - omega**2)
    
    result, _ = quad(integrand, 0, omega_max)
    return 1 + (2 / np.pi) * result

def causality_condition(refractive_index: complex) -> bool:
    """
    验证因果性条件
    
    参数:
        refractive_index: 复折射率
    
    返回:
        是否满足因果性
    """
    # 简化验证：检查虚部是否非负
    return np.imag(refractive_index) >= 0

def dispersion_verification(dispersion_obj: DispersionRelation, 
                          frequency_range: np.ndarray) -> bool:
    """
    验证色散关系
    
    参数:
        dispersion_obj: 色散关系对象
        frequency_range: 频率范围 (Hz)
    
    返回:
        是否满足色散关系
    """
    for freq in frequency_range:
        n = dispersion_obj.refractive_index(freq)
        v_p = dispersion_obj.phase_velocity(freq)
        v_g = dispersion_obj.group_velocity(freq)
        
        # 基本关系验证
        if abs(v_p - c / n) > 1e-12:
            return False
        
        # 因果性验证
        if not causality_condition(n):
            return False
    
    return True

def dispersion_example():
    """色散关系示例"""
    # 定义介电常数函数（德鲁德模型）
    def epsilon_drude(omega):
        omega_p = 2e15  # 等离子体频率
        gamma = 1e13    # 碰撞频率
        return 1 - omega_p**2 / (omega**2 + 1j * omega * gamma)
    
    # 定义磁导率函数（常数）
    def mu_constant(omega):
        return 1.0
    
    # 创建色散关系对象
    dispersion = DispersionRelation(epsilon_drude, mu_constant)
    
    # 计算不同频率的折射率
    frequencies = np.array([1e14, 2e14, 5e14, 1e15])
    refractive_indices = [dispersion.refractive_index(f) for f in frequencies]
    
    # 计算相速度和群速度
    phase_velocities = [dispersion.phase_velocity(f) for f in frequencies]
    group_velocities = [dispersion.group_velocity(f) for f in frequencies]
    
    # 计算衰减系数
    attenuation_coeffs = [attenuation_coefficient(n, f) for n, f in zip(refractive_indices, frequencies)]
    
    # 验证因果性
    causality_check = [causality_condition(n) for n in refractive_indices]
    
    # 验证色散关系
    verification = dispersion_verification(dispersion, frequencies)
    
    print("色散关系分析:")
    for i, freq in enumerate(frequencies):
        print(f"频率: {freq:.1e} Hz")
        print(f"  折射率: {refractive_indices[i]:.3f}")
        print(f"  相速度: {phase_velocities[i]:.2e} m/s")
        print(f"  群速度: {group_velocities[i]:.2e} m/s")
        print(f"  衰减系数: {attenuation_coeffs[i]:.2e} m⁻¹")
        print(f"  因果性: {'满足' if causality_check[i] else '不满足'}")
        print()
    
    print(f"色散关系验证: {'通过' if verification else '失败'}")
    
    return (dispersion, refractive_indices, phase_velocities, 
            group_velocities, attenuation_coeffs, causality_check, verification)
```

---

*最后更新: 2025-08-01*
*版本: 6.0.0*
