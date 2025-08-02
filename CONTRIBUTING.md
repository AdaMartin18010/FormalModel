# 贡献指南 / Contributing Guide

## 欢迎贡献 / Welcome Contributions

感谢您对形式化模型项目的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 **问题报告**: 报告bug和问题
- 💡 **功能建议**: 提出新功能和改进建议
- 📝 **文档改进**: 改进文档和教程
- 🔧 **代码贡献**: 提交代码和实现
- 🌍 **翻译工作**: 多语言翻译
- 🧪 **测试贡献**: 编写测试用例
- 📚 **示例贡献**: 提供使用示例

## 快速开始 / Quick Start

### 1. Fork项目 / Fork the Project

1. 访问项目GitHub页面
2. 点击右上角的"Fork"按钮
3. 选择您的GitHub账户
4. 等待Fork完成

### 2. 克隆仓库 / Clone the Repository

```bash
# 克隆您的Fork
git clone https://github.com/YOUR_USERNAME/FormalModel.git

# 进入项目目录
cd FormalModel

# 添加上游仓库
git remote add upstream https://github.com/ORIGINAL_OWNER/FormalModel.git
```

### 3. 创建分支 / Create a Branch

```bash
# 创建新分支
git checkout -b feature/your-feature-name

# 或者创建修复分支
git checkout -b fix/your-bug-fix
```

### 4. 进行修改 / Make Changes

根据您要贡献的内容类型，进行相应的修改：

#### 代码贡献 / Code Contributions

```rust
// Rust代码示例
pub struct FormalModel {
    pub name: String,
    pub parameters: HashMap<String, f64>,
    pub equations: Vec<String>,
}

impl FormalModel {
    pub fn new(name: String) -> Self {
        Self {
            name,
            parameters: HashMap::new(),
            equations: Vec::new(),
        }
    }
    
    pub fn add_parameter(&mut self, name: &str, value: f64) {
        self.parameters.insert(name.to_string(), value);
    }
    
    pub fn add_equation(&mut self, equation: &str) {
        self.equations.push(equation.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_formal_model_creation() {
        let model = FormalModel::new("Test Model".to_string());
        assert_eq!(model.name, "Test Model");
        assert!(model.parameters.is_empty());
        assert!(model.equations.is_empty());
    }
    
    #[test]
    fn test_add_parameter() {
        let mut model = FormalModel::new("Test Model".to_string());
        model.add_parameter("mass", 1.0);
        assert_eq!(model.parameters.get("mass"), Some(&1.0));
    }
}
```

#### 文档贡献 / Documentation Contributions

```markdown
# 新模型文档模板

## 模型名称 / Model Name

### 概述 / Overview

简要描述模型的目的和应用场景。

### 数学基础 / Mathematical Foundation

#### 基本定义 / Basic Definitions

**定义1**: 模型的基本概念定义

$$f(x) = \int_{-\infty}^{\infty} g(t) e^{-i\omega t} dt$$

#### 主要定理 / Main Theorems

**定理1**: 重要定理的陈述和证明

**证明**: 详细的数学证明过程

### 实现示例 / Implementation Examples

#### Rust实现 / Rust Implementation

```rust
pub struct ModelName {
    // 模型参数
}

impl ModelName {
    pub fn new() -> Self {
        // 构造函数
    }
    
    pub fn compute(&self, input: f64) -> f64 {
        // 计算实现
    }
}
```

#### Python实现 / Python Implementation

```python
class ModelName:
    def __init__(self, parameters):
        self.parameters = parameters
    
    def compute(self, input_value):
        """计算模型输出"""
        return result
    
    def verify(self):
        """验证模型性质"""
        return verification_result
```

### 应用案例 / Application Cases

描述模型在实际问题中的应用。

### 参考文献 / References

1. 作者. (年份). 标题. 期刊/会议.
2. 作者. (年份). 标题. 出版社.

```

### 5. 提交更改 / Commit Changes

```bash
# 添加修改的文件
git add .

# 提交更改
git commit -m "feat: add new formal model for quantum systems

- Add quantum harmonic oscillator model
- Implement wave function visualization
- Add comprehensive test suite
- Update documentation with examples

Closes #123"

# 推送到您的Fork
git push origin feature/your-feature-name
```

### 6. 创建Pull Request / Create Pull Request

1. 访问您的Fork页面
2. 点击"Compare & pull request"
3. 填写PR描述
4. 提交PR

## 贡献类型 / Types of Contributions

### 🐛 Bug报告 / Bug Reports

#### 报告模板 / Report Template

```markdown
## Bug描述 / Bug Description

### 问题概述 / Summary
简要描述遇到的问题。

### 重现步骤 / Steps to Reproduce
1. 打开...
2. 点击...
3. 看到错误...

### 预期行为 / Expected Behavior
描述您期望看到的行为。

### 实际行为 / Actual Behavior
描述实际发生的行为。

### 环境信息 / Environment
- 操作系统: [如 Windows 10, macOS 11.0, Ubuntu 20.04]
- 编程语言版本: [如 Python 3.9, Rust 1.70]
- 依赖库版本: [如 numpy 1.21, matplotlib 3.5]

### 附加信息 / Additional Information
- 错误日志
- 截图
- 相关代码片段
```

### 💡 功能建议 / Feature Requests

#### 建议模板 / Request Template

```markdown
## 功能建议 / Feature Request

### 问题描述 / Problem Statement
描述当前缺少的功能或需要改进的地方。

### 解决方案 / Proposed Solution
描述您建议的解决方案。

### 替代方案 / Alternative Solutions
如果有其他解决方案，请列出。

### 影响评估 / Impact Assessment
- 对现有功能的影响
- 性能影响
- 兼容性影响

### 实现建议 / Implementation Suggestions
如果可能，提供实现建议或代码示例。
```

### 📝 文档改进 / Documentation Improvements

#### 文档标准 / Documentation Standards

1. **结构清晰**: 使用清晰的标题和层次结构
2. **内容准确**: 确保技术内容的准确性
3. **示例丰富**: 提供充分的代码示例
4. **多语言支持**: 中英双语对照
5. **格式统一**: 遵循项目的文档格式

```markdown
# 文档改进示例

## 原文档
```python
def calculate_energy(mass, velocity):
    return 0.5 * mass * velocity**2
```

## 改进后

```python
def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """
    计算物体的动能
    
    Args:
        mass: 物体质量 (kg)
        velocity: 物体速度 (m/s)
    
    Returns:
        动能 (J)
    
    Raises:
        ValueError: 当质量或速度为负数时
    
    Examples:
        >>> calculate_kinetic_energy(2.0, 3.0)
        9.0
        >>> calculate_kinetic_energy(1.0, 5.0)
        12.5
    """
    if mass < 0 or velocity < 0:
        raise ValueError("质量和速度必须为正数")
    
    return 0.5 * mass * velocity**2
```

```

### 🔧 代码贡献 / Code Contributions

#### 代码标准 / Code Standards

##### Rust代码标准 / Rust Code Standards

```rust
// 1. 命名规范
pub struct QuantumSystem {
    pub hamiltonian: Matrix,
    pub wave_function: Vector,
}

impl QuantumSystem {
    // 2. 文档注释
    /// 创建新的量子系统
    /// 
    /// # Arguments
    /// 
    /// * `hamiltonian` - 哈密顿算符矩阵
    /// * `initial_state` - 初始波函数
    /// 
    /// # Returns
    /// 
    /// 返回新的量子系统实例
    /// 
    /// # Examples
    /// 
    /// ```
    /// use quantum_models::QuantumSystem;
    /// 
    /// let hamiltonian = Matrix::new(2, 2);
    /// let initial_state = Vector::new(2);
    /// let system = QuantumSystem::new(hamiltonian, initial_state);
    /// ```
    pub fn new(hamiltonian: Matrix, initial_state: Vector) -> Self {
        Self {
            hamiltonian,
            wave_function: initial_state,
        }
    }
    
    // 3. 错误处理
    pub fn evolve(&mut self, time: f64) -> Result<(), QuantumError> {
        if time < 0.0 {
            return Err(QuantumError::InvalidTime(time));
        }
        
        // 时间演化计算
        self.wave_function = self.calculate_evolution(time)?;
        Ok(())
    }
    
    // 4. 私有方法
    fn calculate_evolution(&self, time: f64) -> Result<Vector, QuantumError> {
        // 实现细节
        Ok(Vector::new(2))
    }
}

// 5. 测试
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_system_creation() {
        let hamiltonian = Matrix::new(2, 2);
        let initial_state = Vector::new(2);
        let system = QuantumSystem::new(hamiltonian, initial_state);
        
        assert_eq!(system.wave_function.dim(), 2);
    }
    
    #[test]
    fn test_evolution_with_negative_time() {
        let hamiltonian = Matrix::new(2, 2);
        let initial_state = Vector::new(2);
        let mut system = QuantumSystem::new(hamiltonian, initial_state);
        
        let result = system.evolve(-1.0);
        assert!(result.is_err());
    }
}
```

### Haskell代码标准 / Haskell Code Standards

```haskell
-- 1. 模块声明
module Physics.Quantum.HarmonicOscillator
  ( HarmonicOscillator(..)
  , createOscillator
  , evolveState
  , energy
  ) where

-- 2. 导入声明
import Data.Complex
import Data.Vector (Vector)
import qualified Data.Vector as V

-- 3. 类型定义
data HarmonicOscillator = HarmonicOscillator
  { mass :: Double
  , frequency :: Double
  , position :: Complex Double
  , momentum :: Complex Double
  } deriving (Show, Eq)

-- 4. 类型类实例
instance Num HarmonicOscillator where
  (+) = addOscillators
  (*) = multiplyOscillators
  abs = absOscillator
  signum = signumOscillator
  fromInteger = fromIntegerOscillator
  negate = negateOscillator

-- 5. 主要函数
-- | 创建简谐振荡器
-- 
-- @
-- createOscillator 1.0 2.0 0.0 1.0
-- @
createOscillator :: Double -> Double -> Double -> Double -> HarmonicOscillator
createOscillator m omega x p = HarmonicOscillator
  { mass = m
  , frequency = omega
  , position = x :+ 0
  , momentum = p :+ 0
  }

-- | 计算系统能量
energy :: HarmonicOscillator -> Double
energy osc = kinetic + potential
  where
    kinetic = (magnitude (momentum osc) ^ 2) / (2 * mass osc)
    potential = 0.5 * mass osc * frequency osc ^ 2 * (magnitude (position osc) ^ 2)

-- 6. 辅助函数
addOscillators :: HarmonicOscillator -> HarmonicOscillator -> HarmonicOscillator
addOscillators osc1 osc2 = HarmonicOscillator
  { mass = mass osc1 + mass osc2
  , frequency = frequency osc1
  , position = position osc1 + position osc2
  , momentum = momentum osc1 + momentum osc2
  }

-- 7. 测试
-- | 属性测试
prop_energy_conservation :: HarmonicOscillator -> Bool
prop_energy_conservation osc = 
  let initial_energy = energy osc
      evolved_osc = evolveState osc 1.0
      final_energy = energy evolved_osc
  in abs (initial_energy - final_energy) < 1e-10
```

#### Python代码标准 / Python Code Standards

```python
"""
量子力学模型实现

本模块提供了量子力学相关的基础模型和计算工具。
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class QuantumState:
    """量子状态类"""
    amplitude: np.ndarray
    energy: float
    
    def __post_init__(self):
        """验证量子状态的有效性"""
        if not np.isclose(np.sum(np.abs(self.amplitude)**2), 1.0):
            raise ValueError("量子状态必须归一化")

class QuantumSystem(ABC):
    """量子系统抽象基类"""
    
    def __init__(self, hamiltonian: np.ndarray, initial_state: QuantumState):
        """
        初始化量子系统
        
        Args:
            hamiltonian: 哈密顿算符矩阵
            initial_state: 初始量子状态
        
        Raises:
            ValueError: 当哈密顿矩阵不是厄米矩阵时
        """
        self.hamiltonian = hamiltonian
        self.current_state = initial_state
        
        # 验证哈密顿矩阵的厄米性
        if not np.allclose(self.hamiltonian, self.hamiltonian.conj().T):
            raise ValueError("哈密顿矩阵必须是厄米矩阵")
    
    @abstractmethod
    def evolve(self, time: float) -> QuantumState:
        """
        时间演化
        
        Args:
            time: 演化时间
        
        Returns:
            演化后的量子状态
        """
        pass
    
    def energy(self) -> float:
        """计算系统能量"""
        return self.current_state.energy
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """
        计算可观测量期望值
        
        Args:
            observable: 可观测量算符
        
        Returns:
            期望值
        """
        return np.dot(self.current_state.amplitude.conj(),
                     np.dot(observable, self.current_state.amplitude))

class HarmonicOscillator(QuantumSystem):
    """简谐振荡器量子系统"""
    
    def __init__(self, omega: float, n_levels: int = 10):
        """
        初始化简谐振荡器
        
        Args:
            omega: 角频率
            n_levels: 能级数量
        """
        # 构建哈密顿矩阵
        hamiltonian = self._build_hamiltonian(omega, n_levels)
        
        # 初始状态（基态）
        initial_state = QuantumState(
            amplitude=np.zeros(n_levels, dtype=complex),
            energy=0.5 * omega
        )
        initial_state.amplitude[0] = 1.0
        
        super().__init__(hamiltonian, initial_state)
        self.omega = omega
        self.n_levels = n_levels
    
    def _build_hamiltonian(self, omega: float, n_levels: int) -> np.ndarray:
        """构建哈密顿矩阵"""
        hamiltonian = np.zeros((n_levels, n_levels), dtype=complex)
        
        for n in range(n_levels):
            hamiltonian[n, n] = (n + 0.5) * omega
        
        return hamiltonian
    
    def evolve(self, time: float) -> QuantumState:
        """时间演化"""
        # 计算演化算符
        evolution_operator = np.exp(-1j * self.hamiltonian * time)
        
        # 演化量子状态
        new_amplitude = np.dot(evolution_operator, self.current_state.amplitude)
        
        # 更新状态
        self.current_state = QuantumState(
            amplitude=new_amplitude,
            energy=self.current_state.energy
        )
        
        return self.current_state

# 测试
def test_harmonic_oscillator():
    """测试简谐振荡器"""
    # 创建系统
    oscillator = HarmonicOscillator(omega=1.0, n_levels=5)
    
    # 检查初始能量
    assert np.isclose(oscillator.energy(), 0.5)
    
    # 时间演化
    evolved_state = oscillator.evolve(time=1.0)
    
    # 检查归一化
    norm = np.sum(np.abs(evolved_state.amplitude)**2)
    assert np.isclose(norm, 1.0, atol=1e-10)
    
    print("简谐振荡器测试通过！")

if __name__ == "__main__":
    test_harmonic_oscillator()
```

### 🌍 翻译贡献 / Translation Contributions

#### 翻译指南 / Translation Guidelines

1. **保持一致性**: 使用统一的术语翻译
2. **保持准确性**: 确保技术术语的准确性
3. **保持可读性**: 确保翻译后的文本易于理解
4. **保持完整性**: 确保所有内容都被翻译

```markdown
# 翻译对照表

## 基础术语 / Basic Terms
- Formal Model → 形式化模型
- Mathematical Model → 数学模型
- Physical Model → 物理模型
- Computer Model → 计算机模型
- Verification → 验证
- Validation → 确认
- Simulation → 模拟
- Analysis → 分析

## 技术术语 / Technical Terms
- Theorem Proving → 定理证明
- Model Checking → 模型检查
- Type System → 类型系统
- Algebraic Structure → 代数结构
- Topological Space → 拓扑空间
- Quantum State → 量子状态
- Wave Function → 波函数
- Hamiltonian → 哈密顿量

## 行业术语 / Industry Terms
- Risk Management → 风险管理
- Portfolio Optimization → 投资组合优化
- Supply Chain → 供应链
- Energy System → 能源系统
- Manufacturing Process → 制造过程
- Healthcare System → 医疗系统
- Education Platform → 教育平台
```

### 🧪 测试贡献 / Testing Contributions

#### 测试标准 / Testing Standards

```python
# Python测试示例
import unittest
import numpy as np
from physics.quantum import HarmonicOscillator

class TestHarmonicOscillator(unittest.TestCase):
    """简谐振荡器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.oscillator = HarmonicOscillator(omega=1.0, n_levels=5)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.oscillator.omega, 1.0)
        self.assertEqual(self.oscillator.n_levels, 5)
        self.assertIsNotNone(self.oscillator.hamiltonian)
    
    def test_energy_conservation(self):
        """测试能量守恒"""
        initial_energy = self.oscillator.energy()
        
        # 演化一段时间
        self.oscillator.evolve(time=1.0)
        
        final_energy = self.oscillator.energy()
        self.assertAlmostEqual(initial_energy, final_energy, places=10)
    
    def test_normalization(self):
        """测试归一化"""
        state = self.oscillator.current_state
        norm = np.sum(np.abs(state.amplitude)**2)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        with self.assertRaises(ValueError):
            HarmonicOscillator(omega=-1.0)
        
        with self.assertRaises(ValueError):
            HarmonicOscillator(omega=1.0, n_levels=0)

if __name__ == '__main__':
    unittest.main()
```

```rust
// Rust测试示例
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harmonic_oscillator_creation() {
        let oscillator = HarmonicOscillator::new(1.0, 5);
        assert_eq!(oscillator.omega(), 1.0);
        assert_eq!(oscillator.n_levels(), 5);
    }
    
    #[test]
    fn test_energy_conservation() {
        let mut oscillator = HarmonicOscillator::new(1.0, 5);
        let initial_energy = oscillator.energy();
        
        oscillator.evolve(1.0).unwrap();
        let final_energy = oscillator.energy();
        
        assert!((initial_energy - final_energy).abs() < 1e-10);
    }
    
    #[test]
    fn test_normalization() {
        let oscillator = HarmonicOscillator::new(1.0, 5);
        let state = oscillator.current_state();
        let norm: f64 = state.amplitude.iter().map(|x| x.norm_sqr()).sum();
        
        assert!((norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(HarmonicOscillator::new(-1.0, 5).is_err());
        assert!(HarmonicOscillator::new(1.0, 0).is_err());
    }
}
```

## 代码审查 / Code Review

### 审查清单 / Review Checklist

#### 功能审查 / Functional Review

- [ ] 功能实现正确
- [ ] 边界条件处理
- [ ] 错误处理完善
- [ ] 性能考虑充分

#### 代码质量审查 / Code Quality Review

- [ ] 代码风格一致
- [ ] 命名规范合理
- [ ] 注释充分
- [ ] 文档完整

#### 测试审查 / Testing Review

- [ ] 单元测试覆盖
- [ ] 集成测试完整
- [ ] 边界测试充分
- [ ] 性能测试合理

#### 安全审查 / Security Review

- [ ] 输入验证
- [ ] 权限控制
- [ ] 数据保护
- [ ] 漏洞检查

### 审查流程 / Review Process

1. **自动检查**: CI/CD自动运行检查
2. **同行审查**: 团队成员审查
3. **专家审查**: 领域专家审查
4. **最终审查**: 维护者最终审查

## 发布流程 / Release Process

### 版本管理 / Version Management

#### 语义化版本 / Semantic Versioning

```bash
# 版本格式: MAJOR.MINOR.PATCH
# 示例: 1.2.3

# MAJOR: 不兼容的API修改
# MINOR: 向下兼容的功能性新增
# PATCH: 向下兼容的问题修正
```

#### 发布分支 / Release Branches

```bash
# 创建发布分支
git checkout -b release/v1.2.0

# 更新版本号
# 更新CHANGELOG.md
# 更新文档

# 合并到主分支
git checkout main
git merge release/v1.2.0

# 创建标签
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0
```

### 变更日志 / Changelog

```markdown
# 变更日志格式

## [1.2.0] - 2025-08-01

### 新增 / Added
- 新增量子力学模型
- 新增机器学习算法
- 新增可视化工具

### 修改 / Changed
- 优化性能算法
- 改进错误处理
- 更新文档结构

### 修复 / Fixed
- 修复内存泄漏问题
- 修复并发安全问题
- 修复文档错误

### 移除 / Removed
- 移除过时的API
- 移除不安全的函数
```

## 社区准则 / Community Guidelines

### 行为准则 / Code of Conduct

1. **尊重他人**: 尊重所有贡献者
2. **建设性讨论**: 保持建设性的讨论氛围
3. **包容性**: 欢迎不同背景的贡献者
4. **专业性**: 保持专业的技术讨论

### 沟通指南 / Communication Guidelines

#### 问题讨论 / Issue Discussion

- 使用清晰的语言描述问题
- 提供充分的信息和上下文
- 保持礼貌和建设性
- 及时回应和跟进

#### 1代码审查 / Code Review

- 提供建设性的反馈
- 关注代码质量和功能正确性
- 尊重不同的编程风格
- 鼓励学习和改进

### 奖励机制 / Recognition System

#### 贡献者等级 / Contributor Levels

- **新手贡献者**: 首次贡献
- **活跃贡献者**: 定期贡献
- **核心贡献者**: 重要贡献
- **维护者**: 项目维护

#### 奖励方式 / Recognition Methods

- 贡献者名单
- 特殊徽章
- 项目致谢
- 推荐信

## 联系方式 / Contact Information

### 📧 主要联系方式 / Primary Contact

- **GitHub Issues**: 通过GitHub Issues提交问题
- **讨论区**: 参与GitHub Discussions
- **邮件列表**: 订阅项目邮件列表

### 🤝 社区渠道 / Community Channels

- **Slack**: 实时讨论
- **Discord**: 社区交流
- **微信群**: 中文用户交流
- **Telegram**: 国际用户交流

### 📚 学习资源 / Learning Resources

- **官方文档**: 项目官方文档
- **教程视频**: 在线教程视频
- **示例代码**: 丰富的代码示例
- **最佳实践**: 开发最佳实践

---

*最后更新: 2025-08-01*  
*版本: 1.0.0*  
*状态: 活跃维护 / Actively Maintained*
