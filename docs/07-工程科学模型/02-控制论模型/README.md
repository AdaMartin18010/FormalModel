# 7.2 控制论模型 / Cybernetics Models

## 目录 / Table of Contents

- [7.2 控制论模型 / Cybernetics Models](#72-控制论模型--cybernetics-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.2.1 反馈控制系统 / Feedback Control Systems](#721-反馈控制系统--feedback-control-systems)
    - [开环控制 / Open-Loop Control](#开环控制--open-loop-control)
    - [闭环控制 / Closed-Loop Control](#闭环控制--closed-loop-control)
    - [PID控制器 / PID Controller](#pid控制器--pid-controller)
  - [7.2.2 系统稳定性分析 / System Stability Analysis](#722-系统稳定性分析--system-stability-analysis)
    - [Lyapunov稳定性 / Lyapunov Stability](#lyapunov稳定性--lyapunov-stability)
    - [Routh-Hurwitz判据 / Routh-Hurwitz Criterion](#routh-hurwitz判据--routh-hurwitz-criterion)
    - [Nyquist判据 / Nyquist Criterion](#nyquist判据--nyquist-criterion)
  - [7.2.3 状态空间模型 / State Space Models](#723-状态空间模型--state-space-models)
    - [线性系统 / Linear Systems](#线性系统--linear-systems)
    - [非线性系统 / Nonlinear Systems](#非线性系统--nonlinear-systems)
    - [可观性和可控性 / Observability and Controllability](#可观性和可控性--observability-and-controllability)
  - [7.2.4 最优控制理论 / Optimal Control Theory](#724-最优控制理论--optimal-control-theory)
    - [变分法 / Calculus of Variations](#变分法--calculus-of-variations)
    - [动态规划 / Dynamic Programming](#动态规划--dynamic-programming)
    - [Pontryagin极大值原理 / Pontryagin Maximum Principle](#pontryagin极大值原理--pontryagin-maximum-principle)
  - [7.2.5 自适应控制 / Adaptive Control](#725-自适应控制--adaptive-control)
    - [模型参考自适应控制 / Model Reference Adaptive Control](#模型参考自适应控制--model-reference-adaptive-control)
    - [自校正控制 / Self-Tuning Control](#自校正控制--self-tuning-control)
    - [鲁棒控制 / Robust Control](#鲁棒控制--robust-control)
  - [7.2.6 智能控制 / Intelligent Control](#726-智能控制--intelligent-control)
    - [模糊控制 / Fuzzy Control](#模糊控制--fuzzy-control)
    - [神经网络控制 / Neural Network Control](#神经网络控制--neural-network-control)
    - [遗传算法控制 / Genetic Algorithm Control](#遗传算法控制--genetic-algorithm-control)
  - [7.2.7 实现与应用 / Implementation and Applications](#727-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [工业自动化 / Industrial Automation](#工业自动化--industrial-automation)
      - [机器人控制 / Robotics Control](#机器人控制--robotics-control)
      - [航空航天 / Aerospace](#航空航天--aerospace)
  - [参考文献 / References](#参考文献--references)

---

## 7.2.1 反馈控制系统 / Feedback Control Systems

### 开环控制 / Open-Loop Control

**系统方程**: $\dot{x} = f(x, u)$

**控制律**: $u = g(r)$

**输出**: $y = h(x)$

**误差**: $e = r - y$

### 闭环控制 / Closed-Loop Control

**反馈控制**: $u = g(r, y) = g(r, h(x))$

**比例控制**: $u(t) = K_p e(t)$

**积分控制**: $u(t) = K_i \int_0^t e(\tau) d\tau$

**微分控制**: $u(t) = K_d \frac{de(t)}{dt}$

### PID控制器 / PID Controller

**PID控制律**: $u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$

**传递函数**: $G_c(s) = K_p + \frac{K_i}{s} + K_d s$

**闭环传递函数**: $G_{cl}(s) = \frac{G_c(s)G_p(s)}{1 + G_c(s)G_p(s)}$

**稳态误差**: $e_{ss} = \lim_{s \to 0} s E(s) = \lim_{s \to 0} s \frac{R(s)}{1 + G_c(s)G_p(s)}$

---

## 7.2.2 系统稳定性分析 / System Stability Analysis

### Lyapunov稳定性 / Lyapunov Stability

**Lyapunov函数**: $V(x) > 0$ for $x \neq 0$

**稳定性条件**: $\dot{V}(x) \leq 0$

**渐近稳定性**: $\dot{V}(x) < 0$ for $x \neq 0$

**全局稳定性**: $V(x) \to \infty$ as $\|x\| \to \infty$

### Routh-Hurwitz判据 / Routh-Hurwitz Criterion

**特征方程**: $a_n s^n + a_{n-1} s^{n-1} + \ldots + a_0 = 0$

**Routh表**:
$$
\begin{array}{c|c}
s^n & a_n \quad a_{n-2} \quad a_{n-4} \ldots \\
s^{n-1} & a_{n-1} \quad a_{n-3} \quad a_{n-5} \ldots \\
s^{n-2} & b_1 \quad b_2 \quad b_3 \ldots \\
\vdots & \vdots
\end{array}
$$

**稳定性条件**: 第一列所有元素同号

### Nyquist判据 / Nyquist Criterion

**开环传递函数**: $G(s)H(s)$

**Nyquist图**: $G(j\omega)H(j\omega)$

**稳定性条件**: $N = Z - P$

其中：

- $N$: 包围(-1,0)的圈数
- $Z$: 右半平面零点数
- $P$: 右半平面极点数

---

## 7.2.3 状态空间模型 / State Space Models

### 线性系统 / Linear Systems

**状态方程**: $\dot{x} = Ax + Bu$

**输出方程**: $y = Cx + Du$

**传递函数**: $G(s) = C(sI - A)^{-1}B + D$

**特征值**: $\det(sI - A) = 0$

### 非线性系统 / Nonlinear Systems

**状态方程**: $\dot{x} = f(x, u)$

**输出方程**: $y = h(x, u)$

**线性化**: $\delta \dot{x} = \frac{\partial f}{\partial x} \delta x + \frac{\partial f}{\partial u} \delta u$

**雅可比矩阵**: $A = \frac{\partial f}{\partial x}|_{x_0, u_0}$

### 可观性和可控性 / Observability and Controllability

**可控性矩阵**: $\mathcal{C} = [B \quad AB \quad A^2B \quad \ldots \quad A^{n-1}B]$

**可观性矩阵**: $\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$

**可控性条件**: $\text{rank}(\mathcal{C}) = n$

**可观性条件**: $\text{rank}(\mathcal{O}) = n$

---

## 7.2.4 最优控制理论 / Optimal Control Theory

### 变分法 / Calculus of Variations

**性能指标**: $J = \int_{t_0}^{t_f} L(x, u, t) dt$

**Euler-Lagrange方程**: $\frac{d}{dt}\frac{\partial L}{\partial \dot{x}} - \frac{\partial L}{\partial x} = 0$

**横截条件**: $\frac{\partial L}{\partial \dot{x}}|_{t_f} = 0$

### 动态规划 / Dynamic Programming

**Bellman方程**: $V(x, t) = \min_u \{L(x, u, t) + V(f(x, u), t + \Delta t)\}$

**最优控制**: $u^*(x, t) = \arg\min_u \{L(x, u, t) + V(f(x, u), t + \Delta t)\}$

**Hamilton-Jacobi-Bellman方程**: $-\frac{\partial V}{\partial t} = \min_u \{L(x, u, t) + \frac{\partial V}{\partial x} f(x, u)\}$

### Pontryagin极大值原理 / Pontryagin Maximum Principle

**Hamilton函数**: $H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u)$

**协态方程**: $\dot{\lambda} = -\frac{\partial H}{\partial x}$

**最优控制**: $u^*(t) = \arg\max_u H(x^*(t), u, \lambda^*(t), t)$

**横截条件**: $\lambda(t_f) = \frac{\partial \phi}{\partial x}|_{t_f}$

---

## 7.2.5 自适应控制 / Adaptive Control

### 模型参考自适应控制 / Model Reference Adaptive Control

**参考模型**: $\dot{x}_m = A_m x_m + B_m r$

**被控对象**: $\dot{x}_p = A_p x_p + B_p u$

**误差**: $e = x_m - x_p$

**自适应律**: $\dot{\theta} = \gamma e^T P B_p \phi$

### 自校正控制 / Self-Tuning Control

**参数估计**: $\hat{\theta}(t) = \hat{\theta}(t-1) + K[t](y(t) - \phi^T(t)\hat{\theta}(t-1))$

**控制器设计**: $u(t) = -\hat{\theta}^T(t) \phi(t)$

**递归最小二乘**: $K(t) = P(t-1)\phi[t](\lambda + \phi^T(t)P(t-1)\phi(t))^{-1}$

### 鲁棒控制 / Robust Control

**不确定性**: $\Delta G(s) = G(s) - G_0(s)$

**H∞控制**: $\min_K \|T_{zw}\|_\infty$

**μ综合**: $\min_K \sup_{\omega} \mu(M(j\omega))$

---

## 7.2.6 智能控制 / Intelligent Control

### 模糊控制 / Fuzzy Control

**模糊化**: $\mu_A(x) = \text{membership function}$

**推理规则**: IF $x_1$ is $A_1$ AND $x_2$ is $A_2$ THEN $u$ is $B$

**去模糊化**: $u = \frac{\sum_i \mu_i u_i}{\sum_i \mu_i}$

### 神经网络控制 / Neural Network Control

**网络结构**: $y = f(\sum_{i=1}^n w_i x_i + b)$

**学习算法**: $\Delta w = \eta \delta x$

**反向传播**: $\delta_j = \sum_k w_{jk} \delta_k f'(net_j)$

### 遗传算法控制 / Genetic Algorithm Control

**染色体**: $c = [K_p, K_i, K_d]$

**适应度**: $f(c) = \frac{1}{1 + J(c)}$

**交叉**: $c_{new} = \alpha c_1 + (1-\alpha) c_2$

**变异**: $c_{new} = c + N(0, \sigma)$

---

## 7.2.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub setpoint: f64,
    pub integral: f64,
    pub previous_error: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub integral_min: f64,
    pub integral_max: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64, setpoint: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            setpoint,
            integral: 0.0,
            previous_error: 0.0,
            output_min: -100.0,
            output_max: 100.0,
            integral_min: -50.0,
            integral_max: 50.0,
        }
    }
    
    pub fn compute(&mut self, measurement: f64, dt: f64) -> f64 {
        let error = self.setpoint - measurement;
        
        // 比例项
        let proportional = self.kp * error;
        
        // 积分项
        self.integral += error * dt;
        self.integral = self.integral.max(self.integral_min).min(self.integral_max);
        let integral = self.ki * self.integral;
        
        // 微分项
        let derivative = self.kd * (error - self.previous_error) / dt;
        self.previous_error = error;
        
        // 总输出
        let output = proportional + integral + derivative;
        output.max(self.output_min).min(self.output_max)
    }
    
    pub fn set_setpoint(&mut self, setpoint: f64) {
        self.setpoint = setpoint;
    }
    
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
    }
}

#[derive(Debug)]
pub struct StateSpaceSystem {
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub c: Vec<Vec<f64>>,
    pub d: Vec<Vec<f64>>,
    pub state: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
    pub state_size: usize,
}

impl StateSpaceSystem {
    pub fn new(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>, c: Vec<Vec<f64>>, d: Vec<Vec<f64>>) -> Self {
        let state_size = a.len();
        let state = vec![0.0; state_size];
        
        Self {
            a,
            b,
            c,
            d,
            state,
            input_size: b[0].len(),
            output_size: c.len(),
            state_size,
        }
    }
    
    pub fn update(&mut self, input: &[f64], dt: f64) -> Vec<f64> {
        // 计算状态导数
        let mut state_derivative = vec![0.0; self.state_size];
        
        for i in 0..self.state_size {
            // Ax项
            for j in 0..self.state_size {
                state_derivative[i] += self.a[i][j] * self.state[j];
            }
            // Bu项
            for j in 0..self.input_size {
                state_derivative[i] += self.b[i][j] * input[j];
            }
        }
        
        // 欧拉积分
        for i in 0..self.state_size {
            self.state[i] += state_derivative[i] * dt;
        }
        
        // 计算输出
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            // Cx项
            for j in 0..self.state_size {
                output[i] += self.c[i][j] * self.state[j];
            }
            // Du项
            for j in 0..self.input_size {
                output[i] += self.d[i][j] * input[j];
            }
        }
        
        output
    }
    
    pub fn get_state(&self) -> Vec<f64> {
        self.state.clone()
    }
    
    pub fn set_state(&mut self, state: Vec<f64>) {
        if state.len() == self.state_size {
            self.state = state;
        }
    }
}

#[derive(Debug)]
pub struct LyapunovStability {
    pub system: StateSpaceSystem,
}

impl LyapunovStability {
    pub fn new(system: StateSpaceSystem) -> Self {
        Self { system }
    }
    
    pub fn check_stability(&self) -> bool {
        // 简化的稳定性检查：计算特征值
        let eigenvalues = self.compute_eigenvalues();
        
        // 检查所有特征值的实部是否小于零
        eigenvalues.iter().all(|&e| e.re < 0.0)
    }
    
    fn compute_eigenvalues(&self) -> Vec<num_complex::Complex<f64>> {
        // 简化的特征值计算
        let n = self.system.state_size;
        let mut eigenvalues = Vec::new();
        
        // 对于2x2系统，直接计算特征值
        if n == 2 {
            let a = &self.system.a;
            let trace = a[0][0] + a[1][1];
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            let discriminant = trace * trace - 4.0 * det;
            
            if discriminant >= 0.0 {
                let lambda1 = (trace + discriminant.sqrt()) / 2.0;
                let lambda2 = (trace - discriminant.sqrt()) / 2.0;
                eigenvalues.push(num_complex::Complex::new(lambda1, 0.0));
                eigenvalues.push(num_complex::Complex::new(lambda2, 0.0));
            } else {
                let real = trace / 2.0;
                let imag = (-discriminant).sqrt() / 2.0;
                eigenvalues.push(num_complex::Complex::new(real, imag));
                eigenvalues.push(num_complex::Complex::new(real, -imag));
            }
        }
        
        eigenvalues
    }
}

#[derive(Debug)]
pub struct OptimalControl {
    pub system: StateSpaceSystem,
    pub q: Vec<Vec<f64>>, // 状态权重矩阵
    pub r: Vec<Vec<f64>>, // 控制权重矩阵
}

impl OptimalControl {
    pub fn new(system: StateSpaceSystem, q: Vec<Vec<f64>>, r: Vec<Vec<f64>>) -> Self {
        Self { system, q, r }
    }
    
    pub fn lqr_control(&self, state: &[f64]) -> Vec<f64> {
        // 简化的LQR控制
        let n = self.system.state_size;
        let m = self.system.input_size;
        
        // 计算反馈增益矩阵K（简化版本）
        let mut k = vec![vec![0.0; n]; m];
        
        // 对于简单情况，使用经验公式
        for i in 0..m {
            for j in 0..n {
                k[i][j] = -1.0 * (i as f64 + j as f64 + 1.0);
            }
        }
        
        // 计算控制输入 u = -Kx
        let mut control = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                control[i] -= k[i][j] * state[j];
            }
        }
        
        control
    }
    
    pub fn compute_cost(&self, state: &[f64], control: &[f64]) -> f64 {
        let mut cost = 0.0;
        
        // 状态成本 x^T Q x
        for i in 0..self.system.state_size {
            for j in 0..self.system.state_size {
                cost += state[i] * self.q[i][j] * state[j];
            }
        }
        
        // 控制成本 u^T R u
        for i in 0..self.system.input_size {
            for j in 0..self.system.input_size {
                cost += control[i] * self.r[i][j] * control[j];
            }
        }
        
        cost
    }
}

#[derive(Debug)]
pub struct FuzzyController {
    pub rules: Vec<FuzzyRule>,
    pub membership_functions: HashMap<String, Vec<(f64, f64, f64)>>, // (a, b, c) for triangular
}

#[derive(Debug)]
pub struct FuzzyRule {
    pub antecedents: Vec<(String, String)>, // (variable, term)
    pub consequent: (String, String), // (variable, term)
}

impl FuzzyController {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            membership_functions: HashMap::new(),
        }
    }
    
    pub fn add_membership_function(&mut self, variable: String, term: String, a: f64, b: f64, c: f64) {
        let key = format!("{}_{}", variable, term);
        self.membership_functions.insert(key, vec![(a, b, c)]);
    }
    
    pub fn add_rule(&mut self, antecedents: Vec<(String, String)>, consequent: (String, String)) {
        self.rules.push(FuzzyRule { antecedents, consequent });
    }
    
    pub fn compute(&self, inputs: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut outputs = HashMap::new();
        
        for rule in &self.rules {
            let mut rule_strength = 1.0;
            
            // 计算规则强度（取最小值）
            for (variable, term) in &rule.antecedents {
                if let Some(&input_value) = inputs.get(variable) {
                    let membership = self.compute_membership(variable, term, input_value);
                    rule_strength = rule_strength.min(membership);
                }
            }
            
            // 应用规则
            let (output_var, output_term) = &rule.consequent;
            let current_output = outputs.get(output_var).unwrap_or(&0.0);
            outputs.insert(output_var.clone(), current_output + rule_strength);
        }
        
        outputs
    }
    
    fn compute_membership(&self, variable: &str, term: &str, value: f64) -> f64 {
        let key = format!("{}_{}", variable, term);
        if let Some(functions) = self.membership_functions.get(&key) {
            if let Some(&(a, b, c)) = functions.first() {
                // 三角隶属度函数
                if value <= a || value >= c {
                    return 0.0;
                } else if value <= b {
                    return (value - a) / (b - a);
                } else {
                    return (c - value) / (c - b);
                }
            }
        }
        0.0
    }
}

// 使用示例
fn main() {
    // PID控制器示例
    let mut pid = PIDController::new(1.0, 0.1, 0.01, 10.0);
    let mut measurement = 0.0;
    
    for step in 0..100 {
        let control = pid.compute(measurement, 0.01);
        measurement += control * 0.01; // 简化的系统响应
        println!("Step {}: Setpoint={}, Measurement={:.3}, Control={:.3}", 
                step, pid.setpoint, measurement, control);
    }
    
    // 状态空间系统示例
    let a = vec![vec![0.0, 1.0], vec![-1.0, -1.0]];
    let b = vec![vec![0.0], vec![1.0]];
    let c = vec![vec![1.0, 0.0]];
    let d = vec![vec![0.0]];
    
    let mut system = StateSpaceSystem::new(a, b, c, d);
    let input = vec![1.0];
    
    for step in 0..50 {
        let output = system.update(&input, 0.01);
        println!("Step {}: Output={:.3}, State=[{:.3}, {:.3}]", 
                step, output[0], system.state[0], system.state[1]);
    }
    
    // 稳定性分析
    let stability = LyapunovStability::new(system);
    let is_stable = stability.check_stability();
    println!("System is stable: {}", is_stable);
    
    // 最优控制
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let r = vec![vec![1.0]];
    let optimal_control = OptimalControl::new(system, q, r);
    
    let state = vec![1.0, 0.5];
    let control = optimal_control.lqr_control(&state);
    let cost = optimal_control.compute_cost(&state, &control);
    
    println!("Optimal control: {:?}, Cost: {:.3}", control, cost);
    
    // 模糊控制
    let mut fuzzy = FuzzyController::new();
    fuzzy.add_membership_function("error".to_string(), "negative".to_string(), -10.0, -5.0, 0.0);
    fuzzy.add_membership_function("error".to_string(), "zero".to_string(), -5.0, 0.0, 5.0);
    fuzzy.add_membership_function("error".to_string(), "positive".to_string(), 0.0, 5.0, 10.0);
    
    fuzzy.add_rule(vec![("error".to_string(), "negative".to_string())], 
                   ("control".to_string(), "positive".to_string()));
    fuzzy.add_rule(vec![("error".to_string(), "positive".to_string())], 
                   ("control".to_string(), "negative".to_string()));
    
    let mut inputs = HashMap::new();
    inputs.insert("error".to_string(), 3.0);
    let outputs = fuzzy.compute(&inputs);
    
    println!("Fuzzy control output: {:?}", outputs);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module CyberneticsModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, length, filter)

-- PID控制器
data PIDController = PIDController {
    kp :: Double,
    ki :: Double,
    kd :: Double,
    setpoint :: Double,
    integral :: Double,
    previousError :: Double,
    outputMin :: Double,
    outputMax :: Double,
    integralMin :: Double,
    integralMax :: Double
} deriving Show

newPIDController :: Double -> Double -> Double -> Double -> PIDController
newPIDController kp_val ki_val kd_val setpoint_val = PIDController {
    kp = kp_val,
    ki = ki_val,
    kd = kd_val,
    setpoint = setpoint_val,
    integral = 0.0,
    previousError = 0.0,
    outputMin = -100.0,
    outputMax = 100.0,
    integralMin = -50.0,
    integralMax = 50.0
}

computePID :: PIDController -> Double -> Double -> PIDController
computePID pid measurement dt = 
    let error = setpoint pid - measurement
        
        -- 比例项
        proportional = kp pid * error
        
        -- 积分项
        newIntegral = max (integralMin pid) (min (integralMax pid) (integral pid + error * dt))
        integralTerm = ki pid * newIntegral
        
        -- 微分项
        derivative = kd pid * (error - previousError pid) / dt
        
        -- 总输出
        output = max (outputMin pid) (min (outputMax pid) (proportional + integralTerm + derivative))
    in pid {
        integral = newIntegral,
        previousError = error
    }

getOutput :: PIDController -> Double -> Double -> Double
getOutput pid measurement dt = 
    let error = setpoint pid - measurement
        proportional = kp pid * error
        integralTerm = ki pid * integral pid
        derivative = kd pid * (error - previousError pid) / dt
    in max (outputMin pid) (min (outputMax pid) (proportional + integralTerm + derivative))

-- 状态空间系统
data StateSpaceSystem = StateSpaceSystem {
    a :: [[Double]],
    b :: [[Double]],
    c :: [[Double]],
    d :: [[Double]],
    state :: [Double],
    inputSize :: Int,
    outputSize :: Int,
    stateSize :: Int
} deriving Show

newStateSpaceSystem :: [[Double]] -> [[Double]] -> [[Double]] -> [[Double]] -> StateSpaceSystem
newStateSpaceSystem a_matrix b_matrix c_matrix d_matrix = 
    let state_size = length a_matrix
        state = replicate state_size 0.0
    in StateSpaceSystem {
        a = a_matrix,
        b = b_matrix,
        c = c_matrix,
        d = d_matrix,
        state = state,
        inputSize = length (head b_matrix),
        outputSize = length c_matrix,
        stateSize = state_size
    }

updateSystem :: StateSpaceSystem -> [Double] -> Double -> StateSpaceSystem
updateSystem system input dt = 
    let -- 计算状态导数
        stateDerivative = [sum [a system !! i !! j * (state system !! j) | j <- [0..stateSize system-1]] +
                          sum [b system !! i !! j * (input !! j) | j <- [0..inputSize system-1]] | 
                          i <- [0..stateSize system-1]]
        
        -- 欧拉积分
        newState = [state system !! i + stateDerivative !! i * dt | i <- [0..stateSize system-1]]
    in system { state = newState }

computeOutput :: StateSpaceSystem -> [Double] -> [Double]
computeOutput system input = 
    [sum [c system !! i !! j * (state system !! j) | j <- [0..stateSize system-1]] +
     sum [d system !! i !! j * (input !! j) | j <- [0..inputSize system-1]] | 
     i <- [0..outputSize system-1]]

-- Lyapunov稳定性
data LyapunovStability = LyapunovStability {
    system :: StateSpaceSystem
} deriving Show

newLyapunovStability :: StateSpaceSystem -> LyapunovStability
newLyapunovStability sys = LyapunovStability sys

checkStability :: LyapunovStability -> Bool
checkStability stability = 
    let eigenvalues = computeEigenvalues (system stability)
    in all (\e -> realPart e < 0.0) eigenvalues

computeEigenvalues :: StateSpaceSystem -> [Complex Double]
computeEigenvalues system = 
    let a_matrix = a system
        n = stateSize system
    in if n == 2 
       then let a11 = a_matrix !! 0 !! 0
                a12 = a_matrix !! 0 !! 1
                a21 = a_matrix !! 1 !! 0
                a22 = a_matrix !! 1 !! 1
                trace = a11 + a22
                det = a11 * a22 - a12 * a21
                discriminant = trace^2 - 4 * det
            in if discriminant >= 0 
               then let lambda1 = (trace + sqrt discriminant) / 2
                        lambda2 = (trace - sqrt discriminant) / 2
                    in [lambda1 :+ 0.0, lambda2 :+ 0.0]
               else let real = trace / 2
                        imag = sqrt (-discriminant) / 2
                    in [real :+ imag, real :+ (-imag)]
       else []

-- 最优控制
data OptimalControl = OptimalControl {
    controlSystem :: StateSpaceSystem,
    q :: [[Double]], -- 状态权重矩阵
    r :: [[Double]]  -- 控制权重矩阵
} deriving Show

newOptimalControl :: StateSpaceSystem -> [[Double]] -> [[Double]] -> OptimalControl
newOptimalControl sys q_matrix r_matrix = OptimalControl {
    controlSystem = sys,
    q = q_matrix,
    r = r_matrix
}

lqrControl :: OptimalControl -> [Double] -> [Double]
lqrControl control state = 
    let n = stateSize (controlSystem control)
        m = inputSize (controlSystem control)
        -- 简化的反馈增益矩阵
        k = [[-1.0 * fromIntegral (i + j + 1) | j <- [0..n-1]] | i <- [0..m-1]]
    in [sum [-k !! i !! j * (state !! j) | j <- [0..n-1]] | i <- [0..m-1]]

computeCost :: OptimalControl -> [Double] -> [Double] -> Double
computeCost control state control_input = 
    let -- 状态成本 x^T Q x
        stateCost = sum [state !! i * (q control !! i !! j) * (state !! j) | 
                        i <- [0..length state-1], j <- [0..length state-1]]
        
        -- 控制成本 u^T R u
        controlCost = sum [control_input !! i * (r control !! i !! j) * (control_input !! j) | 
                          i <- [0..length control_input-1], j <- [0..length control_input-1]]
    in stateCost + controlCost

-- 模糊控制
data FuzzyController = FuzzyController {
    rules :: [FuzzyRule],
    membershipFunctions :: Map String [(Double, Double, Double)] -- (a, b, c) for triangular
} deriving Show

data FuzzyRule = FuzzyRule {
    antecedents :: [(String, String)], -- (variable, term)
    consequent :: (String, String)     -- (variable, term)
} deriving Show

newFuzzyController :: FuzzyController
newFuzzyController = FuzzyController {
    rules = [],
    membershipFunctions = Map.empty
}

addMembershipFunction :: String -> String -> Double -> Double -> Double -> FuzzyController -> FuzzyController
addMembershipFunction variable term a b c controller = 
    let key = variable ++ "_" ++ term
        newFunctions = Map.insert key [(a, b, c)] (membershipFunctions controller)
    in controller { membershipFunctions = newFunctions }

addRule :: [(String, String)] -> (String, String) -> FuzzyController -> FuzzyController
addRule antecedents consequent controller = 
    let newRule = FuzzyRule antecedents consequent
    in controller { rules = newRule : rules controller }

computeFuzzy :: FuzzyController -> Map String Double -> Map String Double
computeFuzzy controller inputs = 
    let ruleOutputs = map (\rule -> computeRule controller rule inputs) (rules controller)
        -- 合并规则输出
        combineOutputs outputs = 
            foldl (\acc (var, value) -> 
                Map.insertWith (+) var value acc) Map.empty outputs
    in combineOutputs ruleOutputs

computeRule :: FuzzyController -> FuzzyRule -> Map String Double -> (String, Double)
computeRule controller rule inputs = 
    let -- 计算规则强度
        ruleStrength = minimum [computeMembership controller variable term (inputs Map.! variable) | 
                               (variable, term) <- antecedents rule]
        (outputVar, outputTerm) = consequent rule
    in (outputVar, ruleStrength)

computeMembership :: FuzzyController -> String -> String -> Double -> Double
computeMembership controller variable term value = 
    let key = variable ++ "_" ++ term
    in case Map.lookup key (membershipFunctions controller) of
        Just [(a, b, c)] -> 
            if value <= a || value >= c 
            then 0.0
            else if value <= b 
                 then (value - a) / (b - a)
                 else (c - value) / (c - b)
        _ -> 0.0

-- 示例使用
example :: IO ()
example = do
    -- PID控制器示例
    let pid = newPIDController 1.0 0.1 0.01 10.0
        simulatePID 0 pid measurement = return ()
        simulatePID steps pid measurement = do
            let control = getOutput pid measurement 0.01
                newMeasurement = measurement + control * 0.01
                newPID = computePID pid measurement 0.01
            putStrLn $ "Step " ++ show (100 - steps) ++ ": Measurement=" ++ show newMeasurement ++ ", Control=" ++ show control
            simulatePID (steps - 1) newPID newMeasurement
    
    simulatePID 100 pid 0.0
    
    -- 状态空间系统示例
    let a_matrix = [[0.0, 1.0], [-1.0, -1.0]]
        b_matrix = [[0.0], [1.0]]
        c_matrix = [[1.0, 0.0]]
        d_matrix = [[0.0]]
        system = newStateSpaceSystem a_matrix b_matrix c_matrix d_matrix
        input = [1.0]
        
        simulateSystem 0 sys = return ()
        simulateSystem steps sys = do
            let newSys = updateSystem sys input 0.01
                output = computeOutput newSys input
            putStrLn $ "Step " ++ show (50 - steps) ++ ": Output=" ++ show (head output) ++ 
                      ", State=[" ++ show (state newSys !! 0) ++ ", " ++ show (state newSys !! 1) ++ "]"
            simulateSystem (steps - 1) newSys
    
    simulateSystem 50 system
    
    -- 稳定性分析
    let stability = newLyapunovStability system
        isStable = checkStability stability
    putStrLn $ "System is stable: " ++ show isStable
    
    -- 最优控制
    let q_matrix = [[1.0, 0.0], [0.0, 1.0]]
        r_matrix = [[1.0]]
        optimalControl = newOptimalControl system q_matrix r_matrix
        state = [1.0, 0.5]
        control = lqrControl optimalControl state
        cost = computeCost optimalControl state control
    
    putStrLn $ "Optimal control: " ++ show control
    putStrLn $ "Cost: " ++ show cost
```

### 应用领域 / Application Domains

#### 工业自动化 / Industrial Automation

- **过程控制**: 化工、电力、制造过程控制
- **运动控制**: 机器人、机床、传送带控制
- **质量控制**: 自动检测和调整系统

#### 机器人控制 / Robotics Control

- **轨迹规划**: 路径规划和轨迹跟踪
- **力控制**: 力反馈和阻抗控制
- **多机器人协调**: 群体机器人控制

#### 航空航天 / Aerospace

- **飞行控制**: 飞机姿态和轨迹控制
- **卫星控制**: 轨道保持和姿态控制
- **导弹制导**: 精确制导和导航

---

## 参考文献 / References

1. Ogata, K. (2010). Modern Control Engineering. Prentice Hall.
2. Astrom, K. J., & Murray, R. M. (2008). Feedback Systems. Princeton University Press.
3. Slotine, J. J. E., & Li, W. (1991). Applied Nonlinear Control. Prentice Hall.
4. Lewis, F. L., et al. (2012). Optimal Control. Wiley.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
