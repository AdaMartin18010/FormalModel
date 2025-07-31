# 7.2 控制论模型 / Control Theory Models

## 目录 / Table of Contents

- [7.2 控制论模型 / Control Theory Models](#72-控制论模型--control-theory-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.2.1 线性系统模型 / Linear System Models](#721-线性系统模型--linear-system-models)
    - [传递函数模型 / Transfer Function Model](#传递函数模型--transfer-function-model)
    - [频率响应模型 / Frequency Response Model](#频率响应模型--frequency-response-model)
    - [稳定性分析 / Stability Analysis](#稳定性分析--stability-analysis)
  - [7.2.2 PID控制模型 / PID Control Models](#722-pid控制模型--pid-control-models)
    - [连续PID控制器 / Continuous PID Controller](#连续pid控制器--continuous-pid-controller)
    - [离散PID控制器 / Discrete PID Controller](#离散pid控制器--discrete-pid-controller)
    - [参数整定方法 / Tuning Methods](#参数整定方法--tuning-methods)
  - [7.2.3 状态空间模型 / State Space Models](#723-状态空间模型--state-space-models)
    - [连续时间状态空间 / Continuous State Space](#连续时间状态空间--continuous-state-space)
    - [离散时间状态空间 / Discrete State Space](#离散时间状态空间--discrete-state-space)
    - [可控性和可观性 / Controllability and Observability](#可控性和可观性--controllability-and-observability)
  - [7.2.4 鲁棒控制模型 / Robust Control Models](#724-鲁棒控制模型--robust-control-models)
    - [H∞控制 / H∞ Control](#h控制--h-control)
    - [μ综合 / μ-Synthesis](#μ综合--μ-synthesis)
    - [滑模控制 / Sliding Mode Control](#滑模控制--sliding-mode-control)
  - [7.2.5 自适应控制模型 / Adaptive Control Models](#725-自适应控制模型--adaptive-control-models)
    - [模型参考自适应控制 / Model Reference Adaptive Control](#模型参考自适应控制--model-reference-adaptive-control)
    - [自校正控制 / Self-Tuning Control](#自校正控制--self-tuning-control)
    - [神经网络控制 / Neural Network Control](#神经网络控制--neural-network-control)
  - [7.2.6 模糊控制模型 / Fuzzy Control Models](#726-模糊控制模型--fuzzy-control-models)
    - [模糊推理 / Fuzzy Inference](#模糊推理--fuzzy-inference)
    - [模糊PID控制器 / Fuzzy PID Controller](#模糊pid控制器--fuzzy-pid-controller)
    - [自适应模糊控制 / Adaptive Fuzzy Control](#自适应模糊控制--adaptive-fuzzy-control)
  - [7.2.7 实现与应用 / Implementation and Applications](#727-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [工业控制 / Industrial Control](#工业控制--industrial-control)
      - [航空航天 / Aerospace](#航空航天--aerospace)
      - [汽车控制 / Automotive Control](#汽车控制--automotive-control)
  - [参考文献 / References](#参考文献--references)

---

## 7.2.1 线性系统模型 / Linear System Models

### 传递函数模型 / Transfer Function Model

**连续时间**: $G(s) = \frac{Y(s)}{U(s)} = \frac{b_ms^m + b_{m-1}s^{m-1} + \cdots + b_0}{a_ns^n + a_{n-1}s^{n-1} + \cdots + a_0}$

**离散时间**: $G(z) = \frac{Y(z)}{U(z)} = \frac{b_mz^m + b_{m-1}z^{m-1} + \cdots + b_0}{a_nz^n + a_{n-1}z^{n-1} + \cdots + a_0}$

**零极点形式**: $G(s) = K \frac{\prod_{i=1}^m (s - z_i)}{\prod_{i=1}^n (s - p_i)}$

### 频率响应模型 / Frequency Response Model

**幅频特性**: $|G(j\omega)| = \sqrt{\text{Re}^2[G(j\omega)] + \text{Im}^2[G(j\omega)]}$

**相频特性**: $\angle G(j\omega) = \tan^{-1}\left(\frac{\text{Im}[G(j\omega)]}{\text{Re}[G(j\omega)]}\right)$

**伯德图**: $20\log_{10}|G(j\omega)|$ vs $\log_{10}\omega$

### 稳定性分析 / Stability Analysis

**劳斯判据**: 特征方程 $a_ns^n + a_{n-1}s^{n-1} + \cdots + a_0 = 0$

**赫尔维茨判据**: 赫尔维茨矩阵的所有主子式为正

**奈奎斯特判据**: $N = Z - P$，其中 $N$ 为包围次数

---

## 7.2.2 PID控制模型 / PID Control Models

### 连续PID控制器 / Continuous PID Controller

**控制律**: $u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$

**传递函数**: $G_c(s) = K_p + \frac{K_i}{s} + K_d s$

**理想形式**: $G_c(s) = K_p \left(1 + \frac{1}{T_i s} + T_d s\right)$

### 离散PID控制器 / Discrete PID Controller

**位置式**: $u(k) = K_p e(k) + K_i \sum_{i=0}^k e(i) + K_d [e(k) - e(k-1)]$

**增量式**: $\Delta u(k) = K_p [e(k) - e(k-1)] + K_i e(k) + K_d [e(k) - 2e(k-1) + e(k-2)]$

**积分抗饱和**: $u(k) = K_p e(k) + K_i \sum_{i=0}^k e(i) + K_d [e(k) - e(k-1)] + \text{anti-windup}$

### 参数整定方法 / Tuning Methods

**Ziegler-Nichols方法**:

- $K_p = 0.6K_u$
- $T_i = 0.5T_u$
- $T_d = 0.125T_u$

**Cohen-Coon方法**:

- $K_p = \frac{1}{K} \frac{\tau + 0.5\theta}{\theta}$
- $T_i = \theta \frac{9 + 20\tau/\theta}{7 + 20\tau/\theta}$
- $T_d = \theta \frac{2\tau}{11 + 2\tau/\theta}$

---

## 7.2.3 状态空间模型 / State Space Models

### 连续时间状态空间 / Continuous State Space

**状态方程**: $\dot{x}(t) = Ax(t) + Bu(t)$

**输出方程**: $y(t) = Cx(t) + Du(t)$

**传递函数**: $G(s) = C(sI - A)^{-1}B + D$

### 离散时间状态空间 / Discrete State Space

**状态方程**: $x(k+1) = \Phi x(k) + \Gamma u(k)$

**输出方程**: $y(k) = Cx(k) + Du(k)$

**传递函数**: $G(z) = C(zI - \Phi)^{-1}\Gamma + D$

### 可控性和可观性 / Controllability and Observability

**可控性矩阵**: $C = [B \quad AB \quad A^2B \quad \cdots \quad A^{n-1}B]$

**可观性矩阵**: $O = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$

**可控性判据**: $\text{rank}(C) = n$

**可观性判据**: $\text{rank}(O) = n$

---

## 7.2.4 鲁棒控制模型 / Robust Control Models

### H∞控制 / H∞ Control

**性能指标**: $\|T_{zw}\|_\infty < \gamma$

**广义对象**: $P(s) = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix}$

**闭环传递函数**: $T_{zw} = P_{11} + P_{12}K(I - P_{22}K)^{-1}P_{21}$

### μ综合 / μ-Synthesis

**结构奇异值**: $\mu(M) = \frac{1}{\min\{\bar{\sigma}(\Delta) : \det(I - M\Delta) = 0\}}$

**鲁棒稳定性**: $\mu(M_{11}) < 1$

**鲁棒性能**: $\mu(M) < 1$

### 滑模控制 / Sliding Mode Control

**滑模面**: $s(x) = c^T x = 0$

**控制律**: $u = u_{eq} + u_{sw}$

**等效控制**: $u_{eq} = -(c^T B)^{-1}c^T Ax$

**切换控制**: $u_{sw} = -\eta \text{sign}(s)$

---

## 7.2.5 自适应控制模型 / Adaptive Control Models

### 模型参考自适应控制 / Model Reference Adaptive Control

**参考模型**: $\dot{x}_m = A_m x_m + B_m r$

**控制律**: $u = \theta^T \phi$

**参数更新**: $\dot{\theta} = -\gamma e^T PB\phi$

**李雅普诺夫函数**: $V = e^T Pe + \frac{1}{\gamma} \tilde{\theta}^T \tilde{\theta}$

### 自校正控制 / Self-Tuning Control

**参数估计**: $\hat{\theta}(k) = \hat{\theta}(k-1) + K[k](y(k) - \phi^T(k)\hat{\theta}(k-1))$

**卡尔曼增益**: $K(k) = P(k-1)\phi[k](\lambda + \phi^T(k)P(k-1)\phi(k))^{-1}$

**协方差更新**: $P(k) = \frac{1}{\lambda}[I - K(k)\phi^T(k)]P(k-1)$

### 神经网络控制 / Neural Network Control

**网络输出**: $u = W^T \sigma(V^T x)$

**权重更新**: $\dot{W} = -\eta e \sigma(V^T x)$

**隐层更新**: $\dot{V} = -\eta e W \sigma'(V^T x) x^T$

---

## 7.2.6 模糊控制模型 / Fuzzy Control Models

### 模糊推理 / Fuzzy Inference

**模糊化**: $\mu_A(x) = f(x)$

**规则库**: IF $x_1$ is $A_1$ AND $x_2$ is $A_2$ THEN $y$ is $B$

**推理**: $\mu_B'(y) = \sup_{x} \min[\mu_{A_1}(x_1), \mu_{A_2}(x_2), \mu_{A_1'}(x_1), \mu_{A_2'}(x_2)]$

**去模糊化**: $y = \frac{\sum_i y_i \mu_B(y_i)}{\sum_i \mu_B(y_i)}$

### 模糊PID控制器 / Fuzzy PID Controller

**误差模糊化**: $E = \text{fuzzify}(e)$

**误差变化率**: $\Delta E = \text{fuzzify}(\dot{e})$

**输出模糊化**: $U = \text{fuzzy\_inference}(E, \Delta E)$

**去模糊化**: $u = \text{defuzzify}(U)$

### 自适应模糊控制 / Adaptive Fuzzy Control

**参数更新**: $\dot{\theta} = -\gamma e^T PB \xi$

**逼近误差**: $\epsilon = f(x) - \hat{f}(x|\theta)$

**稳定性**: $\dot{V} = -e^T Qe + \epsilon^T e$

---

## 7.2.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct LinearSystem {
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub c: Vec<Vec<f64>>,
    pub d: Vec<Vec<f64>>,
    pub state: Vec<f64>,
}

impl LinearSystem {
    pub fn new(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>, c: Vec<Vec<f64>>, d: Vec<Vec<f64>>) -> Self {
        let state_dim = a.len();
        let state = vec![0.0; state_dim];
        
        Self { a, b, c, d, state }
    }
    
    pub fn update(&mut self, u: &[f64], dt: f64) -> Vec<f64> {
        let n = self.state.len();
        let mut new_state = vec![0.0; n];
        
        // 状态更新: x(k+1) = Ax(k) + Bu(k)
        for i in 0..n {
            new_state[i] = self.state[i];
            for j in 0..n {
                new_state[i] += self.a[i][j] * self.state[j] * dt;
            }
            for j in 0..u.len() {
                new_state[i] += self.b[i][j] * u[j] * dt;
            }
        }
        
        self.state = new_state;
        
        // 输出计算: y(k) = Cx(k) + Du(k)
        let output_dim = self.c.len();
        let mut y = vec![0.0; output_dim];
        
        for i in 0..output_dim {
            for j in 0..n {
                y[i] += self.c[i][j] * self.state[j];
            }
            for j in 0..u.len() {
                y[i] += self.d[i][j] * u[j];
            }
        }
        
        y
    }
    
    pub fn get_state(&self) -> &[f64] {
        &self.state
    }
    
    pub fn set_state(&mut self, state: Vec<f64>) {
        self.state = state;
    }
}

#[derive(Debug)]
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub setpoint: f64,
    pub integral: f64,
    pub prev_error: f64,
    pub output_min: f64,
    pub output_max: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            setpoint: 0.0,
            integral: 0.0,
            prev_error: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
        }
    }
    
    pub fn set_setpoint(&mut self, setpoint: f64) {
        self.setpoint = setpoint;
    }
    
    pub fn set_limits(&mut self, min: f64, max: f64) {
        self.output_min = min;
        self.output_max = max;
    }
    
    pub fn compute(&mut self, measurement: f64, dt: f64) -> f64 {
        let error = self.setpoint - measurement;
        
        // 比例项
        let proportional = self.kp * error;
        
        // 积分项
        self.integral += error * dt;
        let integral = self.ki * self.integral;
        
        // 微分项
        let derivative = self.kd * (error - self.prev_error) / dt;
        
        // 计算输出
        let mut output = proportional + integral + derivative;
        
        // 输出限幅
        output = output.max(self.output_min).min(self.output_max);
        
        // 积分抗饱和
        if output == self.output_min || output == self.output_max {
            self.integral -= error * dt;
        }
        
        self.prev_error = error;
        output
    }
    
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

#[derive(Debug)]
pub struct FuzzyController {
    pub rules: Vec<FuzzyRule>,
    pub input_mfs: Vec<Vec<MembershipFunction>>,
    pub output_mfs: Vec<MembershipFunction>,
}

#[derive(Debug, Clone)]
pub struct FuzzyRule {
    pub antecedents: Vec<(usize, usize)>, // (input_index, mf_index)
    pub consequent: usize,
}

#[derive(Debug, Clone)]
pub struct MembershipFunction {
    pub name: String,
    pub mf_type: String,
    pub params: Vec<f64>,
}

impl MembershipFunction {
    pub fn new(name: String, mf_type: String, params: Vec<f64>) -> Self {
        Self { name, mf_type, params }
    }
    
    pub fn evaluate(&self, x: f64) -> f64 {
        match self.mf_type.as_str() {
            "trapmf" => self.trapmf(x),
            "trimf" => self.trimf(x),
            "gaussmf" => self.gaussmf(x),
            _ => 0.0,
        }
    }
    
    fn trapmf(&self, x: f64) -> f64 {
        let a = self.params[0];
        let b = self.params[1];
        let c = self.params[2];
        let d = self.params[3];
        
        if x <= a || x >= d {
            0.0
        } else if x >= b && x <= c {
            1.0
        } else if x > a && x < b {
            (x - a) / (b - a)
        } else {
            (d - x) / (d - c)
        }
    }
    
    fn trimf(&self, x: f64) -> f64 {
        let a = self.params[0];
        let b = self.params[1];
        let c = self.params[2];
        
        if x <= a || x >= c {
            0.0
        } else if x == b {
            1.0
        } else if x > a && x < b {
            (x - a) / (b - a)
        } else {
            (c - x) / (c - b)
        }
    }
    
    fn gaussmf(&self, x: f64) -> f64 {
        let sigma = self.params[0];
        let c = self.params[1];
        (-0.5 * ((x - c) / sigma).powi(2)).exp()
    }
}

impl FuzzyController {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            input_mfs: Vec::new(),
            output_mfs: Vec::new(),
        }
    }
    
    pub fn add_input_mf(&mut self, input_index: usize, mf: MembershipFunction) {
        while self.input_mfs.len() <= input_index {
            self.input_mfs.push(Vec::new());
        }
        self.input_mfs[input_index].push(mf);
    }
    
    pub fn add_output_mf(&mut self, mf: MembershipFunction) {
        self.output_mfs.push(mf);
    }
    
    pub fn add_rule(&mut self, rule: FuzzyRule) {
        self.rules.push(rule);
    }
    
    pub fn compute(&self, inputs: &[f64]) -> f64 {
        let mut rule_outputs = Vec::new();
        let mut rule_weights = Vec::new();
        
        for rule in &self.rules {
            // 计算规则强度
            let mut strength = 1.0;
            for (input_idx, mf_idx) in &rule.antecedents {
                let membership = self.input_mfs[*input_idx][*mf_idx].evaluate(inputs[*input_idx]);
                strength = strength.min(membership);
            }
            
            rule_weights.push(strength);
            
            // 计算规则输出（简化：使用重心法）
            let output_center = 0.0; // 简化，实际应该根据输出MF计算
            rule_outputs.push(output_center);
        }
        
        // 去模糊化（加权平均）
        let total_weight: f64 = rule_weights.iter().sum();
        if total_weight > 0.0 {
            let weighted_sum: f64 = rule_outputs.iter()
                .zip(rule_weights.iter())
                .map(|(output, weight)| output * weight)
                .sum();
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct RobustController {
    pub nominal_controller: PIDController,
    pub uncertainty_bound: f64,
    pub adaptive_gain: f64,
}

impl RobustController {
    pub fn new(kp: f64, ki: f64, kd: f64, uncertainty_bound: f64) -> Self {
        Self {
            nominal_controller: PIDController::new(kp, ki, kd),
            uncertainty_bound,
            adaptive_gain: 1.0,
        }
    }
    
    pub fn compute_robust(&mut self, measurement: f64, dt: f64) -> f64 {
        let nominal_output = self.nominal_controller.compute(measurement, dt);
        
        // 简化的鲁棒补偿项
        let error = self.nominal_controller.setpoint - measurement;
        let robust_term = self.adaptive_gain * self.uncertainty_bound * error.signum();
        
        nominal_output + robust_term
    }
    
    pub fn update_adaptive_gain(&mut self, tracking_error: f64) {
        // 简化的自适应律
        self.adaptive_gain += 0.01 * tracking_error.abs();
        self.adaptive_gain = self.adaptive_gain.max(0.1).min(10.0);
    }
}

// 使用示例
fn main() {
    // 创建二阶系统
    let a = vec![
        vec![0.0, 1.0],
        vec![-1.0, -0.5],
    ];
    let b = vec![
        vec![0.0],
        vec![1.0],
    ];
    let c = vec![
        vec![1.0, 0.0],
    ];
    let d = vec![
        vec![0.0],
    ];
    
    let mut system = LinearSystem::new(a, b, c, d);
    
    // 创建PID控制器
    let mut controller = PIDController::new(1.0, 0.5, 0.1);
    controller.set_setpoint(1.0);
    controller.set_limits(-10.0, 10.0);
    
    // 仿真
    let dt = 0.01;
    let simulation_time = 10.0;
    let steps = (simulation_time / dt) as usize;
    
    println!("Time\tSetpoint\tOutput\tControl");
    
    for i in 0..steps {
        let time = i as f64 * dt;
        let output = system.update(&[0.0], dt)[0];
        let control = controller.compute(output, dt);
        let _ = system.update(&[control], dt);
        
        if i % 100 == 0 {
            println!("{:.2}\t{:.3}\t{:.3}\t{:.3}", 
                    time, controller.setpoint, output, control);
        }
    }
    
    // 模糊控制示例
    let mut fuzzy_controller = FuzzyController::new();
    
    // 添加输入隶属度函数
    fuzzy_controller.add_input_mf(0, MembershipFunction::new(
        "negative".to_string(), "trimf".to_string(), vec![-1.0, -0.5, 0.0]
    ));
    fuzzy_controller.add_input_mf(0, MembershipFunction::new(
        "zero".to_string(), "trimf".to_string(), vec![-0.5, 0.0, 0.5]
    ));
    fuzzy_controller.add_input_mf(0, MembershipFunction::new(
        "positive".to_string(), "trimf".to_string(), vec![0.0, 0.5, 1.0]
    ));
    
    // 添加规则
    fuzzy_controller.add_rule(FuzzyRule {
        antecedents: vec![(0, 0)], // negative
        consequent: 0,
    });
    fuzzy_controller.add_rule(FuzzyRule {
        antecedents: vec![(0, 1)], // zero
        consequent: 1,
    });
    fuzzy_controller.add_rule(FuzzyRule {
        antecedents: vec![(0, 2)], // positive
        consequent: 2,
    });
    
    let input = vec![0.3];
    let fuzzy_output = fuzzy_controller.compute(&input);
    println!("Fuzzy control output: {:.3}", fuzzy_output);
    
    // 鲁棒控制示例
    let mut robust_controller = RobustController::new(1.0, 0.5, 0.1, 0.1);
    robust_controller.nominal_controller.set_setpoint(1.0);
    
    let robust_output = robust_controller.compute_robust(0.5, 0.01);
    println!("Robust control output: {:.3}", robust_output);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module ControlTheoryModels where

import Data.List (transpose)
import Data.Vector (Vector, (!), fromList, toList)
import qualified Data.Vector as V

-- 线性系统
data LinearSystem = LinearSystem {
    a :: [[Double]],
    b :: [[Double]],
    c :: [[Double]],
    d :: [[Double]],
    state :: [Double]
} deriving Show

newLinearSystem :: [[Double]] -> [[Double]] -> [[Double]] -> [[Double]] -> LinearSystem
newLinearSystem a_matrix b_matrix c_matrix d_matrix = LinearSystem {
    a = a_matrix,
    b = b_matrix,
    c = c_matrix,
    d = d_matrix,
    state = replicate (length a_matrix) 0.0
}

updateSystem :: LinearSystem -> [Double] -> Double -> (LinearSystem, [Double])
updateSystem system inputs dt = 
    let n = length (state system)
        new_state = zipWith (\i _ -> 
            let state_contribution = sum [a system !! i !! j * state system !! j | j <- [0..n-1]]
                input_contribution = sum [b system !! i !! j * inputs !! j | j <- [0..length inputs-1]]
            in state system !! i + (state_contribution + input_contribution) * dt) 
            [0..n-1] (state system)
        
        updated_system = system { state = new_state }
        
        outputs = [sum [c system !! i !! j * new_state !! j | j <- [0..n-1]] +
                   sum [d system !! i !! j * inputs !! j | j <- [0..length inputs-1]]
                  | i <- [0..length (c system) - 1]]
    in (updated_system, outputs)

-- PID控制器
data PIDController = PIDController {
    kp :: Double,
    ki :: Double,
    kd :: Double,
    setpoint :: Double,
    integral :: Double,
    prevError :: Double,
    outputMin :: Double,
    outputMax :: Double
} deriving Show

newPIDController :: Double -> Double -> Double -> PIDController
newPIDController kp_val ki_val kd_val = PIDController {
    kp = kp_val,
    ki = ki_val,
    kd = kd_val,
    setpoint = 0.0,
    integral = 0.0,
    prevError = 0.0,
    outputMin = negate infinity,
    outputMax = infinity
}
  where
    infinity = 1e10

setSetpoint :: Double -> PIDController -> PIDController
setSetpoint sp controller = controller { setpoint = sp }

setLimits :: Double -> Double -> PIDController -> PIDController
setLimits min_val max_val controller = controller {
    outputMin = min_val,
    outputMax = max_val
}

computePID :: PIDController -> Double -> Double -> (PIDController, Double)
computePID controller measurement dt = 
    let error = setpoint controller - measurement
        proportional = kp controller * error
        new_integral = integral controller + error * dt
        integral_term = ki controller * new_integral
        derivative = kd controller * (error - prevError controller) / dt
        raw_output = proportional + integral_term + derivative
        output = max (outputMin controller) (min (outputMax controller) raw_output)
        
        -- 积分抗饱和
        final_integral = if output == outputMin controller || output == outputMax controller
                         then integral controller
                         else new_integral
    in (controller { 
            integral = final_integral,
            prevError = error
        }, output)

resetPID :: PIDController -> PIDController
resetPID controller = controller {
    integral = 0.0,
    prevError = 0.0
}

-- 模糊控制器
data MembershipFunction = MembershipFunction {
    mfName :: String,
    mfType :: String,
    mfParams :: [Double]
} deriving Show

data FuzzyRule = FuzzyRule {
    antecedents :: [(Int, Int)], -- (input_index, mf_index)
    consequent :: Int
} deriving Show

data FuzzyController = FuzzyController {
    rules :: [FuzzyRule],
    inputMFs :: [[MembershipFunction]],
    outputMFs :: [MembershipFunction]
} deriving Show

newFuzzyController :: FuzzyController
newFuzzyController = FuzzyController [] [] []

addInputMF :: Int -> MembershipFunction -> FuzzyController -> FuzzyController
addInputMF inputIndex mf controller = 
    let current_inputs = inputMFs controller
        new_inputs = take inputIndex current_inputs ++ 
                     [mf : (if inputIndex < length current_inputs 
                            then current_inputs !! inputIndex 
                            else [])] ++
                     drop (inputIndex + 1) current_inputs
    in controller { inputMFs = new_inputs }

addOutputMF :: MembershipFunction -> FuzzyController -> FuzzyController
addOutputMF mf controller = controller {
    outputMFs = mf : outputMFs controller
}

addRule :: FuzzyRule -> FuzzyController -> FuzzyController
addRule rule controller = controller {
    rules = rule : rules controller
}

evaluateMF :: MembershipFunction -> Double -> Double
evaluateMF mf x = case mfType mf of
    "trimf" -> trimf x (mfParams mf)
    "trapmf" -> trapmf x (mfParams mf)
    "gaussmf" -> gaussmf x (mfParams mf)
    _ -> 0.0

trimf :: Double -> [Double] -> Double
trimf x params = 
    let a = params !! 0
        b = params !! 1
        c = params !! 2
    in if x <= a || x >= c
       then 0.0
       else if x == b
            then 1.0
            else if x > a && x < b
                 then (x - a) / (b - a)
                 else (c - x) / (c - b)

trapmf :: Double -> [Double] -> Double
trapmf x params = 
    let a = params !! 0
        b = params !! 1
        c = params !! 2
        d = params !! 3
    in if x <= a || x >= d
       then 0.0
       else if x >= b && x <= c
            then 1.0
            else if x > a && x < b
                 then (x - a) / (b - a)
                 else (d - x) / (d - c)

gaussmf :: Double -> [Double] -> Double
gaussmf x params = 
    let sigma = params !! 0
        c = params !! 1
    in exp (-0.5 * ((x - c) / sigma) ^ 2)

computeFuzzy :: FuzzyController -> [Double] -> Double
computeFuzzy controller inputs = 
    let rule_outputs = map (evaluateRule controller inputs) (rules controller)
        rule_weights = map (calculateRuleWeight controller inputs) (rules controller)
        total_weight = sum rule_weights
    in if total_weight > 0
       then sum (zipWith (*) rule_outputs rule_weights) / total_weight
       else 0.0

evaluateRule :: FuzzyController -> [Double] -> FuzzyRule -> Double
evaluateRule controller inputs rule = 
    let strength = minimum [evaluateMF (inputMFs controller !! input_idx !! mf_idx) (inputs !! input_idx)
                           | (input_idx, mf_idx) <- antecedents rule]
    in strength -- 简化输出

calculateRuleWeight :: FuzzyController -> [Double] -> FuzzyRule -> Double
calculateRuleWeight controller inputs rule = 
    minimum [evaluateMF (inputMFs controller !! input_idx !! mf_idx) (inputs !! input_idx)
             | (input_idx, mf_idx) <- antecedents rule]

-- 鲁棒控制器
data RobustController = RobustController {
    nominalController :: PIDController,
    uncertaintyBound :: Double,
    adaptiveGain :: Double
} deriving Show

newRobustController :: Double -> Double -> Double -> Double -> RobustController
newRobustController kp ki kd bound = RobustController {
    nominalController = newPIDController kp ki kd,
    uncertaintyBound = bound,
    adaptiveGain = 1.0
}

computeRobust :: RobustController -> Double -> Double -> (RobustController, Double)
computeRobust controller measurement dt = 
    let (updated_pid, nominal_output) = computePID (nominalController controller) measurement dt
        error = setpoint (nominalController controller) - measurement
        robust_term = adaptiveGain controller * uncertaintyBound controller * signum error
        output = nominal_output + robust_term
    in (controller { nominalController = updated_pid }, output)

updateAdaptiveGain :: RobustController -> Double -> RobustController
updateAdaptiveGain controller tracking_error = 
    let new_gain = adaptiveGain controller + 0.01 * abs tracking_error
        clamped_gain = max 0.1 (min 10.0 new_gain)
    in controller { adaptiveGain = clamped_gain }

-- 示例使用
example :: IO ()
example = do
    -- 创建二阶系统
    let a_matrix = [[0.0, 1.0], [-1.0, -0.5]]
        b_matrix = [[0.0], [1.0]]
        c_matrix = [[1.0, 0.0]]
        d_matrix = [[0.0]]
        system = newLinearSystem a_matrix b_matrix c_matrix d_matrix
    
    -- 创建PID控制器
    let controller = setLimits (-10.0) 10.0 $ setSetpoint 1.0 $ newPIDController 1.0 0.5 0.1
    
    -- 仿真
    let dt = 0.01
        simulation_time = 10.0
        steps = floor (simulation_time / dt)
        
        simulate 0 system controller = return ()
        simulate step system controller = do
            let (updated_system, outputs) = updateSystem system [0.0] dt
                output = head outputs
                (updated_controller, control) = computePID controller output dt
                (final_system, _) = updateSystem updated_system [control] dt
            
            if step `mod` 100 == 0
            then putStrLn $ "Step " ++ show step ++ ": Output = " ++ show output ++ ", Control = " ++ show control
            else return ()
            
            simulate (step - 1) final_system updated_controller
    
    simulate steps system controller
    
    -- 模糊控制示例
    let fuzzy_controller = addRule (FuzzyRule [(0, 0)] 0) $
                          addRule (FuzzyRule [(0, 1)] 1) $
                          addRule (FuzzyRule [(0, 2)] 2) $
                          addOutputMF (MembershipFunction "out1" "trimf" [-1.0, -0.5, 0.0]) $
                          addOutputMF (MembershipFunction "out2" "trimf" [-0.5, 0.0, 0.5]) $
                          addOutputMF (MembershipFunction "out3" "trimf" [0.0, 0.5, 1.0]) $
                          addInputMF 0 (MembershipFunction "negative" "trimf" [-1.0, -0.5, 0.0]) $
                          addInputMF 0 (MembershipFunction "zero" "trimf" [-0.5, 0.0, 0.5]) $
                          addInputMF 0 (MembershipFunction "positive" "trimf" [0.0, 0.5, 1.0]) $
                          newFuzzyController
        
        input = [0.3]
        fuzzy_output = computeFuzzy fuzzy_controller input
    
    putStrLn $ "Fuzzy control output: " ++ show fuzzy_output
    
    -- 鲁棒控制示例
    let robust_controller = newRobustController 1.0 0.5 0.1 0.1
        (_, robust_output) = computeRobust robust_controller 0.5 0.01
    
    putStrLn $ "Robust control output: " ++ show robust_output
```

### 应用领域 / Application Domains

#### 工业控制 / Industrial Control

- **过程控制**: 温度、压力、流量控制
- **运动控制**: 机器人、数控机床
- **电力系统**: 电压、频率控制

#### 航空航天 / Aerospace

- **飞行控制**: 姿态控制、轨迹跟踪
- **导航系统**: GPS、惯性导航
- **推进系统**: 发动机控制

#### 汽车控制 / Automotive Control

- **发动机控制**: 燃油喷射、点火控制
- **底盘控制**: 制动、转向、悬架
- **驾驶辅助**: 自适应巡航、车道保持

---

## 参考文献 / References

1. Franklin, G. F., et al. (2015). Feedback Control of Dynamic Systems. Pearson.
2. Åström, K. J., & Murray, R. M. (2008). Feedback Systems. Princeton University Press.
3. Skogestad, S., & Postlethwaite, I. (2005). Multivariable Feedback Control. Wiley.
4. Zadeh, L. A. (1965). Fuzzy Sets. Information and Control.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
