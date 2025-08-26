# 5.4 神经科学模型 / Neuroscience Models

## 目录 / Table of Contents

- [5.4 神经科学模型 / Neuroscience Models](#54-神经科学模型--neuroscience-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [5.4.1 神经元模型 / Neuron Models](#541-神经元模型--neuron-models)
    - [Hodgkin-Huxley模型 / Hodgkin-Huxley Model](#hodgkin-huxley模型--hodgkin-huxley-model)
    - [Integrate-and-Fire模型 / Integrate-and-Fire Model](#integrate-and-fire模型--integrate-and-fire-model)
    - [Izhikevich模型 / Izhikevich Model](#izhikevich模型--izhikevich-model)
  - [5.4.2 突触可塑性模型 / Synaptic Plasticity Models](#542-突触可塑性模型--synaptic-plasticity-models)
    - [Hebbian学习 / Hebbian Learning](#hebbian学习--hebbian-learning)
    - [STDP模型 / STDP Model](#stdp模型--stdp-model)
    - [突触强度调节 / Synaptic Strength Regulation](#突触强度调节--synaptic-strength-regulation)
  - [5.4.3 神经网络模型 / Neural Network Models](#543-神经网络模型--neural-network-models)
    - [前馈网络 / Feedforward Networks](#前馈网络--feedforward-networks)
    - [循环网络 / Recurrent Networks](#循环网络--recurrent-networks)
    - [脉冲神经网络 / Spiking Neural Networks](#脉冲神经网络--spiking-neural-networks)
  - [5.4.4 学习算法模型 / Learning Algorithm Models](#544-学习算法模型--learning-algorithm-models)
    - [反向传播 / Backpropagation](#反向传播--backpropagation)
    - [强化学习 / Reinforcement Learning](#强化学习--reinforcement-learning)
    - [无监督学习 / Unsupervised Learning](#无监督学习--unsupervised-learning)
  - [5.4.5 认知模型 / Cognitive Models](#545-认知模型--cognitive-models)
    - [工作记忆 / Working Memory](#工作记忆--working-memory)
    - [注意力机制 / Attention Mechanisms](#注意力机制--attention-mechanisms)
    - [决策模型 / Decision Models](#决策模型--decision-models)
  - [5.4.6 脑区功能模型 / Brain Region Function Models](#546-脑区功能模型--brain-region-function-models)
    - [视觉皮层 / Visual Cortex](#视觉皮层--visual-cortex)
    - [运动皮层 / Motor Cortex](#运动皮层--motor-cortex)
    - [海马体 / Hippocampus](#海马体--hippocampus)
  - [5.4.7 实现与应用 / Implementation and Applications](#547-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [脑机接口 / Brain-Computer Interface](#脑机接口--brain-computer-interface)
      - [神经康复 / Neural Rehabilitation](#神经康复--neural-rehabilitation)
      - [神经药物开发 / Neuropharmaceutical Development](#神经药物开发--neuropharmaceutical-development)
  - [参考文献 / References](#参考文献--references)

---

## 5.4.1 神经元模型 / Neuron Models

### Hodgkin-Huxley模型 / Hodgkin-Huxley Model

**膜电位方程**: $C_m \frac{dV}{dt} = I_{ext} - I_{Na} - I_K - I_L$

**钠离子电流**: $I_{Na} = g_{Na} m^3 h (V - E_{Na})$

**钾离子电流**: $I_K = g_K n^4 (V - E_K)$

**漏电流**: $I_L = g_L (V - E_L)$

**门控变量**: $\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V)m$

**激活函数**: $\alpha_m(V) = \frac{0.1(V+40)}{1-e^{-(V+40)/10}}$

### Integrate-and-Fire模型 / Integrate-and-Fire Model

**膜电位**: $\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R I(t)$

**阈值**: $V(t) \geq V_{threshold} \Rightarrow \text{spike}$

**重置**: $V(t) = V_{reset}$

**绝对不应期**: $t_{ref} = 2ms$

**发放率**: $\nu = \frac{1}{t_{ref} + \tau_m \ln(\frac{V_{threshold} - V_{rest}}{V_{reset} - V_{rest}})}$

### Izhikevich模型 / Izhikevich Model

**膜电位**: $\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$

**恢复变量**: $\frac{du}{dt} = a(bv - u)$

**重置条件**: $v \geq 30 \Rightarrow v = c, u = u + d$

**参数**: $a, b, c, d$ 控制神经元类型

---

## 5.4.2 突触可塑性模型 / Synaptic Plasticity Models

### Hebbian学习 / Hebbian Learning

**基本规则**: $\Delta w_{ij} = \eta x_i y_j$

**协方差规则**: $\Delta w_{ij} = \eta \text{Cov}(x_i, y_j)$

**BCM规则**: $\Delta w_{ij} = \eta x_i y_j(y_j - \theta)$

**Oja规则**: $\Delta w_{ij} = \eta x_i(y_j - w_{ij}x_i)$

### STDP模型 / STDP Model

**时间窗口**: $\Delta t = t_{post} - t_{pre}$

**权重变化**: $\Delta w = \begin{cases} A_+ e^{-\Delta t/\tau_+} & \text{if } \Delta t > 0 \\ A_- e^{\Delta t/\tau_-} & \text{if } \Delta t < 0 \end{cases}$

**参数**: $A_+, A_-, \tau_+, \tau_-$

**积分形式**: $\frac{dw}{dt} = \int_{-\infty}^{\infty} W(\Delta t) \rho_{pre}(t) \rho_{post}(t+\Delta t) d\Delta t$

### 突触强度调节 / Synaptic Strength Regulation

**稳态可塑性**: $\frac{dw}{dt} = \eta(\bar{r} - r_{target})$

**突触缩放**: $w_{new} = w_{old} \cdot \frac{\bar{r}_{target}}{\bar{r}_{current}}$

**突触竞争**: $\sum_i w_i = \text{constant}$

---

## 5.4.3 神经网络模型 / Neural Network Models

### 前馈网络 / Feedforward Networks

**前向传播**: $h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$

**激活函数**: $\sigma(x) = \frac{1}{1 + e^{-x}}$

**输出层**: $y = \text{softmax}(W^{(L)} h^{(L-1)} + b^{(L)})$

**损失函数**: $L = -\sum_i y_i \log(\hat{y}_i)$

### 循环网络 / Recurrent Networks

**隐藏状态**: $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$

**输出**: $y_t = W_{hy} h_t + b_y$

**梯度**: $\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial y_t} W_{hy} + \frac{\partial L}{\partial h_{t+1}} W_{hh}$

**LSTM**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

### 脉冲神经网络 / Spiking Neural Networks

**膜电位**: $V_i(t) = V_{rest} + \sum_j w_{ij} \sum_k \epsilon(t - t_j^k)$

**脉冲响应**: $\epsilon(t) = \frac{t}{\tau} e^{-t/\tau} \Theta(t)$

**发放概率**: $P(\text{spike}) = \frac{1}{1 + e^{-\beta(V - \theta)}}$

**STDP学习**: $\Delta w_{ij} = \eta \sum_k \sum_l W(\Delta t_{ij}^{kl})$

---

## 5.4.4 学习算法模型 / Learning Algorithm Models

### 反向传播 / Backpropagation

**误差梯度**: $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$

**权重更新**: $\Delta W^{(l)} = -\eta \delta^{(l)} (h^{(l-1)})^T$

**偏置更新**: $\Delta b^{(l)} = -\eta \delta^{(l)}$

**动量**: $\Delta W_t = \mu \Delta W_{t-1} - \eta \nabla L$

### 强化学习 / Reinforcement Learning

**Q学习**: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

**策略梯度**: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]$

**Actor-Critic**: $A(s,a) = Q(s,a) - V(s)$

**TD误差**: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 无监督学习 / Unsupervised Learning

**自编码器**: $\min \|x - \text{Dec}(\text{Enc}(x))\|^2$

**稀疏编码**: $\min \|x - D\alpha\|^2 + \lambda \|\alpha\|_1$

**竞争学习**: $\Delta w_i = \eta(x - w_i)$ if $i = \arg\min_j \|x - w_j\|$

**Hebbian聚类**: $\Delta w_{ij} = \eta(x_i - w_{ij})y_j$

---

## 5.4.5 认知模型 / Cognitive Models

### 工作记忆 / Working Memory

**持续活动**: $\tau \frac{dA}{dt} = -A + f(I + \sum_j w_{ij} A_j)$

**注意力门控**: $g_i = \sigma(\sum_j v_{ij} A_j + b_i)$

**记忆容量**: $C = \frac{1}{\Delta \theta}$

**遗忘**: $A(t) = A_0 e^{-t/\tau_{forget}}$

### 注意力机制 / Attention Mechanisms

**注意力权重**: $\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$

**注意力分数**: $e_i = \text{MLP}([h_i, s])$

**上下文向量**: $c = \sum_i \alpha_i h_i$

**多头注意力**: $\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, \ldots, head_h)W^O$

### 决策模型 / Decision Models

**漂移扩散**: $\frac{dx}{dt} = \mu + \sigma \eta(t)$

**决策阈值**: $x(t) \geq \theta \Rightarrow \text{decision}$

**反应时间**: $RT = \frac{\theta}{\mu} + \text{non-decision time}$

**准确率**: $P(\text{correct}) = \frac{1}{1 + e^{-2\mu\theta/\sigma^2}}$

---

## 5.4.6 脑区功能模型 / Brain Region Function Models

### 视觉皮层 / Visual Cortex

**感受野**: $RF(x,y) = \sum_i w_i I(x_i, y_i)$

**方向选择性**: $G(\theta) = \exp(-\frac{(\theta - \theta_0)^2}{2\sigma_\theta^2})$

**空间频率**: $F(f) = \exp(-\frac{(f - f_0)^2}{2\sigma_f^2})$

**简单细胞**: $S = \sum_i w_i I_i$

**复杂细胞**: $C = \sqrt{\sum_i S_i^2}$

### 运动皮层 / Motor Cortex

**运动方向**: $\vec{v} = \sum_i w_i \vec{d}_i$

**运动规划**: $\frac{d\vec{x}}{dt} = \vec{v}$

**运动执行**: $\vec{F} = M \frac{d^2\vec{x}}{dt^2} + B \frac{d\vec{x}}{dt} + K\vec{x}$

**运动学习**: $\Delta w_i = \eta \delta \vec{d}_i$

### 海马体 / Hippocampus

**位置细胞**: $f_i(\vec{x}) = \exp(-\frac{\|\vec{x} - \vec{c}_i\|^2}{2\sigma_i^2})$

**网格细胞**: $g_{i,j}(\vec{x}) = \cos(\vec{k}_{i,j} \cdot \vec{x} + \phi_{i,j})$

**时间细胞**: $h_i(t) = \exp(-\frac{(t - t_i)^2}{2\tau_i^2})$

**记忆整合**: $M(\vec{x}, t) = \sum_i w_i f_i(\vec{x}) h_i(t)$

---

## 5.4.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub membrane_potential: f64,
    pub threshold: f64,
    pub reset_potential: f64,
    pub resting_potential: f64,
    pub membrane_time_constant: f64,
    pub refractory_period: f64,
    pub last_spike_time: f64,
    pub current_time: f64,
}

impl Neuron {
    pub fn new() -> Self {
        Self {
            membrane_potential: -65.0,
            threshold: -55.0,
            reset_potential: -65.0,
            resting_potential: -65.0,
            membrane_time_constant: 20.0,
            refractory_period: 2.0,
            last_spike_time: -1000.0,
            current_time: 0.0,
        }
    }
    
    pub fn integrate_and_fire(&mut self, input_current: f64, dt: f64) -> bool {
        self.current_time += dt;
        
        // 检查不应期
        if self.current_time - self.last_spike_time < self.refractory_period {
            self.membrane_potential = self.reset_potential;
            return false;
        }
        
        // 膜电位更新
        let dv_dt = (-(self.membrane_potential - self.resting_potential) + input_current) / self.membrane_time_constant;
        self.membrane_potential += dv_dt * dt;
        
        // 检查是否发放
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            self.last_spike_time = self.current_time;
            return true;
        }
        
        false
    }
    
    pub fn hodgkin_huxley(&mut self, input_current: f64, dt: f64) -> bool {
        // 简化的Hodgkin-Huxley模型
        let g_na = 120.0;
        let g_k = 36.0;
        let g_l = 0.3;
        let e_na = 55.0;
        let e_k = -77.0;
        let e_l = -54.4;
        
        let v = self.membrane_potential;
        
        // 门控变量
        let alpha_m = 0.1 * (v + 40.0) / (1.0 - (-(v + 40.0) / 10.0).exp());
        let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();
        let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
        let alpha_n = 0.01 * (v + 55.0) / (1.0 - (-(v + 55.0) / 10.0).exp());
        let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();
        
        // 简化的门控变量更新
        let m = alpha_m / (alpha_m + beta_m);
        let h = alpha_h / (alpha_h + beta_h);
        let n = alpha_n / (alpha_n + beta_n);
        
        // 离子电流
        let i_na = g_na * m.powi(3) * h * (v - e_na);
        let i_k = g_k * n.powi(4) * (v - e_k);
        let i_l = g_l * (v - e_l);
        
        // 膜电位更新
        let dv_dt = (input_current - i_na - i_k - i_l) / 1.0; // 膜电容设为1
        self.membrane_potential += dv_dt * dt;
        
        // 检查发放
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            return true;
        }
        
        false
    }
}

#[derive(Debug)]
pub struct Synapse {
    pub weight: f64,
    pub delay: f64,
    pub spike_times: Vec<f64>,
    pub pre_neuron: usize,
    pub post_neuron: usize,
}

impl Synapse {
    pub fn new(pre: usize, post: usize, weight: f64) -> Self {
        Self {
            weight,
            delay: 1.0,
            spike_times: Vec::new(),
            pre_neuron: pre,
            post_neuron: post,
        }
    }
    
    pub fn add_spike(&mut self, time: f64) {
        self.spike_times.push(time);
    }
    
    pub fn get_current(&self, current_time: f64) -> f64 {
        let mut current = 0.0;
        for &spike_time in &self.spike_times {
            let t_diff = current_time - spike_time - self.delay;
            if t_diff > 0.0 {
                current += self.weight * (t_diff / 5.0) * (-t_diff / 5.0).exp();
            }
        }
        current
    }
    
    pub fn stdp_update(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        let delta_t = post_spike_time - pre_spike_time;
        let a_plus = 0.1;
        let a_minus = -0.1;
        let tau_plus = 20.0;
        let tau_minus = 20.0;
        
        if delta_t > 0.0 {
            self.weight += a_plus * (-delta_t / tau_plus).exp();
        } else {
            self.weight += a_minus * (delta_t / tau_minus).exp();
        }
        
        // 限制权重范围
        self.weight = self.weight.max(-5.0).min(5.0);
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub time: f64,
    pub dt: f64,
}

impl NeuralNetwork {
    pub fn new(num_neurons: usize) -> Self {
        let neurons = (0..num_neurons).map(|_| Neuron::new()).collect();
        
        Self {
            neurons,
            synapses: Vec::new(),
            time: 0.0,
            dt: 0.1,
        }
    }
    
    pub fn add_synapse(&mut self, pre: usize, post: usize, weight: f64) {
        self.synapses.push(Synapse::new(pre, post, weight));
    }
    
    pub fn simulate_step(&mut self) -> Vec<bool> {
        let mut spikes = vec![false; self.neurons.len()];
        
        // 更新神经元
        for i in 0..self.neurons.len() {
            let mut input_current = 0.0;
            
            // 计算突触输入
            for synapse in &self.synapses {
                if synapse.post_neuron == i {
                    input_current += synapse.get_current(self.time);
                }
            }
            
            // 更新神经元
            let spiked = self.neurons[i].integrate_and_fire(input_current, self.dt);
            spikes[i] = spiked;
            
            // 更新突触
            if spiked {
                for synapse in &mut self.synapses {
                    if synapse.post_neuron == i {
                        synapse.add_spike(self.time);
                    }
                }
            }
        }
        
        // STDP学习
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j && spikes[i] && spikes[j] {
                    for synapse in &mut self.synapses {
                        if synapse.pre_neuron == i && synapse.post_neuron == j {
                            synapse.stdp_update(self.time, self.time);
                        }
                    }
                }
            }
        }
        
        self.time += self.dt;
        spikes
    }
    
    pub fn simulate(&mut self, duration: f64) -> Vec<Vec<bool>> {
        let num_steps = (duration / self.dt) as usize;
        let mut spike_history = Vec::new();
        
        for _ in 0..num_steps {
            let spikes = self.simulate_step();
            spike_history.push(spikes);
        }
        
        spike_history
    }
    
    pub fn calculate_firing_rate(&self, spike_history: &Vec<Vec<bool>>) -> Vec<f64> {
        let num_neurons = self.neurons.len();
        let mut firing_rates = vec![0.0; num_neurons];
        
        for neuron_id in 0..num_neurons {
            let spike_count = spike_history.iter()
                .filter(|&step| step[neuron_id])
                .count();
            firing_rates[neuron_id] = spike_count as f64 / spike_history.len() as f64 / self.dt;
        }
        
        firing_rates
    }
}

#[derive(Debug)]
pub struct WorkingMemory {
    pub neurons: Vec<Neuron>,
    pub recurrent_weights: Vec<Vec<f64>>,
    pub input_weights: Vec<f64>,
    pub output_weights: Vec<f64>,
    pub time_constant: f64,
}

impl WorkingMemory {
    pub fn new(num_neurons: usize) -> Self {
        let neurons = (0..num_neurons).map(|_| Neuron::new()).collect();
        let recurrent_weights = vec![vec![0.1; num_neurons]; num_neurons];
        let input_weights = vec![1.0; num_neurons];
        let output_weights = vec![1.0; num_neurons];
        
        Self {
            neurons,
            recurrent_weights,
            input_weights,
            output_weights,
            time_constant: 100.0,
        }
    }
    
    pub fn update(&mut self, input: f64, dt: f64) -> f64 {
        let mut new_activities = Vec::new();
        
        for i in 0..self.neurons.len() {
            let mut total_input = input * self.input_weights[i];
            
            // 循环连接
            for j in 0..self.neurons.len() {
                total_input += self.recurrent_weights[i][j] * self.neurons[j].membrane_potential;
            }
            
            // 更新神经元
            let dv_dt = (-self.neurons[i].membrane_potential + total_input) / self.time_constant;
            self.neurons[i].membrane_potential += dv_dt * dt;
            
            new_activities.push(self.neurons[i].membrane_potential);
        }
        
        // 计算输出
        let output = new_activities.iter()
            .zip(&self.output_weights)
            .map(|(a, w)| a * w)
            .sum::<f64>();
        
        output
    }
    
    pub fn store_pattern(&mut self, pattern: Vec<f64>) {
        // 简化的模式存储
        for i in 0..self.neurons.len() {
            if i < pattern.len() {
                self.neurons[i].membrane_potential = pattern[i];
            }
        }
    }
}

// 使用示例
fn main() {
    // 单神经元模拟
    let mut neuron = Neuron::new();
    let mut spike_times = Vec::new();
    
    for step in 0..1000 {
        let input_current = if step < 100 { 10.0 } else { 0.0 };
        let spiked = neuron.integrate_and_fire(input_current, 0.1);
        if spiked {
            spike_times.push(step as f64 * 0.1);
        }
    }
    
    println!("Spike times: {:?}", spike_times);
    
    // 神经网络模拟
    let mut network = NeuralNetwork::new(10);
    
    // 添加突触连接
    for i in 0..9 {
        network.add_synapse(i, i + 1, 1.0);
    }
    network.add_synapse(9, 0, 1.0); // 循环连接
    
    let spike_history = network.simulate(100.0);
    let firing_rates = network.calculate_firing_rate(&spike_history);
    
    println!("Firing rates: {:?}", firing_rates);
    
    // 工作记忆模拟
    let mut wm = WorkingMemory::new(5);
    let pattern = vec![1.0, 0.5, 0.0, 0.5, 1.0];
    wm.store_pattern(pattern);
    
    let mut output_history = Vec::new();
    for step in 0..1000 {
        let output = wm.update(0.0, 0.1);
        output_history.push(output);
    }
    
    println!("Working memory output: {:?}", output_history[output_history.len()-1]);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module NeuroscienceModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, length, filter)
import System.Random (randomRs, newStdGen)

-- 神经元模型
data Neuron = Neuron {
    membranePotential :: Double,
    threshold :: Double,
    resetPotential :: Double,
    restingPotential :: Double,
    membraneTimeConstant :: Double,
    refractoryPeriod :: Double,
    lastSpikeTime :: Double,
    currentTime :: Double
} deriving Show

newNeuron :: Neuron
newNeuron = Neuron {
    membranePotential = -65.0,
    threshold = -55.0,
    resetPotential = -65.0,
    restingPotential = -65.0,
    membraneTimeConstant = 20.0,
    refractoryPeriod = 2.0,
    lastSpikeTime = -1000.0,
    currentTime = 0.0
}

integrateAndFire :: Neuron -> Double -> Double -> (Neuron, Bool)
integrateAndFire neuron inputCurrent dt = 
    let newTime = currentTime neuron + dt
        newNeuron = neuron { currentTime = newTime }
    in if newTime - lastSpikeTime newNeuron < refractoryPeriod newNeuron
       then (newNeuron { membranePotential = resetPotential newNeuron }, False)
       else let dv_dt = (-(membranePotential newNeuron - restingPotential newNeuron) + inputCurrent) / membraneTimeConstant newNeuron
                newPotential = membranePotential newNeuron + dv_dt * dt
            in if newPotential >= threshold newNeuron
               then (newNeuron { membranePotential = resetPotential newNeuron, lastSpikeTime = newTime }, True)
               else (newNeuron { membranePotential = newPotential }, False)

hodgkinHuxley :: Neuron -> Double -> Double -> (Neuron, Bool)
hodgkinHuxley neuron inputCurrent dt = 
    let v = membranePotential neuron
        g_na = 120.0
        g_k = 36.0
        g_l = 0.3
        e_na = 55.0
        e_k = -77.0
        e_l = -54.4
        
        -- 门控变量
        alpha_m = 0.1 * (v + 40.0) / (1.0 - exp (-(v + 40.0) / 10.0))
        beta_m = 4.0 * exp (-(v + 65.0) / 18.0)
        alpha_h = 0.07 * exp (-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp (-(v + 35.0) / 10.0))
        alpha_n = 0.01 * (v + 55.0) / (1.0 - exp (-(v + 55.0) / 10.0))
        beta_n = 0.125 * exp (-(v + 65.0) / 80.0)
        
        m = alpha_m / (alpha_m + beta_m)
        h = alpha_h / (alpha_h + beta_h)
        n = alpha_n / (alpha_n + beta_n)
        
        -- 离子电流
        i_na = g_na * m^3 * h * (v - e_na)
        i_k = g_k * n^4 * (v - e_k)
        i_l = g_l * (v - e_l)
        
        -- 膜电位更新
        dv_dt = (inputCurrent - i_na - i_k - i_l) / 1.0
        newPotential = v + dv_dt * dt
    in if newPotential >= threshold neuron
       then (neuron { membranePotential = resetPotential neuron }, True)
       else (neuron { membranePotential = newPotential }, False)

-- 突触模型
data Synapse = Synapse {
    weight :: Double,
    delay :: Double,
    spikeTimes :: [Double],
    preNeuron :: Int,
    postNeuron :: Int
} deriving Show

newSynapse :: Int -> Int -> Double -> Synapse
newSynapse pre post w = Synapse {
    weight = w,
    delay = 1.0,
    spikeTimes = [],
    preNeuron = pre,
    postNeuron = post
}

addSpike :: Synapse -> Double -> Synapse
addSpike synapse time = synapse { spikeTimes = time : spikeTimes synapse }

getCurrent :: Synapse -> Double -> Double
getCurrent synapse currentTime = 
    sum [weight synapse * (tDiff / 5.0) * exp (-tDiff / 5.0) | 
         spikeTime <- spikeTimes synapse,
         let tDiff = currentTime - spikeTime - delay synapse,
         tDiff > 0.0]

stdpUpdate :: Synapse -> Double -> Double -> Synapse
stdpUpdate synapse preSpikeTime postSpikeTime = 
    let deltaT = postSpikeTime - preSpikeTime
        a_plus = 0.1
        a_minus = -0.1
        tau_plus = 20.0
        tau_minus = 20.0
        weightChange = if deltaT > 0.0
                      then a_plus * exp (-deltaT / tau_plus)
                      else a_minus * exp (deltaT / tau_minus)
        newWeight = max (-5.0) (min 5.0 (weight synapse + weightChange))
    in synapse { weight = newWeight }

-- 神经网络
data NeuralNetwork = NeuralNetwork {
    neurons :: [Neuron],
    synapses :: [Synapse],
    time :: Double,
    dt :: Double
} deriving Show

newNeuralNetwork :: Int -> NeuralNetwork
newNeuralNetwork numNeurons = NeuralNetwork {
    neurons = replicate numNeurons newNeuron,
    synapses = [],
    time = 0.0,
    dt = 0.1
}

addSynapse :: Int -> Int -> Double -> NeuralNetwork -> NeuralNetwork
addSynapse pre post weight network = network {
    synapses = newSynapse pre post weight : synapses network
}

simulateStep :: NeuralNetwork -> (NeuralNetwork, [Bool])
simulateStep network = 
    let (newNeurons, spikes) = unzip $ zipWith (\i neuron -> 
        let inputCurrent = sum [getCurrent synapse (time network) | 
                               synapse <- synapses network, 
                               postNeuron synapse == i]
            (newNeuron, spiked) = integrateAndFire neuron inputCurrent (dt network)
        in (newNeuron, spiked)) [0..] (neurons network)
        
        newSynapses = map (\synapse -> 
            if any (\i -> spikes !! i) [preNeuron synapse, postNeuron synapse]
            then addSpike synapse (time network)
            else synapse) (synapses network)
        
        newTime = time network + dt network
    in (network { neurons = newNeurons, synapses = newSynapses, time = newTime }, spikes)

simulate :: NeuralNetwork -> Int -> [[Bool]]
simulate network numSteps = 
    let go 0 net = []
        go steps net = 
            let (newNet, spikes) = simulateStep net
            in spikes : go (steps - 1) newNet
    in go numSteps network

calculateFiringRate :: [[Bool]] -> [Double]
calculateFiringRate spikeHistory = 
    let numNeurons = length (head spikeHistory)
        numSteps = length spikeHistory
    in [fromIntegral (length (filter (!! i) spikeHistory)) / fromIntegral numSteps | i <- [0..numNeurons-1]]

-- 工作记忆
data WorkingMemory = WorkingMemory {
    wmNeurons :: [Neuron],
    recurrentWeights :: [[Double]],
    inputWeights :: [Double],
    outputWeights :: [Double],
    timeConstant :: Double
} deriving Show

newWorkingMemory :: Int -> WorkingMemory
newWorkingMemory numNeurons = WorkingMemory {
    wmNeurons = replicate numNeurons newNeuron,
    recurrentWeights = replicate numNeurons (replicate numNeurons 0.1),
    inputWeights = replicate numNeurons 1.0,
    outputWeights = replicate numNeurons 1.0,
    timeConstant = 100.0
}

updateWorkingMemory :: WorkingMemory -> Double -> Double -> (WorkingMemory, Double)
updateWorkingMemory wm input dt = 
    let newActivities = zipWith (\i neuron -> 
        let totalInput = input * (inputWeights wm !! i) + 
                        sum [recurrentWeights wm !! i !! j * membranePotential (wmNeurons wm !! j) | 
                             j <- [0..length (wmNeurons wm) - 1]]
            dv_dt = (-membranePotential neuron + totalInput) / timeConstant wm
            newPotential = membranePotential neuron + dv_dt * dt
        in newPotential) [0..] (wmNeurons wm)
        
        newNeurons = zipWith (\neuron newActivity -> 
            neuron { membranePotential = newActivity }) (wmNeurons wm) newActivities
        
        output = sum [newActivities !! i * (outputWeights wm !! i) | 
                     i <- [0..length newActivities - 1]]
    in (wm { wmNeurons = newNeurons }, output)

storePattern :: WorkingMemory -> [Double] -> WorkingMemory
storePattern wm pattern = 
    let newNeurons = zipWith (\neuron patternValue -> 
        if patternValue /= 0.0 
        then neuron { membranePotential = patternValue }
        else neuron) (wmNeurons wm) pattern
    in wm { wmNeurons = newNeurons }

-- 示例使用
example :: IO ()
example = do
    -- 单神经元模拟
    let neuron = newNeuron
        simulateNeuron 0 neuron = []
        simulateNeuron steps neuron = 
            let inputCurrent = if steps < 100 then 10.0 else 0.0
                (newNeuron, spiked) = integrateAndFire neuron inputCurrent 0.1
            in if spiked 
               then (currentTime newNeuron) : simulateNeuron (steps - 1) newNeuron
               else simulateNeuron (steps - 1) newNeuron
    
    let spikeTimes = simulateNeuron 1000 neuron
    putStrLn $ "Spike times: " ++ show (take 5 spikeTimes)
    
    -- 神经网络模拟
    let network = addSynapse 9 0 1.0 $ 
                  foldl (\net i -> addSynapse i (i + 1) 1.0 net) 
                        (newNeuralNetwork 10) [0..8]
        
        spikeHistory = simulate network 1000
        firingRates = calculateFiringRate spikeHistory
    
    putStrLn $ "Firing rates: " ++ show firingRates
    
    -- 工作记忆模拟
    let wm = newWorkingMemory 5
        pattern = [1.0, 0.5, 0.0, 0.5, 1.0]
        wmWithPattern = storePattern wm pattern
        
        simulateWM 0 wm = []
        simulateWM steps wm = 
            let (newWM, output) = updateWorkingMemory wm 0.0 0.1
            in output : simulateWM (steps - 1) newWM
        
        outputHistory = simulateWM 1000 wmWithPattern
    
    putStrLn $ "Working memory output: " ++ show (last outputHistory)
```

### 应用领域 / Application Domains

#### 脑机接口 / Brain-Computer Interface

- **神经解码**: 从脑信号解码意图
- **神经编码**: 向大脑传递信息
- **反馈控制**: 实时神经反馈系统

#### 神经康复 / Neural Rehabilitation

- **运动恢复**: 中风后运动功能恢复
- **感觉恢复**: 感觉神经修复
- **认知康复**: 记忆和注意力训练

#### 神经药物开发 / Neuropharmaceutical Development

- **药物筛选**: 神经活性化合物筛选
- **毒性评估**: 神经毒性预测
- **药效建模**: 药物作用机制建模

---

## 5.4.8 算法实现 / Algorithm Implementation

```python
import numpy as np
from typing import Tuple

def lif_simulate(T: float = 1.0, dt: float = 1e-4, I: float = 1.5, 
                 V_rest: float = -65.0, V_th: float = -50.0, V_reset: float = -65.0, 
                 tau_m: float = 0.02, t_ref: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    n = int(T / dt)
    V = np.ones(n) * V_rest
    spikes = np.zeros(n, dtype=bool)
    ref_counter = 0.0
    R = 1.0
    for t in range(1, n):
        if ref_counter > 0.0:
            V[t] = V_reset
            ref_counter -= dt
            continue
        dV = (-(V[t-1] - V_rest) + R * I) / tau_m
        V[t] = V[t-1] + dt * dV
        if V[t] >= V_th:
            spikes[t] = True
            V[t] = V_reset
            ref_counter = t_ref
    time = np.arange(n) * dt
    return time, V, spikes

def rate_model_step(r: float, I: float, tau: float, dt: float, phi=lambda x: np.maximum(0.0, x)) -> float:
    dr = (-r + phi(I)) / tau
    return max(0.0, r + dt * dr)

def stdp_update(w: float, delta_t: float, A_plus: float = 0.01, A_minus: float = 0.012, 
                tau_plus: float = 0.02, tau_minus: float = 0.02, wmin: float = 0.0, wmax: float = 1.0) -> float:
    if delta_t > 0:
        w += A_plus * np.exp(-delta_t / tau_plus)
    else:
        w -= A_minus * np.exp(delta_t / tau_minus)
    return min(wmax, max(wmin, w))

def network_dynamics_WilsonCowan(E: float, I: float, dt: float, params: dict) -> Tuple[float, float]:
    tau_E, tau_I = params['tau_E'], params['tau_I']
    w_EE, w_EI, w_IE, w_II = params['w_EE'], params['w_EI'], params['w_IE'], params['w_II']
    I_E, I_I = params['I_E'], params['I_I']
    phi = lambda x: 1.0 / (1.0 + np.exp(-x))
    dE = (-E + phi(w_EE * E - w_EI * I + I_E)) / tau_E
    dI = (-I + phi(w_IE * E - w_II * I + I_I)) / tau_I
    E_new = np.clip(E + dt * dE, 0.0, 1.0)
    I_new = np.clip(I + dt * dI, 0.0, 1.0)
    return E_new, I_new

def neuroscience_verification():
    # LIF产生脉冲
    t, V, spikes = lif_simulate(T=0.5, dt=1e-4, I=2.0)
    assert spikes.sum() > 0 and np.isfinite(V).all()
    
    # 发放率模型单步稳定
    r = 0.0
    for _ in range(1000):
        r = rate_model_step(r, I=1.0, tau=0.05, dt=1e-3)
    assert 0.0 <= r <= 1e3
    
    # STDP权重更新方向正确
    w = 0.5
    w_pot = stdp_update(w, delta_t=0.01)
    w_dep = stdp_update(w, delta_t=-0.01)
    assert w_pot > w and w_dep < w
    
    # Wilson-Cowan网络稳定迭代
    E, I = 0.1, 0.1
    params = {'tau_E': 0.02, 'tau_I': 0.01, 'w_EE': 10.0, 'w_EI': 12.0, 'w_IE': 10.0, 'w_II': 0.0, 'I_E': 0.5, 'I_I': 0.0}
    for _ in range(2000):
        E, I = network_dynamics_WilsonCowan(E, I, 1e-3, params)
    assert 0.0 <= E <= 1.0 and 0.0 <= I <= 1.0
    print("Neuroscience models verified.")

if __name__ == "__main__":
    neuroscience_verification()
```

## 参考文献 / References

1. Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology.
2. Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks.
3. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons. Journal of Neuroscience.
4. Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience. MIT Press.

---

*最后更新: 2025-08-26*
*版本: 1.1.0*
