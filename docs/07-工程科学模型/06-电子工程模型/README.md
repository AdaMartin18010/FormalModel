# 7.6 电子工程模型 / Electrical Engineering Models

## 目录 / Table of Contents

- [7.6 电子工程模型 / Electrical Engineering Models](#76-电子工程模型--electrical-engineering-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.6.1 电路分析模型 / Circuit Analysis Models](#761-电路分析模型--circuit-analysis-models)
    - [基尔霍夫定律 / Kirchhoff's Laws](#基尔霍夫定律--kirchhoffs-laws)
    - [线性电路分析 / Linear Circuit Analysis](#线性电路分析--linear-circuit-analysis)
    - [动态电路分析 / Dynamic Circuit Analysis](#动态电路分析--dynamic-circuit-analysis)
  - [7.6.2 电磁场模型 / Electromagnetic Field Models](#762-电磁场模型--electromagnetic-field-models)
    - [麦克斯韦方程 / Maxwell's Equations](#麦克斯韦方程--maxwells-equations)
    - [电磁波传播 / Electromagnetic Wave Propagation](#电磁波传播--electromagnetic-wave-propagation)
    - [天线理论 / Antenna Theory](#天线理论--antenna-theory)
  - [7.6.3 信号处理模型 / Signal Processing Models](#763-信号处理模型--signal-processing-models)
    - [模拟信号处理 / Analog Signal Processing](#模拟信号处理--analog-signal-processing)
    - [数字信号处理 / Digital Signal Processing](#数字信号处理--digital-signal-processing)
    - [调制解调 / Modulation and Demodulation](#调制解调--modulation-and-demodulation)
  - [7.6.4 通信系统模型 / Communication System Models](#764-通信系统模型--communication-system-models)
    - [信息论 / Information Theory](#信息论--information-theory)
    - [数字通信 / Digital Communication](#数字通信--digital-communication)
    - [多路复用 / Multiplexing](#多路复用--multiplexing)
  - [7.6.5 电力系统模型 / Power System Models](#765-电力系统模型--power-system-models)
    - [三相系统 / Three-Phase Systems](#三相系统--three-phase-systems)
    - [电力传输 / Power Transmission](#电力传输--power-transmission)
    - [电力电子 / Power Electronics](#电力电子--power-electronics)
  - [7.6.6 实现与应用 / Implementation and Applications](#766-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [电子设计 / Electronic Design](#电子设计--electronic-design)
      - [通信系统 / Communication Systems](#通信系统--communication-systems)
      - [电力系统 / Power Systems](#电力系统--power-systems)
  - [参考文献 / References](#参考文献--references)

---

## 7.6.1 电路分析模型 / Circuit Analysis Models

### 基尔霍夫定律 / Kirchhoff's Laws

**电流定律**: $\sum_{k=1}^n I_k = 0$

**电压定律**: $\sum_{k=1}^n V_k = 0$

**节点分析**: $[Y][V] = [I]$

**网孔分析**: $[Z][I] = [V]$

### 线性电路分析 / Linear Circuit Analysis

**戴维南定理**: $V_{oc} = V_{th}$, $R_{th} = \frac{V_{oc}}{I_{sc}}$

**诺顿定理**: $I_{sc} = I_n$, $R_n = \frac{V_{oc}}{I_{sc}}$

**叠加定理**: $V = \sum_{k=1}^n V_k$

### 动态电路分析 / Dynamic Circuit Analysis

**RC电路**: $v(t) = V_0 e^{-\frac{t}{RC}}$

**RL电路**: $i(t) = I_0 e^{-\frac{Rt}{L}}$

**RLC电路**: $\frac{d^2i}{dt^2} + \frac{R}{L}\frac{di}{dt} + \frac{1}{LC}i = 0$

---

## 7.6.2 电磁场模型 / Electromagnetic Field Models

### 麦克斯韦方程 / Maxwell's Equations

**高斯定律**: $\nabla \cdot \mathbf{D} = \rho$

**高斯磁定律**: $\nabla \cdot \mathbf{B} = 0$

**法拉第定律**: $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$

**安培定律**: $\nabla \times \mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}$

### 电磁波传播 / Electromagnetic Wave Propagation

**波动方程**: $\nabla^2 \mathbf{E} = \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}$

**波阻抗**: $Z = \sqrt{\frac{\mu}{\epsilon}}$

**传播常数**: $\gamma = \alpha + j\beta$

### 天线理论 / Antenna Theory

**辐射强度**: $U(\theta, \phi) = \frac{r^2}{2\eta} |E_\theta|^2$

**方向性**: $D = \frac{4\pi U_{max}}{P_{rad}}$

**增益**: $G = \eta D$

---

## 7.6.3 信号处理模型 / Signal Processing Models

### 模拟信号处理 / Analog Signal Processing

**滤波器传递函数**: $H(s) = \frac{Y(s)}{X(s)}$

**低通滤波器**: $H(s) = \frac{1}{1 + sRC}$

**高通滤波器**: $H(s) = \frac{sRC}{1 + sRC}$

**带通滤波器**: $H(s) = \frac{s/Q}{s^2 + s/Q + 1}$

### 数字信号处理 / Digital Signal Processing

**Z变换**: $X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}$

**差分方程**: $y[n] = \sum_{k=0}^M b_k x[n-k] - \sum_{k=1}^N a_k y[n-k]$

**频率响应**: $H(e^{j\omega}) = \frac{\sum_{k=0}^M b_k e^{-j\omega k}}{1 + \sum_{k=1}^N a_k e^{-j\omega k}}$

### 调制解调 / Modulation and Demodulation

**幅度调制**: $s(t) = A_c[1 + m(t)]\cos(\omega_c t)$

**频率调制**: $s(t) = A_c\cos[\omega_c t + k_f \int m(\tau) d\tau]$

**相位调制**: $s(t) = A_c\cos[\omega_c t + k_p m(t)]$

---

## 7.6.4 通信系统模型 / Communication System Models

### 信息论 / Information Theory

**香农熵**: $H(X) = -\sum_{i=1}^n p_i \log_2 p_i$

**信道容量**: $C = B \log_2(1 + \frac{S}{N})$

**误码率**: $P_e = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$

### 数字通信 / Digital Communication

**脉冲编码调制**: $x[n] = \text{quantize}(x(t))$

**正交幅度调制**: $s(t) = I(t)\cos(\omega_c t) + Q(t)\sin(\omega_c t)$

**码分多址**: $s_i(t) = d_i(t)c_i(t)\cos(\omega_c t)$

### 多路复用 / Multiplexing

**频分多路复用**: $s(t) = \sum_{k=1}^N s_k(t)\cos(\omega_k t)$

**时分多路复用**: $s(t) = \sum_{k=1}^N s_k(t - kT_s)$

**波分多路复用**: $s(t) = \sum_{k=1}^N s_k(t)e^{j\omega_k t}$

---

## 7.6.5 电力系统模型 / Power System Models

### 三相系统 / Three-Phase Systems

**相电压**: $V_a = V_m\cos(\omega t)$

**线电压**: $V_{ab} = \sqrt{3}V_a\angle 30°$

**功率**: $P = \sqrt{3}V_L I_L \cos(\phi)$

### 电力传输 / Power Transmission

**传输线方程**: $\frac{\partial^2 V}{\partial x^2} = LC\frac{\partial^2 V}{\partial t^2}$

**特性阻抗**: $Z_0 = \sqrt{\frac{L}{C}}$

**传播速度**: $v = \frac{1}{\sqrt{LC}}$

### 电力电子 / Power Electronics

**整流器**: $V_{dc} = \frac{2V_m}{\pi}$

**逆变器**: $v(t) = \sum_{n=1}^{\infty} \frac{4V_{dc}}{n\pi}\sin(n\omega t)$

**PWM控制**: $d(t) = \frac{V_{ref}(t)}{V_{carrier}}$

---

## 7.6.6 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct CircuitNode {
    pub id: usize,
    pub voltage: f64,
    pub connections: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CircuitElement {
    pub id: usize,
    pub element_type: String,
    pub value: f64,
    pub node1: usize,
    pub node2: usize,
}

#[derive(Debug)]
pub struct Circuit {
    pub nodes: Vec<CircuitNode>,
    pub elements: Vec<CircuitElement>,
}

impl Circuit {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, id: usize) {
        self.nodes.push(CircuitNode {
            id,
            voltage: 0.0,
            connections: Vec::new(),
        });
    }
    
    pub fn add_element(&mut self, element: CircuitElement) {
        self.elements.push(element.clone());
        
        // 更新节点连接
        if let Some(node1) = self.nodes.get_mut(element.node1) {
            node1.connections.push(element.node2);
        }
        if let Some(node2) = self.nodes.get_mut(element.node2) {
            node2.connections.push(element.node1);
        }
    }
    
    pub fn nodal_analysis(&mut self) -> Vec<f64> {
        let n = self.nodes.len();
        let mut conductance_matrix = vec![vec![0.0; n]; n];
        let mut current_vector = vec![0.0; n];
        
        // 构建导纳矩阵
        for element in &self.elements {
            let i = element.node1;
            let j = element.node2;
            
            match element.element_type.as_str() {
                "resistor" => {
                    let conductance = 1.0 / element.value;
                    conductance_matrix[i][i] += conductance;
                    conductance_matrix[j][j] += conductance;
                    conductance_matrix[i][j] -= conductance;
                    conductance_matrix[j][i] -= conductance;
                }
                "current_source" => {
                    current_vector[i] -= element.value;
                    current_vector[j] += element.value;
                }
                _ => {}
            }
        }
        
        // 求解线性方程组（简化实现）
        let mut voltages = vec![0.0; n];
        for i in 0..n {
            if i > 0 { // 跳过参考节点
                voltages[i] = current_vector[i] / conductance_matrix[i][i];
            }
        }
        
        voltages
    }
}

#[derive(Debug)]
pub struct ElectromagneticField {
    pub electric_field: [f64; 3],
    pub magnetic_field: [f64; 3],
    pub frequency: f64,
    pub wavelength: f64,
}

impl ElectromagneticField {
    pub fn new(frequency: f64) -> Self {
        let wavelength = 3e8 / frequency;
        Self {
            electric_field: [0.0; 3],
            magnetic_field: [0.0; 3],
            frequency,
            wavelength,
        }
    }
    
    pub fn plane_wave(&mut self, amplitude: f64, direction: [f64; 3], position: [f64; 3], time: f64) {
        let k = 2.0 * PI / self.wavelength;
        let omega = 2.0 * PI * self.frequency;
        
        let phase = k * (direction[0] * position[0] + direction[1] * position[1] + direction[2] * position[2]) - omega * time;
        
        self.electric_field[0] = amplitude * phase.cos();
        self.electric_field[1] = 0.0;
        self.electric_field[2] = 0.0;
        
        // 磁场垂直于电场
        let impedance = 377.0; // 自由空间阻抗
        self.magnetic_field[0] = 0.0;
        self.magnetic_field[1] = -amplitude / impedance * phase.cos();
        self.magnetic_field[2] = 0.0;
    }
    
    pub fn power_density(&self) -> f64 {
        let e_magnitude = (self.electric_field[0].powi(2) + self.electric_field[1].powi(2) + self.electric_field[2].powi(2)).sqrt();
        let h_magnitude = (self.magnetic_field[0].powi(2) + self.magnetic_field[1].powi(2) + self.magnetic_field[2].powi(2)).sqrt();
        
        e_magnitude * h_magnitude
    }
}

#[derive(Debug)]
pub struct Filter {
    pub filter_type: String,
    pub cutoff_frequency: f64,
    pub order: usize,
}

impl Filter {
    pub fn new(filter_type: String, cutoff_frequency: f64, order: usize) -> Self {
        Self {
            filter_type,
            cutoff_frequency,
            order,
        }
    }
    
    pub fn transfer_function(&self, frequency: f64) -> f64 {
        let normalized_frequency = frequency / self.cutoff_frequency;
        
        match self.filter_type.as_str() {
            "lowpass" => {
                let denominator = (1.0 + normalized_frequency.powi(2 * self.order as i32)).sqrt();
                1.0 / denominator
            }
            "highpass" => {
                let denominator = (1.0 + (1.0 / normalized_frequency).powi(2 * self.order as i32)).sqrt();
                1.0 / denominator
            }
            "bandpass" => {
                let q = 1.0; // 品质因数
                let denominator = (1.0 + q.powi(2) * (normalized_frequency - 1.0 / normalized_frequency).powi(2)).sqrt();
                1.0 / denominator
            }
            _ => 1.0
        }
    }
    
    pub fn group_delay(&self, frequency: f64) -> f64 {
        let delta_f = 1e-6;
        let phase1 = self.phase_response(frequency);
        let phase2 = self.phase_response(frequency + delta_f);
        
        -(phase2 - phase1) / (2.0 * PI * delta_f)
    }
    
    fn phase_response(&self, frequency: f64) -> f64 {
        let normalized_frequency = frequency / self.cutoff_frequency;
        
        match self.filter_type.as_str() {
            "lowpass" => -normalized_frequency.atan(),
            "highpass" => (PI / 2.0) - (1.0 / normalized_frequency).atan(),
            _ => 0.0
        }
    }
}

#[derive(Debug)]
pub struct CommunicationSystem {
    pub carrier_frequency: f64,
    pub sampling_rate: f64,
    pub modulation_type: String,
}

impl CommunicationSystem {
    pub fn new(carrier_frequency: f64, sampling_rate: f64, modulation_type: String) -> Self {
        Self {
            carrier_frequency,
            sampling_rate,
            modulation_type,
        }
    }
    
    pub fn amplitude_modulation(&self, message: &[f64], modulation_index: f64) -> Vec<f64> {
        let mut modulated_signal = Vec::new();
        
        for (i, &sample) in message.iter().enumerate() {
            let t = i as f64 / self.sampling_rate;
            let carrier = (2.0 * PI * self.carrier_frequency * t).cos();
            let modulated = carrier * (1.0 + modulation_index * sample);
            modulated_signal.push(modulated);
        }
        
        modulated_signal
    }
    
    pub fn frequency_modulation(&self, message: &[f64], frequency_deviation: f64) -> Vec<f64> {
        let mut modulated_signal = Vec::new();
        let mut phase = 0.0;
        
        for (i, &sample) in message.iter().enumerate() {
            let t = i as f64 / self.sampling_rate;
            let instantaneous_frequency = self.carrier_frequency + frequency_deviation * sample;
            phase += 2.0 * PI * instantaneous_frequency / self.sampling_rate;
            
            let modulated = phase.cos();
            modulated_signal.push(modulated);
        }
        
        modulated_signal
    }
    
    pub fn calculate_ber(&self, snr_db: f64, modulation_type: &str) -> f64 {
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        
        match modulation_type {
            "BPSK" => {
                let erfc_arg = (2.0 * snr_linear).sqrt();
                0.5 * (1.0 - erf(erfc_arg / 2.0_f64.sqrt()))
            }
            "QPSK" => {
                let erfc_arg = snr_linear.sqrt();
                0.5 * (1.0 - erf(erfc_arg / 2.0_f64.sqrt()))
            }
            "QAM16" => {
                let erfc_arg = (0.1 * snr_linear).sqrt();
                0.75 * (1.0 - erf(erfc_arg / 2.0_f64.sqrt()))
            }
            _ => 0.5
        }
    }
}

fn erf(x: f64) -> f64 {
    // 误差函数近似
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

#[derive(Debug)]
pub struct PowerSystem {
    pub voltage: f64,
    pub current: f64,
    pub frequency: f64,
    pub power_factor: f64,
}

impl PowerSystem {
    pub fn new(voltage: f64, current: f64, frequency: f64, power_factor: f64) -> Self {
        Self {
            voltage,
            current,
            frequency,
            power_factor,
        }
    }
    
    pub fn apparent_power(&self) -> f64 {
        self.voltage * self.current
    }
    
    pub fn real_power(&self) -> f64 {
        self.apparent_power() * self.power_factor
    }
    
    pub fn reactive_power(&self) -> f64 {
        let sin_phi = (1.0 - self.power_factor.powi(2)).sqrt();
        self.apparent_power() * sin_phi
    }
    
    pub fn three_phase_power(&self) -> f64 {
        3.0_f64.sqrt() * self.voltage * self.current * self.power_factor
    }
    
    pub fn transmission_line_loss(&self, resistance: f64, length: f64) -> f64 {
        let current_squared = self.current.powi(2);
        resistance * length * current_squared
    }
    
    pub fn voltage_regulation(&self, no_load_voltage: f64) -> f64 {
        ((no_load_voltage - self.voltage) / self.voltage) * 100.0
    }
}

// 使用示例
fn main() {
    // 电路分析示例
    let mut circuit = Circuit::new();
    
    // 添加节点
    circuit.add_node(0);
    circuit.add_node(1);
    circuit.add_node(2);
    
    // 添加元件
    circuit.add_element(CircuitElement {
        id: 1,
        element_type: "resistor".to_string(),
        value: 1000.0,
        node1: 0,
        node2: 1,
    });
    
    circuit.add_element(CircuitElement {
        id: 2,
        element_type: "resistor".to_string(),
        value: 2000.0,
        node1: 1,
        node2: 2,
    });
    
    circuit.add_element(CircuitElement {
        id: 3,
        element_type: "current_source".to_string(),
        value: 0.01,
        node1: 0,
        node2: 2,
    });
    
    let voltages = circuit.nodal_analysis();
    println!("Node voltages: {:?}", voltages);
    
    // 电磁场示例
    let mut em_field = ElectromagneticField::new(1e9); // 1 GHz
    em_field.plane_wave(100.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0);
    
    println!("Electric field: {:?}", em_field.electric_field);
    println!("Magnetic field: {:?}", em_field.magnetic_field);
    println!("Power density: {:.3} W/m²", em_field.power_density());
    
    // 滤波器示例
    let filter = Filter::new("lowpass".to_string(), 1000.0, 2);
    let transfer_function = filter.transfer_function(500.0);
    let group_delay = filter.group_delay(500.0);
    
    println!("Transfer function at 500 Hz: {:.3}", transfer_function);
    println!("Group delay at 500 Hz: {:.3} s", group_delay);
    
    // 通信系统示例
    let comm_system = CommunicationSystem::new(1e6, 1e7, "AM".to_string());
    let message = vec![0.5, 0.8, 0.3, 0.9, 0.2];
    let am_signal = comm_system.amplitude_modulation(&message, 0.5);
    let fm_signal = comm_system.frequency_modulation(&message, 1e5);
    
    println!("AM signal samples: {:?}", &am_signal[..3]);
    println!("FM signal samples: {:?}", &fm_signal[..3]);
    
    let ber = comm_system.calculate_ber(10.0, "BPSK");
    println!("Bit error rate: {:.6}", ber);
    
    // 电力系统示例
    let power_system = PowerSystem::new(220.0, 10.0, 50.0, 0.85);
    
    println!("Apparent power: {:.1} VA", power_system.apparent_power());
    println!("Real power: {:.1} W", power_system.real_power());
    println!("Reactive power: {:.1} VAR", power_system.reactive_power());
    println!("Three-phase power: {:.1} W", power_system.three_phase_power());
    
    let transmission_loss = power_system.transmission_line_loss(0.1, 1000.0);
    println!("Transmission line loss: {:.1} W", transmission_loss);
    
    let voltage_regulation = power_system.voltage_regulation(240.0);
    println!("Voltage regulation: {:.1}%", voltage_regulation);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module ElectricalEngineeringModels where

import Data.List (sum, length)
import Data.Complex (Complex(..), magnitude, phase)

-- 电路节点
data CircuitNode = CircuitNode {
    nodeId :: Int,
    nodeVoltage :: Double,
    nodeConnections :: [Int]
} deriving Show

-- 电路元件
data CircuitElement = CircuitElement {
    elementId :: Int,
    elementType :: String,
    elementValue :: Double,
    elementNode1 :: Int,
    elementNode2 :: Int
} deriving Show

-- 电路
data Circuit = Circuit {
    circuitNodes :: [CircuitNode],
    circuitElements :: [CircuitElement]
} deriving Show

newCircuit :: Circuit
newCircuit = Circuit [] []

addNode :: Int -> Circuit -> Circuit
addNode node_id circuit = circuit {
    circuitNodes = CircuitNode node_id 0.0 [] : circuitNodes circuit
}

addElement :: CircuitElement -> Circuit -> Circuit
addElement element circuit = circuit {
    circuitElements = element : circuitElements circuit
}

nodalAnalysis :: Circuit -> [Double]
nodalAnalysis circuit = 
    let n = length (circuitNodes circuit)
        conductance_matrix = replicate n (replicate n 0.0)
        current_vector = replicate n 0.0
    in solveLinearSystem conductance_matrix current_vector
  where
    solveLinearSystem matrix vector = 
        -- 简化的线性方程组求解
        take (length vector) $ repeat 0.0

-- 电磁场
data ElectromagneticField = ElectromagneticField {
    electricField :: [Double],
    magneticField :: [Double],
    fieldFrequency :: Double,
    fieldWavelength :: Double
} deriving Show

newElectromagneticField :: Double -> ElectromagneticField
newElectromagneticField frequency = ElectromagneticField {
    electricField = [0.0, 0.0, 0.0],
    magneticField = [0.0, 0.0, 0.0],
    fieldFrequency = frequency,
    fieldWavelength = 3e8 / frequency
}

planeWave :: ElectromagneticField -> Double -> [Double] -> [Double] -> Double -> ElectromagneticField
planeWave field amplitude direction position time = 
    let k = 2.0 * pi / fieldWavelength field
        omega = 2.0 * pi * fieldFrequency field
        phase = k * (direction !! 0 * position !! 0 + direction !! 1 * position !! 1 + direction !! 2 * position !! 2) - omega * time
        impedance = 377.0
    in field {
        electricField = [amplitude * cos phase, 0.0, 0.0],
        magneticField = [0.0, -amplitude / impedance * cos phase, 0.0]
    }

powerDensity :: ElectromagneticField -> Double
powerDensity field = 
    let e_magnitude = sqrt (sum (map (^2) (electricField field)))
        h_magnitude = sqrt (sum (map (^2) (magneticField field)))
    in e_magnitude * h_magnitude

-- 滤波器
data Filter = Filter {
    filterType :: String,
    cutoffFrequency :: Double,
    filterOrder :: Int
} deriving Show

newFilter :: String -> Double -> Int -> Filter
newFilter f_type cutoff order = Filter f_type cutoff order

transferFunction :: Filter -> Double -> Double
transferFunction filter frequency = 
    let normalized_frequency = frequency / cutoffFrequency filter
    in case filterType filter of
        "lowpass" -> 1.0 / sqrt (1.0 + normalized_frequency^(2 * filterOrder filter))
        "highpass" -> 1.0 / sqrt (1.0 + (1.0 / normalized_frequency)^(2 * filterOrder filter))
        "bandpass" -> 
            let q = 1.0
            in 1.0 / sqrt (1.0 + q^2 * (normalized_frequency - 1.0 / normalized_frequency)^2)
        _ -> 1.0

groupDelay :: Filter -> Double -> Double
groupDelay filter frequency = 
    let delta_f = 1e-6
        phase1 = phaseResponse filter frequency
        phase2 = phaseResponse filter (frequency + delta_f)
    in -(phase2 - phase1) / (2.0 * pi * delta_f)

phaseResponse :: Filter -> Double -> Double
phaseResponse filter frequency = 
    let normalized_frequency = frequency / cutoffFrequency filter
    in case filterType filter of
        "lowpass" -> -atan normalized_frequency
        "highpass" -> pi / 2.0 - atan (1.0 / normalized_frequency)
        _ -> 0.0

-- 通信系统
data CommunicationSystem = CommunicationSystem {
    carrierFrequency :: Double,
    samplingRate :: Double,
    modulationType :: String
} deriving Show

newCommunicationSystem :: Double -> Double -> String -> CommunicationSystem
newCommunicationSystem carrier_freq sample_rate mod_type = CommunicationSystem {
    carrierFrequency = carrier_freq,
    samplingRate = sample_rate,
    modulationType = mod_type
}

amplitudeModulation :: CommunicationSystem -> [Double] -> Double -> [Double]
amplitudeModulation system message mod_index = 
    zipWith (\i sample -> 
        let t = fromIntegral i / samplingRate system
            carrier = cos (2.0 * pi * carrierFrequency system * t)
        in carrier * (1.0 + mod_index * sample)) 
        [0..] message

frequencyModulation :: CommunicationSystem -> [Double] -> Double -> [Double]
frequencyModulation system message freq_deviation = 
    let go _ [] _ _ = []
        go phase (sample:samples) i freq_dev = 
            let t = fromIntegral i / samplingRate system
                instantaneous_freq = carrierFrequency system + freq_deviation * sample
                new_phase = phase + 2.0 * pi * instantaneous_freq / samplingRate system
            in cos new_phase : go new_phase samples (i + 1) freq_deviation
    in go 0.0 message 0 freq_deviation

calculateBER :: CommunicationSystem -> Double -> String -> Double
calculateBER system snr_db mod_type = 
    let snr_linear = 10.0**(snr_db / 10.0)
    in case mod_type of
        "BPSK" -> 
            let erfc_arg = sqrt (2.0 * snr_linear)
            in 0.5 * (1.0 - errorFunction (erfc_arg / sqrt 2.0))
        "QPSK" -> 
            let erfc_arg = sqrt snr_linear
            in 0.5 * (1.0 - errorFunction (erfc_arg / sqrt 2.0))
        "QAM16" -> 
            let erfc_arg = sqrt (0.1 * snr_linear)
            in 0.75 * (1.0 - errorFunction (erfc_arg / sqrt 2.0))
        _ -> 0.5

errorFunction :: Double -> Double
errorFunction x = 
    let a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = if x < 0.0 then -1.0 else 1.0
        x_abs = abs x
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp (-x_abs^2)
    in sign * y

-- 电力系统
data PowerSystem = PowerSystem {
    voltage :: Double,
    current :: Double,
    frequency :: Double,
    powerFactor :: Double
} deriving Show

newPowerSystem :: Double -> Double -> Double -> Double -> PowerSystem
newPowerSystem v i freq pf = PowerSystem v i freq pf

apparentPower :: PowerSystem -> Double
apparentPower system = voltage system * current system

realPower :: PowerSystem -> Double
realPower system = apparentPower system * powerFactor system

reactivePower :: PowerSystem -> Double
reactivePower system = 
    let sin_phi = sqrt (1.0 - powerFactor system^2)
    in apparentPower system * sin_phi

threePhasePower :: PowerSystem -> Double
threePhasePower system = sqrt 3.0 * voltage system * current system * powerFactor system

transmissionLineLoss :: PowerSystem -> Double -> Double -> Double
transmissionLineLoss system resistance length = 
    resistance * length * current system^2

voltageRegulation :: PowerSystem -> Double -> Double
voltageRegulation system no_load_voltage = 
    ((no_load_voltage - voltage system) / voltage system) * 100.0

-- 示例使用
example :: IO ()
example = do
    -- 电路分析示例
    let circuit = addElement (CircuitElement 1 "resistor" 1000.0 0 1) $
                  addElement (CircuitElement 2 "resistor" 2000.0 1 2) $
                  addElement (CircuitElement 3 "current_source" 0.01 0 2) $
                  addNode 2 $ addNode 1 $ addNode 0 newCircuit
        
        voltages = nodalAnalysis circuit
    
    putStrLn $ "Node voltages: " ++ show voltages
    
    -- 电磁场示例
    let em_field = newElectromagneticField 1e9 -- 1 GHz
        wave_field = planeWave em_field 100.0 [1.0, 0.0, 0.0] [0.0, 0.0, 0.0] 0.0
        power_density = powerDensity wave_field
    
    putStrLn $ "Electric field: " ++ show (electricField wave_field)
    putStrLn $ "Magnetic field: " ++ show (magneticField wave_field)
    putStrLn $ "Power density: " ++ show power_density ++ " W/m²"
    
    -- 滤波器示例
    let filter = newFilter "lowpass" 1000.0 2
        transfer_func = transferFunction filter 500.0
        group_delay = groupDelay filter 500.0
    
    putStrLn $ "Transfer function at 500 Hz: " ++ show transfer_func
    putStrLn $ "Group delay at 500 Hz: " ++ show group_delay ++ " s"
    
    -- 通信系统示例
    let comm_system = newCommunicationSystem 1e6 1e7 "AM"
        message = [0.5, 0.8, 0.3, 0.9, 0.2]
        am_signal = amplitudeModulation comm_system message 0.5
        fm_signal = frequencyModulation comm_system message 1e5
        ber = calculateBER comm_system 10.0 "BPSK"
    
    putStrLn $ "AM signal samples: " ++ show (take 3 am_signal)
    putStrLn $ "FM signal samples: " ++ show (take 3 fm_signal)
    putStrLn $ "Bit error rate: " ++ show ber
    
    -- 电力系统示例
    let power_system = newPowerSystem 220.0 10.0 50.0 0.85
    
    putStrLn $ "Apparent power: " ++ show (apparentPower power_system) ++ " VA"
    putStrLn $ "Real power: " ++ show (realPower power_system) ++ " W"
    putStrLn $ "Reactive power: " ++ show (reactivePower power_system) ++ " VAR"
    putStrLn $ "Three-phase power: " ++ show (threePhasePower power_system) ++ " W"
    
    let transmission_loss = transmissionLineLoss power_system 0.1 1000.0
        voltage_regulation = voltageRegulation power_system 240.0
    
    putStrLn $ "Transmission line loss: " ++ show transmission_loss ++ " W"
    putStrLn $ "Voltage regulation: " ++ show voltage_regulation ++ "%"
```

### 应用领域 / Application Domains

#### 电子设计 / Electronic Design

- **模拟电路**: 放大器、滤波器、振荡器
- **数字电路**: 逻辑门、时序电路、微处理器
- **混合信号**: ADC/DAC、PLL、数据转换

#### 通信系统 / Communication Systems

- **无线通信**: 移动通信、卫星通信
- **有线通信**: 光纤通信、电缆通信
- **网络协议**: TCP/IP、以太网、WiFi

#### 电力系统 / Power Systems

- **发电系统**: 火力、水力、核能发电
- **输电系统**: 高压输电、配电网络
- **电力电子**: 变频器、整流器、逆变器

---

## 参考文献 / References

1. Hayt, W. H., et al. (2018). Engineering Circuit Analysis. McGraw-Hill.
2. Griffiths, D. J. (2017). Introduction to Electrodynamics. Cambridge University Press.
3. Proakis, J. G., & Salehi, M. (2007). Digital Communications. McGraw-Hill.
4. Glover, J. D., et al. (2016). Power System Analysis & Design. Cengage Learning.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
