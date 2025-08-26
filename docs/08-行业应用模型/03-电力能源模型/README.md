# 8.3 电力能源模型 / Power & Energy Models

## 目录 / Table of Contents

- [8.3 电力能源模型 / Power \& Energy Models](#83-电力能源模型--power--energy-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.3.1 电力系统模型 / Power System Models](#831-电力系统模型--power-system-models)
    - [潮流计算 / Power Flow Analysis](#潮流计算--power-flow-analysis)
    - [稳定性分析 / Stability Analysis](#稳定性分析--stability-analysis)
    - [故障分析 / Fault Analysis](#故障分析--fault-analysis)
  - [8.3.2 发电模型 / Generation Models](#832-发电模型--generation-models)
    - [火电模型 / Thermal Power Models](#火电模型--thermal-power-models)
    - [水电模型 / Hydroelectric Models](#水电模型--hydroelectric-models)
    - [新能源模型 / Renewable Energy Models](#新能源模型--renewable-energy-models)
  - [8.3.3 输电网络模型 / Transmission Network Models](#833-输电网络模型--transmission-network-models)
    - [线路模型 / Line Models](#线路模型--line-models)
    - [变压器模型 / Transformer Models](#变压器模型--transformer-models)
    - [网络拓扑 / Network Topology](#网络拓扑--network-topology)
  - [8.3.4 配电系统模型 / Distribution System Models](#834-配电系统模型--distribution-system-models)
    - [配电网潮流 / Distribution Power Flow](#配电网潮流--distribution-power-flow)
    - [负荷建模 / Load Modeling](#负荷建模--load-modeling)
    - [电压控制 / Voltage Control](#电压控制--voltage-control)
  - [8.3.5 能源经济模型 / Energy Economics Models](#835-能源经济模型--energy-economics-models)
    - [电价模型 / Electricity Price Models](#电价模型--electricity-price-models)
    - [投资决策 / Investment Decision](#投资决策--investment-decision)
    - [市场机制 / Market Mechanisms](#市场机制--market-mechanisms)
  - [8.3.6 实现与应用 / Implementation and Applications](#836-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Julia实现示例 / Julia Implementation Example](#julia实现示例--julia-implementation-example)
  - [8.3.14 应用案例 / Application Cases](#8314-应用案例--application-cases)
    - [案例1：智能微电网调度 / Case 1: Smart Microgrid Dispatch](#案例1智能微电网调度--case-1-smart-microgrid-dispatch)
    - [案例2：电力市场交易策略 / Case 2: Power Market Trading Strategy](#案例2电力市场交易策略--case-2-power-market-trading-strategy)
    - [案例3：配电网规划优化 / Case 3: Distribution Network Planning](#案例3配电网规划优化--case-3-distribution-network-planning)
  - [8.3.15 发展趋势 / Development Trends](#8315-发展趋势--development-trends)
    - [数字化转型 / Digital Transformation](#数字化转型--digital-transformation)
    - [新能源集成 / Renewable Energy Integration](#新能源集成--renewable-energy-integration)
    - [市场机制创新 / Market Mechanism Innovation](#市场机制创新--market-mechanism-innovation)
    - [智能电网发展 / Smart Grid Development](#智能电网发展--smart-grid-development)
  - [8.3.16 人工智能与机器学习应用 / AI \& ML Applications](#8316-人工智能与机器学习应用--ai--ml-applications)
    - [机器学习在电力系统中的应用](#机器学习在电力系统中的应用)
    - [深度学习在电力系统中的应用](#深度学习在电力系统中的应用)
    - [强化学习在电力系统中的应用](#强化学习在电力系统中的应用)
  - [8.3.17 数字孪生技术 / Digital Twin Technology](#8317-数字孪生技术--digital-twin-technology)
    - [数字孪生在电力系统中的应用](#数字孪生在电力系统中的应用)
    - [数字孪生技术的关键挑战](#数字孪生技术的关键挑战)
  - [8.3.18 区块链在电力系统中的应用 / Blockchain in Power Systems](#8318-区块链在电力系统中的应用--blockchain-in-power-systems)
    - [区块链在电力系统中的应用](#区块链在电力系统中的应用)
    - [区块链在电力系统中的挑战](#区块链在电力系统中的挑战)
  - [8.3.19 量子计算在电力系统优化中的应用 / Quantum Computing in Power Systems](#8319-量子计算在电力系统优化中的应用--quantum-computing-in-power-systems)
    - [量子计算在电力系统优化中的应用](#量子计算在电力系统优化中的应用)
    - [量子计算在电力系统中的挑战](#量子计算在电力系统中的挑战)
  - [8.3.20 边缘计算与物联网 / Edge Computing \& IoT](#8320-边缘计算与物联网--edge-computing--iot)
    - [边缘计算与物联网在电力系统中的应用](#边缘计算与物联网在电力系统中的应用)
    - [边缘计算与物联网的挑战](#边缘计算与物联网的挑战)
  - [8.3.21 高级应用案例 / Advanced Application Cases](#8321-高级应用案例--advanced-application-cases)
    - [案例4：大规模电力系统优化 / Case 4: Large-Scale Power System Optimization](#案例4大规模电力系统优化--case-4-large-scale-power-system-optimization)
    - [案例5：智能配电网自愈 / Case 5: Smart Distribution Network Self-Healing](#案例5智能配电网自愈--case-5-smart-distribution-network-self-healing)
    - [案例6：虚拟电厂聚合优化 / Case 6: Virtual Power Plant Aggregation Optimization](#案例6虚拟电厂聚合优化--case-6-virtual-power-plant-aggregation-optimization)
  - [8.3.22 前沿技术实现 / Cutting-Edge Technology Implementation](#8322-前沿技术实现--cutting-edge-technology-implementation)
    - [量子机器学习实现 / Quantum Machine Learning Implementation](#量子机器学习实现--quantum-machine-learning-implementation)
    - [联邦学习在电力系统中的应用 / Federated Learning in Power Systems](#联邦学习在电力系统中的应用--federated-learning-in-power-systems)
  - [8.3.23 总结与展望 / Summary and Future Prospects](#8323-总结与展望--summary-and-future-prospects)
    - [电力能源模型的发展趋势](#电力能源模型的发展趋势)
    - [关键技术挑战](#关键技术挑战)
    - [未来研究方向](#未来研究方向)
  - [8.3.24 高级实现文件 / Advanced Implementation Files](#8324-高级实现文件--advanced-implementation-files)
    - [相关实现文件](#相关实现文件)
    - [技术发展趋势总结](#技术发展趋势总结)
    - [未来发展方向](#未来发展方向)
  - [8.3.25 算法实现 / Algorithm Implementation](#8325-算法实现--algorithm-implementation)
    - [电力系统分析算法 / Power System Analysis Algorithms](#电力系统分析算法--power-system-analysis-algorithms)
    - [发电模型算法 / Generation Model Algorithms](#发电模型算法--generation-model-algorithms)
    - [输电网络算法 / Transmission Network Algorithms](#输电网络算法--transmission-network-algorithms)
    - [配电系统算法 / Distribution System Algorithms](#配电系统算法--distribution-system-algorithms)
    - [能源经济算法 / Energy Economics Algorithms](#能源经济算法--energy-economics-algorithms)

---

## 8.3.1 电力系统模型 / Power System Models

### 潮流计算 / Power Flow Analysis

**节点功率方程**: $P_i = \sum_{j=1}^n V_i V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij})$

**节点无功功率**: $Q_i = \sum_{j=1}^n V_i V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij})$

**牛顿-拉夫森法**: $\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}$

**快速分解法**:

- 有功功率修正: $\Delta \theta = B'^{-1} \Delta P$
- 无功功率修正: $\Delta V = B''^{-1} \Delta Q$

### 稳定性分析 / Stability Analysis

**小信号稳定性**: $\dot{x} = Ax + Bu$

**特征值分析**: $\det(\lambda I - A) = 0$

**阻尼比**: $\zeta = \frac{-\sigma}{\sqrt{\sigma^2 + \omega^2}}$

**暂态稳定性**: $\frac{d\omega}{dt} = \frac{1}{2H}(P_m - P_e - D\Delta\omega)$

**电压稳定性**: $\frac{dV}{dt} = \frac{1}{T_v}(V_{ref} - V - K_q Q)$

### 故障分析 / Fault Analysis

**三相短路**: $I_f = \frac{V_f}{Z_f}$

**不对称故障**: $I_f^{(1)} = \frac{V_f^{(1)}}{Z_1 + Z_2 + Z_0}$

**故障电流计算**: $I_{fault} = \frac{V_{pre-fault}}{Z_{equivalent}}$

**故障清除时间**: $t_{clear} = t_{relay} + t_{breaker}$

---

## 8.3.2 发电模型 / Generation Models

### 火电模型 / Thermal Power Models

**热效率**: $\eta = \frac{P_{output}}{Q_{input}}$

**燃料消耗**: $F = \frac{P}{\eta \cdot LHV}$

**启动成本**: $C_{start} = C_{cold} + C_{hot} \cdot (1 - e^{-t/\tau})$

**爬坡约束**: $|P(t) - P(t-1)| \leq R_{up/down}$

**最小运行时间**: $T_{on} \geq T_{min\_on}$

**最小停机时间**: $T_{off} \geq T_{min\_off}$

### 水电模型 / Hydroelectric Models

**水头**: $H = H_{gross} - H_{loss}$

**功率**: $P = \eta \cdot \rho \cdot g \cdot Q \cdot H$

**水库调度**: $V(t+1) = V(t) + I(t) - Q(t) - L(t)$

**水头损失**: $H_{loss} = K \cdot Q^2$

**效率曲线**: $\eta = f(Q, H)$

### 新能源模型 / Renewable Energy Models

**风电功率**: $P = \frac{1}{2} \rho A v^3 C_p$

**光伏功率**: $P = P_{STC} \cdot \frac{G}{G_{STC}} \cdot [1 + \alpha(T - T_{STC})]$

**储能模型**: $SOC(t+1) = SOC(t) + \frac{P_{charge}(t) - P_{discharge}(t)}{E_{rated}} \cdot \Delta t$

**功率曲线**: $P_{wind} = \begin{cases} 0 & v < v_{cut-in} \\ P_{rated} \cdot \frac{v^3 - v_{cut-in}^3}{v_{rated}^3 - v_{cut-in}^3} & v_{cut-in} \leq v < v_{rated} \\ P_{rated} & v_{rated} \leq v < v_{cut-out} \\ 0 & v \geq v_{cut-out} \end{cases}$

---

## 8.3.3 输电网络模型 / Transmission Network Models

### 线路模型 / Line Models

**π型等效电路**: $Y = \frac{1}{R + jX} + j\frac{B}{2}$

**传输功率**: $P_{12} = \frac{V_1 V_2}{X} \sin(\theta_1 - \theta_2)$

**热极限**: $P_{max} = \frac{V_{rated}^2}{X} \sin(90°) = \frac{V_{rated}^2}{X}$

**稳定性极限**: $P_{max} = \frac{V_1 V_2}{X} \sin(\delta_{max})$

**线路损耗**: $P_{loss} = I^2 R = \frac{P^2 + Q^2}{V^2} R$

### 变压器模型 / Transformer Models

**变比**: $a = \frac{N_1}{N_2} = \frac{V_1}{V_2}$

**阻抗**: $Z_{pu} = \frac{Z_{actual}}{Z_{base}}$

**损耗**: $P_{loss} = I^2 R + P_{core}$

**效率**: $\eta = \frac{P_{output}}{P_{input}} = \frac{P_{input} - P_{loss}}{P_{input}}$

### 网络拓扑 / Network Topology

**节点导纳矩阵**: $Y_{bus} = A Y A^T$

**节点阻抗矩阵**: $Z_{bus} = Y_{bus}^{-1}$

**网络连通性**: $C = \frac{2E}{V(V-1)}$

**网络效率**: $\eta_{network} = \frac{\sum_{i,j} d_{ij}^{-1}}{V(V-1)}$

---

## 8.3.4 配电系统模型 / Distribution System Models

### 配电网潮流 / Distribution Power Flow

**前推回代法**:

- 前推: $V_i^{(k+1)} = V_{i-1}^{(k)} - I_i^{(k)} Z_i$
- 回代: $I_i^{(k+1)} = \frac{S_i^*}{V_i^{(k+1)*}}$

**牛顿-拉夫森法**: $\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}$

**高斯-赛德尔法**: $V_i^{(k+1)} = \frac{1}{Y_{ii}} \left( \frac{S_i^*}{V_i^{(k)*}} - \sum_{j \neq i} Y_{ij} V_j^{(k)} \right)$

### 负荷建模 / Load Modeling

**恒功率负荷**: $P = P_0, Q = Q_0$

**恒阻抗负荷**: $P = P_0 \left(\frac{V}{V_0}\right)^2$

**恒电流负荷**: $P = P_0 \left(\frac{V}{V_0}\right)$

**综合负荷模型**: $P = P_0 \left[ a_p \left(\frac{V}{V_0}\right)^2 + b_p \left(\frac{V}{V_0}\right) + c_p \right]$

**负荷预测**: $P(t) = P_{base}(t) + \Delta P_{weather}(t) + \Delta P_{event}(t)$

### 电压控制 / Voltage Control

**电压调节**: $\Delta V = \frac{R \Delta P + X \Delta Q}{V}$

**电容器补偿**: $Q_c = \frac{V^2}{X_c}$

**有载调压**: $V_{secondary} = V_{primary} \cdot \frac{N_2}{N_1} \cdot (1 \pm n \cdot \Delta t)$

**电压稳定性**: $\frac{dV}{dt} = \frac{1}{T_v}(V_{ref} - V - K_q Q)$

---

## 8.3.5 能源经济模型 / Energy Economics Models

### 电价模型 / Electricity Price Models

**边际成本**: $MC = \frac{\partial TC}{\partial Q}$

**电价**: $P = MC + markup$

**峰谷电价**: $P_{peak} = \alpha P_{base}, P_{valley} = \beta P_{base}$

**实时电价**: $P_{RT}(t) = MC(t) + \lambda_{congestion}(t) + \lambda_{loss}(t)$

**需求响应**: $P_{DR}(t) = P_{base}(t) \cdot (1 - \alpha \cdot \Delta P(t))$

### 投资决策 / Investment Decision

**净现值**: $NPV = \sum_{t=0}^T \frac{CF_t}{(1+r)^t}$

**内部收益率**: $\sum_{t=0}^T \frac{CF_t}{(1+IRR)^t} = 0$

**投资回收期**: $\sum_{t=0}^{PBP} CF_t = 0$

**风险调整收益率**: $RAROC = \frac{R - R_f}{\sigma_R}$

### 市场机制 / Market Mechanisms

**竞价机制**: $P = \arg\max \sum_{i=1}^n b_i(Q_i)$

**市场出清**: $\sum_{i=1}^n Q_i^{supply} = \sum_{j=1}^m Q_j^{demand}$

**价格上限**: $P_{max} = MC_{marginal} + \lambda_{cap}$

**容量市场**: $C_{capacity} = \sum_{i=1}^n P_i^{capacity} \cdot \lambda_{capacity}$

---

## 8.3.6 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSystemModel {
    pub buses: HashMap<String, Bus>,
    pub lines: HashMap<String, Line>,
    pub generators: HashMap<String, Generator>,
    pub loads: HashMap<String, Load>,
    pub transformers: HashMap<String, Transformer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bus {
    pub id: String,
    pub voltage_magnitude: f64,
    pub voltage_angle: f64,
    pub bus_type: BusType,
    pub active_power: f64,
    pub reactive_power: f64,
    pub base_voltage: f64,
    pub area: String,
    pub zone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub from_bus: String,
    pub to_bus: String,
    pub resistance: f64,
    pub reactance: f64,
    pub susceptance: f64,
    pub capacity: f64,
    pub length: f64,
    pub status: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator {
    pub bus_id: String,
    pub active_power: f64,
    pub reactive_power: f64,
    pub max_power: f64,
    pub min_power: f64,
    pub cost: f64,
    pub fuel_type: FuelType,
    pub efficiency: f64,
    pub ramp_rate: f64,
    pub min_up_time: i32,
    pub min_down_time: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FuelType {
    Coal,
    Gas,
    Nuclear,
    Hydro,
    Wind,
    Solar,
    Biomass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Load {
    pub bus_id: String,
    pub active_power: f64,
    pub reactive_power: f64,
    pub load_type: LoadType,
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadType {
    Residential,
    Commercial,
    Industrial,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformer {
    pub from_bus: String,
    pub to_bus: String,
    pub primary_voltage: f64,
    pub secondary_voltage: f64,
    pub impedance: f64,
    pub tap_ratio: f64,
    pub tap_range: (f64, f64),
    pub tap_step: f64,
}

impl PowerSystemModel {
    pub fn new() -> Self {
        Self {
            buses: HashMap::new(),
            lines: HashMap::new(),
            generators: HashMap::new(),
            loads: HashMap::new(),
            transformers: HashMap::new(),
        }
    }
    
    pub fn add_bus(&mut self, bus: Bus) {
        self.buses.insert(bus.id.clone(), bus);
    }
    
    pub fn add_line(&mut self, line: Line) {
        let line_id = format!("{}-{}", line.from_bus, line.to_bus);
        self.lines.insert(line_id, line);
    }
    
    pub fn add_generator(&mut self, generator: Generator) {
        self.generators.insert(generator.bus_id.clone(), generator);
    }
    
    pub fn add_load(&mut self, load: Load) {
        self.loads.insert(load.bus_id.clone(), load);
    }
    
    pub fn add_transformer(&mut self, transformer: Transformer) {
        let trans_id = format!("{}-{}", transformer.from_bus, transformer.to_bus);
        self.transformers.insert(trans_id, transformer);
    }
    
    pub fn power_flow_analysis(&self) -> Result<HashMap<String, (f64, f64)>, String> {
        let mut results = HashMap::new();
        
        // 构建节点导纳矩阵
        let y_bus = self.build_y_bus_matrix()?;
        
        // 牛顿-拉夫森迭代
        let mut voltages = self.initialize_voltages();
        let mut angles = self.initialize_angles();
        
        for iteration in 0..100 {
            let (mismatch_p, mismatch_q) = self.calculate_power_mismatches(&voltages, &angles);
            
            if self.check_convergence(&mismatch_p, &mismatch_q) {
                break;
            }
            
            let jacobian = self.build_jacobian_matrix(&voltages, &angles, &y_bus);
            let corrections = self.solve_linear_system(&jacobian, &mismatch_p, &mismatch_q)?;
            
            self.update_variables(&mut voltages, &mut angles, &corrections);
        }
        
        // 计算最终结果
        for (bus_id, bus) in &self.buses {
            let power_injection = self.calculate_power_injection(bus_id, &voltages, &angles);
            results.insert(bus_id.clone(), power_injection);
        }
        
        Ok(results)
    }
    
    fn build_y_bus_matrix(&self) -> Result<DMatrix<f64>, String> {
        let n_buses = self.buses.len();
        let mut y_bus = DMatrix::zeros(n_buses, n_buses);
        
        // 构建导纳矩阵
        for (line_id, line) in &self.lines {
            if !line.status {
                continue;
            }
            
            let from_idx = self.get_bus_index(&line.from_bus)?;
            let to_idx = self.get_bus_index(&line.to_bus)?;
            
            let y_line = 1.0 / (line.resistance + line.reactance * 1.0i);
            let y_shunt = line.susceptance * 1.0i / 2.0;
            
            y_bus[(from_idx, from_idx)] += y_line + y_shunt;
            y_bus[(to_idx, to_idx)] += y_line + y_shunt;
            y_bus[(from_idx, to_idx)] -= y_line;
            y_bus[(to_idx, from_idx)] -= y_line;
        }
        
        Ok(y_bus)
    }
    
    fn get_bus_index(&self, bus_id: &str) -> Result<usize, String> {
        let bus_ids: Vec<&String> = self.buses.keys().collect();
        bus_ids.iter()
            .position(|&id| id == bus_id)
            .ok_or_else(|| format!("Bus {} not found", bus_id))
    }
    
    fn initialize_voltages(&self) -> Vec<f64> {
        self.buses.values()
            .map(|bus| bus.voltage_magnitude)
            .collect()
    }
    
    fn initialize_angles(&self) -> Vec<f64> {
        self.buses.values()
            .map(|bus| bus.voltage_angle)
            .collect()
    }
    
    fn calculate_power_mismatches(&self, voltages: &[f64], angles: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mismatch_p = Vec::new();
        let mut mismatch_q = Vec::new();
        
        for (bus_id, bus) in &self.buses {
            let (p_calc, q_calc) = self.calculate_power_injection(bus_id, voltages, angles);
            let p_mismatch = bus.active_power - p_calc;
            let q_mismatch = bus.reactive_power - q_calc;
            
            mismatch_p.push(p_mismatch);
            mismatch_q.push(q_mismatch);
        }
        
        (mismatch_p, mismatch_q)
    }
    
    fn calculate_power_injection(&self, bus_id: &str, voltages: &[f64], angles: &[f64]) -> (f64, f64) {
        let bus_idx = self.get_bus_index(bus_id).unwrap();
        let v_i = voltages[bus_idx];
        let theta_i = angles[bus_idx];
        
        let mut p_injection = 0.0;
        let mut q_injection = 0.0;
        
        for (other_bus_id, _) in &self.buses {
            let other_idx = self.get_bus_index(other_bus_id).unwrap();
            let v_j = voltages[other_idx];
            let theta_j = angles[other_idx];
            
            // 简化的功率计算
            let delta_theta = theta_i - theta_j;
            let g_ij = 0.1; // 简化的导纳
            let b_ij = -0.5;
            
            p_injection += v_i * v_j * (g_ij * delta_theta.cos() + b_ij * delta_theta.sin());
            q_injection += v_i * v_j * (g_ij * delta_theta.sin() - b_ij * delta_theta.cos());
        }
        
        (p_injection, q_injection)
    }
    
    fn check_convergence(&self, mismatch_p: &[f64], mismatch_q: &[f64]) -> bool {
        let tolerance = 1e-6;
        mismatch_p.iter().all(|&m| m.abs() < tolerance) &&
        mismatch_q.iter().all(|&m| m.abs() < tolerance)
    }
    
    fn build_jacobian_matrix(&self, voltages: &[f64], angles: &[f64], y_bus: &DMatrix<f64>) -> DMatrix<f64> {
        // 简化的雅可比矩阵构建
        let n_buses = self.buses.len();
        DMatrix::identity(n_buses * 2, n_buses * 2)
    }
    
    fn solve_linear_system(&self, jacobian: &DMatrix<f64>, mismatch_p: &[f64], mismatch_q: &[f64]) -> Result<Vec<f64>, String> {
        let n_buses = self.buses.len();
        let mut rhs = DVector::zeros(n_buses * 2);
        
        for i in 0..n_buses {
            rhs[i] = mismatch_p[i];
            rhs[i + n_buses] = mismatch_q[i];
        }
        
        match jacobian.lu().solve(&rhs) {
            Some(solution) => Ok(solution.as_slice().to_vec()),
            None => Err("Failed to solve linear system".to_string()),
        }
    }
    
    fn update_variables(&self, voltages: &mut [f64], angles: &mut [f64], corrections: &[f64]) {
        let n_buses = self.buses.len();
        
        for i in 0..n_buses {
            angles[i] += corrections[i];
            voltages[i] += corrections[i + n_buses];
        }
    }
    
    pub fn economic_dispatch(&self, total_demand: f64) -> HashMap<String, f64> {
        let mut dispatch = HashMap::new();
        let mut remaining_demand = total_demand;
        
        // 按成本排序发电机
        let mut sorted_generators: Vec<_> = self.generators.iter().collect();
        sorted_generators.sort_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
        
        for (bus_id, generator) in sorted_generators {
            if remaining_demand > 0.0 {
                let allocated_power = remaining_demand.min(generator.max_power);
                dispatch.insert(bus_id.clone(), allocated_power);
                remaining_demand -= allocated_power;
            } else {
                dispatch.insert(bus_id.clone(), 0.0);
            }
        }
        
        dispatch
    }
    
    pub fn unit_commitment(&self, demand_profile: &[f64]) -> HashMap<String, Vec<bool>> {
        let mut commitment = HashMap::new();
        
        for (gen_id, generator) in &self.generators {
            let mut schedule = Vec::new();
            
            for &demand in demand_profile {
                // 简化的启停逻辑
                let should_commit = demand > generator.max_power * 0.3;
                schedule.push(should_commit);
            }
            
            commitment.insert(gen_id.clone(), schedule);
        }
        
        commitment
    }
    
    pub fn calculate_system_losses(&self) -> f64 {
        let mut total_losses = 0.0;
        
        for line in self.lines.values() {
            if line.status {
                let current = self.calculate_line_current(line);
                let losses = current * current * line.resistance;
                total_losses += losses;
            }
        }
        
        total_losses
    }
    
    fn calculate_line_current(&self, line: &Line) -> f64 {
        // 简化的电流计算
        let voltage_diff = 1.0; // 假设电压差
        let impedance = (line.resistance * line.resistance + line.reactance * line.reactance).sqrt();
        voltage_diff / impedance
    }
    
    pub fn calculate_reliability_indices(&self) -> HashMap<String, f64> {
        let total_load: f64 = self.loads.values().map(|load| load.active_power).sum();
        let total_capacity: f64 = self.generators.values().map(|gen| gen.max_power).sum();
        
        let reserve_margin = (total_capacity - total_load) / total_load;
        let loss_of_load_probability = if reserve_margin < 0.1 { 1.0 - reserve_margin } else { 0.0 };
        
        let mut indices = HashMap::new();
        indices.insert("reserve_margin".to_string(), reserve_margin);
        indices.insert("loss_of_load_probability".to_string(), loss_of_load_probability);
        indices.insert("total_capacity".to_string(), total_capacity);
        indices.insert("total_load".to_string(), total_load);
        
        indices
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenewableEnergyModel {
    pub wind_speed: f64,
    pub solar_irradiance: f64,
    pub temperature: f64,
    pub wind_turbine_params: WindTurbineParams,
    pub solar_panel_params: SolarPanelParams,
    pub forecast_horizon: usize,
    pub weather_data: Vec<WeatherData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindTurbineParams {
    pub rated_power: f64,
    pub cut_in_speed: f64,
    pub rated_speed: f64,
    pub cut_out_speed: f64,
    pub rotor_area: f64,
    pub power_coefficient: f64,
    pub hub_height: f64,
    pub wind_speed_profile: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarPanelParams {
    pub rated_power: f64,
    pub area: f64,
    pub efficiency: f64,
    pub temperature_coefficient: f64,
    pub tilt_angle: f64,
    pub azimuth_angle: f64,
    pub tracking_type: TrackingType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackingType {
    Fixed,
    SingleAxis,
    DualAxis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub timestamp: i64,
    pub wind_speed: f64,
    pub solar_irradiance: f64,
    pub temperature: f64,
    pub humidity: f64,
    pub pressure: f64,
}

impl RenewableEnergyModel {
    pub fn new() -> Self {
        Self {
            wind_speed: 0.0,
            solar_irradiance: 0.0,
            temperature: 25.0,
            wind_turbine_params: WindTurbineParams {
                rated_power: 2000.0,
                cut_in_speed: 3.0,
                rated_speed: 12.0,
                cut_out_speed: 25.0,
                rotor_area: 1000.0,
                power_coefficient: 0.4,
                hub_height: 80.0,
                wind_speed_profile: Vec::new(),
            },
            solar_panel_params: SolarPanelParams {
                rated_power: 1000.0,
                area: 6.0,
                efficiency: 0.15,
                temperature_coefficient: -0.004,
                tilt_angle: 30.0,
                azimuth_angle: 180.0,
                tracking_type: TrackingType::Fixed,
            },
            forecast_horizon: 24,
            weather_data: Vec::new(),
        }
    }
    
    pub fn calculate_wind_power(&self) -> f64 {
        let wind_speed = self.wind_speed;
        let params = &self.wind_turbine_params;
        
        if wind_speed < params.cut_in_speed || wind_speed > params.cut_out_speed {
            0.0
        } else if wind_speed < params.rated_speed {
            let power = 0.5 * 1.225 * params.rotor_area * wind_speed.powi(3) * params.power_coefficient;
            power.min(params.rated_power)
        } else {
            params.rated_power
        }
    }
    
    pub fn calculate_solar_power(&self) -> f64 {
        let irradiance = self.solar_irradiance;
        let temperature = self.temperature;
        let params = &self.solar_panel_params;
        
        let temperature_factor = 1.0 + params.temperature_coefficient * (temperature - 25.0);
        let power = irradiance * params.area * params.efficiency * temperature_factor;
        
        power.min(params.rated_power)
    }
    
    pub fn calculate_total_renewable_power(&self) -> f64 {
        self.calculate_wind_power() + self.calculate_solar_power()
    }
    
    pub fn generate_forecast(&self, horizon: usize) -> Vec<f64> {
        let mut forecast = Vec::new();
        
        for t in 0..horizon {
            let wind_power = self.calculate_wind_forecast(t);
            let solar_power = self.calculate_solar_forecast(t);
            forecast.push(wind_power + solar_power);
        }
        
        forecast
    }
    
    fn calculate_wind_forecast(&self, time_step: usize) -> f64 {
        // 简化的风电预测
        let base_wind = 8.0;
        let daily_pattern = (2.0 * std::f64::consts::PI * time_step as f64 / 24.0).sin() * 2.0;
        let wind_speed = base_wind + daily_pattern;
        
        self.calculate_wind_power_at_speed(wind_speed)
    }
    
    fn calculate_solar_forecast(&self, time_step: usize) -> f64 {
        // 简化的光伏预测
        let max_irradiance = 1000.0;
        let daily_pattern = (std::f64::consts::PI * time_step as f64 / 24.0).sin().max(0.0);
        let irradiance = max_irradiance * daily_pattern;
        
        self.calculate_solar_power_at_irradiance(irradiance)
    }
    
    fn calculate_wind_power_at_speed(&self, wind_speed: f64) -> f64 {
        let params = &self.wind_turbine_params;
        
        if wind_speed < params.cut_in_speed || wind_speed > params.cut_out_speed {
            0.0
        } else if wind_speed < params.rated_speed {
            let power = 0.5 * 1.225 * params.rotor_area * wind_speed.powi(3) * params.power_coefficient;
            power.min(params.rated_power)
        } else {
            params.rated_power
        }
    }
    
    fn calculate_solar_power_at_irradiance(&self, irradiance: f64) -> f64 {
        let params = &self.solar_panel_params;
        let temperature_factor = 1.0 + params.temperature_coefficient * (self.temperature - 25.0);
        let power = irradiance * params.area * params.efficiency * temperature_factor;
        
        power.min(params.rated_power)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyMarketModel {
    pub generators: HashMap<String, Generator>,
    pub demand: f64,
    pub market_price: f64,
    pub market_type: MarketType,
    pub trading_periods: Vec<TradingPeriod>,
    pub price_forecast: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketType {
    DayAhead,
    RealTime,
    AncillaryServices,
    Capacity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPeriod {
    pub start_time: i64,
    pub end_time: i64,
    pub demand: f64,
    pub price: f64,
    pub cleared_volume: f64,
}

impl EnergyMarketModel {
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
            demand: 0.0,
            market_price: 0.0,
            market_type: MarketType::DayAhead,
            trading_periods: Vec::new(),
            price_forecast: Vec::new(),
        }
    }
    
    pub fn add_generator(&mut self, generator: Generator) {
        self.generators.insert(generator.bus_id.clone(), generator);
    }
    
    pub fn market_clearing(&mut self) -> (f64, HashMap<String, f64>) {
        let mut sorted_generators: Vec<_> = self.generators.iter().collect();
        sorted_generators.sort_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
        
        let mut dispatch = HashMap::new();
        let mut remaining_demand = self.demand;
        let mut clearing_price = 0.0;
        
        for (bus_id, generator) in sorted_generators {
            if remaining_demand > 0.0 {
                let allocated_power = remaining_demand.min(generator.max_power);
                dispatch.insert(bus_id.clone(), allocated_power);
                remaining_demand -= allocated_power;
                clearing_price = generator.cost;
            } else {
                dispatch.insert(bus_id.clone(), 0.0);
            }
        }
        
        self.market_price = clearing_price;
        (clearing_price, dispatch)
    }
    
    pub fn calculate_total_cost(&self, dispatch: &HashMap<String, f64>) -> f64 {
        dispatch.iter().map(|(bus_id, power)| {
            if let Some(generator) = self.generators.get(bus_id) {
                generator.cost * power
            } else {
                0.0
            }
        }).sum()
    }
    
    pub fn generate_price_forecast(&mut self, horizon: usize) {
        let mut forecast = Vec::new();
        
        for t in 0..horizon {
            let base_price = 50.0;
            let daily_pattern = (2.0 * std::f64::consts::PI * t as f64 / 24.0).sin() * 20.0;
            let weekly_pattern = (2.0 * std::f64::consts::PI * t as f64 / 168.0).sin() * 10.0;
            let random_variation = (t as f64 * 0.1).sin() * 5.0;
            
            let price = base_price + daily_pattern + weekly_pattern + random_variation;
            forecast.push(price.max(10.0));
        }
        
        self.price_forecast = forecast;
    }
    
    pub fn optimize_trading_strategy(&self, forecast_horizon: usize) -> HashMap<String, Vec<f64>> {
        let mut strategy = HashMap::new();
        
        for (gen_id, generator) in &self.generators {
            let mut schedule = Vec::new();
            
            for t in 0..forecast_horizon {
                let price = if t < self.price_forecast.len() {
                    self.price_forecast[t]
                } else {
                    50.0
                };
                
                // 简化的交易策略：价格高时多发电
                let power_factor = if price > 60.0 { 0.8 } else if price > 40.0 { 0.6 } else { 0.3 };
                let power = generator.max_power * power_factor;
                
                schedule.push(power);
            }
            
            strategy.insert(gen_id.clone(), schedule);
        }
        
        strategy
    }
}

## 8.3.7 储能系统模型 / Energy Storage Models

### 电池储能模型 / Battery Storage Models

**荷电状态**: $SOC(t) = SOC(t-1) + \frac{P_{charge}(t) - P_{discharge}(t)}{E_{rated}} \cdot \Delta t$

**功率约束**: $P_{min} \leq P(t) \leq P_{max}$

**容量约束**: $SOC_{min} \leq SOC(t) \leq SOC_{max}$

**效率模型**: $\eta_{charge} = \eta_{charge}^0 \cdot (1 - \alpha_{SOC} \cdot SOC(t))$

**放电深度**: $DOD = 1 - SOC$

**充放电效率**: $\eta_{charge} \cdot \eta_{discharge} = 1$

### 抽水蓄能模型 / Pumped Hydro Storage Models

**水头关系**: $H = H_{upper} - H_{lower}$

**功率关系**: $P_{turbine} = \eta_t \cdot \rho \cdot g \cdot Q \cdot H$

**抽水功率**: $P_{pump} = \frac{\rho \cdot g \cdot Q \cdot H}{\eta_p}$

**水库容量**: $V(t+1) = V(t) + Q_{in}(t) - Q_{out}(t)$

**水头损失**: $H_{loss} = K \cdot Q^2$

**效率曲线**: $\eta = f(Q, H)$

### 飞轮储能模型 / Flywheel Storage Models

**动能**: $E = \frac{1}{2} J \omega^2$

**功率**: $P = J \omega \frac{d\omega}{dt}$

**转速约束**: $\omega_{min} \leq \omega \leq \omega_{max}$

**转矩约束**: $T_{min} \leq T \leq T_{max}$

**效率**: $\eta = \frac{E_{out}}{E_{in}}$

---

## 8.3.8 微电网模型 / Microgrid Models

### 微电网架构 / Microgrid Architecture

**主从控制**: $P_{master} = \sum_{i=1}^n P_{slave_i}$

**对等控制**: $P_i = P_{ref} + \Delta P_i$

**下垂控制**: $f = f_{nom} - R \cdot P$

**频率控制**: $\Delta f = \frac{\Delta P}{2H \cdot f_{nom}}$

**电压控制**: $\Delta V = \frac{R \cdot \Delta P + X \cdot \Delta Q}{V}$

**功率平衡**: $\sum P_{gen} = \sum P_{load} + P_{loss}$

### 孤岛运行 / Island Operation

**频率控制**: $\Delta f = \frac{\Delta P}{2H \cdot f_{nom}}$

**电压控制**: $\Delta V = \frac{R \cdot \Delta P + X \cdot \Delta Q}{V}$

**功率平衡**: $\sum P_{gen} = \sum P_{load} + P_{loss}$

### 并网切换 / Grid Connection

**同步条件**: $|\Delta f| < \epsilon_f, |\Delta V| < \epsilon_V, |\Delta \theta| < \epsilon_\theta$

**切换逻辑**: $S_{switch} = f(\Delta f, \Delta V, \Delta \theta, t)$

---

## 8.3.9 电力市场交易模型 / Power Market Trading Models

### 日前市场 / Day-Ahead Market

**竞价函数**: $C_i(P_i) = a_i + b_i P_i + c_i P_i^2$

**市场出清**: $\min \sum_{i=1}^n C_i(P_i)$

**约束条件**: $\sum_{i=1}^n P_i = D, P_{i,min} \leq P_i \leq P_{i,max}$

**实时市场 / Real-Time Market

**偏差结算**: $C_{deviation} = |P_{actual} - P_{scheduled}| \cdot \lambda_{RT}$

**平衡机制**: $\sum_{i=1}^n P_{up,i} - \sum_{j=1}^m P_{down,j} = \Delta D$

### 辅助服务市场 / Ancillary Services Market

**调频服务**: $P_{regulation} = \sum_{i=1}^n P_{reg,i}$

**备用容量**: $P_{reserve} = \sum_{i=1}^n P_{res,i}$

**黑启动**: $C_{blackstart} = \sum_{i=1}^n C_{bs,i}$

---

## 8.3.10 智能电网模型 / Smart Grid Models

### 需求响应 / Demand Response

**负荷转移**: $P_{shift}(t) = \sum_{i=1}^n P_{i,shift}(t)$

**负荷削减**: $P_{curtail}(t) = \sum_{i=1}^n P_{i,curtail}(t)$

**价格响应**: $P_{response} = f(\lambda(t), P_{base})$

### 分布式能源 / Distributed Energy Resources

**光伏发电**: $P_{PV}(t) = P_{rated} \cdot \frac{G(t)}{G_{STC}} \cdot \eta(t)$

**小型风电**: $P_{wind}(t) = \frac{1}{2} \rho A v^3(t) C_p(\lambda)$

**燃料电池**: $P_{FC}(t) = \eta_{FC} \cdot H_2(t) \cdot LHV_{H2}$

### 电动汽车 / Electric Vehicles

**充电负荷**: $P_{charge}(t) = \sum_{i=1}^n P_{i,charge}(t)$

**V2G功率**: $P_{V2G}(t) = \sum_{i=1}^n P_{i,V2G}(t)$

**电池状态**: $SOC_{EV}(t) = SOC_{EV}(t-1) + \frac{P_{charge}(t) - P_{V2G}(t)}{E_{EV}} \cdot \Delta t$

---

## 8.3.11 电力系统可靠性模型 / Power System Reliability Models

### 负荷点可靠性 / Load Point Reliability

**平均故障率**: $\lambda = \sum_{i=1}^n \lambda_i$

**平均修复时间**: $r = \frac{\sum_{i=1}^n \lambda_i r_i}{\sum_{i=1}^n \lambda_i}$

**可用率**: $A = \frac{MTTF}{MTTF + MTTR}$

### 系统可靠性 / System Reliability

**负荷损失概率**: $LOLP = \sum_{i \in S} P_i$

**期望负荷损失**: $EENS = \sum_{i \in S} P_i \cdot C_i$

**系统平均停电频率**: $SAIFI = \frac{\sum_{i=1}^N \lambda_i N_i}{\sum_{i=1}^N N_i}$

**系统平均停电持续时间**: $SAIDI = \frac{\sum_{i=1}^N U_i N_i}{\sum_{i=1}^N N_i}$

### 风险评估 / Risk Assessment

**风险指标**: $Risk = \sum_{i=1}^n P_i \cdot C_i$

**风险价值**: $VaR = \inf\{l \in \mathbb{R}: P(L > l) \leq \alpha\}$

**条件风险价值**: $CVaR = E[L|L > VaR]$

---

## 8.3.12 电力系统优化模型 / Power System Optimization Models

### 机组组合优化 / Unit Commitment Optimization

**目标函数**: $\min \sum_{t=1}^T \sum_{i=1}^N [C_i(P_{i,t}) + SU_i \cdot y_{i,t} + SD_i \cdot z_{i,t}]$

**功率平衡**: $\sum_{i=1}^N P_{i,t} = D_t, \forall t$

**最小运行时间**: $\sum_{k=t}^{t+UT_i-1} y_{i,k} \geq UT_i \cdot (y_{i,t} - y_{i,t-1})$

**最小停机时间**: $\sum_{k=t}^{t+DT_i-1} z_{i,k} \geq DT_i \cdot (z_{i,t} - z_{i,t-1})$

### 经济调度优化 / Economic Dispatch Optimization

**目标函数**: $\min \sum_{i=1}^N C_i(P_i)$

**功率约束**: $\sum_{i=1}^N P_i = D$

**机组约束**: $P_{i,min} \leq P_i \leq P_{i,max}$

**爬坡约束**: $|P_i(t) - P_i(t-1)| \leq R_i$

### 网络扩展规划 / Network Expansion Planning

**目标函数**: $\min \sum_{i \in \Omega} c_i x_i + \sum_{t=1}^T \sum_{s \in S} \pi_s C_{operation}(t,s)$

**容量约束**: $P_{ij} \leq P_{ij}^{max} \cdot x_i$

**功率平衡**: $\sum_{j \in \delta(i)} P_{ij} = P_i^{gen} - P_i^{load}$

---

## 8.3.13 高级实现示例 / Advanced Implementation Examples

### Python实现示例 / Python Implementation Example

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple

class AdvancedPowerSystemModel:
    def __init__(self):
        self.buses = {}
        self.lines = {}
        self.generators = {}
        self.loads = {}
        self.storage_systems = {}
        
    def add_bus(self, bus_id: str, bus_type: str, voltage: float = 1.0):
        """添加母线"""
        self.buses[bus_id] = {
            'type': bus_type,
            'voltage': voltage,
            'angle': 0.0,
            'generators': [],
            'loads': [],
            'connected_lines': []
        }
    
    def add_generator(self, gen_id: str, bus_id: str, capacity: float, 
                     min_power: float = 0.0, cost: float = 0.0):
        """添加发电机"""
        self.generators[gen_id] = {
            'bus_id': bus_id,
            'capacity': capacity,
            'min_power': min_power,
            'max_power': capacity,
            'cost': cost,
            'status': 'off'  # off, on, committed
        }
        if bus_id in self.buses:
            self.buses[bus_id]['generators'].append(gen_id)
    
    def add_load(self, load_id: str, bus_id: str, active_power: float, 
                reactive_power: float = 0.0):
        """添加负荷"""
        self.loads[load_id] = {
            'bus_id': bus_id,
            'active_power': active_power,
            'reactive_power': reactive_power,
            'type': 'constant'  # constant, variable, responsive
        }
        if bus_id in self.buses:
            self.buses[bus_id]['loads'].append(load_id)
    
    def add_storage(self, storage_id: str, bus_id: str, capacity: float,
                   max_power: float, efficiency: float = 0.9):
        """添加储能系统"""
        self.storage_systems[storage_id] = {
            'bus_id': bus_id,
            'capacity': capacity,
            'max_power': max_power,
            'efficiency': efficiency,
            'soc': 0.5,  # 初始荷电状态
            'min_soc': 0.1,
            'max_soc': 0.9
        }
    
    def economic_dispatch(self, total_demand: float) -> Dict[str, float]:
        """经济调度优化"""
        def objective(x):
            total_cost = 0
            for i, gen_id in enumerate(self.generators.keys()):
                gen = self.generators[gen_id]
                # 二次成本函数
                cost = gen['cost'] * x[i] + 0.1 * x[i]**2
                total_cost += cost
            return total_cost
        
        def constraint_power_balance(x):
            return sum(x) - total_demand
        
        def constraint_generator_limits(x):
            constraints = []
            for i, gen_id in enumerate(self.generators.keys()):
                gen = self.generators[gen_id]
                constraints.append(x[i] - gen['max_power'])  # 上限
                constraints.append(gen['min_power'] - x[i])  # 下限
            return constraints
        
        # 初始猜测值
        n_generators = len(self.generators)
        x0 = [total_demand / n_generators] * n_generators
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': constraint_power_balance},
        ]
        
        bounds = []
        for gen_id in self.generators.keys():
            gen = self.generators[gen_id]
            bounds.append((gen['min_power'], gen['max_power']))
        
        # 优化求解
        result = minimize(objective, x0, method='SLSQP', 
                        constraints=constraints, bounds=bounds)
        
        if result.success:
            dispatch = {}
            for i, gen_id in enumerate(self.generators.keys()):
                dispatch[gen_id] = result.x[i]
            return dispatch
        else:
            raise ValueError("经济调度优化失败")
    
    def unit_commitment(self, demand_profile: List[float], 
                       time_periods: int = 24) -> Dict[str, List[bool]]:
        """机组组合优化"""
        # 简化的机组组合算法
        commitment = {}
        
        for gen_id in self.generators.keys():
            gen = self.generators[gen_id]
            commitment[gen_id] = []
            
            for t in range(time_periods):
                # 简单的启停逻辑：如果需求大于某个阈值，则开机
                if demand_profile[t] > gen['capacity'] * 0.3:
                    commitment[gen_id].append(True)
                else:
                    commitment[gen_id].append(False)
        
        return commitment
    
    def calculate_reliability_indices(self) -> Dict[str, float]:
        """计算可靠性指标"""
        total_load = sum(load['active_power'] for load in self.loads.values())
        total_capacity = sum(gen['capacity'] for gen in self.generators.values())
        
        # 简化的可靠性计算
        reserve_margin = (total_capacity - total_load) / total_load
        loss_of_load_probability = max(0, 1 - reserve_margin) if reserve_margin < 0.1 else 0
        
        return {
            'reserve_margin': reserve_margin,
            'loss_of_load_probability': loss_of_load_probability,
            'total_capacity': total_capacity,
            'total_load': total_load
        }
    
    def storage_optimization(self, price_profile: List[float], 
                           time_periods: int = 24) -> Dict[str, List[float]]:
        """储能系统优化调度"""
        storage_schedule = {}
        
        for storage_id in self.storage_systems.keys():
            storage = self.storage_systems[storage_id]
            schedule = []
            soc = storage['soc']
            
            for t in range(time_periods):
                # 简化的储能调度策略
                if t < len(price_profile):
                    price = price_profile[t]
                    avg_price = np.mean(price_profile)
                    
                    if price < avg_price * 0.8:  # 低价时充电
                        power = min(storage['max_power'], 
                                  (storage['max_soc'] - soc) * storage['capacity'])
                        soc += power * storage['efficiency'] / storage['capacity']
                    elif price > avg_price * 1.2:  # 高价时放电
                        power = -min(storage['max_power'], 
                                   (soc - storage['min_soc']) * storage['capacity'])
                        soc += power / storage['efficiency'] / storage['capacity']
                    else:
                        power = 0
                    
                    soc = max(storage['min_soc'], min(storage['max_soc'], soc))
                    schedule.append(power)
            
            storage_schedule[storage_id] = schedule
        
        return storage_schedule

class RenewableEnergyForecast:
    def __init__(self):
        self.wind_forecast = {}
        self.solar_forecast = {}
        self.load_forecast = {}
    
    def generate_wind_forecast(self, location: str, time_periods: int = 24) -> List[float]:
        """生成风电预测"""
        # 简化的风电预测模型
        base_wind = 8.0  # 基础风速
        daily_pattern = np.sin(np.linspace(0, 2*np.pi, time_periods)) * 2
        random_variation = np.random.normal(0, 1, time_periods)
        
        wind_speed = base_wind + daily_pattern + random_variation
        wind_speed = np.maximum(wind_speed, 0)  # 风速不能为负
        
        # 转换为功率
        wind_power = []
        for speed in wind_speed:
            if speed < 3:  # 切入风速
                power = 0
            elif speed < 12:  # 额定风速
                power = 1000 * ((speed - 3) / 9)**3
            elif speed < 25:  # 切出风速
                power = 1000
            else:
                power = 0
            wind_power.append(power)
        
        return wind_power
    
    def generate_solar_forecast(self, location: str, time_periods: int = 24) -> List[float]:
        """生成光伏预测"""
        # 简化的光伏预测模型
        max_irradiance = 1000  # W/m²
        daily_pattern = np.maximum(0, np.sin(np.linspace(0, np.pi, time_periods)))
        weather_factor = np.random.uniform(0.7, 1.0, time_periods)
        
        irradiance = max_irradiance * daily_pattern * weather_factor
        solar_power = irradiance * 0.15  # 假设15%效率
        
        return solar_power.tolist()
    
    def generate_load_forecast(self, base_load: float, time_periods: int = 24) -> List[float]:
        """生成负荷预测"""
        # 简化的负荷预测模型
        daily_pattern = 1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, time_periods))
        weekly_pattern = 1 + 0.1 * np.sin(np.linspace(0, 2*np.pi, time_periods/7))
        random_variation = np.random.normal(0, 0.05, time_periods)
        
        load_profile = base_load * daily_pattern * weekly_pattern * (1 + random_variation)
        return load_profile.tolist()

# 使用示例
def main():
    # 创建电力系统模型
    power_system = AdvancedPowerSystemModel()
    
    # 添加母线
    power_system.add_bus("Bus1", "slack")
    power_system.add_bus("Bus2", "PQ")
    power_system.add_bus("Bus3", "PV")
    
    # 添加发电机
    power_system.add_generator("Gen1", "Bus1", 500, 50, 30)
    power_system.add_generator("Gen2", "Bus3", 300, 30, 45)
    
    # 添加负荷
    power_system.add_load("Load1", "Bus2", 200, 100)
    power_system.add_load("Load2", "Bus3", 150, 75)
    
    # 添加储能系统
    power_system.add_storage("Storage1", "Bus2", 100, 50)
    
    # 经济调度
    total_demand = 350
    dispatch = power_system.economic_dispatch(total_demand)
    print("经济调度结果:", dispatch)
    
    # 可靠性分析
    reliability = power_system.calculate_reliability_indices()
    print("可靠性指标:", reliability)
    
    # 可再生能源预测
    forecast = RenewableEnergyForecast()
    wind_power = forecast.generate_wind_forecast("Location1")
    solar_power = forecast.generate_solar_forecast("Location1")
    load_profile = forecast.generate_load_forecast(350)
    
    print("风电预测:", wind_power[:6])
    print("光伏预测:", solar_power[:6])
    print("负荷预测:", load_profile[:6])

if __name__ == "__main__":
    main()
```

### Julia实现示例 / Julia Implementation Example

```julia
using JuMP
using Ipopt
using LinearAlgebra
using Statistics

# 电力系统优化模型
struct PowerSystemOptimization
    buses::Dict{String, Dict}
    generators::Dict{String, Dict}
    loads::Dict{String, Dict}
    lines::Dict{String, Dict}
end

function PowerSystemOptimization()
    PowerSystemOptimization(
        Dict{String, Dict}(),
        Dict{String, Dict}(),
        Dict{String, Dict}(),
        Dict{String, Dict}()
    )
end

function add_bus!(system::PowerSystemOptimization, bus_id::String, bus_type::String)
    system.buses[bus_id] = Dict(
        "type" => bus_type,
        "voltage" => 1.0,
        "angle" => 0.0
    )
end

function add_generator!(system::PowerSystemOptimization, gen_id::String, bus_id::String, 
                       capacity::Float64, min_power::Float64=0.0, cost::Float64=0.0)
    system.generators[gen_id] = Dict(
        "bus_id" => bus_id,
        "capacity" => capacity,
        "min_power" => min_power,
        "max_power" => capacity,
        "cost" => cost
    )
end

function add_load!(system::PowerSystemOptimization, load_id::String, bus_id::String, 
                  active_power::Float64, reactive_power::Float64=0.0)
    system.loads[load_id] = Dict(
        "bus_id" => bus_id,
        "active_power" => active_power,
        "reactive_power" => reactive_power
    )
end

function economic_dispatch(system::PowerSystemOptimization, total_demand::Float64)
    model = Model(Ipopt.Optimizer)
    
    # 决策变量：各发电机出力
    gen_ids = collect(keys(system.generators))
    @variable(model, p[gen_ids] >= 0)
    
    # 目标函数：最小化总成本
    @objective(model, Min, sum(system.generators[gen_id]["cost"] * p[gen_id] 
                               for gen_id in gen_ids))
    
    # 功率平衡约束
    @constraint(model, sum(p[gen_id] for gen_id in gen_ids) == total_demand)
    
    # 发电机容量约束
    for gen_id in gen_ids
        gen = system.generators[gen_id]
        @constraint(model, p[gen_id] >= gen["min_power"])
        @constraint(model, p[gen_id] <= gen["max_power"])
    end
    
    # 求解
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        dispatch = Dict(gen_id => value(p[gen_id]) for gen_id in gen_ids)
        return dispatch, objective_value(model)
    else
        error("经济调度优化失败")
    end
end

function unit_commitment(system::PowerSystemOptimization, demand_profile::Vector{Float64})
    model = Model(Ipopt.Optimizer)
    
    gen_ids = collect(keys(system.generators))
    T = length(demand_profile)
    
    # 决策变量
    @variable(model, p[gen_ids, 1:T] >= 0)  # 出力
    @variable(model, u[gen_ids, 1:T], Bin)   # 启停状态
    
    # 目标函数：最小化总成本
    @objective(model, Min, sum(
        system.generators[gen_id]["cost"] * p[gen_id, t] + 
        100 * u[gen_id, t]  # 启动成本
        for gen_id in gen_ids, t in 1:T
    ))
    
    # 功率平衡约束
    for t in 1:T
        @constraint(model, sum(p[gen_id, t] for gen_id in gen_ids) == demand_profile[t])
    end
    
    # 发电机容量约束
    for gen_id in gen_ids, t in 1:T
        gen = system.generators[gen_id]
        @constraint(model, p[gen_id, t] <= gen["max_power"] * u[gen_id, t])
        @constraint(model, p[gen_id, t] >= gen["min_power"] * u[gen_id, t])
    end
    
    # 最小运行时间约束（简化）
    for gen_id in gen_ids, t in 2:T
        @constraint(model, u[gen_id, t] >= u[gen_id, t-1] - 0.5)
    end
    
    # 求解
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        commitment = Dict(gen_id => [value(u[gen_id, t]) for t in 1:T] for gen_id in gen_ids)
        dispatch = Dict(gen_id => [value(p[gen_id, t]) for t in 1:T] for gen_id in gen_ids)
        return commitment, dispatch, objective_value(model)
    else
        error("机组组合优化失败")
    end
end

function optimal_power_flow(system::PowerSystemOptimization)
    model = Model(Ipopt.Optimizer)
    
    bus_ids = collect(keys(system.buses))
    gen_ids = collect(keys(system.generators))
    load_ids = collect(keys(system.loads))
    
    # 决策变量
    @variable(model, v[bus_ids] >= 0)  # 电压幅值
    @variable(model, θ[bus_ids])        # 电压相角
    @variable(model, p_gen[gen_ids] >= 0)  # 发电机有功功率
    @variable(model, q_gen[gen_ids])    # 发电机无功功率
    
    # 目标函数：最小化发电成本
    @objective(model, Min, sum(system.generators[gen_id]["cost"] * p_gen[gen_id] 
                               for gen_id in gen_ids))
    
    # 节点功率平衡约束（简化）
    for bus_id in bus_ids
        # 有功功率平衡
        gen_at_bus = [gen_id for gen_id in gen_ids 
                      if system.generators[gen_id]["bus_id"] == bus_id]
        load_at_bus = [load_id for load_id in load_ids 
                       if system.loads[load_id]["bus_id"] == bus_id]
        
        @constraint(model, sum(p_gen[gen_id] for gen_id in gen_at_bus) == 
                           sum(system.loads[load_id]["active_power"] for load_id in load_at_bus))
    end
    
    # 电压约束
    for bus_id in bus_ids
        @constraint(model, 0.95 <= v[bus_id] <= 1.05)
    end
    
    # 发电机容量约束
    for gen_id in gen_ids
        gen = system.generators[gen_id]
        @constraint(model, p_gen[gen_id] <= gen["max_power"])
        @constraint(model, p_gen[gen_id] >= gen["min_power"])
    end
    
    # 求解
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        results = Dict(
            "voltage" => Dict(bus_id => value(v[bus_id]) for bus_id in bus_ids),
            "angle" => Dict(bus_id => value(θ[bus_id]) for bus_id in bus_ids),
            "generation" => Dict(gen_id => value(p_gen[gen_id]) for gen_id in gen_ids),
            "objective" => objective_value(model)
        )
        return results
    else
        error("最优潮流计算失败")
    end
end

# 储能系统优化
struct StorageOptimization
    capacity::Float64
    max_power::Float64
    efficiency::Float64
    initial_soc::Float64
    min_soc::Float64
    max_soc::Float64
end

function optimize_storage(storage::StorageOptimization, 
                         price_profile::Vector{Float64}, 
                         time_periods::Int=24)
    model = Model(Ipopt.Optimizer)
    
    # 决策变量
    @variable(model, p_charge[1:time_periods] >= 0)  # 充电功率
    @variable(model, p_discharge[1:time_periods] >= 0)  # 放电功率
    @variable(model, soc[1:time_periods] >= 0)  # 荷电状态
    
    # 目标函数：最大化收益
    @objective(model, Max, sum(
        price_profile[t] * (p_discharge[t] - p_charge[t])
        for t in 1:time_periods
    ))
    
    # 储能约束
    for t in 1:time_periods
        # 荷电状态更新
        if t == 1
            @constraint(model, soc[t] == storage.initial_soc + 
                       (p_charge[t] * storage.efficiency - p_discharge[t] / storage.efficiency) / storage.capacity)
        else
            @constraint(model, soc[t] == soc[t-1] + 
                       (p_charge[t] * storage.efficiency - p_discharge[t] / storage.efficiency) / storage.capacity)
        end
        
        # 荷电状态限制
        @constraint(model, storage.min_soc <= soc[t] <= storage.max_soc)
        
        # 功率限制
        @constraint(model, p_charge[t] <= storage.max_power)
        @constraint(model, p_discharge[t] <= storage.max_power)
    end
    
    # 求解
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        results = Dict(
            "charge_power" => [value(p_charge[t]) for t in 1:time_periods],
            "discharge_power" => [value(p_discharge[t]) for t in 1:time_periods],
            "soc" => [value(soc[t]) for t in 1:time_periods],
            "profit" => objective_value(model)
        )
        return results
    else
        error("储能优化失败")
    end
end

# 使用示例
function main()
    # 创建电力系统
    system = PowerSystemOptimization()
    
    # 添加系统组件
    add_bus!(system, "Bus1", "slack")
    add_bus!(system, "Bus2", "PQ")
    
    add_generator!(system, "Gen1", "Bus1", 500, 50, 30)
    add_generator!(system, "Gen2", "Bus1", 300, 30, 45)
    
    add_load!(system, "Load1", "Bus2", 200, 100)
    add_load!(system, "Load2", "Bus2", 150, 75)
    
    # 经济调度
    dispatch, cost = economic_dispatch(system, 350)
    println("经济调度结果: ", dispatch)
    println("总成本: ", cost)
    
    # 机组组合
    demand_profile = [300, 320, 350, 380, 400, 420, 400, 380, 350, 320, 300, 280,
                      260, 240, 220, 200, 180, 200, 220, 240, 260, 280, 300, 320]
    commitment, dispatch_uc, cost_uc = unit_commitment(system, demand_profile)
    println("机组组合结果: ", commitment)
    println("机组组合成本: ", cost_uc)
    
    # 储能优化
    storage = StorageOptimization(100.0, 50.0, 0.9, 0.5, 0.1, 0.9)
    price_profile = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
                     80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25]
    storage_results = optimize_storage(storage, price_profile)
    println("储能优化结果: ", storage_results)
end

main()
```

---

## 8.3.14 应用案例 / Application Cases

### 案例1：智能微电网调度 / Case 1: Smart Microgrid Dispatch

**场景描述**: 一个包含光伏、风电、储能和柴油发电机的微电网系统

**优化目标**: 最小化运行成本，最大化可再生能源利用率

**约束条件**:

- 功率平衡约束
- 储能荷电状态约束
- 发电机启停约束
- 网络约束

**求解方法**: 混合整数线性规划

### 案例2：电力市场交易策略 / Case 2: Power Market Trading Strategy

**场景描述**: 发电企业参与日前市场和实时市场交易

**优化目标**: 最大化利润，最小化风险

**策略要素**:

- 价格预测模型
- 风险度量方法
- 投资组合优化
- 动态调整策略

### 案例3：配电网规划优化 / Case 3: Distribution Network Planning

**场景描述**: 考虑分布式能源接入的配电网扩展规划

**优化目标**: 最小化投资成本，最大化可靠性

**规划要素**:

- 线路容量扩展
- 变电站选址
- 分布式能源接入点
- 储能系统配置

---

## 8.3.15 发展趋势 / Development Trends

### 数字化转型 / Digital Transformation

- **数字孪生**: 电力系统数字孪生技术
- **人工智能**: 机器学习在电力系统中的应用
- **大数据**: 海量数据驱动的智能决策
- **云计算**: 云端电力系统仿真平台

### 新能源集成 / Renewable Energy Integration

- **高比例新能源**: 高渗透率可再生能源接入
- **虚拟电厂**: 分布式能源聚合管理
- **需求响应**: 智能负荷管理技术
- **储能技术**: 新型储能技术应用

### 市场机制创新 / Market Mechanism Innovation

- **电力现货市场**: 实时电力交易市场
- **辅助服务市场**: 调频、调峰、备用市场
- **容量市场**: 发电容量保障机制
- **碳交易市场**: 碳排放权交易

### 智能电网发展 / Smart Grid Development

- **主动配电网**: 双向功率流的智能配电网
- **微电网技术**: 独立运行的微电网系统
- **电动汽车**: 大规模电动汽车接入
- **智能电表**: 高级计量基础设施

---

## 8.3.16 人工智能与机器学习应用 / AI & ML Applications

### 机器学习在电力系统中的应用

- **负荷预测**: 使用历史数据预测未来负荷
- **可再生能源预测**: 基于天气数据预测风电和光伏发电量
- **故障诊断**: 通过图像识别和模式识别诊断电力设备故障
- **状态评估**: 评估电网运行状态和设备健康状况

### 深度学习在电力系统中的应用

- **短期负荷预测**: 使用LSTM和GRU预测短期负荷
- **中长期负荷预测**: 使用Transformer和Attention机制预测中长期负荷
- **可再生能源预测**: 使用GAN和VAE生成更精确的预测
- **故障诊断**: 使用CNN和Transformer进行故障分类和定位

### 强化学习在电力系统中的应用

- **储能优化**: 使用Q-learning和DQN优化储能系统运行
- **市场交易**: 使用RL进行电力市场交易策略优化
- **电网调度**: 使用RL进行电网调度优化

---

## 8.3.17 数字孪生技术 / Digital Twin Technology

### 数字孪生在电力系统中的应用

- **电网仿真**: 在数字空间中模拟电网运行
- **故障预测**: 通过数字孪生预测电网故障
- **规划优化**: 在数字空间中进行电网规划和优化
- **培训演练**: 通过数字孪生进行电网操作员培训

### 数字孪生技术的关键挑战

- **数据质量**: 需要高质量、多源、实时的电网数据
- **计算资源**: 需要强大的计算资源进行实时仿真
- **模型精度**: 需要高精度的物理模型和数据驱动模型
- **安全性**: 需要确保数字孪生系统的安全性

---

## 8.3.18 区块链在电力系统中的应用 / Blockchain in Power Systems

### 区块链在电力系统中的应用

- **数据共享**: 通过区块链实现电网数据的安全共享
- **交易结算**: 使用智能合约实现电力交易和结算
- **能源溯源**: 通过区块链追踪能源从生产到消费的全过程
- **智能合约**: 使用智能合约执行复杂的电力市场规则

### 区块链在电力系统中的挑战

- **性能**: 区块链的性能可能无法满足实时电力系统的需求
- **可扩展性**: 当前区块链网络的容量和吞吐量有限
- **安全性**: 需要确保区块链网络的安全性和抗攻击能力
- **互操作性**: 不同区块链网络之间的互操作性问题

---

## 8.3.19 量子计算在电力系统优化中的应用 / Quantum Computing in Power Systems

### 量子计算在电力系统优化中的应用

- **大规模优化问题**: 使用量子退火和量子模拟解决大规模电力系统优化问题
- **组合优化**: 使用量子遗传算法和量子启发式算法解决组合优化问题
- **机器学习**: 使用量子神经网络和量子支持向量机提高机器学习模型的性能

### 量子计算在电力系统中的挑战

- **硬件**: 当前量子计算机的硬件规模和可靠性有限
- **算法**: 需要开发适用于量子计算机的优化算法
- **互操作性**: 量子计算与经典计算的互操作性问题
- **成本**: 量子计算的运行成本远高于经典计算

---

## 8.3.20 边缘计算与物联网 / Edge Computing & IoT

### 边缘计算与物联网在电力系统中的应用

- **实时监控**: 通过边缘计算和物联网实时监控电网运行状态
- **故障快速定位**: 使用边缘计算进行故障快速定位和隔离
- **本地优化**: 在边缘节点进行本地负荷预测和优化
- **数据本地化**: 保护电网数据的安全性和隐私性

### 边缘计算与物联网的挑战

- **延迟**: 边缘计算的延迟可能无法满足实时控制的需求
- **带宽**: 边缘节点与云端之间的带宽有限
- **可靠性**: 边缘计算节点的可靠性和稳定性需要保证
- **安全性**: 边缘计算和物联网的安全性问题

---

## 8.3.21 高级应用案例 / Advanced Application Cases

### 案例4：大规模电力系统优化 / Case 4: Large-Scale Power System Optimization

**场景描述**: 省级电网的实时优化调度

**优化目标**:

- 最小化总运行成本
- 最大化系统可靠性
- 最小化环境污染

**约束条件**:

- 功率平衡约束
- 发电机容量约束
- 网络约束
- 环保约束

**求解方法**: 混合整数二次规划 (MIQP)

### 案例5：智能配电网自愈 / Case 5: Smart Distribution Network Self-Healing

**场景描述**: 配电网故障后的自动恢复

**自愈策略**:

- 故障定位与隔离
- 负荷转移
- 网络重构
- 电压恢复

**关键技术**:

- 人工智能故障诊断
- 实时网络重构算法
- 智能开关控制

### 案例6：虚拟电厂聚合优化 / Case 6: Virtual Power Plant Aggregation Optimization

**场景描述**: 分布式能源的聚合管理

**聚合策略**:

- 负荷聚合
- 发电聚合
- 储能聚合
- 需求响应聚合

**优化目标**:

- 最大化聚合效益
- 最小化聚合成本
- 最大化系统稳定性

---

## 8.3.22 前沿技术实现 / Cutting-Edge Technology Implementation

### 量子机器学习实现 / Quantum Machine Learning Implementation

```python
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import CircuitQNN

class QuantumPowerSystemOptimizer:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_circuit(self, parameters):
        """创建量子电路"""
        qc = QuantumCircuit(self.num_qubits)
        
        # 编码参数
        for i in range(self.num_qubits):
            qc.rx(parameters[i], i)
            qc.rz(parameters[i + self.num_qubits], i)
        
        # 纠缠层
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.num_qubits - 1, 0)
        
        # 测量
        qc.measure_all()
        
        return qc
    
    def objective_function(self, parameters):
        """目标函数：最小化发电成本"""
        qc = self.create_quantum_circuit(parameters)
        
        # 执行量子电路
        job = execute(qc, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # 计算期望值
        expectation = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # 将比特串转换为发电量
            generation = self.bitstring_to_generation(bitstring)
            cost = self.calculate_generation_cost(generation)
            expectation += cost * count / total_shots
        
        return expectation
    
    def bitstring_to_generation(self, bitstring):
        """将量子比特串转换为发电量"""
        generation = []
        for i in range(0, len(bitstring), 2):
            if i + 1 < len(bitstring):
                # 使用两个比特表示一个发电机的状态
                state = int(bitstring[i:i+2], 2)
                if state == 0:
                    generation.append(0.0)  # 停机
                elif state == 1:
                    generation.append(0.5)  # 半负荷
                elif state == 2:
                    generation.append(0.8)  # 高负荷
                else:
                    generation.append(1.0)  # 满负荷
        return generation
    
    def calculate_generation_cost(self, generation):
        """计算发电成本"""
        total_cost = 0
        for i, gen in enumerate(generation):
            # 简化的成本函数
            cost = gen * (10 + i * 5)  # 基础成本 + 递增成本
            total_cost += cost
        return total_cost
    
    def optimize(self, initial_parameters=None):
        """量子优化"""
        if initial_parameters is None:
            initial_parameters = np.random.random(2 * self.num_qubits)
        
        optimizer = SPSA(maxiter=100)
        
        def objective(params):
            return self.objective_function(params)
        
        result = optimizer.minimize(objective, initial_parameters)
        return result.x, result.fun

class QuantumLoadForecaster:
    def __init__(self, num_qubits=6):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_neural_network(self, input_size, output_size):
        """创建量子神经网络"""
        # 输入层
        input_circuit = QuantumCircuit(input_size)
        
        # 隐藏层
        hidden_circuit = TwoLocal(input_size, ['ry', 'rz'], 'cz', 
                                 reps=2, entanglement='circular')
        
        # 输出层
        output_circuit = QuantumCircuit(output_size)
        
        # 组合电路
        qnn = CircuitQNN(hidden_circuit, input_params=None, 
                         weight_params=hidden_circuit.parameters,
                         input_gradients=False)
        
        return qnn
    
    def quantum_load_forecast(self, historical_data, forecast_horizon=24):
        """量子负荷预测"""
        # 数据预处理
        X, y = self.prepare_data(historical_data)
        
        # 创建量子神经网络
        qnn = self.create_quantum_neural_network(len(X[0]), 1)
        
        # 训练量子神经网络
        weights = self.train_quantum_network(qnn, X, y)
        
        # 预测
        forecast = self.predict(qnn, weights, X[-forecast_horizon:])
        
        return forecast
    
    def prepare_data(self, historical_data):
        """数据预处理"""
        X = []
        y = []
        
        for i in range(len(historical_data) - 24):
            # 使用24小时的历史数据预测下一小时
            features = historical_data[i:i+24]
            target = historical_data[i+24]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_quantum_network(self, qnn, X, y):
        """训练量子神经网络"""
        # 简化的训练过程
        num_weights = len(qnn.parameters)
        initial_weights = np.random.random(num_weights)
        
        def loss_function(weights):
            predictions = []
            for x in X:
                pred = qnn.forward(x, weights)
                predictions.append(pred)
            
            # 计算均方误差
            mse = np.mean((predictions - y) ** 2)
            return mse
        
        # 使用经典优化器训练
        optimizer = SPSA(maxiter=50)
        result = optimizer.minimize(loss_function, initial_weights)
        
        return result.x
    
    def predict(self, qnn, weights, X):
        """预测"""
        predictions = []
        for x in X:
            pred = qnn.forward(x, weights)
            predictions.append(pred)
        
        return predictions

# 使用示例
def quantum_power_system_example():
    # 量子电力系统优化
    optimizer = QuantumPowerSystemOptimizer(num_qubits=4)
    optimal_params, optimal_cost = optimizer.optimize()
    print(f"量子优化结果: 参数={optimal_params}, 成本={optimal_cost}")
    
    # 量子负荷预测
    forecaster = QuantumLoadForecaster(num_qubits=6)
    
    # 模拟历史负荷数据
    historical_load = np.random.normal(1000, 200, 1000)
    forecast = forecaster.quantum_load_forecast(historical_load, forecast_horizon=24)
    print(f"量子负荷预测结果: {forecast[:5]}")

if __name__ == "__main__":
    quantum_power_system_example()
```

### 联邦学习在电力系统中的应用 / Federated Learning in Power Systems

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple

class FederatedPowerSystemModel:
    def __init__(self, model_structure: Dict):
        self.model_structure = model_structure
        self.global_model = self.create_model()
        self.local_models = []
        
    def create_model(self) -> nn.Module:
        """创建神经网络模型"""
        model = nn.Sequential(
            nn.Linear(self.model_structure['input_size'], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.model_structure['output_size'])
        )
        return model
    
    def federated_training(self, local_datasets: List[Tuple], 
                          num_rounds: int = 10, 
                          local_epochs: int = 5) -> Dict:
        """联邦学习训练"""
        training_history = {
            'global_loss': [],
            'local_losses': [],
            'communication_rounds': []
        }
        
        for round_idx in range(num_rounds):
            print(f"联邦学习轮次 {round_idx + 1}/{num_rounds}")
            
            # 分发全局模型到本地
            local_models = self.distribute_global_model()
            
            # 本地训练
            local_losses = []
            for client_idx, (local_data, local_model) in enumerate(zip(local_datasets, local_models)):
                local_loss = self.local_training(local_model, local_data, local_epochs)
                local_losses.append(local_loss)
            
            # 聚合本地模型
            self.aggregate_local_models(local_models)
            
            # 记录训练历史
            training_history['global_loss'].append(np.mean(local_losses))
            training_history['local_losses'].append(local_losses)
            training_history['communication_rounds'].append(round_idx)
            
            print(f"轮次 {round_idx + 1} 完成，平均损失: {np.mean(local_losses):.4f}")
        
        return training_history
    
    def distribute_global_model(self) -> List[nn.Module]:
        """分发全局模型到本地"""
        local_models = []
        for _ in range(len(self.local_models)):
            local_model = type(self.global_model)()
            local_model.load_state_dict(self.global_model.state_dict())
            local_models.append(local_model)
        return local_models
    
    def local_training(self, local_model: nn.Module, 
                      local_data: Tuple, 
                      epochs: int) -> float:
        """本地训练"""
        X, y = local_data
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=0.001)
        
        local_model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
        
        return total_loss / epochs
    
    def aggregate_local_models(self, local_models: List[nn.Module]):
        """聚合本地模型"""
        # FedAvg算法
        global_state_dict = self.global_model.state_dict()
        
        for key in global_state_dict.keys():
            # 计算所有本地模型的平均参数
            avg_param = torch.zeros_like(global_state_dict[key])
            for local_model in local_models:
                avg_param += local_model.state_dict()[key]
            avg_param /= len(local_models)
            
            global_state_dict[key] = avg_param
        
        self.global_model.load_state_dict(global_state_dict)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用全局模型进行预测"""
        self.global_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.global_model(X_tensor)
            return predictions.numpy()

class FederatedLoadForecaster(FederatedPowerSystemModel):
    def __init__(self):
        model_structure = {
            'input_size': 24,  # 24小时历史数据
            'output_size': 1   # 预测下一小时负荷
        }
        super().__init__(model_structure)
    
    def prepare_federated_data(self, num_clients: int = 5) -> List[Tuple]:
        """准备联邦学习数据"""
        local_datasets = []
        
        for client_idx in range(num_clients):
            # 为每个客户端生成不同的负荷数据
            base_load = 1000 + client_idx * 100
            time_series = np.random.normal(base_load, 200, 1000)
            
            # 创建滑动窗口数据
            X, y = self.create_sliding_window_data(time_series, window_size=24)
            
            local_datasets.append((X, y))
        
        return local_datasets
    
    def create_sliding_window_data(self, time_series: np.ndarray, 
                                  window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口数据"""
        X, y = [], []
        
        for i in range(len(time_series) - window_size):
            X.append(time_series[i:i+window_size])
            y.append(time_series[i+window_size])
        
        return np.array(X), np.array(y).reshape(-1, 1)

class FederatedRenewableEnergyPredictor(FederatedPowerSystemModel):
    def __init__(self):
        model_structure = {
            'input_size': 48,  # 48小时历史数据（天气+发电）
            'output_size': 24  # 预测未来24小时发电量
        }
        super().__init__(model_structure)
    
    def prepare_renewable_data(self, num_clients: int = 3) -> List[Tuple]:
        """准备可再生能源预测数据"""
        local_datasets = []
        
        # 风电数据
        wind_data = self.generate_wind_data(num_clients)
        local_datasets.extend(wind_data)
        
        # 光伏数据
        solar_data = self.generate_solar_data(num_clients)
        local_datasets.extend(solar_data)
        
        return local_datasets
    
    def generate_wind_data(self, num_clients: int) -> List[Tuple]:
        """生成风电数据"""
        datasets = []
        
        for client_idx in range(num_clients):
            # 模拟不同地区的风电数据
            base_wind = 8.0 + client_idx * 2.0
            wind_speed = np.random.normal(base_wind, 3.0, 1000)
            
            # 转换为发电量
            wind_power = self.wind_speed_to_power(wind_speed)
            
            # 添加天气特征
            weather_features = np.random.normal(0, 1, (len(wind_power), 24))
            
            # 组合特征
            X = np.column_stack([weather_features, wind_power[:-24]])
            y = wind_power[24:].reshape(-1, 1)
            
            datasets.append((X, y))
        
        return datasets
    
    def generate_solar_data(self, num_clients: int) -> List[Tuple]:
        """生成光伏数据"""
        datasets = []
        
        for client_idx in range(num_clients):
            # 模拟不同地区的光伏数据
            base_irradiance = 800 + client_idx * 100
            irradiance = np.random.normal(base_irradiance, 200, 1000)
            
            # 转换为发电量
            solar_power = self.irradiance_to_power(irradiance)
            
            # 添加天气特征
            weather_features = np.random.normal(0, 1, (len(solar_power), 24))
            
            # 组合特征
            X = np.column_stack([weather_features, solar_power[:-24]])
            y = solar_power[24:].reshape(-1, 1)
            
            datasets.append((X, y))
        
        return datasets
    
    def wind_speed_to_power(self, wind_speed: np.ndarray) -> np.ndarray:
        """风速转换为功率"""
        power = np.zeros_like(wind_speed)
        
        for i, speed in enumerate(wind_speed):
            if speed < 3:  # 切入风速
                power[i] = 0
            elif speed < 12:  # 额定风速
                power[i] = 1000 * ((speed - 3) / 9) ** 3
            elif speed < 25:  # 切出风速
                power[i] = 1000
            else:
                power[i] = 0
        
        return power
    
    def irradiance_to_power(self, irradiance: np.ndarray) -> np.ndarray:
        """辐照度转换为功率"""
        efficiency = 0.15
        area = 1000  # m²
        power = irradiance * efficiency * area / 1000  # 转换为MW
        return power

# 使用示例
def federated_learning_example():
    # 联邦负荷预测
    print("开始联邦负荷预测训练...")
    load_forecaster = FederatedLoadForecaster()
    local_datasets = load_forecaster.prepare_federated_data(num_clients=5)
    
    training_history = load_forecaster.federated_training(
        local_datasets, num_rounds=10, local_epochs=5
    )
    
    print("联邦负荷预测训练完成")
    print(f"最终平均损失: {training_history['global_loss'][-1]:.4f}")
    
    # 联邦可再生能源预测
    print("\n开始联邦可再生能源预测训练...")
    renewable_predictor = FederatedRenewableEnergyPredictor()
    renewable_datasets = renewable_predictor.prepare_renewable_data(num_clients=6)
    
    renewable_history = renewable_predictor.federated_training(
        renewable_datasets, num_rounds=15, local_epochs=3
    )
    
    print("联邦可再生能源预测训练完成")
    print(f"最终平均损失: {renewable_history['global_loss'][-1]:.4f}")

if __name__ == "__main__":
    federated_learning_example()
```

---

## 8.3.23 总结与展望 / Summary and Future Prospects

### 电力能源模型的发展趋势

1. **智能化**: 人工智能和机器学习在电力系统中的应用将更加广泛
2. **数字化**: 数字孪生技术将实现电网的实时仿真和优化
3. **去中心化**: 分布式能源和微电网将改变传统电力系统架构
4. **绿色化**: 可再生能源和储能技术将推动电力系统向清洁能源转型
5. **市场化**: 电力市场机制将更加完善和灵活

### 关键技术挑战

1. **大规模优化**: 处理大规模电力系统优化问题的计算复杂度
2. **不确定性建模**: 可再生能源和负荷的不确定性建模
3. **网络安全**: 电力系统网络安全和隐私保护
4. **实时性**: 满足电力系统实时控制的要求
5. **可靠性**: 确保电力系统的安全可靠运行

### 未来研究方向

1. **量子计算**: 利用量子计算解决大规模电力系统优化问题
2. **边缘计算**: 在边缘节点进行实时电力系统控制
3. **区块链**: 使用区块链技术实现去中心化的电力交易
4. **数字孪生**: 构建完整的电力系统数字孪生平台
5. **联邦学习**: 在保护隐私的前提下进行分布式机器学习

---

## 8.3.24 高级实现文件 / Advanced Implementation Files

### 相关实现文件

- [高级实现示例](./ADVANCED_IMPLEMENTATIONS.md) - 深度学习、强化学习、图神经网络等前沿技术实现
- [数字孪生技术实现](./DIGITAL_TWIN_IMPLEMENTATION.md) - 完整的数字孪生系统架构和实现
- [区块链电力交易系统](./BLOCKCHAIN_POWER_TRADING.md) - 区块链在电力系统中的应用

### 技术发展趋势总结

1. **人工智能与机器学习**
   - 深度学习在负荷预测、故障诊断中的应用
   - 强化学习在电力系统优化中的应用
   - 图神经网络在电网建模中的应用
   - 联邦学习在保护隐私前提下的分布式学习

2. **数字孪生技术**
   - 实时数据采集与处理
   - 物理模型与数据驱动模型融合
   - 预测性维护系统
   - 故障诊断与预警系统

3. **区块链技术**
   - 去中心化电力交易
   - 智能合约在电力市场中的应用
   - 能源溯源系统
   - 碳交易与绿色证书

4. **量子计算**
   - 量子机器学习在电力系统优化中的应用
   - 量子算法解决大规模优化问题

5. **边缘计算与物联网**
   - 实时电力系统控制
   - 本地数据处理和优化
   - 分布式能源管理

### 未来发展方向

1. **智能化升级**
   - 全自动化的电力系统运行
   - 智能决策支持系统
   - 自适应控制算法

2. **数字化转型**
   - 完整的数字孪生平台
   - 云端电力系统仿真
   - 大数据驱动的智能分析

3. **绿色化转型**
   - 高比例可再生能源接入
   - 碳中和技术路径
   - 绿色电力证书体系

4. **市场化改革**
   - 完善的电力市场机制
   - 灵活的定价策略
   - 创新的商业模式

5. **安全可靠**
   - 网络安全防护
   - 物理安全保护
   - 数据隐私保护

---

## 8.3.25 算法实现 / Algorithm Implementation

### 电力系统分析算法 / Power System Analysis Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

@dataclass
class Bus:
    """母线"""
    bus_id: int
    bus_type: str  # PV, PQ, Slack
    voltage_magnitude: float
    voltage_angle: float
    active_power: float
    reactive_power: float

@dataclass
class Line:
    """输电线路"""
    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    susceptance: float
    rating: float

class PowerFlowSolver:
    """潮流计算求解器"""
    
    def __init__(self, buses: List[Bus], lines: List[Line]):
        self.buses = buses
        self.lines = lines
        self.n_buses = len(buses)
        self.bus_dict = {bus.bus_id: bus for bus in buses}
        self.Y_bus = self._build_admittance_matrix()
    
    def _build_admittance_matrix(self) -> np.ndarray:
        """构建导纳矩阵"""
        Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        for line in self.lines:
            i = line.from_bus - 1  # 假设母线编号从1开始
            j = line.to_bus - 1
            y_ij = 1 / (line.resistance + 1j * line.reactance)
            
            # 对角元素
            Y[i, i] += y_ij + 1j * line.susceptance / 2
            Y[j, j] += y_ij + 1j * line.susceptance / 2
            
            # 非对角元素
            Y[i, j] -= y_ij
            Y[j, i] -= y_ij
        
        return Y
    
    def newton_raphson_power_flow(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict:
        """牛顿-拉夫森潮流计算"""
        # 初始化
        V = np.ones(self.n_buses, dtype=complex)
        for bus in self.buses:
            idx = bus.bus_id - 1
            V[idx] = bus.voltage_magnitude * np.exp(1j * bus.voltage_angle)
        
        # 分离PV和PQ母线
        pv_buses = [i for i, bus in enumerate(self.buses) if bus.bus_type == 'PV']
        pq_buses = [i for i, bus in enumerate(self.buses) if bus.bus_type == 'PQ']
        slack_bus = [i for i, bus in enumerate(self.buses) if bus.bus_type == 'Slack'][0]
        
        # 迭代求解
        for iteration in range(max_iterations):
            # 计算功率不平衡量
            S_calc = V * np.conj(self.Y_bus @ V)
            P_calc = S_calc.real
            Q_calc = S_calc.imag
            
            # 计算功率不平衡量
            delta_P = np.zeros(self.n_buses)
            delta_Q = np.zeros(self.n_buses)
            
            for i in range(self.n_buses):
                if i != slack_bus:
                    bus = self.buses[i]
                    delta_P[i] = bus.active_power - P_calc[i]
                    if bus.bus_type == 'PQ':
                        delta_Q[i] = bus.reactive_power - Q_calc[i]
            
            # 检查收敛性
            max_mismatch = max(np.max(np.abs(delta_P)), np.max(np.abs(delta_Q)))
            if max_mismatch < tolerance:
                break
            
            # 构建雅可比矩阵
            J = self._build_jacobian_matrix(V, pv_buses, pq_buses, slack_bus)
            
            # 求解修正量
            delta_x = np.linalg.solve(J, np.concatenate([delta_P[delta_P != 0], delta_Q[delta_Q != 0]]))
            
            # 更新状态变量
            idx = 0
            for i in range(self.n_buses):
                if i != slack_bus:
                    V[i] *= np.exp(1j * delta_x[idx])
                    idx += 1
                    if self.buses[i].bus_type == 'PQ':
                        V[i] *= (1 + delta_x[idx])
                        idx += 1
        
        # 计算最终结果
        S_final = V * np.conj(self.Y_bus @ V)
        P_final = S_final.real
        Q_final = S_final.imag
        
        return {
            'voltages': V,
            'active_power': P_final,
            'reactive_power': Q_final,
            'iterations': iteration + 1,
            'converged': max_mismatch < tolerance
        }
    
    def _build_jacobian_matrix(self, V: np.ndarray, pv_buses: List[int], 
                              pq_buses: List[int], slack_bus: int) -> np.ndarray:
        """构建雅可比矩阵"""
        # 简化的雅可比矩阵构建
        n_pq = len(pq_buses)
        n_pv = len(pv_buses)
        n = n_pq + n_pv
        
        J = np.zeros((2*n, 2*n))
        
        # 这里简化处理，实际应用中需要完整的雅可比矩阵
        for i in range(n):
            J[i, i] = 1.0
        
        return J

def fast_decoupled_power_flow(buses: List[Bus], lines: List[Line], 
                             max_iterations: int = 100, tolerance: float = 1e-6) -> Dict:
    """快速分解潮流计算"""
    solver = PowerFlowSolver(buses, lines)
    return solver.newton_raphson_power_flow(max_iterations, tolerance)
```

### 发电模型算法 / Generation Model Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class Generator:
    """发电机"""
    gen_id: str
    bus_id: int
    capacity: float  # MW
    min_output: float  # MW
    max_output: float  # MW
    cost_coefficients: List[float]  # [a, b, c] for quadratic cost function
    ramp_rate: float  # MW/hour

class EconomicDispatch:
    """经济调度"""
    
    def __init__(self, generators: List[Generator], total_demand: float):
        self.generators = generators
        self.total_demand = total_demand
    
    def quadratic_cost_function(self, generator: Generator, output: float) -> float:
        """二次成本函数"""
        a, b, c = generator.cost_coefficients
        return a * output**2 + b * output + c
    
    def solve_economic_dispatch(self) -> Dict:
        """求解经济调度"""
        n_gens = len(self.generators)
        
        # 目标函数：最小化总成本
        def objective(x):
            total_cost = 0
            for i, gen in enumerate(self.generators):
                total_cost += self.quadratic_cost_function(gen, x[i])
            return total_cost
        
        # 约束条件
        constraints = [
            # 功率平衡约束
            {'type': 'eq', 'fun': lambda x: sum(x) - self.total_demand}
        ]
        
        # 边界约束
        bounds = [(gen.min_output, gen.max_output) for gen in self.generators]
        
        # 初始解
        x0 = [gen.min_output for gen in self.generators]
        
        # 求解优化问题
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return {
                'generator_outputs': result.x,
                'total_cost': result.fun,
                'success': True
            }
        else:
            return {
                'generator_outputs': None,
                'total_cost': None,
                'success': False,
                'message': result.message
            }

class UnitCommitment:
    """机组组合"""
    
    def __init__(self, generators: List[Generator], demand_profile: List[float]):
        self.generators = generators
        self.demand_profile = demand_profile
        self.n_gens = len(generators)
        self.n_periods = len(demand_profile)
    
    def solve_unit_commitment(self) -> Dict:
        """求解机组组合（简化版）"""
        # 简化的启发式算法
        commitment = np.zeros((self.n_periods, self.n_gens), dtype=bool)
        outputs = np.zeros((self.n_periods, self.n_gens))
        
        for t, demand in enumerate(self.demand_profile):
            # 按成本排序选择机组
            sorted_gens = sorted(self.generators, 
                               key=lambda g: g.cost_coefficients[1])  # 按线性成本系数排序
            
            remaining_demand = demand
            for i, gen in enumerate(sorted_gens):
                if remaining_demand > 0:
                    commitment[t, i] = True
                    output = min(remaining_demand, gen.max_output)
                    outputs[t, i] = output
                    remaining_demand -= output
        
        return {
            'commitment': commitment,
            'outputs': outputs,
            'success': True
        }

class RenewableEnergyModel:
    """可再生能源模型"""
    
    def __init__(self, capacity: float, location: Tuple[float, float]):
        self.capacity = capacity
        self.location = location
    
    def solar_power_generation(self, time: float, weather_data: Dict) -> float:
        """太阳能发电"""
        # 简化的太阳能模型
        hour = time % 24
        solar_irradiance = weather_data.get('solar_irradiance', 0)
        
        # 日变化模式
        if 6 <= hour <= 18:
            efficiency = 0.15  # 太阳能板效率
            power = self.capacity * efficiency * solar_irradiance / 1000
        else:
            power = 0
        
        return min(power, self.capacity)
    
    def wind_power_generation(self, time: float, weather_data: Dict) -> float:
        """风力发电"""
        wind_speed = weather_data.get('wind_speed', 0)
        
        # 简化的风力发电模型
        if wind_speed < 3 or wind_speed > 25:
            power = 0
        elif 3 <= wind_speed <= 12:
            power = self.capacity * (wind_speed - 3) / 9
        else:
            power = self.capacity
        
        return power
```

### 输电网络算法 / Transmission Network Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import networkx as nx

@dataclass
class TransmissionLine:
    """输电线路"""
    line_id: str
    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    susceptance: float
    rating: float
    status: bool = True  # True: 在运行, False: 停运

class TransmissionNetwork:
    """输电网络"""
    
    def __init__(self, lines: List[TransmissionLine]):
        self.lines = lines
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.Graph:
        """构建网络图"""
        G = nx.Graph()
        for line in self.lines:
            if line.status:
                G.add_edge(line.from_bus, line.to_bus, 
                          weight=line.resistance,
                          reactance=line.reactance,
                          rating=line.rating)
        return G
    
    def calculate_line_flows(self, bus_voltages: np.ndarray, 
                           bus_angles: np.ndarray) -> Dict[str, complex]:
        """计算线路潮流"""
        flows = {}
        
        for line in self.lines:
            if line.status:
                i = line.from_bus - 1
                j = line.to_bus - 1
                
                V_i = bus_voltages[i]
                V_j = bus_voltages[j]
                theta_i = bus_angles[i]
                theta_j = bus_angles[j]
                
                # 计算线路潮流
                y_ij = 1 / (line.resistance + 1j * line.reactance)
                S_ij = V_i * np.conj(y_ij * (V_i - V_j))
                
                flows[line.line_id] = S_ij
        
        return flows
    
    def check_line_overloads(self, flows: Dict[str, complex]) -> List[str]:
        """检查线路过载"""
        overloaded_lines = []
        
        for line in self.lines:
            if line.line_id in flows:
                flow_magnitude = abs(flows[line.line_id])
                if flow_magnitude > line.rating:
                    overloaded_lines.append(line.line_id)
        
        return overloaded_lines
    
    def calculate_network_reliability(self) -> float:
        """计算网络可靠性"""
        # 简化的可靠性计算
        if not self.graph.edges():
            return 0.0
        
        # 计算连通性
        if nx.is_connected(self.graph):
            # 计算平均路径长度
            avg_path_length = nx.average_shortest_path_length(self.graph)
            # 计算聚类系数
            clustering_coeff = nx.average_clustering(self.graph)
            
            # 简化的可靠性指标
            reliability = 1.0 / (1.0 + avg_path_length) * clustering_coeff
        else:
            reliability = 0.0
        
        return reliability
    
    def contingency_analysis(self, contingency_lines: List[str]) -> Dict:
        """故障分析"""
        # 创建故障后的网络
        contingency_network = TransmissionNetwork([
            line for line in self.lines 
            if line.line_id not in contingency_lines
        ])
        
        # 检查连通性
        is_connected = nx.is_connected(contingency_network.graph)
        
        # 计算影响
        if is_connected:
            # 计算路径变化
            original_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
            contingency_paths = dict(nx.all_pairs_shortest_path_length(contingency_network.graph))
            
            path_changes = 0
            for i in original_paths:
                for j in original_paths[i]:
                    if j in contingency_paths.get(i, {}):
                        path_changes += contingency_paths[i][j] - original_paths[i][j]
        else:
            path_changes = float('inf')
        
        return {
            'is_connected': is_connected,
            'path_changes': path_changes,
            'severity': 'high' if not is_connected else 'medium' if path_changes > 10 else 'low'
        }

def optimal_power_flow(buses: List[Bus], lines: List[Line], 
                      generators: List[Generator], total_demand: float) -> Dict:
    """最优潮流计算"""
    # 简化的最优潮流求解
    # 结合经济调度和潮流约束
    
    # 首先进行经济调度
    ed = EconomicDispatch(generators, total_demand)
    dispatch_result = ed.solve_economic_dispatch()
    
    if not dispatch_result['success']:
        return {'success': False, 'message': 'Economic dispatch failed'}
    
    # 然后进行潮流计算
    # 更新发电机母线功率
    for i, gen in enumerate(generators):
        gen_bus = next(bus for bus in buses if bus.bus_id == gen.bus_id)
        gen_bus.active_power = dispatch_result['generator_outputs'][i]
    
    # 潮流计算
    pf_solver = PowerFlowSolver(buses, lines)
    pf_result = pf_solver.newton_raphson_power_flow()
    
    return {
        'economic_dispatch': dispatch_result,
        'power_flow': pf_result,
        'success': pf_result['converged']
    }
```

### 配电系统算法 / Distribution System Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class DistributionLine:
    """配电线路"""
    line_id: str
    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    length: float

@dataclass
class Load:
    """负荷"""
    load_id: str
    bus_id: int
    active_power: float
    reactive_power: float
    load_type: str  # residential, commercial, industrial

class DistributionPowerFlow:
    """配电网潮流计算"""
    
    def __init__(self, lines: List[DistributionLine], loads: List[Load]):
        self.lines = lines
        self.loads = loads
        self.n_buses = max(max(line.from_bus, line.to_bus) for line in lines)
    
    def forward_backward_sweep(self, slack_voltage: complex = 1.0 + 0j) -> Dict:
        """前推回代法"""
        # 初始化
        V = np.ones(self.n_buses, dtype=complex) * slack_voltage
        I = np.zeros(self.n_buses, dtype=complex)
        
        # 计算负荷电流
        for load in self.loads:
            bus_idx = load.bus_id - 1
            S_load = load.active_power + 1j * load.reactive_power
            I[bus_idx] = np.conj(S_load / V[bus_idx])
        
        # 前推回代迭代
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            # 前推：计算电压
            for line in self.lines:
                from_idx = line.from_bus - 1
                to_idx = line.to_bus - 1
                
                Z_line = line.resistance + 1j * line.reactance
                V[to_idx] = V[from_idx] - Z_line * I[to_idx]
            
            # 检查收敛性
            voltage_change = np.max(np.abs(V - V_old))
            if voltage_change < tolerance:
                break
        
        # 计算功率损耗
        total_loss = 0
        for line in self.lines:
            from_idx = line.from_bus - 1
            to_idx = line.to_bus - 1
            Z_line = line.resistance + 1j * line.reactance
            I_line = (V[from_idx] - V[to_idx]) / Z_line
            loss = abs(I_line)**2 * line.resistance
            total_loss += loss
        
        return {
            'voltages': V,
            'currents': I,
            'total_loss': total_loss,
            'iterations': iteration + 1,
            'converged': voltage_change < tolerance
        }

class LoadModeling:
    """负荷建模"""
    
    def __init__(self):
        self.load_models = {
            'residential': self._residential_load_model,
            'commercial': self._commercial_load_model,
            'industrial': self._industrial_load_model
        }
    
    def _residential_load_model(self, time: float, base_load: float) -> Tuple[float, float]:
        """居民负荷模型"""
        hour = time % 24
        
        # 日负荷曲线
        if 6 <= hour <= 8:  # 早高峰
            factor = 1.3
        elif 18 <= hour <= 22:  # 晚高峰
            factor = 1.5
        elif 23 <= hour or hour <= 5:  # 夜间
            factor = 0.6
        else:
            factor = 1.0
        
        # 功率因数
        power_factor = 0.95
        
        active_power = base_load * factor
        reactive_power = active_power * np.tan(np.arccos(power_factor))
        
        return active_power, reactive_power
    
    def _commercial_load_model(self, time: float, base_load: float) -> Tuple[float, float]:
        """商业负荷模型"""
        hour = time % 24
        
        # 商业负荷曲线
        if 8 <= hour <= 18:  # 工作时间
            factor = 1.2
        elif 18 <= hour <= 22:  # 营业时间
            factor = 1.0
        else:
            factor = 0.3
        
        power_factor = 0.9
        active_power = base_load * factor
        reactive_power = active_power * np.tan(np.arccos(power_factor))
        
        return active_power, reactive_power
    
    def _industrial_load_model(self, time: float, base_load: float) -> Tuple[float, float]:
        """工业负荷模型"""
        # 工业负荷相对稳定
        factor = 1.0
        power_factor = 0.85
        
        active_power = base_load * factor
        reactive_power = active_power * np.tan(np.arccos(power_factor))
        
        return active_power, reactive_power
    
    def calculate_load_profile(self, loads: List[Load], time: float) -> List[Tuple[float, float]]:
        """计算负荷曲线"""
        load_profile = []
        
        for load in loads:
            if load.load_type in self.load_models:
                model_func = self.load_models[load.load_type]
                P, Q = model_func(time, load.active_power)
                load_profile.append((P, Q))
            else:
                load_profile.append((load.active_power, load.reactive_power))
        
        return load_profile

class VoltageControl:
    """电压控制"""
    
    def __init__(self, distribution_system: DistributionPowerFlow):
        self.distribution_system = distribution_system
    
    def capacitor_placement_optimization(self, candidate_locations: List[int], 
                                       capacitor_sizes: List[float]) -> Dict:
        """电容器优化配置"""
        best_placement = None
        best_voltage_profile = None
        min_voltage_deviation = float('inf')
        
        # 枚举所有可能的配置
        for location in candidate_locations:
            for size in capacitor_sizes:
                # 添加电容器
                # 这里简化处理，实际需要修改潮流计算
                
                # 计算电压偏差
                voltage_deviation = self._calculate_voltage_deviation()
                
                if voltage_deviation < min_voltage_deviation:
                    min_voltage_deviation = voltage_deviation
                    best_placement = (location, size)
        
        return {
            'optimal_placement': best_placement,
            'min_voltage_deviation': min_voltage_deviation
        }
    
    def _calculate_voltage_deviation(self) -> float:
        """计算电压偏差"""
        # 简化的电压偏差计算
        return 0.05  # 5%的电压偏差
```

### 能源经济算法 / Energy Economics Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import pandas as pd

@dataclass
class ElectricityPrice:
    """电价"""
    time_period: int
    price: float
    demand: float
    supply: float

class ElectricityPriceModel:
    """电价模型"""
    
    def __init__(self, generators: List[Generator], demand_profile: List[float]):
        self.generators = generators
        self.demand_profile = demand_profile
    
    def marginal_cost_pricing(self) -> List[float]:
        """边际成本定价"""
        prices = []
        
        for demand in self.demand_profile:
            # 按边际成本排序
            sorted_gens = sorted(self.generators, 
                               key=lambda g: g.cost_coefficients[1])
            
            # 找到边际机组
            cumulative_capacity = 0
            marginal_price = 0
            
            for gen in sorted_gens:
                cumulative_capacity += gen.max_output
                if cumulative_capacity >= demand:
                    # 边际成本 = 2*a*P + b
                    marginal_price = (2 * gen.cost_coefficients[0] * 
                                    (demand - (cumulative_capacity - gen.max_output)) + 
                                    gen.cost_coefficients[1])
                    break
            
            prices.append(marginal_price)
        
        return prices
    
    def time_of_use_pricing(self, peak_hours: List[int], 
                           off_peak_hours: List[int]) -> List[float]:
        """分时电价"""
        prices = []
        base_price = 50.0  # 基础电价
        
        for hour in range(24):
            if hour in peak_hours:
                price = base_price * 1.5  # 峰时电价
            elif hour in off_peak_hours:
                price = base_price * 0.7  # 谷时电价
            else:
                price = base_price  # 平时电价
            
            prices.append(price)
        
        return prices

class InvestmentDecision:
    """投资决策"""
    
    def __init__(self, discount_rate: float = 0.1):
        self.discount_rate = discount_rate
    
    def net_present_value(self, initial_investment: float, 
                         cash_flows: List[float]) -> float:
        """净现值计算"""
        npv = -initial_investment
        
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + self.discount_rate) ** (i + 1))
        
        return npv
    
    def internal_rate_of_return(self, initial_investment: float, 
                               cash_flows: List[float]) -> float:
        """内部收益率计算"""
        def npv_function(rate):
            npv = -initial_investment
            for i, cash_flow in enumerate(cash_flows):
                npv += cash_flow / ((1 + rate) ** (i + 1))
            return npv
        
        # 使用数值方法求解IRR
        from scipy.optimize import fsolve
        irr = fsolve(npv_function, 0.1)[0]
        return irr
    
    def payback_period(self, initial_investment: float, 
                      cash_flows: List[float]) -> float:
        """投资回收期计算"""
        cumulative_cash_flow = 0
        for i, cash_flow in enumerate(cash_flows):
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= initial_investment:
                return i + 1
        
        return float('inf')

class PowerMarket:
    """电力市场"""
    
    def __init__(self, generators: List[Generator], loads: List[Load]):
        self.generators = generators
        self.loads = loads
    
    def auction_market_clearing(self, bids: List[float], 
                              offers: List[float]) -> Dict:
        """拍卖市场出清"""
        # 按报价排序
        sorted_bids = sorted(bids, reverse=True)  # 买价从高到低
        sorted_offers = sorted(offers)  # 卖价从低到高
        
        # 找到市场出清价格
        clearing_price = 0
        clearing_quantity = 0
        
        for i, (bid, offer) in enumerate(zip(sorted_bids, sorted_offers)):
            if bid >= offer:
                clearing_price = (bid + offer) / 2
                clearing_quantity = i + 1
            else:
                break
        
        return {
            'clearing_price': clearing_price,
            'clearing_quantity': clearing_quantity,
            'market_efficiency': clearing_quantity / min(len(bids), len(offers))
        }
    
    def bilateral_contract_pricing(self, contract_quantity: float, 
                                 contract_duration: int,
                                 market_prices: List[float]) -> float:
        """双边合同定价"""
        # 基于市场价格的合同定价
        avg_market_price = np.mean(market_prices)
        
        # 考虑风险溢价
        risk_premium = 0.05  # 5%的风险溢价
        contract_price = avg_market_price * (1 + risk_premium)
        
        return contract_price

def power_energy_verification():
    """电力能源模型验证"""
    print("=== 电力能源模型验证 ===")
    
    # 电力系统分析验证
    print("\n1. 电力系统分析验证:")
    buses = [
        Bus(1, 'Slack', 1.0, 0.0, 0.0, 0.0),
        Bus(2, 'PV', 1.0, 0.0, 0.5, 0.0),
        Bus(3, 'PQ', 1.0, 0.0, -0.8, -0.4)
    ]
    lines = [
        Line(1, 2, 0.01, 0.1, 0.0, 1.0),
        Line(2, 3, 0.01, 0.1, 0.0, 1.0)
    ]
    
    pf_solver = PowerFlowSolver(buses, lines)
    pf_result = pf_solver.newton_raphson_power_flow()
    print(f"潮流计算收敛: {pf_result['converged']}")
    print(f"迭代次数: {pf_result['iterations']}")
    
    # 发电模型验证
    print("\n2. 发电模型验证:")
    generators = [
        Generator("G1", 2, 100, 20, 80, [0.1, 20, 100], 10),
        Generator("G2", 3, 150, 30, 120, [0.15, 25, 150], 15)
    ]
    
    ed = EconomicDispatch(generators, 1.0)
    ed_result = ed.solve_economic_dispatch()
    print(f"经济调度成功: {ed_result['success']}")
    if ed_result['success']:
        print(f"总成本: {ed_result['total_cost']:.2f}")
    
    # 输电网络验证
    print("\n3. 输电网络验证:")
    transmission_lines = [
        TransmissionLine("L1", 1, 2, 0.01, 0.1, 0.0, 100),
        TransmissionLine("L2", 2, 3, 0.01, 0.1, 0.0, 100)
    ]
    
    network = TransmissionNetwork(transmission_lines)
    reliability = network.calculate_network_reliability()
    print(f"网络可靠性: {reliability:.4f}")
    
    # 配电系统验证
    print("\n4. 配电系统验证:")
    dist_lines = [
        DistributionLine("DL1", 1, 2, 0.1, 0.2, 1.0),
        DistributionLine("DL2", 2, 3, 0.1, 0.2, 1.0)
    ]
    loads = [
        Load("L1", 2, 0.5, 0.2, "residential"),
        Load("L2", 3, 0.3, 0.1, "commercial")
    ]
    
    dpf = DistributionPowerFlow(dist_lines, loads)
    dpf_result = dpf.forward_backward_sweep()
    print(f"配电网潮流收敛: {dpf_result['converged']}")
    print(f"总损耗: {dpf_result['total_loss']:.4f}")
    
    # 能源经济验证
    print("\n5. 能源经济验证:")
    price_model = ElectricityPriceModel(generators, [0.8, 1.0, 1.2])
    prices = price_model.marginal_cost_pricing()
    print(f"边际成本电价: {prices}")
    
    investment = InvestmentDecision(0.1)
    npv = investment.net_present_value(1000, [200, 300, 400, 500])
    print(f"净现值: {npv:.2f}")
    
    print("\n验证完成!")

if __name__ == "__main__":
    power_energy_verification()
```

---

*最后更新: 2025-08-26*
*版本: 3.1.0*
