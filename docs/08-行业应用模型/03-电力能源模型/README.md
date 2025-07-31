# 8.3 电力能源模型 / Power & Energy Models

## 目录 / Table of Contents

- [8.3 电力能源模型 / Power \& Energy Models](#83-电力能源模型--power--energy-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.3.1 电力系统模型 / Power System Models](#831-电力系统模型--power-system-models)
    - [电力系统基本方程 / Basic Power System Equations](#电力系统基本方程--basic-power-system-equations)
    - [传输线模型 / Transmission Line Model](#传输线模型--transmission-line-model)
    - [发电机模型 / Generator Model](#发电机模型--generator-model)
  - [8.3.2 潮流计算模型 / Power Flow Models](#832-潮流计算模型--power-flow-models)
    - [牛顿-拉夫森法 / Newton-Raphson Method](#牛顿-拉夫森法--newton-raphson-method)
    - [高斯-赛德尔法 / Gauss-Seidel Method](#高斯-赛德尔法--gauss-seidel-method)
    - [快速分解法 / Fast Decoupled Method](#快速分解法--fast-decoupled-method)
  - [8.3.3 电力市场模型 / Electricity Market Models](#833-电力市场模型--electricity-market-models)
    - [经济调度 / Economic Dispatch](#经济调度--economic-dispatch)
    - [机组组合 / Unit Commitment](#机组组合--unit-commitment)
    - [电力拍卖模型 / Electricity Auction Model](#电力拍卖模型--electricity-auction-model)
  - [8.3.4 可再生能源模型 / Renewable Energy Models](#834-可再生能源模型--renewable-energy-models)
    - [风电模型 / Wind Power Model](#风电模型--wind-power-model)
    - [光伏模型 / Solar PV Model](#光伏模型--solar-pv-model)
    - [水能模型 / Hydropower Model](#水能模型--hydropower-model)
  - [8.3.5 智能电网模型 / Smart Grid Models](#835-智能电网模型--smart-grid-models)
    - [需求响应 / Demand Response](#需求响应--demand-response)
    - [微电网模型 / Microgrid Model](#微电网模型--microgrid-model)
    - [分布式能源 / Distributed Energy Resources](#分布式能源--distributed-energy-resources)
  - [8.3.6 能源优化模型 / Energy Optimization Models](#836-能源优化模型--energy-optimization-models)
    - [能源规划 / Energy Planning](#能源规划--energy-planning)
    - [多目标优化 / Multi-objective Optimization](#多目标优化--multi-objective-optimization)
    - [能源效率 / Energy Efficiency](#能源效率--energy-efficiency)
  - [8.3.7 储能系统模型 / Energy Storage Models](#837-储能系统模型--energy-storage-models)
    - [电池储能 / Battery Storage](#电池储能--battery-storage)
    - [抽水蓄能 / Pumped Hydro Storage](#抽水蓄能--pumped-hydro-storage)
    - [压缩空气储能 / Compressed Air Energy Storage](#压缩空气储能--compressed-air-energy-storage)
  - [8.3.8 实现与应用 / Implementation and Applications](#838-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [电力系统运行 / Power System Operation](#电力系统运行--power-system-operation)
      - [电力市场 / Electricity Markets](#电力市场--electricity-markets)
      - [智能电网 / Smart Grid](#智能电网--smart-grid)
  - [参考文献 / References](#参考文献--references)

---

## 8.3.1 电力系统模型 / Power System Models

### 电力系统基本方程 / Basic Power System Equations

**节点电压方程**: $V_i = |V_i| \angle \theta_i$

**复功率**: $S_i = P_i + jQ_i = V_i I_i^*$

**功率平衡**: $\sum_{i=1}^n P_i = 0$, $\sum_{i=1}^n Q_i = 0$

### 传输线模型 / Transmission Line Model

**π型等效电路**:

```text
     Y/2
A ────┴──── B
     Z      Y/2
```

**阻抗**: $Z = R + jX$

**导纳**: $Y = G + jB$

**传输线方程**:
$$\begin{pmatrix} V_A \\ I_A \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} V_B \\ I_B \end{pmatrix}$$

### 发电机模型 / Generator Model

**同步发电机**: $E_g = V_t + jX_d I_a$

**功率角方程**: $P = \frac{E_g V_t}{X_d} \sin \delta$

**转子运动方程**: $\frac{2H}{\omega_0} \frac{d^2\delta}{dt^2} = P_m - P_e$

---

## 8.3.2 潮流计算模型 / Power Flow Models

### 牛顿-拉夫森法 / Newton-Raphson Method

**雅可比矩阵**: $J = \begin{pmatrix} \frac{\partial P}{\partial \theta} & \frac{\partial P}{\partial V} \\ \frac{\partial Q}{\partial \theta} & \frac{\partial Q}{\partial V} \end{pmatrix}$

**迭代方程**: $\begin{pmatrix} \Delta \theta \\ \Delta V \end{pmatrix} = J^{-1} \begin{pmatrix} \Delta P \\ \Delta Q \end{pmatrix}$

**收敛条件**: $|\Delta P_i| < \epsilon$, $|\Delta Q_i| < \epsilon$

### 高斯-赛德尔法 / Gauss-Seidel Method

**电压更新**: $V_i^{(k+1)} = \frac{1}{Y_{ii}} \left( \frac{P_i - jQ_i}{V_i^{(k)*}} - \sum_{j \neq i} Y_{ij} V_j^{(k)} \right)$

**收敛判据**: $\max |V_i^{(k+1)} - V_i^{(k)}| < \epsilon$

### 快速分解法 / Fast Decoupled Method

**P-θ分解**: $\Delta P = B' \Delta \theta$

**Q-V分解**: $\Delta Q = B'' \Delta V$

**简化雅可比矩阵**: $B'_{ij} = -B_{ij}$, $B''_{ij} = -B_{ij}$

---

## 8.3.3 电力市场模型 / Electricity Market Models

### 经济调度 / Economic Dispatch

**目标函数**: $\min \sum_{i=1}^n C_i(P_i)$

**约束条件**:

- $\sum_{i=1}^n P_i = P_D$ (功率平衡)
- $P_{i,min} \leq P_i \leq P_{i,max}$ (发电限制)

**拉格朗日函数**: $L = \sum_{i=1}^n C_i(P_i) + \lambda(P_D - \sum_{i=1}^n P_i)$

**最优条件**: $\frac{dC_i}{dP_i} = \lambda$ (等增量成本)

### 机组组合 / Unit Commitment

**目标函数**: $\min \sum_{t=1}^T \sum_{i=1}^n [C_i(P_{it}) + SU_i u_{it} + SD_i v_{it}]$

**约束条件**:

- 功率平衡: $\sum_{i=1}^n P_{it} = P_{Dt}$
- 机组限制: $P_{i,min} u_{it} \leq P_{it} \leq P_{i,max} u_{it}$
- 最小运行时间: $u_{it} - u_{i,t-1} \leq u_{i,t+UT_i-1}$
- 最小停机时间: $u_{i,t-1} - u_{it} \leq 1 - u_{i,t+DT_i-1}$

### 电力拍卖模型 / Electricity Auction Model

**统一价格拍卖**: $p = \max \{b_i : i \in S\}$

**按报价支付**: $p_i = b_i$

**市场出清**: $\sum_{i \in S} q_i = D$

---

## 8.3.4 可再生能源模型 / Renewable Energy Models

### 风电模型 / Wind Power Model

**风速分布**: $f(v) = \frac{k}{c} \left(\frac{v}{c}\right)^{k-1} e^{-(v/c)^k}$

**功率曲线**: $P(v) = \begin{cases} 0 & v < v_{ci} \\ P_r \frac{v^3 - v_{ci}^3}{v_r^3 - v_{ci}^3} & v_{ci} \leq v < v_r \\ P_r & v_r \leq v < v_{co} \\ 0 & v \geq v_{co} \end{cases}$

**容量因子**: $CF = \frac{\int_{v_{ci}}^{v_{co}} P(v) f(v) dv}{P_r}$

### 光伏模型 / Solar PV Model

**太阳辐射**: $G = G_0 \cos \theta \cdot T$

**光伏功率**: $P = \eta A G (1 - 0.005(T_c - 25))$

**温度系数**: $T_c = T_a + \frac{NOCT - 20}{800} G$

### 水能模型 / Hydropower Model

**水头**: $H = H_{gross} - H_{loss}$

**功率**: $P = \eta \rho g Q H$

**水库调度**: $\frac{dV}{dt} = I - Q - E$

---

## 8.3.5 智能电网模型 / Smart Grid Models

### 需求响应 / Demand Response

**价格响应**: $D_t = D_0 \left(1 + \epsilon \frac{p_t - p_0}{p_0}\right)$

**负荷转移**: $\sum_{t=1}^T D_t = \sum_{t=1}^T D_0$

**用户效用**: $U = \sum_{t=1}^T [B(D_t) - p_t D_t]$

### 微电网模型 / Microgrid Model

**功率平衡**: $P_{DG} + P_{BESS} + P_{GRID} = P_{LOAD}$

**储能约束**: $SOC_{min} \leq SOC_t \leq SOC_{max}$

**经济调度**: $\min \sum_{i=1}^n C_i(P_i) + C_{BESS}(P_{BESS}) + C_{GRID}(P_{GRID})$

### 分布式能源 / Distributed Energy Resources

**光伏系统**: $P_{PV} = \eta_{PV} A_{PV} G$

**风力发电**: $P_{WT} = \frac{1}{2} \rho A v^3 C_p$

**燃料电池**: $P_{FC} = \eta_{FC} \dot{m}_{H2} LHV_{H2}$

---

## 8.3.6 能源优化模型 / Energy Optimization Models

### 能源规划 / Energy Planning

**目标函数**: $\min \sum_{i=1}^n \sum_{t=1}^T [C_i(P_{it}) + E_i(P_{it})]$

**约束条件**:

- 能源平衡: $\sum_{i=1}^n P_{it} = D_t$
- 容量限制: $P_{it} \leq C_i$
- 环境约束: $\sum_{i=1}^n E_i(P_{it}) \leq E_{max}$

### 多目标优化 / Multi-objective Optimization

**目标函数**:

- 最小化成本: $\min \sum_{i=1}^n C_i(P_i)$
- 最小化排放: $\min \sum_{i=1}^n E_i(P_i)$
- 最大化可靠性: $\max R(P_1, \ldots, P_n)$

**帕累托最优**: 使用遗传算法或粒子群优化

### 能源效率 / Energy Efficiency

**能效比**: $EER = \frac{Q}{W}$

**性能系数**: $COP = \frac{Q}{W}$

**综合效率**: $\eta_{total} = \eta_1 \cdot \eta_2 \cdot \ldots \cdot \eta_n$

---

## 8.3.7 储能系统模型 / Energy Storage Models

### 电池储能 / Battery Storage

**荷电状态**: $SOC(t) = SOC_0 + \frac{1}{C} \int_0^t i(\tau) d\tau$

**功率约束**: $-P_{max} \leq P(t) \leq P_{max}$

**能量约束**: $SOC_{min} \leq SOC(t) \leq SOC_{max}$

**寿命模型**: $L = L_0 \left(\frac{C}{C_0}\right)^\alpha \left(\frac{I}{I_0}\right)^\beta$

### 抽水蓄能 / Pumped Hydro Storage

**能量**: $E = \rho g V H \eta$

**功率**: $P = \rho g Q H \eta$

**效率**: $\eta = \eta_p \cdot \eta_t \cdot \eta_m \cdot \eta_g$

### 压缩空气储能 / Compressed Air Energy Storage

**能量**: $E = \frac{1}{2} m v^2 + mgh$

**压力**: $P = P_0 \left(\frac{V_0}{V}\right)^\gamma$

**温度**: $T = T_0 \left(\frac{V_0}{V}\right)^{\gamma-1}$

---

## 8.3.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PowerSystem {
    pub buses: Vec<Bus>,
    pub lines: Vec<TransmissionLine>,
    pub generators: Vec<Generator>,
}

#[derive(Debug, Clone)]
pub struct Bus {
    pub id: String,
    pub voltage_magnitude: f64,
    pub voltage_angle: f64,
    pub bus_type: BusType,
    pub active_power: f64,
    pub reactive_power: f64,
}

#[derive(Debug, Clone)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

#[derive(Debug, Clone)]
pub struct TransmissionLine {
    pub from_bus: String,
    pub to_bus: String,
    pub resistance: f64,
    pub reactance: f64,
    pub susceptance: f64,
    pub capacity: f64,
}

#[derive(Debug, Clone)]
pub struct Generator {
    pub bus_id: String,
    pub active_power: f64,
    pub reactive_power: f64,
    pub max_power: f64,
    pub min_power: f64,
    pub cost_function: Box<dyn Fn(f64) -> f64>,
}

impl PowerSystem {
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            lines: Vec::new(),
            generators: Vec::new(),
        }
    }
    
    pub fn add_bus(&mut self, bus: Bus) {
        self.buses.push(bus);
    }
    
    pub fn add_line(&mut self, line: TransmissionLine) {
        self.lines.push(line);
    }
    
    pub fn add_generator(&mut self, generator: Generator) {
        self.generators.push(generator);
    }
    
    pub fn newton_raphson_power_flow(&mut self, max_iterations: usize, tolerance: f64) -> Result<(), String> {
        for iteration in 0..max_iterations {
            let mut max_mismatch = 0.0;
            
            // 计算功率不平衡量
            for bus in &mut self.buses {
                if let BusType::PQ = bus.bus_type {
                    let (p_calc, q_calc) = self.calculate_power_at_bus(bus);
                    let p_mismatch = bus.active_power - p_calc;
                    let q_mismatch = bus.reactive_power - q_calc;
                    
                    max_mismatch = max_mismatch.max(p_mismatch.abs()).max(q_mismatch.abs());
                    
                    // 更新电压和相角
                    // 这里简化处理，实际需要构建雅可比矩阵
                    bus.voltage_angle += 0.1 * p_mismatch;
                    bus.voltage_magnitude += 0.1 * q_mismatch;
                }
            }
            
            if max_mismatch < tolerance {
                println!("Power flow converged in {} iterations", iteration + 1);
                return Ok(());
            }
        }
        
        Err("Power flow did not converge".to_string())
    }
    
    fn calculate_power_at_bus(&self, bus: &Bus) -> (f64, f64) {
        let mut p_calc = 0.0;
        let mut q_calc = 0.0;
        
        for line in &self.lines {
            if line.from_bus == bus.id {
                // 简化的功率计算
                let v_i = bus.voltage_magnitude;
                let v_j = 1.0; // 假设其他节点电压为标幺值
                let theta_ij = bus.voltage_angle;
                
                p_calc += v_i * v_j * (line.resistance * theta_ij.cos() + line.reactance * theta_ij.sin());
                q_calc += v_i * v_j * (line.reactance * theta_ij.cos() - line.resistance * theta_ij.sin());
            }
        }
        
        (p_calc, q_calc)
    }
    
    pub fn economic_dispatch(&self, total_demand: f64) -> HashMap<String, f64> {
        let mut dispatch = HashMap::new();
        let mut lambda = 10.0; // 初始增量成本
        
        for _ in 0..100 { // 迭代求解
            let mut total_power = 0.0;
            
            for generator in &self.generators {
                // 简化的等增量成本法
                let power = (lambda - 5.0) / 0.1; // 假设成本函数为二次函数
                let power = power.max(generator.min_power).min(generator.max_power);
                dispatch.insert(generator.bus_id.clone(), power);
                total_power += power;
            }
            
            if (total_power - total_demand).abs() < 0.01 {
                break;
            }
            
            lambda += 0.1 * (total_demand - total_power);
        }
        
        dispatch
    }
}

#[derive(Debug)]
pub struct WindFarm {
    pub capacity: f64,
    pub wind_speed: f64,
    pub cut_in_speed: f64,
    pub rated_speed: f64,
    pub cut_out_speed: f64,
}

impl WindFarm {
    pub fn new(capacity: f64) -> Self {
        Self {
            capacity,
            wind_speed: 0.0,
            cut_in_speed: 3.0,
            rated_speed: 12.0,
            cut_out_speed: 25.0,
        }
    }
    
    pub fn set_wind_speed(&mut self, speed: f64) {
        self.wind_speed = speed;
    }
    
    pub fn calculate_power(&self) -> f64 {
        if self.wind_speed < self.cut_in_speed || self.wind_speed > self.cut_out_speed {
            0.0
        } else if self.wind_speed < self.rated_speed {
            self.capacity * ((self.wind_speed.powi(3) - self.cut_in_speed.powi(3)) / 
                            (self.rated_speed.powi(3) - self.cut_in_speed.powi(3)))
        } else {
            self.capacity
        }
    }
}

#[derive(Debug)]
pub struct BatteryStorage {
    pub capacity: f64,
    pub max_power: f64,
    pub state_of_charge: f64,
    pub efficiency: f64,
}

impl BatteryStorage {
    pub fn new(capacity: f64, max_power: f64) -> Self {
        Self {
            capacity,
            max_power,
            state_of_charge: 0.5, // 初始50%荷电状态
            efficiency: 0.9,
        }
    }
    
    pub fn charge(&mut self, power: f64, duration: f64) -> f64 {
        let actual_power = power.min(self.max_power);
        let energy = actual_power * duration * self.efficiency;
        let energy_change = energy / self.capacity;
        
        self.state_of_charge = (self.state_of_charge + energy_change).min(1.0);
        actual_power
    }
    
    pub fn discharge(&mut self, power: f64, duration: f64) -> f64 {
        let actual_power = power.min(self.max_power);
        let energy = actual_power * duration / self.efficiency;
        let energy_change = energy / self.capacity;
        
        if self.state_of_charge >= energy_change {
            self.state_of_charge -= energy_change;
            actual_power
        } else {
            let available_energy = self.state_of_charge * self.capacity * self.efficiency;
            let available_power = available_energy / duration;
            self.state_of_charge = 0.0;
            available_power
        }
    }
}

// 使用示例
fn main() {
    // 创建电力系统
    let mut system = PowerSystem::new();
    
    // 添加母线
    system.add_bus(Bus {
        id: "Bus1".to_string(),
        voltage_magnitude: 1.0,
        voltage_angle: 0.0,
        bus_type: BusType::Slack,
        active_power: 0.0,
        reactive_power: 0.0,
    });
    
    system.add_bus(Bus {
        id: "Bus2".to_string(),
        voltage_magnitude: 1.0,
        voltage_angle: 0.0,
        bus_type: BusType::PQ,
        active_power: -1.0,
        reactive_power: -0.5,
    });
    
    // 添加传输线
    system.add_line(TransmissionLine {
        from_bus: "Bus1".to_string(),
        to_bus: "Bus2".to_string(),
        resistance: 0.01,
        reactance: 0.1,
        susceptance: 0.0,
        capacity: 2.0,
    });
    
    // 添加发电机
    system.add_generator(Generator {
        bus_id: "Bus1".to_string(),
        active_power: 1.0,
        reactive_power: 0.5,
        max_power: 2.0,
        min_power: 0.0,
        cost_function: Box::new(|p| 5.0 * p + 0.1 * p * p),
    });
    
    // 潮流计算
    match system.newton_raphson_power_flow(100, 0.001) {
        Ok(_) => println!("Power flow calculation successful"),
        Err(e) => println!("Power flow calculation failed: {}", e),
    }
    
    // 经济调度
    let dispatch = system.economic_dispatch(1.0);
    println!("Economic dispatch: {:?}", dispatch);
    
    // 风电模型
    let mut wind_farm = WindFarm::new(100.0);
    wind_farm.set_wind_speed(10.0);
    println!("Wind power: {:.2} MW", wind_farm.calculate_power());
    
    // 储能系统
    let mut battery = BatteryStorage::new(100.0, 20.0);
    let charged = battery.charge(15.0, 1.0);
    let discharged = battery.discharge(10.0, 1.0);
    println!("Battery SOC: {:.2}, Charged: {:.2} MW, Discharged: {:.2} MW", 
             battery.state_of_charge, charged, discharged);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module PowerEnergyModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, length)

-- 电力系统数据类型
data PowerSystem = PowerSystem {
    buses :: [Bus],
    lines :: [TransmissionLine],
    generators :: [Generator]
} deriving Show

data Bus = Bus {
    busId :: String,
    voltageMagnitude :: Double,
    voltageAngle :: Double,
    busType :: BusType,
    activePower :: Double,
    reactivePower :: Double
} deriving Show

data BusType = Slack | PV | PQ deriving Show

data TransmissionLine = TransmissionLine {
    fromBus :: String,
    toBus :: String,
    resistance :: Double,
    reactance :: Double,
    susceptance :: Double,
    capacity :: Double
} deriving Show

data Generator = Generator {
    genBusId :: String,
    genActivePower :: Double,
    genReactivePower :: Double,
    maxPower :: Double,
    minPower :: Double
} deriving Show

-- 创建电力系统
newPowerSystem :: PowerSystem
newPowerSystem = PowerSystem [] [] []

-- 添加母线
addBus :: Bus -> PowerSystem -> PowerSystem
addBus bus system = system { buses = bus : buses system }

-- 添加传输线
addLine :: TransmissionLine -> PowerSystem -> PowerSystem
addLine line system = system { lines = line : lines system }

-- 添加发电机
addGenerator :: Generator -> PowerSystem -> PowerSystem
addGenerator gen system = system { generators = gen : generators system }

-- 简化的潮流计算
powerFlow :: PowerSystem -> Int -> Double -> Either String PowerSystem
powerFlow system maxIterations tolerance = go system 0
  where
    go sys iteration
        | iteration >= maxIterations = Left "Power flow did not converge"
        | maxMismatch < tolerance = Right sys
        | otherwise = go updatedSystem (iteration + 1)
      where
        maxMismatch = calculateMaxMismatch sys
        updatedSystem = updateVoltages sys

-- 计算最大不平衡量
calculateMaxMismatch :: PowerSystem -> Double
calculateMaxMismatch system = maximum [abs (activePower bus - pCalc) | bus <- buses system, busType bus == PQ]
  where
    pCalc = 1.0 -- 简化的功率计算

-- 更新电压
updateVoltages :: PowerSystem -> PowerSystem
updateVoltages system = system { buses = map updateBus (buses system) }
  where
    updateBus bus = case busType bus of
        PQ -> bus { voltageAngle = voltageAngle bus + 0.1 }
        _ -> bus

-- 经济调度
economicDispatch :: PowerSystem -> Double -> Map String Double
economicDispatch system totalDemand = go initialLambda
  where
    initialLambda = 10.0
    
    go lambda
        | abs (totalPower - totalDemand) < 0.01 = dispatch
        | otherwise = go (lambda + 0.1 * (totalDemand - totalPower))
      where
        dispatch = Map.fromList [(genBusId gen, power) | gen <- generators system]
        power = (lambda - 5.0) / 0.1
        totalPower = sum [power | gen <- generators system]

-- 风电模型
data WindFarm = WindFarm {
    capacity :: Double,
    windSpeed :: Double,
    cutInSpeed :: Double,
    ratedSpeed :: Double,
    cutOutSpeed :: Double
} deriving Show

newWindFarm :: Double -> WindFarm
newWindFarm cap = WindFarm cap 0.0 3.0 12.0 25.0

setWindSpeed :: Double -> WindFarm -> WindFarm
setWindSpeed speed wf = wf { windSpeed = speed }

calculateWindPower :: WindFarm -> Double
calculateWindPower wf
    | windSpeed wf < cutInSpeed wf || windSpeed wf > cutOutSpeed wf = 0.0
    | windSpeed wf < ratedSpeed wf = capacity wf * powerRatio
    | otherwise = capacity wf
  where
    powerRatio = ((windSpeed wf)^3 - (cutInSpeed wf)^3) / ((ratedSpeed wf)^3 - (cutInSpeed wf)^3)

-- 储能系统
data BatteryStorage = BatteryStorage {
    batCapacity :: Double,
    maxPower :: Double,
    stateOfCharge :: Double,
    efficiency :: Double
} deriving Show

newBatteryStorage :: Double -> Double -> BatteryStorage
newBatteryStorage cap maxP = BatteryStorage cap maxP 0.5 0.9

charge :: Double -> Double -> BatteryStorage -> (Double, BatteryStorage)
charge power duration battery
    | newSOC > 1.0 = (actualPower, battery { stateOfCharge = 1.0 })
    | otherwise = (actualPower, battery { stateOfCharge = newSOC })
  where
    actualPower = min power (maxPower battery)
    energy = actualPower * duration * efficiency battery
    energyChange = energy / batCapacity battery
    newSOC = stateOfCharge battery + energyChange

discharge :: Double -> Double -> BatteryStorage -> (Double, BatteryStorage)
discharge power duration battery
    | stateOfCharge battery < energyChange = (availablePower, battery { stateOfCharge = 0.0 })
    | otherwise = (actualPower, battery { stateOfCharge = newSOC })
  where
    actualPower = min power (maxPower battery)
    energy = actualPower * duration / efficiency battery
    energyChange = energy / batCapacity battery
    availableEnergy = stateOfCharge battery * batCapacity battery * efficiency battery
    availablePower = availableEnergy / duration
    newSOC = stateOfCharge battery - energyChange

-- 示例使用
example :: IO ()
example = do
    -- 创建电力系统
    let system = addGenerator (Generator "Bus1" 1.0 0.5 2.0 0.0) $
                 addLine (TransmissionLine "Bus1" "Bus2" 0.01 0.1 0.0 2.0) $
                 addBus (Bus "Bus2" 1.0 0.0 PQ (-1.0) (-0.5)) $
                 addBus (Bus "Bus1" 1.0 0.0 Slack 0.0 0.0) newPowerSystem
    
    -- 潮流计算
    case powerFlow system 100 0.001 of
        Right _ -> putStrLn "Power flow calculation successful"
        Left err -> putStrLn $ "Power flow calculation failed: " ++ err
    
    -- 经济调度
    let dispatch = economicDispatch system 1.0
    putStrLn $ "Economic dispatch: " ++ show dispatch
    
    -- 风电模型
    let windFarm = setWindSpeed 10.0 (newWindFarm 100.0)
    putStrLn $ "Wind power: " ++ show (calculateWindPower windFarm) ++ " MW"
    
    -- 储能系统
    let battery = newBatteryStorage 100.0 20.0
        (charged, battery1) = charge 15.0 1.0 battery
        (discharged, battery2) = discharge 10.0 1.0 battery1
    
    putStrLn $ "Battery SOC: " ++ show (stateOfCharge battery2) ++ 
               ", Charged: " ++ show charged ++ " MW, Discharged: " ++ show discharged ++ " MW"
```

### 应用领域 / Application Domains

#### 电力系统运行 / Power System Operation

- **潮流计算**: 稳态分析、电压稳定性
- **经济调度**: 成本最小化、燃料优化
- **安全分析**: 暂态稳定性、故障分析

#### 电力市场 / Electricity Markets

- **市场设计**: 拍卖机制、价格形成
- **风险管理**: 价格风险、容量风险
- **市场监管**: 反垄断、公平竞争

#### 智能电网 / Smart Grid

- **需求响应**: 负荷管理、价格响应
- **分布式能源**: 微电网、虚拟电厂
- **储能系统**: 电池储能、抽水蓄能

---

## 参考文献 / References

1. Grainger, J. J., & Stevenson, W. D. (1994). Power System Analysis. McGraw-Hill.
2. Wood, A. J., & Wollenberg, B. F. (2012). Power Generation, Operation, and Control. Wiley.
3. Kirschen, D. S., & Strbac, G. (2018). Fundamentals of Power System Economics. Wiley.
4. Kundur, P. (1994). Power System Stability and Control. McGraw-Hill.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
