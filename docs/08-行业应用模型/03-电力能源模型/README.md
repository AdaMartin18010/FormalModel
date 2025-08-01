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
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [电网规划 / Grid Planning](#电网规划--grid-planning)
      - [运行调度 / Operation Dispatch](#运行调度--operation-dispatch)
      - [市场交易 / Market Trading](#市场交易--market-trading)
  - [参考文献 / References](#参考文献--references)

---

## 8.3.1 电力系统模型 / Power System Models

### 潮流计算 / Power Flow Analysis

**节点功率方程**: $P_i = \sum_{j=1}^n V_i V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij})$

**节点无功功率**: $Q_i = \sum_{j=1}^n V_i V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij})$

**牛顿-拉夫森法**: $\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}$

### 稳定性分析 / Stability Analysis

**小信号稳定性**: $\dot{x} = Ax + Bu$

**特征值分析**: $\det(\lambda I - A) = 0$

**阻尼比**: $\zeta = \frac{-\sigma}{\sqrt{\sigma^2 + \omega^2}}$

### 故障分析 / Fault Analysis

**三相短路**: $I_f = \frac{V_f}{Z_f}$

**不对称故障**: $I_f^{(1)} = \frac{V_f^{(1)}}{Z_1 + Z_2 + Z_0}$

---

## 8.3.2 发电模型 / Generation Models

### 火电模型 / Thermal Power Models

**热效率**: $\eta = \frac{P_{output}}{Q_{input}}$

**燃料消耗**: $F = \frac{P}{\eta \cdot LHV}$

**启动成本**: $C_{start} = C_{cold} + C_{hot} \cdot (1 - e^{-t/\tau})$

### 水电模型 / Hydroelectric Models

**水头**: $H = H_{gross} - H_{loss}$

**功率**: $P = \eta \cdot \rho \cdot g \cdot Q \cdot H$

**水库调度**: $V(t+1) = V(t) + I(t) - Q(t) - L(t)$

### 新能源模型 / Renewable Energy Models

**风电功率**: $P = \frac{1}{2} \rho A v^3 C_p$

**光伏功率**: $P = P_{STC} \cdot \frac{G}{G_{STC}} \cdot [1 + \alpha(T - T_{STC})]$

**储能模型**: $SOC(t+1) = SOC(t) + \frac{P_{charge}(t) - P_{discharge}(t)}{E_{rated}} \cdot \Delta t$

---

## 8.3.3 输电网络模型 / Transmission Network Models

### 线路模型 / Line Models

**π型等效电路**: $Y = \frac{1}{R + jX} + j\frac{B}{2}$

**传输功率**: $P_{12} = \frac{V_1 V_2}{X} \sin(\theta_1 - \theta_2)$

**热极限**: $P_{max} = \frac{V_{rated}^2}{X} \sin(90°) = \frac{V_{rated}^2}{X}$

### 变压器模型 / Transformer Models

**变比**: $a = \frac{N_1}{N_2} = \frac{V_1}{V_2}$

**阻抗**: $Z_{pu} = \frac{Z_{actual}}{Z_{base}}$

**损耗**: $P_{loss} = I^2 R + P_{core}$

### 网络拓扑 / Network Topology

**节点导纳矩阵**: $Y_{bus} = A Y A^T$

**节点阻抗矩阵**: $Z_{bus} = Y_{bus}^{-1}$

---

## 8.3.4 配电系统模型 / Distribution System Models

### 配电网潮流 / Distribution Power Flow

**前推回代法**:

- 前推: $V_i^{(k+1)} = V_{i-1}^{(k)} - I_i^{(k)} Z_i$
- 回代: $I_i^{(k+1)} = \frac{S_i^*}{V_i^{(k+1)*}}$

### 负荷建模 / Load Modeling

**恒功率负荷**: $P = P_0, Q = Q_0$

**恒阻抗负荷**: $P = P_0 \left(\frac{V}{V_0}\right)^2$

**恒电流负荷**: $P = P_0 \left(\frac{V}{V_0}\right)$

### 电压控制 / Voltage Control

**电压调节**: $\Delta V = \frac{R \Delta P + X \Delta Q}{V}$

**电容器补偿**: $Q_c = \frac{V^2}{X_c}$

---

## 8.3.5 能源经济模型 / Energy Economics Models

### 电价模型 / Electricity Price Models

**边际成本**: $MC = \frac{\partial TC}{\partial Q}$

**电价**: $P = MC + markup$

**峰谷电价**: $P_{peak} = \alpha P_{base}, P_{valley} = \beta P_{base}$

### 投资决策 / Investment Decision

**净现值**: $NPV = \sum_{t=0}^T \frac{CF_t}{(1+r)^t}$

**内部收益率**: $\sum_{t=0}^T \frac{CF_t}{(1+IRR)^t} = 0$

**投资回收期**: $\sum_{t=0}^{PBP} CF_t = 0$

### 市场机制 / Market Mechanisms

**竞价机制**: $P = \arg\max \sum_{i=1}^n b_i(Q_i)$

**市场出清**: $\sum_{i=1}^n Q_i^{supply} = \sum_{j=1}^m Q_j^{demand}$

---

## 8.3.6 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PowerSystemModel {
    pub buses: HashMap<String, Bus>,
    pub lines: HashMap<String, Line>,
    pub generators: HashMap<String, Generator>,
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
pub struct Line {
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
    pub cost: f64,
}

impl PowerSystemModel {
    pub fn new() -> Self {
        Self {
            buses: HashMap::new(),
            lines: HashMap::new(),
            generators: HashMap::new(),
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
    
    pub fn power_flow_analysis(&self) -> HashMap<String, (f64, f64)> {
        let mut results = HashMap::new();
        
        // 简化的潮流计算
        for (bus_id, bus) in &self.buses {
            let mut total_power = 0.0;
            let mut total_reactive = 0.0;
            
            // 计算注入功率
            for (line_id, line) in &self.lines {
                if line.from_bus == *bus_id {
                    let power_flow = self.calculate_line_power_flow(line, bus);
                    total_power += power_flow.0;
                    total_reactive += power_flow.1;
                }
            }
            
            results.insert(bus_id.clone(), (total_power, total_reactive));
        }
        
        results
    }
    
    fn calculate_line_power_flow(&self, line: &Line, from_bus: &Bus) -> (f64, f64) {
        let v1 = from_bus.voltage_magnitude;
        let theta1 = from_bus.voltage_angle;
        
        // 简化的功率流计算
        let impedance = (line.resistance * line.resistance + line.reactance * line.reactance).sqrt();
        let power = (v1 * v1) / impedance;
        let reactive = power * line.reactance / impedance;
        
        (power, reactive)
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
    
    pub fn calculate_system_losses(&self) -> f64 {
        let mut total_losses = 0.0;
        
        for line in self.lines.values() {
            let current = self.calculate_line_current(line);
            let losses = current * current * line.resistance;
            total_losses += losses;
        }
        
        total_losses
    }
    
    fn calculate_line_current(&self, line: &Line) -> f64 {
        // 简化的电流计算
        let voltage_diff = 1.0; // 假设电压差
        let impedance = (line.resistance * line.resistance + line.reactance * line.reactance).sqrt();
        voltage_diff / impedance
    }
}

#[derive(Debug, Clone)]
pub struct RenewableEnergyModel {
    pub wind_speed: f64,
    pub solar_irradiance: f64,
    pub temperature: f64,
    pub wind_turbine_params: WindTurbineParams,
    pub solar_panel_params: SolarPanelParams,
}

#[derive(Debug, Clone)]
pub struct WindTurbineParams {
    pub rated_power: f64,
    pub cut_in_speed: f64,
    pub rated_speed: f64,
    pub cut_out_speed: f64,
    pub rotor_area: f64,
    pub power_coefficient: f64,
}

#[derive(Debug, Clone)]
pub struct SolarPanelParams {
    pub rated_power: f64,
    pub area: f64,
    pub efficiency: f64,
    pub temperature_coefficient: f64,
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
            },
            solar_panel_params: SolarPanelParams {
                rated_power: 1000.0,
                area: 6.0,
                efficiency: 0.15,
                temperature_coefficient: -0.004,
            },
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
}

#[derive(Debug, Clone)]
pub struct EnergyMarketModel {
    pub generators: HashMap<String, Generator>,
    pub demand: f64,
    pub market_price: f64,
}

impl EnergyMarketModel {
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
            demand: 0.0,
            market_price: 0.0,
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
}

// 使用示例
fn main() {
    // 电力系统模型示例
    let mut power_system = PowerSystemModel::new();
    
    // 添加母线
    power_system.add_bus(Bus {
        id: "Bus1".to_string(),
        voltage_magnitude: 1.0,
        voltage_angle: 0.0,
        bus_type: BusType::Slack,
        active_power: 0.0,
        reactive_power: 0.0,
    });
    
    power_system.add_bus(Bus {
        id: "Bus2".to_string(),
        voltage_magnitude: 1.0,
        voltage_angle: 0.0,
        bus_type: BusType::PQ,
        active_power: -100.0,
        reactive_power: -50.0,
    });
    
    // 添加线路
    power_system.add_line(Line {
        from_bus: "Bus1".to_string(),
        to_bus: "Bus2".to_string(),
        resistance: 0.1,
        reactance: 0.2,
        susceptance: 0.0,
        capacity: 200.0,
    });
    
    // 添加发电机
    power_system.add_generator(Generator {
        bus_id: "Bus1".to_string(),
        active_power: 0.0,
        reactive_power: 0.0,
        max_power: 500.0,
        min_power: 0.0,
        cost: 50.0,
    });
    
    let power_flow_results = power_system.power_flow_analysis();
    let economic_dispatch = power_system.economic_dispatch(100.0);
    let system_losses = power_system.calculate_system_losses();
    
    println!("电力系统模型示例:");
    println!("潮流分析结果: {:?}", power_flow_results);
    println!("经济调度: {:?}", economic_dispatch);
    println!("系统损耗: {:.2} MW", system_losses);
    
    // 新能源模型示例
    let mut renewable_model = RenewableEnergyModel::new();
    renewable_model.wind_speed = 10.0;
    renewable_model.solar_irradiance = 800.0;
    renewable_model.temperature = 30.0;
    
    let wind_power = renewable_model.calculate_wind_power();
    let solar_power = renewable_model.calculate_solar_power();
    let total_renewable_power = renewable_model.calculate_total_renewable_power();
    
    println!("\n新能源模型示例:");
    println!("风电功率: {:.2} kW", wind_power);
    println!("光伏功率: {:.2} kW", solar_power);
    println!("总可再生能源功率: {:.2} kW", total_renewable_power);
    
    // 能源市场模型示例
    let mut market = EnergyMarketModel::new();
    market.demand = 1000.0;
    
    market.add_generator(Generator {
        bus_id: "Gen1".to_string(),
        active_power: 0.0,
        reactive_power: 0.0,
        max_power: 500.0,
        min_power: 0.0,
        cost: 30.0,
    });
    
    market.add_generator(Generator {
        bus_id: "Gen2".to_string(),
        active_power: 0.0,
        reactive_power: 0.0,
        max_power: 600.0,
        min_power: 0.0,
        cost: 45.0,
    });
    
    let (clearing_price, dispatch) = market.market_clearing();
    let total_cost = market.calculate_total_cost(&dispatch);
    
    println!("\n能源市场模型示例:");
    println!("市场出清价格: {:.2} $/MWh", clearing_price);
    println!("发电调度: {:?}", dispatch);
    println!("总成本: {:.2} $", total_cost);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module PowerEnergyModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sortBy, sum)

-- 电力系统模型
data BusType = Slack | PV | PQ deriving (Show, Eq)

data Bus = Bus {
    busId :: String,
    voltageMagnitude :: Double,
    voltageAngle :: Double,
    busType :: BusType,
    activePower :: Double,
    reactivePower :: Double
} deriving Show

data Line = Line {
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
    minPower :: Double,
    cost :: Double
} deriving Show

data PowerSystemModel = PowerSystemModel {
    buses :: Map String Bus,
    lines :: Map String Line,
    generators :: Map String Generator
} deriving Show

newPowerSystemModel :: PowerSystemModel
newPowerSystemModel = PowerSystemModel {
    buses = Map.empty,
    lines = Map.empty,
    generators = Map.empty
}

addBus :: Bus -> PowerSystemModel -> PowerSystemModel
addBus bus model = model { buses = Map.insert (busId bus) bus (buses model) }

addLine :: Line -> PowerSystemModel -> PowerSystemModel
addLine line model = 
    let lineId = fromBus line ++ "-" ++ toBus line
    in model { lines = Map.insert lineId line (lines model) }

addGenerator :: Generator -> PowerSystemModel -> PowerSystemModel
addGenerator generator model = 
    model { generators = Map.insert (genBusId generator) generator (generators model) }

powerFlowAnalysis :: PowerSystemModel -> Map String (Double, Double)
powerFlowAnalysis model = 
    Map.fromList [(busId, calculateBusPower model busId) | 
                  (busId, _) <- Map.toList (buses model)]

calculateBusPower :: PowerSystemModel -> String -> (Double, Double)
calculateBusPower model busId = 
    let connectedLines = filter (\line -> fromBus line == busId) (Map.elems (lines model))
        powerFlows = map (\line -> calculateLinePowerFlow model line) connectedLines
    in foldr (\(p, q) (totalP, totalQ) -> (totalP + p, totalQ + q)) (0.0, 0.0) powerFlows

calculateLinePowerFlow :: PowerSystemModel -> Line -> (Double, Double)
calculateLinePowerFlow model line = 
    let impedance = sqrt (resistance line ^ 2 + reactance line ^ 2)
        power = 1.0 / impedance  -- 简化的功率流计算
        reactive = power * reactance line / impedance
    in (power, reactive)

economicDispatch :: PowerSystemModel -> Double -> Map String Double
economicDispatch model totalDemand = 
    let sortedGenerators = sortBy (\a b -> compare (cost a) (cost b)) (Map.elems (generators model))
    in dispatchHelper sortedGenerators totalDemand Map.empty

dispatchHelper :: [Generator] -> Double -> Map String Double -> Map String Double
dispatchHelper [] _ dispatch = dispatch
dispatchHelper (gen:gens) remainingDemand dispatch
    | remainingDemand <= 0 = Map.insert (genBusId gen) 0.0 dispatch
    | otherwise = 
        let allocatedPower = min remainingDemand (maxPower gen)
            newDispatch = Map.insert (genBusId gen) allocatedPower dispatch
        in dispatchHelper gens (remainingDemand - allocatedPower) newDispatch

calculateSystemLosses :: PowerSystemModel -> Double
calculateSystemLosses model = 
    sum [calculateLineLosses line | line <- Map.elems (lines model)]

calculateLineLosses :: Line -> Double
calculateLineLosses line = 
    let current = 1.0 / sqrt (resistance line ^ 2 + reactance line ^ 2)
    in current ^ 2 * resistance line

-- 新能源模型
data WindTurbineParams = WindTurbineParams {
    ratedPower :: Double,
    cutInSpeed :: Double,
    ratedSpeed :: Double,
    cutOutSpeed :: Double,
    rotorArea :: Double,
    powerCoefficient :: Double
} deriving Show

data SolarPanelParams = SolarPanelParams {
    panelRatedPower :: Double,
    area :: Double,
    efficiency :: Double,
    temperatureCoefficient :: Double
} deriving Show

data RenewableEnergyModel = RenewableEnergyModel {
    windSpeed :: Double,
    solarIrradiance :: Double,
    temperature :: Double,
    windTurbineParams :: WindTurbineParams,
    solarPanelParams :: SolarPanelParams
} deriving Show

newRenewableEnergyModel :: RenewableEnergyModel
newRenewableEnergyModel = RenewableEnergyModel {
    windSpeed = 0.0,
    solarIrradiance = 0.0,
    temperature = 25.0,
    windTurbineParams = WindTurbineParams {
        ratedPower = 2000.0,
        cutInSpeed = 3.0,
        ratedSpeed = 12.0,
        cutOutSpeed = 25.0,
        rotorArea = 1000.0,
        powerCoefficient = 0.4
    },
    solarPanelParams = SolarPanelParams {
        panelRatedPower = 1000.0,
        area = 6.0,
        efficiency = 0.15,
        temperatureCoefficient = -0.004
    }
}

calculateWindPower :: RenewableEnergyModel -> Double
calculateWindPower model = 
    let windSpeed = windSpeed model
        params = windTurbineParams model
    in if windSpeed < cutInSpeed params || windSpeed > cutOutSpeed params
       then 0.0
       else if windSpeed < ratedSpeed params
            then let power = 0.5 * 1.225 * rotorArea params * windSpeed ^ 3 * powerCoefficient params
                 in min power (ratedPower params)
            else ratedPower params

calculateSolarPower :: RenewableEnergyModel -> Double
calculateSolarPower model = 
    let irradiance = solarIrradiance model
        temperature = temperature model
        params = solarPanelParams model
        temperatureFactor = 1.0 + temperatureCoefficient params * (temperature - 25.0)
        power = irradiance * area params * efficiency params * temperatureFactor
    in min power (panelRatedPower params)

calculateTotalRenewablePower :: RenewableEnergyModel -> Double
calculateTotalRenewablePower model = 
    calculateWindPower model + calculateSolarPower model

-- 能源市场模型
data EnergyMarketModel = EnergyMarketModel {
    marketGenerators :: Map String Generator,
    demand :: Double,
    marketPrice :: Double
} deriving Show

newEnergyMarketModel :: EnergyMarketModel
newEnergyMarketModel = EnergyMarketModel {
    marketGenerators = Map.empty,
    demand = 0.0,
    marketPrice = 0.0
}

addMarketGenerator :: Generator -> EnergyMarketModel -> EnergyMarketModel
addMarketGenerator generator model = 
    model { marketGenerators = Map.insert (genBusId generator) generator (marketGenerators model) }

marketClearing :: EnergyMarketModel -> (Double, Map String Double)
marketClearing model = 
    let sortedGenerators = sortBy (\a b -> compare (cost a) (cost b)) (Map.elems (marketGenerators model))
    in clearingHelper sortedGenerators (demand model) Map.empty 0.0

clearingHelper :: [Generator] -> Double -> Map String Double -> Double -> (Double, Map String Double)
clearingHelper [] _ dispatch price = (price, dispatch)
clearingHelper (gen:gens) remainingDemand dispatch price
    | remainingDemand <= 0 = (price, Map.insert (genBusId gen) 0.0 dispatch)
    | otherwise = 
        let allocatedPower = min remainingDemand (maxPower gen)
            newDispatch = Map.insert (genBusId gen) allocatedPower dispatch
            newPrice = cost gen
        in clearingHelper gens (remainingDemand - allocatedPower) newDispatch newPrice

calculateTotalCost :: EnergyMarketModel -> Map String Double -> Double
calculateTotalCost model dispatch = 
    sum [cost * power | (genId, power) <- Map.toList dispatch,
                       let gen = marketGenerators model Map.! genId,
                       let cost = PowerEnergyModels.cost gen]

-- 示例使用
example :: IO ()
example = do
    -- 电力系统模型示例
    let powerSystem = addGenerator (Generator "Bus1" 0.0 0.0 500.0 0.0 50.0) $
                     addLine (Line "Bus1" "Bus2" 0.1 0.2 0.0 200.0) $
                     addBus (Bus "Bus2" 1.0 0.0 PQ (-100.0) (-50.0)) $
                     addBus (Bus "Bus1" 1.0 0.0 Slack 0.0 0.0) $
                     newPowerSystemModel
    
    let powerFlowResults = powerFlowAnalysis powerSystem
        economicDispatch = economicDispatch powerSystem 100.0
        systemLosses = calculateSystemLosses powerSystem
    
    putStrLn "电力系统模型示例:"
    putStrLn $ "潮流分析结果: " ++ show powerFlowResults
    putStrLn $ "经济调度: " ++ show economicDispatch
    putStrLn $ "系统损耗: " ++ show systemLosses ++ " MW"
    
    -- 新能源模型示例
    let renewableModel = (newRenewableEnergyModel) {
        windSpeed = 10.0,
        solarIrradiance = 800.0,
        temperature = 30.0
    }
    
    let windPower = calculateWindPower renewableModel
        solarPower = calculateSolarPower renewableModel
        totalRenewablePower = calculateTotalRenewablePower renewableModel
    
    putStrLn "\n新能源模型示例:"
    putStrLn $ "风电功率: " ++ show windPower ++ " kW"
    putStrLn $ "光伏功率: " ++ show solarPower ++ " kW"
    putStrLn $ "总可再生能源功率: " ++ show totalRenewablePower ++ " kW"
    
    -- 能源市场模型示例
    let market = addMarketGenerator (Generator "Gen2" 0.0 0.0 600.0 0.0 45.0) $
                 addMarketGenerator (Generator "Gen1" 0.0 0.0 500.0 0.0 30.0) $
                 (newEnergyMarketModel) { demand = 1000.0 }
    
    let (clearingPrice, dispatch) = marketClearing market
        totalCost = calculateTotalCost market dispatch
    
    putStrLn "\n能源市场模型示例:"
    putStrLn $ "市场出清价格: " ++ show clearingPrice ++ " $/MWh"
    putStrLn $ "发电调度: " ++ show dispatch
    putStrLn $ "总成本: " ++ show totalCost ++ " $"
```

### 应用领域 / Application Domains

#### 电网规划 / Grid Planning

- **网络扩展**: 输电线路规划、变电站选址
- **容量规划**: 发电容量规划、负荷预测
- **投资决策**: 经济性分析、风险评估

#### 运行调度 / Operation Dispatch

- **经济调度**: 发电机组组合、负荷分配
- **安全运行**: 稳定性分析、故障处理
- **实时控制**: 自动发电控制、电压控制

#### 市场交易 / Market Trading

- **电力市场**: 竞价机制、市场出清
- **辅助服务**: 调频、调峰、备用容量
- **价格机制**: 峰谷电价、实时电价

---

## 参考文献 / References

1. Kundur, P. (1994). Power system stability and control. McGraw-Hill.
2. Wood, A. J., & Wollenberg, B. F. (2012). Power generation, operation, and control. Wiley.
3. Saadat, H. (2011). Power system analysis. McGraw-Hill.
4. Kirschen, D. S., & Strbac, G. (2018). Fundamentals of power system economics. Wiley.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
