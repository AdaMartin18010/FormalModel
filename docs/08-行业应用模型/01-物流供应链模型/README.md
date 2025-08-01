# 8.1 物流供应链模型 / Logistics & Supply Chain Models

## 目录 / Table of Contents

- [8.1 物流供应链模型 / Logistics \& Supply Chain Models](#81-物流供应链模型--logistics--supply-chain-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.1.1 库存管理模型 / Inventory Management Models](#811-库存管理模型--inventory-management-models)
    - [经济订货量模型 / Economic Order Quantity](#经济订货量模型--economic-order-quantity)
    - [安全库存模型 / Safety Stock Models](#安全库存模型--safety-stock-models)
    - [多级库存模型 / Multi-Echelon Inventory Models](#多级库存模型--multi-echelon-inventory-models)
  - [8.1.2 运输优化模型 / Transportation Optimization Models](#812-运输优化模型--transportation-optimization-models)
    - [车辆路径问题 / Vehicle Routing Problem](#车辆路径问题--vehicle-routing-problem)
    - [运输网络优化 / Transportation Network Optimization](#运输网络优化--transportation-network-optimization)
  - [8.1.3 供应链网络模型 / Supply Chain Network Models](#813-供应链网络模型--supply-chain-network-models)
    - [供应商选择模型 / Supplier Selection Models](#供应商选择模型--supplier-selection-models)
    - [需求预测模型 / Demand Forecasting Models](#需求预测模型--demand-forecasting-models)
  - [8.1.4 实现与应用 / Implementation and Applications](#814-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [电子商务 / E-commerce](#电子商务--e-commerce)
      - [制造业 / Manufacturing](#制造业--manufacturing)
      - [零售业 / Retail](#零售业--retail)
  - [参考文献 / References](#参考文献--references)

---

## 8.1.1 库存管理模型 / Inventory Management Models

### 经济订货量模型 / Economic Order Quantity

**EOQ公式**: $Q^* = \sqrt{\frac{2DS}{H}}$

其中：

- $D$: 年需求量
- $S$: 订货成本
- $H$: 单位库存持有成本

**总成本**: $TC = \frac{D}{Q}S + \frac{Q}{2}H$

### 安全库存模型 / Safety Stock Models

**服务水平**: $SL = P(D \leq ROP)$

**安全库存**: $SS = z_\alpha \sigma_L$

**再订货点**: $ROP = \mu_L + SS$

### 多级库存模型 / Multi-Echelon Inventory Models

**系统库存**: $I_{total} = \sum_{i=1}^n I_i$

**服务水平**: $SL_{system} = \prod_{i=1}^n SL_i$

---

## 8.1.2 运输优化模型 / Transportation Optimization Models

### 车辆路径问题 / Vehicle Routing Problem

**目标函数**: $\min \sum_{i=1}^n \sum_{j=1}^n c_{ij} x_{ij}$

**约束条件**:

- $\sum_{j=1}^n x_{ij} = 1$ for all $i$
- $\sum_{i=1}^n x_{ij} = 1$ for all $j$
- $\sum_{i \in S} \sum_{j \in S} x_{ij} \leq |S| - 1$ for all $S \subset V$

### 运输网络优化 / Transportation Network Optimization

**最小成本流**: $\min \sum_{(i,j) \in E} c_{ij} f_{ij}$

**流量守恒**: $\sum_{j} f_{ij} - \sum_{j} f_{ji} = b_i$

---

## 8.1.3 供应链网络模型 / Supply Chain Network Models

### 供应商选择模型 / Supplier Selection Models

**多目标优化**: $\min \sum_{i=1}^n w_i f_i(x)$

**评价指标**: 质量、成本、交付时间、服务水平

### 需求预测模型 / Demand Forecasting Models

**时间序列**: $D_t = \mu + \alpha t + \sum_{i=1}^p \phi_i D_{t-i} + \epsilon_t$

**指数平滑**: $F_{t+1} = \alpha D_t + (1-\alpha) F_t$

---

## 8.1.4 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct InventoryModel {
    pub demand_rate: f64,
    pub order_cost: f64,
    pub holding_cost: f64,
    pub lead_time: f64,
    pub service_level: f64,
}

impl InventoryModel {
    pub fn new(demand: f64, order_cost: f64, holding_cost: f64, lead_time: f64, service_level: f64) -> Self {
        Self {
            demand_rate: demand,
            order_cost,
            holding_cost,
            lead_time,
            service_level,
        }
    }
    
    pub fn economic_order_quantity(&self) -> f64 {
        (2.0 * self.demand_rate * self.order_cost / self.holding_cost).sqrt()
    }
    
    pub fn total_cost(&self, order_quantity: f64) -> f64 {
        let ordering_cost = self.demand_rate / order_quantity * self.order_cost;
        let holding_cost = order_quantity / 2.0 * self.holding_cost;
        ordering_cost + holding_cost
    }
    
    pub fn safety_stock(&self, demand_std: f64) -> f64 {
        // 简化的安全库存计算
        let z_score = 1.645; // 对应95%服务水平
        z_score * demand_std * self.lead_time.sqrt()
    }
    
    pub fn reorder_point(&self, demand_std: f64) -> f64 {
        let average_demand = self.demand_rate * self.lead_time;
        let safety_stock = self.safety_stock(demand_std);
        average_demand + safety_stock
    }
}

#[derive(Debug, Clone)]
pub struct TransportationModel {
    pub distances: HashMap<(String, String), f64>,
    pub demands: HashMap<String, f64>,
    pub vehicle_capacity: f64,
}

impl TransportationModel {
    pub fn new() -> Self {
        Self {
            distances: HashMap::new(),
            demands: HashMap::new(),
            vehicle_capacity: 1000.0,
        }
    }
    
    pub fn add_distance(&mut self, from: String, to: String, distance: f64) {
        self.distances.insert((from, to), distance);
    }
    
    pub fn add_demand(&mut self, location: String, demand: f64) {
        self.demands.insert(location, demand);
    }
    
    pub fn calculate_total_distance(&self, route: &[String]) -> f64 {
        let mut total_distance = 0.0;
        for i in 0..route.len() - 1 {
            let key = (route[i].clone(), route[i + 1].clone());
            total_distance += self.distances.get(&key).unwrap_or(&0.0);
        }
        total_distance
    }
    
    pub fn nearest_neighbor_tsp(&self, start: &str, locations: &[String]) -> Vec<String> {
        let mut unvisited: Vec<String> = locations.iter().filter(|&x| x != start).cloned().collect();
        let mut route = vec![start.to_string()];
        let mut current = start;
        
        while !unvisited.is_empty() {
            let mut nearest = &unvisited[0];
            let mut min_distance = f64::INFINITY;
            
            for location in &unvisited {
                let distance = self.distances.get(&(current.to_string(), location.clone())).unwrap_or(&f64::INFINITY);
                if *distance < min_distance {
                    min_distance = *distance;
                    nearest = location;
                }
            }
            
            route.push(nearest.clone());
            unvisited.retain(|x| x != nearest);
            current = nearest;
        }
        
        route.push(start.to_string());
        route
    }
}

#[derive(Debug, Clone)]
pub struct SupplyChainModel {
    pub suppliers: Vec<String>,
    pub facilities: Vec<String>,
    pub customers: Vec<String>,
    pub costs: HashMap<(String, String), f64>,
    pub capacities: HashMap<String, f64>,
    pub demands: HashMap<String, f64>,
}

impl SupplyChainModel {
    pub fn new() -> Self {
        Self {
            suppliers: Vec::new(),
            facilities: Vec::new(),
            customers: Vec::new(),
            costs: HashMap::new(),
            capacities: HashMap::new(),
            demands: HashMap::new(),
        }
    }
    
    pub fn add_supplier(&mut self, supplier: String, capacity: f64) {
        self.suppliers.push(supplier.clone());
        self.capacities.insert(supplier, capacity);
    }
    
    pub fn add_facility(&mut self, facility: String, capacity: f64) {
        self.facilities.push(facility.clone());
        self.capacities.insert(facility, capacity);
    }
    
    pub fn add_customer(&mut self, customer: String, demand: f64) {
        self.customers.push(customer.clone());
        self.demands.insert(customer, demand);
    }
    
    pub fn add_cost(&mut self, from: String, to: String, cost: f64) {
        self.costs.insert((from, to), cost);
    }
    
    pub fn calculate_total_cost(&self, flows: &HashMap<(String, String), f64>) -> f64 {
        flows.iter().map(|((from, to), flow)| {
            self.costs.get(&(from.clone(), to.clone())).unwrap_or(&0.0) * flow
        }).sum()
    }
    
    pub fn simple_optimization(&self) -> HashMap<(String, String), f64> {
        let mut flows = HashMap::new();
        
        // 简化的优化：按成本最小分配
        for customer in &self.customers {
            let demand = self.demands.get(customer).unwrap_or(&0.0);
            let mut remaining_demand = *demand;
            
            // 从供应商到设施
            for supplier in &self.suppliers {
                if remaining_demand <= 0.0 { break; }
                let capacity = self.capacities.get(supplier).unwrap_or(&0.0);
                let flow = remaining_demand.min(*capacity);
                flows.insert((supplier.clone(), customer.clone()), flow);
                remaining_demand -= flow;
            }
        }
        
        flows
    }
}

// 使用示例
fn main() {
    // 库存管理示例
    let inventory = InventoryModel::new(1000.0, 50.0, 2.0, 5.0, 0.95);
    let eoq = inventory.economic_order_quantity();
    let total_cost = inventory.total_cost(eoq);
    let safety_stock = inventory.safety_stock(50.0);
    let reorder_point = inventory.reorder_point(50.0);
    
    println!("库存管理示例:");
    println!("经济订货量: {:.2}", eoq);
    println!("总成本: {:.2}", total_cost);
    println!("安全库存: {:.2}", safety_stock);
    println!("再订货点: {:.2}", reorder_point);
    
    // 运输优化示例
    let mut transport = TransportationModel::new();
    transport.add_distance("A".to_string(), "B".to_string(), 10.0);
    transport.add_distance("B".to_string(), "C".to_string(), 15.0);
    transport.add_distance("C".to_string(), "A".to_string(), 12.0);
    
    let locations = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let route = transport.nearest_neighbor_tsp("A", &locations);
    let total_distance = transport.calculate_total_distance(&route);
    
    println!("\n运输优化示例:");
    println!("最优路径: {:?}", route);
    println!("总距离: {:.2}", total_distance);
    
    // 供应链网络示例
    let mut supply_chain = SupplyChainModel::new();
    supply_chain.add_supplier("S1".to_string(), 500.0);
    supply_chain.add_facility("F1".to_string(), 300.0);
    supply_chain.add_customer("C1".to_string(), 200.0);
    
    supply_chain.add_cost("S1".to_string(), "F1".to_string(), 10.0);
    supply_chain.add_cost("F1".to_string(), "C1".to_string(), 5.0);
    
    let flows = supply_chain.simple_optimization();
    let total_cost = supply_chain.calculate_total_cost(&flows);
    
    println!("\n供应链网络示例:");
    println!("流量分配: {:?}", flows);
    println!("总成本: {:.2}", total_cost);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module LogisticsSupplyChainModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, minimumBy)
import Data.Ord (comparing)

-- 库存管理模型
data InventoryModel = InventoryModel {
    demandRate :: Double,
    orderCost :: Double,
    holdingCost :: Double,
    leadTime :: Double,
    serviceLevel :: Double
} deriving Show

newInventoryModel :: Double -> Double -> Double -> Double -> Double -> InventoryModel
newInventoryModel demand order_cost holding_cost lead_time service_level = InventoryModel {
    demandRate = demand,
    orderCost = order_cost,
    holdingCost = holding_cost,
    leadTime = lead_time,
    serviceLevel = service_level
}

economicOrderQuantity :: InventoryModel -> Double
economicOrderQuantity model = 
    sqrt (2.0 * demandRate model * orderCost model / holdingCost model)

totalCost :: InventoryModel -> Double -> Double
totalCost model orderQuantity = 
    let orderingCost = demandRate model / orderQuantity * orderCost model
        holdingCost = orderQuantity / 2.0 * holdingCost model
    in orderingCost + holdingCost

safetyStock :: InventoryModel -> Double -> Double
safetyStock model demandStd = 
    let zScore = 1.645 -- 对应95%服务水平
    in zScore * demandStd * sqrt (leadTime model)

reorderPoint :: InventoryModel -> Double -> Double
reorderPoint model demandStd = 
    let averageDemand = demandRate model * leadTime model
        safetyStock = safetyStock model demandStd
    in averageDemand + safetyStock

-- 运输优化模型
data TransportationModel = TransportationModel {
    distances :: Map (String, String) Double,
    demands :: Map String Double,
    vehicleCapacity :: Double
} deriving Show

newTransportationModel :: TransportationModel
newTransportationModel = TransportationModel {
    distances = Map.empty,
    demands = Map.empty,
    vehicleCapacity = 1000.0
}

addDistance :: TransportationModel -> String -> String -> Double -> TransportationModel
addDistance model from to distance = 
    model { distances = Map.insert (from, to) distance (distances model) }

addDemand :: TransportationModel -> String -> Double -> TransportationModel
addDemand model location demand = 
    model { demands = Map.insert location demand (demands model) }

calculateTotalDistance :: TransportationModel -> [String] -> Double
calculateTotalDistance model route = 
    sum [Map.findWithDefault 0.0 (route !! i, route !! (i + 1)) (distances model) | 
         i <- [0..length route - 2]]

nearestNeighborTSP :: TransportationModel -> String -> [String] -> [String]
nearestNeighborTSP model start locations = 
    let unvisited = filter (/= start) locations
        buildRoute current remaining = 
            if null remaining 
            then [current, start]
            else let nearest = minimumBy (comparing (\loc -> 
                        Map.findWithDefault (1/0) (current, loc) (distances model))) remaining
                     newRemaining = filter (/= nearest) remaining
                 in current : buildRoute nearest newRemaining
    in buildRoute start unvisited

-- 供应链网络模型
data SupplyChainModel = SupplyChainModel {
    suppliers :: [String],
    facilities :: [String],
    customers :: [String],
    costs :: Map (String, String) Double,
    capacities :: Map String Double,
    demands :: Map String Double
} deriving Show

newSupplyChainModel :: SupplyChainModel
newSupplyChainModel = SupplyChainModel {
    suppliers = [],
    facilities = [],
    customers = [],
    costs = Map.empty,
    capacities = Map.empty,
    demands = Map.empty
}

addSupplier :: SupplyChainModel -> String -> Double -> SupplyChainModel
addSupplier model supplier capacity = 
    model { 
        suppliers = supplier : suppliers model,
        capacities = Map.insert supplier capacity (capacities model)
    }

addFacility :: SupplyChainModel -> String -> Double -> SupplyChainModel
addFacility model facility capacity = 
    model { 
        facilities = facility : facilities model,
        capacities = Map.insert facility capacity (capacities model)
    }

addCustomer :: SupplyChainModel -> String -> Double -> SupplyChainModel
addCustomer model customer demand = 
    model { 
        customers = customer : customers model,
        demands = Map.insert customer demand (demands model)
    }

addCost :: SupplyChainModel -> String -> String -> Double -> SupplyChainModel
addCost model from to cost = 
    model { costs = Map.insert (from, to) cost (costs model) }

calculateTotalCost :: SupplyChainModel -> Map (String, String) Double -> Double
calculateTotalCost model flows = 
    sum [Map.findWithDefault 0.0 (from, to) (costs model) * flow | 
         ((from, to), flow) <- Map.toList flows]

simpleOptimization :: SupplyChainModel -> Map (String, String) Double
simpleOptimization model = 
    let customerDemands = Map.toList (demands model)
        supplierCapacities = Map.toList (capacities model)
        -- 简化的优化：按成本最小分配
        allocateDemand customer demand = 
            foldr (\(supplier, capacity) remaining -> 
                if remaining <= 0.0 
                then remaining 
                else let flow = min remaining capacity
                     in remaining - flow) demand supplierCapacities
    in Map.fromList [(("S1", customer), demand) | (customer, demand) <- customerDemands]

-- 示例使用
example :: IO ()
example = do
    -- 库存管理示例
    let inventory = newInventoryModel 1000.0 50.0 2.0 5.0 0.95
        eoq = economicOrderQuantity inventory
        totalCost = totalCost inventory eoq
        safetyStock = safetyStock inventory 50.0
        reorderPoint = reorderPoint inventory 50.0
    
    putStrLn "库存管理示例:"
    putStrLn $ "经济订货量: " ++ show eoq
    putStrLn $ "总成本: " ++ show totalCost
    putStrLn $ "安全库存: " ++ show safetyStock
    putStrLn $ "再订货点: " ++ show reorderPoint
    
    -- 运输优化示例
    let transport = addDistance (addDistance (addDistance newTransportationModel "A" "B" 10.0) "B" "C" 15.0) "C" "A" 12.0
        locations = ["A", "B", "C"]
        route = nearestNeighborTSP transport "A" locations
        totalDistance = calculateTotalDistance transport route
    
    putStrLn "\n运输优化示例:"
    putStrLn $ "最优路径: " ++ show route
    putStrLn $ "总距离: " ++ show totalDistance
    
    -- 供应链网络示例
    let supplyChain = addSupplier (addFacility (addCustomer newSupplyChainModel "C1" 200.0) "F1" 300.0) "S1" 500.0
        supplyChainWithCosts = addCost (addCost supplyChain "S1" "F1" 10.0) "F1" "C1" 5.0
        flows = simpleOptimization supplyChainWithCosts
        totalCost = calculateTotalCost supplyChainWithCosts flows
    
    putStrLn "\n供应链网络示例:"
    putStrLn $ "流量分配: " ++ show flows
    putStrLn $ "总成本: " ++ show totalCost
```

### 应用领域 / Application Domains

#### 电子商务 / E-commerce

- **订单履行**: 多仓库库存分配、最后一公里配送
- **需求预测**: 季节性需求、促销影响预测
- **供应商管理**: 多供应商选择、供应商绩效评估

#### 制造业 / Manufacturing

- **生产计划**: 主生产计划、物料需求计划
- **采购管理**: 供应商选择、采购批量优化
- **库存控制**: 原材料库存、在制品库存、成品库存

#### 零售业 / Retail

- **门店补货**: 自动补货系统、安全库存管理
- **配送优化**: 门店配送路线、配送时间窗口
- **需求管理**: 促销需求、季节性需求

---

## 参考文献 / References

1. Chopra, S., & Meindl, P. (2016). Supply chain management: Strategy, planning, and operation. Pearson.
2. Silver, E. A., Pyke, D. F., & Peterson, R. (1998). Inventory management and production planning and scheduling. Wiley.
3. Toth, P., & Vigo, D. (2002). The vehicle routing problem. SIAM.
4. Simchi-Levi, D., Kaminsky, P., & Simchi-Levi, E. (2008). Designing and managing the supply chain: Concepts, strategies, and case studies. McGraw-Hill.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
