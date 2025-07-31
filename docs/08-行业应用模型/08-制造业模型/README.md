# 8.8 制造业模型 / Manufacturing Models

## 目录 / Table of Contents

- [8.8 制造业模型 / Manufacturing Models](#88-制造业模型--manufacturing-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.8.1 生产计划模型 / Production Planning Models](#881-生产计划模型--production-planning-models)
    - [主生产计划 (MPS) / Master Production Schedule](#主生产计划-mps--master-production-schedule)
    - [物料需求计划 (MRP) / Material Requirements Planning](#物料需求计划-mrp--material-requirements-planning)
    - [产能规划 / Capacity Planning](#产能规划--capacity-planning)
  - [8.8.2 质量控制模型 / Quality Control Models](#882-质量控制模型--quality-control-models)
    - [统计过程控制 (SPC) / Statistical Process Control](#统计过程控制-spc--statistical-process-control)
    - [抽样检验 / Sampling Inspection](#抽样检验--sampling-inspection)
    - [六西格玛 / Six Sigma](#六西格玛--six-sigma)
  - [8.8.3 供应链管理模型 / Supply Chain Management Models](#883-供应链管理模型--supply-chain-management-models)
    - [库存管理 / Inventory Management](#库存管理--inventory-management)
    - [供应商选择 / Supplier Selection](#供应商选择--supplier-selection)
    - [网络设计 / Network Design](#网络设计--network-design)
  - [8.8.4 精益生产模型 / Lean Manufacturing Models](#884-精益生产模型--lean-manufacturing-models)
    - [价值流图 / Value Stream Mapping](#价值流图--value-stream-mapping)
    - [看板系统 / Kanban System](#看板系统--kanban-system)
    - [5S管理 / 5S Management](#5s管理--5s-management)
  - [8.8.5 智能制造模型 / Smart Manufacturing Models](#885-智能制造模型--smart-manufacturing-models)
    - [数字孪生 / Digital Twin](#数字孪生--digital-twin)
    - [工业物联网 (IIoT) / Industrial Internet of Things](#工业物联网-iiot--industrial-internet-of-things)
    - [机器学习应用 / Machine Learning Applications](#机器学习应用--machine-learning-applications)
  - [8.8.6 设备维护模型 / Equipment Maintenance Models](#886-设备维护模型--equipment-maintenance-models)
    - [预防性维护 / Preventive Maintenance](#预防性维护--preventive-maintenance)
    - [预测性维护 / Predictive Maintenance](#预测性维护--predictive-maintenance)
    - [可靠性分析 / Reliability Analysis](#可靠性分析--reliability-analysis)
  - [8.8.7 成本控制模型 / Cost Control Models](#887-成本控制模型--cost-control-models)
    - [标准成本法 / Standard Costing](#标准成本法--standard-costing)
    - [作业成本法 (ABC) / Activity-Based Costing](#作业成本法-abc--activity-based-costing)
    - [目标成本法 / Target Costing](#目标成本法--target-costing)
  - [8.8.8 实现与应用 / Implementation and Applications](#888-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [生产管理 / Production Management](#生产管理--production-management)
      - [质量管理 / Quality Management](#质量管理--quality-management)
      - [供应链管理 / Supply Chain Management](#供应链管理--supply-chain-management)
  - [参考文献 / References](#参考文献--references)

---

## 8.8.1 生产计划模型 / Production Planning Models

### 主生产计划 (MPS) / Master Production Schedule

**目标函数**: $\min \sum_{t=1}^T \sum_{i=1}^n [c_i x_{it} + h_i I_{it} + s_i y_{it}]$

**约束条件**:

- 需求满足: $I_{i,t-1} + x_{it} - I_{it} = d_{it}$
- 产能限制: $\sum_{i=1}^n a_{ij} x_{it} \leq C_{jt}$
- 生产启动: $x_{it} \leq M y_{it}$

其中：

- $x_{it}$: 产品 $i$ 在时期 $t$ 的生产量
- $I_{it}$: 产品 $i$ 在时期 $t$ 的库存量
- $y_{it}$: 产品 $i$ 在时期 $t$ 是否生产 (0/1)
- $c_i$: 单位生产成本
- $h_i$: 单位库存持有成本
- $s_i$: 生产启动成本

### 物料需求计划 (MRP) / Material Requirements Planning

**总需求**: $GR_{it} = d_{it} + \sum_{j=1}^n r_{ij} x_{jt}$

**净需求**: $NR_{it} = \max(0, GR_{it} - I_{i,t-1})$

**计划订单**: $PO_{it} = \lceil \frac{NR_{it}}{LS_i} \rceil \cdot LS_i$

其中：

- $r_{ij}$: 产品 $i$ 对物料 $j$ 的需求系数
- $LS_i$: 产品 $i$ 的批量大小

### 产能规划 / Capacity Planning

**产能需求**: $CR_{jt} = \sum_{i=1}^n a_{ij} x_{it}$

**产能利用率**: $CU_{jt} = \frac{CR_{jt}}{C_{jt}}$

**产能平衡**: $\sum_{j=1}^m w_j CU_{jt} \leq 1$

---

## 8.8.2 质量控制模型 / Quality Control Models

### 统计过程控制 (SPC) / Statistical Process Control

**控制图**: $UCL = \mu + 3\sigma$, $LCL = \mu - 3\sigma$

**X-bar图**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

**R图**: $R = \max(x_i) - \min(x_i)$

**过程能力指数**: $C_p = \frac{USL - LSL}{6\sigma}$

**过程能力指数**: $C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right)$

### 抽样检验 / Sampling Inspection

**接收概率**: $P_a = \sum_{d=0}^c \binom{n}{d} p^d (1-p)^{n-d}$

**OC曲线**: 接收概率与不合格品率的关系

**平均检出质量**: $AOQ = p \cdot P_a$

**平均检验数**: $ASN = n \cdot P_a + N \cdot (1-P_a)$

### 六西格玛 / Six Sigma

**缺陷率**: $DPMO = \frac{\text{缺陷数} \times 1,000,000}{\text{机会数}}$

**西格玛水平**: $\sigma = \text{norminv}(1 - \frac{DPMO}{1,000,000})$

**过程改进**: $Y = f(X_1, X_2, \ldots, X_n)$

---

## 8.8.3 供应链管理模型 / Supply Chain Management Models

### 库存管理 / Inventory Management

**经济订货量 (EOQ)**: $Q^* = \sqrt{\frac{2DS}{H}}$

**安全库存**: $SS = z \cdot \sigma_L \cdot \sqrt{L}$

**再订货点**: $ROP = \mu_L + SS$

**总成本**: $TC = \frac{D}{Q} S + \frac{Q}{2} H + DC$

其中：

- $D$: 年需求量
- $S$: 订货成本
- $H$: 单位库存持有成本
- $L$: 提前期
- $z$: 服务水平对应的标准正态分位数

### 供应商选择 / Supplier Selection

**多目标优化**:
$$\min \sum_{i=1}^n w_i f_i(x)$$

**目标函数**:

- 成本最小化: $f_1(x) = \sum_{j=1}^m c_j x_j$
- 质量最大化: $f_2(x) = \sum_{j=1}^m q_j x_j$
- 交付时间最小化: $f_3(x) = \sum_{j=1}^m t_j x_j$

**约束条件**: $\sum_{j=1}^m x_j = 1$, $x_j \geq 0$

### 网络设计 / Network Design

**设施选址**: $\min \sum_{i=1}^n \sum_{j=1}^m c_{ij} x_{ij} + \sum_{j=1}^m f_j y_j$

**约束条件**:

- 需求满足: $\sum_{j=1}^m x_{ij} = d_i$
- 容量限制: $\sum_{i=1}^n x_{ij} \leq C_j y_j$
- 设施选择: $y_j \in \{0,1\}$

---

## 8.8.4 精益生产模型 / Lean Manufacturing Models

### 价值流图 / Value Stream Mapping

**增值时间**: $VA = \sum_{i=1}^n t_{va,i}$

**非增值时间**: $NVA = \sum_{i=1}^n t_{nva,i}$

**总周期时间**: $TCT = VA + NVA$

**价值流效率**: $\eta = \frac{VA}{TCT}$

### 看板系统 / Kanban System

**看板数量**: $K = \frac{D \cdot L \cdot (1 + \alpha)}{C}$

**看板循环**: $T_k = \frac{K}{D}$

**库存水平**: $I = K \cdot C$

其中：

- $D$: 日需求量
- $L$: 提前期
- $\alpha$: 安全系数
- $C$: 容器容量

### 5S管理 / 5S Management

**整理 (Sort)**: 区分必要和不必要的物品

**整顿 (Set)**: 物品定位、标识

**清扫 (Shine)**: 清洁工作环境

**清洁 (Standardize)**: 标准化清洁程序

**素养 (Sustain)**: 维持良好习惯

---

## 8.8.5 智能制造模型 / Smart Manufacturing Models

### 数字孪生 / Digital Twin

**物理模型**: $M_p = f_p(x, t)$

**数字模型**: $M_d = f_d(x, t, \theta)$

**模型校准**: $\min \sum_{i=1}^n (M_p^i - M_d^i)^2$

**预测模型**: $\hat{y} = f(x, \theta) + \epsilon$

### 工业物联网 (IIoT) / Industrial Internet of Things

**传感器数据**: $s(t) = f(x(t)) + \eta(t)$

**数据融合**: $\hat{x} = \arg\min_x \sum_{i=1}^n w_i (s_i - f_i(x))^2$

**异常检测**: $a(t) = \begin{cases} 1 & \text{if } |s(t) - \hat{s}(t)| > \tau \\ 0 & \text{otherwise} \end{cases}$

### 机器学习应用 / Machine Learning Applications

**预测性维护**: $P(failure|t) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}$

**质量预测**: $y = f(x_1, x_2, \ldots, x_n) + \epsilon$

**生产优化**: $\max f(x)$ s.t. $g_i(x) \leq 0$

---

## 8.8.6 设备维护模型 / Equipment Maintenance Models

### 预防性维护 / Preventive Maintenance

**维护间隔**: $T^* = \sqrt{\frac{2C_p}{C_f \lambda}}$

**总成本**: $TC = \frac{C_p}{T} + \frac{C_f \lambda T}{2}$

**可用性**: $A = \frac{MTBF}{MTBF + MTTR}$

其中：

- $C_p$: 预防性维护成本
- $C_f$: 故障维修成本
- $\lambda$: 故障率
- $MTBF$: 平均故障间隔时间
- $MTTR$: 平均修复时间

### 预测性维护 / Predictive Maintenance

**剩余寿命**: $RUL = \frac{1}{\lambda(t)} \int_t^\infty R(\tau) d\tau$

**健康指标**: $HI = \sum_{i=1}^n w_i \cdot f_i(x_i)$

**维护决策**: $a(t) = \begin{cases} \text{维护} & \text{if } HI(t) < \tau \\ \text{继续运行} & \text{otherwise} \end{cases}$

### 可靠性分析 / Reliability Analysis

**故障率函数**: $\lambda(t) = \frac{f(t)}{R(t)}$

**累积故障率**: $\Lambda(t) = \int_0^t \lambda(\tau) d\tau$

**可靠度函数**: $R(t) = e^{-\Lambda(t)}$

---

## 8.8.7 成本控制模型 / Cost Control Models

### 标准成本法 / Standard Costing

**标准成本**: $SC = \sum_{i=1}^n q_i \cdot p_i$

**实际成本**: $AC = \sum_{i=1}^n q_i' \cdot p_i'$

**成本差异**: $\Delta C = AC - SC$

**价格差异**: $\Delta P = \sum_{i=1}^n q_i' \cdot (p_i' - p_i)$

**数量差异**: $\Delta Q = \sum_{i=1}^n p_i \cdot (q_i' - q_i)$

### 作业成本法 (ABC) / Activity-Based Costing

**作业成本**: $AC_j = \sum_{i=1}^n r_{ij} \cdot c_i$

**产品成本**: $PC_k = \sum_{j=1}^m a_{jk} \cdot AC_j$

**成本动因**: $CD_i = \frac{\text{作业成本}}{\text{成本动因量}}$

### 目标成本法 / Target Costing

**目标成本**: $TC = \text{市场价格} - \text{目标利润}$

**成本差距**: $\Delta C = \text{当前成本} - TC$

**成本降低**: $\Delta C = \sum_{i=1}^n \Delta c_i$

---

## 8.8.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ProductionPlan {
    pub products: Vec<Product>,
    pub periods: usize,
    pub capacity: Vec<f64>,
    pub demand: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Product {
    pub id: String,
    pub production_cost: f64,
    pub holding_cost: f64,
    pub setup_cost: f64,
    pub capacity_requirement: f64,
}

impl ProductionPlan {
    pub fn new(periods: usize) -> Self {
        Self {
            products: Vec::new(),
            periods,
            capacity: vec![1000.0; periods],
            demand: Vec::new(),
        }
    }
    
    pub fn add_product(&mut self, product: Product) {
        self.products.push(product);
        self.demand.push(vec![0.0; self.periods]);
    }
    
    pub fn set_demand(&mut self, product_id: &str, period: usize, demand: f64) {
        if let Some(product) = self.products.iter().position(|p| p.id == product_id) {
            if period < self.periods {
                self.demand[product][period] = demand;
            }
        }
    }
    
    pub fn optimize_production(&self) -> Vec<Vec<f64>> {
        let mut production = vec![vec![0.0; self.periods]; self.products.len()];
        let mut inventory = vec![vec![0.0; self.periods + 1]; self.products.len()];
        
        for p in 0..self.products.len() {
            for t in 0..self.periods {
                // 简化的生产计划算法
                let needed = self.demand[p][t] - inventory[p][t];
                if needed > 0.0 {
                    let max_production = self.capacity[t] / self.products[p].capacity_requirement;
                    production[p][t] = needed.min(max_production);
                    inventory[p][t + 1] = inventory[p][t] + production[p][t] - self.demand[p][t];
                }
            }
        }
        
        production
    }
    
    pub fn calculate_total_cost(&self, production: &Vec<Vec<f64>>) -> f64 {
        let mut total_cost = 0.0;
        
        for p in 0..self.products.len() {
            for t in 0..self.periods {
                if production[p][t] > 0.0 {
                    total_cost += production[p][t] * self.products[p].production_cost;
                    total_cost += self.products[p].setup_cost;
                }
            }
        }
        
        total_cost
    }
}

#[derive(Debug)]
pub struct QualityControl {
    pub process_mean: f64,
    pub process_std: f64,
    pub usl: f64,
    pub lsl: f64,
}

impl QualityControl {
    pub fn new(mean: f64, std: f64, usl: f64, lsl: f64) -> Self {
        Self {
            process_mean: mean,
            process_std: std,
            usl,
            lsl,
        }
    }
    
    pub fn calculate_cp(&self) -> f64 {
        (self.usl - self.lsl) / (6.0 * self.process_std)
    }
    
    pub fn calculate_cpk(&self) -> f64 {
        let cpu = (self.usl - self.process_mean) / (3.0 * self.process_std);
        let cpl = (self.process_mean - self.lsl) / (3.0 * self.process_std);
        cpu.min(cpl)
    }
    
    pub fn calculate_defect_rate(&self) -> f64 {
        // 简化的缺陷率计算
        let z1 = (self.usl - self.process_mean) / self.process_std;
        let z2 = (self.lsl - self.process_mean) / self.process_std;
        
        // 使用标准正态分布计算缺陷率
        let p1 = 1.0 - normal_cdf(z1);
        let p2 = normal_cdf(z2);
        
        p1 + p2
    }
}

#[derive(Debug)]
pub struct InventorySystem {
    pub demand_rate: f64,
    pub setup_cost: f64,
    pub holding_cost: f64,
    pub lead_time: f64,
    pub safety_factor: f64,
}

impl InventorySystem {
    pub fn new(demand: f64, setup: f64, holding: f64, lead_time: f64) -> Self {
        Self {
            demand_rate: demand,
            setup_cost: setup,
            holding_cost: holding,
            lead_time,
            safety_factor: 1.96, // 95% 服务水平
        }
    }
    
    pub fn calculate_eoq(&self) -> f64 {
        (2.0 * self.demand_rate * self.setup_cost / self.holding_cost).sqrt()
    }
    
    pub fn calculate_safety_stock(&self, demand_std: f64) -> f64 {
        self.safety_factor * demand_std * self.lead_time.sqrt()
    }
    
    pub fn calculate_reorder_point(&self, demand_std: f64) -> f64 {
        self.demand_rate * self.lead_time + self.calculate_safety_stock(demand_std)
    }
    
    pub fn calculate_total_cost(&self, order_quantity: f64) -> f64 {
        let annual_orders = self.demand_rate / order_quantity;
        let average_inventory = order_quantity / 2.0;
        
        annual_orders * self.setup_cost + average_inventory * self.holding_cost
    }
}

// 简化的正态分布累积分布函数
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // 简化的误差函数近似
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

// 使用示例
fn main() {
    // 生产计划示例
    let mut plan = ProductionPlan::new(4);
    
    plan.add_product(Product {
        id: "A".to_string(),
        production_cost: 10.0,
        holding_cost: 2.0,
        setup_cost: 100.0,
        capacity_requirement: 1.0,
    });
    
    plan.set_demand("A", 0, 100.0);
    plan.set_demand("A", 1, 150.0);
    plan.set_demand("A", 2, 200.0);
    plan.set_demand("A", 3, 120.0);
    
    let production = plan.optimize_production();
    let total_cost = plan.calculate_total_cost(&production);
    
    println!("Production plan: {:?}", production);
    println!("Total cost: {:.2}", total_cost);
    
    // 质量控制示例
    let qc = QualityControl::new(100.0, 2.0, 106.0, 94.0);
    println!("Cp: {:.3}", qc.calculate_cp());
    println!("Cpk: {:.3}", qc.calculate_cpk());
    println!("Defect rate: {:.3}%", qc.calculate_defect_rate() * 100.0);
    
    // 库存管理示例
    let inventory = InventorySystem::new(1000.0, 50.0, 5.0, 2.0);
    let eoq = inventory.calculate_eoq();
    let total_cost = inventory.calculate_total_cost(eoq);
    
    println!("EOQ: {:.0}", eoq);
    println!("Total inventory cost: {:.2}", total_cost);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module ManufacturingModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, length)

-- 生产计划数据类型
data ProductionPlan = ProductionPlan {
    products :: [Product],
    periods :: Int,
    capacity :: [Double],
    demand :: [[Double]]
} deriving Show

data Product = Product {
    productId :: String,
    productionCost :: Double,
    holdingCost :: Double,
    setupCost :: Double,
    capacityRequirement :: Double
} deriving Show

-- 创建生产计划
newProductionPlan :: Int -> ProductionPlan
newProductionPlan periods = ProductionPlan {
    products = [],
    periods = periods,
    capacity = replicate periods 1000.0,
    demand = []
}

-- 添加产品
addProduct :: Product -> ProductionPlan -> ProductionPlan
addProduct product plan = plan { 
    products = product : products plan,
    demand = replicate (periods plan) 0.0 : demand plan
}

-- 设置需求
setDemand :: String -> Int -> Double -> ProductionPlan -> ProductionPlan
setDemand productId period demandValue plan = plan { demand = updatedDemand }
  where
    productIndex = findProductIndex productId (products plan)
    updatedDemand = case productIndex of
        Just idx -> updateList (demand plan) idx period demandValue
        Nothing -> demand plan

-- 查找产品索引
findProductIndex :: String -> [Product] -> Maybe Int
findProductIndex id products = go 0 products
  where
    go _ [] = Nothing
    go n (p:ps) = if productId p == id then Just n else go (n+1) ps

-- 更新列表中的特定元素
updateList :: [[Double]] -> Int -> Int -> Double -> [[Double]]
updateList lists row col value = take row lists ++ [updateRow (lists !! row) col value] ++ drop (row+1) lists
  where
    updateRow row col value = take col row ++ [value] ++ drop (col+1) row

-- 优化生产计划
optimizeProduction :: ProductionPlan -> [[Double]]
optimizeProduction plan = go initialProduction initialInventory
  where
    initialProduction = replicate (length (products plan)) (replicate (periods plan) 0.0)
    initialInventory = replicate (length (products plan)) (replicate ((periods plan) + 1) 0.0)
    
    go prod inv = foldl (\acc p -> foldl (\acc2 t -> updateProduction acc2 p t inv) acc [0..(periods plan)-1]) prod [0..(length (products plan))-1]
    
    updateProduction prod p t inv = 
        let needed = (demand plan) !! p !! t - (inv !! p !! t)
            maxProduction = (capacity plan) !! t / capacityRequirement ((products plan) !! p)
        in if needed > 0 
           then updateMatrix prod p t (min needed maxProduction)
           else prod

-- 更新矩阵
updateMatrix :: [[Double]] -> Int -> Int -> Double -> [[Double]]
updateMatrix matrix row col value = take row matrix ++ [updateRow (matrix !! row) col value] ++ drop (row+1) matrix
  where
    updateRow row col value = take col row ++ [value] ++ drop (col+1) row

-- 计算总成本
calculateTotalCost :: ProductionPlan -> [[Double]] -> Double
calculateTotalCost plan production = sum [cost p t | p <- [0..length (products plan)-1], t <- [0..periods plan-1]]
  where
    cost p t = if (production !! p !! t) > 0 
               then (production !! p !! t) * productionCost ((products plan) !! p) + setupCost ((products plan) !! p)
               else 0

-- 质量控制
data QualityControl = QualityControl {
    processMean :: Double,
    processStd :: Double,
    usl :: Double,
    lsl :: Double
} deriving Show

newQualityControl :: Double -> Double -> Double -> Double -> QualityControl
newQualityControl mean std usl lsl = QualityControl mean std usl lsl

calculateCp :: QualityControl -> Double
calculateCp qc = (usl qc - lsl qc) / (6.0 * processStd qc)

calculateCpk :: QualityControl -> Double
calculateCpk qc = min cpu cpl
  where
    cpu = (usl qc - processMean qc) / (3.0 * processStd qc)
    cpl = (processMean qc - lsl qc) / (3.0 * processStd qc)

calculateDefectRate :: QualityControl -> Double
calculateDefectRate qc = p1 + p2
  where
    z1 = (usl qc - processMean qc) / processStd qc
    z2 = (lsl qc - processMean qc) / processStd qc
    p1 = 1.0 - normalCdf z1
    p2 = normalCdf z2

-- 简化的正态分布累积分布函数
normalCdf :: Double -> Double
normalCdf x = 0.5 * (1.0 + erf (x / sqrt 2.0))

erf :: Double -> Double
erf x = sign * y
  where
    sign = if x < 0 then -1.0 else 1.0
    x_abs = abs x
    t = 1.0 / (1.0 + 0.3275911 * x_abs)
    y = 1.0 - (((((1.061405429 * t + (-1.453152027)) * t + 1.421413741) * t + (-0.284496736)) * t + 0.254829592) * t) * exp (-x_abs * x_abs)

-- 库存系统
data InventorySystem = InventorySystem {
    demandRate :: Double,
    setupCost :: Double,
    holdingCost :: Double,
    leadTime :: Double,
    safetyFactor :: Double
} deriving Show

newInventorySystem :: Double -> Double -> Double -> Double -> InventorySystem
newInventorySystem demand setup holding leadTime = InventorySystem {
    demandRate = demand,
    setupCost = setup,
    holdingCost = holding,
    leadTime = leadTime,
    safetyFactor = 1.96  -- 95% 服务水平
}

calculateEOQ :: InventorySystem -> Double
calculateEOQ inv = sqrt (2.0 * demandRate inv * setupCost inv / holdingCost inv)

calculateSafetyStock :: InventorySystem -> Double -> Double
calculateSafetyStock inv demandStd = safetyFactor inv * demandStd * sqrt (leadTime inv)

calculateReorderPoint :: InventorySystem -> Double -> Double
calculateReorderPoint inv demandStd = demandRate inv * leadTime inv + calculateSafetyStock inv demandStd

calculateTotalCost :: InventorySystem -> Double -> Double
calculateTotalCost inv orderQuantity = annualOrders * setupCost inv + averageInventory * holdingCost inv
  where
    annualOrders = demandRate inv / orderQuantity
    averageInventory = orderQuantity / 2.0

-- 示例使用
example :: IO ()
example = do
    -- 生产计划示例
    let plan = setDemand "A" 0 100.0 $
               setDemand "A" 1 150.0 $
               setDemand "A" 2 200.0 $
               setDemand "A" 3 120.0 $
               addProduct (Product "A" 10.0 2.0 100.0 1.0) (newProductionPlan 4)
        
        production = optimizeProduction plan
        totalCost = calculateTotalCost plan production
    
    putStrLn $ "Production plan: " ++ show production
    putStrLn $ "Total cost: " ++ show totalCost
    
    -- 质量控制示例
    let qc = newQualityControl 100.0 2.0 106.0 94.0
    
    putStrLn $ "Cp: " ++ show (calculateCp qc)
    putStrLn $ "Cpk: " ++ show (calculateCpk qc)
    putStrLn $ "Defect rate: " ++ show (calculateDefectRate qc * 100.0) ++ "%"
    
    -- 库存管理示例
    let inventory = newInventorySystem 1000.0 50.0 5.0 2.0
        eoq = calculateEOQ inventory
        totalCost = calculateTotalCost inventory eoq
    
    putStrLn $ "EOQ: " ++ show eoq
    putStrLn $ "Total inventory cost: " ++ show totalCost
```

### 应用领域 / Application Domains

#### 生产管理 / Production Management

- **生产计划**: 主生产计划、物料需求计划
- **产能规划**: 产能平衡、瓶颈分析
- **调度优化**: 作业调度、资源分配

#### 质量管理 / Quality Management

- **统计过程控制**: 控制图、过程能力分析
- **六西格玛**: 过程改进、缺陷减少
- **质量成本**: 预防成本、鉴定成本、故障成本

#### 供应链管理 / Supply Chain Management

- **库存管理**: EOQ模型、安全库存
- **供应商管理**: 供应商选择、绩效评估
- **物流优化**: 运输优化、仓储管理

---

## 参考文献 / References

1. Nahmias, S. (2009). Production and Operations Analysis. McGraw-Hill.
2. Montgomery, D. C. (2012). Introduction to Statistical Quality Control. Wiley.
3. Chopra, S., & Meindl, P. (2015). Supply Chain Management. Pearson.
4. Womack, J. P., & Jones, D. T. (2003). Lean Thinking. Free Press.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
