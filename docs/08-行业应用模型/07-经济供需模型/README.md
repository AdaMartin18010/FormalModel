# 8.7 经济供需模型 / Economic Supply-Demand Models

## 目录 / Table of Contents

- [8.7 经济供需模型 / Economic Supply-Demand Models](#87-经济供需模型--economic-supply-demand-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.7.1 供需基础模型 / Basic Supply-Demand Models](#871-供需基础模型--basic-supply-demand-models)
    - [需求函数 / Demand Function](#需求函数--demand-function)
    - [供给函数 / Supply Function](#供给函数--supply-function)
    - [需求定律 / Law of Demand](#需求定律--law-of-demand)
    - [供给定律 / Law of Supply](#供给定律--law-of-supply)
  - [8.7.2 市场均衡模型 / Market Equilibrium Models](#872-市场均衡模型--market-equilibrium-models)
    - [均衡条件 / Equilibrium Conditions](#均衡条件--equilibrium-conditions)
    - [线性供需均衡 / Linear Supply-Demand Equilibrium](#线性供需均衡--linear-supply-demand-equilibrium)
    - [稳定性分析 / Stability Analysis](#稳定性分析--stability-analysis)
  - [8.7.3 价格机制模型 / Price Mechanism Models](#873-价格机制模型--price-mechanism-models)
    - [价格弹性 / Price Elasticity](#价格弹性--price-elasticity)
    - [收入弹性 / Income Elasticity](#收入弹性--income-elasticity)
    - [交叉价格弹性 / Cross-Price Elasticity](#交叉价格弹性--cross-price-elasticity)
    - [价格管制 / Price Controls](#价格管制--price-controls)
  - [8.7.4 弹性理论模型 / Elasticity Theory Models](#874-弹性理论模型--elasticity-theory-models)
    - [点弹性 / Point Elasticity](#点弹性--point-elasticity)
    - [弧弹性 / Arc Elasticity](#弧弹性--arc-elasticity)
    - [弹性分类 / Elasticity Classification](#弹性分类--elasticity-classification)
    - [弹性与总收入 / Elasticity and Total Revenue](#弹性与总收入--elasticity-and-total-revenue)
  - [8.7.5 一般均衡模型 / General Equilibrium Models](#875-一般均衡模型--general-equilibrium-models)
    - [瓦尔拉斯均衡 / Walrasian Equilibrium](#瓦尔拉斯均衡--walrasian-equilibrium)
    - [帕累托最优 / Pareto Optimality](#帕累托最优--pareto-optimality)
    - [阿罗-德布鲁模型 / Arrow-Debreu Model](#阿罗-德布鲁模型--arrow-debreu-model)
  - [8.7.6 博弈论模型 / Game Theory Models](#876-博弈论模型--game-theory-models)
    - [纳什均衡 / Nash Equilibrium](#纳什均衡--nash-equilibrium)
    - [古诺模型 / Cournot Model](#古诺模型--cournot-model)
    - [伯特兰模型 / Bertrand Model](#伯特兰模型--bertrand-model)
  - [8.7.7 宏观经济模型 / Macroeconomic Models](#877-宏观经济模型--macroeconomic-models)
    - [IS-LM模型 / IS-LM Model](#is-lm模型--is-lm-model)
    - [AD-AS模型 / AD-AS Model](#ad-as模型--ad-as-model)
    - [菲利普斯曲线 / Phillips Curve](#菲利普斯曲线--phillips-curve)
  - [8.7.8 实现与应用 / Implementation and Applications](#878-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [微观经济学 / Microeconomics](#微观经济学--microeconomics)
      - [宏观经济学 / Macroeconomics](#宏观经济学--macroeconomics)
      - [国际贸易 / International Trade](#国际贸易--international-trade)
  - [参考文献 / References](#参考文献--references)

---

## 8.7.1 供需基础模型 / Basic Supply-Demand Models

### 需求函数 / Demand Function

**需求函数**: $Q_d = D(P, Y, P_s, P_c, T, N)$

其中：

- $Q_d$: 需求量
- $P$: 商品价格
- $Y$: 消费者收入
- $P_s$: 替代品价格
- $P_c$: 互补品价格
- $T$: 消费者偏好
- $N$: 消费者数量

**线性需求函数**: $Q_d = a - bP$

其中 $a > 0, b > 0$ 是参数。

### 供给函数 / Supply Function

**供给函数**: $Q_s = S(P, C, T, N, E)$

其中：

- $Q_s$: 供给量
- $P$: 商品价格
- $C$: 生产成本
- $T$: 技术水平
- $N$: 生产者数量
- $E$: 预期价格

**线性供给函数**: $Q_s = c + dP$

其中 $c, d > 0$ 是参数。

### 需求定律 / Law of Demand

**需求定律**: 在其他条件不变的情况下，价格上升，需求量下降。

$$\frac{\partial Q_d}{\partial P} < 0$$

### 供给定律 / Law of Supply

**供给定律**: 在其他条件不变的情况下，价格上升，供给量增加。

$$\frac{\partial Q_s}{\partial P} > 0$$

---

## 8.7.2 市场均衡模型 / Market Equilibrium Models

### 均衡条件 / Equilibrium Conditions

**市场均衡**: $Q_d = Q_s$

**均衡价格**: $P^*$ 满足 $D(P^*) = S(P^*)$

**均衡数量**: $Q^* = D(P^*) = S(P^*)$

### 线性供需均衡 / Linear Supply-Demand Equilibrium

**需求函数**: $Q_d = a - bP$

**供给函数**: $Q_s = c + dP$

**均衡条件**: $a - bP^* = c + dP^*$

**均衡价格**: $P^* = \frac{a - c}{b + d}$

**均衡数量**: $Q^* = \frac{ad + bc}{b + d}$

### 稳定性分析 / Stability Analysis

**瓦尔拉斯稳定性**: 当 $Q_d > Q_s$ 时，价格上升；当 $Q_d < Q_s$ 时，价格下降。

**马歇尔稳定性**: 当价格高于均衡价格时，供给过剩，价格下降；当价格低于均衡价格时，需求过剩，价格上升。

---

## 8.7.3 价格机制模型 / Price Mechanism Models

### 价格弹性 / Price Elasticity

**需求价格弹性**: $\epsilon_d = \frac{\partial Q_d}{\partial P} \cdot \frac{P}{Q_d}$

**供给价格弹性**: $\epsilon_s = \frac{\partial Q_s}{\partial P} \cdot \frac{P}{Q_s}$

### 收入弹性 / Income Elasticity

**需求收入弹性**: $\eta_d = \frac{\partial Q_d}{\partial Y} \cdot \frac{Y}{Q_d}$

### 交叉价格弹性 / Cross-Price Elasticity

**需求交叉价格弹性**: $\epsilon_{xy} = \frac{\partial Q_x}{\partial P_y} \cdot \frac{P_y}{Q_x}$

### 价格管制 / Price Controls

**价格上限**: $P_{max} < P^*$ 导致短缺

**价格下限**: $P_{min} > P^*$ 导致过剩

**短缺量**: $Q_d(P_{max}) - Q_s(P_{max})$

**过剩量**: $Q_s(P_{min}) - Q_d(P_{min})$

---

## 8.7.4 弹性理论模型 / Elasticity Theory Models

### 点弹性 / Point Elasticity

**需求点弹性**: $\epsilon_d = \frac{dQ_d}{dP} \cdot \frac{P}{Q_d}$

**供给点弹性**: $\epsilon_s = \frac{dQ_s}{dP} \cdot \frac{P}{Q_s}$

### 弧弹性 / Arc Elasticity

**需求弧弹性**: $\epsilon_d = \frac{Q_2 - Q_1}{P_2 - P_1} \cdot \frac{P_1 + P_2}{Q_1 + Q_2}$

### 弹性分类 / Elasticity Classification

**完全弹性**: $|\epsilon| = \infty$

**富有弹性**: $|\epsilon| > 1$

**单位弹性**: $|\epsilon| = 1$

**缺乏弹性**: $0 < |\epsilon| < 1$

**完全无弹性**: $|\epsilon| = 0$

### 弹性与总收入 / Elasticity and Total Revenue

**总收入**: $TR = P \cdot Q$

**总收入变化**: $\frac{dTR}{dP} = Q(1 + \epsilon_d)$

- 当 $|\epsilon_d| > 1$ 时，价格上升，总收入下降
- 当 $|\epsilon_d| < 1$ 时，价格上升，总收入上升
- 当 $|\epsilon_d| = 1$ 时，总收入不变

---

## 8.7.5 一般均衡模型 / General Equilibrium Models

### 瓦尔拉斯均衡 / Walrasian Equilibrium

**经济**: $E = (X_i, \preceq_i, \omega_i)_{i=1}^n$

**可行分配**: $\sum_{i=1}^n x_i \leq \sum_{i=1}^n \omega_i$

**瓦尔拉斯均衡**: $(x^*, p^*)$ 满足：

1. 每个消费者在预算约束下最大化效用
2. 市场出清: $\sum_{i=1}^n x_i^* = \sum_{i=1}^n \omega_i$

### 帕累托最优 / Pareto Optimality

**帕累托最优**: 不存在其他可行分配使得至少一个人的效用提高，且没有人效用降低。

**福利经济学第一定理**: 瓦尔拉斯均衡是帕累托最优的。

**福利经济学第二定理**: 在适当条件下，任何帕累托最优分配都可以通过瓦尔拉斯均衡实现。

### 阿罗-德布鲁模型 / Arrow-Debreu Model

**状态空间**: $\Omega = \{\omega_1, \ldots, \omega_S\}$

**或有商品**: $x_{is}$ 表示消费者 $i$ 在状态 $s$ 的商品消费

**预算约束**: $\sum_{s=1}^S p_s \cdot x_{is} \leq \sum_{s=1}^S p_s \cdot \omega_{is}$

---

## 8.7.6 博弈论模型 / Game Theory Models

### 纳什均衡 / Nash Equilibrium

**策略组合**: $\sigma = (\sigma_1, \ldots, \sigma_n)$

**纳什均衡**: 对于每个玩家 $i$，$\sigma_i$ 是对其他玩家策略的最优反应。

**数学表示**: $u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*)$ 对所有 $\sigma_i$ 成立。

### 古诺模型 / Cournot Model

**双寡头竞争**: 两个企业选择产量 $q_1, q_2$

**市场需求**: $P = a - b(q_1 + q_2)$

**企业利润**: $\pi_i = (a - b(q_1 + q_2))q_i - c_i q_i$

**反应函数**: $q_i = \frac{a - c_i - bq_j}{2b}$

**纳什均衡**: $q_1^* = q_2^* = \frac{a - c}{3b}$

### 伯特兰模型 / Bertrand Model

**价格竞争**: 两个企业选择价格 $p_1, p_2$

**需求分配**: $D_i = \begin{cases} D(p_i) & \text{if } p_i < p_j \\ \frac{1}{2}D(p_i) & \text{if } p_i = p_j \\ 0 & \text{if } p_i > p_j \end{cases}$

**伯特兰悖论**: 在完全同质产品竞争中，均衡价格等于边际成本。

---

## 8.7.7 宏观经济模型 / Macroeconomic Models

### IS-LM模型 / IS-LM Model

**IS曲线**: $Y = C(Y - T) + I(r) + G$

**LM曲线**: $\frac{M}{P} = L(Y, r)$

**均衡**: IS曲线与LM曲线的交点

### AD-AS模型 / AD-AS Model

**总需求曲线**: $Y = C(Y - T) + I(r) + G + NX$

**总供给曲线**: $Y = F(K, L)$

**短期均衡**: AD曲线与SRAS曲线的交点

**长期均衡**: AD曲线与LRAS曲线的交点

### 菲利普斯曲线 / Phillips Curve

**短期菲利普斯曲线**: $\pi = \pi^e + \alpha(u^* - u) + \epsilon$

**长期菲利普斯曲线**: 垂直，自然失业率水平

---

## 8.7.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Market {
    pub demand_function: Box<dyn Fn(f64) -> f64>,
    pub supply_function: Box<dyn Fn(f64) -> f64>,
    pub equilibrium_price: Option<f64>,
    pub equilibrium_quantity: Option<f64>,
}

impl Market {
    pub fn new<D, S>(demand: D, supply: S) -> Self 
    where
        D: Fn(f64) -> f64 + 'static,
        S: Fn(f64) -> f64 + 'static,
    {
        Self {
            demand_function: Box::new(demand),
            supply_function: Box::new(supply),
            equilibrium_price: None,
            equilibrium_quantity: None,
        }
    }
    
    pub fn find_equilibrium(&mut self, tolerance: f64, max_iterations: usize) -> Option<(f64, f64)> {
        let mut price = 1.0;
        
        for _ in 0..max_iterations {
            let demand = (self.demand_function)(price);
            let supply = (self.supply_function)(price);
            
            if (demand - supply).abs() < tolerance {
                self.equilibrium_price = Some(price);
                self.equilibrium_quantity = Some(demand);
                return Some((price, demand));
            }
            
            // 简单的价格调整机制
            if demand > supply {
                price *= 1.01; // 价格上涨
            } else {
                price *= 0.99; // 价格下跌
            }
        }
        
        None
    }
    
    pub fn calculate_elasticity(&self, price: f64, delta: f64) -> (f64, f64) {
        let demand_at_price = (self.demand_function)(price);
        let demand_at_price_plus_delta = (self.demand_function)(price + delta);
        let supply_at_price = (self.supply_function)(price);
        let supply_at_price_plus_delta = (self.supply_function)(price + delta);
        
        let demand_elasticity = (demand_at_price_plus_delta - demand_at_price) / delta * price / demand_at_price;
        let supply_elasticity = (supply_at_price_plus_delta - supply_at_price) / delta * price / supply_at_price;
        
        (demand_elasticity, supply_elasticity)
    }
    
    pub fn calculate_consumer_surplus(&self, price: f64) -> f64 {
        let mut surplus = 0.0;
        let mut current_price = 0.0;
        let step = 0.01;
        
        while current_price < price {
            let demand = (self.demand_function)(current_price);
            surplus += demand * step;
            current_price += step;
        }
        
        surplus - price * (self.demand_function)(price)
    }
    
    pub fn calculate_producer_surplus(&self, price: f64) -> f64 {
        let mut surplus = 0.0;
        let mut current_price = 0.0;
        let step = 0.01;
        
        while current_price < price {
            let supply = (self.supply_function)(current_price);
            surplus += supply * step;
            current_price += step;
        }
        
        price * (self.supply_function)(price) - surplus
    }
}

#[derive(Debug)]
pub struct CournotModel {
    pub firms: Vec<f64>, // 边际成本
    pub market_demand: Box<dyn Fn(f64) -> f64>,
    pub equilibrium_quantities: Vec<f64>,
}

impl CournotModel {
    pub fn new(firms: Vec<f64>, market_demand: Box<dyn Fn(f64) -> f64>) -> Self {
        Self {
            firms,
            market_demand,
            equilibrium_quantities: Vec::new(),
        }
    }
    
    pub fn find_equilibrium(&mut self, tolerance: f64, max_iterations: usize) -> Vec<f64> {
        let n = self.firms.len();
        let mut quantities = vec![1.0; n];
        
        for _ in 0..max_iterations {
            let mut new_quantities = quantities.clone();
            let mut converged = true;
            
            for i in 0..n {
                let total_quantity: f64 = quantities.iter().sum();
                let market_price = (self.market_demand)(total_quantity);
                
                // 计算企业i的最优产量
                let marginal_revenue = market_price - quantities[i] * self.derivative_at(total_quantity);
                let optimal_quantity = (marginal_revenue - self.firms[i]) / self.derivative_at(total_quantity);
                
                new_quantities[i] = optimal_quantity.max(0.0);
                
                if (new_quantities[i] - quantities[i]).abs() > tolerance {
                    converged = false;
                }
            }
            
            quantities = new_quantities;
            
            if converged {
                break;
            }
        }
        
        self.equilibrium_quantities = quantities.clone();
        quantities
    }
    
    fn derivative_at(&self, quantity: f64) -> f64 {
        let delta = 0.01;
        let price1 = (self.market_demand)(quantity);
        let price2 = (self.market_demand)(quantity + delta);
        (price2 - price1) / delta
    }
}

// 使用示例
fn main() {
    // 线性供需模型
    let demand = |p: f64| (100.0 - p).max(0.0);
    let supply = |p: f64| (p - 20.0).max(0.0);
    
    let mut market = Market::new(demand, supply);
    
    if let Some((price, quantity)) = market.find_equilibrium(0.01, 1000) {
        println!("Equilibrium price: {:.2}", price);
        println!("Equilibrium quantity: {:.2}", quantity);
        
        let (demand_elasticity, supply_elasticity) = market.calculate_elasticity(price, 0.01);
        println!("Demand elasticity: {:.2}", demand_elasticity);
        println!("Supply elasticity: {:.2}", supply_elasticity);
        
        let consumer_surplus = market.calculate_consumer_surplus(price);
        let producer_surplus = market.calculate_producer_surplus(price);
        println!("Consumer surplus: {:.2}", consumer_surplus);
        println!("Producer surplus: {:.2}", producer_surplus);
    }
    
    // 古诺模型示例
    let market_demand = |q: f64| (100.0 - q).max(0.0);
    let mut cournot = CournotModel::new(vec![10.0, 15.0], Box::new(market_demand));
    
    let equilibrium_quantities = cournot.find_equilibrium(0.01, 1000);
    println!("Cournot equilibrium quantities: {:?}", equilibrium_quantities);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module EconomicModels where

import Data.List (sum, length)
import Data.Vector (Vector)
import qualified Data.Vector as V

-- 市场数据类型
data Market = Market {
    demandFunction :: Double -> Double,
    supplyFunction :: Double -> Double,
    equilibriumPrice :: Maybe Double,
    equilibriumQuantity :: Maybe Double
} deriving Show

-- 创建市场
newMarket :: (Double -> Double) -> (Double -> Double) -> Market
newMarket demand supply = Market {
    demandFunction = demand,
    supplyFunction = supply,
    equilibriumPrice = Nothing,
    equilibriumQuantity = Nothing
}

-- 寻找均衡
findEquilibrium :: Market -> Double -> Int -> Maybe (Double, Double)
findEquilibrium market tolerance maxIterations = go 1.0 0
  where
    go price iterations
        | iterations >= maxIterations = Nothing
        | abs (demand - supply) < tolerance = Just (price, demand)
        | demand > supply = go (price * 1.01) (iterations + 1)
        | otherwise = go (price * 0.99) (iterations + 1)
      where
        demand = demandFunction market price
        supply = supplyFunction market price

-- 计算弹性
calculateElasticity :: Market -> Double -> Double -> (Double, Double)
calculateElasticity market price delta = (demandElasticity, supplyElasticity)
  where
    demandAtPrice = demandFunction market price
    demandAtPricePlusDelta = demandFunction market (price + delta)
    supplyAtPrice = supplyFunction market price
    supplyAtPricePlusDelta = supplyFunction market (price + delta)
    
    demandElasticity = (demandAtPricePlusDelta - demandAtPrice) / delta * price / demandAtPrice
    supplyElasticity = (supplyAtPricePlusDelta - supplyAtPrice) / delta * price / supplyAtPrice

-- 计算消费者剩余
calculateConsumerSurplus :: Market -> Double -> Double
calculateConsumerSurplus market price = surplus - price * demandAtPrice
  where
    demandAtPrice = demandFunction market price
    surplus = sum [demandFunction market p * 0.01 | p <- [0, 0.01..price]]

-- 古诺模型
data CournotModel = CournotModel {
    firms :: Vector Double,  -- 边际成本
    marketDemand :: Double -> Double,
    equilibriumQuantities :: Vector Double
} deriving Show

-- 创建古诺模型
newCournotModel :: [Double] -> (Double -> Double) -> CournotModel
newCournotModel costs demand = CournotModel {
    firms = V.fromList costs,
    marketDemand = demand,
    equilibriumQuantities = V.empty
}

-- 寻找古诺均衡
findCournotEquilibrium :: CournotModel -> Double -> Int -> Vector Double
findCournotEquilibrium model tolerance maxIterations = go (V.replicate (V.length (firms model)) 1.0) 0
  where
    go quantities iterations
        | iterations >= maxIterations = quantities
        | converged = quantities
        | otherwise = go newQuantities (iterations + 1)
      where
        totalQuantity = V.sum quantities
        marketPrice = marketDemand model totalQuantity
        
        newQuantities = V.imap (\i q -> 
            let marginalRevenue = marketPrice - q * derivativeAt totalQuantity
                optimalQuantity = (marginalRevenue - firms model V.! i) / derivativeAt totalQuantity
            in max 0.0 optimalQuantity) quantities
        
        converged = V.all (\i -> abs (newQuantities V.! i - quantities V.! i) < tolerance) 
                          (V.enumFromN 0 (V.length quantities))
    
    derivativeAt quantity = (marketDemand model (quantity + 0.01) - marketDemand model quantity) / 0.01

-- 示例使用
example :: IO ()
example = do
    -- 线性供需模型
    let demand p = max 0.0 (100.0 - p)
        supply p = max 0.0 (p - 20.0)
        market = newMarket demand supply
    
    case findEquilibrium market 0.01 1000 of
        Just (price, quantity) -> do
            putStrLn $ "Equilibrium price: " ++ show price
            putStrLn $ "Equilibrium quantity: " ++ show quantity
            
            let (demandElasticity, supplyElasticity) = calculateElasticity market price 0.01
            putStrLn $ "Demand elasticity: " ++ show demandElasticity
            putStrLn $ "Supply elasticity: " ++ show supplyElasticity
            
            let consumerSurplus = calculateConsumerSurplus market price
            putStrLn $ "Consumer surplus: " ++ show consumerSurplus
        
        Nothing -> putStrLn "No equilibrium found"
    
    -- 古诺模型示例
    let marketDemand q = max 0.0 (100.0 - q)
        cournot = newCournotModel [10.0, 15.0] marketDemand
        equilibrium = findCournotEquilibrium cournot 0.01 1000
    
    putStrLn $ "Cournot equilibrium quantities: " ++ show (V.toList equilibrium)
```

### 应用领域 / Application Domains

#### 微观经济学 / Microeconomics

- **消费者理论**: 效用最大化、需求函数
- **生产者理论**: 成本最小化、供给函数
- **市场结构**: 完全竞争、垄断、寡头垄断

#### 宏观经济学 / Macroeconomics

- **国民收入**: GDP、消费、投资、政府支出
- **货币理论**: 货币供给、需求、利率
- **经济周期**: 扩张、收缩、失业、通货膨胀

#### 国际贸易 / International Trade

- **比较优势**: 李嘉图模型
- **要素禀赋**: 赫克歇尔-俄林模型
- **贸易政策**: 关税、配额、补贴

---

## 参考文献 / References

1. Varian, H. R. (2014). Intermediate Microeconomics: A Modern Approach. W.W. Norton.
2. Mankiw, N. G. (2014). Principles of Economics. Cengage Learning.
3. Mas-Colell, A., Whinston, M. D., & Green, J. R. (1995). Microeconomic Theory. Oxford University Press.
4. Blanchard, O. (2017). Macroeconomics. Pearson.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
