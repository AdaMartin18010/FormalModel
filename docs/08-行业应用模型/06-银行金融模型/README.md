# 8.6 银行金融模型 / Banking & Finance Models

## 目录 / Table of Contents

- [8.6 银行金融模型 / Banking \& Finance Models](#86-银行金融模型--banking--finance-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.6.1 风险管理模型 / Risk Management Models](#861-风险管理模型--risk-management-models)
    - [VaR (Value at Risk) 模型 / VaR Model](#var-value-at-risk-模型--var-model)
    - [历史模拟法 / Historical Simulation](#历史模拟法--historical-simulation)
    - [蒙特卡洛模拟 / Monte Carlo Simulation](#蒙特卡洛模拟--monte-carlo-simulation)
    - [压力测试 / Stress Testing](#压力测试--stress-testing)
  - [8.6.2 投资组合模型 / Portfolio Models](#862-投资组合模型--portfolio-models)
    - [马科维茨均值-方差模型 / Markowitz Mean-Variance Model](#马科维茨均值-方差模型--markowitz-mean-variance-model)
    - [投资组合收益率 / Portfolio Return](#投资组合收益率--portfolio-return)
    - [有效前沿 / Efficient Frontier](#有效前沿--efficient-frontier)
    - [资本资产定价模型 (CAPM) / Capital Asset Pricing Model](#资本资产定价模型-capm--capital-asset-pricing-model)
  - [8.6.3 期权定价模型 / Option Pricing Models](#863-期权定价模型--option-pricing-models)
    - [Black-Scholes模型 / Black-Scholes Model](#black-scholes模型--black-scholes-model)
    - [二叉树模型 / Binomial Model](#二叉树模型--binomial-model)
    - [希腊字母 / Greeks](#希腊字母--greeks)
  - [8.6.4 信用风险模型 / Credit Risk Models](#864-信用风险模型--credit-risk-models)
    - [Merton模型 / Merton Model](#merton模型--merton-model)
    - [KMV模型 / KMV Model](#kmv模型--kmv-model)
    - [CreditMetrics模型 / CreditMetrics Model](#creditmetrics模型--creditmetrics-model)
  - [8.6.5 利率模型 / Interest Rate Models](#865-利率模型--interest-rate-models)
    - [Vasicek模型 / Vasicek Model](#vasicek模型--vasicek-model)
    - [Cox-Ingersoll-Ross (CIR) 模型 / CIR Model](#cox-ingersoll-ross-cir-模型--cir-model)
    - [Hull-White模型 / Hull-White Model](#hull-white模型--hull-white-model)
  - [8.6.6 银行运营模型 / Banking Operations Models](#866-银行运营模型--banking-operations-models)
    - [流动性管理 / Liquidity Management](#流动性管理--liquidity-management)
    - [资本充足率 / Capital Adequacy](#资本充足率--capital-adequacy)
    - [资产负债管理 / Asset-Liability Management](#资产负债管理--asset-liability-management)
  - [8.6.7 金融科技模型 / FinTech Models](#867-金融科技模型--fintech-models)
    - [机器学习信用评分 / Machine Learning Credit Scoring](#机器学习信用评分--machine-learning-credit-scoring)
    - [区块链模型 / Blockchain Models](#区块链模型--blockchain-models)
    - [高频交易模型 / High-Frequency Trading Models](#高频交易模型--high-frequency-trading-models)
  - [8.6.8 实现与应用 / Implementation and Applications](#868-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [银行风险管理 / Banking Risk Management](#银行风险管理--banking-risk-management)
      - [投资管理 / Investment Management](#投资管理--investment-management)
      - [金融工程 / Financial Engineering](#金融工程--financial-engineering)
  - [参考文献 / References](#参考文献--references)

---

## 8.6.1 风险管理模型 / Risk Management Models

### VaR (Value at Risk) 模型 / VaR Model

**定义**: 在给定置信水平下，投资组合在特定时间内的最大可能损失。

**数学表示**:
$$\text{VaR}_\alpha = \inf\{l \in \mathbb{R}: P(L > l) \leq 1-\alpha\}$$

其中：

- $L$: 损失随机变量
- $\alpha$: 置信水平 (通常为95%或99%)

### 历史模拟法 / Historical Simulation

**步骤**:

1. 收集历史收益率数据
2. 计算投资组合价值变化
3. 按升序排列损失
4. 取第 $(1-\alpha) \times 100\%$ 分位数

### 蒙特卡洛模拟 / Monte Carlo Simulation

**步骤**:

1. 假设收益率分布 (如正态分布)
2. 生成大量随机样本
3. 计算投资组合价值
4. 统计损失分布

### 压力测试 / Stress Testing

**情景分析**:

- 市场崩盘情景
- 利率急剧上升情景
- 汇率大幅波动情景

---

## 8.6.2 投资组合模型 / Portfolio Models

### 马科维茨均值-方差模型 / Markowitz Mean-Variance Model

**目标函数**: 最大化夏普比率
$$\max \frac{\mu_p - r_f}{\sigma_p}$$

**约束条件**:
$$\sum_{i=1}^n w_i = 1$$
$$w_i \geq 0, \quad \forall i$$

其中：

- $\mu_p$: 投资组合期望收益率
- $\sigma_p$: 投资组合标准差
- $r_f$: 无风险利率
- $w_i$: 资产 $i$ 的权重

### 投资组合收益率 / Portfolio Return

$$\mu_p = \sum_{i=1}^n w_i \mu_i$$

$$\sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}$$

其中 $\sigma_{ij}$ 是资产 $i$ 和 $j$ 的协方差。

### 有效前沿 / Efficient Frontier

**帕累托最优**: 在给定风险水平下最大化收益，或在给定收益水平下最小化风险。

### 资本资产定价模型 (CAPM) / Capital Asset Pricing Model

$$\mu_i = r_f + \beta_i(\mu_m - r_f)$$

其中：

- $\mu_i$: 资产 $i$ 的期望收益率
- $\mu_m$: 市场组合期望收益率
- $\beta_i$: 资产 $i$ 的贝塔系数

---

## 8.6.3 期权定价模型 / Option Pricing Models

### Black-Scholes模型 / Black-Scholes Model

**期权定价公式**:
$$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$

$$P = Ke^{-rT} N(-d_2) - S_0 N(-d_1)$$

其中：
$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

**参数**:

- $S_0$: 当前股价
- $K$: 执行价格
- $T$: 到期时间
- $r$: 无风险利率
- $\sigma$: 波动率
- $N(\cdot)$: 标准正态分布函数

### 二叉树模型 / Binomial Model

**单期模型**:
$$C = \frac{pC_u + (1-p)C_d}{1+r}$$

其中：
$$p = \frac{1+r-d}{u-d}$$

**多期扩展**: 通过递归计算得到期权价格。

### 希腊字母 / Greeks

**Delta**: $\Delta = \frac{\partial C}{\partial S} = N(d_1)$

**Gamma**: $\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S\sigma\sqrt{T}}$

**Theta**: $\Theta = \frac{\partial C}{\partial t} = -\frac{S\sigma N'(d_1)}{2\sqrt{T}} - rKe^{-rT}N(d_2)$

**Vega**: $\mathcal{V} = \frac{\partial C}{\partial \sigma} = S\sqrt{T}N'(d_1)$

---

## 8.6.4 信用风险模型 / Credit Risk Models

### Merton模型 / Merton Model

**公司价值**: $V_t = V_0 e^{(\mu-\sigma^2/2)t + \sigma W_t}$

**违约概率**: $P(\text{default}) = N\left(\frac{\ln(D/V_0) - (\mu-\sigma^2/2)T}{\sigma\sqrt{T}}\right)$

其中：

- $V_t$: 公司价值
- $D$: 债务面值
- $T$: 债务到期时间

### KMV模型 / KMV Model

**违约距离**: $DD = \frac{\ln(V_0/D) + (\mu-\sigma^2/2)T}{\sigma\sqrt{T}}$

**期望违约频率**: $EDF = N(-DD)$

### CreditMetrics模型 / CreditMetrics Model

**信用迁移矩阵**: $P_{ij}$ 表示从评级 $i$ 迁移到评级 $j$ 的概率。

**投资组合损失**: $L = \sum_{i=1}^n L_i \cdot \mathbf{1}_{\{\text{default}_i\}}$

---

## 8.6.5 利率模型 / Interest Rate Models

### Vasicek模型 / Vasicek Model

**短期利率过程**:
$$dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$$

**解**: $r_t = \theta + (r_0 - \theta)e^{-\kappa t} + \sigma \int_0^t e^{-\kappa(t-s)}dW_s$

### Cox-Ingersoll-Ross (CIR) 模型 / CIR Model

**短期利率过程**:
$$dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t$$

**特征**: 利率始终为正。

### Hull-White模型 / Hull-White Model

**短期利率过程**:
$$dr_t = (\theta(t) - \kappa r_t)dt + \sigma dW_t$$

**优势**: 可以精确拟合初始收益率曲线。

---

## 8.6.6 银行运营模型 / Banking Operations Models

### 流动性管理 / Liquidity Management

**流动性比率**: $LR = \frac{\text{流动资产}}{\text{流动负债}}$

**净稳定资金比率**: $NSFR = \frac{\text{可用稳定资金}}{\text{所需稳定资金}}$

### 资本充足率 / Capital Adequacy

**巴塞尔协议III**:
$$CAR = \frac{\text{核心资本}}{\text{风险加权资产}} \geq 8\%$$

**杠杆率**: $LR = \frac{\text{一级资本}}{\text{总资产}} \geq 3\%$

### 资产负债管理 / Asset-Liability Management

**久期匹配**: $D_A = D_L$

**缺口分析**: $GAP = RSA - RSL$

其中：

- $D_A$: 资产久期
- $D_L$: 负债久期
- $RSA$: 利率敏感资产
- $RSL$: 利率敏感负债

---

## 8.6.7 金融科技模型 / FinTech Models

### 机器学习信用评分 / Machine Learning Credit Scoring

**逻辑回归模型**:
$$P(\text{default}) = \frac{1}{1 + e^{-\beta^T x}}$$

**特征工程**:

- 收入水平
- 信用历史
- 债务收入比
- 就业状况

### 区块链模型 / Blockchain Models

**哈希函数**: $H(x) = \text{SHA256}(x)$

**工作量证明**: 寻找 $nonce$ 使得 $H(block + nonce) < target$

**共识机制**: 最长链原则

### 高频交易模型 / High-Frequency Trading Models

**市场微观结构**:

- 订单簿动态
- 价格冲击
- 流动性提供

**算法交易**:

- 统计套利
- 做市策略
- 动量交易

---

## 8.6.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;
use rand::Rng;
use statrs::distribution::{Normal, ContinuousCDF};

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub weights: Vec<f64>,
    pub returns: Vec<Vec<f64>>,
    pub assets: Vec<String>,
}

impl Portfolio {
    pub fn new(assets: Vec<String>) -> Self {
        Self {
            weights: vec![1.0 / assets.len() as f64; assets.len()],
            returns: Vec::new(),
            assets,
        }
    }
    
    pub fn add_returns(&mut self, returns: Vec<f64>) {
        self.returns.push(returns);
    }
    
    pub fn calculate_portfolio_return(&self) -> f64 {
        let mut portfolio_return = 0.0;
        for (i, weight) in self.weights.iter().enumerate() {
            let asset_return = self.returns.iter()
                .map(|r| r[i])
                .sum::<f64>() / self.returns.len() as f64;
            portfolio_return += weight * asset_return;
        }
        portfolio_return
    }
    
    pub fn calculate_portfolio_variance(&self) -> f64 {
        let mut variance = 0.0;
        for i in 0..self.weights.len() {
            for j in 0..self.weights.len() {
                let covariance = self.calculate_covariance(i, j);
                variance += self.weights[i] * self.weights[j] * covariance;
            }
        }
        variance
    }
    
    pub fn calculate_covariance(&self, i: usize, j: usize) -> f64 {
        let returns_i: Vec<f64> = self.returns.iter().map(|r| r[i]).collect();
        let returns_j: Vec<f64> = self.returns.iter().map(|r| r[j]).collect();
        
        let mean_i = returns_i.iter().sum::<f64>() / returns_i.len() as f64;
        let mean_j = returns_j.iter().sum::<f64>() / returns_j.len() as f64;
        
        returns_i.iter().zip(returns_j.iter())
            .map(|(r_i, r_j)| (r_i - mean_i) * (r_j - mean_j))
            .sum::<f64>() / (returns_i.len() - 1) as f64
    }
    
    pub fn calculate_var(&self, confidence_level: f64) -> f64 {
        let portfolio_returns: Vec<f64> = self.returns.iter()
            .map(|r| {
                r.iter().zip(self.weights.iter())
                    .map(|(ret, weight)| ret * weight)
                    .sum::<f64>()
            })
            .collect();
        
        let mean_return = portfolio_returns.iter().sum::<f64>() / portfolio_returns.len() as f64;
        let std_return = (portfolio_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (portfolio_returns.len() - 1) as f64).sqrt();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_score = normal.inverse_cdf(1.0 - confidence_level);
        
        mean_return + z_score * std_return
    }
}

#[derive(Debug)]
pub struct BlackScholes {
    pub s0: f64,  // 当前股价
    pub k: f64,   // 执行价格
    pub t: f64,   // 到期时间
    pub r: f64,   // 无风险利率
    pub sigma: f64, // 波动率
}

impl BlackScholes {
    pub fn new(s0: f64, k: f64, t: f64, r: f64, sigma: f64) -> Self {
        Self { s0, k, t, r, sigma }
    }
    
    pub fn call_price(&self) -> f64 {
        let d1 = (self.s0 / self.k).ln() + (self.r + self.sigma.powi(2) / 2.0) * self.t;
        let d1 = d1 / (self.sigma * self.t.sqrt());
        let d2 = d1 - self.sigma * self.t.sqrt();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        self.s0 * normal.cdf(d1) - self.k * (-self.r * self.t).exp() * normal.cdf(d2)
    }
    
    pub fn put_price(&self) -> f64 {
        let d1 = (self.s0 / self.k).ln() + (self.r + self.sigma.powi(2) / 2.0) * self.t;
        let d1 = d1 / (self.sigma * self.t.sqrt());
        let d2 = d1 - self.sigma * self.t.sqrt();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        self.k * (-self.r * self.t).exp() * normal.cdf(-d2) - self.s0 * normal.cdf(-d1)
    }
    
    pub fn delta(&self) -> f64 {
        let d1 = (self.s0 / self.k).ln() + (self.r + self.sigma.powi(2) / 2.0) * self.t;
        let d1 = d1 / (self.sigma * self.t.sqrt());
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(d1)
    }
}

// 使用示例
fn main() {
    // 投资组合示例
    let mut portfolio = Portfolio::new(vec!["AAPL".to_string(), "GOOGL".to_string()]);
    
    // 添加历史收益率数据
    for _ in 0..100 {
        let mut rng = rand::thread_rng();
        portfolio.add_returns(vec![rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)]);
    }
    
    println!("Portfolio return: {:.4}", portfolio.calculate_portfolio_return());
    println!("Portfolio variance: {:.4}", portfolio.calculate_portfolio_variance());
    println!("VaR (95%): {:.4}", portfolio.calculate_var(0.95));
    
    // Black-Scholes期权定价示例
    let bs = BlackScholes::new(100.0, 100.0, 1.0, 0.05, 0.2);
    println!("Call price: {:.4}", bs.call_price());
    println!("Put price: {:.4}", bs.put_price());
    println!("Delta: {:.4}", bs.delta());
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module BankingFinance where

import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.List (sum, length)
import System.Random (randomRs, newStdGen)

-- 投资组合数据类型
data Portfolio = Portfolio {
    weights :: Vector Double,
    returns :: [Vector Double],
    assets :: [String]
} deriving (Show)

-- 创建投资组合
newPortfolio :: [String] -> Portfolio
newPortfolio assets = Portfolio {
    weights = V.replicate (length assets) (1.0 / fromIntegral (length assets)),
    returns = [],
    assets = assets
}

-- 添加收益率数据
addReturns :: Vector Double -> Portfolio -> Portfolio
addReturns returns p = p { returns = returns : returns p }

-- 计算投资组合收益率
calculatePortfolioReturn :: Portfolio -> Double
calculatePortfolioReturn p = V.sum (V.zipWith (*) (weights p) avgReturns)
  where
    avgReturns = V.map (\i -> sum (map (V.! i) (returns p)) / fromIntegral (length (returns p))) 
                       (V.enumFromN 0 (V.length (weights p)))

-- 计算协方差
calculateCovariance :: Vector Double -> Vector Double -> Double
calculateCovariance xs ys = sum (zipWith (*) (V.map (subtract meanX) xs) (V.map (subtract meanY) ys)) / fromIntegral (V.length xs - 1)
  where
    meanX = V.sum xs / fromIntegral (V.length xs)
    meanY = V.sum ys / fromIntegral (V.length ys)

-- 计算投资组合方差
calculatePortfolioVariance :: Portfolio -> Double
calculatePortfolioVariance p = sum [weights p V.! i * weights p V.! j * covariance i j | i <- [0..V.length (weights p)-1], j <- [0..V.length (weights p)-1]]
  where
    covariance i j = calculateCovariance (V.map (V.! i) (V.fromList (returns p))) (V.map (V.! j) (V.fromList (returns p)))

-- Black-Scholes期权定价
data BlackScholes = BlackScholes {
    s0 :: Double,    -- 当前股价
    k :: Double,     -- 执行价格
    t :: Double,     -- 到期时间
    r :: Double,     -- 无风险利率
    sigma :: Double  -- 波动率
} deriving (Show)

-- 正态分布累积分布函数 (简化实现)
normalCDF :: Double -> Double
normalCDF x = 0.5 * (1 + erf (x / sqrt 2))
  where
    erf z = 2 / sqrt pi * sum [((-1)^n * z^(2*n+1)) / (fromIntegral (factorial n) * (2*n+1)) | n <- [0..10]]
    factorial n = product [1..n]

-- 计算d1和d2
calculateD1D2 :: BlackScholes -> (Double, Double)
calculateD1D2 bs = (d1, d2)
  where
    d1 = (log (s0 bs / k bs) + (r bs + sigma bs^2 / 2) * t bs) / (sigma bs * sqrt (t bs))
    d2 = d1 - sigma bs * sqrt (t bs)

-- 看涨期权价格
callPrice :: BlackScholes -> Double
callPrice bs = s0 bs * normalCDF d1 - k bs * exp (-r bs * t bs) * normalCDF d2
  where
    (d1, d2) = calculateD1D2 bs

-- 看跌期权价格
putPrice :: BlackScholes -> Double
putPrice bs = k bs * exp (-r bs * t bs) * normalCDF (-d2) - s0 bs * normalCDF (-d1)
  where
    (d1, d2) = calculateD1D2 bs

-- Delta
delta :: BlackScholes -> Double
delta bs = normalCDF d1
  where
    (d1, _) = calculateD1D2 bs

-- 示例使用
example :: IO ()
example = do
    -- 投资组合示例
    let portfolio = newPortfolio ["AAPL", "GOOGL"]
        returns1 = V.fromList [0.01, 0.02, -0.01, 0.03]
        returns2 = V.fromList [0.02, -0.01, 0.01, 0.02]
        portfolio' = addReturns returns1 $ addReturns returns2 portfolio
    
    putStrLn $ "Portfolio return: " ++ show (calculatePortfolioReturn portfolio')
    putStrLn $ "Portfolio variance: " ++ show (calculatePortfolioVariance portfolio')
    
    -- Black-Scholes示例
    let bs = BlackScholes 100.0 100.0 1.0 0.05 0.2
    
    putStrLn $ "Call price: " ++ show (callPrice bs)
    putStrLn $ "Put price: " ++ show (putPrice bs)
    putStrLn $ "Delta: " ++ show (delta bs)
```

### 应用领域 / Application Domains

#### 银行风险管理 / Banking Risk Management

- **信用风险**: 贷款违约概率评估
- **市场风险**: 利率、汇率、股价波动
- **操作风险**: 内部流程、系统故障

#### 投资管理 / Investment Management

- **资产配置**: 最优投资组合构建
- **业绩评估**: 风险调整后收益
- **基准管理**: 跟踪误差控制

#### 金融工程 / Financial Engineering

- **衍生品定价**: 期权、期货、互换
- **结构化产品**: 复杂金融工具设计
- **量化交易**: 算法交易策略

---

## 参考文献 / References

1. Hull, J. C. (2018). Options, Futures, and Other Derivatives. Pearson.
2. Markowitz, H. M. (1952). Portfolio Selection. Journal of Finance.
3. Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk. McGraw-Hill.
4. Merton, R. C. (1974). On the Pricing of Corporate Debt. Journal of Finance.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
