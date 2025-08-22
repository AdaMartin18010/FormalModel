# 贝叶斯统计多表征示例 / Bayesian Statistics Multi-Representation Example

## 概述 / Overview

本文档提供贝叶斯统计模型的多表征实现示例，包括贝叶斯推断、马尔可夫链蒙特卡洛和贝叶斯网络。

This document provides multi-representation implementation examples for Bayesian statistics models, including Bayesian inference, Markov Chain Monte Carlo, and Bayesian networks.

## 1. 贝叶斯推断 / Bayesian Inference

### 1.1 贝叶斯定理 / Bayes' Theorem

#### 数学表示 / Mathematical Representation

贝叶斯定理是贝叶斯统计的核心：

Bayes' theorem is the core of Bayesian statistics:

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

其中：

- $P(\theta|D)$ 是后验概率
- $P(D|\theta)$ 是似然函数
- $P(\theta)$ 是先验概率
- $P(D)$ 是证据（归一化常数）

where:

- $P(\theta|D)$ is the posterior probability
- $P(D|\theta)$ is the likelihood function
- $P(\theta)$ is the prior probability
- $P(D)$ is the evidence (normalization constant)

#### 可视化表示 / Visual Representation

```mermaid
graph TD
    A[先验 P(θ)] --> C[后验 P(θ|D)]
    B[似然 P(D|θ)] --> C
    D[数据 D] --> B
    
    subgraph "贝叶斯更新"
        E[先验知识] --> F[观察数据]
        F --> G[更新信念]
    end
```

#### Rust实现 / Rust Implementation

```rust
use std::f64::consts::PI;
use rand::Rng;
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
struct BayesianInference {
    prior_mean: f64,
    prior_std: f64,
    likelihood_std: f64,
}

impl BayesianInference {
    fn new(prior_mean: f64, prior_std: f64, likelihood_std: f64) -> Self {
        Self {
            prior_mean,
            prior_std,
            likelihood_std,
        }
    }
    
    fn prior_pdf(&self, theta: f64) -> f64 {
        let normal = Normal::new(self.prior_mean, self.prior_std).unwrap();
        normal.pdf(theta)
    }
    
    fn likelihood(&self, data: &[f64], theta: f64) -> f64 {
        let normal = Normal::new(theta, self.likelihood_std).unwrap();
        data.iter().map(|&x| normal.pdf(x)).product()
    }
    
    fn posterior_pdf(&self, data: &[f64], theta: f64) -> f64 {
        let prior = self.prior_pdf(theta);
        let likelihood = self.likelihood(data, theta);
        prior * likelihood
    }
    
    fn conjugate_update(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let data_mean = data.iter().sum::<f64>() / n;
        
        // 正态-正态共轭先验的更新公式
        let prior_precision = 1.0 / (self.prior_std * self.prior_std);
        let likelihood_precision = n / (self.likelihood_std * self.likelihood_std);
        
        let posterior_precision = prior_precision + likelihood_precision;
        let posterior_std = 1.0 / posterior_precision.sqrt();
        
        let posterior_mean = (prior_precision * self.prior_mean + 
                             likelihood_precision * data_mean) / posterior_precision;
        
        (posterior_mean, posterior_std)
    }
    
    fn metropolis_hastings(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n_samples);
        let mut current_theta = self.prior_mean;
        
        for _ in 0..n_samples {
            // 提议分布：正态分布
            let proposal_std = 0.1;
            let proposal = Normal::new(current_theta, proposal_std).unwrap();
            let proposed_theta = proposal.sample(&mut rng);
            
            // 计算接受概率
            let current_posterior = self.posterior_pdf(data, current_theta);
            let proposed_posterior = self.posterior_pdf(data, proposed_theta);
            
            let acceptance_ratio = proposed_posterior / current_posterior;
            let acceptance_prob = acceptance_ratio.min(1.0);
            
            if rng.gen::<f64>() < acceptance_prob {
                current_theta = proposed_theta;
            }
            
            samples.push(current_theta);
        }
        
        samples
    }
}

fn main() {
    // 生成模拟数据
    let true_theta = 2.0;
    let data_std = 1.0;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(true_theta, data_std).unwrap();
    
    let data: Vec<f64> = (0..100).map(|_| normal.sample(&mut rng)).collect();
    
    // 贝叶斯推断
    let bayes = BayesianInference::new(0.0, 2.0, data_std);
    
    // 共轭更新
    let (posterior_mean, posterior_std) = bayes.conjugate_update(&data);
    println!("共轭后验 - 均值: {:.3}, 标准差: {:.3}", posterior_mean, posterior_std);
    
    // MCMC采样
    let samples = bayes.metropolis_hastings(&data, 10000);
    let mcmc_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let mcmc_var = samples.iter()
        .map(|&x| (x - mcmc_mean).powi(2))
        .sum::<f64>() / samples.len() as f64;
    let mcmc_std = mcmc_var.sqrt();
    
    println!("MCMC后验 - 均值: {:.3}, 标准差: {:.3}", mcmc_mean, mcmc_std);
    println!("真实值: {:.3}", true_theta);
}
```

#### Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from scipy import stats
from scipy.stats import norm
import seaborn as sns

@dataclass
class BayesianInference:
    """贝叶斯推断"""
    prior_mean: float
    prior_std: float
    likelihood_std: float
    
    def prior_pdf(self, theta: np.ndarray) -> np.ndarray:
        """先验概率密度函数"""
        return norm.pdf(theta, self.prior_mean, self.prior_std)
    
    def likelihood(self, data: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """似然函数"""
        if np.isscalar(theta):
            theta = np.array([theta])
        
        likelihoods = np.ones(len(theta))
        for i, t in enumerate(theta):
            likelihoods[i] = np.prod(norm.pdf(data, t, self.likelihood_std))
        
        return likelihoods
    
    def posterior_pdf(self, data: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """后验概率密度函数"""
        prior = self.prior_pdf(theta)
        likelihood = self.likelihood(data, theta)
        return prior * likelihood
    
    def conjugate_update(self, data: np.ndarray) -> Tuple[float, float]:
        """共轭先验更新"""
        n = len(data)
        data_mean = np.mean(data)
        
        # 正态-正态共轭先验的更新公式
        prior_precision = 1.0 / (self.prior_std ** 2)
        likelihood_precision = n / (self.likelihood_std ** 2)
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_std = 1.0 / np.sqrt(posterior_precision)
        
        posterior_mean = (prior_precision * self.prior_mean + 
                         likelihood_precision * data_mean) / posterior_precision
        
        return posterior_mean, posterior_std
    
    def metropolis_hastings(self, data: np.ndarray, n_samples: int, 
                           proposal_std: float = 0.1) -> np.ndarray:
        """Metropolis-Hastings算法"""
        samples = np.zeros(n_samples)
        current_theta = self.prior_mean
        
        for i in range(n_samples):
            # 提议分布
            proposed_theta = current_theta + np.random.normal(0, proposal_std)
            
            # 计算接受概率
            current_posterior = self.posterior_pdf(data, current_theta)[0]
            proposed_posterior = self.posterior_pdf(data, proposed_theta)[0]
            
            acceptance_ratio = proposed_posterior / current_posterior
            acceptance_prob = min(1.0, acceptance_ratio)
            
            if np.random.random() < acceptance_prob:
                current_theta = proposed_theta
            
            samples[i] = current_theta
        
        return samples

def visualize_bayesian_inference(bayes: BayesianInference, data: np.ndarray, 
                                samples: np.ndarray, true_theta: float) -> None:
    """可视化贝叶斯推断结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 先验分布
    theta_range = np.linspace(-5, 5, 1000)
    prior = bayes.prior_pdf(theta_range)
    axes[0, 0].plot(theta_range, prior, 'b-', label='Prior')
    axes[0, 0].set_title('Prior Distribution')
    axes[0, 0].set_xlabel('θ')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 似然函数
    likelihood = bayes.likelihood(data, theta_range)
    axes[0, 1].plot(theta_range, likelihood, 'g-', label='Likelihood')
    axes[0, 1].set_title('Likelihood Function')
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_ylabel('Likelihood')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 后验分布
    posterior = bayes.posterior_pdf(data, theta_range)
    axes[1, 0].plot(theta_range, posterior, 'r-', label='Posterior')
    axes[1, 0].axvline(true_theta, color='k', linestyle='--', label=f'True θ = {true_theta}')
    axes[1, 0].set_title('Posterior Distribution')
    axes[1, 0].set_xlabel('θ')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MCMC样本分布
    axes[1, 1].hist(samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
    axes[1, 1].plot(theta_range, posterior, 'r-', label='Analytical Posterior')
    axes[1, 1].axvline(true_theta, color='k', linestyle='--', label=f'True θ = {true_theta}')
    axes[1, 1].set_title('MCMC Samples vs Analytical Posterior')
    axes[1, 1].set_xlabel('θ')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 测试贝叶斯推断
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    true_theta = 2.0
    data_std = 1.0
    data = np.random.normal(true_theta, data_std, 100)
    
    # 贝叶斯推断
    bayes = BayesianInference(prior_mean=0.0, prior_std=2.0, likelihood_std=data_std)
    
    # 共轭更新
    posterior_mean, posterior_std = bayes.conjugate_update(data)
    print(f"共轭后验 - 均值: {posterior_mean:.3f}, 标准差: {posterior_std:.3f}")
    
    # MCMC采样
    samples = bayes.metropolis_hastings(data, n_samples=10000)
    mcmc_mean = np.mean(samples)
    mcmc_std = np.std(samples)
    print(f"MCMC后验 - 均值: {mcmc_mean:.3f}, 标准差: {mcmc_std:.3f}")
    print(f"真实值: {true_theta:.3f}")
    
    # 可视化结果
    visualize_bayesian_inference(bayes, data, samples, true_theta)
```

## 2. 马尔可夫链蒙特卡洛 / Markov Chain Monte Carlo

### 2.1 Gibbs采样 / Gibbs Sampling

#### 2.1.1 数学表示 / Mathematical Representation

Gibbs采样是一种特殊的MCMC方法，用于从多维分布中采样：

Gibbs sampling is a special MCMC method for sampling from multivariate distributions:

$$P(\theta_i|\theta_{-i}, D) \propto P(D|\theta)P(\theta)$$

其中 $\theta_{-i}$ 表示除 $\theta_i$ 外的所有参数。

where $\theta_{-i}$ represents all parameters except $\theta_i$.

#### 2.1.2 可视化表示 / Visual Representation

```mermaid
graph TD
    A[初始化参数] --> B[更新θ₁|θ₂,θ₃]
    B --> C[更新θ₂|θ₁,θ₃]
    C --> D[更新θ₃|θ₁,θ₂]
    D --> E{收敛?}
    E -->|否| B
    E -->|是| F[输出样本]
    
    subgraph "Gibbs采样循环"
        G[条件分布] --> H[参数更新]
        H --> I[样本收集]
    end
```

#### 2.1.3 Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from scipy.stats import norm, gamma, multivariate_normal

@dataclass
class GibbsSampler:
    """Gibbs采样器"""
    data: np.ndarray
    n_samples: int
    burn_in: int = 1000
    
    def __post_init__(self):
        self.n_data = len(self.data)
        self.data_mean = np.mean(self.data)
        self.data_var = np.var(self.data)
    
    def sample_mu_given_sigma(self, sigma: float, mu_prior_mean: float = 0.0, 
                             mu_prior_std: float = 10.0) -> float:
        """给定σ采样μ"""
        # 先验精度
        prior_precision = 1.0 / (mu_prior_std ** 2)
        # 似然精度
        likelihood_precision = self.n_data / (sigma ** 2)
        
        # 后验参数
        posterior_precision = prior_precision + likelihood_precision
        posterior_std = 1.0 / np.sqrt(posterior_precision)
        posterior_mean = (prior_precision * mu_prior_mean + 
                         likelihood_precision * self.data_mean) / posterior_precision
        
        return np.random.normal(posterior_mean, posterior_std)
    
    def sample_sigma_given_mu(self, mu: float, alpha: float = 1.0, 
                             beta: float = 1.0) -> float:
        """给定μ采样σ²"""
        # 计算残差平方和
        ss = np.sum((self.data - mu) ** 2)
        
        # 后验参数
        posterior_alpha = alpha + self.n_data / 2
        posterior_beta = beta + ss / 2
        
        # 从逆Gamma分布采样σ²
        sigma_squared = np.random.gamma(posterior_alpha, 1.0 / posterior_beta)
        return np.sqrt(sigma_squared)
    
    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """运行Gibbs采样"""
        mu_samples = np.zeros(self.n_samples)
        sigma_samples = np.zeros(self.n_samples)
        
        # 初始化
        mu = self.data_mean
        sigma = np.sqrt(self.data_var)
        
        for i in range(self.n_samples + self.burn_in):
            # Gibbs采样步骤
            mu = self.sample_mu_given_sigma(sigma)
            sigma = self.sample_sigma_given_mu(mu)
            
            if i >= self.burn_in:
                mu_samples[i - self.burn_in] = mu
                sigma_samples[i - self.burn_in] = sigma
        
        return mu_samples, sigma_samples
    
    def diagnose_convergence(self, samples: np.ndarray, param_name: str) -> None:
        """诊断收敛性"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 轨迹图
        axes[0, 0].plot(samples)
        axes[0, 0].set_title(f'{param_name} Trace Plot')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel(param_name)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 自相关图
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(samples, ax=axes[0, 1], lags=50)
        axes[0, 1].set_title(f'{param_name} Autocorrelation')
        
        # 直方图
        axes[1, 0].hist(samples, bins=50, density=True, alpha=0.7)
        axes[1, 0].set_title(f'{param_name} Histogram')
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累积均值
        cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        axes[1, 1].plot(cumulative_mean)
        axes[1, 1].set_title(f'{param_name} Cumulative Mean')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Cumulative Mean')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def visualize_gibbs_results(mu_samples: np.ndarray, sigma_samples: np.ndarray, 
                           true_mu: float, true_sigma: float) -> None:
    """可视化Gibbs采样结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # μ的分布
    axes[0, 0].hist(mu_samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
    axes[0, 0].axvline(true_mu, color='r', linestyle='--', label=f'True μ = {true_mu}')
    axes[0, 0].axvline(np.mean(mu_samples), color='g', linestyle='--', 
                       label=f'Estimated μ = {np.mean(mu_samples):.3f}')
    axes[0, 0].set_title('μ Posterior Distribution')
    axes[0, 0].set_xlabel('μ')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # σ的分布
    axes[0, 1].hist(sigma_samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
    axes[0, 1].axvline(true_sigma, color='r', linestyle='--', label=f'True σ = {true_sigma}')
    axes[0, 1].axvline(np.mean(sigma_samples), color='g', linestyle='--', 
                       label=f'Estimated σ = {np.mean(sigma_samples):.3f}')
    axes[0, 1].set_title('σ Posterior Distribution')
    axes[0, 1].set_xlabel('σ')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 联合分布
    axes[1, 0].scatter(mu_samples, sigma_samples, alpha=0.5, s=1)
    axes[1, 0].axvline(true_mu, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(true_sigma, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Joint Posterior Distribution')
    axes[1, 0].set_xlabel('μ')
    axes[1, 0].set_ylabel('σ')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 数据拟合
    x_range = np.linspace(true_mu - 3*true_sigma, true_mu + 3*true_sigma, 100)
    axes[1, 1].hist(data, bins=30, density=True, alpha=0.7, label='Data')
    
    # 真实分布
    true_pdf = norm.pdf(x_range, true_mu, true_sigma)
    axes[1, 1].plot(x_range, true_pdf, 'r-', label='True Distribution')
    
    # 估计分布
    estimated_pdf = norm.pdf(x_range, np.mean(mu_samples), np.mean(sigma_samples))
    axes[1, 1].plot(x_range, estimated_pdf, 'g-', label='Estimated Distribution')
    
    axes[1, 1].set_title('Data Fitting')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 测试Gibbs采样
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    true_mu = 2.0
    true_sigma = 1.5
    data = np.random.normal(true_mu, true_sigma, 100)
    
    # Gibbs采样
    gibbs = GibbsSampler(data, n_samples=5000, burn_in=1000)
    mu_samples, sigma_samples = gibbs.run()
    
    print(f"真实参数 - μ: {true_mu}, σ: {true_sigma}")
    print(f"估计参数 - μ: {np.mean(mu_samples):.3f} ± {np.std(mu_samples):.3f}")
    print(f"估计参数 - σ: {np.mean(sigma_samples):.3f} ± {np.std(sigma_samples):.3f}")
    
    # 诊断收敛性
    gibbs.diagnose_convergence(mu_samples, "μ")
    gibbs.diagnose_convergence(sigma_samples, "σ")
    
    # 可视化结果
    visualize_gibbs_results(mu_samples, sigma_samples, true_mu, true_sigma)
```

## 3. 贝叶斯网络 / Bayesian Networks

### 3.1 简单贝叶斯网络 / Simple Bayesian Network

#### 3.1.1 数学表示 / Mathematical Representation

贝叶斯网络表示联合概率分布：

Bayesian networks represent joint probability distributions:

$$P(X_1, X_2, ..., X_n) = \prod_{i=1}^{n} P(X_i|Pa(X_i))$$

其中 $Pa(X_i)$ 是 $X_i$ 的父节点。

where $Pa(X_i)$ are the parents of $X_i$.

#### 3.1.2 可视化表示 / Visual Representation

```mermaid
graph TD
    A[天气] --> C[洒水器]
    A --> D[草地湿润]
    C --> D
    D --> E[鞋子湿]
    
    subgraph "条件概率表"
        F[P(洒水器|天气)]
        G[P(草地湿润|天气,洒水器)]
        H[P(鞋子湿|草地湿润)]
    end
```

#### 3.1.3 Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx as nx

@dataclass
class BayesianNetwork:
    """贝叶斯网络"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    cpt: Dict[str, np.ndarray]
    
    def __post_init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
    
    def sample(self, n_samples: int) -> Dict[str, np.ndarray]:
        """从贝叶斯网络采样"""
        samples = {node: np.zeros(n_samples, dtype=int) for node in self.nodes}
        
        # 拓扑排序
        topo_order = list(nx.topological_sort(self.graph))
        
        for i in range(n_samples):
            for node in topo_order:
                parents = list(self.graph.predecessors(node))
                
                if not parents:
                    # 根节点
                    p = self.cpt[node]
                    samples[node][i] = np.random.choice(len(p), p=p)
                else:
                    # 有父节点的节点
                    parent_values = [samples[parent][i] for parent in parents]
                    parent_key = tuple(parent_values)
                    p = self.cpt[node][parent_key]
                    samples[node][i] = np.random.choice(len(p), p=p)
        
        return samples
    
    def joint_probability(self, evidence: Dict[str, int]) -> float:
        """计算联合概率"""
        prob = 1.0
        
        for node in self.nodes:
            parents = list(self.graph.predecessors(node))
            
            if not parents:
                # 根节点
                prob *= self.cpt[node][evidence[node]]
            else:
                # 有父节点的节点
                parent_values = tuple(evidence[parent] for parent in parents)
                prob *= self.cpt[node][parent_values][evidence[node]]
        
        return prob
    
    def inference(self, query: str, evidence: Dict[str, int], 
                  n_samples: int = 10000) -> np.ndarray:
        """贝叶斯推断"""
        samples = self.sample(n_samples)
        
        # 过滤符合证据的样本
        valid_samples = np.ones(n_samples, dtype=bool)
        for var, value in evidence.items():
            valid_samples &= (samples[var] == value)
        
        if not np.any(valid_samples):
            return np.array([0.5, 0.5])  # 默认均匀分布
        
        # 计算查询变量的条件概率
        query_samples = samples[query][valid_samples]
        counts = np.bincount(query_samples, minlength=2)
        return counts / np.sum(counts)
    
    def visualize_network(self) -> None:
        """可视化贝叶斯网络"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=12, font_weight='bold',
                arrows=True, edge_color='gray', arrowsize=20)
        
        plt.title("Bayesian Network Structure")
        plt.show()

def create_sprinkler_network() -> BayesianNetwork:
    """创建洒水器贝叶斯网络"""
    nodes = ['Weather', 'Sprinkler', 'WetGrass', 'WetShoes']
    edges = [('Weather', 'Sprinkler'), ('Weather', 'WetGrass'), 
             ('Sprinkler', 'WetGrass'), ('WetGrass', 'WetShoes')]
    
    # 条件概率表
    cpt = {
        'Weather': np.array([0.7, 0.3]),  # P(Weather=Sunny), P(Weather=Rainy)
        
        'Sprinkler': {
            (0,): np.array([0.4, 0.6]),  # P(Sprinkler|Weather=Sunny)
            (1,): np.array([0.01, 0.99])  # P(Sprinkler|Weather=Rainy)
        },
        
        'WetGrass': {
            (0, 0): np.array([0.99, 0.01]),  # P(WetGrass|Weather=Sunny, Sprinkler=Off)
            (0, 1): np.array([0.1, 0.9]),    # P(WetGrass|Weather=Sunny, Sprinkler=On)
            (1, 0): np.array([0.2, 0.8]),    # P(WetGrass|Weather=Rainy, Sprinkler=Off)
            (1, 1): np.array([0.01, 0.99])   # P(WetGrass|Weather=Rainy, Sprinkler=On)
        },
        
        'WetShoes': {
            (0,): np.array([0.9, 0.1]),   # P(WetShoes|WetGrass=No)
            (1,): np.array([0.1, 0.9])    # P(WetShoes|WetGrass=Yes)
        }
    }
    
    return BayesianNetwork(nodes, edges, cpt)

def visualize_inference_results(network: BayesianNetwork, n_samples: int = 10000) -> None:
    """可视化推断结果"""
    # 生成样本
    samples = network.sample(n_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 天气分布
    weather_counts = np.bincount(samples['Weather'])
    axes[0, 0].bar(['Sunny', 'Rainy'], weather_counts)
    axes[0, 0].set_title('Weather Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # 洒水器分布
    sprinkler_counts = np.bincount(samples['Sprinkler'])
    axes[0, 1].bar(['Off', 'On'], sprinkler_counts)
    axes[0, 1].set_title('Sprinkler Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # 草地湿润分布
    wetgrass_counts = np.bincount(samples['WetGrass'])
    axes[1, 0].bar(['No', 'Yes'], wetgrass_counts)
    axes[1, 0].set_title('Wet Grass Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # 鞋子湿分布
    wetshoes_counts = np.bincount(samples['WetShoes'])
    axes[1, 1].bar(['No', 'Yes'], wetshoes_counts)
    axes[1, 1].set_title('Wet Shoes Distribution')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # 条件概率推断
    print("贝叶斯推断结果:")
    
    # P(WetShoes=Yes|Weather=Rainy)
    evidence = {'Weather': 1}  # Rainy
    prob = network.inference('WetShoes', evidence, n_samples)
    print(f"P(WetShoes=Yes|Weather=Rainy) = {prob[1]:.3f}")
    
    # P(WetShoes=Yes|Weather=Sunny)
    evidence = {'Weather': 0}  # Sunny
    prob = network.inference('WetShoes', evidence, n_samples)
    print(f"P(WetShoes=Yes|Weather=Sunny) = {prob[1]:.3f}")
    
    # P(WetShoes=Yes|Sprinkler=On)
    evidence = {'Sprinkler': 1}  # On
    prob = network.inference('WetShoes', evidence, n_samples)
    print(f"P(WetShoes=Yes|Sprinkler=On) = {prob[1]:.3f}")

# 测试贝叶斯网络
if __name__ == "__main__":
    # 创建洒水器网络
    network = create_sprinkler_network()
    
    # 可视化网络结构
    network.visualize_network()
    
    # 生成样本并可视化
    visualize_inference_results(network)
    
    # 计算一些联合概率
    evidence = {'Weather': 1, 'Sprinkler': 0, 'WetGrass': 1, 'WetShoes': 1}
    joint_prob = network.joint_probability(evidence)
    print(f"P(Weather=Rainy, Sprinkler=Off, WetGrass=Yes, WetShoes=Yes) = {joint_prob:.6f}")
```

## 总结 / Summary

本文档提供了贝叶斯统计模型的多表征实现示例，包括：

This document provides multi-representation implementation examples for Bayesian statistics models, including:

1. **贝叶斯推断** / Bayesian Inference
   - 贝叶斯定理 / Bayes' Theorem
   - 共轭先验更新 / Conjugate Prior Updates
   - Metropolis-Hastings算法 / Metropolis-Hastings Algorithm

2. **马尔可夫链蒙特卡洛** / Markov Chain Monte Carlo
   - Gibbs采样 / Gibbs Sampling
   - 收敛性诊断 / Convergence Diagnostics
   - 后验分布可视化 / Posterior Distribution Visualization

3. **贝叶斯网络** / Bayesian Networks
   - 网络结构 / Network Structure
   - 条件概率表 / Conditional Probability Tables
   - 贝叶斯推断 / Bayesian Inference

每个模型都包含数学表示、可视化图表、Rust/Python实现，展示了贝叶斯统计在不同领域的应用。

Each model includes mathematical representation, visual diagrams, and Rust/Python implementations, demonstrating the applications of Bayesian statistics in different domains.
