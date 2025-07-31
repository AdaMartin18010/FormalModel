# 7.1 优化模型 / Optimization Models

## 目录 / Table of Contents

- [7.1 优化模型 / Optimization Models](#71-优化模型--optimization-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.1.1 线性规划模型 / Linear Programming Models](#711-线性规划模型--linear-programming-models)
    - [标准形式 / Standard Form](#标准形式--standard-form)
    - [单纯形法 / Simplex Method](#单纯形法--simplex-method)
    - [内点法 / Interior Point Method](#内点法--interior-point-method)
  - [7.1.2 非线性规划模型 / Nonlinear Programming Models](#712-非线性规划模型--nonlinear-programming-models)
    - [无约束优化 / Unconstrained Optimization](#无约束优化--unconstrained-optimization)
    - [约束优化 / Constrained Optimization](#约束优化--constrained-optimization)
  - [7.1.3 动态规划模型 / Dynamic Programming Models](#713-动态规划模型--dynamic-programming-models)
    - [贝尔曼方程 / Bellman Equation](#贝尔曼方程--bellman-equation)
    - [背包问题 / Knapsack Problem](#背包问题--knapsack-problem)
    - [最短路径 / Shortest Path](#最短路径--shortest-path)
  - [7.1.4 遗传算法模型 / Genetic Algorithm Models](#714-遗传算法模型--genetic-algorithm-models)
    - [编码与解码 / Encoding and Decoding](#编码与解码--encoding-and-decoding)
    - [选择算子 / Selection Operators](#选择算子--selection-operators)
    - [交叉算子 / Crossover Operators](#交叉算子--crossover-operators)
    - [变异算子 / Mutation Operators](#变异算子--mutation-operators)
  - [7.1.5 模拟退火模型 / Simulated Annealing Models](#715-模拟退火模型--simulated-annealing-models)
    - [温度调度 / Temperature Schedule](#温度调度--temperature-schedule)
    - [接受准则 / Acceptance Criterion](#接受准则--acceptance-criterion)
    - [邻域结构 / Neighborhood Structure](#邻域结构--neighborhood-structure)
  - [7.1.6 粒子群优化模型 / Particle Swarm Optimization Models](#716-粒子群优化模型--particle-swarm-optimization-models)
    - [速度更新 / Velocity Update](#速度更新--velocity-update)
    - [位置更新 / Position Update](#位置更新--position-update)
    - [拓扑结构 / Topology Structure](#拓扑结构--topology-structure)
  - [7.1.7 实现与应用 / Implementation and Applications](#717-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [运筹学 / Operations Research](#运筹学--operations-research)
      - [工程设计 / Engineering Design](#工程设计--engineering-design)
      - [人工智能 / Artificial Intelligence](#人工智能--artificial-intelligence)
  - [参考文献 / References](#参考文献--references)

---

## 7.1.1 线性规划模型 / Linear Programming Models

### 标准形式 / Standard Form

**目标函数**: $\min c^T x$

**约束条件**:

- $Ax = b$
- $x \geq 0$

**对偶问题**: $\max b^T y$ s.t. $A^T y \leq c$

### 单纯形法 / Simplex Method

**基变量**: $x_B = B^{-1}b$

**非基变量**: $x_N = 0$

**检验数**: $\bar{c}_j = c_j - c_B^T B^{-1} A_j$

**最优性条件**: $\bar{c}_j \geq 0$ for all $j$

### 内点法 / Interior Point Method

**中心路径**: $x(\mu) = \arg\min \{c^T x - \mu \sum_i \ln x_i\}$

**障碍函数**: $f(x) = c^T x - \mu \sum_i \ln x_i$

**牛顿方向**: $\Delta x = -H^{-1} \nabla f(x)$

---

## 7.1.2 非线性规划模型 / Nonlinear Programming Models

### 无约束优化 / Unconstrained Optimization

**梯度下降**: $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$

**牛顿法**: $x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)$

**拟牛顿法**: $H_{k+1} = H_k + \frac{(y_k - H_k s_k)(y_k - H_k s_k)^T}{(y_k - H_k s_k)^T s_k}$

### 约束优化 / Constrained Optimization

**拉格朗日函数**: $L(x, \lambda) = f(x) + \sum_i \lambda_i g_i(x)$

**KKT条件**:

- $\nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) = 0$
- $g_i(x^*) \leq 0$
- $\lambda_i^* \geq 0$
- $\lambda_i^* g_i(x^*) = 0$

**惩罚函数**: $P(x, \mu) = f(x) + \frac{\mu}{2} \sum_i [g_i(x)]_+^2$

---

## 7.1.3 动态规划模型 / Dynamic Programming Models

### 贝尔曼方程 / Bellman Equation

**值函数**: $V(s) = \max_a \{r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\}$

**最优策略**: $\pi^*(s) = \arg\max_a Q(s,a)$

**Q函数**: $Q(s,a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q(s',a')$

### 背包问题 / Knapsack Problem

**状态转移**: $dp[i][w] = \max(dp[i-1][w], dp[i-1][w-w_i] + v_i)$

**最优解**: $dp[n][W]$

**回溯路径**: $selected[i] = (dp[i][w] > dp[i-1][w])$

### 最短路径 / Shortest Path

**Floyd-Warshall**: $d[i][j] = \min(d[i][j], d[i][k] + d[k][j])$

**Dijkstra算法**: $d[v] = \min(d[v], d[u] + w(u,v))$

**Bellman-Ford**: $d[v] = \min(d[v], d[u] + w(u,v))$

---

## 7.1.4 遗传算法模型 / Genetic Algorithm Models

### 编码与解码 / Encoding and Decoding

**二进制编码**: $x = \sum_{i=0}^{n-1} b_i 2^i$

**实数编码**: $x = x_{min} + \frac{x_{max} - x_{min}}{2^n - 1} \sum_{i=0}^{n-1} b_i 2^i$

**适应度函数**: $f(x) = \frac{1}{1 + \text{objective}(x)}$

### 选择算子 / Selection Operators

**轮盘赌选择**: $P(i) = \frac{f_i}{\sum_j f_j}$

**锦标赛选择**: $P(i) = \binom{n-1}{k-1} p^{k-1}(1-p)^{n-k}$

**排序选择**: $P(i) = \frac{2(n-i+1)}{n(n+1)}$

### 交叉算子 / Crossover Operators

**单点交叉**: $child_1 = parent_1[:k] + parent_2[k:]$

**双点交叉**: $child_1 = parent_1[:k_1] + parent_2[k_1:k_2] + parent_1[k_2:]$

**均匀交叉**: $child_1[i] = \begin{cases} parent_1[i] & \text{if } r < 0.5 \\ parent_2[i] & \text{otherwise} \end{cases}$

### 变异算子 / Mutation Operators

**位翻转变异**: $P(mutation) = \frac{1}{L}$

**高斯变异**: $x' = x + N(0, \sigma^2)$

**多项式变异**: $x' = x + \eta_m (x_{max} - x_{min})$

---

## 7.1.5 模拟退火模型 / Simulated Annealing Models

### 温度调度 / Temperature Schedule

**指数冷却**: $T(t) = T_0 \alpha^t$

**线性冷却**: $T(t) = T_0 - \frac{T_0 - T_f}{t_{max}} t$

**对数冷却**: $T(t) = \frac{T_0}{\ln(1 + t)}$

### 接受准则 / Acceptance Criterion

**Metropolis准则**: $P(accept) = \min(1, e^{-\frac{\Delta E}{T}})$

**Boltzmann准则**: $P(accept) = \frac{1}{1 + e^{\frac{\Delta E}{T}}}$

**阈值准则**: $P(accept) = \begin{cases} 1 & \text{if } \Delta E \leq 0 \\ e^{-\frac{\Delta E}{T}} & \text{otherwise} \end{cases}$

### 邻域结构 / Neighborhood Structure

**2-opt**: 交换两条边

**3-opt**: 交换三条边

**插入**: 将城市插入到新位置

---

## 7.1.6 粒子群优化模型 / Particle Swarm Optimization Models

### 速度更新 / Velocity Update

**标准PSO**: $v_i(t+1) = w v_i(t) + c_1 r_1 (p_i - x_i(t)) + c_2 r_2 (g - x_i(t))$

**惯性权重**: $w = w_{max} - \frac{w_{max} - w_{min}}{t_{max}} t$

**学习因子**: $c_1 = c_{1i} - \frac{c_{1i} - c_{1f}}{t_{max}} t$

### 位置更新 / Position Update

**位置更新**: $x_i(t+1) = x_i(t) + v_i(t+1)$

**边界处理**: $x_i = \max(x_{min}, \min(x_{max}, x_i))$

**速度限制**: $v_i = \max(-v_{max}, \min(v_{max}, v_i))$

### 拓扑结构 / Topology Structure

**全局拓扑**: $g = \arg\min_{j} f(p_j)$

**环形拓扑**: $g_i = \arg\min_{j \in N_i} f(p_j)$

**冯诺依曼拓扑**: 4个邻居

---

## 7.1.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LinearProgram {
    pub objective: Vec<f64>,
    pub constraints: Vec<Vec<f64>>,
    pub rhs: Vec<f64>,
    pub bounds: Vec<(f64, f64)>,
}

impl LinearProgram {
    pub fn new(objective: Vec<f64>) -> Self {
        Self {
            objective,
            constraints: Vec::new(),
            rhs: Vec::new(),
            bounds: Vec::new(),
        }
    }
    
    pub fn add_constraint(&mut self, constraint: Vec<f64>, rhs: f64) {
        self.constraints.push(constraint);
        self.rhs.push(rhs);
    }
    
    pub fn set_bounds(&mut self, bounds: Vec<(f64, f64)>) {
        self.bounds = bounds;
    }
    
    pub fn solve_simplex(&self) -> Option<Vec<f64>> {
        // 简化的单纯形法实现
        let n_vars = self.objective.len();
        let n_constraints = self.constraints.len();
        
        // 构造标准形式
        let mut tableau = vec![vec![0.0; n_vars + n_constraints + 1]; n_constraints + 1];
        
        // 目标函数行
        for j in 0..n_vars {
            tableau[0][j] = -self.objective[j];
        }
        
        // 约束条件
        for i in 0..n_constraints {
            for j in 0..n_vars {
                tableau[i + 1][j] = self.constraints[i][j];
            }
            tableau[i + 1][n_vars + i] = 1.0; // 松弛变量
            tableau[i + 1][n_vars + n_constraints] = self.rhs[i];
        }
        
        // 迭代求解
        for _ in 0..100 {
            // 选择入基变量
            let mut entering = None;
            for j in 0..n_vars + n_constraints {
                if tableau[0][j] < -1e-10 {
                    entering = Some(j);
                    break;
                }
            }
            
            if entering.is_none() {
                break; // 最优解
            }
            
            let entering_col = entering.unwrap();
            
            // 选择出基变量
            let mut leaving = None;
            let mut min_ratio = f64::INFINITY;
            
            for i in 1..n_constraints + 1 {
                if tableau[i][entering_col] > 1e-10 {
                    let ratio = tableau[i][n_vars + n_constraints] / tableau[i][entering_col];
                    if ratio < min_ratio {
                        min_ratio = ratio;
                        leaving = Some(i);
                    }
                }
            }
            
            if leaving.is_none() {
                return None; // 无界解
            }
            
            let leaving_row = leaving.unwrap();
            
            // 高斯消元
            let pivot = tableau[leaving_row][entering_col];
            for j in 0..n_vars + n_constraints + 1 {
                tableau[leaving_row][j] /= pivot;
            }
            
            for i in 0..n_constraints + 1 {
                if i != leaving_row {
                    let factor = tableau[i][entering_col];
                    for j in 0..n_vars + n_constraints + 1 {
                        tableau[i][j] -= factor * tableau[leaving_row][j];
                    }
                }
            }
        }
        
        // 提取解
        let mut solution = vec![0.0; n_vars];
        for i in 1..n_constraints + 1 {
            let mut basic_var = None;
            for j in 0..n_vars + n_constraints {
                if (tableau[i][j] - 1.0).abs() < 1e-10 {
                    basic_var = Some(j);
                    break;
                }
            }
            
            if let Some(j) = basic_var {
                if j < n_vars {
                    solution[j] = tableau[i][n_vars + n_constraints];
                }
            }
        }
        
        Some(solution)
    }
}

#[derive(Debug, Clone)]
pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub chromosome_length: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub population: Vec<Vec<bool>>,
}

impl GeneticAlgorithm {
    pub fn new(population_size: usize, chromosome_length: usize) -> Self {
        let mut population = Vec::new();
        for _ in 0..population_size {
            let chromosome = (0..chromosome_length)
                .map(|_| rand::random::<bool>())
                .collect();
            population.push(chromosome);
        }
        
        Self {
            population_size,
            chromosome_length,
            mutation_rate: 0.01,
            crossover_rate: 0.8,
            population,
        }
    }
    
    pub fn fitness(&self, chromosome: &[bool]) -> f64 {
        // 示例适应度函数：计算1的个数
        chromosome.iter().filter(|&&x| x).count() as f64
    }
    
    pub fn selection(&self) -> Vec<Vec<bool>> {
        let mut new_population = Vec::new();
        let fitnesses: Vec<f64> = self.population.iter()
            .map(|chrom| self.fitness(chrom))
            .collect();
        let total_fitness: f64 = fitnesses.iter().sum();
        
        for _ in 0..self.population_size {
            let r = rand::random::<f64>() * total_fitness;
            let mut cumsum = 0.0;
            for (i, &fitness) in fitnesses.iter().enumerate() {
                cumsum += fitness;
                if cumsum >= r {
                    new_population.push(self.population[i].clone());
                    break;
                }
            }
        }
        
        new_population
    }
    
    pub fn crossover(&self, parent1: &[bool], parent2: &[bool]) -> (Vec<bool>, Vec<bool>) {
        if rand::random::<f64>() > self.crossover_rate {
            return (parent1.to_vec(), parent2.to_vec());
        }
        
        let crossover_point = rand::random::<usize>() % self.chromosome_length;
        let mut child1 = parent1[..crossover_point].to_vec();
        child1.extend_from_slice(&parent2[crossover_point..]);
        
        let mut child2 = parent2[..crossover_point].to_vec();
        child2.extend_from_slice(&parent1[crossover_point..]);
        
        (child1, child2)
    }
    
    pub fn mutation(&self, chromosome: &mut [bool]) {
        for gene in chromosome.iter_mut() {
            if rand::random::<f64>() < self.mutation_rate {
                *gene = !*gene;
            }
        }
    }
    
    pub fn evolve(&mut self, generations: usize) -> Vec<f64> {
        let mut best_fitnesses = Vec::new();
        
        for generation in 0..generations {
            // 选择
            let selected = self.selection();
            
            // 交叉和变异
            let mut new_population = Vec::new();
            for i in (0..self.population_size).step_by(2) {
                let (child1, child2) = self.crossover(&selected[i], &selected[i + 1]);
                let mut child1 = child1;
                let mut child2 = child2;
                
                self.mutation(&mut child1);
                self.mutation(&mut child2);
                
                new_population.push(child1);
                new_population.push(child2);
            }
            
            self.population = new_population;
            
            // 记录最佳适应度
            let best_fitness = self.population.iter()
                .map(|chrom| self.fitness(chrom))
                .fold(0.0, f64::max);
            best_fitnesses.push(best_fitness);
            
            if generation % 10 == 0 {
                println!("Generation {}: Best fitness = {}", generation, best_fitness);
            }
        }
        
        best_fitnesses
    }
}

#[derive(Debug)]
pub struct SimulatedAnnealing {
    pub initial_temperature: f64,
    pub final_temperature: f64,
    pub cooling_rate: f64,
    pub current_solution: Vec<f64>,
    pub best_solution: Vec<f64>,
    pub best_energy: f64,
}

impl SimulatedAnnealing {
    pub fn new(initial_temp: f64, final_temp: f64, cooling_rate: f64, solution: Vec<f64>) -> Self {
        let best_solution = solution.clone();
        let best_energy = Self::energy(&solution);
        
        Self {
            initial_temperature: initial_temp,
            final_temperature: final_temp,
            cooling_rate: cooling_rate,
            current_solution: solution,
            best_solution,
            best_energy,
        }
    }
    
    pub fn energy(solution: &[f64]) -> f64 {
        // 示例能量函数：Rastrigin函数
        let a = 10.0;
        let n = solution.len() as f64;
        let sum = solution.iter()
            .map(|&x| x * x - a * (2.0 * std::f64::consts::PI * x).cos())
            .sum::<f64>();
        a * n + sum
    }
    
    pub fn neighbor(&self, solution: &[f64]) -> Vec<f64> {
        let mut new_solution = solution.to_vec();
        let i = rand::random::<usize>() % solution.len();
        new_solution[i] += (rand::random::<f64>() - 0.5) * 0.1;
        new_solution
    }
    
    pub fn accept_probability(&self, current_energy: f64, new_energy: f64, temperature: f64) -> f64 {
        if new_energy < current_energy {
            1.0
        } else {
            ((current_energy - new_energy) / temperature).exp()
        }
    }
    
    pub fn optimize(&mut self, iterations: usize) -> (Vec<f64>, f64) {
        let mut temperature = self.initial_temperature;
        
        for iteration in 0..iterations {
            let current_energy = Self::energy(&self.current_solution);
            let new_solution = self.neighbor(&self.current_solution);
            let new_energy = Self::energy(&new_solution);
            
            if self.accept_probability(current_energy, new_energy, temperature) > rand::random::<f64>() {
                self.current_solution = new_solution;
                
                if new_energy < self.best_energy {
                    self.best_solution = self.current_solution.clone();
                    self.best_energy = new_energy;
                }
            }
            
            temperature *= self.cooling_rate;
            
            if iteration % 100 == 0 {
                println!("Iteration {}: Temperature = {:.6}, Best energy = {:.6}", 
                        iteration, temperature, self.best_energy);
            }
        }
        
        (self.best_solution.clone(), self.best_energy)
    }
}

// 使用示例
fn main() {
    // 线性规划示例
    let mut lp = LinearProgram::new(vec![3.0, 2.0]);
    lp.add_constraint(vec![1.0, 1.0], 4.0);
    lp.add_constraint(vec![2.0, 1.0], 5.0);
    lp.set_bounds(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]);
    
    if let Some(solution) = lp.solve_simplex() {
        println!("LP Solution: {:?}", solution);
    }
    
    // 遗传算法示例
    let mut ga = GeneticAlgorithm::new(50, 20);
    let fitness_history = ga.evolve(100);
    println!("Final best fitness: {}", fitness_history.last().unwrap());
    
    // 模拟退火示例
    let initial_solution = vec![0.0; 10];
    let mut sa = SimulatedAnnealing::new(100.0, 0.01, 0.95, initial_solution);
    let (best_solution, best_energy) = sa.optimize(1000);
    println!("SA Best solution: {:?}", best_solution);
    println!("SA Best energy: {}", best_energy);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module OptimizationModels where

import Data.List (minimumBy, maximumBy)
import Data.Ord (comparing)
import System.Random (randomRs, newStdGen)
import Control.Monad.State

-- 线性规划
data LinearProgram = LinearProgram {
    objective :: [Double],
    constraints :: [[Double]],
    rhs :: [Double],
    bounds :: [(Double, Double)]
} deriving Show

newLinearProgram :: [Double] -> LinearProgram
newLinearProgram obj = LinearProgram obj [] [] []

addConstraint :: [Double] -> Double -> LinearProgram -> LinearProgram
addConstraint constraint rhs_val lp = lp {
    constraints = constraint : constraints lp,
    rhs = rhs_val : rhs lp
}

setBounds :: [(Double, Double)] -> LinearProgram -> LinearProgram
setBounds bounds_list lp = lp { bounds = bounds_list }

solveSimplex :: LinearProgram -> Maybe [Double]
solveSimplex lp = 
    let n_vars = length (objective lp)
        n_constraints = length (constraints lp)
        tableau = constructTableau lp
    in solveTableau tableau n_vars n_constraints

constructTableau :: LinearProgram -> [[Double]]
constructTableau lp = 
    let n_vars = length (objective lp)
        n_constraints = length (constraints lp)
        objective_row = map negate (objective lp) ++ replicate n_constraints 0.0 ++ [0.0]
        constraint_rows = zipWith (\constraint rhs_val -> 
            constraint ++ replicate n_constraints 0.0 ++ [rhs_val]) 
            (constraints lp) (rhs lp)
    in objective_row : constraint_rows

solveTableau :: [[Double]] -> Int -> Int -> Maybe [Double]
solveTableau tableau n_vars n_constraints = 
    let iterations = iterate (simplexStep n_vars n_constraints) tableau
        converged = takeWhile (not . isOptimal) iterations
    in if null converged 
       then Nothing 
       else Just (extractSolution (last converged) n_vars n_constraints)

isOptimal :: [[Double]] -> Bool
isOptimal tableau = all (>= -1e-10) (head tableau)

simplexStep :: Int -> Int -> [[Double]] -> [[Double]]
simplexStep n_vars n_constraints tableau = 
    let entering_col = findEnteringColumn tableau
        leaving_row = findLeavingRow tableau entering_col
    in pivot tableau leaving_row entering_col

findEnteringColumn :: [[Double]] -> Int
findEnteringColumn tableau = 
    case findIndex (< -1e-10) (head tableau) of
        Just col -> col
        Nothing -> 0

findLeavingRow :: [[Double]] -> Int -> Int
findLeavingRow tableau col = 
    let ratios = zipWith (\i row -> 
        if row !! col > 1e-10 
        then Just (i, (row !! (length row - 1)) / (row !! col))
        else Nothing) [1..] (tail tableau)
        valid_ratios = catMaybes ratios
    in if null valid_ratios 
       then 0 
       else fst (minimumBy (comparing snd) valid_ratios)

pivot :: [[Double]] -> Int -> Int -> [[Double]]
pivot tableau row col = 
    let pivot_val = tableau !! row !! col
        normalized_row = map (/ pivot_val) (tableau !! row)
        new_tableau = map (\i -> 
            if i == row 
            then normalized_row 
            else map (\j -> 
                let factor = tableau !! i !! col
                in (tableau !! i !! j) - factor * (normalized_row !! j)) 
                [0..length (head tableau) - 1]) 
            [0..length tableau - 1]
    in new_tableau

extractSolution :: [[Double]] -> Int -> Int -> [Double]
extractSolution tableau n_vars n_constraints = 
    let solution = replicate n_vars 0.0
        basic_vars = findBasicVariables tableau
    in foldl (\sol (var, val) -> 
        if var < n_vars 
        then take var sol ++ [val] ++ drop (var + 1) sol
        else sol) solution basic_vars

findBasicVariables :: [[Double]] -> [(Int, Double)]
findBasicVariables tableau = 
    concatMap (\i -> 
        let row = tableau !! i
            basic_col = findIndex (\x -> abs (x - 1.0) < 1e-10) row
        in case basic_col of
            Just col -> [(col, row !! (length row - 1))]
            Nothing -> []) [1..length tableau - 1]

-- 遗传算法
data GeneticAlgorithm = GeneticAlgorithm {
    populationSize :: Int,
    chromosomeLength :: Int,
    mutationRate :: Double,
    crossoverRate :: Double,
    population :: [[Bool]]
} deriving Show

newGeneticAlgorithm :: Int -> Int -> IO GeneticAlgorithm
newGeneticAlgorithm pop_size chrom_length = do
    gen <- newStdGen
    let population = take pop_size $ iterate (randomChromosome chrom_length) []
    return GeneticAlgorithm {
        populationSize = pop_size,
        chromosomeLength = chrom_length,
        mutationRate = 0.01,
        crossoverRate = 0.8,
        population = population
    }
  where
    randomChromosome len = take len $ randomRs (False, True) gen

fitness :: [Bool] -> Double
fitness chromosome = fromIntegral $ length $ filter id chromosome

selection :: GeneticAlgorithm -> IO [[Bool]]
selection ga = do
    gen <- newStdGen
    let fitnesses = map fitness (population ga)
        total_fitness = sum fitnesses
        roulette = map (\f -> f / total_fitness) fitnesses
    return $ take (populationSize ga) $ iterate (selectIndividual roulette) []

selectIndividual :: [Double] -> [Bool]
selectIndividual roulette = 
    let r = head $ randomRs (0.0, 1.0) gen
        cumsum = scanl1 (+) roulette
        selected = length $ takeWhile (< r) cumsum
    in population ga !! selected

crossover :: GeneticAlgorithm -> [Bool] -> [Bool] -> IO ([Bool], [Bool])
crossover ga parent1 parent2 = do
    gen <- newStdGen
    let r = head $ randomRs (0.0, 1.0) gen
    if r > crossoverRate ga
    then return (parent1, parent2)
    else do
        let crossover_point = head $ randomRs (0, chromosomeLength ga) gen
            child1 = take crossover_point parent1 ++ drop crossover_point parent2
            child2 = take crossover_point parent2 ++ drop crossover_point parent1
        return (child1, child2)

mutation :: GeneticAlgorithm -> [Bool] -> IO [Bool]
mutation ga chromosome = do
    gen <- newStdGen
    let mutation_rates = randomRs (0.0, 1.0) gen
        mutated = zipWith (\gene rate -> 
            if rate < mutationRate ga then not gene else gene) 
            chromosome mutation_rates
    return mutated

evolve :: GeneticAlgorithm -> Int -> IO ([Double], GeneticAlgorithm)
evolve ga generations = go ga generations []
  where
    go current_ga 0 history = return (reverse history, current_ga)
    go current_ga gen history = do
        -- 选择
        selected <- selection current_ga
        
        -- 交叉和变异
        new_population <- crossoverAndMutate current_ga selected
        
        let new_ga = current_ga { population = new_population }
            best_fitness = maximum $ map fitness new_population
        
        go new_ga (gen - 1) (best_fitness : history)

crossoverAndMutate :: GeneticAlgorithm -> [[Bool]] -> IO [[Bool]]
crossoverAndMutate ga selected = do
    let pairs = zip (take (populationSize ga `div` 2) selected) 
                   (drop (populationSize ga `div` 2) selected)
    children <- mapM (\(p1, p2) -> do
        (c1, c2) <- crossover ga p1 p2
        mc1 <- mutation ga c1
        mc2 <- mutation ga c2
        return [mc1, mc2]) pairs
    return $ concat children

-- 模拟退火
data SimulatedAnnealing = SimulatedAnnealing {
    initialTemperature :: Double,
    finalTemperature :: Double,
    coolingRate :: Double,
    currentSolution :: [Double],
    bestSolution :: [Double],
    bestEnergy :: Double
} deriving Show

newSimulatedAnnealing :: Double -> Double -> Double -> [Double] -> SimulatedAnnealing
newSimulatedAnnealing init_temp final_temp cooling_rate solution = 
    SimulatedAnnealing {
        initialTemperature = init_temp,
        finalTemperature = final_temp,
        coolingRate = cooling_rate,
        currentSolution = solution,
        bestSolution = solution,
        bestEnergy = energy solution
    }

energy :: [Double] -> Double
energy solution = 
    let a = 10.0
        n = fromIntegral $ length solution
        sum_terms = sum $ map (\x -> x * x - a * cos (2 * pi * x)) solution
    in a * n + sum_terms

neighbor :: [Double] -> IO [Double]
neighbor solution = do
    gen <- newStdGen
    let i = head $ randomRs (0, length solution - 1) gen
        delta = head $ randomRs (-0.1, 0.1) gen
    return $ take i solution ++ [solution !! i + delta] ++ drop (i + 1) solution

acceptProbability :: Double -> Double -> Double -> Double
acceptProbability current_energy new_energy temperature = 
    if new_energy < current_energy 
    then 1.0 
    else exp ((current_energy - new_energy) / temperature)

optimize :: SimulatedAnnealing -> Int -> IO ([Double], Double)
optimize sa iterations = go sa iterations (initialTemperature sa)
  where
    go current_sa 0 temp = return (bestSolution current_sa, bestEnergy current_sa)
    go current_sa iter temp = do
        let current_energy = energy (currentSolution current_sa)
        new_solution <- neighbor (currentSolution current_sa)
        let new_energy = energy new_solution
        
        gen <- newStdGen
        let r = head $ randomRs (0.0, 1.0) gen
            accept = acceptProbability current_energy new_energy temp > r
        
        let updated_sa = if accept 
                         then current_sa { currentSolution = new_solution }
                         else current_sa
        
        let final_sa = if new_energy < bestEnergy updated_sa
                       then updated_sa { 
                           bestSolution = new_solution, 
                           bestEnergy = new_energy 
                       }
                       else updated_sa
        
        let new_temp = temp * coolingRate current_sa
        go final_sa (iter - 1) new_temp

-- 示例使用
example :: IO ()
example = do
    -- 线性规划示例
    let lp = setBounds [(0.0, infinity), (0.0, infinity)] $
             addConstraint [2.0, 1.0] 5.0 $
             addConstraint [1.0, 1.0] 4.0 $
             newLinearProgram [3.0, 2.0]
    
    case solveSimplex lp of
        Just solution -> putStrLn $ "LP Solution: " ++ show solution
        Nothing -> putStrLn "No solution found"
    
    -- 遗传算法示例
    ga <- newGeneticAlgorithm 50 20
    (fitness_history, final_ga) <- evolve ga 100
    putStrLn $ "Final best fitness: " ++ show (last fitness_history)
    
    -- 模拟退火示例
    let initial_solution = replicate 10 0.0
        sa = newSimulatedAnnealing 100.0 0.01 0.95 initial_solution
    (best_solution, best_energy) <- optimize sa 1000
    putStrLn $ "SA Best solution: " ++ show best_solution
    putStrLn $ "SA Best energy: " ++ show best_energy
  where
    infinity = 1e10
```

### 应用领域 / Application Domains

#### 运筹学 / Operations Research

- **生产计划**: 资源分配、调度优化
- **物流优化**: 路径规划、库存管理
- **金融优化**: 投资组合、风险管理

#### 工程设计 / Engineering Design

- **结构优化**: 重量最小化、强度最大化
- **参数优化**: 性能调优、成本控制
- **多目标优化**: 权衡分析、帕累托前沿

#### 人工智能 / Artificial Intelligence

- **机器学习**: 超参数优化、特征选择
- **神经网络**: 权重优化、架构搜索
- **强化学习**: 策略优化、价值函数

---

## 参考文献 / References

1. Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
2. Bertsekas, D. P. (2016). Nonlinear Programming. Athena Scientific.
3. Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley.
4. Kirkpatrick, S., et al. (1983). Optimization by Simulated Annealing. Science.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
