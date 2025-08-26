# 4.5 人工智能模型 / AI Models

## 目录 / Table of Contents

- [4.5 人工智能模型 / AI Models](#45-人工智能模型--ai-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [4.5.1 机器学习基础模型 / Basic Machine Learning Models](#451-机器学习基础模型--basic-machine-learning-models)
    - [线性回归模型 / Linear Regression Model](#线性回归模型--linear-regression-model)
    - [逻辑回归模型 / Logistic Regression Model](#逻辑回归模型--logistic-regression-model)
    - [支持向量机 (SVM) / Support Vector Machine](#支持向量机-svm--support-vector-machine)
    - [决策树模型 / Decision Tree Model](#决策树模型--decision-tree-model)
  - [4.5.2 深度学习模型 / Deep Learning Models](#452-深度学习模型--deep-learning-models)
    - [多层感知机 (MLP) / Multi-Layer Perceptron](#多层感知机-mlp--multi-layer-perceptron)
    - [卷积神经网络 (CNN) / Convolutional Neural Network](#卷积神经网络-cnn--convolutional-neural-network)
    - [循环神经网络 (RNN) / Recurrent Neural Network](#循环神经网络-rnn--recurrent-neural-network)
  - [4.5.3 强化学习模型 / Reinforcement Learning Models](#453-强化学习模型--reinforcement-learning-models)
    - [Q-Learning模型 / Q-Learning Model](#q-learning模型--q-learning-model)
    - [策略梯度模型 / Policy Gradient Model](#策略梯度模型--policy-gradient-model)
    - [Actor-Critic模型 / Actor-Critic Model](#actor-critic模型--actor-critic-model)
  - [4.5.4 自然语言处理模型 / Natural Language Processing Models](#454-自然语言处理模型--natural-language-processing-models)
    - [Word2Vec模型 / Word2Vec Model](#word2vec模型--word2vec-model)
    - [Transformer模型 / Transformer Model](#transformer模型--transformer-model)
    - [BERT模型 / BERT Model](#bert模型--bert-model)
  - [4.5.5 计算机视觉模型 / Computer Vision Models](#455-计算机视觉模型--computer-vision-models)
    - [残差网络 (ResNet) / Residual Network](#残差网络-resnet--residual-network)
    - [注意力机制 / Attention Mechanism](#注意力机制--attention-mechanism)
    - [目标检测模型 / Object Detection Models](#目标检测模型--object-detection-models)
  - [4.5.6 生成对抗网络模型 / Generative Adversarial Network Models](#456-生成对抗网络模型--generative-adversarial-network-models)
    - [GAN基础模型 / Basic GAN Model](#gan基础模型--basic-gan-model)
    - [Wasserstein GAN / Wasserstein GAN](#wasserstein-gan--wasserstein-gan)
    - [条件GAN / Conditional GAN](#条件gan--conditional-gan)
  - [4.5.7 图神经网络模型 / Graph Neural Network Models](#457-图神经网络模型--graph-neural-network-models)
    - [图卷积网络 (GCN) / Graph Convolutional Network](#图卷积网络-gcn--graph-convolutional-network)
    - [图注意力网络 (GAT) / Graph Attention Network](#图注意力网络-gat--graph-attention-network)
    - [图池化 / Graph Pooling](#图池化--graph-pooling)
  - [4.5.8 实现与应用 / Implementation and Applications](#458-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [计算机视觉 / Computer Vision](#计算机视觉--computer-vision)
      - [自然语言处理 / Natural Language Processing](#自然语言处理--natural-language-processing)
      - [推荐系统 / Recommendation Systems](#推荐系统--recommendation-systems)
  - [参考文献 / References](#参考文献--references)

---

## 4.5.1 机器学习基础模型 / Basic Machine Learning Models

### 线性回归模型 / Linear Regression Model

**模型**: $y = \mathbf{w}^T \mathbf{x} + b$

**损失函数**: $L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$

**梯度下降更新**:
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

### 逻辑回归模型 / Logistic Regression Model

**模型**: $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$

**sigmoid函数**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

**损失函数**: $L(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$

### 支持向量机 (SVM) / Support Vector Machine

**目标函数**:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

**约束条件**:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**核函数**: $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$

### 决策树模型 / Decision Tree Model

**信息增益**: $IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$

**基尼指数**: $Gini(D) = 1 - \sum_{i=1}^k p_i^2$

**剪枝**: 最小化 $C(T) = \sum_{t \in T} N_t H_t(T) + \alpha |T|$

---

## 4.5.2 深度学习模型 / Deep Learning Models

### 多层感知机 (MLP) / Multi-Layer Perceptron

**前向传播**:
$$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$$

**反向传播**:
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

**权重更新**:
$$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$

### 卷积神经网络 (CNN) / Convolutional Neural Network

**卷积层**:
$$(f * k)(i, j) = \sum_{m} \sum_{n} f(m, n) k(i-m, j-n)$$

**池化层**:
$$y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$$

**批量归一化**:
$$BN(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 循环神经网络 (RNN) / Recurrent Neural Network

**隐藏状态**:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

**输出**:
$$y_t = W_{hy} h_t + b_y$$

**LSTM门控机制**:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

---

## 4.5.3 强化学习模型 / Reinforcement Learning Models

### Q-Learning模型 / Q-Learning Model

**Q值更新**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**策略**: $\pi(s) = \arg\max_a Q(s, a)$

**ε-贪婪策略**: $\pi(s) = \begin{cases} \arg\max_a Q(s, a) & \text{w.p. } 1-\epsilon \\ \text{random action} & \text{w.p. } \epsilon \end{cases}$

### 策略梯度模型 / Policy Gradient Model

**目标函数**: $J(\theta) = \mathbb{E}_{\pi_\theta} [R(\tau)]$

**梯度**: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) R(\tau)]$

**REINFORCE算法**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) R_t$$

### Actor-Critic模型 / Actor-Critic Model

**Actor网络**: $\pi_\theta(a|s)$

**Critic网络**: $V_\phi(s)$

**优势函数**: $A(s, a) = Q(s, a) - V(s)$

**更新规则**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) A(s, a)$$
$$\phi \leftarrow \phi + \beta \nabla_\phi (R - V_\phi(s))^2$$

---

## 4.5.4 自然语言处理模型 / Natural Language Processing Models

### Word2Vec模型 / Word2Vec Model

**Skip-gram目标**:
$$J = -\frac{1}{T} \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)$$

**负采样**:
$$P(w_O|w_I) = \frac{\exp(v_{w_O}'^T v_{w_I})}{\sum_{w=1}^W \exp(v_w'^T v_{w_I})}$$

### Transformer模型 / Transformer Model

**多头注意力**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \ldots, head_h)W^O$$

**自注意力**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**位置编码**:
$$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

### BERT模型 / BERT Model

**掩码语言模型**: $P(w_i|w_1, \ldots, w_{i-1}, [MASK], w_{i+1}, \ldots, w_n)$

**下一句预测**: $P(\text{IsNext}|A, B)$

**预训练目标**:
$$L = L_{MLM} + L_{NSP}$$

---

## 4.5.5 计算机视觉模型 / Computer Vision Models

### 残差网络 (ResNet) / Residual Network

**残差连接**: $F(x) = H(x) - x$

**前向传播**: $H(x) = F(x) + x$

**梯度流动**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial L}{\partial H} \cdot (1 + \frac{\partial F}{\partial x})$

### 注意力机制 / Attention Mechanism

**查询-键-值**: $Q, K, V$

**注意力权重**: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$

**注意力输出**: $c_i = \sum_j \alpha_{ij} v_j$

### 目标检测模型 / Object Detection Models

**YOLO算法**: $P(\text{object}) \times \text{IoU}(pred, truth)$

**边界框回归**: $b_x, b_y, b_w, b_h$

**非极大值抑制**: 移除重叠检测框

---

## 4.5.6 生成对抗网络模型 / Generative Adversarial Network Models

### GAN基础模型 / Basic GAN Model

**生成器**: $G: \mathcal{Z} \to \mathcal{X}$

**判别器**: $D: \mathcal{X} \to [0, 1]$

**目标函数**:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1-D(G(z)))]$$

### Wasserstein GAN / Wasserstein GAN

**Wasserstein距离**: $W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x-y\|]$

**目标函数**:
$$\min_G \max_D \mathbb{E}_{x \sim p_r} [D(x)] - \mathbb{E}_{\tilde{x} \sim p_g} [D(\tilde{x})]$$

**梯度惩罚**: $\lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} [(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$

### 条件GAN / Conditional GAN

**生成器**: $G: \mathcal{Z} \times \mathcal{Y} \to \mathcal{X}$

**判别器**: $D: \mathcal{X} \times \mathcal{Y} \to [0, 1]$

**目标函数**:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x,y \sim p_{data}(x,y)} [\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)} [\log(1-D(G(z, y), y))]$$

---

## 4.5.7 图神经网络模型 / Graph Neural Network Models

### 图卷积网络 (GCN) / Graph Convolutional Network

**图卷积层**:
$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$

其中：

- $\tilde{A} = A + I$ (添加自环)
- $\tilde{D}$ 是 $\tilde{A}$ 的度矩阵

### 图注意力网络 (GAT) / Graph Attention Network

**注意力系数**:
$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$$

**聚合**:
$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} Wh_j\right)$$

### 图池化 / Graph Pooling

**DiffPool**: 学习聚类分配矩阵 $S^{(l)} \in \mathbb{R}^{n_l \times n_{l+1}}$

**图粗化**: $A^{(l+1)} = S^{(l)^T} A^{(l)} S^{(l)}$

**特征聚合**: $X^{(l+1)} = S^{(l)^T} X^{(l)}$

---

## 4.5.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug)]
pub struct LinearRegression {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl LinearRegression {
    pub fn new(input_size: usize, learning_rate: f64) -> Self {
        Self {
            weights: Array1::random(input_size, Uniform::new(-0.1, 0.1)),
            bias: 0.0,
            learning_rate,
        }
    }
    
    pub fn predict(&self, x: &ArrayView1<f64>) -> f64 {
        x.dot(&self.weights) + self.bias
    }
    
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize) {
        let n_samples = x.nrows();
        
        for _ in 0..epochs {
            let mut weight_gradients = Array1::zeros(self.weights.len());
            let mut bias_gradient = 0.0;
            
            for i in 0..n_samples {
                let prediction = self.predict(&x.row(i));
                let error = prediction - y[i];
                
                // 计算梯度
                for j in 0..self.weights.len() {
                    weight_gradients[j] += error * x[[i, j]];
                }
                bias_gradient += error;
            }
            
            // 更新参数
            for j in 0..self.weights.len() {
                self.weights[j] -= self.learning_rate * weight_gradients[j] / n_samples as f64;
            }
            self.bias -= self.learning_rate * bias_gradient / n_samples as f64;
        }
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<usize>,
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, learning_rate: f64) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        for i in 0..layers.len() - 1 {
            weights.push(Array2::random((layers[i + 1], layers[i]), Uniform::new(-0.1, 0.1)));
            biases.push(Array1::random(layers[i + 1], Uniform::new(-0.1, 0.1)));
        }
        
        Self {
            layers,
            weights,
            biases,
            learning_rate,
        }
    }
    
    pub fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    pub fn sigmoid_derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }
    
    pub fn forward(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut activations = vec![input.clone()];
        
        for i in 0..self.weights.len() {
            let z = self.weights[i].dot(&activations[i]) + &self.biases[i];
            let activation = z.mapv(|x| self.sigmoid(x));
            activations.push(activation);
        }
        
        activations
    }
    
    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: usize) {
        let n_samples = x.nrows();
        
        for _ in 0..epochs {
            for i in 0..n_samples {
                // 前向传播
                let input = x.row(i).to_owned();
                let target = y.row(i).to_owned();
                let activations = self.forward(&input);
                
                // 反向传播
                let mut deltas = vec![Array1::zeros(self.layers[self.layers.len() - 1])];
                
                // 输出层误差
                let output_error = &activations[activations.len() - 1] - &target;
                let output_delta = output_error * &activations[activations.len() - 1].mapv(|x| self.sigmoid_derivative(x));
                deltas[0] = output_delta;
                
                // 隐藏层误差
                for layer in 1..self.weights.len() {
                    let layer_idx = self.weights.len() - layer - 1;
                    let error = self.weights[layer_idx + 1].t().dot(&deltas[layer - 1]);
                    let delta = error * &activations[layer_idx].mapv(|x| self.sigmoid_derivative(x));
                    deltas.push(delta);
                }
                deltas.reverse();
                
                // 更新权重和偏置
                for layer in 0..self.weights.len() {
                    let weight_gradient = deltas[layer].outer(&activations[layer]);
                    self.weights[layer] -= &(self.learning_rate * weight_gradient);
                    self.biases[layer] -= &(self.learning_rate * &deltas[layer]);
                }
            }
        }
    }
}

// 使用示例
fn main() {
    // 线性回归示例
    let mut lr = LinearRegression::new(2, 0.01);
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
    let y = Array1::from_shape_vec(100, (0..100).map(|x| 2.0 * x as f64 + 1.0).collect()).unwrap();
    
    lr.train(&x, &y, 1000);
    println!("Linear Regression weights: {:?}", lr.weights);
    println!("Linear Regression bias: {:?}", lr.bias);
    
    // 神经网络示例
    let mut nn = NeuralNetwork::new(vec![2, 3, 1], 0.1);
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    
    nn.train(&x, &y, 10000);
    println!("Neural Network trained!");
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module AIModels where

import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.List (sum, length)
import System.Random (randomRs, newStdGen)

-- 线性回归模型
data LinearRegression = LinearRegression {
    weights :: Vector Double,
    bias :: Double,
    learningRate :: Double
} deriving (Show)

-- 创建线性回归模型
newLinearRegression :: Int -> Double -> LinearRegression
newLinearRegression inputSize lr = LinearRegression {
    weights = V.replicate inputSize 0.0,
    bias = 0.0,
    learningRate = lr
}

-- 预测
predict :: LinearRegression -> Vector Double -> Double
predict model x = V.sum (V.zipWith (*) (weights model) x) + bias model

-- 训练
train :: LinearRegression -> [Vector Double] -> [Double] -> Int -> LinearRegression
train model xs ys epochs = foldl trainEpoch model (replicate epochs ())
  where
    trainEpoch model _ = foldl (updateWeights xs ys) model [0..length xs - 1]
    
    updateWeights xs ys model i = model {
        weights = V.zipWith (\w x -> w - learningRate model * gradient w x) 
                            (weights model) (xs !! i),
        bias = bias model - learningRate model * biasGradient
    }
      where
        prediction = predict model (xs !! i)
        error = prediction - (ys !! i)
        gradient w x = error * x
        biasGradient = error

-- 神经网络
data NeuralNetwork = NeuralNetwork {
    layers :: [Int],
    weights :: [Vector (Vector Double)],
    biases :: [Vector Double],
    learningRate :: Double
} deriving (Show)

-- sigmoid函数
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

-- sigmoid导数
sigmoidDerivative :: Double -> Double
sigmoidDerivative x = x * (1.0 - x)

-- 前向传播
forward :: NeuralNetwork -> Vector Double -> [Vector Double]
forward nn input = scanl (\acc (w, b) -> V.map sigmoid (V.zipWith (+) (V.map (V.sum . V.zipWith (*) acc) w) b)) 
                        [input] (zip (weights nn) (biases nn))

-- 训练神经网络
trainNN :: NeuralNetwork -> [Vector Double] -> [Vector Double] -> Int -> NeuralNetwork
trainNN nn xs ys epochs = foldl trainEpoch nn (replicate epochs ())
  where
    trainEpoch nn _ = foldl (updateWeights xs ys) nn [0..length xs - 1]

-- 示例使用
example :: IO ()
example = do
    -- 线性回归示例
    let lr = newLinearRegression 2 0.01
        xs = [V.fromList [1.0, 2.0], V.fromList [2.0, 3.0], V.fromList [3.0, 4.0]]
        ys = [5.0, 8.0, 11.0]
        trainedLR = train lr xs ys 1000
    
    putStrLn $ "Trained Linear Regression: " ++ show trainedLR
    
    -- 测试预测
    let testInput = V.fromList [4.0, 5.0]
        prediction = predict trainedLR testInput
    putStrLn $ "Prediction for [4.0, 5.0]: " ++ show prediction
```

### 应用领域 / Application Domains

#### 计算机视觉 / Computer Vision

- **图像分类**: 识别图像中的物体
- **目标检测**: 定位图像中的物体
- **图像分割**: 像素级别的分类

#### 自然语言处理 / Natural Language Processing

- **机器翻译**: 语言间的自动翻译
- **文本生成**: 自动生成文本内容
- **情感分析**: 分析文本的情感倾向

#### 推荐系统 / Recommendation Systems

- **协同过滤**: 基于用户行为的推荐
- **内容过滤**: 基于物品特征的推荐
- **深度学习推荐**: 端到端的推荐模型

---

## 4.5.9 算法实现 / Algorithm Implementation

### 机器学习基础算法 / Basic Machine Learning Algorithms

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import random

class LinearRegression:
    """线性回归模型"""
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.cost_history: List[float] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """训练线性回归模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算损失
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 早停
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-6:
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.weights is None:
            raise ValueError("Model not fitted yet")
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class LogisticRegression:
    """逻辑回归模型"""
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.cost_history: List[float] = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """训练逻辑回归模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # 计算损失（交叉熵）
            epsilon = 1e-15
            cost = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 早停
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-6:
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.weights is None:
            raise ValueError("Model not fitted yet")
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        return (self.predict(X) >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return accuracy_score(y, self.predict_classes(X))

class SupportVectorMachine:
    """支持向量机（简化版）"""
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, C: float = 1.0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.C = C
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorMachine':
        """训练SVM模型"""
        n_samples, n_features = X.shape
        
        # 将标签转换为-1和1
        y = np.where(y == 0, -1, 1)
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            for i in range(n_samples):
                # 计算决策函数
                decision = y[i] * (np.dot(X[i], self.weights) + self.bias)
                
                # 更新规则
                if decision < 1:
                    self.weights += self.learning_rate * (self.C * y[i] * X[i] - 2 * (1/self.max_iterations) * self.weights)
                    self.bias += self.learning_rate * self.C * y[i]
                else:
                    self.weights += self.learning_rate * (-2 * (1/self.max_iterations) * self.weights)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.weights is None:
            raise ValueError("Model not fitted yet")
        return np.sign(np.dot(X, self.weights) + self.bias)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        y = np.where(y == 0, -1, 1)
        return accuracy_score(y, self.predict(X))

### 深度学习算法 / Deep Learning Algorithms

class NeuralNetwork:
    """多层感知机"""
    def __init__(self, layers: List[int], learning_rate: float = 0.01, max_iterations: int = 1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.cost_history: List[float] = []
        
        # 初始化权重和偏置
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * 0.01
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """sigmoid导数"""
        return z * (1 - z)
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """前向传播"""
        activations = [X.T]  # 输入层
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """反向传播"""
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        # 计算输出层误差
        delta = activations[-1] - y.T
        
        # 计算梯度
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            weight_gradients.insert(0, np.dot(delta, activations[i].T) / m)
            bias_gradients.insert(0, np.sum(delta, axis=1, keepdims=True) / m)
            
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(activations[i])
        
        return weight_gradients, bias_gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetwork':
        """训练神经网络"""
        for iteration in range(self.max_iterations):
            # 前向传播
            activations, z_values = self.forward_propagation(X)
            
            # 计算损失
            cost = -np.mean(y * np.log(activations[-1].T + 1e-15) + (1 - y) * np.log(1 - activations[-1].T + 1e-15))
            self.cost_history.append(cost)
            
            # 反向传播
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations, z_values)
            
            # 更新参数
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]
            
            # 早停
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-6:
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        activations, _ = self.forward_propagation(X)
        return activations[-1].T
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        return (self.predict(X) >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return accuracy_score(y, self.predict_classes(X))

class ConvolutionalNeuralNetwork:
    """简化的卷积神经网络"""
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float = 0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.filters = []
        self.weights = None
        self.bias = None
        
        # 初始化卷积核
        self.filters = [np.random.randn(3, 3) * 0.1 for _ in range(6)]
        
        # 计算全连接层的输入大小
        conv_output_size = (input_shape[0] - 2) * (input_shape[1] - 2) * len(self.filters)
        self.weights = np.random.randn(num_classes, conv_output_size) * 0.1
        self.bias = np.zeros((num_classes, 1))
    
    def convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """卷积操作"""
        h, w = image.shape
        kh, kw = kernel.shape
        output_h = h - kh + 1
        output_w = w - kw + 1
        
        output = np.zeros((output_h, output_w))
        for i in range(output_h):
            for j in range(output_w):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
        
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """前向传播"""
        batch_size = X.shape[0]
        conv_outputs = []
        
        for i in range(batch_size):
            image = X[i, :, :, 0]  # 假设是灰度图像
            conv_output = []
            
            for kernel in self.filters:
                conv_result = self.convolve(image, kernel)
                conv_result = self.relu(conv_result)
                conv_output.append(conv_result)
            
            conv_outputs.append(np.array(conv_output))
        
        # 展平
        flattened = np.array([output.flatten() for output in conv_outputs])
        
        # 全连接层
        logits = np.dot(self.weights, flattened.T) + self.bias
        probabilities = self.softmax(logits)
        
        return probabilities.T, conv_outputs
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> 'ConvolutionalNeuralNetwork':
        """训练CNN"""
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # 前向传播
                probabilities, conv_outputs = self.forward(X[i:i+1])
                
                # 计算损失（交叉熵）
                target = np.zeros((1, self.num_classes))
                target[0, y[i]] = 1
                loss = -np.sum(target * np.log(probabilities + 1e-15))
                
                # 简化的反向传播（这里只是示例）
                # 实际实现需要更复杂的梯度计算
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}, Sample {i}, Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        probabilities, _ = self.forward(X)
        return np.argmax(probabilities, axis=1)

### 强化学习算法 / Reinforcement Learning Algorithms

class QLearning:
    """Q-Learning算法"""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state: int) -> int:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """更新Q值"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value
    
    def get_policy(self) -> np.ndarray:
        """获取最优策略"""
        return np.argmax(self.q_table, axis=1)

class PolicyGradient:
    """策略梯度算法"""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.policy = np.random.rand(state_size, action_size)
        self.policy = self.policy / np.sum(self.policy, axis=1, keepdims=True)
    
    def choose_action(self, state: int) -> int:
        """根据策略选择动作"""
        return np.random.choice(self.action_size, p=self.policy[state])
    
    def update_policy(self, states: List[int], actions: List[int], rewards: List[float]) -> None:
        """更新策略"""
        # 计算折扣奖励
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards):
            running_reward = reward + 0.95 * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # 标准化奖励
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # 更新策略
        for state, action, reward in zip(states, actions, discounted_rewards):
            # 计算策略梯度
            action_probs = self.policy[state]
            action_prob = action_probs[action]
            
            # 更新策略参数
            self.policy[state, action] += self.learning_rate * reward * (1 - action_prob)
            
            # 重新归一化
            self.policy[state] = self.policy[state] / np.sum(self.policy[state])

### 自然语言处理算法 / Natural Language Processing Algorithms

class Word2Vec:
    """简化的Word2Vec实现"""
    def __init__(self, vocab_size: int, embedding_dim: int, learning_rate: float = 0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # 初始化词向量
        self.word_vectors = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_vectors = np.random.randn(vocab_size, embedding_dim) * 0.1
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train_on_batch(self, center_word: int, context_words: List[int], negative_words: List[int]) -> float:
        """训练一个批次"""
        loss = 0
        
        # 正样本
        for context_word in context_words:
            # 前向传播
            center_vec = self.word_vectors[center_word]
            context_vec = self.context_vectors[context_word]
            score = np.dot(center_vec, context_vec)
            prob = self.sigmoid(score)
            
            # 计算损失
            loss -= np.log(prob + 1e-15)
            
            # 反向传播
            grad = prob - 1
            self.word_vectors[center_word] -= self.learning_rate * grad * context_vec
            self.context_vectors[context_word] -= self.learning_rate * grad * center_vec
        
        # 负样本
        for negative_word in negative_words:
            center_vec = self.word_vectors[center_word]
            negative_vec = self.context_vectors[negative_word]
            score = np.dot(center_vec, negative_vec)
            prob = self.sigmoid(score)
            
            # 计算损失
            loss -= np.log(1 - prob + 1e-15)
            
            # 反向传播
            grad = prob
            self.word_vectors[center_word] -= self.learning_rate * grad * negative_vec
            self.context_vectors[negative_word] -= self.learning_rate * grad * center_vec
        
        return loss
    
    def get_word_vector(self, word_id: int) -> np.ndarray:
        """获取词向量"""
        return self.word_vectors[word_id]
    
    def find_similar_words(self, word_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """找到相似词"""
        word_vec = self.word_vectors[word_id]
        similarities = []
        
        for i in range(self.vocab_size):
            if i != word_id:
                other_vec = self.word_vectors[i]
                similarity = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

### 人工智能模型验证函数 / AI Models Verification Functions

def ai_models_verification():
    """人工智能模型综合验证"""
    print("=== 人工智能模型验证 ===\n")
    
    # 1. 线性回归验证
    print("1. 线性回归验证:")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(100) * 0.1
    
    lr = LinearRegression(learning_rate=0.01, max_iterations=1000)
    lr.fit(X, y)
    
    score = lr.score(X, y)
    print(f"   线性回归R²分数: {score:.4f}")
    print(f"   权重: {lr.weights}")
    print(f"   偏置: {lr.bias:.4f}")
    
    # 2. 逻辑回归验证
    print("\n2. 逻辑回归验证:")
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X, y)
    
    score = log_reg.score(X, y)
    print(f"   逻辑回归准确率: {score:.4f}")
    
    # 3. 神经网络验证
    print("\n3. 神经网络验证:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR问题
    
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.1, max_iterations=10000)
    nn.fit(X, y)
    
    predictions = nn.predict_classes(X)
    score = nn.score(X, y)
    print(f"   神经网络准确率: {score:.4f}")
    print(f"   预测结果: {predictions}")
    
    # 4. Q-Learning验证
    print("\n4. Q-Learning验证:")
    # 简单的网格世界环境
    q_learning = QLearning(state_size=4, action_size=4)
    
    # 模拟一些训练数据
    for episode in range(100):
        state = 0
        for step in range(10):
            action = q_learning.choose_action(state)
            next_state = min(state + 1, 3)  # 简单的环境
            reward = 1 if next_state == 3 else 0
            q_learning.update(state, action, reward, next_state)
            state = next_state
            if state == 3:
                break
    
    policy = q_learning.get_policy()
    print(f"   最优策略: {policy}")
    
    # 5. Word2Vec验证
    print("\n5. Word2Vec验证:")
    vocab_size = 100
    embedding_dim = 10
    word2vec = Word2Vec(vocab_size, embedding_dim)
    
    # 模拟训练
    for _ in range(100):
        center_word = random.randint(0, vocab_size - 1)
        context_words = [random.randint(0, vocab_size - 1) for _ in range(2)]
        negative_words = [random.randint(0, vocab_size - 1) for _ in range(3)]
        loss = word2vec.train_on_batch(center_word, context_words, negative_words)
    
    word_vector = word2vec.get_word_vector(0)
    print(f"   词向量维度: {word_vector.shape}")
    print(f"   词向量范数: {np.linalg.norm(word_vector):.4f}")
    
    print("\n=== 所有人工智能模型验证完成 ===")

if __name__ == "__main__":
    ai_models_verification()
```

## 参考文献 / References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
3. Vaswani, A., et al. (2017). Attention is All You Need. NIPS.
4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. Nature.

---

*最后更新: 2025-08-26*
*版本: 1.1.0*
