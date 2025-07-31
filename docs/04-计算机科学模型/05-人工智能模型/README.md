# 4.5 人工智能模型 / AI Models

## 目录 / Table of Contents

1. [4.5.1 机器学习基础模型](#451-机器学习基础模型)
2. [4.5.2 深度学习模型](#452-深度学习模型)
3. [4.5.3 强化学习模型](#453-强化学习模型)
4. [4.5.4 自然语言处理模型](#454-自然语言处理模型)
5. [4.5.5 计算机视觉模型](#455-计算机视觉模型)
6. [4.5.6 生成对抗网络模型](#456-生成对抗网络模型)
7. [4.5.7 图神经网络模型](#457-图神经网络模型)
8. [4.5.8 实现与应用](#458-实现与应用)

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

## 参考文献 / References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
3. Vaswani, A., et al. (2017). Attention is All You Need. NIPS.
4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. Nature.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
