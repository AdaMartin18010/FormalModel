# 强化学习多表征示例 / Reinforcement Learning Multi-Representation Example

## 概述 / Overview

本文档提供强化学习模型的多表征实现示例，包括Q-Learning、策略梯度和深度Q网络。

This document provides multi-representation implementation examples for reinforcement learning models, including Q-Learning, Policy Gradient, and Deep Q-Network.

## 1. Q-Learning算法 / Q-Learning Algorithm

### 1.1 Q-Learning理论 / Q-Learning Theory

#### 数学表示 / Mathematical Representation

Q-Learning是一种基于值函数的强化学习算法：

Q-Learning is a value-based reinforcement learning algorithm:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

其中：

- $Q(s,a)$ 是状态-动作值函数
- $\alpha$ 是学习率
- $r$ 是即时奖励
- $\gamma$ 是折扣因子
- $s'$ 是下一个状态

where:

- $Q(s,a)$ is the state-action value function
- $\alpha$ is the learning rate
- $r$ is the immediate reward
- $\gamma$ is the discount factor
- $s'$ is the next state

#### 可视化表示 / Visual Representation

```mermaid
graph TD
    A[当前状态 s] --> B[选择动作 a]
    B --> C[执行动作]
    C --> D[获得奖励 r]
    D --> E[观察新状态 s']
    E --> F[更新Q值]
    F --> G[新状态 s']
    G --> A
    
    subgraph "Q值更新"
        H[Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]]
    end
```

#### Rust实现 / Rust Implementation

```rust
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct State {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug)]
struct QLearning {
    q_table: HashMap<(State, Action), f64>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
}

impl QLearning {
    fn new(learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
        }
    }
    
    fn get_q_value(&self, state: &State, action: &Action) -> f64 {
        *self.q_table.get(&(state.clone(), action.clone())).unwrap_or(&0.0)
    }
    
    fn update_q_value(&mut self, state: &State, action: &Action, reward: f64, next_state: &State) {
        let current_q = self.get_q_value(state, action);
        
        // 计算最大Q值
        let max_next_q = self.get_max_q_value(next_state);
        
        // Q-Learning更新公式
        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);
        
        self.q_table.insert((state.clone(), action.clone()), new_q);
    }
    
    fn get_max_q_value(&self, state: &State) -> f64 {
        let actions = vec![Action::Up, Action::Down, Action::Left, Action::Right];
        actions.iter()
            .map(|action| self.get_q_value(state, action))
            .fold(f64::NEG_INFINITY, f64::max)
    }
    
    fn choose_action(&self, state: &State) -> Action {
        let mut rng = rand::thread_rng();
        
        // ε-贪婪策略
        if rng.gen::<f64>() < self.epsilon {
            // 随机选择
            let actions = vec![Action::Up, Action::Down, Action::Left, Action::Right];
            actions[rng.gen_range(0..actions.len())].clone()
        } else {
            // 选择最大Q值的动作
            let actions = vec![Action::Up, Action::Down, Action::Left, Action::Right];
            let mut best_action = &actions[0];
            let mut best_q = self.get_q_value(state, best_action);
            
            for action in &actions {
                let q_value = self.get_q_value(state, action);
                if q_value > best_q {
                    best_q = q_value;
                    best_action = action;
                }
            }
            
            best_action.clone()
        }
    }
    
    fn train(&mut self, episodes: usize) -> Vec<f64> {
        let mut rewards = Vec::new();
        
        for episode in 0..episodes {
            let mut state = State { x: 0, y: 0 };
            let mut total_reward = 0.0;
            
            for step in 0..100 {
                let action = self.choose_action(&state);
                let (next_state, reward) = self.take_action(&state, &action);
                
                self.update_q_value(&state, &action, reward, &next_state);
                
                total_reward += reward;
                state = next_state;
                
                // 到达目标
                if state.x == 4 && state.y == 4 {
                    break;
                }
            }
            
            rewards.push(total_reward);
            
            if episode % 100 == 0 {
                println!("Episode {}, Total Reward: {:.2}", episode, total_reward);
            }
        }
        
        rewards
    }
    
    fn take_action(&self, state: &State, action: &Action) -> (State, f64) {
        let mut next_state = state.clone();
        
        match action {
            Action::Up => next_state.y = (next_state.y - 1).max(0),
            Action::Down => next_state.y = (next_state.y + 1).min(4),
            Action::Left => next_state.x = (next_state.x - 1).max(0),
            Action::Right => next_state.x = (next_state.x + 1).min(4),
        }
        
        let reward = if next_state.x == 4 && next_state.y == 4 {
            100.0  // 目标奖励
        } else {
            -1.0   // 每步成本
        };
        
        (next_state, reward)
    }
}

fn main() {
    let mut q_learning = QLearning::new(0.1, 0.9, 0.1);
    let rewards = q_learning.train(1000);
    
    println!("训练完成！");
    println!("最终Q表大小: {}", q_learning.q_table.len());
    
    // 测试最优策略
    let mut state = State { x: 0, y: 0 };
    let mut path = vec![state.clone()];
    
    for _ in 0..20 {
        let action = q_learning.choose_action(&state);
        let (next_state, _) = q_learning.take_action(&state, &action);
        path.push(next_state.clone());
        state = next_state;
        
        if state.x == 4 && state.y == 4 {
            break;
        }
    }
    
    println!("最优路径: {:?}", path);
}
```

#### Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class State:
    """状态"""
    x: int
    y: int

@dataclass
class Action:
    """动作"""
    name: str
    dx: int
    dy: int

class QLearning:
    """Q-Learning算法"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
        # 定义动作
        self.actions = [
            Action("Up", 0, -1),
            Action("Down", 0, 1),
            Action("Left", -1, 0),
            Action("Right", 1, 0)
        ]
    
    def get_q_value(self, state: State, action: Action) -> float:
        """获取Q值"""
        return self.q_table.get((state.x, state.y, action.name), 0.0)
    
    def update_q_value(self, state: State, action: Action, reward: float, 
                      next_state: State) -> None:
        """更新Q值"""
        current_q = self.get_q_value(state, action)
        max_next_q = self.get_max_q_value(next_state)
        
        # Q-Learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state.x, state.y, action.name)] = new_q
    
    def get_max_q_value(self, state: State) -> float:
        """获取最大Q值"""
        q_values = [self.get_q_value(state, action) for action in self.actions]
        return max(q_values) if q_values else 0.0
    
    def choose_action(self, state: State) -> Action:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            # 随机选择
            return random.choice(self.actions)
        else:
            # 选择最大Q值的动作
            q_values = [self.get_q_value(state, action) for action in self.actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def take_action(self, state: State, action: Action) -> Tuple[State, float]:
        """执行动作"""
        new_x = max(0, min(4, state.x + action.dx))
        new_y = max(0, min(4, state.y + action.dy))
        next_state = State(new_x, new_y)
        
        # 奖励函数
        if next_state.x == 4 and next_state.y == 4:
            reward = 100.0  # 目标奖励
        else:
            reward = -1.0   # 每步成本
        
        return next_state, reward
    
    def train(self, episodes: int) -> List[float]:
        """训练"""
        rewards_history = []
        
        for episode in range(episodes):
            state = State(0, 0)
            total_reward = 0.0
            
            for step in range(100):
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                
                # 到达目标
                if state.x == 4 and state.y == 4:
                    break
            
            rewards_history.append(total_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
        
        return rewards_history
    
    def get_optimal_policy(self) -> Dict[Tuple[int, int], str]:
        """获取最优策略"""
        policy = {}
        
        for x in range(5):
            for y in range(5):
                state = State(x, y)
                q_values = [self.get_q_value(state, action) for action in self.actions]
                best_action_idx = np.argmax(q_values)
                policy[(x, y)] = self.actions[best_action_idx].name
        
        return policy

def visualize_q_learning_results(rewards: List[float], policy: Dict[Tuple[int, int], str]) -> None:
    """可视化Q-Learning结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励曲线
    axes[0].plot(rewards)
    axes[0].set_title('Training Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # 最优策略可视化
    grid = np.zeros((5, 5))
    action_map = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
    
    for (x, y), action in policy.items():
        grid[y, x] = action_map[action]
    
    im = axes[1].imshow(grid, cmap='viridis')
    axes[1].set_title('Optimal Policy')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # 添加动作标签
    for x in range(5):
        for y in range(5):
            action = policy.get((x, y), "None")
            axes[1].text(x, y, action, ha='center', va='center', color='white')
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()

# 测试Q-Learning
if __name__ == "__main__":
    # 创建Q-Learning实例
    q_learning = QLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    
    # 训练
    rewards = q_learning.train(episodes=1000)
    
    # 获取最优策略
    policy = q_learning.get_optimal_policy()
    
    print("训练完成！")
    print(f"Q表大小: {len(q_learning.q_table)}")
    
    # 可视化结果
    visualize_q_learning_results(rewards, policy)
    
    # 测试最优路径
    state = State(0, 0)
    path = [state]
    
    for _ in range(20):
        action = q_learning.choose_action(state)
        next_state, _ = q_learning.take_action(state, action)
        path.append(next_state)
        state = next_state
        
        if state.x == 4 and state.y == 4:
            break
    
    print(f"最优路径: {path}")
```

## 2. 策略梯度算法 / Policy Gradient Algorithm

### 2.1 策略梯度理论 / Policy Gradient Theory

#### 2.1.1 数学表示 / Mathematical Representation

策略梯度算法直接优化策略函数：

Policy gradient algorithms directly optimize the policy function:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]$$

其中：

- $J(\theta)$ 是目标函数
- $\pi_\theta(a|s)$ 是策略函数
- $Q^\pi(s,a)$ 是状态-动作值函数

where:

- $J(\theta)$ is the objective function
- $\pi_\theta(a|s)$ is the policy function
- $Q^\pi(s,a)$ is the state-action value function

#### 2.1.2 可视化表示 / Visual Representation

```mermaid
graph TD
    A[策略网络 π(a|s)] --> B[选择动作]
    B --> C[执行动作]
    C --> D[获得奖励]
    D --> E[计算梯度]
    E --> F[更新策略参数]
    F --> A
    
    subgraph "策略梯度更新"
        G[θ ← θ + α ∇J(θ)]
    end
```

#### 2.1.3 Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class PolicyGradient:
    """策略梯度算法"""
    state_size: int
    action_size: int
    learning_rate: float = 0.01
    
    def __post_init__(self):
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy_net(state_tensor)
        
        # 采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_probs[action].item()
    
    def update_policy(self, states: List[np.ndarray], actions: List[int], 
                     rewards: List[float]) -> None:
        """更新策略"""
        # 计算折扣奖励
        discounted_rewards = self.compute_discounted_rewards(rewards)
        
        # 标准化奖励
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # 计算策略梯度
        policy_loss = 0.0
        
        for state, action, reward in zip(states, actions, discounted_rewards):
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy_net(state_tensor)
            
            # 计算对数概率
            log_prob = torch.log(action_probs[action])
            
            # 策略梯度损失
            policy_loss -= log_prob * reward
        
        # 更新网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
    
    def compute_discounted_rewards(self, rewards: List[float], gamma: float = 0.99) -> np.ndarray:
        """计算折扣奖励"""
        discounted_rewards = []
        running_reward = 0
        
        for reward in reversed(rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        return np.array(discounted_rewards)
    
    def train(self, episodes: int) -> List[float]:
        """训练"""
        episode_rewards = []
        
        for episode in range(episodes):
            states, actions, rewards = self.run_episode()
            
            if len(states) > 0:
                self.update_policy(states, actions, rewards)
                episode_rewards.append(sum(rewards))
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def run_episode(self, max_steps: int = 1000) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """运行一个episode"""
        states, actions, rewards = [], [], []
        
        # 简化的环境：随机游走
        state = np.random.randn(self.state_size)
        
        for step in range(max_steps):
            states.append(state.copy())
            
            action, _ = self.get_action(state)
            actions.append(action)
            
            # 简化的奖励函数
            reward = np.random.normal(0, 1)  # 随机奖励
            rewards.append(reward)
            
            # 更新状态
            state = np.random.randn(self.state_size)
            
            # 随机结束条件
            if np.random.random() < 0.1:
                break
        
        return states, actions, rewards

def visualize_policy_gradient_results(rewards: List[float]) -> None:
    """可视化策略梯度结果"""
    plt.figure(figsize=(12, 6))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Policy Gradient Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # 移动平均奖励
    plt.subplot(1, 2, 2)
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title('Moving Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 测试策略梯度
if __name__ == "__main__":
    # 创建策略梯度实例
    policy_gradient = PolicyGradient(state_size=4, action_size=4, learning_rate=0.01)
    
    # 训练
    rewards = policy_gradient.train(episodes=1000)
    
    print("训练完成！")
    
    # 可视化结果
    visualize_policy_gradient_results(rewards)
```

## 3. 深度Q网络 / Deep Q-Network

### 3.1 DQN理论 / DQN Theory

#### 3.1.1 数学表示 / Mathematical Representation

深度Q网络使用神经网络近似Q函数：

Deep Q-Network uses neural networks to approximate Q-functions:

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

其中：

- $\theta$ 是主网络参数
- $\theta^-$ 是目标网络参数
- $L(\theta)$ 是损失函数

where:

- $\theta$ are the main network parameters
- $\theta^-$ are the target network parameters
- $L(\theta)$ is the loss function

#### 3.1.2 可视化表示 / Visual Representation

```mermaid
graph TD
    A[状态 s] --> B[主网络 Q(s,a;θ)]
    B --> C[选择动作]
    C --> D[执行动作]
    D --> E[获得奖励 r]
    E --> F[观察新状态 s']
    F --> G[目标网络 Q(s',a';θ⁻)]
    G --> H[计算目标值]
    H --> I[更新主网络]
    I --> J[定期更新目标网络]
    J --> A
    
    subgraph "经验回放"
        K[存储经验 (s,a,r,s')]
        L[随机采样批次]
    end
```

#### 3.1.3 Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Deque
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

@dataclass
class DeepQNetwork:
    """深度Q网络算法"""
    state_size: int
    action_size: int
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update: int = 100
    
    def __post_init__(self):
        # 主网络和目标网络
        self.q_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        self.update_count = 0
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state: np.ndarray) -> int:
        """选择动作"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self) -> float:
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, episodes: int) -> List[float]:
        """训练"""
        episode_rewards = []
        losses = []
        
        for episode in range(episodes):
            state = np.random.randn(self.state_size)
            total_reward = 0.0
            episode_loss = 0.0
            
            for step in range(1000):
                action = self.choose_action(state)
                
                # 执行动作
                next_state = np.random.randn(self.state_size)
                reward = np.random.normal(0, 1)
                done = np.random.random() < 0.1
                
                # 存储经验
                self.remember(state, action, reward, next_state, done)
                
                # 经验回放
                loss = self.replay()
                if loss > 0:
                    episode_loss += loss
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            losses.append(episode_loss)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards

def visualize_dqn_results(rewards: List[float]) -> None:
    """可视化DQN结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励曲线
    axes[0].plot(rewards)
    axes[0].set_title('DQN Training Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # 移动平均奖励
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    axes[1].plot(moving_avg)
    axes[1].set_title('Moving Average Reward')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Reward')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 测试DQN
if __name__ == "__main__":
    # 创建DQN实例
    dqn = DeepQNetwork(state_size=4, action_size=4)
    
    # 训练
    rewards = dqn.train(episodes=1000)
    
    print("训练完成！")
    print(f"最终探索率: {dqn.epsilon:.3f}")
    
    # 可视化结果
    visualize_dqn_results(rewards)
```

## 总结 / Summary

本文档提供了强化学习模型的多表征实现示例，包括：

This document provides multi-representation implementation examples for reinforcement learning models, including:

1. **Q-Learning算法** / Q-Learning Algorithm
   - Q值更新公式 / Q-Value Update Formula
   - ε-贪婪策略 / ε-Greedy Policy
   - 表格方法 / Tabular Method

2. **策略梯度算法** / Policy Gradient Algorithm
   - 策略梯度定理 / Policy Gradient Theorem
   - 神经网络策略 / Neural Network Policy
   - 策略优化 / Policy Optimization

3. **深度Q网络** / Deep Q-Network
   - 经验回放 / Experience Replay
   - 目标网络 / Target Network
   - 深度强化学习 / Deep Reinforcement Learning

每个模型都包含数学表示、可视化图表、Rust/Python实现，展示了强化学习在不同领域的应用。

Each model includes mathematical representation, visual diagrams, and Rust/Python implementations, demonstrating the applications of reinforcement learning in different domains.
