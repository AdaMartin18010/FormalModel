# 电力能源模型高级实现 / Advanced Power & Energy Model Implementations

## 目录 / Table of Contents

- [电力能源模型高级实现 / Advanced Power \& Energy Model Implementations](#电力能源模型高级实现--advanced-power--energy-model-implementations)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.3.24 深度学习在电力系统中的应用 / Deep Learning in Power Systems](#8324-深度学习在电力系统中的应用--deep-learning-in-power-systems)
    - [深度神经网络负荷预测 / Deep Neural Network Load Forecasting](#深度神经网络负荷预测--deep-neural-network-load-forecasting)
    - [深度强化学习电力系统优化 / Deep Reinforcement Learning for Power System Optimization](#深度强化学习电力系统优化--deep-reinforcement-learning-for-power-system-optimization)
  - [8.3.25 图神经网络在电力系统中的应用 / Graph Neural Networks in Power Systems](#8325-图神经网络在电力系统中的应用--graph-neural-networks-in-power-systems)
    - [图神经网络电力系统建模 / Graph Neural Network Power System Modeling](#图神经网络电力系统建模--graph-neural-network-power-system-modeling)
  - [8.3.26 多智能体系统在电力系统中的应用 / Multi-Agent Systems in Power Systems](#8326-多智能体系统在电力系统中的应用--multi-agent-systems-in-power-systems)
    - [多智能体电力系统控制 / Multi-Agent Power System Control](#多智能体电力系统控制--multi-agent-power-system-control)

---

## 8.3.24 深度学习在电力系统中的应用 / Deep Learning in Power Systems

### 深度神经网络负荷预测 / Deep Neural Network Load Forecasting

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DeepLoadForecaster(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=3, output_size=24):
        super(DeepLoadForecaster, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全连接层
        output = self.fc_layers(attn_out[:, -1, :])
        
        return output

class TransformerLoadForecaster(nn.Module):
    def __init__(self, input_size=24, d_model=128, nhead=8, num_layers=6, output_size=24):
        super(TransformerLoadForecaster, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = self.create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)
        
        # 位置编码
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer编码器
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        transformer_out = self.transformer(x)
        
        # 输出层
        output = self.fc(transformer_out[-1])  # 使用最后一个时间步的输出
        
        return output

class LoadForecastingSystem:
    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'lstm':
            self.model = DeepLoadForecaster().to(self.device)
        elif model_type == 'transformer':
            self.model = TransformerLoadForecaster().to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def prepare_data(self, load_data, window_size=24, forecast_horizon=24):
        """准备数据"""
        X, y = [], []
        
        for i in range(len(load_data) - window_size - forecast_horizon + 1):
            X.append(load_data[i:i+window_size])
            y.append(load_data[i+window_size:i+window_size+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled, y
    
    def train(self, X, y, epochs=100, batch_size=32):
        """训练模型"""
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return train_losses
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions = self.predict(X_test)
        
        # 计算各种指标
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': np.sqrt(mse)
        }

# 使用示例
def deep_learning_load_forecasting_example():
    # 生成模拟负荷数据
    np.random.seed(42)
    time_steps = 1000
    base_load = 1000
    seasonal_pattern = 200 * np.sin(np.linspace(0, 4*np.pi, time_steps))
    trend = np.linspace(0, 100, time_steps)
    noise = np.random.normal(0, 50, time_steps)
    
    load_data = base_load + seasonal_pattern + trend + noise
    
    # 创建预测系统
    forecaster = LoadForecastingSystem(model_type='transformer')
    
    # 准备数据
    X, y = forecaster.prepare_data(load_data, window_size=24, forecast_horizon=24)
    
    # 分割训练和测试数据
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 训练模型
    print("开始训练深度学习负荷预测模型...")
    train_losses = forecaster.train(X_train, y_train, epochs=50)
    
    # 评估模型
    metrics = forecaster.evaluate(X_test, y_test)
    print(f"模型评估结果: {metrics}")
    
    # 预测
    predictions = forecaster.predict(X_test[:5])
    print(f"预测结果示例: {predictions[0][:5]}")

if __name__ == "__main__":
    deep_learning_load_forecasting_example()
```

### 深度强化学习电力系统优化 / Deep Reinforcement Learning for Power System Optimization

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

class PowerSystemEnvironment(gym.Env):
    def __init__(self, num_generators=5, time_horizon=24):
        super(PowerSystemEnvironment, self).__init__()
        
        self.num_generators = num_generators
        self.time_horizon = time_horizon
        self.current_time = 0
        
        # 发电机参数
        self.generator_costs = np.array([30, 35, 40, 45, 50])  # 成本
        self.generator_capacities = np.array([100, 150, 200, 250, 300])  # 容量
        self.generator_min_powers = np.array([20, 30, 40, 50, 60])  # 最小出力
        
        # 负荷数据
        self.demand_profile = self.generate_demand_profile()
        
        # 动作空间：每个发电机的出力
        self.action_space = gym.spaces.Box(
            low=np.zeros(num_generators),
            high=self.generator_capacities,
            dtype=np.float32
        )
        
        # 状态空间：时间、负荷、各发电机状态
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + 1 + num_generators,),  # 时间 + 负荷 + 发电机状态
            dtype=np.float32
        )
        
    def generate_demand_profile(self):
        """生成负荷曲线"""
        base_demand = 800
        daily_pattern = 200 * np.sin(np.linspace(0, 2*np.pi, self.time_horizon))
        weekly_pattern = 50 * np.sin(np.linspace(0, 2*np.pi, self.time_horizon/7))
        noise = np.random.normal(0, 30, self.time_horizon)
        
        return base_demand + daily_pattern + weekly_pattern + noise
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.current_demand = self.demand_profile[0]
        
        # 初始状态：所有发电机停机
        initial_state = np.concatenate([
            [self.current_time / self.time_horizon],  # 归一化时间
            [self.current_demand / 1000],  # 归一化负荷
            np.zeros(self.num_generators)  # 发电机状态
        ])
        
        return initial_state.astype(np.float32)
    
    def step(self, action):
        """执行动作"""
        # 确保动作在合理范围内
        action = np.clip(action, 0, self.generator_capacities)
        
        # 计算总发电量
        total_generation = np.sum(action)
        current_demand = self.demand_profile[self.current_time]
        
        # 计算奖励
        reward = self.calculate_reward(action, total_generation, current_demand)
        
        # 更新状态
        self.current_time += 1
        
        if self.current_time >= self.time_horizon:
            done = True
        else:
            done = False
            self.current_demand = self.demand_profile[self.current_time]
        
        # 构建新状态
        next_state = np.concatenate([
            [self.current_time / self.time_horizon],
            [self.current_demand / 1000],
            action / self.generator_capacities  # 归一化发电机出力
        ])
        
        info = {
            'total_generation': total_generation,
            'demand': current_demand,
            'cost': self.calculate_cost(action)
        }
        
        return next_state.astype(np.float32), reward, done, info
    
    def calculate_reward(self, action, total_generation, demand):
        """计算奖励"""
        # 功率平衡奖励
        balance_penalty = -abs(total_generation - demand) * 10
        
        # 成本奖励
        cost = self.calculate_cost(action)
        cost_penalty = -cost / 1000
        
        # 约束违反惩罚
        constraint_penalty = 0
        
        # 最小出力约束
        for i, power in enumerate(action):
            if power > 0 and power < self.generator_min_powers[i]:
                constraint_penalty -= 100
        
        # 总奖励
        reward = balance_penalty + cost_penalty + constraint_penalty
        
        return reward
    
    def calculate_cost(self, action):
        """计算发电成本"""
        return np.sum(action * self.generator_costs)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # 神经网络
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        # 训练参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.95
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def build_model(self):
        """构建神经网络"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.detach().numpy()[0]
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([data[0] for data in minibatch])
        actions = torch.FloatTensor([data[1] for data in minibatch])
        rewards = torch.FloatTensor([data[2] for data in minibatch])
        next_states = torch.FloatTensor([data[3] for data in minibatch])
        dones = torch.BoolTensor([data[4] for data in minibatch])
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        target_q_values = current_q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i] = rewards[i]
            else:
                target_q_values[i] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PowerSystemRLTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
    def train(self, episodes=1000):
        """训练强化学习智能体"""
        scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.env.time_horizon):
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.agent.replay()
            
            if episode % 100 == 0:
                self.agent.update_target_network()
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f'Episode: {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.2f}')
        
        return scores

# 使用示例
def reinforcement_learning_example():
    # 创建环境
    env = PowerSystemEnvironment(num_generators=5, time_horizon=24)
    
    # 创建智能体
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)
    
    # 训练
    trainer = PowerSystemRLTrainer(env, agent)
    scores = trainer.train(episodes=500)
    
    print("强化学习训练完成")
    print(f"最终平均得分: {np.mean(scores[-100:]):.2f}")

if __name__ == "__main__":
    reinforcement_learning_example()
```

---

## 8.3.25 图神经网络在电力系统中的应用 / Graph Neural Networks in Power Systems

### 图神经网络电力系统建模 / Graph Neural Network Power System Modeling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt

class PowerSystemGraph:
    def __init__(self, num_buses=10, num_lines=15):
        self.num_buses = num_buses
        self.num_lines = num_lines
        self.graph = self.create_power_system_graph()
        
    def create_power_system_graph(self):
        """创建电力系统图"""
        G = nx.Graph()
        
        # 添加母线节点
        for i in range(self.num_buses):
            G.add_node(i, 
                      voltage=1.0 + 0.1 * np.random.randn(),
                      angle=0.0 + 0.1 * np.random.randn(),
                      bus_type=np.random.choice(['slack', 'PV', 'PQ']),
                      load=np.random.exponential(100),
                      generation=np.random.exponential(150))
        
        # 添加线路边
        edges = []
        edge_features = []
        
        for _ in range(self.num_lines):
            from_bus = np.random.randint(0, self.num_buses)
            to_bus = np.random.randint(0, self.num_buses)
            
            if from_bus != to_bus and not G.has_edge(from_bus, to_bus):
                resistance = np.random.exponential(0.1)
                reactance = np.random.exponential(0.2)
                capacity = np.random.exponential(200)
                
                G.add_edge(from_bus, to_bus, 
                          resistance=resistance,
                          reactance=reactance,
                          capacity=capacity)
                
                edges.append([from_bus, to_bus])
                edge_features.append([resistance, reactance, capacity])
        
        return G
    
    def to_pytorch_geometric(self):
        """转换为PyTorch Geometric格式"""
        # 节点特征
        node_features = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            features = [
                node_data['voltage'],
                node_data['angle'],
                1.0 if node_data['bus_type'] == 'slack' else 0.0,
                1.0 if node_data['bus_type'] == 'PV' else 0.0,
                1.0 if node_data['bus_type'] == 'PQ' else 0.0,
                node_data['load'],
                node_data['generation']
            ]
            node_features.append(features)
        
        node_features = torch.FloatTensor(node_features)
        
        # 边索引
        edge_index = []
        edge_attr = []
        
        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge]
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # 无向图
            
            edge_features = [
                edge_data['resistance'],
                edge_data['reactance'],
                edge_data['capacity']
            ]
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)  # 无向图
        
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

class GNNPowerSystemModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=3, num_layers=3):
        super(GNNPowerSystemModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN层
        for i, (gcn_layer, batch_norm) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

class GATPowerSystemModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=3, num_layers=3, heads=8):
        super(GATPowerSystemModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim * heads, output_dim)
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GAT层
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x = gat_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

class PowerSystemGNNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, epochs=100):
        """训练模型"""
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # 前向传播
                output = self.model(batch)
                
                # 计算损失（这里使用简化的目标）
                target = torch.randn_like(output)  # 简化的目标
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return train_losses
    
    def predict(self, data):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            return output

# 使用示例
def graph_neural_network_example():
    # 创建电力系统图
    power_system = PowerSystemGraph(num_buses=20, num_lines=30)
    
    # 转换为PyTorch Geometric格式
    data = power_system.to_pytorch_geometric()
    
    print(f"节点数量: {data.x.shape[0]}")
    print(f"边数量: {data.edge_index.shape[1]}")
    print(f"节点特征维度: {data.x.shape[1]}")
    print(f"边特征维度: {data.edge_attr.shape[1]}")
    
    # 创建GNN模型
    gnn_model = GNNPowerSystemModel(input_dim=7, hidden_dim=64, output_dim=3)
    
    # 训练模型
    trainer = PowerSystemGNNTrainer(gnn_model)
    
    # 创建数据加载器
    train_loader = DataLoader([data], batch_size=1, shuffle=True)
    
    # 训练
    train_losses = trainer.train(train_loader, epochs=50)
    
    # 预测
    predictions = trainer.predict(data)
    print(f"预测结果形状: {predictions.shape}")
    
    # 可视化训练损失
    plt.plot(train_losses)
    plt.title('GNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    graph_neural_network_example()
```

---

## 8.3.26 多智能体系统在电力系统中的应用 / Multi-Agent Systems in Power Systems

### 多智能体电力系统控制 / Multi-Agent Power System Control

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class PowerSystemAgent:
    def __init__(self, agent_id, state_size, action_size, learning_rate=0.001):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # 神经网络
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        # 训练参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.95
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def build_model(self):
        """构建神经网络"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.detach().numpy()[0]
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([data[0] for data in minibatch])
        actions = torch.FloatTensor([data[1] for data in minibatch])
        rewards = torch.FloatTensor([data[2] for data in minibatch])
        next_states = torch.FloatTensor([data[3] for data in minibatch])
        dones = torch.BoolTensor([data[4] for data in minibatch])
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        target_q_values = current_q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i] = rewards[i]
            else:
                target_q_values[i] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MultiAgentPowerSystem:
    def __init__(self, num_agents=5, time_horizon=24):
        self.num_agents = num_agents
        self.time_horizon = time_horizon
        self.current_time = 0
        
        # 系统参数
        self.demand_profile = self.generate_demand_profile()
        self.agent_capacities = np.random.exponential(100, num_agents)
        self.agent_costs = np.random.exponential(30, num_agents)
        
        # 创建智能体
        self.agents = []
        for i in range(num_agents):
            state_size = 3  # 时间、负荷、自身状态
            action_size = 1  # 出力
            agent = PowerSystemAgent(i, state_size, action_size)
            self.agents.append(agent)
        
        # 通信网络
        self.communication_matrix = self.create_communication_matrix()
        
    def generate_demand_profile(self):
        """生成负荷曲线"""
        base_demand = 500
        daily_pattern = 100 * np.sin(np.linspace(0, 2*np.pi, self.time_horizon))
        noise = np.random.normal(0, 20, self.time_horizon)
        return base_demand + daily_pattern + noise
    
    def create_communication_matrix(self):
        """创建通信矩阵"""
        # 随机生成通信拓扑
        matrix = np.random.rand(self.num_agents, self.num_agents)
        matrix = (matrix > 0.7).astype(float)  # 稀疏连接
        np.fill_diagonal(matrix, 0)  # 自己不能和自己通信
        return matrix
    
    def get_agent_state(self, agent_id):
        """获取智能体状态"""
        current_demand = self.demand_profile[self.current_time]
        
        # 计算其他智能体的平均出力
        other_agents_power = []
        for i in range(self.num_agents):
            if i != agent_id and self.communication_matrix[agent_id, i] > 0:
                # 这里简化处理，实际应该从其他智能体获取信息
                other_agents_power.append(np.random.exponential(50))
        
        avg_other_power = np.mean(other_agents_power) if other_agents_power else 0
        
        state = [
            self.current_time / self.time_horizon,  # 归一化时间
            current_demand / 1000,  # 归一化负荷
            avg_other_power / 100  # 归一化其他智能体出力
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, actions):
        """执行所有智能体的动作"""
        # 计算总发电量
        total_generation = np.sum(actions)
        current_demand = self.demand_profile[self.current_time]
        
        # 计算每个智能体的奖励
        rewards = []
        for i, action in enumerate(actions):
            reward = self.calculate_agent_reward(i, action, total_generation, current_demand)
            rewards.append(reward)
        
        # 更新状态
        self.current_time += 1
        
        if self.current_time >= self.time_horizon:
            done = True
        else:
            done = False
        
        # 获取新状态
        next_states = []
        for i in range(self.num_agents):
            next_state = self.get_agent_state(i)
            next_states.append(next_state)
        
        return next_states, rewards, done
    
    def calculate_agent_reward(self, agent_id, action, total_generation, demand):
        """计算智能体奖励"""
        # 功率平衡奖励
        balance_penalty = -abs(total_generation - demand) * 5
        
        # 成本奖励
        cost = action * self.agent_costs[agent_id]
        cost_penalty = -cost / 1000
        
        # 容量约束惩罚
        capacity_penalty = 0
        if action > self.agent_capacities[agent_id]:
            capacity_penalty = -100
        
        # 协作奖励（鼓励智能体之间的协作）
        collaboration_reward = 0
        if abs(total_generation - demand) < 50:  # 如果功率平衡良好
            collaboration_reward = 10
        
        total_reward = balance_penalty + cost_penalty + capacity_penalty + collaboration_reward
        return total_reward
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        initial_states = []
        for i in range(self.num_agents):
            state = self.get_agent_state(i)
            initial_states.append(state)
        return initial_states

class MultiAgentTrainer:
    def __init__(self, environment):
        self.environment = environment
        self.agents = environment.agents
        
    def train(self, episodes=1000):
        """训练多智能体系统"""
        episode_rewards = []
        
        for episode in range(episodes):
            states = self.environment.reset()
            episode_reward = 0
            
            for step in range(self.environment.time_horizon):
                # 所有智能体选择动作
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(states[i])
                    actions.append(action[0])  # 取第一个动作值
                
                # 执行动作
                next_states, rewards, done = self.environment.step(actions)
                
                # 存储经验
                for i, agent in enumerate(self.agents):
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
                
                states = next_states
                episode_reward += np.mean(rewards)
                
                if done:
                    break
            
            # 所有智能体进行经验回放
            for agent in self.agents:
                agent.replay()
            
            episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f'Episode: {episode}, Average Reward: {avg_reward:.2f}')
        
        return episode_rewards

# 使用示例
def multi_agent_power_system_example():
    # 创建多智能体电力系统
    env = MultiAgentPowerSystem(num_agents=5, time_horizon=24)
    
    # 创建训练器
    trainer = MultiAgentTrainer(env)
    
    # 训练
    rewards = trainer.train(episodes=500)
    
    print("多智能体训练完成")
    print(f"最终平均奖励: {np.mean(rewards[-100:]):.2f}")
    
    # 可视化训练过程
    plt.plot(rewards)
    plt.title('Multi-Agent Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

if __name__ == "__main__":
    multi_agent_power_system_example()
```

---

*最后更新: 2025-01-01*
*版本: 1.0.0*
