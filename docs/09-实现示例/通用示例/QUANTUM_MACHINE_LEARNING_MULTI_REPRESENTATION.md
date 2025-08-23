# 量子机器学习扩展多表征示例 / Quantum Machine Learning Extension Multi-Representation

## 概述 / Overview

本文档提供量子机器学习扩展的完整多表征实现，包括量子神经网络、量子优化算法和量子-经典混合计算。

This document provides comprehensive multi-representation implementations for Quantum Machine Learning extension, including Quantum Neural Networks, Quantum Optimization Algorithms, and Quantum-Classical Hybrid Computing.

## 1. 量子神经网络 / Quantum Neural Networks

### 1.1 数学表示 / Mathematical Representation

#### 量子比特状态 / Quantum Bit State
```
|ψ⟩ = α|0⟩ + β|1⟩, where |α|² + |β|² = 1
```

#### 量子门操作 / Quantum Gate Operations
```
H = 1/√2 [1  1]    (Hadamard gate)
         [1 -1]

X = [0 1]          (Pauli-X gate)
    [1 0]

Y = [0 -i]         (Pauli-Y gate)
    [i  0]

Z = [1  0]         (Pauli-Z gate)
    [0 -1]
```

#### 量子神经网络层 / Quantum Neural Network Layer
```
U(θ) = exp(-iθσ/2)  (Parameterized quantum gate)
|ψ_out⟩ = U(θ)|ψ_in⟩
```

#### 量子测量 / Quantum Measurement
```
P(0) = |⟨0|ψ⟩|², P(1) = |⟨1|ψ⟩|²
```

### 1.2 流程图表示 / Flowchart Representation

```mermaid
flowchart TD
    A[输入量子态 |ψ_in⟩] --> B[量子门层 U1]
    B --> C[量子门层 U2]
    C --> D[量子门层 U3]
    D --> E[量子测量]
    E --> F[经典后处理]
    F --> G[输出预测]
    
    H[参数θ] --> I[参数优化]
    I --> J[梯度计算]
    J --> K[参数更新]
    K --> H
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
    style H fill:#fff3e0
```

### 1.3 Python实现 / Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantumBit:
    """量子比特类 / Quantum Bit Class"""
    
    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0):
        self.alpha = alpha
        self.beta = beta
        self._normalize()
    
    def _normalize(self):
        """归一化量子态 / Normalize quantum state"""
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        self.alpha /= norm
        self.beta /= norm
    
    def measure(self) -> int:
        """测量量子比特 / Measure quantum bit"""
        p0 = abs(self.alpha)**2
        return 0 if np.random.random() < p0 else 1
    
    def get_state_vector(self) -> np.ndarray:
        """获取状态向量 / Get state vector"""
        return np.array([self.alpha, self.beta])

class QuantumGates:
    """量子门类 / Quantum Gates Class"""
    
    @staticmethod
    def hadamard():
        """Hadamard门 / Hadamard gate"""
        return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    
    @staticmethod
    def pauli_x():
        """Pauli-X门 / Pauli-X gate"""
        return np.array([[0, 1], [1, 0]])
    
    @staticmethod
    def pauli_y():
        """Pauli-Y门 / Pauli-Y gate"""
        return np.array([[0, -1j], [1j, 0]])
    
    @staticmethod
    def pauli_z():
        """Pauli-Z门 / Pauli-Z gate"""
        return np.array([[1, 0], [0, -1]])
    
    @staticmethod
    def rotation_x(theta: float):
        """X轴旋转门 / X-axis rotation gate"""
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]])
    
    @staticmethod
    def rotation_y(theta: float):
        """Y轴旋转门 / Y-axis rotation gate"""
        return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                        [np.sin(theta/2), np.cos(theta/2)]])
    
    @staticmethod
    def rotation_z(theta: float):
        """Z轴旋转门 / Z-axis rotation gate"""
        return np.array([[np.exp(-1j*theta/2), 0],
                        [0, np.exp(1j*theta/2)]])

class QuantumNeuralNetwork:
    """量子神经网络 / Quantum Neural Network"""
    
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.parameters = np.random.randn(num_layers, num_qubits, 3) * 0.1
        self.gates = QuantumGates()
    
    def apply_gate(self, qubit: QuantumBit, gate: np.ndarray) -> QuantumBit:
        """应用量子门 / Apply quantum gate"""
        state = qubit.get_state_vector()
        new_state = gate @ state
        return QuantumBit(new_state[0], new_state[1])
    
    def quantum_layer(self, qubits: List[QuantumBit], layer_params: np.ndarray) -> List[QuantumBit]:
        """量子层 / Quantum layer"""
        new_qubits = []
        for i, qubit in enumerate(qubits):
            # 应用旋转门 / Apply rotation gates
            rx_gate = self.gates.rotation_x(layer_params[i, 0])
            ry_gate = self.gates.rotation_y(layer_params[i, 1])
            rz_gate = self.gates.rotation_z(layer_params[i, 2])
            
            # 依次应用门 / Apply gates sequentially
            qubit = self.apply_gate(qubit, rx_gate)
            qubit = self.apply_gate(qubit, ry_gate)
            qubit = self.apply_gate(qubit, rz_gate)
            new_qubits.append(qubit)
        
        return new_qubits
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """前向传播 / Forward pass"""
        # 初始化量子比特 / Initialize quantum bits
        qubits = [QuantumBit() for _ in range(self.num_qubits)]
        
        # 编码输入数据 / Encode input data
        for i, qubit in enumerate(qubits):
            if i < len(input_data):
                angle = input_data[i] * np.pi
                qubit = self.apply_gate(qubit, self.gates.rotation_y(angle))
        
        # 应用量子层 / Apply quantum layers
        for layer in range(self.num_layers):
            qubits = self.quantum_layer(qubits, self.parameters[layer])
        
        # 测量输出 / Measure output
        measurements = [qubit.measure() for qubit in qubits]
        return np.array(measurements, dtype=float)
    
    def compute_gradient(self, input_data: np.ndarray, target: np.ndarray) -> np.ndarray:
        """计算梯度 / Compute gradient"""
        epsilon = 0.01
        gradients = np.zeros_like(self.parameters)
        
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                for param in range(3):
                    # 参数偏移 / Parameter shift
                    self.parameters[layer, qubit, param] += epsilon
                    output_plus = self.forward(input_data)
                    self.parameters[layer, qubit, param] -= 2 * epsilon
                    output_minus = self.forward(input_data)
                    self.parameters[layer, qubit, param] += epsilon
                    
                    # 计算梯度 / Compute gradient
                    loss_plus = np.mean((output_plus - target)**2)
                    loss_minus = np.mean((output_minus - target)**2)
                    gradients[layer, qubit, param] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def train(self, input_data: np.ndarray, target: np.ndarray, 
              learning_rate: float = 0.1, epochs: int = 100):
        """训练量子神经网络 / Train quantum neural network"""
        losses = []
        
        for epoch in range(epochs):
            # 前向传播 / Forward pass
            output = self.forward(input_data)
            loss = np.mean((output - target)**2)
            losses.append(loss)
            
            # 计算梯度 / Compute gradient
            gradients = self.compute_gradient(input_data, target)
            
            # 更新参数 / Update parameters
            self.parameters -= learning_rate * gradients
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

def visualize_quantum_neural_network():
    """可视化量子神经网络 / Visualize quantum neural network"""
    # 创建量子神经网络 / Create quantum neural network
    qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=3)
    
    # 生成训练数据 / Generate training data
    input_data = np.random.randn(4)
    target = np.array([0, 1, 0, 1])
    
    # 训练网络 / Train network
    losses = qnn.train(input_data, target, epochs=50)
    
    # 可视化训练过程 / Visualize training process
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('量子神经网络训练损失 / QNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 测试网络 / Test network
    test_input = np.random.randn(4)
    prediction = qnn.forward(test_input)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(prediction)), prediction)
    plt.title('量子神经网络预测结果 / QNN Prediction Results')
    plt.xlabel('Qubit')
    plt.ylabel('Measurement')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return qnn, losses

if __name__ == "__main__":
    print("量子神经网络示例 / Quantum Neural Network Example")
    qnn, losses = visualize_quantum_neural_network()
```
