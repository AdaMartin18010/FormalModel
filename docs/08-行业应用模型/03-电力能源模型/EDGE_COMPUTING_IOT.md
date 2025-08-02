# 边缘计算与物联网在电力系统中的应用实现 / Edge Computing & IoT Implementation for Power Systems

## 目录 / Table of Contents

- [边缘计算与物联网在电力系统中的应用实现 / Edge Computing \& IoT Implementation for Power Systems](#边缘计算与物联网在电力系统中的应用实现--edge-computing--iot-implementation-for-power-systems)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.3.49 边缘计算架构 / Edge Computing Architecture](#8349-边缘计算架构--edge-computing-architecture)
    - [边缘计算电力系统架构 / Edge Computing Power System Architecture](#边缘计算电力系统架构--edge-computing-power-system-architecture)

---

## 8.3.49 边缘计算架构 / Edge Computing Architecture

### 边缘计算电力系统架构 / Edge Computing Power System Architecture

```python
import asyncio
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import threading
from dataclasses import dataclass, asdict
import queue
import hashlib

@dataclass
class EdgeDevice:
    """边缘设备数据结构"""
    device_id: str
    device_type: str  # 'sensor', 'controller', 'gateway', 'actuator'
    location: str
    capabilities: List[str]
    status: str  # 'online', 'offline', 'maintenance'
    last_update: datetime
    data_buffer: List[Dict[str, Any]]
    max_buffer_size: int = 1000

@dataclass
class EdgeNode:
    """边缘节点数据结构"""
    node_id: str
    location: str
    computing_capacity: float  # CPU cores
    memory_capacity: float  # GB
    storage_capacity: float  # GB
    network_bandwidth: float  # Mbps
    connected_devices: List[str]
    status: str  # 'active', 'inactive', 'overloaded'

class EdgeComputingArchitecture:
    """边缘计算架构"""
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.logger = logging.getLogger(f"EdgeComputing.{system_id}")
        
        # 边缘节点
        self.edge_nodes = {}
        
        # 边缘设备
        self.edge_devices = {}
        
        # 任务队列
        self.task_queue = queue.Queue()
        
        # 数据流
        self.data_streams = {}
        
        # 系统状态
        self.system_status = {
            'total_nodes': 0,
            'total_devices': 0,
            'active_tasks': 0,
            'data_throughput': 0.0,
            'last_update': datetime.now()
        }
        
    def add_edge_node(self, node: EdgeNode):
        """添加边缘节点"""
        self.edge_nodes[node.node_id] = node
        self.system_status['total_nodes'] = len(self.edge_nodes)
        self.logger.info(f"Added edge node: {node.node_id}")
    
    def add_edge_device(self, device: EdgeDevice):
        """添加边缘设备"""
        self.edge_devices[device.device_id] = device
        self.system_status['total_devices'] = len(self.edge_devices)
        self.logger.info(f"Added edge device: {device.device_id}")
    
    def connect_device_to_node(self, device_id: str, node_id: str):
        """连接设备到节点"""
        if device_id in self.edge_devices and node_id in self.edge_nodes:
            self.edge_nodes[node_id].connected_devices.append(device_id)
            self.logger.info(f"Connected device {device_id} to node {node_id}")
    
    def submit_task(self, task: Dict[str, Any]):
        """提交任务到边缘节点"""
        task['id'] = f"task_{int(time.time() * 1000)}"
        task['submit_time'] = datetime.now()
        task['status'] = 'pending'
        
        self.task_queue.put(task)
        self.system_status['active_tasks'] += 1
        
        self.logger.info(f"Submitted task: {task['id']}")
        return task['id']
    
    async def process_tasks(self):
        """处理任务队列"""
        while True:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    
                    # 选择最佳边缘节点
                    best_node = self.select_best_node(task)
                    
                    if best_node:
                        # 执行任务
                        result = await self.execute_task_on_node(task, best_node)
                        task['result'] = result
                        task['status'] = 'completed'
                        task['completion_time'] = datetime.now()
                        
                        self.logger.info(f"Completed task {task['id']} on node {best_node}")
                    else:
                        task['status'] = 'failed'
                        task['error'] = 'No suitable node available'
                
                await asyncio.sleep(0.1)  # 100ms间隔
                
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
                await asyncio.sleep(1)
    
    def select_best_node(self, task: Dict[str, Any]) -> Optional[str]:
        """选择最佳边缘节点"""
        best_node = None
        best_score = -1
        
        for node_id, node in self.edge_nodes.items():
            if node.status != 'active':
                continue
            
            # 计算节点评分
            score = self.calculate_node_score(node, task)
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    def calculate_node_score(self, node: EdgeNode, task: Dict[str, Any]) -> float:
        """计算节点评分"""
        # 基础评分
        score = 0.0
        
        # 计算能力评分
        cpu_score = min(node.computing_capacity / task.get('cpu_requirement', 1), 1.0)
        memory_score = min(node.memory_capacity / task.get('memory_requirement', 1), 1.0)
        
        # 网络带宽评分
        bandwidth_score = min(node.network_bandwidth / task.get('bandwidth_requirement', 1), 1.0)
        
        # 负载评分
        load_score = 1.0 - (len(node.connected_devices) / 100)  # 假设最大100个设备
        
        # 综合评分
        score = (cpu_score + memory_score + bandwidth_score + load_score) / 4
        
        return score
    
    async def execute_task_on_node(self, task: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """在边缘节点上执行任务"""
        try:
            # 模拟任务执行
            execution_time = task.get('estimated_time', 1.0)
            await asyncio.sleep(execution_time)
            
            # 生成结果
            result = {
                'task_id': task['id'],
                'node_id': node_id,
                'execution_time': execution_time,
                'result_data': self.generate_task_result(task),
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            return {'error': str(e)}
    
    def generate_task_result(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """生成任务结果"""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'load_forecast':
            return self.generate_load_forecast_result()
        elif task_type == 'fault_detection':
            return self.generate_fault_detection_result()
        elif task_type == 'optimization':
            return self.generate_optimization_result()
        else:
            return {'status': 'completed', 'data': 'generic_result'}
    
    def generate_load_forecast_result(self) -> Dict[str, Any]:
        """生成负荷预测结果"""
        forecast_horizon = 24
        base_load = 1000
        forecast = []
        
        for hour in range(forecast_horizon):
            # 简化的负荷预测
            daily_pattern = 200 * np.sin(2 * np.pi * hour / 24)
            weekly_pattern = 50 * np.sin(2 * np.pi * hour / 168)
            random_variation = np.random.normal(0, 50)
            
            load = base_load + daily_pattern + weekly_pattern + random_variation
            forecast.append({
                'hour': hour,
                'load': max(0, load),
                'confidence': 0.85 + np.random.uniform(-0.1, 0.1)
            })
        
        return {
            'forecast': forecast,
            'accuracy': 0.87,
            'method': 'edge_ml_forecast'
        }
    
    def generate_fault_detection_result(self) -> Dict[str, Any]:
        """生成故障检测结果"""
        # 模拟故障检测
        fault_probability = np.random.uniform(0, 1)
        
        if fault_probability > 0.8:
            return {
                'fault_detected': True,
                'fault_type': 'overvoltage',
                'severity': 'high',
                'location': 'substation_001',
                'confidence': fault_probability,
                'recommended_action': 'Reduce voltage or isolate section'
            }
        else:
            return {
                'fault_detected': False,
                'system_status': 'normal',
                'confidence': 1 - fault_probability
            }
    
    def generate_optimization_result(self) -> Dict[str, Any]:
        """生成优化结果"""
        return {
            'optimization_type': 'economic_dispatch',
            'total_cost': 50000 + np.random.uniform(-1000, 1000),
            'efficiency_improvement': 0.05 + np.random.uniform(-0.02, 0.02),
            'recommended_actions': [
                'Increase generation at unit 1',
                'Reduce generation at unit 3',
                'Activate reserve capacity'
            ]
        }

class IoTDeviceManager:
    """物联网设备管理器"""
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.logger = logging.getLogger(f"IoTDeviceManager.{manager_id}")
        
        # 设备注册表
        self.device_registry = {}
        
        # 设备状态监控
        self.device_status = {}
        
        # 数据收集器
        self.data_collectors = {}
        
        # 设备配置
        self.device_configs = {}
    
    def register_device(self, device: EdgeDevice):
        """注册设备"""
        self.device_registry[device.device_id] = device
        self.device_status[device.device_id] = {
            'status': 'registered',
            'last_heartbeat': datetime.now(),
            'data_count': 0,
            'error_count': 0
        }
        
        self.logger.info(f"Registered device: {device.device_id}")
    
    def update_device_status(self, device_id: str, status_update: Dict[str, Any]):
        """更新设备状态"""
        if device_id in self.device_status:
            self.device_status[device_id].update(status_update)
            self.device_status[device_id]['last_update'] = datetime.now()
    
    def collect_device_data(self, device_id: str, data: Dict[str, Any]):
        """收集设备数据"""
        if device_id in self.device_registry:
            device = self.device_registry[device_id]
            
            # 添加时间戳
            data['timestamp'] = datetime.now()
            data['device_id'] = device_id
            
            # 存储到设备缓冲区
            device.data_buffer.append(data)
            
            # 保持缓冲区大小
            if len(device.data_buffer) > device.max_buffer_size:
                device.data_buffer.pop(0)
            
            # 更新状态
            self.device_status[device_id]['data_count'] += 1
            self.device_status[device_id]['last_heartbeat'] = datetime.now()
            
            self.logger.debug(f"Collected data from device {device_id}: {len(data)} fields")
    
    def get_device_data(self, device_id: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取设备数据"""
        if device_id in self.device_registry:
            device = self.device_registry[device_id]
            return device.data_buffer[-count:] if device.data_buffer else []
        return []
    
    def configure_device(self, device_id: str, config: Dict[str, Any]):
        """配置设备"""
        if device_id in self.device_registry:
            self.device_configs[device_id] = config
            self.logger.info(f"Configured device {device_id}: {config}")
    
    def get_device_health(self, device_id: str) -> Dict[str, Any]:
        """获取设备健康状态"""
        if device_id not in self.device_status:
            return {'status': 'unknown'}
        
        status = self.device_status[device_id]
        device = self.device_registry.get(device_id)
        
        # 计算健康分数
        health_score = 100.0
        
        # 检查心跳
        time_since_heartbeat = (datetime.now() - status['last_heartbeat']).total_seconds()
        if time_since_heartbeat > 300:  # 5分钟无心跳
            health_score -= 50
        
        # 检查错误率
        if status['data_count'] > 0:
            error_rate = status['error_count'] / status['data_count']
            health_score -= error_rate * 100
        
        # 检查缓冲区使用率
        if device and device.data_buffer:
            buffer_usage = len(device.data_buffer) / device.max_buffer_size
            if buffer_usage > 0.9:
                health_score -= 20
        
        return {
            'device_id': device_id,
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score > 80 else 'warning' if health_score > 50 else 'critical',
            'last_heartbeat': status['last_heartbeat'],
            'data_count': status['data_count'],
            'error_count': status['error_count']
        }

class EdgeIntelligenceProcessor:
    """边缘智能处理器"""
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.logger = logging.getLogger(f"EdgeIntelligence.{processor_id}")
        
        # 机器学习模型
        self.ml_models = {}
        
        # 推理引擎
        self.inference_engine = None
        
        # 数据处理管道
        self.data_pipelines = {}
        
        # 性能监控
        self.performance_metrics = {
            'total_inferences': 0,
            'average_latency': 0.0,
            'accuracy': 0.0,
            'throughput': 0.0
        }
    
    def load_model(self, model_id: str, model_data: bytes, model_type: str = 'tensorflow'):
        """加载机器学习模型"""
        try:
            # 简化的模型加载
            self.ml_models[model_id] = {
                'model_data': model_data,
                'model_type': model_type,
                'load_time': datetime.now(),
                'status': 'loaded'
            }
            
            self.logger.info(f"Loaded model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def run_inference(self, model_id: str, input_data: np.ndarray) -> Dict[str, Any]:
        """运行推理"""
        if model_id not in self.ml_models:
            return {'error': 'Model not found'}
        
        start_time = time.time()
        
        try:
            # 简化的推理过程
            if self.ml_models[model_id]['model_type'] == 'load_forecast':
                result = self.run_load_forecast_inference(input_data)
            elif self.ml_models[model_id]['model_type'] == 'fault_detection':
                result = self.run_fault_detection_inference(input_data)
            elif self.ml_models[model_id]['model_type'] == 'optimization':
                result = self.run_optimization_inference(input_data)
            else:
                result = self.run_generic_inference(input_data)
            
            # 更新性能指标
            inference_time = time.time() - start_time
            self.update_performance_metrics(inference_time)
            
            return {
                'model_id': model_id,
                'result': result,
                'inference_time': inference_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return {'error': str(e)}
    
    def run_load_forecast_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """运行负荷预测推理"""
        # 简化的负荷预测模型
        forecast_horizon = 24
        forecast = []
        
        for hour in range(forecast_horizon):
            # 基于输入数据的预测
            base_prediction = np.mean(input_data) if len(input_data) > 0 else 1000
            time_factor = 1 + 0.2 * np.sin(2 * np.pi * hour / 24)
            random_factor = 1 + np.random.uniform(-0.1, 0.1)
            
            predicted_load = base_prediction * time_factor * random_factor
            forecast.append({
                'hour': hour,
                'load': max(0, predicted_load),
                'confidence': 0.8 + np.random.uniform(-0.1, 0.1)
            })
        
        return {
            'forecast': forecast,
            'method': 'edge_ml_inference',
            'input_features': len(input_data)
        }
    
    def run_fault_detection_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """运行故障检测推理"""
        # 简化的故障检测模型
        if len(input_data) == 0:
            return {'fault_detected': False, 'confidence': 0.0}
        
        # 计算异常分数
        mean_value = np.mean(input_data)
        std_value = np.std(input_data)
        
        # 检测异常
        anomaly_scores = np.abs(input_data - mean_value) / (std_value + 1e-6)
        max_anomaly = np.max(anomaly_scores)
        
        if max_anomaly > 3.0:  # 3倍标准差
            return {
                'fault_detected': True,
                'fault_type': 'anomaly',
                'severity': 'high' if max_anomaly > 5.0 else 'medium',
                'confidence': min(0.95, max_anomaly / 10),
                'anomaly_score': max_anomaly
            }
        else:
            return {
                'fault_detected': False,
                'confidence': 1.0 - max_anomaly / 10
            }
    
    def run_optimization_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """运行优化推理"""
        # 简化的优化模型
        if len(input_data) == 0:
            return {'optimization_result': 'insufficient_data'}
        
        # 计算优化建议
        current_load = np.mean(input_data)
        optimal_generation = current_load * 1.1  # 10%裕度
        cost_reduction = np.random.uniform(0.05, 0.15)
        
        return {
            'optimal_generation': optimal_generation,
            'cost_reduction': cost_reduction,
            'efficiency_improvement': cost_reduction,
            'recommendations': [
                'Adjust generation mix',
                'Optimize load distribution',
                'Activate demand response'
            ]
        }
    
    def run_generic_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """运行通用推理"""
        return {
            'result': 'generic_inference',
            'input_size': len(input_data),
            'processing_time': 0.1
        }
    
    def update_performance_metrics(self, inference_time: float):
        """更新性能指标"""
        self.performance_metrics['total_inferences'] += 1
        
        # 更新平均延迟
        current_avg = self.performance_metrics['average_latency']
        new_avg = (current_avg * (self.performance_metrics['total_inferences'] - 1) + inference_time) / self.performance_metrics['total_inferences']
        self.performance_metrics['average_latency'] = new_avg
        
        # 更新吞吐量
        self.performance_metrics['throughput'] = 1.0 / new_avg if new_avg > 0 else 0.0

class RealTimeDataProcessor:
    """实时数据处理器"""
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.logger = logging.getLogger(f"RealTimeDataProcessor.{processor_id}")
        
        # 数据流
        self.data_streams = {}
        
        # 处理管道
        self.processing_pipelines = {}
        
        # 数据缓存
        self.data_cache = {}
        
        # 性能统计
        self.performance_stats = {
            'total_processed': 0,
            'processing_rate': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
    
    def create_data_stream(self, stream_id: str, stream_config: Dict[str, Any]):
        """创建数据流"""
        self.data_streams[stream_id] = {
            'config': stream_config,
            'created_time': datetime.now(),
            'status': 'active',
            'data_count': 0,
            'last_data_time': None
        }
        
        self.logger.info(f"Created data stream: {stream_id}")
    
    def process_data_stream(self, stream_id: str, data: Dict[str, Any]):
        """处理数据流"""
        if stream_id not in self.data_streams:
            return
        
        start_time = time.time()
        
        try:
            # 数据预处理
            processed_data = self.preprocess_data(data)
            
            # 数据验证
            if not self.validate_data(processed_data):
                raise ValueError("Data validation failed")
            
            # 数据转换
            transformed_data = self.transform_data(processed_data)
            
            # 数据缓存
            self.cache_data(stream_id, transformed_data)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.update_performance_stats(processing_time)
            
            # 更新流状态
            self.data_streams[stream_id]['data_count'] += 1
            self.data_streams[stream_id]['last_data_time'] = datetime.now()
            
            self.logger.debug(f"Processed data for stream {stream_id}: {len(data)} fields")
            
        except Exception as e:
            self.logger.error(f"Data processing error: {e}")
            self.performance_stats['error_rate'] += 1
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据预处理"""
        processed_data = data.copy()
        
        # 添加处理时间戳
        processed_data['processed_timestamp'] = datetime.now()
        
        # 数据清洗
        for key, value in processed_data.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    processed_data[key] = None
        
        # 数据类型转换
        if 'voltage' in processed_data:
            processed_data['voltage'] = float(processed_data['voltage'])
        if 'current' in processed_data:
            processed_data['current'] = float(processed_data['current'])
        if 'power' in processed_data:
            processed_data['power'] = float(processed_data['power'])
        
        return processed_data
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """数据验证"""
        # 检查必需字段
        required_fields = ['timestamp', 'device_id']
        for field in required_fields:
            if field not in data:
                return False
        
        # 检查数值范围
        if 'voltage' in data and data['voltage'] is not None:
            if not (100 <= data['voltage'] <= 500):
                return False
        
        if 'current' in data and data['current'] is not None:
            if not (0 <= data['current'] <= 1000):
                return False
        
        return True
    
    def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据转换"""
        transformed_data = data.copy()
        
        # 计算派生值
        if 'voltage' in data and 'current' in data:
            if data['voltage'] is not None and data['current'] is not None:
                transformed_data['apparent_power'] = data['voltage'] * data['current']
        
        # 添加时间特征
        if 'timestamp' in data:
            timestamp = data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            transformed_data['hour'] = timestamp.hour
            transformed_data['day_of_week'] = timestamp.weekday()
            transformed_data['is_weekend'] = timestamp.weekday() >= 5
        
        return transformed_data
    
    def cache_data(self, stream_id: str, data: Dict[str, Any]):
        """缓存数据"""
        if stream_id not in self.data_cache:
            self.data_cache[stream_id] = []
        
        self.data_cache[stream_id].append(data)
        
        # 限制缓存大小
        max_cache_size = 1000
        if len(self.data_cache[stream_id]) > max_cache_size:
            self.data_cache[stream_id] = self.data_cache[stream_id][-max_cache_size:]
    
    def update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self.performance_stats['total_processed'] += 1
        
        # 更新处理速率
        if processing_time > 0:
            self.performance_stats['processing_rate'] = 1.0 / processing_time

class DistributedControlSystem:
    """分布式控制系统"""
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.logger = logging.getLogger(f"DistributedControl.{system_id}")
        
        # 控制节点
        self.control_nodes = {}
        
        # 控制策略
        self.control_strategies = {}
        
        # 控制状态
        self.control_states = {}
        
        # 协调机制
        self.coordination_mechanism = None
    
    def add_control_node(self, node_id: str, node_config: Dict[str, Any]):
        """添加控制节点"""
        self.control_nodes[node_id] = {
            'config': node_config,
            'status': 'active',
            'last_heartbeat': datetime.now(),
            'control_history': []
        }
        
        self.logger.info(f"Added control node: {node_id}")
    
    def execute_control_action(self, node_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制动作"""
        if node_id not in self.control_nodes:
            return {'error': 'Node not found'}
        
        start_time = time.time()
        
        try:
            # 验证控制动作
            if not self.validate_control_action(action):
                raise ValueError("Invalid control action")
            
            # 执行控制
            result = self.perform_control_action(node_id, action)
            
            # 记录控制历史
            control_record = {
                'action': action,
                'result': result,
                'timestamp': datetime.now(),
                'execution_time': time.time() - start_time
            }
            
            self.control_nodes[node_id]['control_history'].append(control_record)
            
            # 限制历史记录大小
            max_history = 1000
            if len(self.control_nodes[node_id]['control_history']) > max_history:
                self.control_nodes[node_id]['control_history'] = self.control_nodes[node_id]['control_history'][-max_history:]
            
            self.logger.info(f"Executed control action on node {node_id}: {action['type']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Control action error: {e}")
            return {'error': str(e)}
    
    def validate_control_action(self, action: Dict[str, Any]) -> bool:
        """验证控制动作"""
        required_fields = ['type', 'target', 'parameters']
        
        for field in required_fields:
            if field not in action:
                return False
        
        # 检查动作类型
        valid_types = ['voltage_control', 'frequency_control', 'load_shedding', 'generation_adjustment']
        if action['type'] not in valid_types:
            return False
        
        # 检查参数范围
        if 'voltage_setpoint' in action['parameters']:
            voltage = action['parameters']['voltage_setpoint']
            if not (200 <= voltage <= 400):
                return False
        
        return True
    
    def perform_control_action(self, node_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制动作"""
        action_type = action['type']
        
        if action_type == 'voltage_control':
            return self.perform_voltage_control(action)
        elif action_type == 'frequency_control':
            return self.perform_frequency_control(action)
        elif action_type == 'load_shedding':
            return self.perform_load_shedding(action)
        elif action_type == 'generation_adjustment':
            return self.perform_generation_adjustment(action)
        else:
            return {'error': 'Unknown action type'}
    
    def perform_voltage_control(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行电压控制"""
        target_voltage = action['parameters'].get('voltage_setpoint', 230)
        control_time = action['parameters'].get('control_time', 1.0)
        
        # 模拟电压控制
        current_voltage = 225 + np.random.uniform(-10, 10)
        voltage_error = target_voltage - current_voltage
        
        return {
            'action_type': 'voltage_control',
            'target_voltage': target_voltage,
            'current_voltage': current_voltage,
            'voltage_error': voltage_error,
            'control_success': abs(voltage_error) < 5,
            'control_time': control_time
        }
    
    def perform_frequency_control(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行频率控制"""
        target_frequency = action['parameters'].get('frequency_setpoint', 50.0)
        control_time = action['parameters'].get('control_time', 1.0)
        
        # 模拟频率控制
        current_frequency = 49.8 + np.random.uniform(-0.2, 0.2)
        frequency_error = target_frequency - current_frequency
        
        return {
            'action_type': 'frequency_control',
            'target_frequency': target_frequency,
            'current_frequency': current_frequency,
            'frequency_error': frequency_error,
            'control_success': abs(frequency_error) < 0.1,
            'control_time': control_time
        }
    
    def perform_load_shedding(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行负荷切除"""
        load_amount = action['parameters'].get('load_amount', 100)
        priority = action['parameters'].get('priority', 'low')
        
        # 模拟负荷切除
        actual_shed = load_amount * (0.9 + np.random.uniform(-0.1, 0.1))
        
        return {
            'action_type': 'load_shedding',
            'target_load': load_amount,
            'actual_shed': actual_shed,
            'shedding_efficiency': actual_shed / load_amount,
            'priority': priority
        }
    
    def perform_generation_adjustment(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行发电调整"""
        adjustment_amount = action['parameters'].get('adjustment_amount', 50)
        generator_id = action['parameters'].get('generator_id', 'gen_001')
        
        # 模拟发电调整
        actual_adjustment = adjustment_amount * (0.95 + np.random.uniform(-0.05, 0.05))
        
        return {
            'action_type': 'generation_adjustment',
            'target_adjustment': adjustment_amount,
            'actual_adjustment': actual_adjustment,
            'adjustment_efficiency': actual_adjustment / adjustment_amount,
            'generator_id': generator_id
        }

# 使用示例
def edge_computing_iot_example():
    print("=== 边缘计算与物联网在电力系统中的应用示例 ===")
    
    # 1. 边缘计算架构
    print("\n1. 边缘计算架构")
    edge_arch = EdgeComputingArchitecture("power_system_edge")
    
    # 添加边缘节点
    node1 = EdgeNode("node_001", "substation_1", 8.0, 16.0, 1000.0, 1000.0, [], "active")
    node2 = EdgeNode("node_002", "substation_2", 4.0, 8.0, 500.0, 500.0, [], "active")
    
    edge_arch.add_edge_node(node1)
    edge_arch.add_edge_node(node2)
    
    # 添加边缘设备
    device1 = EdgeDevice("sensor_001", "sensor", "substation_1", ["voltage", "current"], "online", datetime.now(), [])
    device2 = EdgeDevice("controller_001", "controller", "substation_1", ["voltage_control", "frequency_control"], "online", datetime.now(), [])
    
    edge_arch.add_edge_device(device1)
    edge_arch.add_edge_device(device2)
    
    # 连接设备到节点
    edge_arch.connect_device_to_node("sensor_001", "node_001")
    edge_arch.connect_device_to_node("controller_001", "node_001")
    
    # 2. 物联网设备管理
    print("\n2. 物联网设备管理")
    iot_manager = IoTDeviceManager("power_iot_manager")
    
    # 注册设备
    iot_manager.register_device(device1)
    iot_manager.register_device(device2)
    
    # 收集设备数据
    sensor_data = {
        'voltage': 230.5,
        'current': 100.2,
        'power': 23100.1,
        'frequency': 50.0,
        'temperature': 45.2
    }
    
    iot_manager.collect_device_data("sensor_001", sensor_data)
    
    # 获取设备健康状态
    health = iot_manager.get_device_health("sensor_001")
    print(f"设备健康状态: {health}")
    
    # 3. 边缘智能处理
    print("\n3. 边缘智能处理")
    edge_processor = EdgeIntelligenceProcessor("edge_ml_processor")
    
    # 模拟模型加载
    model_data = b"simulated_model_data"
    edge_processor.load_model("load_forecast_model", model_data, "load_forecast")
    
    # 运行推理
    input_data = np.array([1000, 1100, 1200, 1150, 1050])
    inference_result = edge_processor.run_inference("load_forecast_model", input_data)
    print(f"推理结果: {inference_result}")
    
    # 4. 实时数据处理
    print("\n4. 实时数据处理")
    data_processor = RealTimeDataProcessor("realtime_processor")
    
    # 创建数据流
    stream_config = {
        'data_type': 'power_measurements',
        'sampling_rate': 1.0,  # Hz
        'buffer_size': 1000
    }
    data_processor.create_data_stream("power_stream_001", stream_config)
    
    # 处理数据
    realtime_data = {
        'timestamp': datetime.now(),
        'device_id': 'sensor_001',
        'voltage': 230.5,
        'current': 100.2,
        'power': 23100.1
    }
    
    data_processor.process_data_stream("power_stream_001", realtime_data)
    
    # 5. 分布式控制
    print("\n5. 分布式控制")
    control_system = DistributedControlSystem("distributed_control")
    
    # 添加控制节点
    node_config = {
        'control_type': 'voltage_frequency',
        'control_range': {'voltage': [200, 400], 'frequency': [49.5, 50.5]},
        'response_time': 0.1
    }
    control_system.add_control_node("control_node_001", node_config)
    
    # 执行控制动作
    control_action = {
        'type': 'voltage_control',
        'target': 'substation_1',
        'parameters': {
            'voltage_setpoint': 235.0,
            'control_time': 2.0
        }
    }
    
    control_result = control_system.execute_control_action("control_node_001", control_action)
    print(f"控制结果: {control_result}")

if __name__ == "__main__":
    edge_computing_iot_example()
```

---

*最后更新: 2025-01-01*
*版本: 1.0.0*
