# 区块链电力交易系统实现 / Blockchain Power Trading System Implementation

## 目录 / Table of Contents

- [区块链电力交易系统实现 / Blockchain Power Trading System Implementation](#区块链电力交易系统实现--blockchain-power-trading-system-implementation)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.3.37 区块链电力交易架构 / Blockchain Power Trading Architecture](#8337-区块链电力交易架构--blockchain-power-trading-architecture)
    - [区块链电力交易系统架构 / Blockchain Power Trading System Architecture](#区块链电力交易系统架构--blockchain-power-trading-system-architecture)

---

## 8.3.37 区块链电力交易架构 / Blockchain Power Trading Architecture

### 区块链电力交易系统架构 / Blockchain Power Trading System Architecture

```python
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class PowerTransaction:
    """电力交易数据结构"""
    transaction_id: str
    seller_id: str
    buyer_id: str
    power_amount: float  # kWh
    price_per_kwh: float  # 元/kWh
    timestamp: datetime
    transaction_type: str  # 'spot', 'forward', 'reserve'
    energy_source: str  # 'solar', 'wind', 'hydro', 'thermal'
    location: str
    status: str  # 'pending', 'confirmed', 'completed', 'cancelled'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def calculate_hash(self) -> str:
        """计算交易哈希"""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

@dataclass
class EnergyCertificate:
    """能源证书数据结构"""
    certificate_id: str
    producer_id: str
    energy_source: str
    power_amount: float
    generation_time: datetime
    location: str
    carbon_intensity: float  # gCO2/kWh
    certificate_type: str  # 'green', 'renewable', 'carbon_offset'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['generation_time'] = self.generation_time.isoformat()
        return data

class BlockchainNode:
    """区块链节点"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f"BlockchainNode.{node_id}")
        
        # 区块链数据
        self.blocks = []
        self.pending_transactions = []
        self.energy_certificates = []
        
        # 网络连接
        self.peers = set()
        
        # 共识参数
        self.difficulty = 4  # 挖矿难度
        self.block_reward = 10  # 区块奖励
        
        # 创建创世区块
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """创建创世区块"""
        genesis_block = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'transactions': [],
            'previous_hash': '0' * 64,
            'nonce': 0,
            'hash': self.calculate_block_hash({
                'index': 0,
                'timestamp': datetime.now().isoformat(),
                'transactions': [],
                'previous_hash': '0' * 64,
                'nonce': 0
            })
        }
        self.blocks.append(genesis_block)
        self.logger.info("Created genesis block")
    
    def calculate_block_hash(self, block_data: Dict[str, Any]) -> str:
        """计算区块哈希"""
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self) -> Optional[Dict[str, Any]]:
        """挖矿"""
        if not self.pending_transactions:
            return None
        
        last_block = self.blocks[-1]
        new_block = {
            'index': last_block['index'] + 1,
            'timestamp': datetime.now().isoformat(),
            'transactions': self.pending_transactions[:10],  # 限制每块交易数
            'previous_hash': last_block['hash'],
            'nonce': 0
        }
        
        # 工作量证明
        while True:
            new_block['nonce'] += 1
            block_hash = self.calculate_block_hash(new_block)
            
            if block_hash[:self.difficulty] == '0' * self.difficulty:
                new_block['hash'] = block_hash
                break
        
        # 添加区块奖励交易
        reward_transaction = {
            'transaction_id': f"reward_{new_block['index']}",
            'seller_id': 'system',
            'buyer_id': self.node_id,
            'power_amount': 0,
            'price_per_kwh': 0,
            'timestamp': datetime.now(),
            'transaction_type': 'reward',
            'energy_source': 'system',
            'location': 'system',
            'status': 'confirmed'
        }
        new_block['transactions'].append(reward_transaction)
        
        return new_block
    
    def add_block(self, block: Dict[str, Any]) -> bool:
        """添加新区块"""
        # 验证区块
        if not self.validate_block(block):
            return False
        
        self.blocks.append(block)
        
        # 更新待处理交易
        for transaction in block['transactions']:
            if transaction['transaction_id'] in [t['transaction_id'] for t in self.pending_transactions]:
                self.pending_transactions = [t for t in self.pending_transactions 
                                          if t['transaction_id'] != transaction['transaction_id']]
        
        self.logger.info(f"Added block {block['index']}")
        return True
    
    def validate_block(self, block: Dict[str, Any]) -> bool:
        """验证区块"""
        # 检查索引
        if block['index'] != len(self.blocks):
            return False
        
        # 检查前一个区块哈希
        if block['previous_hash'] != self.blocks[-1]['hash']:
            return False
        
        # 检查工作量证明
        block_hash = self.calculate_block_hash(block)
        if block_hash[:self.difficulty] != '0' * self.difficulty:
            return False
        
        return True
    
    def add_transaction(self, transaction: PowerTransaction):
        """添加交易"""
        # 验证交易
        if not self.validate_transaction(transaction):
            return False
        
        self.pending_transactions.append(transaction.to_dict())
        self.logger.info(f"Added transaction {transaction.transaction_id}")
        return True
    
    def validate_transaction(self, transaction: PowerTransaction) -> bool:
        """验证交易"""
        # 检查基本字段
        if not all([transaction.seller_id, transaction.buyer_id, 
                   transaction.power_amount > 0, transaction.price_per_kwh >= 0]):
            return False
        
        # 检查交易ID唯一性
        existing_ids = [t['transaction_id'] for t in self.pending_transactions]
        if transaction.transaction_id in existing_ids:
            return False
        
        return True
    
    def get_balance(self, user_id: str) -> Dict[str, float]:
        """获取用户余额"""
        balance = {'power': 0.0, 'money': 0.0}
        
        for block in self.blocks:
            for transaction in block['transactions']:
                if transaction['buyer_id'] == user_id:
                    balance['power'] += transaction['power_amount']
                    balance['money'] -= transaction['power_amount'] * transaction['price_per_kwh']
                elif transaction['seller_id'] == user_id:
                    balance['power'] -= transaction['power_amount']
                    balance['money'] += transaction['power_amount'] * transaction['price_per_kwh']
        
        return balance

class PowerTradingMarket:
    """电力交易市场"""
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.logger = logging.getLogger(f"PowerTradingMarket.{market_id}")
        
        # 市场参与者
        self.participants = {}  # user_id -> participant_info
        
        # 订单簿
        self.buy_orders = []  # 买单
        self.sell_orders = []  # 卖单
        
        # 交易历史
        self.trade_history = []
        
        # 市场参数
        self.market_params = {
            'min_order_size': 1.0,  # kWh
            'max_order_size': 10000.0,  # kWh
            'price_tick': 0.01,  # 元
            'trading_hours': (8, 22)  # 交易时间
        }
    
    def register_participant(self, user_id: str, participant_info: Dict[str, Any]):
        """注册市场参与者"""
        self.participants[user_id] = {
            'user_id': user_id,
            'type': participant_info.get('type', 'consumer'),  # producer, consumer, prosumer
            'location': participant_info.get('location', ''),
            'capacity': participant_info.get('capacity', 0.0),
            'registration_time': datetime.now(),
            'status': 'active'
        }
        self.logger.info(f"Registered participant: {user_id}")
    
    def place_buy_order(self, user_id: str, amount: float, price: float, 
                       order_type: str = 'limit') -> Optional[str]:
        """下买单"""
        if user_id not in self.participants:
            return None
        
        if not self.validate_order(amount, price):
            return None
        
        order_id = f"buy_{int(time.time() * 1000)}"
        buy_order = {
            'order_id': order_id,
            'user_id': user_id,
            'amount': amount,
            'price': price,
            'order_type': order_type,
            'timestamp': datetime.now(),
            'status': 'active'
        }
        
        self.buy_orders.append(buy_order)
        self.buy_orders.sort(key=lambda x: x['price'], reverse=True)  # 价格优先
        
        self.logger.info(f"Placed buy order: {order_id}")
        return order_id
    
    def place_sell_order(self, user_id: str, amount: float, price: float,
                        order_type: str = 'limit') -> Optional[str]:
        """下卖单"""
        if user_id not in self.participants:
            return None
        
        if not self.validate_order(amount, price):
            return None
        
        order_id = f"sell_{int(time.time() * 1000)}"
        sell_order = {
            'order_id': order_id,
            'user_id': user_id,
            'amount': amount,
            'price': price,
            'order_type': order_type,
            'timestamp': datetime.now(),
            'status': 'active'
        }
        
        self.sell_orders.append(sell_order)
        self.sell_orders.sort(key=lambda x: x['price'])  # 价格优先
        
        self.logger.info(f"Placed sell order: {order_id}")
        return order_id
    
    def validate_order(self, amount: float, price: float) -> bool:
        """验证订单"""
        if amount < self.market_params['min_order_size']:
            return False
        if amount > self.market_params['max_order_size']:
            return False
        if price < 0:
            return False
        
        return True
    
    def match_orders(self) -> List[Dict[str, Any]]:
        """撮合订单"""
        trades = []
        
        while self.buy_orders and self.sell_orders:
            buy_order = self.buy_orders[0]
            sell_order = self.sell_orders[0]
            
            # 检查是否可以成交
            if buy_order['price'] >= sell_order['price']:
                # 确定成交数量和价格
                trade_amount = min(buy_order['amount'], sell_order['amount'])
                trade_price = (buy_order['price'] + sell_order['price']) / 2
                
                # 创建交易
                trade = {
                    'trade_id': f"trade_{int(time.time() * 1000)}",
                    'buy_order_id': buy_order['order_id'],
                    'sell_order_id': sell_order['order_id'],
                    'buyer_id': buy_order['user_id'],
                    'seller_id': sell_order['user_id'],
                    'amount': trade_amount,
                    'price': trade_price,
                    'timestamp': datetime.now()
                }
                
                trades.append(trade)
                self.trade_history.append(trade)
                
                # 更新订单
                buy_order['amount'] -= trade_amount
                sell_order['amount'] -= trade_amount
                
                # 移除已完成的订单
                if buy_order['amount'] <= 0:
                    self.buy_orders.pop(0)
                if sell_order['amount'] <= 0:
                    self.sell_orders.pop(0)
                
                self.logger.info(f"Matched trade: {trade['trade_id']}")
            else:
                break
        
        return trades
    
    def get_market_depth(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取市场深度"""
        buy_depth = {}
        sell_depth = {}
        
        # 统计买单深度
        for order in self.buy_orders:
            price = order['price']
            if price not in buy_depth:
                buy_depth[price] = 0
            buy_depth[price] += order['amount']
        
        # 统计卖单深度
        for order in self.sell_orders:
            price = order['price']
            if price not in sell_depth:
                sell_depth[price] = 0
            sell_depth[price] += order['amount']
        
        return {
            'buy_depth': [{'price': p, 'amount': a} for p, a in sorted(buy_depth.items(), reverse=True)],
            'sell_depth': [{'price': p, 'amount': a} for p, a in sorted(sell_depth.items())]
        }
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """获取市场统计"""
        if not self.trade_history:
            return {}
        
        prices = [trade['price'] for trade in self.trade_history]
        amounts = [trade['amount'] for trade in self.trade_history]
        
        return {
            'total_trades': len(self.trade_history),
            'total_volume': sum(amounts),
            'average_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'active_buy_orders': len(self.buy_orders),
            'active_sell_orders': len(self.sell_orders)
        }

class SmartContract:
    """智能合约基类"""
    def __init__(self, contract_id: str, contract_type: str):
        self.contract_id = contract_id
        self.contract_type = contract_type
        self.logger = logging.getLogger(f"SmartContract.{contract_id}")
        
        # 合约状态
        self.state = {}
        self.conditions = []
        self.actions = []
        
        # 合约参数
        self.parameters = {}
        
    def add_condition(self, condition_func):
        """添加条件"""
        self.conditions.append(condition_func)
    
    def add_action(self, action_func):
        """添加动作"""
        self.actions.append(action_func)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能合约"""
        try:
            # 检查条件
            for condition in self.conditions:
                if not condition(input_data):
                    return {'success': False, 'error': 'Condition not met'}
            
            # 执行动作
            results = []
            for action in self.actions:
                result = action(input_data)
                results.append(result)
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            self.logger.error(f"Contract execution error: {e}")
            return {'success': False, 'error': str(e)}

class PowerPurchaseAgreement(SmartContract):
    """电力购买协议智能合约"""
    def __init__(self, contract_id: str, seller_id: str, buyer_id: str, 
                 total_amount: float, price: float, duration_days: int):
        super().__init__(contract_id, "PowerPurchaseAgreement")
        
        self.seller_id = seller_id
        self.buyer_id = buyer_id
        self.total_amount = total_amount
        self.price = price
        self.duration_days = duration_days
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=duration_days)
        
        # 合约状态
        self.state = {
            'delivered_amount': 0.0,
            'paid_amount': 0.0,
            'last_delivery': None,
            'last_payment': None,
            'status': 'active'
        }
        
        # 设置条件
        self.setup_conditions()
        
        # 设置动作
        self.setup_actions()
    
    def setup_conditions(self):
        """设置合约条件"""
        def delivery_condition(data):
            """交付条件"""
            delivery_amount = data.get('delivery_amount', 0)
            return delivery_amount > 0 and delivery_amount <= self.total_amount - self.state['delivered_amount']
        
        def payment_condition(data):
            """付款条件"""
            payment_amount = data.get('payment_amount', 0)
            expected_payment = self.state['delivered_amount'] * self.price
            return payment_amount >= expected_payment - self.state['paid_amount']
        
        self.add_condition(delivery_condition)
        self.add_condition(payment_condition)
    
    def setup_actions(self):
        """设置合约动作"""
        def delivery_action(data):
            """交付动作"""
            delivery_amount = data.get('delivery_amount', 0)
            self.state['delivered_amount'] += delivery_amount
            self.state['last_delivery'] = datetime.now()
            
            if self.state['delivered_amount'] >= self.total_amount:
                self.state['status'] = 'completed'
            
            return {'action': 'delivery', 'amount': delivery_amount}
        
        def payment_action(data):
            """付款动作"""
            payment_amount = data.get('payment_amount', 0)
            self.state['paid_amount'] += payment_amount
            self.state['last_payment'] = datetime.now()
            
            return {'action': 'payment', 'amount': payment_amount}
        
        self.add_action(delivery_action)
        self.add_action(payment_action)
    
    def get_contract_status(self) -> Dict[str, Any]:
        """获取合约状态"""
        progress = self.state['delivered_amount'] / self.total_amount * 100
        days_remaining = (self.end_date - datetime.now()).days
        
        return {
            'contract_id': self.contract_id,
            'seller_id': self.seller_id,
            'buyer_id': self.buyer_id,
            'total_amount': self.total_amount,
            'delivered_amount': self.state['delivered_amount'],
            'progress': progress,
            'price': self.price,
            'days_remaining': max(0, days_remaining),
            'status': self.state['status']
        }

class EnergyTraceabilitySystem:
    """能源溯源系统"""
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.logger = logging.getLogger(f"EnergyTraceability.{system_id}")
        
        # 溯源链
        self.traceability_chain = []
        
        # 能源证书
        self.energy_certificates = {}
        
        # 溯源节点
        self.traceability_nodes = {}
    
    def create_energy_certificate(self, producer_id: str, energy_source: str,
                                power_amount: float, generation_time: datetime,
                                location: str, carbon_intensity: float) -> str:
        """创建能源证书"""
        certificate_id = f"cert_{int(time.time() * 1000)}"
        
        certificate = EnergyCertificate(
            certificate_id=certificate_id,
            producer_id=producer_id,
            energy_source=energy_source,
            power_amount=power_amount,
            generation_time=generation_time,
            location=location,
            carbon_intensity=carbon_intensity,
            certificate_type='green' if carbon_intensity < 100 else 'renewable'
        )
        
        self.energy_certificates[certificate_id] = certificate
        
        # 添加到溯源链
        trace_entry = {
            'certificate_id': certificate_id,
            'action': 'created',
            'timestamp': datetime.now(),
            'details': certificate.to_dict()
        }
        self.traceability_chain.append(trace_entry)
        
        self.logger.info(f"Created energy certificate: {certificate_id}")
        return certificate_id
    
    def transfer_certificate(self, certificate_id: str, from_user: str, 
                           to_user: str, amount: float) -> bool:
        """转移能源证书"""
        if certificate_id not in self.energy_certificates:
            return False
        
        certificate = self.energy_certificates[certificate_id]
        
        if certificate.power_amount < amount:
            return False
        
        # 更新证书
        certificate.power_amount -= amount
        
        # 创建新证书给接收方
        new_certificate_id = f"cert_{int(time.time() * 1000)}"
        new_certificate = EnergyCertificate(
            certificate_id=new_certificate_id,
            producer_id=to_user,
            energy_source=certificate.energy_source,
            power_amount=amount,
            generation_time=certificate.generation_time,
            location=certificate.location,
            carbon_intensity=certificate.carbon_intensity,
            certificate_type=certificate.certificate_type
        )
        
        self.energy_certificates[new_certificate_id] = new_certificate
        
        # 添加到溯源链
        trace_entry = {
            'certificate_id': certificate_id,
            'action': 'transferred',
            'from_user': from_user,
            'to_user': to_user,
            'amount': amount,
            'timestamp': datetime.now(),
            'new_certificate_id': new_certificate_id
        }
        self.traceability_chain.append(trace_entry)
        
        self.logger.info(f"Transferred certificate {certificate_id} to {to_user}")
        return True
    
    def get_certificate_trace(self, certificate_id: str) -> List[Dict[str, Any]]:
        """获取证书溯源信息"""
        trace = []
        
        for entry in self.traceability_chain:
            if entry.get('certificate_id') == certificate_id:
                trace.append(entry)
            elif entry.get('new_certificate_id') == certificate_id:
                trace.append(entry)
        
        return trace
    
    def verify_certificate(self, certificate_id: str) -> Dict[str, Any]:
        """验证证书"""
        if certificate_id not in self.energy_certificates:
            return {'valid': False, 'error': 'Certificate not found'}
        
        certificate = self.energy_certificates[certificate_id]
        trace = self.get_certificate_trace(certificate_id)
        
        verification_result = {
            'valid': True,
            'certificate_id': certificate_id,
            'energy_source': certificate.energy_source,
            'carbon_intensity': certificate.carbon_intensity,
            'certificate_type': certificate.certificate_type,
            'trace_length': len(trace),
            'creation_time': certificate.generation_time,
            'location': certificate.location
        }
        
        return verification_result

# 使用示例
def blockchain_power_trading_example():
    # 创建区块链节点
    node1 = BlockchainNode("node_001")
    node2 = BlockchainNode("node_002")
    
    # 创建电力交易市场
    market = PowerTradingMarket("market_001")
    
    # 注册参与者
    market.register_participant("producer_001", {
        'type': 'producer',
        'location': 'Beijing',
        'capacity': 1000.0
    })
    
    market.register_participant("consumer_001", {
        'type': 'consumer',
        'location': 'Shanghai',
        'capacity': 500.0
    })
    
    # 下订单
    market.place_sell_order("producer_001", 100.0, 0.5)
    market.place_buy_order("consumer_001", 50.0, 0.6)
    
    # 撮合订单
    trades = market.match_orders()
    print(f"撮合交易: {trades}")
    
    # 创建智能合约
    ppa_contract = PowerPurchaseAgreement(
        "contract_001", "producer_001", "consumer_001", 
        1000.0, 0.5, 30
    )
    
    # 执行合约
    result = ppa_contract.execute({'delivery_amount': 100.0})
    print(f"合约执行结果: {result}")
    
    # 创建能源溯源系统
    traceability = EnergyTraceabilitySystem("trace_001")
    
    # 创建能源证书
    cert_id = traceability.create_energy_certificate(
        "producer_001", "solar", 100.0, 
        datetime.now(), "Beijing", 50.0
    )
    
    # 验证证书
    verification = traceability.verify_certificate(cert_id)
    print(f"证书验证结果: {verification}")

if __name__ == "__main__":
    blockchain_power_trading_example()
```

---

*最后更新: 2025-01-01*
*版本: 1.0.0*
