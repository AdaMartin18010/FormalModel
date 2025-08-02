#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形式化模型项目演示脚本 / Formal Model Project Demo Script

展示项目核心功能和实现示例
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import json
import time

class FormalModelDemo:
    """形式化模型演示类"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def demo_classical_mechanics(self):
        """经典力学模型演示"""
        print("🔬 经典力学模型演示")
        print("=" * 50)
        
        # 简谐运动
        def harmonic_motion(t: float, A: float, omega: float, phi: float) -> float:
            """简谐运动方程: x(t) = A * cos(ωt + φ)"""
            return A * math.cos(omega * t + phi)
        
        # 参数
        A = 1.0  # 振幅
        omega = 2.0  # 角频率
        phi = 0.0  # 初相位
        
        # 计算运动轨迹
        t_values = np.linspace(0, 4*math.pi, 100)
        x_values = [harmonic_motion(t, A, omega, phi) for t in t_values]
        
        # 绘制图形
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, x_values, 'b-', linewidth=2, label='简谐运动')
        plt.xlabel('时间 t')
        plt.ylabel('位置 x')
        plt.title('简谐运动演示')
        plt.grid(True)
        plt.legend()
        plt.savefig('harmonic_motion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 简谐运动演示完成，图像保存为 harmonic_motion.png")
        
        # 能量计算
        def kinetic_energy(m: float, v: float) -> float:
            """动能: E_k = 1/2 * m * v^2"""
            return 0.5 * m * v**2
        
        def potential_energy(k: float, x: float) -> float:
            """势能: E_p = 1/2 * k * x^2"""
            return 0.5 * k * x**2
        
        m = 1.0  # 质量
        k = 4.0  # 弹簧常数
        v_max = A * omega  # 最大速度
        
        E_k = kinetic_energy(m, v_max)
        E_p = potential_energy(k, A)
        E_total = E_k + E_p
        
        print(f"📊 能量分析:")
        print(f"   动能: {E_k:.2f} J")
        print(f"   势能: {E_p:.2f} J")
        print(f"   总能量: {E_total:.2f} J")
        
        return {
            'motion_type': 'harmonic',
            'parameters': {'A': A, 'omega': omega, 'phi': phi},
            'energy': {'kinetic': E_k, 'potential': E_p, 'total': E_total}
        }
    
    def demo_quantum_mechanics(self):
        """量子力学模型演示"""
        print("\n⚛️ 量子力学模型演示")
        print("=" * 50)
        
        # 波函数
        def wave_function(x: float, k: float, t: float) -> complex:
            """平面波函数: ψ(x,t) = e^(i(kx - ωt))"""
            omega = k**2 / 2  # 自由粒子
            return complex(math.cos(k*x - omega*t), math.sin(k*x - omega*t))
        
        # 概率密度
        def probability_density(psi: complex) -> float:
            """概率密度: |ψ|^2"""
            return abs(psi)**2
        
        # 参数
        k = 2.0  # 波数
        x_values = np.linspace(-5, 5, 200)
        t = 0.0  # 时间
        
        # 计算波函数和概率密度
        psi_values = [wave_function(x, k, t) for x in x_values]
        prob_density = [probability_density(psi) for psi in psi_values]
        
        # 绘制图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 波函数实部和虚部
        real_part = [psi.real for psi in psi_values]
        imag_part = [psi.imag for psi in psi_values]
        
        ax1.plot(x_values, real_part, 'b-', label='实部', linewidth=2)
        ax1.plot(x_values, imag_part, 'r-', label='虚部', linewidth=2)
        ax1.set_xlabel('位置 x')
        ax1.set_ylabel('波函数 ψ')
        ax1.set_title('平面波函数')
        ax1.grid(True)
        ax1.legend()
        
        # 概率密度
        ax2.plot(x_values, prob_density, 'g-', linewidth=2)
        ax2.set_xlabel('位置 x')
        ax2.set_ylabel('概率密度 |ψ|²')
        ax2.set_title('概率密度分布')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('quantum_wave.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 量子波函数演示完成，图像保存为 quantum_wave.png")
        
        # 归一化检查
        integral = np.trapz(prob_density, x_values)
        print(f"📊 归一化检查: ∫|ψ|²dx = {integral:.4f}")
        
        return {
            'wave_type': 'plane_wave',
            'parameters': {'k': k, 't': t},
            'normalization': integral
        }
    
    def demo_optimization(self):
        """优化模型演示"""
        print("\n⚙️ 优化模型演示")
        print("=" * 50)
        
        # 目标函数: f(x,y) = x² + y²
        def objective_function(x: float, y: float) -> float:
            return x**2 + y**2
        
        # 梯度下降优化
        def gradient_descent(start_point: Tuple[float, float], 
                           learning_rate: float, 
                           max_iterations: int) -> List[Tuple[float, float]]:
            """梯度下降算法"""
            x, y = start_point
            path = [(x, y)]
            
            for i in range(max_iterations):
                # 计算梯度
                grad_x = 2 * x
                grad_y = 2 * y
                
                # 更新参数
                x -= learning_rate * grad_x
                y -= learning_rate * grad_y
                
                path.append((x, y))
                
                # 收敛检查
                if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
                    break
            
            return path
        
        # 优化参数
        start_point = (2.0, 2.0)
        learning_rate = 0.1
        max_iterations = 100
        
        # 执行优化
        optimization_path = gradient_descent(start_point, learning_rate, max_iterations)
        
        # 绘制优化过程
        x_path = [point[0] for point in optimization_path]
        y_path = [point[1] for point in optimization_path]
        z_path = [objective_function(x, y) for x, y in optimization_path]
        
        # 创建等高线图
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = objective_function(X, Y)
        
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=20, colors='gray', alpha=0.5)
        plt.plot(x_path, y_path, 'ro-', linewidth=2, markersize=4, label='优化路径')
        plt.plot(x_path[0], y_path[0], 'go', markersize=8, label='起始点')
        plt.plot(x_path[-1], y_path[-1], 'bo', markersize=8, label='最优点')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('梯度下降优化演示')
        plt.grid(True)
        plt.legend()
        plt.savefig('optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 优化算法演示完成，图像保存为 optimization.png")
        print(f"📊 优化结果:")
        print(f"   起始点: ({start_point[0]:.4f}, {start_point[1]:.4f})")
        print(f"   最优点: ({x_path[-1]:.4f}, {y_path[-1]:.4f})")
        print(f"   最优值: {z_path[-1]:.6f}")
        print(f"   迭代次数: {len(optimization_path)}")
        
        return {
            'algorithm': 'gradient_descent',
            'start_point': start_point,
            'optimal_point': (x_path[-1], y_path[-1]),
            'optimal_value': z_path[-1],
            'iterations': len(optimization_path)
        }
    
    def demo_machine_learning(self):
        """机器学习模型演示"""
        print("\n🤖 机器学习模型演示")
        print("=" * 50)
        
        # 生成数据
        np.random.seed(42)
        n_samples = 100
        
        # 线性关系: y = 2x + 1 + noise
        x = np.random.uniform(0, 10, n_samples)
        y_true = 2 * x + 1
        y_noisy = y_true + np.random.normal(0, 0.5, n_samples)
        
        # 线性回归
        def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            """最小二乘线性回归"""
            n = len(x)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # 计算回归系数
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            return slope, intercept
        
        # 训练模型
        slope, intercept = linear_regression(x, y_noisy)
        y_pred = slope * x + intercept
        
        # 计算R²
        ss_res = np.sum((y_noisy - y_pred) ** 2)
        ss_tot = np.sum((y_noisy - np.mean(y_noisy)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y_noisy, alpha=0.6, label='数据点')
        plt.plot(x, y_true, 'g-', linewidth=2, label='真实关系')
        plt.plot(x, y_pred, 'r-', linewidth=2, label='预测关系')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('线性回归演示')
        plt.grid(True)
        plt.legend()
        plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 机器学习演示完成，图像保存为 linear_regression.png")
        print(f"📊 模型结果:")
        print(f"   斜率: {slope:.4f}")
        print(f"   截距: {intercept:.4f}")
        print(f"   R²: {r_squared:.4f}")
        
        return {
            'model_type': 'linear_regression',
            'parameters': {'slope': slope, 'intercept': intercept},
            'performance': {'r_squared': r_squared}
        }
    
    def demo_financial_model(self):
        """金融模型演示"""
        print("\n💰 金融模型演示")
        print("=" * 50)
        
        # 蒙特卡洛模拟期权定价
        def monte_carlo_option_pricing(S0: float, K: float, T: float, 
                                     r: float, sigma: float, n_simulations: int) -> float:
            """蒙特卡洛期权定价"""
            np.random.seed(42)
            
            # 生成随机路径
            dt = T / 252  # 日时间步长
            n_steps = int(T * 252)
            
            payoffs = []
            for _ in range(n_simulations):
                # 生成价格路径
                S = S0
                for _ in range(n_steps):
                    dW = np.random.normal(0, np.sqrt(dt))
                    S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
                
                # 计算期权收益
                payoff = max(S - K, 0)
                payoffs.append(payoff)
            
            # 计算期权价格
            option_price = np.exp(-r * T) * np.mean(payoffs)
            return option_price
        
        # 参数
        S0 = 100.0  # 当前股价
        K = 100.0   # 执行价格
        T = 1.0     # 到期时间（年）
        r = 0.05    # 无风险利率
        sigma = 0.2 # 波动率
        n_simulations = 10000
        
        # 计算期权价格
        option_price = monte_carlo_option_pricing(S0, K, T, r, sigma, n_simulations)
        
        # 风险分析
        def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
            """计算VaR (Value at Risk)"""
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        # 生成收益率分布
        returns = np.random.normal(0.05, 0.2, 10000)  # 年化收益率
        var_95 = calculate_var(returns, 0.95)
        var_99 = calculate_var(returns, 0.99)
        
        # 绘制收益率分布
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(var_95, color='red', linestyle='--', label=f'VaR(95%): {var_95:.3f}')
        plt.axvline(var_99, color='orange', linestyle='--', label=f'VaR(99%): {var_99:.3f}')
        plt.xlabel('收益率')
        plt.ylabel('频次')
        plt.title('收益率分布与VaR')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # 期权价格随股价变化
        stock_prices = np.linspace(80, 120, 50)
        option_prices = [monte_carlo_option_pricing(S, K, T, r, sigma, 1000) 
                        for S in stock_prices]
        
        plt.plot(stock_prices, option_prices, 'b-', linewidth=2)
        plt.axvline(S0, color='red', linestyle='--', label=f'当前股价: {S0}')
        plt.xlabel('股价')
        plt.ylabel('期权价格')
        plt.title('期权价格曲线')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('financial_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 金融模型演示完成，图像保存为 financial_model.png")
        print(f"📊 期权定价结果:")
        print(f"   期权价格: {option_price:.4f}")
        print(f"   VaR(95%): {var_95:.4f}")
        print(f"   VaR(99%): {var_99:.4f}")
        
        return {
            'model_type': 'option_pricing',
            'option_price': option_price,
            'risk_metrics': {'var_95': var_95, 'var_99': var_99}
        }
    
    def demo_network_model(self):
        """网络模型演示"""
        print("\n🌐 网络模型演示")
        print("=" * 50)
        
        import networkx as nx
        
        # 创建小世界网络
        n_nodes = 50
        k = 4  # 平均度数
        p = 0.1  # 重连概率
        
        G = nx.watts_strogatz_graph(n_nodes, k, p)
        
        # 计算网络统计量
        avg_degree = np.mean([d for n, d in G.degree()])
        clustering_coef = nx.average_clustering(G)
        avg_path_length = nx.average_shortest_path_length(G)
        
        # 度分布
        degrees = [d for n, d in G.degree()]
        
        # 绘制网络
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.7)
        plt.title('小世界网络结构')
        
        plt.subplot(1, 3, 2)
        plt.hist(degrees, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('度数')
        plt.ylabel('频次')
        plt.title('度分布')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        # 网络演化
        evolution_steps = 10
        clustering_evolution = []
        path_length_evolution = []
        
        for step in range(evolution_steps):
            p_rewire = step / (evolution_steps - 1)
            G_temp = nx.watts_strogatz_graph(n_nodes, k, p_rewire)
            clustering_evolution.append(nx.average_clustering(G_temp))
            path_length_evolution.append(nx.average_shortest_path_length(G_temp))
        
        plt.plot(range(evolution_steps), clustering_evolution, 'b-', label='聚类系数')
        plt.plot(range(evolution_steps), path_length_evolution, 'r-', label='平均路径长度')
        plt.xlabel('重连概率')
        plt.ylabel('网络指标')
        plt.title('网络演化')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('network_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 网络模型演示完成，图像保存为 network_model.png")
        print(f"📊 网络统计:")
        print(f"   平均度数: {avg_degree:.2f}")
        print(f"   聚类系数: {clustering_coef:.4f}")
        print(f"   平均路径长度: {avg_path_length:.4f}")
        
        return {
            'network_type': 'small_world',
            'statistics': {
                'avg_degree': avg_degree,
                'clustering_coef': clustering_coef,
                'avg_path_length': avg_path_length
            }
        }
    
    def run_all_demos(self):
        """运行所有演示"""
        print("🚀 形式化模型项目演示")
        print("=" * 60)
        print("项目: 2025年形式化模型体系梳理")
        print("版本: 1.0.0")
        print("状态: 已完成")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行各个演示
        self.results['classical_mechanics'] = self.demo_classical_mechanics()
        self.results['quantum_mechanics'] = self.demo_quantum_mechanics()
        self.results['optimization'] = self.demo_optimization()
        self.results['machine_learning'] = self.demo_machine_learning()
        self.results['financial_model'] = self.demo_financial_model()
        self.results['network_model'] = self.demo_network_model()
        
        end_time = time.time()
        
        # 生成演示报告
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成!")
        print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
        print("📁 生成的图像文件:")
        print("   - harmonic_motion.png (简谐运动)")
        print("   - quantum_wave.png (量子波函数)")
        print("   - optimization.png (优化算法)")
        print("   - linear_regression.png (线性回归)")
        print("   - financial_model.png (金融模型)")
        print("   - network_model.png (网络模型)")
        print("   - demo_report.json (演示报告)")
        print("=" * 60)
    
    def generate_report(self):
        """生成演示报告"""
        report = {
            'project_info': {
                'name': '2025年形式化模型体系梳理',
                'version': '1.0.0',
                'status': '已完成',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'demo_results': self.results,
            'summary': {
                'total_demos': len(self.results),
                'models_covered': [
                    '经典力学', '量子力学', '优化算法', 
                    '机器学习', '金融模型', '网络模型'
                ]
            }
        }
        
        with open('demo_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 演示报告已保存为 demo_report.json")

def main():
    """主函数"""
    demo = FormalModelDemo()
    demo.run_all_demos()

if __name__ == "__main__":
    main() 