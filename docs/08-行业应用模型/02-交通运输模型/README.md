# 8.2 交通运输模型 / Transportation Models

## 目录 / Table of Contents

- [8.2 交通运输模型 / Transportation Models](#82-交通运输模型--transportation-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [8.2.1 交通流模型 / Traffic Flow Models](#821-交通流模型--traffic-flow-models)
    - [宏观交通流模型 / Macroscopic Traffic Flow Models](#宏观交通流模型--macroscopic-traffic-flow-models)
    - [微观交通流模型 / Microscopic Traffic Flow Models](#微观交通流模型--microscopic-traffic-flow-models)
    - [中观交通流模型 / Mesoscopic Traffic Flow Models](#中观交通流模型--mesoscopic-traffic-flow-models)
  - [8.2.2 路径规划模型 / Route Planning Models](#822-路径规划模型--route-planning-models)
    - [最短路径算法 / Shortest Path Algorithms](#最短路径算法--shortest-path-algorithms)
    - [多目标路径规划 / Multi-Objective Route Planning](#多目标路径规划--multi-objective-route-planning)
    - [动态路径规划 / Dynamic Route Planning](#动态路径规划--dynamic-route-planning)
  - [8.2.3 公共交通模型 / Public Transit Models](#823-公共交通模型--public-transit-models)
    - [公交线路优化 / Bus Route Optimization](#公交线路优化--bus-route-optimization)
    - [时刻表优化 / Timetable Optimization](#时刻表优化--timetable-optimization)
    - [换乘优化 / Transfer Optimization](#换乘优化--transfer-optimization)
  - [8.2.4 智能交通系统模型 / Intelligent Transportation System Models](#824-智能交通系统模型--intelligent-transportation-system-models)
    - [交通信号控制 / Traffic Signal Control](#交通信号控制--traffic-signal-control)
    - [车辆检测与识别 / Vehicle Detection and Recognition](#车辆检测与识别--vehicle-detection-and-recognition)
    - [交通预测 / Traffic Prediction](#交通预测--traffic-prediction)
  - [8.2.5 实现与应用 / Implementation and Applications](#825-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [城市交通管理 / Urban Traffic Management](#城市交通管理--urban-traffic-management)
      - [智能交通系统 / Intelligent Transportation Systems](#智能交通系统--intelligent-transportation-systems)
      - [公共交通优化 / Public Transit Optimization](#公共交通优化--public-transit-optimization)
  - [参考文献 / References](#参考文献--references)

---

## 8.2.1 交通流模型 / Traffic Flow Models

### 宏观交通流模型 / Macroscopic Traffic Flow Models

**基本图关系**: $q = k \cdot v$

**Greenshields模型**: $v = v_f \left(1 - \frac{k}{k_j}\right)$

**流量-密度关系**: $q = k \cdot v_f \left(1 - \frac{k}{k_j}\right)$

**LWR模型**: $\frac{\partial k}{\partial t} + \frac{\partial q}{\partial x} = 0$

### 微观交通流模型 / Microscopic Traffic Flow Models

**跟车模型**: $\frac{d^2 x_n(t)}{dt^2} = \alpha \left[\frac{dx_{n-1}(t-\tau)}{dt} - \frac{dx_n(t)}{dt}\right]$

**IDM模型**: $a = a_0 \left[1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*(v,\Delta v)}{s}\right)^2\right]$

**期望间距**: $s^*(v,\Delta v) = s_0 + vT + \frac{v \cdot \Delta v}{2\sqrt{a_0 b}}$

### 中观交通流模型 / Mesoscopic Traffic Flow Models

**元胞传输模型**: $n_i(t+1) = n_i(t) + y_i(t) - y_{i+1}(t)$

**传输流量**: $y_i(t) = \min\{n_{i-1}(t), Q_i, \frac{w}{v_i}[N_i - n_i(t)]\}$

---

## 8.2.2 路径规划模型 / Route Planning Models

### 最短路径算法 / Shortest Path Algorithms

**Dijkstra算法**: $d[v] = \min\{d[v], d[u] + w(u,v)\}$

**A*算法**: $f(n) = g(n) + h(n)$

**Floyd-Warshall算法**: $d[i][j] = \min\{d[i][j], d[i][k] + d[k][j]\}$

### 多目标路径规划 / Multi-Objective Route Planning

**目标函数**: $\min \sum_{i=1}^n w_i f_i(x)$

**帕累托最优**: $f_i(x^*) \leq f_i(x)$ for all $i$

### 动态路径规划 / Dynamic Route Planning

**实时更新**: $P(t) = f(P(t-1), \Delta T(t))$

**预测模型**: $\hat{T}(t+\Delta t) = T(t) + \alpha \cdot \Delta T(t)$

---

## 8.2.3 公共交通模型 / Public Transit Models

### 公交线路优化 / Bus Route Optimization

**目标函数**: $\min \sum_{i=1}^n c_i x_i$

**约束条件**: $\sum_{j \in N(i)} x_{ij} = 1$ for all $i$

**覆盖率**: $C = \frac{\text{覆盖人口}}{\text{总人口}}$

### 时刻表优化 / Timetable Optimization

**发车间隔**: $h = \frac{T}{n}$

**等待时间**: $W = \frac{h}{2}$

**乘客满意度**: $S = \frac{1}{1 + \alpha W}$

### 换乘优化 / Transfer Optimization

**换乘时间**: $T_{transfer} = T_{arrival} - T_{departure}$

**换乘成本**: $C_{transfer} = \alpha \cdot T_{transfer} + \beta \cdot D_{walk}$

---

## 8.2.4 智能交通系统模型 / Intelligent Transportation System Models

### 交通信号控制 / Traffic Signal Control

**Webster公式**: $C = \frac{1.5L + 5}{1 - Y}$

**绿信比**: $g_i = \frac{q_i}{q_{total}} \cdot (C - L)$

**延误**: $d = \frac{C(1-\lambda)^2}{2(1-\lambda x)} + \frac{x^2}{2q(1-x)}$

### 车辆检测与识别 / Vehicle Detection and Recognition

**检测率**: $DR = \frac{TP}{TP + FN}$

**精确率**: $PR = \frac{TP}{TP + FP}$

**F1分数**: $F1 = \frac{2 \cdot PR \cdot DR}{PR + DR}$

### 交通预测 / Traffic Prediction

**时间序列模型**: $T(t) = \mu + \alpha t + \sum_{i=1}^p \phi_i T(t-i) + \epsilon_t$

**神经网络**: $T(t+1) = f(T(t), T(t-1), \ldots, T(t-n))$

---

## 8.2.5 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct TrafficFlowModel {
    pub free_flow_speed: f64,
    pub jam_density: f64,
    pub critical_density: f64,
}

impl TrafficFlowModel {
    pub fn new(free_flow_speed: f64, jam_density: f64) -> Self {
        Self {
            free_flow_speed,
            jam_density,
            critical_density: jam_density / 2.0,
        }
    }
    
    pub fn greenshields_speed(&self, density: f64) -> f64 {
        if density >= self.jam_density {
            0.0
        } else {
            self.free_flow_speed * (1.0 - density / self.jam_density)
        }
    }
    
    pub fn flow_rate(&self, density: f64) -> f64 {
        density * self.greenshields_speed(density)
    }
    
    pub fn optimal_density(&self) -> f64 {
        self.critical_density
    }
    
    pub fn max_flow_rate(&self) -> f64 {
        self.flow_rate(self.critical_density)
    }
}

#[derive(Debug, Clone)]
pub struct RoutePlanningModel {
    pub graph: HashMap<String, HashMap<String, f64>>,
    pub heuristic: HashMap<String, f64>,
}

impl RoutePlanningModel {
    pub fn new() -> Self {
        Self {
            graph: HashMap::new(),
            heuristic: HashMap::new(),
        }
    }
    
    pub fn add_edge(&mut self, from: String, to: String, weight: f64) {
        self.graph.entry(from.clone()).or_insert_with(HashMap::new).insert(to.clone(), weight);
        self.graph.entry(to).or_insert_with(HashMap::new).insert(from, weight);
    }
    
    pub fn add_heuristic(&mut self, node: String, h_value: f64) {
        self.heuristic.insert(node, h_value);
    }
    
    pub fn dijkstra_shortest_path(&self, start: &str, end: &str) -> Option<(f64, Vec<String>)> {
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, String> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();
        
        // 初始化距离
        for node in self.graph.keys() {
            distances.insert(node.clone(), f64::INFINITY);
        }
        distances.insert(start.to_string(), 0.0);
        
        while !visited.contains(end) {
            // 找到未访问节点中距离最小的
            let current = distances.iter()
                .filter(|(node, _)| !visited.contains(*node))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(node, _)| node.clone())?;
            
            if current == end {
                break;
            }
            
            visited.insert(current.clone());
            
            // 更新邻居距离
            if let Some(neighbors) = self.graph.get(&current) {
                for (neighbor, weight) in neighbors {
                    if !visited.contains(neighbor) {
                        let new_distance = distances[&current] + weight;
                        if new_distance < distances.get(neighbor).unwrap_or(&f64::INFINITY) {
                            distances.insert(neighbor.clone(), new_distance);
                            previous.insert(neighbor.clone(), current.clone());
                        }
                    }
                }
            }
        }
        
        // 重建路径
        let mut path = Vec::new();
        let mut current = end.to_string();
        while current != start {
            path.push(current.clone());
            current = previous.get(&current)?.clone();
        }
        path.push(start.to_string());
        path.reverse();
        
        Some((distances[end], path))
    }
    
    pub fn a_star_search(&self, start: &str, end: &str) -> Option<(f64, Vec<String>)> {
        let mut open_set: BinaryHeap<AStarNode> = BinaryHeap::new();
        let mut g_score: HashMap<String, f64> = HashMap::new();
        let mut f_score: HashMap<String, f64> = HashMap::new();
        let mut came_from: HashMap<String, String> = HashMap::new();
        
        g_score.insert(start.to_string(), 0.0);
        f_score.insert(start.to_string(), self.heuristic.get(start).unwrap_or(&0.0));
        open_set.push(AStarNode {
            node: start.to_string(),
            f_score: *f_score.get(start).unwrap_or(&0.0),
        });
        
        while !open_set.is_empty() {
            let current = open_set.pop()?.node;
            
            if current == end {
                // 重建路径
                let mut path = Vec::new();
                let mut current_node = end.to_string();
                while current_node != start {
                    path.push(current_node.clone());
                    current_node = came_from.get(&current_node)?.clone();
                }
                path.push(start.to_string());
                path.reverse();
                return Some((g_score[end], path));
            }
            
            if let Some(neighbors) = self.graph.get(&current) {
                for (neighbor, weight) in neighbors {
                    let tentative_g_score = g_score.get(&current).unwrap_or(&f64::INFINITY) + weight;
                    
                    if tentative_g_score < *g_score.get(neighbor).unwrap_or(&f64::INFINITY) {
                        came_from.insert(neighbor.clone(), current.clone());
                        g_score.insert(neighbor.clone(), tentative_g_score);
                        let f_score_value = tentative_g_score + self.heuristic.get(neighbor).unwrap_or(&0.0);
                        f_score.insert(neighbor.clone(), f_score_value);
                        open_set.push(AStarNode {
                            node: neighbor.clone(),
                            f_score: f_score_value,
                        });
                    }
                }
            }
        }
        
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AStarNode {
    node: String,
    f_score: f64,
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.f_score.partial_cmp(&self.f_score)
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Clone)]
pub struct PublicTransitModel {
    pub routes: HashMap<String, Vec<String>>,
    pub frequencies: HashMap<String, f64>,
    pub transfer_times: HashMap<(String, String), f64>,
}

impl PublicTransitModel {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            frequencies: HashMap::new(),
            transfer_times: HashMap::new(),
        }
    }
    
    pub fn add_route(&mut self, route_id: String, stops: Vec<String>, frequency: f64) {
        self.routes.insert(route_id.clone(), stops);
        self.frequencies.insert(route_id, frequency);
    }
    
    pub fn add_transfer_time(&mut self, route1: String, route2: String, time: f64) {
        self.transfer_times.insert((route1, route2), time);
    }
    
    pub fn calculate_wait_time(&self, route_id: &str) -> f64 {
        if let Some(frequency) = self.frequencies.get(route_id) {
            1.0 / (2.0 * frequency)
        } else {
            f64::INFINITY
        }
    }
    
    pub fn calculate_total_travel_time(&self, origin: &str, destination: &str) -> f64 {
        // 简化的旅行时间计算
        let mut total_time = 0.0;
        
        // 找到包含起点和终点的路线
        for (route_id, stops) in &self.routes {
            if stops.contains(&origin.to_string()) && stops.contains(&destination.to_string()) {
                let wait_time = self.calculate_wait_time(route_id);
                let travel_time = self.calculate_travel_time(route_id, origin, destination);
                total_time = wait_time + travel_time;
                break;
            }
        }
        
        total_time
    }
    
    fn calculate_travel_time(&self, route_id: &str, origin: &str, destination: &str) -> f64 {
        if let Some(stops) = self.routes.get(route_id) {
            if let (Some(origin_idx), Some(dest_idx)) = (
                stops.iter().position(|s| s == origin),
                stops.iter().position(|s| s == destination)
            ) {
                let stop_count = (dest_idx as i32 - origin_idx as i32).abs() as f64;
                stop_count * 2.0 // 假设每站2分钟
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrafficSignalModel {
    pub intersections: HashMap<String, Vec<String>>,
    pub cycle_lengths: HashMap<String, f64>,
    pub green_splits: HashMap<String, Vec<f64>>,
}

impl TrafficSignalModel {
    pub fn new() -> Self {
        Self {
            intersections: HashMap::new(),
            cycle_lengths: HashMap::new(),
            green_splits: HashMap::new(),
        }
    }
    
    pub fn add_intersection(&mut self, intersection_id: String, approaches: Vec<String>) {
        self.intersections.insert(intersection_id.clone(), approaches);
    }
    
    pub fn set_cycle_length(&mut self, intersection_id: String, cycle_length: f64) {
        self.cycle_lengths.insert(intersection_id, cycle_length);
    }
    
    pub fn set_green_splits(&mut self, intersection_id: String, splits: Vec<f64>) {
        self.green_splits.insert(intersection_id, splits);
    }
    
    pub fn calculate_delay(&self, intersection_id: &str, approach: &str) -> f64 {
        if let (Some(cycle_length), Some(splits)) = (
            self.cycle_lengths.get(intersection_id),
            self.green_splits.get(intersection_id)
        ) {
            if let Some(approaches) = self.intersections.get(intersection_id) {
                if let Some(approach_idx) = approaches.iter().position(|a| a == approach) {
                    if approach_idx < splits.len() {
                        let green_time = cycle_length * splits[approach_idx];
                        let red_time = cycle_length - green_time;
                        red_time * red_time / (2.0 * cycle_length)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    pub fn optimize_signal_timing(&mut self, intersection_id: &str, flows: &[f64]) -> Vec<f64> {
        if let Some(cycle_length) = self.cycle_lengths.get(intersection_id) {
            let total_flow: f64 = flows.iter().sum();
            let splits: Vec<f64> = flows.iter().map(|&flow| flow / total_flow).collect();
            self.green_splits.insert(intersection_id.to_string(), splits.clone());
            splits
        } else {
            vec![]
        }
    }
}

// 使用示例
fn main() {
    // 交通流模型示例
    let traffic_flow = TrafficFlowModel::new(60.0, 120.0);
    let density = 30.0;
    let speed = traffic_flow.greenshields_speed(density);
    let flow = traffic_flow.flow_rate(density);
    let max_flow = traffic_flow.max_flow_rate();
    
    println!("交通流模型示例:");
    println!("密度: {:.1} 车辆/公里", density);
    println!("速度: {:.1} 公里/小时", speed);
    println!("流量: {:.1} 车辆/小时", flow);
    println!("最大流量: {:.1} 车辆/小时", max_flow);
    
    // 路径规划模型示例
    let mut route_planning = RoutePlanningModel::new();
    route_planning.add_edge("A".to_string(), "B".to_string(), 10.0);
    route_planning.add_edge("B".to_string(), "C".to_string(), 15.0);
    route_planning.add_edge("A".to_string(), "C".to_string(), 25.0);
    
    route_planning.add_heuristic("A".to_string(), 0.0);
    route_planning.add_heuristic("B".to_string(), 10.0);
    route_planning.add_heuristic("C".to_string(), 15.0);
    
    if let Some((distance, path)) = route_planning.dijkstra_shortest_path("A", "C") {
        println!("\n路径规划示例 (Dijkstra):");
        println!("最短距离: {:.1}", distance);
        println!("路径: {:?}", path);
    }
    
    if let Some((distance, path)) = route_planning.a_star_search("A", "C") {
        println!("\n路径规划示例 (A*):");
        println!("最短距离: {:.1}", distance);
        println!("路径: {:?}", path);
    }
    
    // 公共交通模型示例
    let mut transit = PublicTransitModel::new();
    transit.add_route("R1".to_string(), 
                     vec!["A".to_string(), "B".to_string(), "C".to_string()], 
                     4.0); // 每小时4班
    transit.add_route("R2".to_string(), 
                     vec!["B".to_string(), "D".to_string(), "E".to_string()], 
                     6.0); // 每小时6班
    
    let wait_time = transit.calculate_wait_time("R1");
    let travel_time = transit.calculate_total_travel_time("A", "C");
    
    println!("\n公共交通模型示例:");
    println!("等待时间: {:.1} 分钟", wait_time * 60.0);
    println!("总旅行时间: {:.1} 分钟", travel_time * 60.0);
    
    // 交通信号模型示例
    let mut signal = TrafficSignalModel::new();
    signal.add_intersection("I1".to_string(), 
                           vec!["North".to_string(), "South".to_string(), "East".to_string(), "West".to_string()]);
    signal.set_cycle_length("I1".to_string(), 90.0); // 90秒周期
    
    let flows = vec![800.0, 600.0, 400.0, 300.0]; // 各方向流量
    let splits = signal.optimize_signal_timing("I1", &flows);
    let delay = signal.calculate_delay("I1", "North");
    
    println!("\n交通信号模型示例:");
    println!("绿信比: {:?}", splits);
    println!("北向延误: {:.1} 秒", delay);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module TransportationModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, minimumBy, find)
import Data.Ord (comparing)
import qualified Data.Set as Set

-- 交通流模型
data TrafficFlowModel = TrafficFlowModel {
    freeFlowSpeed :: Double,
    jamDensity :: Double,
    criticalDensity :: Double
} deriving Show

newTrafficFlowModel :: Double -> Double -> TrafficFlowModel
newTrafficFlowModel freeFlowSpeed jamDensity = TrafficFlowModel {
    freeFlowSpeed = freeFlowSpeed,
    jamDensity = jamDensity,
    criticalDensity = jamDensity / 2.0
}

greenshieldsSpeed :: TrafficFlowModel -> Double -> Double
greenshieldsSpeed model density = 
    if density >= jamDensity model 
    then 0.0
    else freeFlowSpeed model * (1.0 - density / jamDensity model)

flowRate :: TrafficFlowModel -> Double -> Double
flowRate model density = density * greenshieldsSpeed model density

optimalDensity :: TrafficFlowModel -> Double
optimalDensity = criticalDensity

maxFlowRate :: TrafficFlowModel -> Double
maxFlowRate model = flowRate model (criticalDensity model)

-- 路径规划模型
data RoutePlanningModel = RoutePlanningModel {
    graph :: Map String (Map String Double),
    heuristic :: Map String Double
} deriving Show

newRoutePlanningModel :: RoutePlanningModel
newRoutePlanningModel = RoutePlanningModel {
    graph = Map.empty,
    heuristic = Map.empty
}

addEdge :: RoutePlanningModel -> String -> String -> Double -> RoutePlanningModel
addEdge model from to weight = 
    let newGraph = Map.insertWith Map.union from (Map.singleton to weight) (graph model)
        finalGraph = Map.insertWith Map.union to (Map.singleton from weight) newGraph
    in model { graph = finalGraph }

addHeuristic :: RoutePlanningModel -> String -> Double -> RoutePlanningModel
addHeuristic model node hValue = 
    model { heuristic = Map.insert node hValue (heuristic model) }

dijkstraShortestPath :: RoutePlanningModel -> String -> String -> Maybe (Double, [String])
dijkstraShortestPath model start end = 
    let nodes = Map.keys (graph model)
        initialDistances = Map.fromList [(node, 1/0) | node <- nodes]
        distances = Map.insert start 0.0 initialDistances
    in dijkstraHelper model end distances Map.empty Set.empty

dijkstraHelper :: RoutePlanningModel -> String -> Map String Double -> 
                 Map String String -> Set.Set String -> Maybe (Double, [String])
dijkstraHelper model end distances previous visited
    | Set.member end visited = Just (distances Map.! end, reconstructPath previous end)
    | otherwise = 
        let unvisited = Map.keys distances `Set.difference` visited
            current = minimumBy (comparing (\node -> distances Map.! node)) (Set.toList unvisited)
            newVisited = Set.insert current visited
        in if current == end 
           then Just (distances Map.! end, reconstructPath previous end)
           else dijkstraHelper model end (updateDistances model current distances) 
                                    (updatePrevious model current distances previous) newVisited

updateDistances :: RoutePlanningModel -> String -> Map String Double -> Map String Double
updateDistances model current distances = 
    case Map.lookup current (graph model) of
        Just neighbors -> foldr (\neighbor acc -> 
            let newDistance = distances Map.! current + (neighbors Map.! neighbor)
            in if newDistance < (distances Map.! neighbor)
               then Map.insert neighbor newDistance acc
               else acc) distances (Map.keys neighbors)
        Nothing -> distances

updatePrevious :: RoutePlanningModel -> String -> Map String Double -> Map String String -> Map String String
updatePrevious model current distances previous = 
    case Map.lookup current (graph model) of
        Just neighbors -> foldr (\neighbor acc -> 
            let newDistance = distances Map.! current + (neighbors Map.! neighbor)
            in if newDistance < (distances Map.! neighbor)
               then Map.insert neighbor current acc
               else acc) previous (Map.keys neighbors)
        Nothing -> previous

reconstructPath :: Map String String -> String -> [String]
reconstructPath previous end = 
    let path = takeWhile (/= "") (iterate (\node -> Map.findWithDefault "" node previous) end)
    in reverse path

-- 公共交通模型
data PublicTransitModel = PublicTransitModel {
    routes :: Map String [String],
    frequencies :: Map String Double,
    transferTimes :: Map (String, String) Double
} deriving Show

newPublicTransitModel :: PublicTransitModel
newPublicTransitModel = PublicTransitModel {
    routes = Map.empty,
    frequencies = Map.empty,
    transferTimes = Map.empty
}

addRoute :: PublicTransitModel -> String -> [String] -> Double -> PublicTransitModel
addRoute model routeId stops frequency = 
    model { 
        routes = Map.insert routeId stops (routes model),
        frequencies = Map.insert routeId frequency (frequencies model)
    }

addTransferTime :: PublicTransitModel -> String -> String -> Double -> PublicTransitModel
addTransferTime model route1 route2 time = 
    model { transferTimes = Map.insert (route1, route2) time (transferTimes model) }

calculateWaitTime :: PublicTransitModel -> String -> Double
calculateWaitTime model routeId = 
    case Map.lookup routeId (frequencies model) of
        Just frequency -> 1.0 / (2.0 * frequency)
        Nothing -> 1/0

calculateTotalTravelTime :: PublicTransitModel -> String -> String -> Double
calculateTotalTravelTime model origin destination = 
    let routeEntries = Map.toList (routes model)
        matchingRoutes = filter (\(_, stops) -> 
            origin `elem` stops && destination `elem` stops) routeEntries
    in case matchingRoutes of
        ((routeId, _):_) -> 
            let waitTime = calculateWaitTime model routeId
                travelTime = calculateTravelTime model routeId origin destination
            in waitTime + travelTime
        _ -> 1/0

calculateTravelTime :: PublicTransitModel -> String -> String -> String -> Double
calculateTravelTime model routeId origin destination = 
    case Map.lookup routeId (routes model) of
        Just stops -> 
            case (findIndex (== origin) stops, findIndex (== destination) stops) of
                (Just originIdx, Just destIdx) -> 
                    let stopCount = abs (destIdx - originIdx)
                    in fromIntegral stopCount * 2.0 -- 假设每站2分钟
                _ -> 1/0
        Nothing -> 1/0

-- 交通信号模型
data TrafficSignalModel = TrafficSignalModel {
    intersections :: Map String [String],
    cycleLengths :: Map String Double,
    greenSplits :: Map String [Double]
} deriving Show

newTrafficSignalModel :: TrafficSignalModel
newTrafficSignalModel = TrafficSignalModel {
    intersections = Map.empty,
    cycleLengths = Map.empty,
    greenSplits = Map.empty
}

addIntersection :: TrafficSignalModel -> String -> [String] -> TrafficSignalModel
addIntersection model intersectionId approaches = 
    model { intersections = Map.insert intersectionId approaches (intersections model) }

setCycleLength :: TrafficSignalModel -> String -> Double -> TrafficSignalModel
setCycleLength model intersectionId cycleLength = 
    model { cycleLengths = Map.insert intersectionId cycleLength (cycleLengths model) }

setGreenSplits :: TrafficSignalModel -> String -> [Double] -> TrafficSignalModel
setGreenSplits model intersectionId splits = 
    model { greenSplits = Map.insert intersectionId splits (greenSplits model) }

calculateDelay :: TrafficSignalModel -> String -> String -> Double
calculateDelay model intersectionId approach = 
    case (Map.lookup intersectionId (cycleLengths model), 
          Map.lookup intersectionId (greenSplits model),
          Map.lookup intersectionId (intersections model)) of
        (Just cycleLength, Just splits, Just approaches) -> 
            case findIndex (== approach) approaches of
                Just approachIdx -> 
                    if approachIdx < length splits 
                    then let greenTime = cycleLength * splits !! approachIdx
                             redTime = cycleLength - greenTime
                         in redTime * redTime / (2.0 * cycleLength)
                    else 0.0
                Nothing -> 0.0
        _ -> 0.0

optimizeSignalTiming :: TrafficSignalModel -> String -> [Double] -> [Double]
optimizeSignalTiming model intersectionId flows = 
    case Map.lookup intersectionId (cycleLengths model) of
        Just cycleLength -> 
            let totalFlow = sum flows
                splits = map (/ totalFlow) flows
            in splits
        Nothing -> []

-- 辅助函数
findIndex :: Eq a => a -> [a] -> Maybe Int
findIndex x xs = lookup x (zip xs [0..])

-- 示例使用
example :: IO ()
example = do
    -- 交通流模型示例
    let trafficFlow = newTrafficFlowModel 60.0 120.0
        density = 30.0
        speed = greenshieldsSpeed trafficFlow density
        flow = flowRate trafficFlow density
        maxFlow = maxFlowRate trafficFlow
    
    putStrLn "交通流模型示例:"
    putStrLn $ "密度: " ++ show density ++ " 车辆/公里"
    putStrLn $ "速度: " ++ show speed ++ " 公里/小时"
    putStrLn $ "流量: " ++ show flow ++ " 车辆/小时"
    putStrLn $ "最大流量: " ++ show maxFlow ++ " 车辆/小时"
    
    -- 路径规划模型示例
    let routePlanning = addEdge (addEdge (addEdge newRoutePlanningModel "A" "B" 10.0) "B" "C" 15.0) "A" "C" 25.0
        routePlanningWithHeuristic = addHeuristic (addHeuristic (addHeuristic routePlanning "A" 0.0) "B" 10.0) "C" 15.0
    
    case dijkstraShortestPath routePlanningWithHeuristic "A" "C" of
        Just (distance, path) -> do
            putStrLn "\n路径规划示例 (Dijkstra):"
            putStrLn $ "最短距离: " ++ show distance
            putStrLn $ "路径: " ++ show path
        Nothing -> putStrLn "未找到路径"
    
    -- 公共交通模型示例
    let transit = addRoute (addRoute newPublicTransitModel "R1" ["A", "B", "C"] 4.0) "R2" ["B", "D", "E"] 6.0
        waitTime = calculateWaitTime transit "R1"
        travelTime = calculateTotalTravelTime transit "A" "C"
    
    putStrLn "\n公共交通模型示例:"
    putStrLn $ "等待时间: " ++ show (waitTime * 60.0) ++ " 分钟"
    putStrLn $ "总旅行时间: " ++ show (travelTime * 60.0) ++ " 分钟"
    
    -- 交通信号模型示例
    let signal = setCycleLength (addIntersection newTrafficSignalModel "I1" ["North", "South", "East", "West"]) "I1" 90.0
        flows = [800.0, 600.0, 400.0, 300.0]
        splits = optimizeSignalTiming signal "I1" flows
        delay = calculateDelay signal "I1" "North"
    
    putStrLn "\n交通信号模型示例:"
    putStrLn $ "绿信比: " ++ show splits
    putStrLn $ "北向延误: " ++ show delay ++ " 秒"
```

### 应用领域 / Application Domains

#### 城市交通管理 / Urban Traffic Management

- **交通信号控制**: 自适应信号控制、绿波带协调
- **交通流预测**: 实时交通状态预测、拥堵预警
- **交通规划**: 道路网络优化、交通设施布局

#### 智能交通系统 / Intelligent Transportation Systems

- **车辆检测**: 视频检测、雷达检测、地磁检测
- **交通信息**: 实时路况、出行建议、路径推荐
- **交通控制**: 自适应控制、协调控制、优化控制

#### 公共交通优化 / Public Transit Optimization

- **线路规划**: 公交线路设计、站点布局优化
- **时刻表优化**: 发车间隔优化、换乘时间协调
- **运营管理**: 车辆调度、人员排班、成本控制

---

## 参考文献 / References

1. May, A. D. (1990). Traffic flow fundamentals. Prentice Hall.
2. Daganzo, C. F. (1997). Fundamentals of transportation and traffic operations. Pergamon.
3. Sheffi, Y. (1985). Urban transportation networks: Equilibrium analysis with mathematical programming methods. Prentice Hall.
4. Vuchic, V. R. (2005). Urban transit: Operations, planning, and economics. Wiley.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*

---

## 8.2.6 算法实现 / Algorithm Implementation

### 交通流模型算法 / Traffic Flow Model Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.integrate import odeint
import heapq

@dataclass
class TrafficFlowModel:
    """交通流模型"""
    free_flow_speed: float  # 自由流速度 (km/h)
    jam_density: float      # 阻塞密度 (veh/km)
    capacity: float         # 通行能力 (veh/h)

def greenshields_speed(model: TrafficFlowModel, density: float) -> float:
    """Greenshields速度-密度关系"""
    if density >= model.jam_density:
        return 0.0
    return model.free_flow_speed * (1 - density / model.jam_density)

def greenshields_flow(model: TrafficFlowModel, density: float) -> float:
    """Greenshields流量-密度关系"""
    speed = greenshields_speed(model, density)
    return density * speed

def max_flow_rate(model: TrafficFlowModel) -> float:
    """最大流量"""
    optimal_density = model.jam_density / 2
    return greenshields_flow(model, optimal_density)

def lwr_model_simulation(
    initial_density: np.ndarray,
    dx: float,
    dt: float,
    total_time: float,
    model: TrafficFlowModel
) -> np.ndarray:
    """LWR模型数值模拟"""
    nx = len(initial_density)
    nt = int(total_time / dt)
    
    # 初始化密度场
    density = np.zeros((nt, nx))
    density[0] = initial_density
    
    # 时间推进
    for n in range(nt - 1):
        for i in range(nx):
            # 简化的有限差分格式
            if i == 0:
                # 左边界条件
                density[n+1, i] = density[n, i] - dt/dx * (greenshields_flow(model, density[n, i+1]) - greenshields_flow(model, density[n, i]))
            elif i == nx - 1:
                # 右边界条件
                density[n+1, i] = density[n, i] - dt/dx * (greenshields_flow(model, density[n, i]) - greenshields_flow(model, density[n, i-1]))
            else:
                # 内部点
                density[n+1, i] = density[n, i] - dt/dx * (greenshields_flow(model, density[n, i+1]) - greenshields_flow(model, density[n, i-1])) / 2
    
    return density

class IntelligentDriverModel:
    """智能驾驶模型"""
    
    def __init__(self, desired_speed: float, safe_time_headway: float, 
                 max_acceleration: float, comfortable_deceleration: float,
                 minimum_spacing: float, acceleration_exponent: float = 4):
        self.v0 = desired_speed
        self.T = safe_time_headway
        self.a = max_acceleration
        self.b = comfortable_deceleration
        self.s0 = minimum_spacing
        self.delta = acceleration_exponent
    
    def desired_gap(self, speed: float, speed_diff: float) -> float:
        """期望间距"""
        return self.s0 + speed * self.T + (speed * speed_diff) / (2 * np.sqrt(self.a * self.b))
    
    def acceleration(self, speed: float, gap: float, speed_diff: float) -> float:
        """加速度计算"""
        desired_gap = self.desired_gap(speed, speed_diff)
        free_road_term = 1 - (speed / self.v0) ** self.delta
        interaction_term = (desired_gap / gap) ** 2
        
        return self.a * (free_road_term - interaction_term)
    
    def simulate_car_following(self, lead_vehicle_trajectory: List[Tuple[float, float]], 
                              initial_speed: float, initial_position: float,
                              dt: float) -> List[Tuple[float, float, float]]:
        """跟车模拟"""
        trajectory = []
        current_speed = initial_speed
        current_position = initial_position
        
        for t, lead_pos in lead_vehicle_trajectory:
            # 计算间距和速度差
            gap = lead_pos - current_position
            speed_diff = 0  # 简化假设前车速度恒定
            
            if gap > 0:
                # 计算加速度
                acc = self.acceleration(current_speed, gap, speed_diff)
                
                # 更新速度和位置
                current_speed = max(0, current_speed + acc * dt)
                current_position += current_speed * dt
            
            trajectory.append((t, current_position, current_speed))
        
        return trajectory
```

### 路径规划算法 / Route Planning Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import heapq
from collections import defaultdict

class Graph:
    """图结构"""
    
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, u: str, v: str, weight: float):
        """添加边"""
        self.edges[u].append(v)
        self.edges[v].append(u)
        self.weights[(u, v)] = weight
        self.weights[(v, u)] = weight
    
    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """获取邻居节点"""
        neighbors = []
        for neighbor in self.edges[node]:
            weight = self.weights.get((node, neighbor), float('inf'))
            neighbors.append((neighbor, weight))
        return neighbors

def dijkstra_shortest_path(graph: Graph, start: str, end: str) -> Optional[Tuple[float, List[str]]]:
    """Dijkstra最短路径算法"""
    distances = {node: float('inf') for node in graph.edges}
    distances[start] = 0
    previous = {}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == end:
            break
        
        for neighbor, weight in graph.get_neighbors(current):
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    if distances[end] == float('inf'):
        return None
    
    # 重建路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    
    return distances[end], path

def a_star_shortest_path(graph: Graph, start: str, end: str, 
                        heuristic: Dict[str, float]) -> Optional[Tuple[float, List[str]]]:
    """A*最短路径算法"""
    distances = {node: float('inf') for node in graph.edges}
    distances[start] = 0
    previous = {}
    pq = [(heuristic.get(start, 0), 0, start)]
    visited = set()
    
    while pq:
        _, current_distance, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == end:
            break
        
        for neighbor, weight in graph.get_neighbors(current):
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                f_score = distance + heuristic.get(neighbor, 0)
                heapq.heappush(pq, (f_score, distance, neighbor))
    
    if distances[end] == float('inf'):
        return None
    
    # 重建路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    
    return distances[end], path

def floyd_warshall_all_pairs(graph: Graph) -> Dict[Tuple[str, str], float]:
    """Floyd-Warshall全对最短路径"""
    nodes = list(graph.edges.keys())
    n = len(nodes)
    
    # 初始化距离矩阵
    distances = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                distances[(i, j)] = 0
            else:
                distances[(i, j)] = graph.weights.get((i, j), float('inf'))
    
    # Floyd-Warshall算法
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if distances[(i, k)] + distances[(k, j)] < distances[(i, j)]:
                    distances[(i, j)] = distances[(i, k)] + distances[(k, j)]
    
    return distances

def multi_objective_route_planning(
    graph: Graph,
    start: str,
    end: str,
    objectives: List[callable],
    weights: List[float]
) -> Optional[Tuple[float, List[str]]]:
    """多目标路径规划"""
    # 计算每个目标的最短路径
    objective_paths = []
    objective_costs = []
    
    for objective in objectives:
        # 创建加权图
        weighted_graph = Graph()
        for node in graph.edges:
            for neighbor, weight in graph.get_neighbors(node):
                weighted_weight = objective(weight)
                weighted_graph.add_edge(node, neighbor, weighted_weight)
        
        # 计算最短路径
        result = dijkstra_shortest_path(weighted_graph, start, end)
        if result:
            cost, path = result
            objective_paths.append(path)
            objective_costs.append(cost)
        else:
            return None
    
    # 计算综合成本
    total_cost = sum(w * c for w, c in zip(weights, objective_costs))
    
    # 返回第一个目标的最短路径（简化处理）
    return total_cost, objective_paths[0]
```

### 公共交通算法 / Public Transit Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BusRoute:
    """公交线路"""
    route_id: str
    stops: List[str]
    headway: float  # 发车间隔（分钟）
    travel_times: List[float]  # 站间旅行时间

@dataclass
class TransitNetwork:
    """公交网络"""
    routes: List[BusRoute]
    transfer_times: Dict[Tuple[str, str], float]  # 换乘时间

class PublicTransitOptimizer:
    """公共交通优化器"""
    
    def __init__(self, network: TransitNetwork):
        self.network = network
        self.stop_routes = self._build_stop_routes()
    
    def _build_stop_routes(self) -> Dict[str, List[str]]:
        """构建站点-线路映射"""
        stop_routes = defaultdict(list)
        for route in self.network.routes:
            for stop in route.stops:
                stop_routes[stop].append(route.route_id)
        return dict(stop_routes)
    
    def calculate_wait_time(self, route_id: str) -> float:
        """计算等待时间"""
        route = next(r for r in self.network.routes if r.route_id == route_id)
        return route.headway / 2  # 平均等待时间
    
    def calculate_travel_time(self, origin: str, destination: str) -> float:
        """计算旅行时间"""
        # 简化的旅行时间计算
        if origin == destination:
            return 0.0
        
        # 查找直达线路
        for route in self.network.routes:
            if origin in route.stops and destination in route.stops:
                origin_idx = route.stops.index(origin)
                dest_idx = route.stops.index(destination)
                
                if origin_idx < dest_idx:
                    # 计算旅行时间
                    travel_time = sum(route.travel_times[origin_idx:dest_idx])
                    wait_time = self.calculate_wait_time(route.route_id)
                    return travel_time + wait_time
        
        # 需要换乘的情况（简化处理）
        return float('inf')
    
    def optimize_headway(self, route_id: str, passenger_demand: float, 
                        vehicle_capacity: float) -> float:
        """优化发车间隔"""
        route = next(r for r in self.network.routes if r.route_id == route_id)
        
        # 基于乘客需求的发车间隔优化
        # 目标：最小化等待时间，同时满足容量约束
        min_headway = 2.0  # 最小发车间隔（分钟）
        max_headway = 30.0  # 最大发车间隔（分钟）
        
        # 简化的优化公式
        optimal_headway = max(min_headway, 
                            min(max_headway, 
                                np.sqrt(2 * route.headway * passenger_demand / vehicle_capacity)))
        
        return optimal_headway
    
    def optimize_route_coverage(self, population_centers: Dict[str, float], 
                               max_routes: int) -> List[BusRoute]:
        """优化线路覆盖"""
        # 简化的贪心算法
        uncovered_centers = set(population_centers.keys())
        selected_routes = []
        
        while len(selected_routes) < max_routes and uncovered_centers:
            best_route = None
            best_coverage = 0
            
            # 评估每条线路的覆盖效果
            for route in self.network.routes:
                coverage = len(set(route.stops) & uncovered_centers)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_route = route
            
            if best_route:
                selected_routes.append(best_route)
                uncovered_centers -= set(best_route.stops)
            else:
                break
        
        return selected_routes

def calculate_transfer_penalty(transfer_time: float, base_penalty: float = 5.0) -> float:
    """计算换乘惩罚"""
    return base_penalty + transfer_time * 0.5

def optimize_transfer_times(network: TransitNetwork) -> Dict[Tuple[str, str], float]:
    """优化换乘时间"""
    optimized_transfers = {}
    
    for route1 in network.routes:
        for route2 in network.routes:
            if route1.route_id != route2.route_id:
                # 找到共同站点
                common_stops = set(route1.stops) & set(route2.stops)
                
                for stop in common_stops:
                    # 计算最优换乘时间
                    route1_idx = route1.stops.index(stop)
                    route2_idx = route2.stops.index(stop)
                    
                    # 简化的换乘时间优化
                    optimal_transfer = 2.0  # 最小换乘时间
                    optimized_transfers[(route1.route_id, route2.route_id)] = optimal_transfer
    
    return optimized_transfers
```

### 智能交通系统算法 / Intelligent Transportation System Algorithms

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import cvxpy as cp

@dataclass
class TrafficSignal:
    """交通信号"""
    intersection_id: str
    phases: List[List[str]]  # 相位定义
    cycle_length: float      # 周期长度（秒）
    green_times: List[float] # 绿灯时间

class TrafficSignalController:
    """交通信号控制器"""
    
    def __init__(self, intersection_id: str, phases: List[List[str]], 
                 cycle_length: float = 90.0):
        self.intersection_id = intersection_id
        self.phases = phases
        self.cycle_length = cycle_length
        self.n_phases = len(phases)
    
    def calculate_delay(self, flow_rate: float, green_time: float, 
                       cycle_length: float) -> float:
        """计算延误"""
        if flow_rate <= 0:
            return 0.0
        
        # Webster延误公式（简化版）
        saturation_flow = 1800  # 饱和流量（辆/小时）
        capacity = saturation_flow * green_time / cycle_length
        
        if flow_rate >= capacity:
            return float('inf')  # 过饱和
        
        # 均匀延误
        uniform_delay = 0.5 * cycle_length * (1 - green_time / cycle_length) ** 2 / (1 - flow_rate / capacity)
        
        # 随机延误（简化）
        random_delay = 0.1 * uniform_delay
        
        return uniform_delay + random_delay
    
    def optimize_signal_timing(self, flows: List[float]) -> List[float]:
        """优化信号配时"""
        if len(flows) != self.n_phases:
            raise ValueError("流量数量与相位数量不匹配")
        
        # 简化的优化：按流量比例分配绿灯时间
        total_flow = sum(flows)
        if total_flow == 0:
            # 平均分配
            green_times = [self.cycle_length / self.n_phases] * self.n_phases
        else:
            # 按流量比例分配
            green_times = [flow / total_flow * self.cycle_length for flow in flows]
        
        # 确保最小绿灯时间
        min_green = 10.0
        for i in range(len(green_times)):
            green_times[i] = max(min_green, green_times[i])
        
        # 调整总时间
        total_green = sum(green_times)
        if total_green > self.cycle_length:
            # 按比例缩减
            scale = self.cycle_length / total_green
            green_times = [g * scale for g in green_times]
        
        return green_times
    
    def calculate_total_delay(self, flows: List[float], green_times: List[float]) -> float:
        """计算总延误"""
        total_delay = 0.0
        for i, (flow, green_time) in enumerate(zip(flows, green_times)):
            delay = self.calculate_delay(flow, green_time, self.cycle_length)
            if delay != float('inf'):
                total_delay += delay * flow
        
        return total_delay

class TrafficPredictor:
    """交通预测器"""
    
    def __init__(self, historical_data: List[float], window_size: int = 24):
        self.historical_data = historical_data
        self.window_size = window_size
    
    def exponential_smoothing_forecast(self, alpha: float = 0.3) -> List[float]:
        """指数平滑预测"""
        if not self.historical_data:
            return []
        
        forecasts = [self.historical_data[0]]
        
        for i in range(1, len(self.historical_data)):
            forecast = alpha * self.historical_data[i-1] + (1 - alpha) * forecasts[i-1]
            forecasts.append(forecast)
        
        # 预测下一个值
        next_forecast = alpha * self.historical_data[-1] + (1 - alpha) * forecasts[-1]
        forecasts.append(next_forecast)
        
        return forecasts
    
    def moving_average_forecast(self) -> List[float]:
        """移动平均预测"""
        if len(self.historical_data) < self.window_size:
            return self.historical_data
        
        forecasts = []
        for i in range(self.window_size, len(self.historical_data)):
            avg = np.mean(self.historical_data[i-self.window_size:i])
            forecasts.append(avg)
        
        # 预测下一个值
        next_forecast = np.mean(self.historical_data[-self.window_size:])
        forecasts.append(next_forecast)
        
        return forecasts
    
    def seasonal_decomposition(self, period: int = 24) -> Tuple[List[float], List[float], List[float]]:
        """季节性分解"""
        if len(self.historical_data) < 2 * period:
            return self.historical_data, [0] * len(self.historical_data), [0] * len(self.historical_data)
        
        # 简化的季节性分解
        trend = []
        seasonal = []
        residual = []
        
        # 计算趋势（移动平均）
        for i in range(len(self.historical_data)):
            start = max(0, i - period // 2)
            end = min(len(self.historical_data), i + period // 2 + 1)
            trend.append(np.mean(self.historical_data[start:end]))
        
        # 计算季节性
        seasonal_pattern = []
        for i in range(period):
            pattern_values = []
            for j in range(i, len(self.historical_data), period):
                if j < len(self.historical_data):
                    pattern_values.append(self.historical_data[j] - trend[j])
            if pattern_values:
                seasonal_pattern.append(np.mean(pattern_values))
            else:
                seasonal_pattern.append(0)
        
        # 应用季节性模式
        for i in range(len(self.historical_data)):
            seasonal_idx = i % period
            seasonal.append(seasonal_pattern[seasonal_idx])
            residual.append(self.historical_data[i] - trend[i] - seasonal[i])
        
        return trend, seasonal, residual

def transportation_verification():
    """交通运输模型验证"""
    print("=== 交通运输模型验证 ===")
    
    # 交通流模型验证
    print("\n1. 交通流模型验证:")
    traffic_model = TrafficFlowModel(
        free_flow_speed=60.0,
        jam_density=120.0,
        capacity=1800.0
    )
    
    density = 30.0
    speed = greenshields_speed(traffic_model, density)
    flow = greenshields_flow(traffic_model, density)
    max_flow = max_flow_rate(traffic_model)
    
    print(f"密度: {density} 车辆/公里")
    print(f"速度: {speed:.2f} 公里/小时")
    print(f"流量: {flow:.2f} 车辆/小时")
    print(f"最大流量: {max_flow:.2f} 车辆/小时")
    
    # 路径规划验证
    print("\n2. 路径规划验证:")
    graph = Graph()
    graph.add_edge("A", "B", 10)
    graph.add_edge("B", "C", 15)
    graph.add_edge("A", "C", 25)
    graph.add_edge("B", "D", 20)
    graph.add_edge("C", "D", 10)
    
    result = dijkstra_shortest_path(graph, "A", "D")
    if result:
        distance, path = result
        print(f"最短距离: {distance}")
        print(f"路径: {' -> '.join(path)}")
    
    # 公共交通验证
    print("\n3. 公共交通验证:")
    routes = [
        BusRoute("R1", ["A", "B", "C"], 10.0, [5.0, 8.0]),
        BusRoute("R2", ["B", "D", "E"], 15.0, [12.0, 6.0])
    ]
    network = TransitNetwork(routes, {})
    optimizer = PublicTransitOptimizer(network)
    
    wait_time = optimizer.calculate_wait_time("R1")
    travel_time = optimizer.calculate_travel_time("A", "C")
    
    print(f"等待时间: {wait_time:.2f} 分钟")
    print(f"旅行时间: {travel_time:.2f} 分钟")
    
    # 交通信号验证
    print("\n4. 交通信号验证:")
    phases = [["North", "South"], ["East", "West"]]
    controller = TrafficSignalController("I1", phases, 90.0)
    
    flows = [800.0, 600.0]
    green_times = controller.optimize_signal_timing(flows)
    total_delay = controller.calculate_total_delay(flows, green_times)
    
    print(f"绿灯时间: {green_times}")
    print(f"总延误: {total_delay:.2f} 秒")
    
    # 交通预测验证
    print("\n5. 交通预测验证:")
    historical_data = [100, 110, 95, 105, 120, 115, 125, 130, 140, 135]
    predictor = TrafficPredictor(historical_data)
    
    es_forecast = predictor.exponential_smoothing_forecast(0.3)
    ma_forecast = predictor.moving_average_forecast()
    
    print(f"指数平滑预测: {es_forecast[-3:]}")
    print(f"移动平均预测: {ma_forecast[-3:]}")
    
    print("\n验证完成!")

if __name__ == "__main__":
    transportation_verification()
```

---

*最后更新: 2025-08-26*
*版本: 1.1.0*
