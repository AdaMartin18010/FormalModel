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
