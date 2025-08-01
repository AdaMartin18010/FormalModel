# 4.2 算法模型 / Algorithmic Models

## 目录 / Table of Contents

- [4.2 算法模型 / Algorithmic Models](#42-算法模型--algorithmic-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [4.2.1 算法复杂度理论 / Algorithm Complexity Theory](#421-算法复杂度理论--algorithm-complexity-theory)
    - [时间复杂度 / Time Complexity](#时间复杂度--time-complexity)
    - [空间复杂度 / Space Complexity](#空间复杂度--space-complexity)
    - [渐进分析 / Asymptotic Analysis](#渐进分析--asymptotic-analysis)
  - [4.2.2 分治算法 / Divide and Conquer Algorithms](#422-分治算法--divide-and-conquer-algorithms)
    - [归并排序 / Merge Sort](#归并排序--merge-sort)
    - [快速排序 / Quick Sort](#快速排序--quick-sort)
    - [分治策略 / Divide and Conquer Strategy](#分治策略--divide-and-conquer-strategy)
  - [4.2.3 动态规划 / Dynamic Programming](#423-动态规划--dynamic-programming)
    - [最优子结构 / Optimal Substructure](#最优子结构--optimal-substructure)
    - [重叠子问题 / Overlapping Subproblems](#重叠子问题--overlapping-subproblems)
    - [经典问题 / Classical Problems](#经典问题--classical-problems)
  - [4.2.4 贪心算法 / Greedy Algorithms](#424-贪心算法--greedy-algorithms)
    - [贪心选择性质 / Greedy Choice Property](#贪心选择性质--greedy-choice-property)
    - [1最优子结构 / Optimal Substructure](#1最优子结构--optimal-substructure)
    - [贪心算法应用 / Greedy Algorithm Applications](#贪心算法应用--greedy-algorithm-applications)
  - [4.2.5 图算法 / Graph Algorithms](#425-图算法--graph-algorithms)
    - [最短路径算法 / Shortest Path Algorithms](#最短路径算法--shortest-path-algorithms)
    - [最小生成树 / Minimum Spanning Tree](#最小生成树--minimum-spanning-tree)
    - [网络流算法 / Network Flow Algorithms](#网络流算法--network-flow-algorithms)
  - [4.2.6 随机算法 / Randomized Algorithms](#426-随机算法--randomized-algorithms)
    - [拉斯维加斯算法 / Las Vegas Algorithms](#拉斯维加斯算法--las-vegas-algorithms)
    - [蒙特卡洛算法 / Monte Carlo Algorithms](#蒙特卡洛算法--monte-carlo-algorithms)
    - [随机化技术 / Randomization Techniques](#随机化技术--randomization-techniques)
  - [4.2.7 近似算法 / Approximation Algorithms](#427-近似算法--approximation-algorithms)
    - [近似比 / Approximation Ratio](#近似比--approximation-ratio)
    - [PTAS和FPTAS / PTAS and FPTAS](#ptas和fptas--ptas-and-fptas)
    - [启发式算法 / Heuristic Algorithms](#启发式算法--heuristic-algorithms)
  - [参考文献 / References](#参考文献--references)

---

## 4.2.1 算法复杂度理论 / Algorithm Complexity Theory

### 时间复杂度 / Time Complexity

**大O记号**: $O(f(n))$ 表示算法的渐进上界。

**常见复杂度**:

- $O(1)$: 常数时间
- $O(\log n)$: 对数时间
- $O(n)$: 线性时间
- $O(n \log n)$: 线性对数时间
- $O(n^2)$: 二次时间
- $O(2^n)$: 指数时间

**复杂度分析**:

```python
def example_algorithm(n):
    result = 0
    for i in range(n):        # O(n)
        for j in range(n):    # O(n)
            result += i * j    # O(1)
    return result              # 总复杂度: O(n²)
```

### 空间复杂度 / Space Complexity

**空间复杂度**: 算法执行过程中所需的额外空间。

**递归空间**: 递归调用栈的深度。

**示例**:

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    # 空间复杂度: O(n) - 递归栈深度
```

### 渐进分析 / Asymptotic Analysis

**大Ω记号**: $\Omega(f(n))$ 表示渐进下界。

**大Θ记号**: $\Theta(f(n))$ 表示紧确界。

**小o记号**: $o(f(n))$ 表示严格上界。

---

## 4.2.2 分治算法 / Divide and Conquer Algorithms

### 归并排序 / Merge Sort

**算法思想**: 将数组分成两半，递归排序，然后合并。

**时间复杂度**: $O(n \log n)$

**空间复杂度**: $O(n)$

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### 快速排序 / Quick Sort

**算法思想**: 选择基准元素，分区，递归排序。

**平均时间复杂度**: $O(n \log n)$

**最坏时间复杂度**: $O(n^2)$

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```

### 分治策略 / Divide and Conquer Strategy

**三步法**:

1. **分解**: 将问题分解为子问题
2. **解决**: 递归解决子问题
3. **合并**: 将子问题的解合并

**主定理**: 对于递归式 $T(n) = aT(n/b) + f(n)$，

- 如果 $f(n) = O(n^{\log_b a - \epsilon})$，则 $T(n) = \Theta(n^{\log_b a})$
- 如果 $f(n) = \Theta(n^{\log_b a})$，则 $T(n) = \Theta(n^{\log_b a} \log n)$
- 如果 $f(n) = \Omega(n^{\log_b a + \epsilon})$，则 $T(n) = \Theta(f(n))$

---

## 4.2.3 动态规划 / Dynamic Programming

### 最优子结构 / Optimal Substructure

**定义**: 问题的最优解包含其子问题的最优解。

**示例**: 最长公共子序列 (LCS)

```python
def lcs(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### 重叠子问题 / Overlapping Subproblems

**定义**: 递归算法重复解决相同的子问题。

**记忆化**: 存储已计算的子问题结果。

```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### 经典问题 / Classical Problems

**背包问题**:

```python
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

---

## 4.2.4 贪心算法 / Greedy Algorithms

### 贪心选择性质 / Greedy Choice Property

**定义**: 每一步都选择当前看起来最优的选择。

**示例**: 活动选择问题

```python
def activity_selection(start, finish):
    n = len(start)
    selected = [0]  # 选择第一个活动
    j = 0
    
    for i in range(1, n):
        if start[i] >= finish[j]:
            selected.append(i)
            j = i
    
    return selected
```

### 1最优子结构 / Optimal Substructure

**定义**: 贪心选择后，剩余问题的最优解与原问题的最优解一致。

### 贪心算法应用 / Greedy Algorithm Applications

**霍夫曼编码**:

```python
import heapq

def huffman_encoding(freq):
    heap = [[weight, [[symbol, ""]]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return heap[0][1:]
```

---

## 4.2.5 图算法 / Graph Algorithms

### 最短路径算法 / Shortest Path Algorithms

**Dijkstra算法**:

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

**Floyd-Warshall算法**:

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('infinity')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        for j, weight in graph[i].items():
            dist[i][j] = weight
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
```

### 最小生成树 / Minimum Spanning Tree

**Kruskal算法**:

```python
def kruskal(graph):
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))
    edges.sort()
    
    parent = {node: node for node in graph}
    
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    
    def union(u, v):
        parent[find(u)] = find(v)
    
    mst = []
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, weight))
    
    return mst
```

### 网络流算法 / Network Flow Algorithms

**Ford-Fulkerson算法**:

```python
def ford_fulkerson(graph, source, sink):
    def bfs(graph, source, sink, parent):
        visited = [False] * len(graph)
        queue = [source]
        visited[source] = True
        
        while queue:
            u = queue.pop(0)
            for v, capacity in enumerate(graph[u]):
                if not visited[v] and capacity > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False
    
    max_flow = 0
    parent = [-1] * len(graph)
    
    while bfs(graph, source, sink, parent):
        path_flow = float('infinity')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, graph[u][v])
            v = parent[v]
        
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        
        max_flow += path_flow
    
    return max_flow
```

---

## 4.2.6 随机算法 / Randomized Algorithms

### 拉斯维加斯算法 / Las Vegas Algorithms

**定义**: 总是产生正确结果，但运行时间随机。

**示例**: 随机快速排序

```python
import random

def randomized_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return randomized_quick_sort(left) + middle + randomized_quick_sort(right)
```

### 蒙特卡洛算法 / Monte Carlo Algorithms

**定义**: 可能产生错误结果，但错误概率可控。

**示例**: 素数测试

```python
import random

def miller_rabin(n, k=5):
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True
```

### 随机化技术 / Randomization Techniques

**随机化哈希**: 减少哈希冲突。

**随机化搜索**: 避免局部最优。

---

## 4.2.7 近似算法 / Approximation Algorithms

### 近似比 / Approximation Ratio

**定义**: 近似解与最优解的比值。

**示例**: 旅行商问题的2-近似算法

```python
def tsp_2_approximation(graph):
    # 最小生成树
    mst = kruskal(graph)
    
    # 欧拉回路
    euler_tour = eulerian_tour(mst)
    
    # 哈密顿回路
    hamiltonian_cycle = shortcut(euler_tour)
    
    return hamiltonian_cycle
```

### PTAS和FPTAS / PTAS and FPTAS

**PTAS**: 多项式时间近似方案。

**FPTAS**: 完全多项式时间近似方案。

### 启发式算法 / Heuristic Algorithms

**遗传算法**:

```python
def genetic_algorithm(population, fitness, generations):
    for _ in range(generations):
        new_population = []
        for _ in range(len(population)):
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        population = new_population
    return population
```

**模拟退火**:

```python
def simulated_annealing(initial_solution, temperature, cooling_rate):
    current = initial_solution
    best = current
    
    while temperature > 0.1:
        neighbor = generate_neighbor(current)
        delta_e = evaluate(neighbor) - evaluate(current)
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current = neighbor
            
        if evaluate(current) < evaluate(best):
            best = current
            
        temperature *= cooling_rate
    
    return best
```

---

## 参考文献 / References

1. Cormen, T. H., et al. (2009). Introduction to Algorithms. MIT Press.
2. Kleinberg, J., & Tardos, É. (2006). Algorithm Design. Pearson.
3. Dasgupta, S., et al. (2008). Algorithms. McGraw-Hill.
4. Motwani, R., & Raghavan, P. (1995). Randomized Algorithms. Cambridge University Press.
5. Vazirani, V. V. (2001). Approximation Algorithms. Springer.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
