# 4.3 数据结构模型 / Data Structure Models

## 目录 / Table of Contents

- [4.3 数据结构模型 / Data Structure Models](#43-数据结构模型--data-structure-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [4.3.1 基本数据结构 / Basic Data Structures](#431-基本数据结构--basic-data-structures)
    - [数组 / Arrays](#数组--arrays)
    - [链表 / Linked Lists](#链表--linked-lists)
    - [栈和队列 / Stacks and Queues](#栈和队列--stacks-and-queues)
  - [4.3.2 树结构 / Tree Structures](#432-树结构--tree-structures)
    - [二叉树 / Binary Trees](#二叉树--binary-trees)
    - [平衡树 / Balanced Trees](#平衡树--balanced-trees)
    - [B树和B+树 / B-Trees and B+ Trees](#b树和b树--b-trees-and-b-trees)
  - [4.3.3 图结构 / Graph Structures](#433-图结构--graph-structures)
    - [图的表示 / Graph Representation](#图的表示--graph-representation)
    - [图的遍历 / Graph Traversal](#图的遍历--graph-traversal)
    - [最短路径 / Shortest Paths](#最短路径--shortest-paths)
  - [4.3.4 散列结构 / Hash Structures](#434-散列结构--hash-structures)
    - [散列函数 / Hash Functions](#散列函数--hash-functions)
    - [冲突解决 / Collision Resolution](#冲突解决--collision-resolution)
    - [散列表 / Hash Tables](#散列表--hash-tables)
  - [4.3.5 高级数据结构 / Advanced Data Structures](#435-高级数据结构--advanced-data-structures)
    - [堆 / Heaps](#堆--heaps)
    - [并查集 / Union-Find](#并查集--union-find)
    - [跳表 / Skip Lists](#跳表--skip-lists)
  - [4.3.6 空间数据结构 / Spatial Data Structures](#436-空间数据结构--spatial-data-structures)
    - [四叉树 / Quadtrees](#四叉树--quadtrees)
    - [八叉树 / Octrees](#八叉树--octrees)
    - [R树 / R-Trees](#r树--r-trees)
  - [4.3.7 数据结构应用 / Data Structure Applications](#437-数据结构应用--data-structure-applications)
    - [数据库索引 / Database Indexing](#数据库索引--database-indexing)
    - [缓存系统 / Cache Systems](#缓存系统--cache-systems)
    - [内存管理 / Memory Management](#内存管理--memory-management)
  - [参考文献 / References](#参考文献--references)

---

## 4.3.1 基本数据结构 / Basic Data Structures

### 数组 / Arrays

**定义**: 连续内存空间存储的相同类型元素集合。

**操作复杂度**:
- 访问: $O(1)$
- 插入/删除: $O(n)$
- 搜索: $O(n)$

**动态数组**: 自动扩容的数组实现。

```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.array = [None] * self.capacity
    
    def append(self, item):
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.array[self.size] = item
        self.size += 1
    
    def _resize(self, new_capacity):
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
```

### 链表 / Linked Lists

**单链表**: 每个节点包含数据和指向下一个节点的指针。

**双链表**: 每个节点包含指向前后节点的指针。

**操作复杂度**:
- 插入/删除: $O(1)$ (给定位置)
- 搜索: $O(n)$
- 访问: $O(n)$

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete_node(self, key):
        temp = self.head
        if temp and temp.data == key:
            self.head = temp.next
            return
        
        while temp and temp.next:
            if temp.next.data == key:
                temp.next = temp.next.next
                return
            temp = temp.next
```

### 栈和队列 / Stacks and Queues

**栈 (LIFO)**:
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        return self.items[-1] if self.items else None
    
    def is_empty(self):
        return len(self.items) == 0
```

**队列 (FIFO)**:
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft() if self.items else None
    
    def front(self):
        return self.items[0] if self.items else None
```

---

## 4.3.2 树结构 / Tree Structures

### 二叉树 / Binary Trees

**定义**: 每个节点最多有两个子节点的树。

**遍历方式**:
- 前序遍历: 根-左-右
- 中序遍历: 左-根-右
- 后序遍历: 左-右-根
- 层序遍历: 按层访问

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print(node.val, end=" ")
            self.inorder_traversal(node.right)
    
    def preorder_traversal(self, node):
        if node:
            print(node.val, end=" ")
            self.preorder_traversal(node.left)
            self.preorder_traversal(node.right)
    
    def postorder_traversal(self, node):
        if node:
            self.postorder_traversal(node.left)
            self.postorder_traversal(node.right)
            print(node.val, end=" ")
```

### 平衡树 / Balanced Trees

**AVL树**: 自平衡二叉搜索树，左右子树高度差不超过1。

**红黑树**: 自平衡二叉搜索树，满足红黑性质。

**操作复杂度**: 插入、删除、搜索均为 $O(\log n)$。

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        
        x.right = y
        y.left = T2
        
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        
        return x
```

### B树和B+树 / B-Trees and B+ Trees

**B树**: 多路平衡搜索树，用于磁盘存储。

**B+树**: B树的变种，所有数据都在叶子节点。

**特点**: 减少磁盘I/O次数。

---

## 4.3.3 图结构 / Graph Structures

### 图的表示 / Graph Representation

**邻接矩阵**: $n \times n$ 矩阵，$A[i][j] = 1$ 表示边存在。

**邻接表**: 每个顶点维护一个邻接顶点列表。

```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v, weight=1):
        self.graph[u].append((v, weight))
        # 无向图
        self.graph[v].append((u, weight))
    
    def print_graph(self):
        for i in range(self.V):
            print(f"Vertex {i}:", end=" ")
            for j, weight in self.graph[i]:
                print(f"({j}, {weight})", end=" ")
            print()
```

### 图的遍历 / Graph Traversal

**深度优先搜索 (DFS)**:
```python
def dfs(self, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=" ")
    
    for neighbor, _ in self.graph[start]:
        if neighbor not in visited:
            self.dfs(neighbor, visited)
```

**广度优先搜索 (BFS)**:
```python
from collections import deque

def bfs(self, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")
        
        for neighbor, _ in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 最短路径 / Shortest Paths

**Dijkstra算法**: 单源最短路径。

**Floyd-Warshall算法**: 所有顶点对最短路径。

---

## 4.3.4 散列结构 / Hash Structures

### 散列函数 / Hash Functions

**理想散列函数**: 均匀分布，最小化冲突。

**常见散列函数**:
- 除留余数法: $h(k) = k \bmod m$
- 乘法散列: $h(k) = \lfloor m(kA \bmod 1) \rfloor$
- 全域散列: 随机选择散列函数

```python
def hash_function(key, size):
    return hash(key) % size

def hash_string(s, size):
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % size
    return hash_value
```

### 冲突解决 / Collision Resolution

**开放寻址法**:
- 线性探测: $h_i(k) = (h(k) + i) \bmod m$
- 二次探测: $h_i(k) = (h(k) + i^2) \bmod m$
- 双重散列: $h_i(k) = (h_1(k) + ih_2(k)) \bmod m$

**链地址法**: 每个桶维护一个链表。

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def insert(self, key, value):
        hash_key = hash_function(key, self.size)
        for item in self.table[hash_key]:
            if item[0] == key:
                item[1] = value
                return
        self.table[hash_key].append([key, value])
    
    def get(self, key):
        hash_key = hash_function(key, self.size)
        for item in self.table[hash_key]:
            if item[0] == key:
                return item[1]
        return None
```

### 散列表 / Hash Tables

**负载因子**: $\alpha = n/m$，其中 $n$ 是元素数，$m$ 是桶数。

**动态扩容**: 当 $\alpha > \text{threshold}$ 时扩容。

---

## 4.3.5 高级数据结构 / Advanced Data Structures

### 堆 / Heaps

**最大堆**: 父节点值大于等于子节点值。

**最小堆**: 父节点值小于等于子节点值。

**操作复杂度**:
- 插入: $O(\log n)$
- 删除: $O(\log n)$
- 查找最大/最小: $O(1)$

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, item):
        heapq.heappush(self.heap, item)
    
    def pop(self):
        return heapq.heappop(self.heap)
    
    def peek(self):
        return self.heap[0] if self.heap else None
```

### 并查集 / Union-Find

**路径压缩**: 查找时压缩路径。

**按秩合并**: 将较小的树合并到较大的树。

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
```

### 跳表 / Skip Lists

**定义**: 多层链表结构，提供对数时间复杂度的搜索。

**层数**: 随机决定，期望层数为 $\log n$。

---

## 4.3.6 空间数据结构 / Spatial Data Structures

### 四叉树 / Quadtrees

**定义**: 二维空间分割树，每个节点最多有四个子节点。

**应用**: 图像压缩、碰撞检测、空间索引。

```python
class QuadTreeNode:
    def __init__(self, x, y, width, height):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.children = [None] * 4
        self.points = []
    
    def subdivide(self):
        half_w, half_h = self.width // 2, self.height // 2
        mid_x, mid_y = self.x + half_w, self.y + half_h
        
        self.children[0] = QuadTreeNode(self.x, self.y, half_w, half_h)
        self.children[1] = QuadTreeNode(mid_x, self.y, half_w, half_h)
        self.children[2] = QuadTreeNode(self.x, mid_y, half_w, half_h)
        self.children[3] = QuadTreeNode(mid_x, mid_y, half_w, half_h)
```

### 八叉树 / Octrees

**定义**: 三维空间分割树，每个节点最多有八个子节点。

**应用**: 3D图形、体积渲染、空间索引。

### R树 / R-Trees

**定义**: 多维空间索引树，用于空间数据。

**应用**: 地理信息系统、数据库空间索引。

---

## 4.3.7 数据结构应用 / Data Structure Applications

### 数据库索引 / Database Indexing

**B+树索引**: 数据库中最常用的索引结构。

**散列索引**: 等值查询的高效索引。

**位图索引**: 低基数列的压缩索引。

### 缓存系统 / Cache Systems

**LRU缓存**: 最近最少使用策略。

**LFU缓存**: 最不经常使用策略。

**Redis**: 内存数据结构存储系统。

### 内存管理 / Memory Management

**内存池**: 预分配内存块，减少碎片。

**垃圾回收**: 自动内存管理。

**内存对齐**: 提高访问效率。

---

## 参考文献 / References

1. Cormen, T. H., et al. (2009). Introduction to Algorithms. MIT Press.
2. Sedgewick, R., & Wayne, K. (2011). Algorithms. Addison-Wesley.
3. Knuth, D. E. (1997). The Art of Computer Programming. Addison-Wesley.
4. Okasaki, C. (1999). Purely Functional Data Structures. Cambridge University Press.
5. Mehlhorn, K., & Sanders, P. (2008). Algorithms and Data Structures. Springer.

---

*最后更新: 2025-08-01*
*版本: 1.0.0* 