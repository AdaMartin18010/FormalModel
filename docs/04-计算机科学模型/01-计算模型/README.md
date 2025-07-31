# 4.1 计算模型 / Computational Models

## 目录 / Table of Contents

- [4.1 计算模型 / Computational Models](#41-计算模型--computational-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [4.1.1 图灵机模型 / Turing Machine Models](#411-图灵机模型--turing-machine-models)
    - [图灵机定义 / Turing Machine Definition](#图灵机定义--turing-machine-definition)
    - [转移函数 / Transition Function](#转移函数--transition-function)
    - [图灵机计算 / Turing Machine Computation](#图灵机计算--turing-machine-computation)
    - [停机问题 / Halting Problem](#停机问题--halting-problem)
  - [4.1.2 有限状态机模型 / Finite State Machine Models](#412-有限状态机模型--finite-state-machine-models)
    - [确定性有限自动机 (DFA) / Deterministic Finite Automaton](#确定性有限自动机-dfa--deterministic-finite-automaton)
    - [非确定性有限自动机 (NFA) / Nondeterministic Finite Automaton](#非确定性有限自动机-nfa--nondeterministic-finite-automaton)
    - [正则表达式 / Regular Expressions](#正则表达式--regular-expressions)
  - [4.1.3 λ演算模型 / Lambda Calculus Models](#413-λ演算模型--lambda-calculus-models)
    - [λ项 / Lambda Terms](#λ项--lambda-terms)
    - [β归约 / Beta Reduction](#β归约--beta-reduction)
    - [Church编码 / Church Encoding](#church编码--church-encoding)
  - [4.1.4 递归函数模型 / Recursive Function Models](#414-递归函数模型--recursive-function-models)
    - [原始递归函数 / Primitive Recursive Functions](#原始递归函数--primitive-recursive-functions)
    - [μ递归函数 / μ-Recursive Functions](#μ递归函数--μ-recursive-functions)
  - [4.1.5 寄存器机模型 / Register Machine Models](#415-寄存器机模型--register-machine-models)
    - [RAM模型 / Random Access Machine Model](#ram模型--random-access-machine-model)
    - [复杂度分析 / Complexity Analysis](#复杂度分析--complexity-analysis)
  - [4.1.6 并行计算模型 / Parallel Computing Models](#416-并行计算模型--parallel-computing-models)
    - [PRAM模型 / Parallel Random Access Machine Model](#pram模型--parallel-random-access-machine-model)
    - [网络模型 / Network Models](#网络模型--network-models)
  - [4.1.7 量子计算模型 / Quantum Computing Models](#417-量子计算模型--quantum-computing-models)
    - [量子比特 / Qubit](#量子比特--qubit)
    - [量子门 / Quantum Gates](#量子门--quantum-gates)
    - [量子算法 / Quantum Algorithms](#量子算法--quantum-algorithms)
  - [4.1.8 实现与应用 / Implementation and Applications](#418-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [理论计算机科学 / Theoretical Computer Science](#理论计算机科学--theoretical-computer-science)
      - [软件工程 / Software Engineering](#软件工程--software-engineering)
      - [人工智能 / Artificial Intelligence](#人工智能--artificial-intelligence)
  - [参考文献 / References](#参考文献--references)

---

## 4.1.1 图灵机模型 / Turing Machine Models

### 图灵机定义 / Turing Machine Definition

**图灵机** 是一个七元组 $M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$，其中：

- $Q$: 有限状态集合
- $\Sigma$: 输入字母表
- $\Gamma$: 带字母表
- $\delta$: 转移函数
- $q_0$: 初始状态
- $B$: 空白符号
- $F$: 接受状态集合

### 转移函数 / Transition Function

$$\delta: Q \times \Gamma \to Q \times \Gamma \times \{L, R, N\}$$

其中 $L, R, N$ 分别表示左移、右移、不动。

### 图灵机计算 / Turing Machine Computation

**配置**: $(q, \alpha, i)$，其中 $q$ 是当前状态，$\alpha$ 是带内容，$i$ 是读写头位置。

**计算步骤**: $(q, \alpha, i) \vdash (q', \alpha', i')$ 如果 $\delta(q, \alpha_i) = (q', b, d)$。

### 停机问题 / Halting Problem

**定理**: 停机问题是不可判定的。

**证明**: 假设存在停机判定器 $H$，构造矛盾机器 $D$。

---

## 4.1.2 有限状态机模型 / Finite State Machine Models

### 确定性有限自动机 (DFA) / Deterministic Finite Automaton

**定义**: $M = (Q, \Sigma, \delta, q_0, F)$

**转移函数**: $\delta: Q \times \Sigma \to Q$

**接受条件**: 从初始状态开始，按照输入串转移，最终到达接受状态。

### 非确定性有限自动机 (NFA) / Nondeterministic Finite Automaton

**定义**: $M = (Q, \Sigma, \delta, q_0, F)$

**转移函数**: $\delta: Q \times \Sigma \to \mathcal{P}(Q)$

**接受条件**: 存在一条从初始状态到接受状态的路径。

### 正则表达式 / Regular Expressions

**基本操作**:

- **连接**: $R_1 \cdot R_2$
- **选择**: $R_1 | R_2$
- **重复**: $R^*$

**等价性**: 正则表达式、DFA、NFA 三者等价。

---

## 4.1.3 λ演算模型 / Lambda Calculus Models

### λ项 / Lambda Terms

**语法**:
$$M ::= x \mid \lambda x.M \mid M N$$

其中：

- $x$: 变量
- $\lambda x.M$: 抽象
- $M N$: 应用

### β归约 / Beta Reduction

**β归约规则**: $(\lambda x.M)N \to M[x := N]$

**α转换**: $\lambda x.M \equiv \lambda y.M[x := y]$ (如果 $y$ 不在 $M$ 中自由出现)

**η转换**: $\lambda x.M x \equiv M$ (如果 $x$ 不在 $M$ 中自由出现)

### Church编码 / Church Encoding

**自然数**: $\overline{n} = \lambda f.\lambda x.f^n(x)$

**布尔值**: $\text{true} = \lambda x.\lambda y.x$, $\text{false} = \lambda x.\lambda y.y$

**对**: $\langle M, N \rangle = \lambda f.f M N$

---

## 4.1.4 递归函数模型 / Recursive Function Models

### 原始递归函数 / Primitive Recursive Functions

**基本函数**:

- **零函数**: $Z(x) = 0$
- **后继函数**: $S(x) = x + 1$
- **投影函数**: $P_i^n(x_1, \ldots, x_n) = x_i$

**复合**: $h(x_1, \ldots, x_n) = f(g_1(x_1, \ldots, x_n), \ldots, g_m(x_1, \ldots, x_n))$

**原始递归**:
$$h(x_1, \ldots, x_{n-1}, 0) = f(x_1, \ldots, x_{n-1})$$
$$h(x_1, \ldots, x_{n-1}, y+1) = g(x_1, \ldots, x_{n-1}, y, h(x_1, \ldots, x_{n-1}, y))$$

### μ递归函数 / μ-Recursive Functions

**μ算子**: $\mu y[R(x_1, \ldots, x_n, y)]$ 表示最小的 $y$ 使得 $R(x_1, \ldots, x_n, y)$ 为真。

**Church-Turing论题**: 所有可计算函数都是μ递归函数。

---

## 4.1.5 寄存器机模型 / Register Machine Models

### RAM模型 / Random Access Machine Model

**寄存器**: $R_0, R_1, R_2, \ldots$

**指令集**:

- **LOAD**: $R_i \leftarrow M[j]$
- **STORE**: $M[j] \leftarrow R_i$
- **ADD**: $R_i \leftarrow R_j + R_k$
- **SUB**: $R_i \leftarrow R_j - R_k$
- **MULT**: $R_i \leftarrow R_j \times R_k$
- **DIV**: $R_i \leftarrow R_j \div R_k$
- **JUMP**: 无条件跳转
- **JZERO**: 零跳转

### 复杂度分析 / Complexity Analysis

**时间复杂度**: 指令执行次数

**空间复杂度**: 使用的寄存器数量

**多项式时间**: $O(n^k)$ 其中 $k$ 是常数

---

## 4.1.6 并行计算模型 / Parallel Computing Models

### PRAM模型 / Parallel Random Access Machine Model

**共享内存**: 所有处理器共享一个内存空间

**访问模式**:

- **EREW**: 互斥读互斥写
- **CREW**: 并发读互斥写
- **CRCW**: 并发读并发写

**复杂度度量**: 处理器数量 × 时间

### 网络模型 / Network Models

**拓扑结构**:

- **线性阵列**: $P_1 - P_2 - \cdots - P_n$
- **环**: $P_1 - P_2 - \cdots - P_n - P_1$
- **网格**: $n \times n$ 网格
- **超立方**: $Q_d$ 维超立方

**通信复杂度**: 消息传递次数

---

## 4.1.7 量子计算模型 / Quantum Computing Models

### 量子比特 / Qubit

**状态**: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$

**归一化**: $|\alpha|^2 + |\beta|^2 = 1$

**测量**: 以概率 $|\alpha|^2$ 得到 $|0\rangle$，以概率 $|\beta|^2$ 得到 $|1\rangle$

### 量子门 / Quantum Gates

**Hadamard门**: $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$

**CNOT门**: $\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$

**量子傅里叶变换**: $F_N = \frac{1}{\sqrt{N}}\sum_{j,k=0}^{N-1} e^{2\pi ijk/N}|j\rangle\langle k|$

### 量子算法 / Quantum Algorithms

**Shor算法**: 整数分解，时间复杂度 $O((\log n)^3)$

**Grover算法**: 无序搜索，时间复杂度 $O(\sqrt{n})$

---

## 4.1.8 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Symbol {
    Zero,
    One,
    Blank,
}

#[derive(Debug, Clone)]
pub enum Direction {
    Left,
    Right,
    Stay,
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub next_state: String,
    pub write_symbol: Symbol,
    pub direction: Direction,
}

#[derive(Debug)]
pub struct TuringMachine {
    pub states: Vec<String>,
    pub alphabet: Vec<Symbol>,
    pub tape_alphabet: Vec<Symbol>,
    pub transitions: HashMap<(String, Symbol), Transition>,
    pub initial_state: String,
    pub accept_states: Vec<String>,
    pub current_state: String,
    pub tape: Vec<Symbol>,
    pub head_position: i32,
}

impl TuringMachine {
    pub fn new(
        states: Vec<String>,
        alphabet: Vec<Symbol>,
        tape_alphabet: Vec<Symbol>,
        transitions: HashMap<(String, Symbol), Transition>,
        initial_state: String,
        accept_states: Vec<String>,
    ) -> Self {
        Self {
            states,
            alphabet,
            tape_alphabet,
            transitions,
            initial_state: initial_state.clone(),
            accept_states,
            current_state: initial_state,
            tape: vec![Symbol::Blank],
            head_position: 0,
        }
    }
    
    pub fn step(&mut self) -> bool {
        let current_symbol = self.get_current_symbol();
        let key = (self.current_state.clone(), current_symbol.clone());
        
        if let Some(transition) = self.transitions.get(&key) {
            // 写入符号
            self.set_current_symbol(transition.write_symbol.clone());
            
            // 移动读写头
            match transition.direction {
                Direction::Left => self.head_position -= 1,
                Direction::Right => self.head_position += 1,
                Direction::Stay => {}
            }
            
            // 更新状态
            self.current_state = transition.next_state.clone();
            
            // 扩展磁带
            if self.head_position < 0 {
                self.tape.insert(0, Symbol::Blank);
                self.head_position = 0;
            } else if self.head_position >= self.tape.len() as i32 {
                self.tape.push(Symbol::Blank);
            }
            
            true
        } else {
            false
        }
    }
    
    pub fn get_current_symbol(&self) -> Symbol {
        if self.head_position >= 0 && self.head_position < self.tape.len() as i32 {
            self.tape[self.head_position as usize].clone()
        } else {
            Symbol::Blank
        }
    }
    
    pub fn set_current_symbol(&mut self, symbol: Symbol) {
        if self.head_position >= 0 && self.head_position < self.tape.len() as i32 {
            self.tape[self.head_position as usize] = symbol;
        }
    }
    
    pub fn is_accepting(&self) -> bool {
        self.accept_states.contains(&self.current_state)
    }
    
    pub fn run(&mut self, input: Vec<Symbol>) -> bool {
        // 初始化磁带
        self.tape = input;
        self.tape.push(Symbol::Blank);
        self.current_state = self.initial_state.clone();
        self.head_position = 0;
        
        // 运行直到停机
        let mut steps = 0;
        while steps < 1000 { // 防止无限循环
            if !self.step() {
                break;
            }
            steps += 1;
        }
        
        self.is_accepting()
    }
}

// 使用示例：识别包含偶数个1的字符串
fn main() {
    let mut transitions = HashMap::new();
    
    // 状态转移函数
    transitions.insert(
        ("q0".to_string(), Symbol::Zero),
        Transition {
            next_state: "q0".to_string(),
            write_symbol: Symbol::Zero,
            direction: Direction::Right,
        },
    );
    
    transitions.insert(
        ("q0".to_string(), Symbol::One),
        Transition {
            next_state: "q1".to_string(),
            write_symbol: Symbol::One,
            direction: Direction::Right,
        },
    );
    
    transitions.insert(
        ("q1".to_string(), Symbol::Zero),
        Transition {
            next_state: "q1".to_string(),
            write_symbol: Symbol::Zero,
            direction: Direction::Right,
        },
    );
    
    transitions.insert(
        ("q1".to_string(), Symbol::One),
        Transition {
            next_state: "q0".to_string(),
            write_symbol: Symbol::One,
            direction: Direction::Right,
        },
    );
    
    let tm = TuringMachine::new(
        vec!["q0".to_string(), "q1".to_string()],
        vec![Symbol::Zero, Symbol::One],
        vec![Symbol::Zero, Symbol::One, Symbol::Blank],
        transitions,
        "q0".to_string(),
        vec!["q0".to_string()],
    );
    
    let mut machine = tm;
    let input = vec![Symbol::One, Symbol::Zero, Symbol::One, Symbol::One];
    let result = machine.run(input);
    println!("Accepts: {}", result);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module ComputationalModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (find)

-- 符号类型
data Symbol = Zero | One | Blank deriving (Show, Eq, Ord)

-- 方向类型
data Direction = Left | Right | Stay deriving (Show)

-- 转移函数
data Transition = Transition {
    nextState :: String,
    writeSymbol :: Symbol,
    direction :: Direction
} deriving (Show)

-- 图灵机
data TuringMachine = TuringMachine {
    states :: [String],
    alphabet :: [Symbol],
    tapeAlphabet :: [Symbol],
    transitions :: Map (String, Symbol) Transition,
    initialState :: String,
    acceptStates :: [String],
    currentState :: String,
    tape :: [Symbol],
    headPosition :: Int
} deriving (Show)

-- 创建图灵机
newTuringMachine :: [String] -> [Symbol] -> [Symbol] -> Map (String, Symbol) Transition -> String -> [String] -> TuringMachine
newTuringMachine states alphabet tapeAlphabet transitions initial acceptStates = TuringMachine {
    states = states,
    alphabet = alphabet,
    tapeAlphabet = tapeAlphabet,
    transitions = transitions,
    initialState = initial,
    acceptStates = acceptStates,
    currentState = initial,
    tape = [Blank],
    headPosition = 0
}

-- 获取当前符号
getCurrentSymbol :: TuringMachine -> Symbol
getCurrentSymbol tm = tape tm !! headPosition tm

-- 设置当前符号
setCurrentSymbol :: Symbol -> TuringMachine -> TuringMachine
setCurrentSymbol symbol tm = tm { tape = take (headPosition tm) (tape tm) ++ [symbol] ++ drop (headPosition tm + 1) (tape tm) }

-- 执行一步
step :: TuringMachine -> Maybe TuringMachine
step tm = do
    let currentSymbol = getCurrentSymbol tm
    transition <- Map.lookup (currentState tm, currentSymbol) (transitions tm)
    
    let newTape = setCurrentSymbol (writeSymbol transition) tm
    let newHeadPos = case direction transition of
                        Left -> headPosition tm - 1
                        Right -> headPosition tm + 1
                        Stay -> headPosition tm
    let newState = nextState transition
    
    return tm {
        currentState = newState,
        tape = tape newTape,
        headPosition = newHeadPos
    }

-- 检查是否接受
isAccepting :: TuringMachine -> Bool
isAccepting tm = currentState tm `elem` acceptStates tm

-- 运行图灵机
run :: [Symbol] -> TuringMachine -> Bool
run input tm = go 1000 (tm { tape = input ++ [Blank], headPosition = 0 })
  where
    go 0 _ = False  -- 防止无限循环
    go steps machine = case step machine of
        Nothing -> isAccepting machine
        Just newMachine -> go (steps - 1) newMachine

-- 示例：识别包含偶数个1的字符串
example :: IO ()
example = do
    let transitions = Map.fromList [
            (("q0", Zero), Transition "q0" Zero Right),
            (("q0", One), Transition "q1" One Right),
            (("q1", Zero), Transition "q1" Zero Right),
            (("q1", One), Transition "q0" One Right)
        ]
        
        tm = newTuringMachine 
                ["q0", "q1"] 
                [Zero, One] 
                [Zero, One, Blank] 
                transitions 
                "q0" 
                ["q0"]
        
        input = [One, Zero, One, One]
        result = run input tm
    
    putStrLn $ "Input: " ++ show input
    putStrLn $ "Accepts: " ++ show result
```

### 应用领域 / Application Domains

#### 理论计算机科学 / Theoretical Computer Science

- **可计算性理论**: 研究哪些问题可以计算
- **复杂度理论**: 研究计算资源的使用
- **形式语言理论**: 研究语言的层次结构

#### 软件工程 / Software Engineering

- **编译器设计**: 词法分析、语法分析
- **程序验证**: 形式化方法、模型检查
- **并发编程**: 进程间通信、同步

#### 人工智能 / Artificial Intelligence

- **机器学习**: 神经网络、深度学习
- **自然语言处理**: 语法分析、语义理解
- **知识表示**: 逻辑推理、专家系统

---

## 参考文献 / References

1. Sipser, M. (2012). Introduction to the Theory of Computation. Cengage Learning.
2. Hopcroft, J. E., Motwani, R., & Ullman, J. D. (2006). Introduction to Automata Theory, Languages, and Computation. Pearson.
3. Barendregt, H. P. (1984). The Lambda Calculus: Its Syntax and Semantics. North-Holland.
4. Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information. Cambridge University Press.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
