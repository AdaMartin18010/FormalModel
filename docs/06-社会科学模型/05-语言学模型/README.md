# 6.5 语言学模型 / Linguistic Models

## 目录 / Table of Contents

- [6.5 语言学模型 / Linguistic Models](#65-语言学模型--linguistic-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [6.5.1 语音学模型 / Phonological Models](#651-语音学模型--phonological-models)
    - [音位模型 / Phoneme Models](#音位模型--phoneme-models)
    - [音节结构 / Syllable Structure](#音节结构--syllable-structure)
    - [音变规则 / Sound Change Rules](#音变规则--sound-change-rules)
  - [6.5.2 形态学模型 / Morphological Models](#652-形态学模型--morphological-models)
    - [词素分析 / Morpheme Analysis](#词素分析--morpheme-analysis)
    - [构词规则 / Word Formation Rules](#构词规则--word-formation-rules)
    - [屈折变化 / Inflectional Patterns](#屈折变化--inflectional-patterns)
  - [6.5.3 句法学模型 / Syntactic Models](#653-句法学模型--syntactic-models)
    - [短语结构语法 / Phrase Structure Grammar](#短语结构语法--phrase-structure-grammar)
    - [依存语法 / Dependency Grammar](#依存语法--dependency-grammar)
    - [转换语法 / Transformational Grammar](#转换语法--transformational-grammar)
  - [6.5.4 语义学模型 / Semantic Models](#654-语义学模型--semantic-models)
    - [词义模型 / Lexical Semantics](#词义模型--lexical-semantics)
    - [组合语义 / Compositional Semantics](#组合语义--compositional-semantics)
    - [语用学 / Pragmatics](#语用学--pragmatics)
  - [6.5.5 语料库语言学模型 / Corpus Linguistics Models](#655-语料库语言学模型--corpus-linguistics-models)
    - [频率分析 / Frequency Analysis](#频率分析--frequency-analysis)
    - [共现分析 / Co-occurrence Analysis](#共现分析--co-occurrence-analysis)
    - [语料库统计 / Corpus Statistics](#语料库统计--corpus-statistics)
  - [6.5.6 实现与应用 / Implementation and Applications](#656-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [自然语言处理 / Natural Language Processing](#自然语言处理--natural-language-processing)
      - [机器翻译 / Machine Translation](#机器翻译--machine-translation)
      - [语音识别 / Speech Recognition](#语音识别--speech-recognition)
  - [参考文献 / References](#参考文献--references)

---

## 6.5.1 语音学模型 / Phonological Models

### 音位模型 / Phoneme Models

**音位特征矩阵**: $F = [f_{ij}]$ where $f_{ij} \in \{+1, -1, 0\}$

**音位距离**: $d(p_1, p_2) = \sum_{i=1}^n |f_{1i} - f_{2i}|$

**音位相似度**: $sim(p_1, p_2) = \frac{\sum_{i=1}^n f_{1i} \cdot f_{2i}}{\sqrt{\sum_{i=1}^n f_{1i}^2} \cdot \sqrt{\sum_{i=1}^n f_{2i}^2}}$

### 音节结构 / Syllable Structure

**音节模板**: $S = (O)(M)V(C)$

**音节权重**: $W = \sum_{i=1}^n w_i \cdot f_i$

**音节复杂度**: $C = \frac{\text{音素数}}{\text{音节数}}$

### 音变规则 / Sound Change Rules

**音变概率**: $P(change) = \frac{\text{变化次数}}{\text{总出现次数}}$

**音变规则**: $A \rightarrow B / X \_ Y$

**音变强度**: $I = \frac{\text{变化频率}}{\text{环境频率}}$

---

## 6.5.2 形态学模型 / Morphological Models

### 词素分析 / Morpheme Analysis

**词素分解**: $W = m_1 + m_2 + \ldots + m_n$

**词素频率**: $f(m) = \frac{\text{词素出现次数}}{\text{总词数}}$

**词素生产力**: $P(m) = \frac{\text{新词数}}{\text{时间}}$

### 构词规则 / Word Formation Rules

**复合词**: $C = R + M$ where $R$ is root, $M$ is modifier

**派生词**: $D = R + A$ where $A$ is affix

**构词模式**: $Pattern = \frac{\text{符合模式词数}}{\text{总词数}}$

### 屈折变化 / Inflectional Patterns

**屈折范式**: $P = \{f_1, f_2, \ldots, f_n\}$

**屈折规则**: $R: \text{base} \rightarrow \text{inflected}$

**屈折一致性**: $C = \frac{\text{一致形式}}{\text{总形式}}$

---

## 6.5.3 句法学模型 / Syntactic Models

### 短语结构语法 / Phrase Structure Grammar

**重写规则**: $A \rightarrow \alpha$

**句法树**: $T = (N, E)$ where $N$ is nodes, $E$ is edges

**句法复杂度**: $C = \frac{\text{节点数}}{\text{句子长度}}$

### 依存语法 / Dependency Grammar

**依存关系**: $D = \{(w_i, w_j, r) | w_i \text{ depends on } w_j\}$

**依存距离**: $d(w_i, w_j) = |i - j|$

**依存密度**: $\rho = \frac{\text{依存关系数}}{\text{词数}}$

### 转换语法 / Transformational Grammar

**深层结构**: $DS \xrightarrow{T} SS$

**转换规则**: $T: \alpha \rightarrow \beta$

**转换复杂度**: $C = \sum_{i=1}^n c_i \cdot w_i$

---

## 6.5.4 语义学模型 / Semantic Models

### 词义模型 / Lexical Semantics

**词义向量**: $\mathbf{v}_w = \frac{\sum_{c \in C(w)} \mathbf{v}_c}{|C(w)|}$

**词义相似度**: $sim(w_1, w_2) = \frac{\mathbf{v}_{w_1} \cdot \mathbf{v}_{w_2}}{|\mathbf{v}_{w_1}| \cdot |\mathbf{v}_{w_2}|}$

**词义歧义**: $A(w) = \frac{\text{歧义数}}{\text{总义项数}}$

### 组合语义 / Compositional Semantics

**函数应用**: $[[A B]] = [[A]]([[B]])$

**λ抽象**: $[[\lambda x. \phi]] = \lambda d. [[\phi]]_{[x \mapsto d]}$

**语义组合**: $[[\alpha \beta]] = [[\alpha]] \circ [[\beta]]$

### 语用学 / Pragmatics

**会话含义**: $I = \{i | P(i|u) > P(i|\neg u)\}$

**预设**: $Presupposition = \{p | p \text{ is background}\}$

**语用推理**: $P(meaning|context) = \frac{P(context|meaning) \cdot P(meaning)}{P(context)}$

---

## 6.5.5 语料库语言学模型 / Corpus Linguistics Models

### 频率分析 / Frequency Analysis

**词频**: $f(w) = \frac{\text{词出现次数}}{\text{总词数}}$

**相对频率**: $rf(w) = \frac{f(w)}{f_{max}}$

**Zipf定律**: $f(r) = \frac{C}{r^\alpha}$

### 共现分析 / Co-occurrence Analysis

**共现矩阵**: $M_{ij} = \text{co-occurrence}(w_i, w_j)$

**互信息**: $MI(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$

**共现强度**: $S = \frac{\text{共现次数}}{\text{窗口大小}}$

### 语料库统计 / Corpus Statistics

**词汇丰富度**: $R = \frac{\text{不同词数}}{\text{总词数}}$

**平均句长**: $L = \frac{\text{总词数}}{\text{句子数}}$

**词汇密度**: $D = \frac{\text{实词数}}{\text{总词数}}$

---

## 6.5.6 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PhonologicalModel {
    pub phoneme_features: HashMap<String, Vec<i32>>,
    pub syllable_templates: Vec<String>,
}

impl PhonologicalModel {
    pub fn new() -> Self {
        let mut features = HashMap::new();
        features.insert("p".to_string(), vec![1, -1, -1, -1, -1]); // 清辅音
        features.insert("b".to_string(), vec![1, -1, -1, -1, 1]);  // 浊辅音
        features.insert("t".to_string(), vec![1, -1, -1, -1, -1]);
        features.insert("d".to_string(), vec![1, -1, -1, -1, 1]);
        features.insert("a".to_string(), vec![-1, 1, 1, -1, 0]);   // 元音
        features.insert("i".to_string(), vec![-1, 1, 1, 1, 0]);
        features.insert("u".to_string(), vec![-1, 1, 1, -1, 0]);
        
        Self {
            phoneme_features: features,
            syllable_templates: vec!["CV".to_string(), "CVC".to_string(), "V".to_string()],
        }
    }
    
    pub fn phoneme_distance(&self, p1: &str, p2: &str) -> i32 {
        if let (Some(f1), Some(f2)) = (self.phoneme_features.get(p1), self.phoneme_features.get(p2)) {
            f1.iter().zip(f2.iter()).map(|(a, b)| (a - b).abs()).sum()
        } else {
            -1
        }
    }
    
    pub fn phoneme_similarity(&self, p1: &str, p2: &str) -> f64 {
        if let (Some(f1), Some(f2)) = (self.phoneme_features.get(p1), self.phoneme_features.get(p2)) {
            let dot_product: i32 = f1.iter().zip(f2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f64 = (f1.iter().map(|x| x * x).sum::<i32>() as f64).sqrt();
            let norm2: f64 = (f2.iter().map(|x| x * x).sum::<i32>() as f64).sqrt();
            
            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product as f64 / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    pub fn analyze_syllable(&self, syllable: &str) -> String {
        let mut structure = String::new();
        for c in syllable.chars() {
            if self.phoneme_features.contains_key(&c.to_string()) {
                let features = &self.phoneme_features[&c.to_string()];
                if features[0] == 1 {
                    structure.push('C'); // 辅音
                } else {
                    structure.push('V'); // 元音
                }
            }
        }
        structure
    }
}

#[derive(Debug, Clone)]
pub struct MorphologicalModel {
    pub morphemes: HashMap<String, String>,
    pub affixes: HashMap<String, String>,
    pub word_formation_rules: Vec<(String, String)>,
}

impl MorphologicalModel {
    pub fn new() -> Self {
        let mut morphemes = HashMap::new();
        morphemes.insert("run".to_string(), "VERB".to_string());
        morphemes.insert("walk".to_string(), "VERB".to_string());
        morphemes.insert("fast".to_string(), "ADJ".to_string());
        
        let mut affixes = HashMap::new();
        affixes.insert("-ing".to_string(), "PRESENT_PARTICIPLE".to_string());
        affixes.insert("-ed".to_string(), "PAST_TENSE".to_string());
        affixes.insert("-er".to_string(), "COMPARATIVE".to_string());
        
        let rules = vec![
            ("VERB + -ing".to_string(), "PRESENT_PARTICIPLE".to_string()),
            ("VERB + -ed".to_string(), "PAST_TENSE".to_string()),
            ("ADJ + -er".to_string(), "COMPARATIVE".to_string()),
        ];
        
        Self {
            morphemes,
            affixes,
            word_formation_rules: rules,
        }
    }
    
    pub fn analyze_morphemes(&self, word: &str) -> Vec<String> {
        let mut morphemes = Vec::new();
        let mut current = word;
        
        // 简化的词素分析
        for (affix, _) in &self.affixes {
            if current.ends_with(affix) {
                let stem = &current[..current.len() - affix.len()];
                if self.morphemes.contains_key(stem) {
                    morphemes.push(stem.to_string());
                    morphemes.push(affix.clone());
                    break;
                }
            }
        }
        
        if morphemes.is_empty() && self.morphemes.contains_key(current) {
            morphemes.push(current.to_string());
        }
        
        morphemes
    }
    
    pub fn apply_word_formation(&self, base: &str, affix: &str) -> Option<String> {
        if let Some(base_type) = self.morphemes.get(base) {
            for (rule_pattern, result_type) in &self.word_formation_rules {
                if rule_pattern.contains(base_type) && rule_pattern.contains(affix) {
                    return Some(format!("{}{}", base, affix));
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct SyntacticModel {
    pub phrase_structure_rules: Vec<(String, Vec<String>)>,
    pub dependency_patterns: Vec<(String, String, String)>,
}

impl SyntacticModel {
    pub fn new() -> Self {
        let phrase_rules = vec![
            ("S".to_string(), vec!["NP".to_string(), "VP".to_string()]),
            ("NP".to_string(), vec!["Det".to_string(), "N".to_string()]),
            ("VP".to_string(), vec!["V".to_string(), "NP".to_string()]),
        ];
        
        let dependency_patterns = vec![
            ("nsubj".to_string(), "VERB".to_string(), "NOUN".to_string()),
            ("dobj".to_string(), "VERB".to_string(), "NOUN".to_string()),
            ("amod".to_string(), "NOUN".to_string(), "ADJ".to_string()),
        ];
        
        Self {
            phrase_structure_rules: phrase_rules,
            dependency_patterns,
        }
    }
    
    pub fn parse_phrase_structure(&self, sentence: &[String]) -> f64 {
        // 简化的短语结构分析评分
        let mut score = 1.0;
        for i in 0..sentence.len() - 1 {
            let current_pos = self.get_pos_tag(&sentence[i]);
            let next_pos = self.get_pos_tag(&sentence[i + 1]);
            
            // 检查是否符合短语结构规则
            for (rule_head, rule_body) in &self.phrase_structure_rules {
                if rule_body.contains(&current_pos) && rule_body.contains(&next_pos) {
                    score *= 1.1;
                }
            }
        }
        score
    }
    
    pub fn analyze_dependencies(&self, sentence: &[String]) -> Vec<(String, String, String)> {
        let mut dependencies = Vec::new();
        
        for i in 0..sentence.len() {
            for j in 0..sentence.len() {
                if i != j {
                    let pos1 = self.get_pos_tag(&sentence[i]);
                    let pos2 = self.get_pos_tag(&sentence[j]);
                    
                    for (dep_type, head_pos, dep_pos) in &self.dependency_patterns {
                        if &pos1 == head_pos && &pos2 == dep_pos {
                            dependencies.push((dep_type.clone(), sentence[i].clone(), sentence[j].clone()));
                        }
                    }
                }
            }
        }
        
        dependencies
    }
    
    fn get_pos_tag(&self, word: &str) -> String {
        // 简化的词性标注
        if word.ends_with("ing") {
            "VERB".to_string()
        } else if word.ends_with("ed") {
            "VERB".to_string()
        } else if word.ends_with("ly") {
            "ADV".to_string()
        } else if word.ends_with("er") || word.ends_with("est") {
            "ADJ".to_string()
        } else {
            "NOUN".to_string()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticModel {
    pub word_vectors: HashMap<String, Vec<f64>>,
    pub semantic_relations: HashMap<String, Vec<String>>,
}

impl SemanticModel {
    pub fn new() -> Self {
        let mut vectors = HashMap::new();
        vectors.insert("cat".to_string(), vec![0.1, 0.8, 0.2, 0.9]);
        vectors.insert("dog".to_string(), vec![0.2, 0.7, 0.3, 0.8]);
        vectors.insert("run".to_string(), vec![0.8, 0.1, 0.9, 0.3]);
        vectors.insert("walk".to_string(), vec![0.7, 0.2, 0.8, 0.4]);
        
        let mut relations = HashMap::new();
        relations.insert("cat".to_string(), vec!["animal".to_string(), "pet".to_string()]);
        relations.insert("dog".to_string(), vec!["animal".to_string(), "pet".to_string()]);
        relations.insert("run".to_string(), vec!["move".to_string(), "fast".to_string()]);
        
        Self {
            word_vectors: vectors,
            semantic_relations: relations,
        }
    }
    
    pub fn semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        if let (Some(vec1), Some(vec2)) = (self.word_vectors.get(word1), self.word_vectors.get(word2)) {
            let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    pub fn find_semantic_relations(&self, word: &str) -> Vec<String> {
        self.semantic_relations.get(word).cloned().unwrap_or_default()
    }
    
    pub fn calculate_semantic_density(&self, text: &[String]) -> f64 {
        let mut total_similarity = 0.0;
        let mut pair_count = 0;
        
        for i in 0..text.len() {
            for j in i + 1..text.len() {
                let similarity = self.semantic_similarity(&text[i], &text[j]);
                total_similarity += similarity;
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorpusModel {
    pub word_frequencies: HashMap<String, u32>,
    pub co_occurrence_matrix: HashMap<(String, String), u32>,
    pub total_words: u32,
}

impl CorpusModel {
    pub fn new() -> Self {
        Self {
            word_frequencies: HashMap::new(),
            co_occurrence_matrix: HashMap::new(),
            total_words: 0,
        }
    }
    
    pub fn add_text(&mut self, text: &[String]) {
        for word in text {
            *self.word_frequencies.entry(word.clone()).or_insert(0) += 1;
            self.total_words += 1;
        }
        
        // 计算共现矩阵（简化版本）
        for i in 0..text.len() {
            for j in i + 1..text.len() {
                let key = (text[i].clone(), text[j].clone());
                *self.co_occurrence_matrix.entry(key).or_insert(0) += 1;
            }
        }
    }
    
    pub fn word_frequency(&self, word: &str) -> f64 {
        if self.total_words > 0 {
            *self.word_frequencies.get(word).unwrap_or(&0) as f64 / self.total_words as f64
        } else {
            0.0
        }
    }
    
    pub fn co_occurrence_frequency(&self, word1: &str, word2: &str) -> f64 {
        let key1 = (word1.to_string(), word2.to_string());
        let key2 = (word2.to_string(), word1.to_string());
        
        let count = self.co_occurrence_matrix.get(&key1).unwrap_or(&0) + 
                   self.co_occurrence_matrix.get(&key2).unwrap_or(&0);
        
        if self.total_words > 0 {
            count as f64 / self.total_words as f64
        } else {
            0.0
        }
    }
    
    pub fn vocabulary_richness(&self) -> f64 {
        if self.total_words > 0 {
            self.word_frequencies.len() as f64 / self.total_words as f64
        } else {
            0.0
        }
    }
    
    pub fn mutual_information(&self, word1: &str, word2: &str) -> f64 {
        let p_w1 = self.word_frequency(word1);
        let p_w2 = self.word_frequency(word2);
        let p_w1w2 = self.co_occurrence_frequency(word1, word2);
        
        if p_w1 > 0.0 && p_w2 > 0.0 && p_w1w2 > 0.0 {
            (p_w1w2 / (p_w1 * p_w2)).ln()
        } else {
            0.0
        }
    }
}

// 使用示例
fn main() {
    // 语音学模型示例
    let phonological = PhonologicalModel::new();
    let distance = phonological.phoneme_distance("p", "b");
    let similarity = phonological.phoneme_similarity("p", "b");
    let syllable_structure = phonological.analyze_syllable("pat");
    
    println!("语音学模型示例:");
    println!("音位距离: {}", distance);
    println!("音位相似度: {:.3}", similarity);
    println!("音节结构: {}", syllable_structure);
    
    // 形态学模型示例
    let morphological = MorphologicalModel::new();
    let morphemes = morphological.analyze_morphemes("running");
    let new_word = morphological.apply_word_formation("run", "ing");
    
    println!("\n形态学模型示例:");
    println!("词素分析: {:?}", morphemes);
    println!("构词结果: {:?}", new_word);
    
    // 句法学模型示例
    let syntactic = SyntacticModel::new();
    let sentence = vec!["the".to_string(), "cat".to_string(), "runs".to_string()];
    let phrase_score = syntactic.parse_phrase_structure(&sentence);
    let dependencies = syntactic.analyze_dependencies(&sentence);
    
    println!("\n句法学模型示例:");
    println!("短语结构得分: {:.3}", phrase_score);
    println!("依存关系: {:?}", dependencies);
    
    // 语义学模型示例
    let semantic = SemanticModel::new();
    let similarity = semantic.semantic_similarity("cat", "dog");
    let relations = semantic.find_semantic_relations("cat");
    let density = semantic.calculate_semantic_density(&sentence);
    
    println!("\n语义学模型示例:");
    println!("语义相似度: {:.3}", similarity);
    println!("语义关系: {:?}", relations);
    println!("语义密度: {:.3}", density);
    
    // 语料库模型示例
    let mut corpus = CorpusModel::new();
    let text = vec!["the".to_string(), "cat".to_string(), "runs".to_string(), 
                    "the".to_string(), "dog".to_string(), "walks".to_string()];
    corpus.add_text(&text);
    
    let frequency = corpus.word_frequency("the");
    let co_occurrence = corpus.co_occurrence_frequency("cat", "runs");
    let richness = corpus.vocabulary_richness();
    let mi = corpus.mutual_information("cat", "runs");
    
    println!("\n语料库模型示例:");
    println!("词频: {:.3}", frequency);
    println!("共现频率: {:.3}", co_occurrence);
    println!("词汇丰富度: {:.3}", richness);
    println!("互信息: {:.3}", mi);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module LinguisticModels where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sum, length, filter, maximumBy)
import Data.Ord (comparing)

-- 语音学模型
data PhonologicalModel = PhonologicalModel {
    phonemeFeatures :: Map String [Int],
    syllableTemplates :: [String]
} deriving Show

newPhonologicalModel :: PhonologicalModel
newPhonologicalModel = 
    let features = Map.fromList [
            ("p", [1, -1, -1, -1, -1]),
            ("b", [1, -1, -1, -1, 1]),
            ("t", [1, -1, -1, -1, -1]),
            ("d", [1, -1, -1, -1, 1]),
            ("a", [-1, 1, 1, -1, 0]),
            ("i", [-1, 1, 1, 1, 0]),
            ("u", [-1, 1, 1, -1, 0])
        ]
    in PhonologicalModel {
        phonemeFeatures = features,
        syllableTemplates = ["CV", "CVC", "V"]
    }

phonemeDistance :: PhonologicalModel -> String -> String -> Int
phonemeDistance model p1 p2 = 
    case (Map.lookup p1 (phonemeFeatures model), Map.lookup p2 (phonemeFeatures model)) of
        (Just f1, Just f2) -> sum (zipWith (\a b -> abs (a - b)) f1 f2)
        _ -> -1

phonemeSimilarity :: PhonologicalModel -> String -> String -> Double
phonemeSimilarity model p1 p2 = 
    case (Map.lookup p1 (phonemeFeatures model), Map.lookup p2 (phonemeFeatures model)) of
        (Just f1, Just f2) -> 
            let dotProduct = sum (zipWith (*) f1 f2)
                norm1 = sqrt (fromIntegral (sum (map (^2) f1)))
                norm2 = sqrt (fromIntegral (sum (map (^2) f2)))
            in if norm1 > 0.0 && norm2 > 0.0 
               then fromIntegral dotProduct / (norm1 * norm2)
               else 0.0
        _ -> 0.0

analyzeSyllable :: PhonologicalModel -> String -> String
analyzeSyllable model syllable = 
    let structure = map (\c -> 
        case Map.lookup [c] (phonemeFeatures model) of
            Just features -> if head features == 1 then 'C' else 'V'
            Nothing -> '?') syllable
    in structure

-- 形态学模型
data MorphologicalModel = MorphologicalModel {
    morphemes :: Map String String,
    affixes :: Map String String,
    wordFormationRules :: [(String, String)]
} deriving Show

newMorphologicalModel :: MorphologicalModel
newMorphologicalModel = 
    let morphemes = Map.fromList [
            ("run", "VERB"),
            ("walk", "VERB"),
            ("fast", "ADJ")
        ]
        affixes = Map.fromList [
            ("-ing", "PRESENT_PARTICIPLE"),
            ("-ed", "PAST_TENSE"),
            ("-er", "COMPARATIVE")
        ]
        rules = [
            ("VERB + -ing", "PRESENT_PARTICIPLE"),
            ("VERB + -ed", "PAST_TENSE"),
            ("ADJ + -er", "COMPARATIVE")
        ]
    in MorphologicalModel {
        morphemes = morphemes,
        affixes = affixes,
        wordFormationRules = rules
    }

analyzeMorphemes :: MorphologicalModel -> String -> [String]
analyzeMorphemes model word = 
    let findAffix w = 
            foldr (\affix acc -> 
                if w `endsWith` affix 
                then let stem = take (length w - length affix) w
                     in if Map.member stem (morphemes model)
                        then [stem, affix]
                        else acc
                else acc) [] (Map.keys (affixes model))
        result = findAffix word
    in if null result && Map.member word (morphemes model)
       then [word]
       else result

applyWordFormation :: MorphologicalModel -> String -> String -> Maybe String
applyWordFormation model base affix = 
    case Map.lookup base (morphemes model) of
        Just baseType -> 
            let rule = baseType ++ " + " ++ affix
            in if any (\(pattern, _) -> pattern == rule) (wordFormationRules model)
               then Just (base ++ affix)
               else Nothing
        Nothing -> Nothing

-- 句法学模型
data SyntacticModel = SyntacticModel {
    phraseStructureRules :: [(String, [String])],
    dependencyPatterns :: [(String, String, String)]
} deriving Show

newSyntacticModel :: SyntacticModel
newSyntacticModel = 
    let phraseRules = [
            ("S", ["NP", "VP"]),
            ("NP", ["Det", "N"]),
            ("VP", ["V", "NP"])
        ]
        depPatterns = [
            ("nsubj", "VERB", "NOUN"),
            ("dobj", "VERB", "NOUN"),
            ("amod", "NOUN", "ADJ")
        ]
    in SyntacticModel {
        phraseStructureRules = phraseRules,
        dependencyPatterns = depPatterns
    }

parsePhraseStructure :: SyntacticModel -> [String] -> Double
parsePhraseStructure model sentence = 
    let score = foldr (\i acc -> 
        let currentPos = getPosTag (sentence !! i)
            nextPos = if i + 1 < length sentence 
                     then getPosTag (sentence !! (i + 1))
                     else ""
        in acc * (if any (\(_, ruleBody) -> 
                           currentPos `elem` ruleBody && nextPos `elem` ruleBody)
                         (phraseStructureRules model)
                  then 1.1 else 1.0)) 1.0 [0..length sentence - 2]
    in score

analyzeDependencies :: SyntacticModel -> [String] -> [(String, String, String)]
analyzeDependencies model sentence = 
    let dependencies = [(i, j) | i <- [0..length sentence - 1], 
                                   j <- [0..length sentence - 1], i /= j]
        posPairs = map (\(i, j) -> 
            (getPosTag (sentence !! i), getPosTag (sentence !! j))) dependencies
    in [(depType, sentence !! i, sentence !! j) | 
        (i, j) <- dependencies,
        (depType, headPos, depPos) <- dependencyPatterns model,
        getPosTag (sentence !! i) == headPos,
        getPosTag (sentence !! j) == depPos]

-- 语义学模型
data SemanticModel = SemanticModel {
    wordVectors :: Map String [Double],
    semanticRelations :: Map String [String]
} deriving Show

newSemanticModel :: SemanticModel
newSemanticModel = 
    let vectors = Map.fromList [
            ("cat", [0.1, 0.8, 0.2, 0.9]),
            ("dog", [0.2, 0.7, 0.3, 0.8]),
            ("run", [0.8, 0.1, 0.9, 0.3]),
            ("walk", [0.7, 0.2, 0.8, 0.4])
        ]
        relations = Map.fromList [
            ("cat", ["animal", "pet"]),
            ("dog", ["animal", "pet"]),
            ("run", ["move", "fast"])
        ]
    in SemanticModel {
        wordVectors = vectors,
        semanticRelations = relations
    }

semanticSimilarity :: SemanticModel -> String -> String -> Double
semanticSimilarity model word1 word2 = 
    case (Map.lookup word1 (wordVectors model), Map.lookup word2 (wordVectors model)) of
        (Just vec1, Just vec2) -> 
            let dotProduct = sum (zipWith (*) vec1 vec2)
                norm1 = sqrt (sum (map (^2) vec1))
                norm2 = sqrt (sum (map (^2) vec2))
            in if norm1 > 0.0 && norm2 > 0.0 
               then dotProduct / (norm1 * norm2)
               else 0.0
        _ -> 0.0

findSemanticRelations :: SemanticModel -> String -> [String]
findSemanticRelations model word = 
    Map.findWithDefault [] word (semanticRelations model)

calculateSemanticDensity :: SemanticModel -> [String] -> Double
calculateSemanticDensity model text = 
    let pairs = [(i, j) | i <- [0..length text - 1], 
                         j <- [i + 1..length text - 1]]
        similarities = map (\(i, j) -> 
            semanticSimilarity model (text !! i) (text !! j)) pairs
    in if null similarities then 0.0 else sum similarities / fromIntegral (length similarities)

-- 语料库模型
data CorpusModel = CorpusModel {
    wordFrequencies :: Map String Int,
    coOccurrenceMatrix :: Map (String, String) Int,
    totalWords :: Int
} deriving Show

newCorpusModel :: CorpusModel
newCorpusModel = CorpusModel {
    wordFrequencies = Map.empty,
    coOccurrenceMatrix = Map.empty,
    totalWords = 0
}

addText :: CorpusModel -> [String] -> CorpusModel
addText model text = 
    let newFrequencies = foldr (\word acc -> 
            Map.insertWith (+) word 1 acc) (wordFrequencies model) text
        newTotalWords = totalWords model + length text
        newCoOccurrence = foldr (\(i, j) acc -> 
            let key = (text !! i, text !! j)
            in Map.insertWith (+) key 1 acc) (coOccurrenceMatrix model)
            [(i, j) | i <- [0..length text - 1], 
                      j <- [i + 1..length text - 1]]
    in model {
        wordFrequencies = newFrequencies,
        coOccurrenceMatrix = newCoOccurrence,
        totalWords = newTotalWords
    }

wordFrequency :: CorpusModel -> String -> Double
wordFrequency model word = 
    if totalWords model > 0 
    then fromIntegral (Map.findWithDefault 0 word (wordFrequencies model)) / 
         fromIntegral (totalWords model)
    else 0.0

coOccurrenceFrequency :: CorpusModel -> String -> String -> Double
coOccurrenceFrequency model word1 word2 = 
    let key1 = (word1, word2)
        key2 = (word2, word1)
        count = Map.findWithDefault 0 key1 (coOccurrenceMatrix model) + 
                Map.findWithDefault 0 key2 (coOccurrenceMatrix model)
    in if totalWords model > 0 
       then fromIntegral count / fromIntegral (totalWords model)
       else 0.0

vocabularyRichness :: CorpusModel -> Double
vocabularyRichness model = 
    if totalWords model > 0 
    then fromIntegral (Map.size (wordFrequencies model)) / 
         fromIntegral (totalWords model)
    else 0.0

mutualInformation :: CorpusModel -> String -> String -> Double
mutualInformation model word1 word2 = 
    let p_w1 = wordFrequency model word1
        p_w2 = wordFrequency model word2
        p_w1w2 = coOccurrenceFrequency model word1 word2
    in if p_w1 > 0.0 && p_w2 > 0.0 && p_w1w2 > 0.0
       then logBase 2 (p_w1w2 / (p_w1 * p_w2))
       else 0.0

-- 辅助函数
endsWith :: String -> String -> Bool
endsWith str suffix = 
    length str >= length suffix && 
    drop (length str - length suffix) str == suffix

getPosTag :: String -> String
getPosTag word
    | word `endsWith` "ing" = "VERB"
    | word `endsWith` "ed" = "VERB"
    | word `endsWith` "ly" = "ADV"
    | word `endsWith` "er" || word `endsWith` "est" = "ADJ"
    | otherwise = "NOUN"

-- 示例使用
example :: IO ()
example = do
    -- 语音学模型示例
    let phonological = newPhonologicalModel
        distance = phonemeDistance phonological "p" "b"
        similarity = phonemeSimilarity phonological "p" "b"
        syllableStructure = analyzeSyllable phonological "pat"
    
    putStrLn "语音学模型示例:"
    putStrLn $ "音位距离: " ++ show distance
    putStrLn $ "音位相似度: " ++ show similarity
    putStrLn $ "音节结构: " ++ syllableStructure
    
    -- 形态学模型示例
    let morphological = newMorphologicalModel
        morphemes = analyzeMorphemes morphological "running"
        newWord = applyWordFormation morphological "run" "ing"
    
    putStrLn "\n形态学模型示例:"
    putStrLn $ "词素分析: " ++ show morphemes
    putStrLn $ "构词结果: " ++ show newWord
    
    -- 句法学模型示例
    let syntactic = newSyntacticModel
        sentence = ["the", "cat", "runs"]
        phraseScore = parsePhraseStructure syntactic sentence
        dependencies = analyzeDependencies syntactic sentence
    
    putStrLn "\n句法学模型示例:"
    putStrLn $ "短语结构得分: " ++ show phraseScore
    putStrLn $ "依存关系: " ++ show dependencies
    
    -- 语义学模型示例
    let semantic = newSemanticModel
        similarity = semanticSimilarity semantic "cat" "dog"
        relations = findSemanticRelations semantic "cat"
        density = calculateSemanticDensity semantic sentence
    
    putStrLn "\n语义学模型示例:"
    putStrLn $ "语义相似度: " ++ show similarity
    putStrLn $ "语义关系: " ++ show relations
    putStrLn $ "语义密度: " ++ show density
    
    -- 语料库模型示例
    let corpus = addText newCorpusModel ["the", "cat", "runs", "the", "dog", "walks"]
        frequency = wordFrequency corpus "the"
        coOccurrence = coOccurrenceFrequency corpus "cat" "runs"
        richness = vocabularyRichness corpus
        mi = mutualInformation corpus "cat" "runs"
    
    putStrLn "\n语料库模型示例:"
    putStrLn $ "词频: " ++ show frequency
    putStrLn $ "共现频率: " ++ show coOccurrence
    putStrLn $ "词汇丰富度: " ++ show richness
    putStrLn $ "互信息: " ++ show mi
```

### 应用领域 / Application Domains

#### 自然语言处理 / Natural Language Processing

- **文本分析**: 词性标注、句法分析、语义分析
- **信息抽取**: 命名实体识别、关系抽取、事件抽取
- **文本生成**: 机器翻译、摘要生成、对话系统

#### 机器翻译 / Machine Translation

- **统计机器翻译**: 基于短语的翻译、基于句法的翻译
- **神经机器翻译**: 序列到序列模型、注意力机制
- **多语言处理**: 跨语言信息检索、多语言文本分析

#### 语音识别 / Speech Recognition

- **声学模型**: 隐马尔可夫模型、深度神经网络
- **语言模型**: n-gram模型、循环神经网络
- **语音合成**: 文本到语音转换、语音克隆

---

## 参考文献 / References

1. Chomsky, N. (1957). Syntactic structures. Mouton.
2. Jackendoff, R. (2002). Foundations of language: Brain, meaning, grammar, evolution. Oxford University Press.
3. Jurafsky, D., & Martin, J. H. (2009). Speech and language processing. Pearson.
4. Manning, C. D., & Schütze, H. (1999). Foundations of statistical natural language processing. MIT Press.
5. Saussure, F. de. (1916). Course in general linguistics. McGraw-Hill.

---

*最后更新: 2025-08-01*
*版本: 1.0.0* 