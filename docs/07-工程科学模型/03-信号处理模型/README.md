# 7.3 信号处理模型 / Signal Processing Models

## 目录 / Table of Contents

- [7.3 信号处理模型 / Signal Processing Models](#73-信号处理模型--signal-processing-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.3.1 傅里叶变换模型 / Fourier Transform Models](#731-傅里叶变换模型--fourier-transform-models)
    - [连续傅里叶变换 / Continuous Fourier Transform](#连续傅里叶变换--continuous-fourier-transform)
    - [离散傅里叶变换 / Discrete Fourier Transform](#离散傅里叶变换--discrete-fourier-transform)
    - [快速傅里叶变换 / Fast Fourier Transform](#快速傅里叶变换--fast-fourier-transform)
  - [7.3.2 滤波器模型 / Filter Models](#732-滤波器模型--filter-models)
    - [无限冲激响应滤波器 / Infinite Impulse Response Filter](#无限冲激响应滤波器--infinite-impulse-response-filter)
    - [有限冲激响应滤波器 / Finite Impulse Response Filter](#有限冲激响应滤波器--finite-impulse-response-filter)
    - [滤波器设计方法 / Filter Design Methods](#滤波器设计方法--filter-design-methods)
  - [7.3.3 小波变换模型 / Wavelet Transform Models](#733-小波变换模型--wavelet-transform-models)
    - [连续小波变换 / Continuous Wavelet Transform](#连续小波变换--continuous-wavelet-transform)
    - [离散小波变换 / Discrete Wavelet Transform](#离散小波变换--discrete-wavelet-transform)
    - [多分辨率分析 / Multiresolution Analysis](#多分辨率分析--multiresolution-analysis)
  - [7.3.4 数字信号处理模型 / Digital Signal Processing Models](#734-数字信号处理模型--digital-signal-processing-models)
    - [采样定理 / Sampling Theorem](#采样定理--sampling-theorem)
    - [量化模型 / Quantization Model](#量化模型--quantization-model)
    - [数字滤波器设计 / Digital Filter Design](#数字滤波器设计--digital-filter-design)
  - [7.3.5 自适应信号处理模型 / Adaptive Signal Processing Models](#735-自适应信号处理模型--adaptive-signal-processing-models)
    - [最小均方算法 / Least Mean Square Algorithm](#最小均方算法--least-mean-square-algorithm)
    - [递归最小二乘算法 / Recursive Least Squares Algorithm](#递归最小二乘算法--recursive-least-squares-algorithm)
    - [自适应滤波器 / Adaptive Filter](#自适应滤波器--adaptive-filter)
  - [7.3.6 统计信号处理模型 / Statistical Signal Processing Models](#736-统计信号处理模型--statistical-signal-processing-models)
    - [功率谱估计 / Power Spectral Density Estimation](#功率谱估计--power-spectral-density-estimation)
    - [维纳滤波 / Wiener Filter](#维纳滤波--wiener-filter)
    - [卡尔曼滤波 / Kalman Filter](#卡尔曼滤波--kalman-filter)
  - [7.3.7 实现与应用 / Implementation and Applications](#737-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [音频处理 / Audio Processing](#音频处理--audio-processing)
      - [图像处理 / Image Processing](#图像处理--image-processing)
      - [通信系统 / Communication Systems](#通信系统--communication-systems)
  - [参考文献 / References](#参考文献--references)

---

## 7.3.1 傅里叶变换模型 / Fourier Transform Models

### 连续傅里叶变换 / Continuous Fourier Transform

**正变换**: $X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt$

**逆变换**: $x(t) = \int_{-\infty}^{\infty} X(f) e^{j2\pi ft} df$

**能量谱密度**: $S_{xx}(f) = |X(f)|^2$

### 离散傅里叶变换 / Discrete Fourier Transform

**正变换**: $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j\frac{2\pi kn}{N}}$

**逆变换**: $x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j\frac{2\pi kn}{N}}$

**功率谱密度**: $P_{xx}[k] = \frac{1}{N} |X[k]|^2$

### 快速傅里叶变换 / Fast Fourier Transform

**分治策略**: $X[k] = X_e[k] + W_N^k X_o[k]$

**蝶形运算**: $X[k] = X_e[k] + e^{-j\frac{2\pi k}{N}} X_o[k]$

**计算复杂度**: $O(N \log N)$

---

## 7.3.2 滤波器模型 / Filter Models

### 无限冲激响应滤波器 / Infinite Impulse Response Filter

**差分方程**: $y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$

**传递函数**: $H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$

**频率响应**: $H(e^{j\omega}) = \frac{\sum_{k=0}^{M} b_k e^{-j\omega k}}{1 + \sum_{k=1}^{N} a_k e^{-j\omega k}}$

### 有限冲激响应滤波器 / Finite Impulse Response Filter

**差分方程**: $y[n] = \sum_{k=0}^{M} h[k] x[n-k]$

**传递函数**: $H(z) = \sum_{k=0}^{M} h[k] z^{-k}$

**频率响应**: $H(e^{j\omega}) = \sum_{k=0}^{M} h[k] e^{-j\omega k}$

### 滤波器设计方法 / Filter Design Methods

**窗函数法**: $h[n] = h_d[n] w[n]$

**频率采样法**: $H[k] = H_d(e^{j\frac{2\pi k}{N}})$

**最小二乘法**: $\min \sum_{k=0}^{N-1} |H(e^{j\omega_k}) - H_d(e^{j\omega_k})|^2$

---

## 7.3.3 小波变换模型 / Wavelet Transform Models

### 连续小波变换 / Continuous Wavelet Transform

**小波变换**: $W_x(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt$

**逆变换**: $x(t) = \frac{1}{C_\psi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} W_x(a,b) \psi_{a,b}(t) \frac{da db}{a^2}$

**小波函数**: $\psi_{a,b}(t) = \frac{1}{\sqrt{|a|}} \psi\left(\frac{t-b}{a}\right)$

### 离散小波变换 / Discrete Wavelet Transform

**分解**: $c_{j+1}[k] = \sum_n h[n-2k] c_j[n]$

**重构**: $c_j[n] = \sum_k g[n-2k] c_{j+1}[k] + \sum_k h[n-2k] d_{j+1}[k]$

**滤波器组**: $H(z) = \sum_n h[n] z^{-n}$, $G(z) = \sum_n g[n] z^{-n}$

### 多分辨率分析 / Multiresolution Analysis

**尺度函数**: $\phi_{j,k}(t) = 2^{j/2} \phi(2^j t - k)$

**小波函数**: $\psi_{j,k}(t) = 2^{j/2} \psi(2^j t - k)$

**分解关系**: $V_j = V_{j+1} \oplus W_{j+1}$

---

## 7.3.4 数字信号处理模型 / Digital Signal Processing Models

### 采样定理 / Sampling Theorem

**奈奎斯特频率**: $f_s \geq 2f_{max}$

**采样频率**: $T_s = \frac{1}{f_s}$

**重构公式**: $x(t) = \sum_{n=-\infty}^{\infty} x[n] \text{sinc}\left(\frac{t-nT_s}{T_s}\right)$

### 量化模型 / Quantization Model

**量化误差**: $e[n] = x[n] - \hat{x}[n]$

**量化噪声**: $\sigma_e^2 = \frac{\Delta^2}{12}$

**信噪比**: $\text{SNR} = 6.02B + 1.76$ dB

### 数字滤波器设计 / Digital Filter Design

**双线性变换**: $s = \frac{2}{T} \frac{z-1}{z+1}$

**预畸变**: $\omega_d = \frac{2}{T} \tan\left(\frac{\omega_a T}{2}\right)$

**频率响应**: $H(e^{j\omega}) = H_a\left(j\frac{2}{T} \tan\left(\frac{\omega T}{2}\right)\right)$

---

## 7.3.5 自适应信号处理模型 / Adaptive Signal Processing Models

### 最小均方算法 / Least Mean Square Algorithm

**权重更新**: $w[n+1] = w[n] + \mu e[n] x[n]$

**误差信号**: $e[n] = d[n] - y[n]$

**收敛条件**: $0 < \mu < \frac{2}{\lambda_{max}}$

### 递归最小二乘算法 / Recursive Least Squares Algorithm

**卡尔曼增益**: $K[n] = P[n-1]x[n](\lambda + x^T[n]P[n-1]x[n])^{-1}$

**权重更新**: $w[n] = w[n-1] + K[n]e[n]$

**协方差更新**: $P[n] = \frac{1}{\lambda}(I - K[n]x^T[n])P[n-1]$

### 自适应滤波器 / Adaptive Filter

**LMS滤波器**: $y[n] = w^T[n]x[n]$

**RLS滤波器**: $y[n] = w^T[n]x[n]$

**收敛速度**: RLS > LMS

---

## 7.3.6 统计信号处理模型 / Statistical Signal Processing Models

### 功率谱估计 / Power Spectral Density Estimation

**周期图**: $\hat{P}_{xx}(f) = \frac{1}{N} |X(f)|^2$

**Welch方法**: $\hat{P}_{xx}(f) = \frac{1}{K} \sum_{k=1}^{K} \hat{P}_{xx}^{(k)}(f)$

**AR模型**: $P_{xx}(f) = \frac{\sigma_w^2}{|1 + \sum_{k=1}^{p} a_k e^{-j2\pi fk}|^2}$

### 维纳滤波 / Wiener Filter

**最优滤波器**: $H_{opt}(f) = \frac{P_{xs}(f)}{P_{xx}(f)}$

**均方误差**: $\text{MSE} = \int_{-\infty}^{\infty} P_{ss}(f) - \frac{|P_{xs}(f)|^2}{P_{xx}(f)} df$

**信噪比**: $\text{SNR}_{out} = \int_{-\infty}^{\infty} \frac{|H(f)|^2 P_{ss}(f)}{P_{nn}(f)} df$

### 卡尔曼滤波 / Kalman Filter

**预测**: $\hat{x}_k^- = F_k \hat{x}_{k-1} + B_k u_k$

**更新**: $\hat{x}_k = \hat{x}_k^- + K_k(z_k - H_k \hat{x}_k^-)$

**卡尔曼增益**: $K_k = P_k^- H_k^T(H_k P_k^- H_k^T + R_k)^{-1}$

---

## 7.3.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Signal {
    pub data: Vec<f64>,
    pub sampling_rate: f64,
}

impl Signal {
    pub fn new(data: Vec<f64>, sampling_rate: f64) -> Self {
        Self { data, sampling_rate }
    }
    
    pub fn length(&self) -> usize {
        self.data.len()
    }
    
    pub fn duration(&self) -> f64 {
        self.length() as f64 / self.sampling_rate
    }
}

#[derive(Debug)]
pub struct FFT {
    pub n: usize,
}

impl FFT {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
    
    pub fn fft(&self, x: &[f64]) -> Vec<f64> {
        let mut x_complex: Vec<f64> = x.to_vec();
        let mut y_complex: Vec<f64> = vec![0.0; self.n];
        
        // 位反转
        for i in 0..self.n {
            let j = self.bit_reverse(i);
            if i < j {
                x_complex.swap(i, j);
            }
        }
        
        // FFT计算
        for stage in 1..=self.n.trailing_zeros() {
            let m = 1 << stage;
            let w_m = (-2.0 * PI / m as f64).cos();
            
            for k in (0..self.n).step_by(m) {
                let mut w = 1.0;
                for j in 0..m/2 {
                    let t = w * x_complex[k + j + m/2];
                    let u = x_complex[k + j];
                    x_complex[k + j] = u + t;
                    x_complex[k + j + m/2] = u - t;
                    w = w * w_m;
                }
            }
        }
        
        x_complex
    }
    
    fn bit_reverse(&self, mut x: usize) -> usize {
        let mut result = 0;
        for _ in 0..self.n.trailing_zeros() {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }
    
    pub fn power_spectrum(&self, x: &[f64]) -> Vec<f64> {
        let fft_result = self.fft(x);
        let mut power = vec![0.0; self.n / 2];
        
        for i in 0..self.n / 2 {
            let real = fft_result[2 * i];
            let imag = fft_result[2 * i + 1];
            power[i] = (real * real + imag * imag).sqrt();
        }
        
        power
    }
}

#[derive(Debug)]
pub struct FIRFilter {
    pub coefficients: Vec<f64>,
    pub buffer: Vec<f64>,
}

impl FIRFilter {
    pub fn new(coefficients: Vec<f64>) -> Self {
        let buffer_size = coefficients.len();
        Self {
            coefficients,
            buffer: vec![0.0; buffer_size],
        }
    }
    
    pub fn filter(&mut self, input: f64) -> f64 {
        // 更新缓冲区
        for i in (1..self.buffer.len()).rev() {
            self.buffer[i] = self.buffer[i - 1];
        }
        self.buffer[0] = input;
        
        // 卷积计算
        let mut output = 0.0;
        for i in 0..self.coefficients.len() {
            output += self.coefficients[i] * self.buffer[i];
        }
        
        output
    }
    
    pub fn filter_signal(&mut self, signal: &Signal) -> Signal {
        let mut filtered_data = Vec::new();
        
        for &sample in &signal.data {
            let filtered_sample = self.filter(sample);
            filtered_data.push(filtered_sample);
        }
        
        Signal::new(filtered_data, signal.sampling_rate)
    }
}

#[derive(Debug)]
pub struct IIRFilter {
    pub b_coeffs: Vec<f64>, // 分子系数
    pub a_coeffs: Vec<f64>, // 分母系数
    pub x_buffer: Vec<f64>, // 输入缓冲区
    pub y_buffer: Vec<f64>, // 输出缓冲区
}

impl IIRFilter {
    pub fn new(b_coeffs: Vec<f64>, a_coeffs: Vec<f64>) -> Self {
        let max_order = b_coeffs.len().max(a_coeffs.len());
        Self {
            b_coeffs,
            a_coeffs,
            x_buffer: vec![0.0; max_order],
            y_buffer: vec![0.0; max_order],
        }
    }
    
    pub fn filter(&mut self, input: f64) -> f64 {
        // 更新输入缓冲区
        for i in (1..self.x_buffer.len()).rev() {
            self.x_buffer[i] = self.x_buffer[i - 1];
        }
        self.x_buffer[0] = input;
        
        // 计算输出
        let mut output = 0.0;
        
        // 分子部分
        for i in 0..self.b_coeffs.len() {
            output += self.b_coeffs[i] * self.x_buffer[i];
        }
        
        // 分母部分
        for i in 1..self.a_coeffs.len() {
            output -= self.a_coeffs[i] * self.y_buffer[i - 1];
        }
        
        // 归一化
        if self.a_coeffs[0] != 0.0 {
            output /= self.a_coeffs[0];
        }
        
        // 更新输出缓冲区
        for i in (1..self.y_buffer.len()).rev() {
            self.y_buffer[i] = self.y_buffer[i - 1];
        }
        self.y_buffer[0] = output;
        
        output
    }
}

#[derive(Debug)]
pub struct WaveletTransform {
    pub wavelet_type: String,
    pub decomposition_levels: usize,
}

impl WaveletTransform {
    pub fn new(wavelet_type: String, decomposition_levels: usize) -> Self {
        Self {
            wavelet_type,
            decomposition_levels,
        }
    }
    
    pub fn dwt(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let mut approximation = signal.to_vec();
        let mut detail = vec![0.0; n / 2];
        
        // 简化的DWT实现（使用Haar小波）
        for i in 0..n / 2 {
            approximation[i] = (signal[2 * i] + signal[2 * i + 1]) / 2.0;
            detail[i] = (signal[2 * i] - signal[2 * i + 1]) / 2.0;
        }
        
        approximation.truncate(n / 2);
        (approximation, detail)
    }
    
    pub fn idwt(&self, approximation: &[f64], detail: &[f64]) -> Vec<f64> {
        let n = approximation.len() * 2;
        let mut reconstructed = vec![0.0; n];
        
        // 简化的IDWT实现
        for i in 0..approximation.len() {
            reconstructed[2 * i] = approximation[i] + detail[i];
            reconstructed[2 * i + 1] = approximation[i] - detail[i];
        }
        
        reconstructed
    }
    
    pub fn decompose(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let mut coefficients = Vec::new();
        let mut current_signal = signal.to_vec();
        
        for level in 0..self.decomposition_levels {
            let (approx, detail) = self.dwt(&current_signal);
            coefficients.push(detail);
            current_signal = approx;
        }
        
        coefficients.push(current_signal);
        coefficients
    }
}

#[derive(Debug)]
pub struct AdaptiveFilter {
    pub filter_length: usize,
    pub learning_rate: f64,
    pub weights: Vec<f64>,
    pub buffer: Vec<f64>,
}

impl AdaptiveFilter {
    pub fn new(filter_length: usize, learning_rate: f64) -> Self {
        Self {
            filter_length,
            learning_rate,
            weights: vec![0.0; filter_length],
            buffer: vec![0.0; filter_length],
        }
    }
    
    pub fn lms_update(&mut self, input: f64, desired: f64) -> f64 {
        // 更新缓冲区
        for i in (1..self.buffer.len()).rev() {
            self.buffer[i] = self.buffer[i - 1];
        }
        self.buffer[0] = input;
        
        // 计算输出
        let mut output = 0.0;
        for i in 0..self.filter_length {
            output += self.weights[i] * self.buffer[i];
        }
        
        // 计算误差
        let error = desired - output;
        
        // 更新权重
        for i in 0..self.filter_length {
            self.weights[i] += self.learning_rate * error * self.buffer[i];
        }
        
        output
    }
    
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
}

#[derive(Debug)]
pub struct KalmanFilter {
    pub state_dim: usize,
    pub measurement_dim: usize,
    pub f: Vec<Vec<f64>>, // 状态转移矩阵
    pub h: Vec<Vec<f64>>, // 观测矩阵
    pub q: Vec<Vec<f64>>, // 过程噪声协方差
    pub r: Vec<Vec<f64>>, // 测量噪声协方差
    pub p: Vec<Vec<f64>>, // 状态协方差
    pub x: Vec<f64>,      // 状态估计
}

impl KalmanFilter {
    pub fn new(state_dim: usize, measurement_dim: usize) -> Self {
        Self {
            state_dim,
            measurement_dim,
            f: vec![vec![0.0; state_dim]; state_dim],
            h: vec![vec![0.0; state_dim]; measurement_dim],
            q: vec![vec![0.0; state_dim]; state_dim],
            r: vec![vec![0.0; measurement_dim]; measurement_dim],
            p: vec![vec![0.0; state_dim]; state_dim],
            x: vec![0.0; state_dim],
        }
    }
    
    pub fn predict(&mut self) {
        // 状态预测
        let mut x_pred = vec![0.0; self.state_dim];
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                x_pred[i] += self.f[i][j] * self.x[j];
            }
        }
        self.x = x_pred;
        
        // 协方差预测
        let mut p_pred = vec![vec![0.0; self.state_dim]; self.state_dim];
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                for k in 0..self.state_dim {
                    p_pred[i][j] += self.f[i][k] * self.p[k][j];
                }
            }
        }
        
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                self.p[i][j] = 0.0;
                for k in 0..self.state_dim {
                    self.p[i][j] += p_pred[i][k] * self.f[j][k];
                }
                self.p[i][j] += self.q[i][j];
            }
        }
    }
    
    pub fn update(&mut self, measurement: &[f64]) -> Vec<f64> {
        // 计算卡尔曼增益
        let mut s = vec![vec![0.0; self.measurement_dim]; self.measurement_dim];
        for i in 0..self.measurement_dim {
            for j in 0..self.measurement_dim {
                for k in 0..self.state_dim {
                    s[i][j] += self.h[i][k] * self.p[k][j];
                }
                s[i][j] += self.r[i][j];
            }
        }
        
        // 简化的卡尔曼增益计算
        let mut k = vec![vec![0.0; self.measurement_dim]; self.state_dim];
        for i in 0..self.state_dim {
            for j in 0..self.measurement_dim {
                k[i][j] = self.p[i][j] / s[j][j];
            }
        }
        
        // 状态更新
        let mut innovation = vec![0.0; self.measurement_dim];
        for i in 0..self.measurement_dim {
            innovation[i] = measurement[i];
            for j in 0..self.state_dim {
                innovation[i] -= self.h[i][j] * self.x[j];
            }
        }
        
        for i in 0..self.state_dim {
            for j in 0..self.measurement_dim {
                self.x[i] += k[i][j] * innovation[j];
            }
        }
        
        // 协方差更新
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                for k in 0..self.measurement_dim {
                    self.p[i][j] -= k[i][k] * s[k][j];
                }
            }
        }
        
        self.x.clone()
    }
}

// 使用示例
fn main() {
    // 创建测试信号
    let sampling_rate = 1000.0;
    let duration = 1.0;
    let n_samples = (sampling_rate * duration) as usize;
    let mut signal_data = Vec::new();
    
    for i in 0..n_samples {
        let t = i as f64 / sampling_rate;
        let frequency = 10.0;
        let amplitude = 1.0;
        let sample = amplitude * (2.0 * PI * frequency * t).sin();
        signal_data.push(sample);
    }
    
    let signal = Signal::new(signal_data, sampling_rate);
    println!("Signal length: {}", signal.length());
    println!("Signal duration: {:.3} seconds", signal.duration());
    
    // FFT分析
    let fft = FFT::new(signal.length());
    let power_spectrum = fft.power_spectrum(&signal.data);
    println!("Power spectrum max: {:.3}", power_spectrum.iter().fold(0.0, |a, &b| a.max(b)));
    
    // FIR滤波器
    let filter_coeffs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
    let mut fir_filter = FIRFilter::new(filter_coeffs);
    let filtered_signal = fir_filter.filter_signal(&signal);
    println!("Filtered signal length: {}", filtered_signal.length());
    
    // IIR滤波器
    let b_coeffs = vec![0.1, 0.2, 0.1];
    let a_coeffs = vec![1.0, -0.5, 0.2];
    let mut iir_filter = IIRFilter::new(b_coeffs, a_coeffs);
    
    for &sample in &signal.data[..10] {
        let filtered = iir_filter.filter(sample);
        println!("Input: {:.3}, Output: {:.3}", sample, filtered);
    }
    
    // 小波变换
    let wavelet = WaveletTransform::new("haar".to_string(), 3);
    let coefficients = wavelet.decompose(&signal.data);
    println!("Wavelet decomposition levels: {}", coefficients.len());
    
    // 自适应滤波器
    let mut adaptive_filter = AdaptiveFilter::new(10, 0.01);
    let desired_signal = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let noisy_signal = vec![1.1, 0.1, 0.9, -0.1, 1.05];
    
    for (input, desired) in noisy_signal.iter().zip(desired_signal.iter()) {
        let output = adaptive_filter.lms_update(*input, *desired);
        println!("Input: {:.3}, Desired: {:.3}, Output: {:.3}", input, desired, output);
    }
    
    // 卡尔曼滤波器
    let mut kalman = KalmanFilter::new(2, 1);
    
    // 设置状态转移矩阵（简化的匀速运动模型）
    kalman.f[0][0] = 1.0;
    kalman.f[0][1] = 0.01;
    kalman.f[1][1] = 1.0;
    
    // 设置观测矩阵
    kalman.h[0][0] = 1.0;
    
    // 设置噪声协方差
    kalman.q[0][0] = 0.1;
    kalman.q[1][1] = 0.1;
    kalman.r[0][0] = 1.0;
    
    let measurements = vec![1.0, 1.1, 1.2, 1.3, 1.4];
    
    for measurement in measurements {
        kalman.predict();
        let state = kalman.update(&[measurement]);
        println!("Measurement: {:.3}, Estimated position: {:.3}, velocity: {:.3}", 
                measurement, state[0], state[1]);
    }
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module SignalProcessingModels where

import Data.List (transpose)
import Data.Complex (Complex(..), magnitude, phase)
import Data.Vector (Vector, (!), fromList, toList)
import qualified Data.Vector as V

-- 信号数据类型
data Signal = Signal {
    signalData :: [Double],
    samplingRate :: Double
} deriving Show

newSignal :: [Double] -> Double -> Signal
newSignal data_list rate = Signal data_list rate

signalLength :: Signal -> Int
signalLength signal = length (signalData signal)

signalDuration :: Signal -> Double
signalDuration signal = fromIntegral (signalLength signal) / samplingRate signal

-- FFT实现
data FFT = FFT {
    fftSize :: Int
} deriving Show

newFFT :: Int -> FFT
newFFT size = FFT size

fft :: FFT -> [Double] -> [Complex Double]
fft fft_obj data = 
    let n = fftSize fft_obj
        complex_data = zipWith (:+) data (replicate n 0.0)
    in fft_recursive complex_data
  where
    fft_recursive [] = []
    fft_recursive [x] = [x]
    fft_recursive xs = 
        let n = length xs
            even_indices = [xs !! i | i <- [0,2..n-1]]
            odd_indices = [xs !! i | i <- [1,3..n-1]]
            even_fft = fft_recursive even_indices
            odd_fft = fft_recursive odd_indices
            w_n = exp (0 :+ (-2 * pi / fromIntegral n))
            twiddle_factors = [w_n ^ k | k <- [0..n-1]]
        in zipWith (+) even_fft (zipWith (*) twiddle_factors odd_fft) ++
           zipWith (-) even_fft (zipWith (*) twiddle_factors odd_fft)

powerSpectrum :: FFT -> [Double] -> [Double]
powerSpectrum fft_obj data = 
    let fft_result = fft fft_obj data
        n = fftSize fft_obj
    in take (n `div` 2) $ map magnitude fft_result

-- FIR滤波器
data FIRFilter = FIRFilter {
    coefficients :: [Double],
    buffer :: [Double]
} deriving Show

newFIRFilter :: [Double] -> FIRFilter
newFIRFilter coeffs = FIRFilter coeffs (replicate (length coeffs) 0.0)

filterSample :: FIRFilter -> Double -> (FIRFilter, Double)
filterSample filter input = 
    let new_buffer = input : init (buffer filter)
        output = sum $ zipWith (*) (coefficients filter) new_buffer
    in (filter { buffer = new_buffer }, output)

filterSignal :: FIRFilter -> Signal -> Signal
filterSignal filter signal = 
    let (_, filtered_data) = foldl (\(f, acc) sample ->
            let (new_f, output) = filterSample f sample
            in (new_f, acc ++ [output])) (filter, []) (signalData signal)
    in Signal filtered_data (samplingRate signal)

-- IIR滤波器
data IIRFilter = IIRFilter {
    bCoeffs :: [Double],
    aCoeffs :: [Double],
    xBuffer :: [Double],
    yBuffer :: [Double]
} deriving Show

newIIRFilter :: [Double] -> [Double] -> IIRFilter
newIIRFilter b_coeffs a_coeffs = IIRFilter {
    bCoeffs = b_coeffs,
    aCoeffs = a_coeffs,
    xBuffer = replicate (length b_coeffs) 0.0,
    yBuffer = replicate (length a_coeffs) 0.0
}

iirFilterSample :: IIRFilter -> Double -> (IIRFilter, Double)
iirFilterSample filter input = 
    let new_x_buffer = input : init (xBuffer filter)
        numerator = sum $ zipWith (*) (bCoeffs filter) new_x_buffer
        denominator = sum $ zipWith (*) (tail (aCoeffs filter)) (yBuffer filter)
        output = (numerator - denominator) / head (aCoeffs filter)
        new_y_buffer = output : init (yBuffer filter)
    in (filter { 
            xBuffer = new_x_buffer,
            yBuffer = new_y_buffer
        }, output)

-- 小波变换
data WaveletTransform = WaveletTransform {
    waveletType :: String,
    decompositionLevels :: Int
} deriving Show

newWaveletTransform :: String -> Int -> WaveletTransform
newWaveletTransform w_type levels = WaveletTransform w_type levels

dwt :: WaveletTransform -> [Double] -> ([Double], [Double])
dwt wavelet signal = 
    let n = length signal
        approximation = [((signal !! (2*i)) + (signal !! (2*i+1))) / 2.0 | i <- [0..(n`div`2)-1]]
        detail = [((signal !! (2*i)) - (signal !! (2*i+1))) / 2.0 | i <- [0..(n`div`2)-1]]
    in (approximation, detail)

idwt :: WaveletTransform -> [Double] -> [Double] -> [Double]
idwt wavelet approx detail = 
    let n = length approx * 2
        reconstructed = concat [[(approx !! i) + (detail !! i), (approx !! i) - (detail !! i)] 
                               | i <- [0..length approx - 1]]
    in take n reconstructed

decompose :: WaveletTransform -> [Double] -> [[Double]]
decompose wavelet signal = go signal (decompositionLevels wavelet)
  where
    go current_signal 0 = [current_signal]
    go current_signal level = 
        let (approx, detail) = dwt wavelet current_signal
        in detail : go approx (level - 1)

-- 自适应滤波器
data AdaptiveFilter = AdaptiveFilter {
    filterLength :: Int,
    learningRate :: Double,
    weights :: [Double],
    buffer :: [Double]
} deriving Show

newAdaptiveFilter :: Int -> Double -> AdaptiveFilter
newAdaptiveFilter length rate = AdaptiveFilter {
    filterLength = length,
    learningRate = rate,
    weights = replicate length 0.0,
    buffer = replicate length 0.0
}

lmsUpdate :: AdaptiveFilter -> Double -> Double -> (AdaptiveFilter, Double)
lmsUpdate filter input desired = 
    let new_buffer = input : init (buffer filter)
        output = sum $ zipWith (*) (weights filter) new_buffer
        error = desired - output
        new_weights = zipWith (\w x -> w + learningRate filter * error * x) 
                              (weights filter) new_buffer
    in (filter { 
            weights = new_weights,
            buffer = new_buffer
        }, output)

-- 卡尔曼滤波器
data KalmanFilter = KalmanFilter {
    stateDim :: Int,
    measurementDim :: Int,
    f :: [[Double]], -- 状态转移矩阵
    h :: [[Double]], -- 观测矩阵
    q :: [[Double]], -- 过程噪声协方差
    r :: [[Double]], -- 测量噪声协方差
    p :: [[Double]], -- 状态协方差
    x :: [Double]    -- 状态估计
} deriving Show

newKalmanFilter :: Int -> Int -> KalmanFilter
newKalmanFilter state_dim meas_dim = KalmanFilter {
    stateDim = state_dim,
    measurementDim = meas_dim,
    f = replicate state_dim (replicate state_dim 0.0),
    h = replicate meas_dim (replicate state_dim 0.0),
    q = replicate state_dim (replicate state_dim 0.0),
    r = replicate meas_dim (replicate meas_dim 0.0),
    p = replicate state_dim (replicate state_dim 0.0),
    x = replicate state_dim 0.0
}

predict :: KalmanFilter -> KalmanFilter
predict filter = 
    let -- 状态预测
        x_pred = [sum [f filter !! i !! j * x filter !! j | j <- [0..stateDim filter - 1]]
                  | i <- [0..stateDim filter - 1]]
        
        -- 协方差预测（简化实现）
        p_pred = replicate (stateDim filter) (replicate (stateDim filter) 0.0)
    in filter { x = x_pred, p = p_pred }

update :: KalmanFilter -> [Double] -> (KalmanFilter, [Double])
update filter measurement = 
    let -- 简化的卡尔曼增益计算
        k = replicate (stateDim filter) (replicate (measurementDim filter) 0.1)
        
        -- 状态更新
        innovation = [measurement !! i - sum [h filter !! i !! j * x filter !! j 
                                            | j <- [0..stateDim filter - 1]]
                     | i <- [0..measurementDim filter - 1]]
        
        x_updated = [x filter !! i + sum [k !! i !! j * innovation !! j 
                                         | j <- [0..measurementDim filter - 1]]
                    | i <- [0..stateDim filter - 1]]
    in (filter { x = x_updated }, x_updated)

-- 示例使用
example :: IO ()
example = do
    -- 创建测试信号
    let sampling_rate = 1000.0
        duration = 1.0
        n_samples = floor (sampling_rate * duration)
        signal_data = [sin (2 * pi * 10 * t / sampling_rate) | t <- [0..n_samples-1]]
        signal = newSignal signal_data sampling_rate
    
    putStrLn $ "Signal length: " ++ show (signalLength signal)
    putStrLn $ "Signal duration: " ++ show (signalDuration signal) ++ " seconds"
    
    -- FFT分析
    let fft_obj = newFFT (signalLength signal)
        power_spectrum = powerSpectrum fft_obj (signalData signal)
        max_power = maximum power_spectrum
    
    putStrLn $ "Power spectrum max: " ++ show max_power
    
    -- FIR滤波器
    let filter_coeffs = [0.1, 0.2, 0.4, 0.2, 0.1]
        fir_filter = newFIRFilter filter_coeffs
        filtered_signal = filterSignal fir_filter signal
    
    putStrLn $ "Filtered signal length: " ++ show (signalLength filtered_signal)
    
    -- 小波变换
    let wavelet = newWaveletTransform "haar" 3
        coefficients = decompose wavelet (signalData signal)
    
    putStrLn $ "Wavelet decomposition levels: " ++ show (length coefficients)
    
    -- 自适应滤波器
    let adaptive_filter = newAdaptiveFilter 10 0.01
        desired_signal = [1.0, 0.0, 1.0, 0.0, 1.0]
        noisy_signal = [1.1, 0.1, 0.9, -0.1, 1.05]
        
        (final_filter, _) = foldl (\(f, _) (input, desired) ->
            lmsUpdate f input desired) (adaptive_filter, 0.0) 
            (zip noisy_signal desired_signal)
    
    putStrLn $ "Final weights: " ++ show (weights final_filter)
    
    -- 卡尔曼滤波器
    let kalman = newKalmanFilter 2 1
        measurements = [1.0, 1.1, 1.2, 1.3, 1.4]
        
        process_measurements filter [] = return ()
        process_measurements filter (m:ms) = do
            let predicted = predict filter
                (updated, state) = update predicted [m]
            putStrLn $ "Measurement: " ++ show m ++ 
                      ", Position: " ++ show (state !! 0) ++ 
                      ", Velocity: " ++ show (state !! 1)
            process_measurements updated ms
    
    process_measurements kalman measurements
```

### 应用领域 / Application Domains

#### 音频处理 / Audio Processing

- **语音识别**: 特征提取、噪声抑制
- **音乐分析**: 频谱分析、音高检测
- **音频压缩**: MP3、AAC编码

#### 图像处理 / Image Processing

- **图像滤波**: 去噪、锐化、平滑
- **边缘检测**: Sobel、Canny算子
- **图像压缩**: JPEG、小波压缩

#### 通信系统 / Communication Systems

- **调制解调**: AM、FM、QAM
- **信道均衡**: 自适应均衡器
- **多路复用**: FDM、TDM、CDM

---

## 参考文献 / References

1. Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing. Pearson.
2. Proakis, J. G., & Manolakis, D. G. (2006). Digital Signal Processing. Pearson.
3. Mallat, S. (2009). A Wavelet Tour of Signal Processing. Academic Press.
4. Haykin, S. (2014). Adaptive Filter Theory. Pearson.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
