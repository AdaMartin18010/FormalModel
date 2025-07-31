# 7.4 材料科学模型 / Materials Science Models

## 目录 / Table of Contents

- [7.4 材料科学模型 / Materials Science Models](#74-材料科学模型--materials-science-models)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [7.4.1 晶体结构模型 / Crystal Structure Models](#741-晶体结构模型--crystal-structure-models)
    - [布拉格定律 / Bragg's Law](#布拉格定律--braggs-law)
    - [倒易点阵 / Reciprocal Lattice](#倒易点阵--reciprocal-lattice)
    - [缺陷模型 / Defect Models](#缺陷模型--defect-models)
  - [7.4.2 相变模型 / Phase Transformation Models](#742-相变模型--phase-transformation-models)
    - [成核理论 / Nucleation Theory](#成核理论--nucleation-theory)
    - [生长动力学 / Growth Kinetics](#生长动力学--growth-kinetics)
    - [相图模型 / Phase Diagram Models](#相图模型--phase-diagram-models)
  - [7.4.3 力学性能模型 / Mechanical Properties Models](#743-力学性能模型--mechanical-properties-models)
    - [弹性理论 / Elasticity Theory](#弹性理论--elasticity-theory)
    - [塑性变形 / Plastic Deformation](#塑性变形--plastic-deformation)
    - [蠕变模型 / Creep Models](#蠕变模型--creep-models)
  - [7.4.4 热力学模型 / Thermodynamics Models](#744-热力学模型--thermodynamics-models)
    - [吉布斯自由能 / Gibbs Free Energy](#吉布斯自由能--gibbs-free-energy)
    - [相变热力学 / Phase Transformation Thermodynamics](#相变热力学--phase-transformation-thermodynamics)
    - [固溶体模型 / Solid Solution Models](#固溶体模型--solid-solution-models)
  - [7.4.5 扩散模型 / Diffusion Models](#745-扩散模型--diffusion-models)
    - [菲克定律 / Fick's Laws](#菲克定律--ficks-laws)
    - [扩散机制 / Diffusion Mechanisms](#扩散机制--diffusion-mechanisms)
    - [扩散方程解 / Diffusion Equation Solutions](#扩散方程解--diffusion-equation-solutions)
  - [7.4.6 断裂力学模型 / Fracture Mechanics Models](#746-断裂力学模型--fracture-mechanics-models)
    - [线弹性断裂力学 / Linear Elastic Fracture Mechanics](#线弹性断裂力学--linear-elastic-fracture-mechanics)
    - [塑性断裂力学 / Elastic-Plastic Fracture Mechanics](#塑性断裂力学--elastic-plastic-fracture-mechanics)
    - [疲劳模型 / Fatigue Models](#疲劳模型--fatigue-models)
  - [7.4.7 实现与应用 / Implementation and Applications](#747-实现与应用--implementation-and-applications)
    - [Rust实现示例 / Rust Implementation Example](#rust实现示例--rust-implementation-example)
    - [Haskell实现示例 / Haskell Implementation Example](#haskell实现示例--haskell-implementation-example)
    - [应用领域 / Application Domains](#应用领域--application-domains)
      - [材料设计 / Materials Design](#材料设计--materials-design)
      - [工艺优化 / Process Optimization](#工艺优化--process-optimization)
      - [失效分析 / Failure Analysis](#失效分析--failure-analysis)
  - [参考文献 / References](#参考文献--references)

---

## 7.4.1 晶体结构模型 / Crystal Structure Models

### 布拉格定律 / Bragg's Law

**衍射条件**: $n\lambda = 2d\sin\theta$

**晶面间距**: $d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}$

**衍射强度**: $I = I_0 |F_{hkl}|^2$

### 倒易点阵 / Reciprocal Lattice

**倒易基矢**: $\mathbf{b}_i = 2\pi \frac{\mathbf{a}_j \times \mathbf{a}_k}{\mathbf{a}_i \cdot (\mathbf{a}_j \times \mathbf{a}_k)}$

**倒易点阵矢量**: $\mathbf{G}_{hkl} = h\mathbf{b}_1 + k\mathbf{b}_2 + l\mathbf{b}_3$

**结构因子**: $F_{hkl} = \sum_j f_j e^{i\mathbf{G}_{hkl} \cdot \mathbf{r}_j}$

### 缺陷模型 / Defect Models

**点缺陷浓度**: $c = c_0 e^{-\frac{E_f}{k_B T}}$

**位错密度**: $\rho = \frac{N}{V}$

**位错能量**: $E = \frac{\mu b^2}{4\pi} \ln\left(\frac{R}{r_0}\right)$

---

## 7.4.2 相变模型 / Phase Transformation Models

### 成核理论 / Nucleation Theory

**临界半径**: $r^* = \frac{2\gamma}{\Delta G_v}$

**成核功**: $\Delta G^* = \frac{16\pi \gamma^3}{3(\Delta G_v)^2}$

**成核率**: $J = J_0 e^{-\frac{\Delta G^*}{k_B T}}$

### 生长动力学 / Growth Kinetics

**界面速度**: $v = M \Delta G$

**扩散控制**: $v = \frac{D}{x} \Delta c$

**界面控制**: $v = k \Delta T$

### 相图模型 / Phase Diagram Models

**吉布斯相律**: $F = C - P + 2$

**杠杆定律**: $\frac{W_\alpha}{W_\beta} = \frac{C_\beta - C_0}{C_0 - C_\alpha}$

**共析反应**: $\gamma \rightarrow \alpha + \beta$

---

## 7.4.3 力学性能模型 / Mechanical Properties Models

### 弹性理论 / Elasticity Theory

**胡克定律**: $\sigma_{ij} = C_{ijkl} \epsilon_{kl}$

**杨氏模量**: $E = \frac{\sigma}{\epsilon}$

**泊松比**: $\nu = -\frac{\epsilon_{transverse}}{\epsilon_{axial}}$

**剪切模量**: $G = \frac{E}{2(1 + \nu)}$

### 塑性变形 / Plastic Deformation

**屈服准则**: $\sigma_{eq} = \sigma_y$

**冯·米塞斯准则**: $\sigma_{eq} = \sqrt{\frac{1}{2}[(\sigma_1 - \sigma_2)^2 + (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2]}$

**应变硬化**: $\sigma = K\epsilon^n$

**位错运动**: $v = \frac{D_b}{k_B T} \tau$

### 蠕变模型 / Creep Models

**稳态蠕变**: $\dot{\epsilon} = A\sigma^n e^{-\frac{Q}{RT}}$

**幂律蠕变**: $\dot{\epsilon} = A_1\sigma^n$

**扩散蠕变**: $\dot{\epsilon} = A_2\frac{\sigma}{d^2}$

---

## 7.4.4 热力学模型 / Thermodynamics Models

### 吉布斯自由能 / Gibbs Free Energy

**吉布斯函数**: $G = H - TS$

**化学势**: $\mu_i = \left(\frac{\partial G}{\partial n_i}\right)_{T,P,n_j}$

**相平衡**: $\mu_\alpha = \mu_\beta$

### 相变热力学 / Phase Transformation Thermodynamics

**过冷度**: $\Delta T = T_m - T$

**驱动力**: $\Delta G = \Delta H - T\Delta S$

**形核驱动力**: $\Delta G_v = \frac{\Delta H_v \Delta T}{T_m}$

### 固溶体模型 / Solid Solution Models

**理想固溶体**: $\mu_i = \mu_i^0 + RT\ln x_i$

**规则固溶体**: $\mu_i = \mu_i^0 + RT\ln x_i + \Omega x_j^2$

**活度系数**: $\gamma_i = \frac{a_i}{x_i}$

---

## 7.4.5 扩散模型 / Diffusion Models

### 菲克定律 / Fick's Laws

**第一定律**: $J = -D\frac{\partial c}{\partial x}$

**第二定律**: $\frac{\partial c}{\partial t} = D\frac{\partial^2 c}{\partial x^2}$

**扩散系数**: $D = D_0 e^{-\frac{Q}{RT}}$

### 扩散机制 / Diffusion Mechanisms

**空位扩散**: $D_v = D_0^v e^{-\frac{Q_v}{RT}}$

**间隙扩散**: $D_i = D_0^i e^{-\frac{Q_i}{RT}}$

**晶界扩散**: $D_{gb} = D_0^{gb} e^{-\frac{Q_{gb}}{RT}}$

### 扩散方程解 / Diffusion Equation Solutions

**误差函数解**: $\frac{c - c_0}{c_s - c_0} = 1 - \text{erf}\left(\frac{x}{2\sqrt{Dt}}\right)$

**高斯解**: $c(x,t) = \frac{M}{2\sqrt{\pi Dt}} e^{-\frac{x^2}{4Dt}}$

---

## 7.4.6 断裂力学模型 / Fracture Mechanics Models

### 线弹性断裂力学 / Linear Elastic Fracture Mechanics

**应力强度因子**: $K_I = \sigma\sqrt{\pi a}$

**断裂韧性**: $K_{IC} = \sigma_c\sqrt{\pi a_c}$

**能量释放率**: $G = \frac{K_I^2}{E'}$

### 塑性断裂力学 / Elastic-Plastic Fracture Mechanics

**J积分**: $J = \int_\Gamma \left(W dy - T_i \frac{\partial u_i}{\partial x} ds\right)$

**裂纹张开位移**: $\delta = \frac{K_I^2}{E\sigma_y}$

**CTOD**: $\delta_t = \frac{G}{\sigma_y}$

### 疲劳模型 / Fatigue Models

**S-N曲线**: $N = A(\Delta\sigma)^{-m}$

**Paris定律**: $\frac{da}{dN} = C(\Delta K)^m$

**疲劳极限**: $\sigma_f = \sigma_f' (2N_f)^b$

---

## 7.4.7 实现与应用 / Implementation and Applications

### Rust实现示例 / Rust Implementation Example

```rust
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct CrystalStructure {
    pub lattice_parameters: [f64; 3],
    pub lattice_angles: [f64; 3],
    pub space_group: String,
    pub atomic_positions: Vec<AtomicPosition>,
}

#[derive(Debug, Clone)]
pub struct AtomicPosition {
    pub element: String,
    pub position: [f64; 3],
    pub occupancy: f64,
}

impl CrystalStructure {
    pub fn new(lattice_params: [f64; 3], lattice_angles: [f64; 3], space_group: String) -> Self {
        Self {
            lattice_parameters: lattice_params,
            lattice_angles: lattice_angles,
            space_group,
            atomic_positions: Vec::new(),
        }
    }
    
    pub fn add_atom(&mut self, element: String, position: [f64; 3], occupancy: f64) {
        self.atomic_positions.push(AtomicPosition {
            element,
            position,
            occupancy,
        });
    }
    
    pub fn calculate_volume(&self) -> f64 {
        let a = self.lattice_parameters[0];
        let b = self.lattice_parameters[1];
        let c = self.lattice_parameters[2];
        let alpha = self.lattice_angles[0] * PI / 180.0;
        let beta = self.lattice_angles[1] * PI / 180.0;
        let gamma = self.lattice_angles[2] * PI / 180.0;
        
        a * b * c * (1.0 + 2.0 * alpha.cos() * beta.cos() * gamma.cos() 
                      - alpha.cos().powi(2) - beta.cos().powi(2) - gamma.cos().powi(2)).sqrt()
    }
    
    pub fn bragg_angle(&self, wavelength: f64, h: i32, k: i32, l: i32) -> f64 {
        let d_spacing = self.d_spacing(h, k, l);
        (wavelength / (2.0 * d_spacing)).asin()
    }
    
    pub fn d_spacing(&self, h: i32, k: i32, l: i32) -> f64 {
        let a = self.lattice_parameters[0];
        let b = self.lattice_parameters[1];
        let c = self.lattice_parameters[2];
        
        let h_sq = (h as f64).powi(2);
        let k_sq = (k as f64).powi(2);
        let l_sq = (l as f64).powi(2);
        
        (h_sq / a.powi(2) + k_sq / b.powi(2) + l_sq / c.powi(2)).powf(-0.5)
    }
    
    pub fn structure_factor(&self, h: i32, k: i32, l: i32) -> f64 {
        let mut f_total = 0.0;
        
        for atom in &self.atomic_positions {
            let phase = 2.0 * PI * (h as f64 * atom.position[0] + 
                                   k as f64 * atom.position[1] + 
                                   l as f64 * atom.position[2]);
            f_total += atom.occupancy * phase.cos();
        }
        
        f_total
    }
}

#[derive(Debug)]
pub struct PhaseTransformation {
    pub transformation_type: String,
    pub temperature: f64,
    pub activation_energy: f64,
    pub pre_exponential: f64,
}

impl PhaseTransformation {
    pub fn new(transformation_type: String, temperature: f64, 
               activation_energy: f64, pre_exponential: f64) -> Self {
        Self {
            transformation_type,
            temperature,
            activation_energy,
            pre_exponential,
        }
    }
    
    pub fn nucleation_rate(&self, undercooling: f64, interfacial_energy: f64) -> f64 {
        let kb = 8.617333262145e-5; // eV/K
        let t = self.temperature;
        let delta_g = self.gibbs_free_energy_change(undercooling);
        
        let critical_radius = 2.0 * interfacial_energy / delta_g;
        let nucleation_work = 16.0 * PI * interfacial_energy.powi(3) / (3.0 * delta_g.powi(2));
        
        self.pre_exponential * (-nucleation_work / (kb * t)).exp()
    }
    
    pub fn gibbs_free_energy_change(&self, undercooling: f64) -> f64 {
        // 简化的吉布斯自由能变化计算
        let latent_heat = 1000.0; // J/mol
        let melting_temp = 1000.0; // K
        let t = self.temperature;
        
        latent_heat * undercooling / melting_temp
    }
    
    pub fn growth_velocity(&self, driving_force: f64, mobility: f64) -> f64 {
        mobility * driving_force
    }
    
    pub fn avrami_equation(&self, time: f64, n: f64, k: f64) -> f64 {
        1.0 - (-k * time.powf(n)).exp()
    }
}

#[derive(Debug)]
pub struct MechanicalProperties {
    pub youngs_modulus: f64,
    pub poissons_ratio: f64,
    pub yield_strength: f64,
    pub ultimate_strength: f64,
    pub fracture_toughness: f64,
}

impl MechanicalProperties {
    pub fn new(youngs_modulus: f64, poissons_ratio: f64, yield_strength: f64, 
               ultimate_strength: f64, fracture_toughness: f64) -> Self {
        Self {
            youngs_modulus,
            poissons_ratio,
            yield_strength,
            ultimate_strength,
            fracture_toughness,
        }
    }
    
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }
    
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }
    
    pub fn von_mises_stress(&self, sigma1: f64, sigma2: f64, sigma3: f64) -> f64 {
        let term1 = (sigma1 - sigma2).powi(2);
        let term2 = (sigma2 - sigma3).powi(2);
        let term3 = (sigma3 - sigma1).powi(2);
        
        ((term1 + term2 + term3) / 2.0).sqrt()
    }
    
    pub fn strain_hardening(&self, strain: f64, k: f64, n: f64) -> f64 {
        k * strain.powf(n)
    }
    
    pub fn creep_rate(&self, stress: f64, temperature: f64, activation_energy: f64) -> f64 {
        let kb = 8.617333262145e-5; // eV/K
        let pre_exponential = 1e-6; // s^-1
        let stress_exponent = 5.0;
        
        pre_exponential * (stress / self.yield_strength).powf(stress_exponent) * 
        (-activation_energy / (kb * temperature)).exp()
    }
}

#[derive(Debug)]
pub struct DiffusionModel {
    pub diffusion_coefficient: f64,
    pub activation_energy: f64,
    pub pre_exponential: f64,
}

impl DiffusionModel {
    pub fn new(pre_exponential: f64, activation_energy: f64) -> Self {
        Self {
            diffusion_coefficient: 0.0,
            activation_energy,
            pre_exponential,
        }
    }
    
    pub fn calculate_diffusion_coefficient(&mut self, temperature: f64) {
        let kb = 8.617333262145e-5; // eV/K
        self.diffusion_coefficient = self.pre_exponential * 
            (-self.activation_energy / (kb * temperature)).exp();
    }
    
    pub fn fick_first_law(&self, concentration_gradient: f64) -> f64 {
        -self.diffusion_coefficient * concentration_gradient
    }
    
    pub fn concentration_profile(&self, initial_concentration: f64, surface_concentration: f64,
                               distance: f64, time: f64) -> f64 {
        let erf_arg = distance / (2.0 * (self.diffusion_coefficient * time).sqrt());
        initial_concentration + (surface_concentration - initial_concentration) * 
        (1.0 - self.error_function(erf_arg))
    }
    
    fn error_function(&self, x: f64) -> f64 {
        // 简化的误差函数近似
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }
}

#[derive(Debug)]
pub struct FractureMechanics {
    pub fracture_toughness: f64,
    pub youngs_modulus: f64,
    pub poissons_ratio: f64,
}

impl FractureMechanics {
    pub fn new(fracture_toughness: f64, youngs_modulus: f64, poissons_ratio: f64) -> Self {
        Self {
            fracture_toughness,
            youngs_modulus,
            poissons_ratio,
        }
    }
    
    pub fn stress_intensity_factor(&self, applied_stress: f64, crack_length: f64) -> f64 {
        applied_stress * (PI * crack_length).sqrt()
    }
    
    pub fn critical_crack_length(&self, applied_stress: f64) -> f64 {
        (self.fracture_toughness / (applied_stress * PI.sqrt())).powi(2)
    }
    
    pub fn energy_release_rate(&self, stress_intensity_factor: f64) -> f64 {
        let e_prime = self.youngs_modulus / (1.0 - self.poissons_ratio.powi(2));
        stress_intensity_factor.powi(2) / e_prime
    }
    
    pub fn fatigue_crack_growth(&self, delta_k: f64, c: f64, m: f64) -> f64 {
        c * delta_k.powf(m)
    }
    
    pub fn s_n_curve(&self, stress_amplitude: f64, a: f64, m: f64) -> f64 {
        a * stress_amplitude.powf(-m)
    }
}

#[derive(Debug)]
pub struct ThermodynamicsModel {
    pub temperature: f64,
    pub pressure: f64,
    pub composition: Vec<f64>,
}

impl ThermodynamicsModel {
    pub fn new(temperature: f64, pressure: f64, composition: Vec<f64>) -> Self {
        Self {
            temperature,
            pressure,
            composition,
        }
    }
    
    pub fn gibbs_free_energy(&self, enthalpy: f64, entropy: f64) -> f64 {
        enthalpy - self.temperature * entropy
    }
    
    pub fn chemical_potential(&self, standard_potential: f64, activity: f64) -> f64 {
        let r = 8.314; // J/mol/K
        standard_potential + r * self.temperature * activity.ln()
    }
    
    pub fn phase_equilibrium(&self, potential1: f64, potential2: f64) -> bool {
        (potential1 - potential2).abs() < 1e-6
    }
    
    pub fn lever_rule(&self, composition_alpha: f64, composition_beta: f64, 
                      overall_composition: f64) -> (f64, f64) {
        let fraction_alpha = (composition_beta - overall_composition) / 
                            (composition_beta - composition_alpha);
        let fraction_beta = 1.0 - fraction_alpha;
        (fraction_alpha, fraction_beta)
    }
}

// 使用示例
fn main() {
    // 晶体结构示例
    let mut crystal = CrystalStructure::new(
        [3.615, 3.615, 3.615], // 面心立方铁
        [90.0, 90.0, 90.0],
        "Fm-3m".to_string()
    );
    
    crystal.add_atom("Fe".to_string(), [0.0, 0.0, 0.0], 1.0);
    crystal.add_atom("Fe".to_string(), [0.5, 0.5, 0.0], 1.0);
    crystal.add_atom("Fe".to_string(), [0.5, 0.0, 0.5], 1.0);
    crystal.add_atom("Fe".to_string(), [0.0, 0.5, 0.5], 1.0);
    
    let volume = crystal.calculate_volume();
    let d_spacing = crystal.d_spacing(1, 1, 1);
    let bragg_angle = crystal.bragg_angle(1.54, 1, 1, 1); // Cu Kα
    
    println!("Crystal volume: {:.3} Å³", volume);
    println!("d-spacing (111): {:.3} Å", d_spacing);
    println!("Bragg angle: {:.3}°", bragg_angle * 180.0 / PI);
    
    // 相变示例
    let transformation = PhaseTransformation::new(
        "martensitic".to_string(),
        800.0, // K
        0.5,   // eV
        1e12   // s^-1
    );
    
    let nucleation_rate = transformation.nucleation_rate(100.0, 0.5);
    let growth_velocity = transformation.growth_velocity(1000.0, 1e-6);
    let fraction_transformed = transformation.avrami_equation(100.0, 3.0, 1e-6);
    
    println!("Nucleation rate: {:.3e} m⁻³s⁻¹", nucleation_rate);
    println!("Growth velocity: {:.3e} m/s", growth_velocity);
    println!("Fraction transformed: {:.3}", fraction_transformed);
    
    // 力学性能示例
    let mechanical = MechanicalProperties::new(
        200e9,  // GPa
        0.3,    // 泊松比
        250e6,  // Pa
        400e6,  // Pa
        50e6    // Pa·m^0.5
    );
    
    let shear_modulus = mechanical.shear_modulus();
    let von_mises = mechanical.von_mises_stress(100e6, 50e6, 0.0);
    let creep_rate = mechanical.creep_rate(50e6, 800.0, 2.0);
    
    println!("Shear modulus: {:.1e} Pa", shear_modulus);
    println!("Von Mises stress: {:.1e} Pa", von_mises);
    println!("Creep rate: {:.3e} s⁻¹", creep_rate);
    
    // 扩散示例
    let mut diffusion = DiffusionModel::new(1e-4, 2.0);
    diffusion.calculate_diffusion_coefficient(800.0);
    
    let flux = diffusion.fick_first_law(1000.0);
    let concentration = diffusion.concentration_profile(0.0, 1.0, 1e-6, 3600.0);
    
    println!("Diffusion coefficient: {:.3e} m²/s", diffusion.diffusion_coefficient);
    println!("Diffusion flux: {:.3e} mol/m²s", flux);
    println!("Concentration at 1μm: {:.3}", concentration);
    
    // 断裂力学示例
    let fracture = FractureMechanics::new(50e6, 200e9, 0.3);
    
    let k1 = fracture.stress_intensity_factor(100e6, 1e-3);
    let critical_length = fracture.critical_crack_length(100e6);
    let crack_growth = fracture.fatigue_crack_growth(10e6, 1e-12, 3.0);
    
    println!("Stress intensity factor: {:.1e} Pa·m^0.5", k1);
    println!("Critical crack length: {:.3e} m", critical_length);
    println!("Crack growth rate: {:.3e} m/cycle", crack_growth);
    
    // 热力学示例
    let thermo = ThermodynamicsModel::new(800.0, 1e5, vec![0.5, 0.5]);
    
    let gibbs = thermo.gibbs_free_energy(1000.0, 5.0);
    let chemical_potential = thermo.chemical_potential(-1000.0, 0.5);
    let phase_equilibrium = thermo.phase_equilibrium(-1000.0, -999.9);
    
    println!("Gibbs free energy: {:.1e} J/mol", gibbs);
    println!("Chemical potential: {:.1e} J/mol", chemical_potential);
    println!("Phase equilibrium: {}", phase_equilibrium);
}
```

### Haskell实现示例 / Haskell Implementation Example

```haskell
module MaterialsScienceModels where

import Data.List (sum, length)
import Data.Complex (Complex(..), magnitude)

-- 晶体结构
data CrystalStructure = CrystalStructure {
    latticeParameters :: [Double],
    latticeAngles :: [Double],
    spaceGroup :: String,
    atomicPositions :: [AtomicPosition]
} deriving Show

data AtomicPosition = AtomicPosition {
    element :: String,
    position :: [Double],
    occupancy :: Double
} deriving Show

newCrystalStructure :: [Double] -> [Double] -> String -> CrystalStructure
newCrystalStructure params angles group = CrystalStructure {
    latticeParameters = params,
    latticeAngles = angles,
    spaceGroup = group,
    atomicPositions = []
}

addAtom :: String -> [Double] -> Double -> CrystalStructure -> CrystalStructure
addAtom elem pos occ crystal = crystal {
    atomicPositions = AtomicPosition elem pos occ : atomicPositions crystal
}

calculateVolume :: CrystalStructure -> Double
calculateVolume crystal = 
    let [a, b, c] = latticeParameters crystal
        [alpha, beta, gamma] = map (\x -> x * pi / 180.0) (latticeAngles crystal)
        cos_alpha = cos alpha
        cos_beta = cos beta
        cos_gamma = cos gamma
        term = 1.0 + 2.0 * cos_alpha * cos_beta * cos_gamma 
               - cos_alpha^2 - cos_beta^2 - cos_gamma^2
    in a * b * c * sqrt term

braggAngle :: CrystalStructure -> Double -> Int -> Int -> Int -> Double
braggAngle crystal wavelength h k l = 
    let d_spacing = dSpacing crystal h k l
    in asin (wavelength / (2.0 * d_spacing))

dSpacing :: CrystalStructure -> Int -> Int -> Int -> Double
dSpacing crystal h k l = 
    let [a, b, c] = latticeParameters crystal
        h_sq = fromIntegral h ^ 2
        k_sq = fromIntegral k ^ 2
        l_sq = fromIntegral l ^ 2
    in (h_sq / a^2 + k_sq / b^2 + l_sq / c^2) ** (-0.5)

structureFactor :: CrystalStructure -> Int -> Int -> Int -> Double
structureFactor crystal h k l = 
    let phase_contributions = map (\atom -> 
            let [x, y, z] = position atom
                phase = 2.0 * pi * (fromIntegral h * x + fromIntegral k * y + fromIntegral l * z)
            in occupancy atom * cos phase) (atomicPositions crystal)
    in sum phase_contributions

-- 相变模型
data PhaseTransformation = PhaseTransformation {
    transformationType :: String,
    temperature :: Double,
    activationEnergy :: Double,
    preExponential :: Double
} deriving Show

newPhaseTransformation :: String -> Double -> Double -> Double -> PhaseTransformation
newPhaseTransformation t_type temp act_energy pre_exp = PhaseTransformation {
    transformationType = t_type,
    temperature = temp,
    activationEnergy = act_energy,
    preExponential = pre_exp
}

nucleationRate :: PhaseTransformation -> Double -> Double -> Double
nucleationRate transformation undercooling interfacial_energy = 
    let kb = 8.617333262145e-5 -- eV/K
        t = temperature transformation
        delta_g = gibbsFreeEnergyChange transformation undercooling
        critical_radius = 2.0 * interfacial_energy / delta_g
        nucleation_work = 16.0 * pi * interfacial_energy^3 / (3.0 * delta_g^2)
    in preExponential transformation * exp (-nucleation_work / (kb * t))

gibbsFreeEnergyChange :: PhaseTransformation -> Double -> Double
gibbsFreeEnergyChange transformation undercooling = 
    let latent_heat = 1000.0 -- J/mol
        melting_temp = 1000.0 -- K
        t = temperature transformation
    in latent_heat * undercooling / melting_temp

growthVelocity :: PhaseTransformation -> Double -> Double -> Double
growthVelocity transformation driving_force mobility = mobility * driving_force

avramiEquation :: PhaseTransformation -> Double -> Double -> Double -> Double
avramiEquation transformation time n k = 1.0 - exp (-k * time^n)

-- 力学性能
data MechanicalProperties = MechanicalProperties {
    youngsModulus :: Double,
    poissonsRatio :: Double,
    yieldStrength :: Double,
    ultimateStrength :: Double,
    fractureToughness :: Double
} deriving Show

newMechanicalProperties :: Double -> Double -> Double -> Double -> Double -> MechanicalProperties
newMechanicalProperties e nu sigma_y sigma_u k_ic = MechanicalProperties {
    youngsModulus = e,
    poissonsRatio = nu,
    yieldStrength = sigma_y,
    ultimateStrength = sigma_u,
    fractureToughness = k_ic
}

shearModulus :: MechanicalProperties -> Double
shearModulus props = youngsModulus props / (2.0 * (1.0 + poissonsRatio props))

bulkModulus :: MechanicalProperties -> Double
bulkModulus props = youngsModulus props / (3.0 * (1.0 - 2.0 * poissonsRatio props))

vonMisesStress :: MechanicalProperties -> Double -> Double -> Double -> Double
vonMisesStress props sigma1 sigma2 sigma3 = 
    let term1 = (sigma1 - sigma2)^2
        term2 = (sigma2 - sigma3)^2
        term3 = (sigma3 - sigma1)^2
    in sqrt ((term1 + term2 + term3) / 2.0)

strainHardening :: MechanicalProperties -> Double -> Double -> Double -> Double
strainHardening props strain k n = k * strain^n

creepRate :: MechanicalProperties -> Double -> Double -> Double -> Double
creepRate props stress temperature activation_energy = 
    let kb = 8.617333262145e-5 -- eV/K
        pre_exponential = 1e-6 -- s^-1
        stress_exponent = 5.0
    in pre_exponential * (stress / yieldStrength props)^stress_exponent * 
       exp (-activation_energy / (kb * temperature))

-- 扩散模型
data DiffusionModel = DiffusionModel {
    diffusionCoefficient :: Double,
    activationEnergy :: Double,
    preExponential :: Double
} deriving Show

newDiffusionModel :: Double -> Double -> DiffusionModel
newDiffusionModel pre_exp act_energy = DiffusionModel {
    diffusionCoefficient = 0.0,
    activationEnergy = act_energy,
    preExponential = pre_exp
}

calculateDiffusionCoefficient :: DiffusionModel -> Double -> DiffusionModel
calculateDiffusionCoefficient model temperature = 
    let kb = 8.617333262145e-5 -- eV/K
        d_coeff = preExponential model * exp (-activationEnergy model / (kb * temperature))
    in model { diffusionCoefficient = d_coeff }

fickFirstLaw :: DiffusionModel -> Double -> Double
fickFirstLaw model concentration_gradient = 
    -diffusionCoefficient model * concentration_gradient

concentrationProfile :: DiffusionModel -> Double -> Double -> Double -> Double -> Double
concentrationProfile model c0 cs distance time = 
    let erf_arg = distance / (2.0 * sqrt (diffusionCoefficient model * time))
    in c0 + (cs - c0) * (1.0 - errorFunction erf_arg)

errorFunction :: Double -> Double
errorFunction x = 
    let a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = if x < 0.0 then -1.0 else 1.0
        x_abs = abs x
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp (-x_abs^2)
    in sign * y

-- 断裂力学
data FractureMechanics = FractureMechanics {
    fractureToughness :: Double,
    youngsModulus :: Double,
    poissonsRatio :: Double
} deriving Show

newFractureMechanics :: Double -> Double -> Double -> FractureMechanics
newFractureMechanics k_ic e nu = FractureMechanics {
    fractureToughness = k_ic,
    youngsModulus = e,
    poissonsRatio = nu
}

stressIntensityFactor :: FractureMechanics -> Double -> Double -> Double
stressIntensityFactor fracture applied_stress crack_length = 
    applied_stress * sqrt (pi * crack_length)

criticalCrackLength :: FractureMechanics -> Double -> Double
criticalCrackLength fracture applied_stress = 
    (fractureToughness fracture / (applied_stress * sqrt pi))^2

energyReleaseRate :: FractureMechanics -> Double -> Double
energyReleaseRate fracture stress_intensity_factor = 
    let e_prime = youngsModulus fracture / (1.0 - poissonsRatio fracture^2)
    in stress_intensity_factor^2 / e_prime

fatigueCrackGrowth :: FractureMechanics -> Double -> Double -> Double -> Double
fatigueCrackGrowth fracture delta_k c m = c * delta_k^m

sNCurve :: FractureMechanics -> Double -> Double -> Double -> Double
sNCurve fracture stress_amplitude a m = a * stress_amplitude^(-m)

-- 热力学模型
data ThermodynamicsModel = ThermodynamicsModel {
    temperature :: Double,
    pressure :: Double,
    composition :: [Double]
} deriving Show

newThermodynamicsModel :: Double -> Double -> [Double] -> ThermodynamicsModel
newThermodynamicsModel temp press comp = ThermodynamicsModel {
    temperature = temp,
    pressure = press,
    composition = comp
}

gibbsFreeEnergy :: ThermodynamicsModel -> Double -> Double -> Double
gibbsFreeEnergy thermo enthalpy entropy = enthalpy - temperature thermo * entropy

chemicalPotential :: ThermodynamicsModel -> Double -> Double -> Double
chemicalPotential thermo standard_potential activity = 
    let r = 8.314 -- J/mol/K
    in standard_potential + r * temperature thermo * log activity

phaseEquilibrium :: ThermodynamicsModel -> Double -> Double -> Bool
phaseEquilibrium thermo potential1 potential2 = abs (potential1 - potential2) < 1e-6

leverRule :: ThermodynamicsModel -> Double -> Double -> Double -> (Double, Double)
leverRule thermo comp_alpha comp_beta overall_comp = 
    let fraction_alpha = (comp_beta - overall_comp) / (comp_beta - comp_alpha)
        fraction_beta = 1.0 - fraction_alpha
    in (fraction_alpha, fraction_beta)

-- 示例使用
example :: IO ()
example = do
    -- 晶体结构示例
    let crystal = addAtom "Fe" [0.0, 0.0, 0.0] 1.0 $
                 addAtom "Fe" [0.5, 0.5, 0.0] 1.0 $
                 addAtom "Fe" [0.5, 0.0, 0.5] 1.0 $
                 addAtom "Fe" [0.0, 0.5, 0.5] 1.0 $
                 newCrystalStructure [3.615, 3.615, 3.615] [90.0, 90.0, 90.0] "Fm-3m"
    
    let volume = calculateVolume crystal
        d_spacing = dSpacing crystal 1 1 1
        bragg_angle = braggAngle crystal 1.54 1 1 1
    
    putStrLn $ "Crystal volume: " ++ show volume ++ " Å³"
    putStrLn $ "d-spacing (111): " ++ show d_spacing ++ " Å"
    putStrLn $ "Bragg angle: " ++ show (bragg_angle * 180.0 / pi) ++ "°"
    
    -- 相变示例
    let transformation = newPhaseTransformation "martensitic" 800.0 0.5 1e12
        nucleation_rate = nucleationRate transformation 100.0 0.5
        growth_velocity = growthVelocity transformation 1000.0 1e-6
        fraction_transformed = avramiEquation transformation 100.0 3.0 1e-6
    
    putStrLn $ "Nucleation rate: " ++ show nucleation_rate ++ " m⁻³s⁻¹"
    putStrLn $ "Growth velocity: " ++ show growth_velocity ++ " m/s"
    putStrLn $ "Fraction transformed: " ++ show fraction_transformed
    
    -- 力学性能示例
    let mechanical = newMechanicalProperties 200e9 0.3 250e6 400e6 50e6
        shear_modulus = shearModulus mechanical
        von_mises = vonMisesStress mechanical 100e6 50e6 0.0
        creep_rate = creepRate mechanical 50e6 800.0 2.0
    
    putStrLn $ "Shear modulus: " ++ show shear_modulus ++ " Pa"
    putStrLn $ "Von Mises stress: " ++ show von_mises ++ " Pa"
    putStrLn $ "Creep rate: " ++ show creep_rate ++ " s⁻¹"
    
    -- 扩散示例
    let diffusion = calculateDiffusionCoefficient (newDiffusionModel 1e-4 2.0) 800.0
        flux = fickFirstLaw diffusion 1000.0
        concentration = concentrationProfile diffusion 0.0 1.0 1e-6 3600.0
    
    putStrLn $ "Diffusion coefficient: " ++ show (diffusionCoefficient diffusion) ++ " m²/s"
    putStrLn $ "Diffusion flux: " ++ show flux ++ " mol/m²s"
    putStrLn $ "Concentration at 1μm: " ++ show concentration
    
    -- 断裂力学示例
    let fracture = newFractureMechanics 50e6 200e9 0.3
        k1 = stressIntensityFactor fracture 100e6 1e-3
        critical_length = criticalCrackLength fracture 100e6
        crack_growth = fatigueCrackGrowth fracture 10e6 1e-12 3.0
    
    putStrLn $ "Stress intensity factor: " ++ show k1 ++ " Pa·m^0.5"
    putStrLn $ "Critical crack length: " ++ show critical_length ++ " m"
    putStrLn $ "Crack growth rate: " ++ show crack_growth ++ " m/cycle"
    
    -- 热力学示例
    let thermo = newThermodynamicsModel 800.0 1e5 [0.5, 0.5]
        gibbs = gibbsFreeEnergy thermo 1000.0 5.0
        chemical_potential = chemicalPotential thermo (-1000.0) 0.5
        phase_equilibrium = phaseEquilibrium thermo (-1000.0) (-999.9)
    
    putStrLn $ "Gibbs free energy: " ++ show gibbs ++ " J/mol"
    putStrLn $ "Chemical potential: " ++ show chemical_potential ++ " J/mol"
    putStrLn $ "Phase equilibrium: " ++ show phase_equilibrium
```

### 应用领域 / Application Domains

#### 材料设计 / Materials Design

- **合金设计**: 成分优化、相图计算
- **结构材料**: 强度设计、韧性优化
- **功能材料**: 磁性、电性、光学性能

#### 工艺优化 / Process Optimization

- **热处理**: 相变控制、组织调控
- **成形工艺**: 塑性变形、断裂控制
- **表面处理**: 扩散、腐蚀、涂层

#### 失效分析 / Failure Analysis

- **断裂分析**: 裂纹扩展、疲劳寿命
- **腐蚀分析**: 电化学、应力腐蚀
- **磨损分析**: 摩擦磨损、磨粒磨损

---

## 参考文献 / References

1. Callister, W. D., & Rethwisch, D. G. (2018). Materials Science and Engineering. Wiley.
2. Ashby, M. F., & Jones, D. R. H. (2012). Engineering Materials. Butterworth-Heinemann.
3. Porter, D. A., & Easterling, K. E. (2009). Phase Transformations in Metals and Alloys. CRC Press.
4. Anderson, T. L. (2017). Fracture Mechanics. CRC Press.

---

*最后更新: 2025-08-01*
*版本: 1.0.0*
