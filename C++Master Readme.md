# C++ Algorithm Implementations from Scratch

Three production-quality implementations of fundamental machine learning and optimization algorithms, built from first principles in C++ with comprehensive mathematical documentation.

---

## 📦 Contents

1. [Neural Network with Backpropagation](#1-neural-network-with-backpropagation)
2. [Gaussian Distribution Fitting](#2-gaussian-distribution-fitting)
3. [PageRank Algorithm](#3-pagerank-algorithm)

---

## 1. Neural Network with Backpropagation

### Overview
Feed-forward neural network with 3 layers implementing manual backpropagation via chain rule. No ML libraries - pure implementation using only linear algebra.

### Architecture
```
Input Layer (1) → Hidden Layer 1 (6) → Hidden Layer 2 (7) → Output Layer (2)
```

### Mathematical Foundation

#### Forward Propagation

**Layer 1:**
```
z₁ = W₁x + b₁
a₁ = σ(z₁)
```

**Layer 2:**
```
z₂ = W₂a₁ + b₂
a₂ = σ(z₂)
```

**Layer 3 (Output):**
```
z₃ = W₃a₂ + b₃
a₃ = σ(z₃)
```

#### Activation Function

**Sigmoid:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Derivative:**
```
σ'(z) = cosh(z/2)⁻² / 4
```

#### Loss Function

**Mean Squared Error:**
```
C = 1/n Σᵢ ||aᵢ⁽³⁾ - yᵢ||²
```

#### Backpropagation Gradients

**Output Layer (Layer 3):**
```
∂C/∂W₃ = (1/n) · ∂C/∂a₃ ⊙ σ'(z₃) · a₂ᵀ
∂C/∂b₃ = (1/n) · Σⱼ[∂C/∂a₃ ⊙ σ'(z₃)]

where: ∂C/∂a₃ = 2(a₃ - y)
```

**Hidden Layer 2:**
```
∂C/∂W₂ = (1/n) · [W₃ᵀ(∂C/∂a₃ ⊙ σ'(z₃))] ⊙ σ'(z₂) · a₁ᵀ
∂C/∂b₂ = (1/n) · Σⱼ[W₃ᵀ(∂C/∂a₃ ⊙ σ'(z₃)) ⊙ σ'(z₂)]
```

**Hidden Layer 1:**
```
∂C/∂W₁ = (1/n) · [W₂ᵀ(W₃ᵀ(∂C/∂a₃ ⊙ σ'(z₃)) ⊙ σ'(z₂))] ⊙ σ'(z₁) · xᵀ
∂C/∂b₁ = (1/n) · Σⱼ[W₂ᵀ(W₃ᵀ(∂C/∂a₃ ⊙ σ'(z₃)) ⊙ σ'(z₂)) ⊙ σ'(z₁)]
```

**Notation:**
- `⊙` = element-wise (Hadamard) product
- `·` = matrix multiplication
- `ᵀ` = transpose

#### Gradient Descent Update

```
W ← W - α · ∂C/∂W
b ← b - α · ∂C/∂b
```

where α is the learning rate.

### Key Concepts
- **Chain Rule:** Derivatives flow backward through layers
- **Jacobian Matrices:** Gradients for each parameter
- **Batch Processing:** Efficient matrix operations for multiple samples
- **Weight Initialization:** Small random values prevent symmetry

### Complexity
- **Forward pass:** O(n² · m) where n = layer size, m = samples
- **Backward pass:** O(n² · m)
- **Per epoch:** O(n² · m · k) where k = number of layers

---

## 2. Gaussian Distribution Fitting

### Overview
Fits a Gaussian (normal) distribution to empirical data by minimizing chi-squared error using steepest descent with analytical gradients.

### Problem Statement
Given data points (xᵢ, yᵢ), find optimal parameters μ (mean) and σ (standard deviation) that best fit:

```
f(x; μ, σ) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```

### Mathematical Foundation

#### Gaussian PDF

```
f(x; μ, σ) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))
```

**Parameters:**
- μ: Mean (center of distribution)
- σ: Standard deviation (width of distribution)
- σ²: Variance

#### Cost Function

**Chi-Squared Error:**
```
χ² = Σᵢ (yᵢ - f(xᵢ; μ, σ))²
```

Goal: Minimize χ² with respect to μ and σ

#### Analytical Gradients

**Partial Derivative w.r.t. μ:**

Starting from:
```
f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```

Using chain rule:
```
∂f/∂μ = f(x) · ∂/∂μ[-(x-μ)²/(2σ²)]
      = f(x) · [2(x-μ)/(2σ²)]
      = f(x) · (x-μ)/σ²
```

**Partial Derivative w.r.t. σ:**

Rewrite f as:
```
f(x) = (2π)⁻¹/² · σ⁻¹ · exp(-(x-μ)²/(2σ²))
```

Using product rule and chain rule:
```
∂f/∂σ = (2π)⁻¹/² · [(-σ⁻²) · exp(...) + σ⁻¹ · exp(...) · (x-μ)²/σ³]
      = f(x) · [-1/σ + (x-μ)²/σ³]
```

#### Gradient of Chi-Squared

```
∂χ²/∂μ = Σᵢ ∂/∂μ[(yᵢ - f(xᵢ))²]
       = -2 Σᵢ (yᵢ - f(xᵢ)) · ∂f/∂μ

∂χ²/∂σ = -2 Σᵢ (yᵢ - f(xᵢ)) · ∂f/∂σ
```

#### Steepest Descent Update

```
μ⁽ᵗ⁺¹⁾ = μ⁽ᵗ⁾ - α · ∂χ²/∂μ
σ⁽ᵗ⁺¹⁾ = σ⁽ᵗ⁾ - α · ∂χ²/∂σ
```

where α is the learning rate.

### Convergence Analysis

**Necessary Conditions (Karush-Kuhn-Tucker):**
```
∂χ²/∂μ = 0
∂χ²/∂σ = 0
σ > 0 (constraint)
```

**Convergence Rate:**
- Linear convergence: ||θ⁽ᵗ⁺¹⁾ - θ*|| ≤ c||θ⁽ᵗ⁾ - θ*|| for some c < 1
- Typical iterations to convergence: 20-50 for well-conditioned problems

### Complexity
- **Per iteration:** O(n) where n = number of data points
- **Total:** O(k·n) where k = iterations to convergence

---

## 3. PageRank Algorithm

### Overview
Computes node importance in networks using Markov chain theory. Two implementations: power iteration (O(kn²)) and eigendecomposition (O(n³)).

### Problem Statement
Given a network represented by link matrix L, find the stationary distribution r that satisfies:

```
r = Mr
```

where M is the Google matrix.

### Mathematical Foundation

#### Link Matrix L

Column-stochastic matrix where:
```
L[i,j] = probability of transition FROM node j TO node i

Properties:
Σᵢ L[i,j] = 1  (each column sums to 1)
L[i,j] ≥ 0      (non-negative entries)
```

#### Google Matrix

```
M = dL + ((1-d)/n)J
```

**Components:**
- d: Damping factor (typically 0.85)
- L: Link matrix
- J: Matrix of all ones (n×n)
- n: Number of nodes

**Interpretation:**
- With probability d (85%): Follow random outgoing link
- With probability 1-d (15%): Jump to random page (teleportation)

#### PageRank Equation

```
r = Mr
```

This is an **eigenvector equation** with eigenvalue λ = 1.

#### Perron-Frobenius Theorem

For primitive, non-negative matrix M:

1. **Existence:** There exists a unique dominant eigenvalue λ₁ = 1
2. **Positivity:** Corresponding eigenvector r has all positive entries
3. **Dominance:** All other eigenvalues satisfy |λᵢ| < λ₁

The damping factor makes M primitive, guaranteeing these properties.

### Method 1: Power Iteration

#### Algorithm

```
Initialize: r⁽⁰⁾ = [1/n, 1/n, ..., 1/n]ᵀ

Iterate: r⁽ᵏ⁺¹⁾ = M · r⁽ᵏ⁾

Stop when: ||r⁽ᵏ⁺¹⁾ - r⁽ᵏ⁾|| < ε
```

#### Why It Works

Any vector can be expressed as:
```
r⁽⁰⁾ = c₁v₁ + c₂v₂ + ... + cₙvₙ
```
where vᵢ are eigenvectors with eigenvalues λᵢ.

After k iterations:
```
M^k · r⁽⁰⁾ = c₁λ₁ᵏv₁ + c₂λ₂ᵏv₂ + ... + cₙλₙᵏvₙ
           = c₁(1)ᵏv₁ + c₂λ₂ᵏv₂ + ... + cₙλₙᵏvₙ
```

Since λ₁ = 1 and |λᵢ| < 1 for i > 1:
```
lim[k→∞] M^k · r⁽⁰⁾ = c₁v₁
```

The first term dominates!

#### Convergence Rate

```
||r⁽ᵏ⁺¹⁾ - r*|| ≤ |λ₂|ᵏ · ||r⁽⁰⁾ - r*||
```

**Spectral gap:** |λ₂| determines convergence speed
- Larger gap (|λ₂| << 1): Fast convergence
- Smaller gap (|λ₂| ≈ 1): Slow convergence

Typical: |λ₂| ≈ 0.85 (damping factor), so ~100 iterations needed.

#### Complexity

```
Per iteration: O(n²) matrix-vector multiplication
Total: O(k·n²) where k ≈ 100
```

For sparse networks: O(k·edges) ≈ O(k·n)

### Method 2: Eigendecomposition

#### Algorithm

```
1. Compute eigendecomposition: L = VΛV⁻¹
2. Find eigenvalue λᵢ closest to 1
3. Extract corresponding eigenvector vᵢ
4. Normalize: r = vᵢ / ||vᵢ||₁
```

#### Eigenvalue Decomposition

```
L = VΛV⁻¹

where:
V = [v₁ v₂ ... vₙ]  (eigenvectors as columns)
Λ = diag(λ₁, λ₂, ..., λₙ)  (eigenvalues on diagonal)
```

#### Complexity

```
O(n³) for dense matrices
```

**When to use:**
- Small networks (n < 50)
- Need exact solution
- Research/validation purposes

**Advantages:**
- Exact solution (no iteration)
- Computes all eigenvalues

**Disadvantages:**
- Slow for large networks
- High memory usage
- Unnecessary computation (only need one eigenvector)

### Dangling Nodes

**Problem:** Nodes with no outgoing links create zero columns in L.

**Solution:** Replace zero columns with uniform distribution:
```
if Σᵢ L[i,j] = 0:
    L[:,j] = [1/n, 1/n, ..., 1/n]ᵀ
```

### Comparison: Power Iteration vs Eigendecomposition

| Metric | Power Iteration | Eigendecomposition |
|--------|----------------|-------------------|
| Complexity | O(k·n²) | O(n³) |
| Typical k | ~100 | N/A |
| Scalability | Excellent | Poor |
| Accuracy | ~10⁻⁶ | Exact |
| Memory | O(n²) | O(n²) |
| Sparse support | Yes (O(k·n)) | Limited |

For n=100: Power iteration is ~7-10× faster
For n=1000: Power iteration is ~100× faster
