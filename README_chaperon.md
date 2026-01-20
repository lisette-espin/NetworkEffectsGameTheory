# Bayesian Hiring Model: Network Effects and Publication Count

## Overview

This project implements a Bayesian game-theoretic model for evaluating candidate fit in a hiring scenario. The model uses two observable signals—network effect (G) and publication count (N)—to infer the unobservable candidate fit (F) through Bayesian inference.

## Problem Setup

### Variables

- **F (Fit)**: Binary latent variable representing candidate quality
  - $F \in \{\text{good}, \text{bad}\}$
  - This is what we want to infer (unobservable)

- **G (Network Effect)**: Binary observable signal
  - $G \in \{\text{yes}, \text{no}\}$
  - Indicates whether the candidate has network connections (e.g., worked with top researchers)

- **N (Publication Count)**: Discrete observable signal
  - $N \in \{1, 2, 3, \ldots, 10\}$
  - Number of papers/publications

### Goal

Given observed values of $G$ and $N$, compute the **posterior probability**:
$$\Pr(F = \text{good} \mid G = g, N = n)$$

This probability will be organized in a matrix where:
- **Rows**: Network effect $G$ (yes, no)
- **Columns**: Publication count $N$ (1, 2, ..., 10)
- **Values**: $\Pr(F = \text{good} \mid G, N)$

---

## Step 1: Prior Distribution

We start with a **prior belief** about candidate quality before observing any signals:

$$\Pr(F = \text{good}) = p$$

where $p \in [0, 1]$ is a parameter (default: $p = 0.5$).

The prior for bad candidates is simply:
$$\Pr(F = \text{bad}) = 1 - p$$

---

## Step 2: Likelihood Models

### 2.1 Network Effect Likelihood

We model the probability of having a network effect **conditional on fit**:

$$\Pr(G = \text{yes} \mid F = \text{good}) = q_{\text{good}}$$

$$\Pr(G = \text{yes} \mid F = \text{bad}) = q_{\text{bad}}$$

with the assumption that $q_{\text{good}} > q_{\text{bad}}$ (good candidates are more likely to have network effects).

The complementary probabilities are:
$$\Pr(G = \text{no} \mid F = \text{good}) = 1 - q_{\text{good}}$$

$$\Pr(G = \text{no} \mid F = \text{bad}) = 1 - q_{\text{bad}}$$

**Distribution**: Bernoulli distribution with different success probabilities depending on fit.

### 2.2 Publication Count Likelihood

We model the publication count using a **Poisson distribution** conditional on fit:

$$N \mid (F = \text{good}) \sim \text{Poisson}(\lambda_{\text{good}})$$

$$N \mid (F = \text{bad}) \sim \text{Poisson}(\lambda_{\text{bad}})$$

The probability mass function is:

$$\Pr(N = n \mid F = \text{good}) = e^{-\lambda_{\text{good}}} \frac{\lambda_{\text{good}}^n}{n!}$$

$$\Pr(N = n \mid F = \text{bad}) = e^{-\lambda_{\text{bad}}} \frac{\lambda_{\text{bad}}^n}{n!}$$

where $\lambda_{\text{good}} > \lambda_{\text{bad}}$ (good candidates publish more on average).

**Note**: In practice, we clip $N$ to the range $[1, 10]$ to match the discrete constraint.

### 2.3 Conditional Independence

We assume that, **conditional on fit**, the two signals are independent:

$$(G \perp N) \mid F$$

This means:
$$\Pr(G = g, N = n \mid F = f) = \Pr(G = g \mid F = f) \cdot \Pr(N = n \mid F = f)$$

This assumption allows us to factor the joint likelihood into the product of marginal likelihoods.

---

## Step 3: Joint Likelihood

Using conditional independence, the joint likelihood of observing both signals given fit is:

$$\Pr(G = g, N = n \mid F = \text{good}) = \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good})$$

$$\Pr(G = g, N = n \mid F = \text{bad}) = \Pr(G = g \mid F = \text{bad}) \cdot \Pr(N = n \mid F = \text{bad})$$

---

## Step 4: Marginal Probability (Normalizing Constant)

The marginal probability of observing $(G = g, N = n)$ is computed using the **law of total probability**:

$$\Pr(G = g, N = n) = \Pr(G = g, N = n \mid F = \text{good}) \cdot \Pr(F = \text{good}) + \Pr(G = g, N = n \mid F = \text{bad}) \cdot \Pr(F = \text{bad})$$

$$= \Pr(G = g, N = n \mid F = \text{good}) \cdot p + \Pr(G = g, N = n \mid F = \text{bad}) \cdot (1 - p)$$

This serves as the **normalizing constant** in Bayes' theorem.

---

## Step 5: Bayes' Theorem (Posterior Probability)

Using **Bayes' theorem**, we compute the posterior probability:

$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{\Pr(G = g, N = n \mid F = \text{good}) \cdot \Pr(F = \text{good})}{\Pr(G = g, N = n)}$$

Substituting the expressions:

$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{p \cdot \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good})}{p \cdot \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good}) + (1 - p) \cdot \Pr(G = g \mid F = \text{bad}) \cdot \Pr(N = n \mid F = \text{bad})}$$

### Closed-Form Expression

For the Poisson likelihood, the $n!$ terms cancel out, giving us:

$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{p \cdot \Pr(G = g \mid F = \text{good}) \cdot e^{-\lambda_{\text{good}}} \lambda_{\text{good}}^n}{p \cdot \Pr(G = g \mid F = \text{good}) \cdot e^{-\lambda_{\text{good}}} \lambda_{\text{good}}^n + (1 - p) \cdot \Pr(G = g \mid F = \text{bad}) \cdot e^{-\lambda_{\text{bad}}} \lambda_{\text{bad}}^n}$$

The posterior probability of being bad is simply:

$$\Pr(F = \text{bad} \mid G = g, N = n) = 1 - \Pr(F = \text{good} \mid G = g, N = n)$$

---

## Step 6: Probability Matrix

We compute $\Pr(F = \text{good} \mid G, N)$ for all combinations:

|            | N=1 | N=2 | N=3 | ... | N=10 |
|------------|-----|-----|-----|-----|------|
| **G=yes**  | $p_{1,1}$ | $p_{1,2}$ | $p_{1,3}$ | ... | $p_{1,10}$ |
| **G=no**   | $p_{2,1}$ | $p_{2,2}$ | $p_{2,3}$ | ... | $p_{2,10}$ |

where $p_{i,j} = \Pr(F = \text{good} \mid G = g_i, N = j)$.

---

## Model Parameters

The model requires the following parameters:

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| Prior probability | $p$ | $\Pr(F = \text{good})$ | 0.5 |
| Network (good) | $q_{\text{good}}$ | $\Pr(G = \text{yes} \mid F = \text{good})$ | 0.7 |
| Network (bad) | $q_{\text{bad}}$ | $\Pr(G = \text{yes} \mid F = \text{bad})$ | 0.3 |
| Papers (good) | $\lambda_{\text{good}}$ | Mean papers for good candidates | 5.0 |
| Papers (bad) | $\lambda_{\text{bad}}$ | Mean papers for bad candidates | 2.0 |

**Constraints**:
- $0 \leq p, q_{\text{good}}, q_{\text{bad}} \leq 1$
- $q_{\text{good}} > q_{\text{bad}}$ (good candidates more likely to have network)
- $\lambda_{\text{good}} > \lambda_{\text{bad}} > 0$ (good candidates publish more)

---

## Usage

### Basic Usage

```python
from batch_simulation import BayesianHiringModel

# Initialize model
model = BayesianHiringModel(
    p=0.5,           # Prior probability of being good
    q_good=0.7,      # Network probability for good candidates
    q_bad=0.3,       # Network probability for bad candidates
    lambda_good=5.0, # Average papers for good candidates
    lambda_bad=2.0   # Average papers for bad candidates
)

# Compute probability matrix
prob_matrix = model.compute_probability_matrix(n_max=10)
print(prob_matrix)

# Visualize
model.visualize_matrix(prob_matrix, save_path='heatmap.png')
```

### Running the Full Simulation

```bash
python batch_simulation.py
```

This will:
1. Compute the probability matrix
2. Save it to `probability_matrix.csv`
3. Generate a heatmap visualization
4. Simulate 1000 candidates and analyze prediction accuracy

---

## Interpretation

### High Probability Cells

Cells with high values (close to 1) indicate combinations of $(G, N)$ that strongly suggest a good candidate:
- **G=yes, N=high**: Strong signal (network + many papers)
- **G=yes, N=medium**: Moderate signal (network compensates for fewer papers)

### Low Probability Cells

Cells with low values (close to 0) indicate combinations that suggest a bad candidate:
- **G=no, N=low**: Weak signal (no network + few papers)
- **G=no, N=medium**: Still suggests bad (network effect is important)

### Decision Rule

A simple decision rule:
- **Hire** if $\Pr(F = \text{good} \mid G, N) > 0.5$
- **Reject** if $\Pr(F = \text{good} \mid G, N) \leq 0.5$

---

## Mathematical Summary

**Prior**: $\Pr(F = \text{good}) = p$

**Likelihoods**:
- $\Pr(G = g \mid F = f) = \begin{cases} q_f & \text{if } g = \text{yes} \\ 1 - q_f & \text{if } g = \text{no} \end{cases}$
- $\Pr(N = n \mid F = f) = e^{-\lambda_f} \frac{\lambda_f^n}{n!}$

**Posterior**:
$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{p \cdot \Pr(G = g \mid \text{good}) \cdot \Pr(N = n \mid \text{good})}{p \cdot \Pr(G = g \mid \text{good}) \cdot \Pr(N = n \mid \text{good}) + (1-p) \cdot \Pr(G = g \mid \text{bad}) \cdot \Pr(N = n \mid \text{bad})}$$

---

## Extensions

Possible extensions to the model:
1. **Multiple candidates**: Compare posterior probabilities across candidates
2. **Cost-benefit analysis**: Incorporate hiring costs and benefits
3. **Dynamic updating**: Update prior $p$ based on historical hiring outcomes
5. **Multiple signals**: Add additional observable signals (e.g., citations, h-index)

---

## References

This model is based on Bayesian inference principles and follows the structure of signal-based hiring models in game theory and economics. The mathematical framework is similar to models used in:
- Information economics
- Bayesian games
- Statistical decision theory

---

## License

This project is provided as-is for educational and research purposes.
