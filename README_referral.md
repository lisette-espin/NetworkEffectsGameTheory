# Bayesian Referral-Based Hiring Model

## Overview

This project implements a Bayesian game-theoretic model for evaluating candidate fit in a hiring scenario with **multiple referrals and trust-based selection**. The model captures a key behavioral insight: **decision-makers prioritize referrals from trusted sources, even when untrusted sources provide better recommendations**.

## Problem Setup

### The Hiring Scenario

Imagine you're hiring a candidate and receive **two referrals**:
- **Referral 1 (R1)**: From someone you may or may not know/trust
- **Referral 2 (R2)**: From someone else you may or may not know/trust

**Key Behavioral Assumptions**: 
1. If you trust one referrer but not the other, you will use the trusted referrer's recommendation **even if the untrusted referrer gives a better recommendation**.
2. If you trust both referrers, you take a **conservative approach** and use the **worst (lowest quality) referral**.

**Example**:
- Referral 1: "Excellent candidate" (from someone you don't know)
- Referral 2: "Very Bad candidate" (from your trusted colleague)
- **Decision**: You use Referral 2 because you trust that source more

### Variables

- **F (Fit)**: Binary latent variable representing candidate quality
  - $F \in \{\text{good}, \text{bad}\}$
  - This is what we want to infer (unobservable)

- **R1, R2 (Referrals)**: Observable signals from two referrers
  - $R_1, R_2 \in \{\text{excellent}, \text{good}, \text{average}, \text{bad}, \text{very bad}\}$
  - Quality levels ordered from best to worst

- **T1, T2 (Trust)**: Binary indicators of trust in referrers
  - $T_1, T_2 \in \{\text{trust}, \text{no trust}\}$
  - Known to the decision-maker

- **G (Network Effect)**: Observable signal - **same as the selected referral quality**
  - $G \in \{\text{excellent}, \text{good}, \text{average}, \text{bad}, \text{very bad}\}$
  - Network effect G is directly the quality of the selected referral (no mapping)

- **N (Publication Count)**: Discrete observable signal
  - $N \in \{1, 2, 3, \ldots, 10\}$
  - Number of papers/publications

### Goal

Given observed values of referrals $(R_1, R_2)$, trust $(T_1, T_2)$, and paper count $N$, compute the **posterior probability**:
$$\Pr(F = \text{good} \mid R_1, R_2, T_1, T_2, N)$$

However, since we select which referral to use based on trust, we effectively compute:
$$\Pr(F = \text{good} \mid G, N)$$

where $G$ is the network effect derived from the selected referral.

---

## Step 1: Prior Distribution

We start with a **prior belief** about candidate quality:

$$\Pr(F = \text{good}) = p$$

where $p \in [0, 1]$ is a parameter (default: $p = 0.5$).

---

## Step 2: Referral Quality Likelihood

### 2.1 Referral Quality Distribution

Referral quality depends on the candidate's true fit. We model this with different probability distributions:

**For Good Candidates** ($F = \text{good}$):
- Higher probability of positive referrals (excellent, good)
- Lower probability of negative referrals (bad, very bad)

**For Bad Candidates** ($F = \text{bad}$):
- Higher probability of negative referrals (bad, very bad)
- Lower probability of positive referrals (excellent, good)

The exact probabilities depend on two parameters:

1. **Referral Accuracy** ($\alpha$): Probability that a referral correctly reflects candidate quality
   - Higher $\alpha$ → more accurate referrals
   - Lower $\alpha$ → more noise in referrals

2. **Referral Bias** ($\beta$): Probability of positive bias (referrers inflate quality)
   - Accounts for referrers being overly positive

### 2.2 Mathematical Formulation

For good candidates:
$$\Pr(R = r \mid F = \text{good}) = f_{\text{good}}(r, \alpha, \beta)$$

For bad candidates:
$$\Pr(R = r \mid F = \text{bad}) = f_{\text{bad}}(r, \alpha, \beta)$$

where $r \in \{\text{excellent}, \text{good}, \text{average}, \text{bad}, \text{very bad}\}$.

**Example probabilities** (with $\alpha = 0.8$, $\beta = 0.1$):

| Referral Quality | Pr(R | F=good) | Pr(R | F=bad) |
|-----------------|----------------|----------------|
| Excellent       | 0.33          | 0.05          |
| Good            | 0.25          | 0.10          |
| Average         | 0.18          | 0.15          |
| Bad             | 0.12          | 0.30          |
| Very Bad        | 0.12          | 0.40          |

---

## Step 3: Trust-Based Referral Selection

### 3.1 Selection Rule

The decision-maker selects which referral to use based on **trust** and **quality**:

**Decision Rule**:
1. **If trust only R1**: Use R1 (regardless of R2 quality)
2. **If trust only R2**: Use R2 (regardless of R1 quality)
3. **If trust both**: Use the **lowest (worst) quality** referral (conservative approach)
4. **If trust neither**: Use the better referral

**Mathematical Formulation**:

$$R_{\text{selected}} = \begin{cases}
R_1 & \text{if } T_1 = \text{trust} \text{ and } T_2 = \text{no trust} \\
R_2 & \text{if } T_2 = \text{trust} \text{ and } T_1 = \text{no trust} \\
\min(R_1, R_2) & \text{if } T_1 = T_2 = \text{trust} \\
\max(R_1, R_2) & \text{if } T_1 = T_2 = \text{no trust}
\end{cases}$$

where:
- $\max(R_1, R_2)$ means the better quality referral
- $\min(R_1, R_2)$ means the worse quality referral (conservative when both are trusted)

### 3.2 Key Insight: Trust Overrides Quality

This is the **core behavioral assumption**: Trust in a referrer can override referral quality. Even if an untrusted referrer gives an "excellent" recommendation and a trusted referrer gives a "very bad" one, the decision-maker uses the trusted referrer's recommendation.

**Example Scenarios**:

| R1 Quality | R2 Quality | Trust R1 | Trust R2 | Selected Referral |
|-----------|------------|----------|----------|-------------------|
| Excellent | Very Bad   | No       | Yes      | R2 (Very Bad)     |
| Very Bad  | Excellent  | Yes      | No       | R1 (Very Bad)     |
| Good      | Bad        | Yes      | Yes      | R2 (Bad) - lowest |
| Excellent | Good       | No       | No       | R1 (Excellent)    |

---

## Step 4: Network Effect

The network effect **IS** the selected referral quality. There is no mapping - they are the same:

$$G = R_{\text{selected}}$$

**Network Effect Levels**:

| Network Effect G | Description |
|-----------------|-------------|
| Excellent       | Highest network effect |
| Good            | Good network effect |
| Average         | Average network effect |
| Bad             | Poor network effect |
| Very Bad        | Very poor network effect |

---

## Step 5: Publication Count Likelihood

We model the publication count using a **Poisson distribution** conditional on fit:

$$N \mid (F = \text{good}) \sim \text{Poisson}(\lambda_{\text{good}})$$

$$N \mid (F = \text{bad}) \sim \text{Poisson}(\lambda_{\text{bad}})$$

The probability mass function is:

$$\Pr(N = n \mid F = \text{good}) = e^{-\lambda_{\text{good}}} \frac{\lambda_{\text{good}}^n}{n!}$$

$$\Pr(N = n \mid F = \text{bad}) = e^{-\lambda_{\text{bad}}} \frac{\lambda_{\text{bad}}^n}{n!}$$

where $\lambda_{\text{good}} > \lambda_{\text{bad}}$ (good candidates publish more on average).

**Note**: In practice, we clip $N$ to the range $[1, 10]$ to match the discrete constraint.

---

## Step 6: Network Effect Likelihood

Since network effect $G$ **is** the referral quality, the likelihood is simply:

$$\Pr(G = g \mid F = f) = \Pr(R = g \mid F = f)$$

There is no mapping or aggregation needed - the network effect level directly corresponds to the referral quality level.

**Examples**:
$$\Pr(G = \text{excellent} \mid F = \text{good}) = \Pr(R = \text{excellent} \mid F = \text{good})$$

$$\Pr(G = \text{good} \mid F = \text{good}) = \Pr(R = \text{good} \mid F = \text{good})$$

$$\Pr(G = \text{very bad} \mid F = \text{bad}) = \Pr(R = \text{very bad} \mid F = \text{bad})$$

---

## Step 7: Conditional Independence

We assume that, **conditional on fit**, network effect and publication count are independent:

$$(G \perp N) \mid F$$

This means:
$$\Pr(G = g, N = n \mid F = f) = \Pr(G = g \mid F = f) \cdot \Pr(N = n \mid F = f)$$

---

## Step 8: Joint Likelihood

Using conditional independence, the joint likelihood is:

$$\Pr(G = g, N = n \mid F = \text{good}) = \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good})$$

$$\Pr(G = g, N = n \mid F = \text{bad}) = \Pr(G = g \mid F = \text{bad}) \cdot \Pr(N = n \mid F = \text{bad})$$

---

## Step 9: Marginal Probability (Normalizing Constant)

The marginal probability of observing $(G = g, N = n)$ is:

$$\Pr(G = g, N = n) = \Pr(G = g, N = n \mid F = \text{good}) \cdot \Pr(F = \text{good}) + \Pr(G = g, N = n \mid F = \text{bad}) \cdot \Pr(F = \text{bad})$$

$$= \Pr(G = g, N = n \mid F = \text{good}) \cdot p + \Pr(G = g, N = n \mid F = \text{bad}) \cdot (1 - p)$$

---

## Step 10: Bayes' Theorem (Posterior Probability)

Using **Bayes' theorem**, we compute the posterior probability:

$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{\Pr(G = g, N = n \mid F = \text{good}) \cdot \Pr(F = \text{good})}{\Pr(G = g, N = n)}$$

Substituting the expressions:

$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{p \cdot \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good})}{p \cdot \Pr(G = g \mid F = \text{good}) \cdot \Pr(N = n \mid F = \text{good}) + (1 - p) \cdot \Pr(G = g \mid F = \text{bad}) \cdot \Pr(N = n \mid F = \text{bad})}$$

The posterior probability of being bad is:

$$\Pr(F = \text{bad} \mid G = g, N = n) = 1 - \Pr(F = \text{good} \mid G = g, N = n)$$

---

## Step 11: Expected Payoff Calculation

Given the posterior probability, we can compute the **expected employer payoff** from hiring:

$$\text{Expected Payoff} = B \cdot \Pr(F = \text{good} \mid G, N) + b \cdot \Pr(F = \text{bad} \mid G, N)$$

Substituting $\Pr(F = \text{bad} \mid G, N) = 1 - \Pr(F = \text{good} \mid G, N)$:

$$\text{Expected Payoff} = B \cdot P_{\text{good}} + b \cdot (1 - P_{\text{good}}) = b + (B - b) \cdot P_{\text{good}}$$

where $P_{\text{good}} = \Pr(F = \text{good} \mid G, N)$.

**Interpretation**:
- If $P_{\text{good}} = 1$ (certainly good): Expected Payoff = $B$
- If $P_{\text{good}} = 0$ (certainly bad): Expected Payoff = $b$
- If $P_{\text{good}} = 0.5$: Expected Payoff = $(B + b) / 2$

**Decision Rule**:
- **Hire** if Expected Payoff $> 0$ (or some threshold)
- **Reject** if Expected Payoff $\leq 0$ (or below threshold)

---

## Step 12: Probability Matrix

We compute $\Pr(F = \text{good} \mid G, N)$ for all combinations:

|            | N=1 | N=2 | N=3 | ... | N=10 |
|------------|-----|-----|-----|-----|------|
| **G=excellent** | $p_{1,1}$ | $p_{1,2}$ | $p_{1,3}$ | ... | $p_{1,10}$ |
| **G=good** | $p_{2,1}$ | $p_{2,2}$ | $p_{2,3}$ | ... | $p_{2,10}$ |
| **G=average** | $p_{3,1}$ | $p_{3,2}$ | $p_{3,3}$ | ... | $p_{3,10}$ |
| **G=bad** | $p_{4,1}$ | $p_{4,2}$ | $p_{4,3}$ | ... | $p_{4,10}$ |
| **G=very bad** | $p_{5,1}$ | $p_{5,2}$ | $p_{5,3}$ | ... | $p_{5,10}$ |

where $p_{i,j} = \Pr(F = \text{good} \mid G = g_i, N = j)$.

---

## Model Parameters

The model requires the following parameters:

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| Prior probability | $p$ | $\Pr(F = \text{good})$ | 0.5 |
| Papers (good) | $\lambda_{\text{good}}$ | Mean papers for good candidates | 5.0 |
| Papers (bad) | $\lambda_{\text{bad}}$ | Mean papers for bad candidates | 2.0 |
| Referral accuracy | $\alpha$ | Probability referral reflects true quality | 0.8 |
| Referral bias | $\beta$ | Probability of positive bias | 0.1 |
| Trust R1 | $\Pr(T_1 = \text{trust})$ | Probability of trusting referrer 1 | 0.5 |
| Trust R2 | $\Pr(T_2 = \text{trust})$ | Probability of trusting referrer 2 | 0.5 |
| Payoff (good) | $B$ | Employer payoff from hiring good candidate | 10.0 |
| Payoff (bad) | $b$ | Employer payoff from hiring bad candidate | 2.0 |
| Applicant payoff | $w$ | Applicant payoff if hired | 5.0 |

**Constraints**:
- $0 \leq p, \alpha, \beta, \Pr(T_1), \Pr(T_2) \leq 1$
- $\lambda_{\text{good}} > \lambda_{\text{bad}} > 0$
- $B > b$ (hiring good candidate is more valuable)
- $w > 0$ (applicant receives positive payoff if hired)

---

## Key Behavioral Insights

### 1. Trust Overrides Quality

The model captures a realistic behavioral pattern: **trust in information sources can override the quality of information itself**. This is consistent with research on source credibility and social influence.

### 2. Conservative Approach with Multiple Trusted Sources

When both referrers are trusted, the model uses a **conservative approach** by selecting the worst (lowest quality) referral. This reflects risk-averse decision-making: when you trust both sources but they disagree, you take the more pessimistic view.

### 3. Network Effect Equals Referral Quality

Network effect **is** the referral quality - there is no separate mapping. The selected referral quality directly becomes the network effect signal $G$. This simplifies the model and makes the relationship between referrals and network effect transparent.

### 4. Multiple Information Sources

The model handles **multiple information sources** (two referrals) and shows how trust-based selection affects inference.

---

## Example Scenarios

### Scenario 1: Trust Override

- **R1**: Excellent (untrusted)
- **R2**: Very Bad (trusted)
- **Selected**: R2 → G = Very Bad
- **N**: 8 papers
- **Result**: Low posterior probability of being good (despite excellent R1)

### Scenario 2: Both Trusted (Conservative)

- **R1**: Good (trusted)
- **R2**: Bad (trusted)
- **Selected**: R2 (lowest) → G = Bad
- **N**: 5 papers
- **Result**: Lower posterior probability (conservative approach when both trusted)

### Scenario 3: Excellent Network + Many Papers

- **R1**: Excellent (trusted)
- **R2**: Good (untrusted)
- **Selected**: R1 → G = Excellent
- **N**: 9 papers
- **Result**: Very high posterior probability

---

## Usage

### Basic Usage

```python
from batch_simulation_fof import ReferralBasedHiringModel

# Initialize model
model = ReferralBasedHiringModel(
    p=0.5,                    # Prior probability of being good
    lambda_good=5.0,          # Average papers for good candidates
    lambda_bad=2.0,           # Average papers for bad candidates
    referral_accuracy=0.8,    # Referral accuracy
    referral_bias=0.1,        # Positive bias probability
    B=10.0,                   # Payoff from hiring good candidate
    b=2.0,                    # Payoff from hiring bad candidate
    w=5.0                     # Applicant payoff if hired
)

# Compute probability matrix
prob_matrix = model.compute_probability_matrix(n_max=10)
print(prob_matrix)

# Compute expected payoff for a specific candidate
expected_payoff = model.expected_payoff(network_level='excellent', n=5)
print(f"Expected Payoff: {expected_payoff:.2f}")

# Make decision based on probability
should_hire_prob = model.should_hire_probability_based(network_level='excellent', n=5, threshold=0.5)

# Make decision based on payoff
should_hire_payoff = model.should_hire_payoff_based(network_level='excellent', n=5, threshold=0.0)

# Visualize
model.visualize_matrix(prob_matrix, save_path='heatmap.png')
```

### Running the Full Simulation

```bash
# With default parameters
python code/batch_simulation_fof.py --dir results

# With custom payoffs
python code/batch_simulation_fof.py --dir results --B 15.0 --b -5.0 --w 8.0
```

This will:
1. Compute the probability matrix
2. Save it to `referral_probability_matrix.csv`
3. Generate a heatmap visualization
4. Simulate candidates and compute both probability-based and payoff-based decisions
5. Compare accuracy and expected payoffs between the two approaches
6. Analyze trust-based referral selection
7. Save results to `referral_simulated_candidates.csv`

---

## Interpretation

### High Probability Cells

Cells with high values (close to 1) indicate combinations of $(G, N)$ that strongly suggest a good candidate:
- **G=excellent, N=high**: Strong signal (excellent referral + many papers)
- **G=good, N=high**: Moderate-strong signal (good referral + many papers)
- **G=excellent, N=medium**: Strong signal (excellent referral compensates for fewer papers)

### Low Probability Cells

Cells with low values (close to 0) indicate combinations that suggest a bad candidate:
- **G=very bad, N=low**: Weak signal (very bad referral + few papers)
- **G=bad, N=low**: Weak signal (bad referral + few papers)
- **G=very bad, N=medium**: Still suggests bad (referral quality matters more)

### Decision Rules

The model provides two decision-making approaches:

**Probability-Based**:
- **Hire** if $\Pr(F = \text{good} \mid G, N) > 0.5$
- **Reject** if $\Pr(F = \text{good} \mid G, N) \leq 0.5$

**Payoff-Based**:
- **Hire** if Expected Payoff $= b + (B-b) \cdot \Pr(F = \text{good} \mid G, N) > 0$
- **Reject** if Expected Payoff $\leq 0$

The payoff-based approach is more flexible and realistic, as it accounts for the different values of hiring good vs. bad candidates.

---

## Mathematical Summary

**Prior**: $\Pr(F = \text{good}) = p$

**Referral Selection**:
$$R_{\text{selected}} = f(R_1, R_2, T_1, T_2)$$

**Network Effect**:
$$G = \text{map}(R_{\text{selected}})$$

**Likelihoods**:
- $\Pr(G = g \mid F = f) = \Pr(R = g \mid F = f)$ (since $G = R$)
- $\Pr(N = n \mid F = f) = e^{-\lambda_f} \frac{\lambda_f^n}{n!}$

**Posterior**:
$$\Pr(F = \text{good} \mid G = g, N = n) = \frac{p \cdot \Pr(G = g \mid \text{good}) \cdot \Pr(N = n \mid \text{good})}{p \cdot \Pr(G = g \mid \text{good}) \cdot \Pr(N = n \mid \text{good}) + (1-p) \cdot \Pr(G = g \mid \text{bad}) \cdot \Pr(N = n \mid \text{bad})}$$

**Expected Payoff**:
$$\text{Expected Payoff} = B \cdot \Pr(F = \text{good} \mid G, N) + b \cdot \Pr(F = \text{bad} \mid G, N) = b + (B - b) \cdot \Pr(F = \text{good} \mid G, N)$$

---

## Payoff Modeling vs. Probability-Based Decisions

### When to Use Probability-Based Decisions

**Advantages**:
- Simple and intuitive
- No need to estimate payoffs
- Works well when good/bad candidates have similar value
- Good for exploratory analysis

**Limitations**:
- Assumes symmetric payoffs (hiring good = avoiding bad)
- Fixed threshold (0.5) may not be optimal
- Doesn't account for different costs/benefits

### When to Use Payoff-Based Decisions

**Advantages**:
- More realistic and flexible
- Accounts for asymmetric payoffs
- Can optimize for expected utility
- Better for actual hiring decisions

**Limitations**:
- Requires estimating payoff values
- More complex to implement
- Payoff estimates may be uncertain

### Example: When They Differ

Consider a candidate with $\Pr(F = \text{good} \mid G, N) = 0.55$:

**Probability-based**: Hire (55% > 50%)

**Payoff-based** (with $B = 10$, $b = -10$):
- Expected Payoff = $-10 + 20 \times 0.55 = 1 > 0$ → **Hire**

**Payoff-based** (with $B = 10$, $b = -30$):
- Expected Payoff = $-30 + 40 \times 0.55 = -8 < 0$ → **Reject**

When the cost of hiring a bad candidate is very high, the payoff-based approach is more conservative.

## Extensions

Possible extensions to the model:
1. **More referrers**: Extend to $k > 2$ referrers
2. **Trust levels**: Continuous trust scores instead of binary
3. **Referral weighting**: Weight referrals by trust level rather than binary selection
4. **Temporal dynamics**: Update trust based on past referral accuracy
5. **Referrer reputation**: Model referrer reputation as a function of past accuracy
6. **Cost-benefit analysis**: Already incorporated via payoff modeling
7. **Risk aversion**: Incorporate risk preferences (e.g., utility functions)

---

## Comparison with Basic Model

| Feature | Basic Model (Chaperon Effect) | Referral Model |
|---------|-------------|----------------|
| Network signal | Binary (yes/no) | Five levels (excellent/good/average/bad/very bad) |
| Information sources | Single signal | Multiple referrals (2 referrers) |
| Trust mechanism | Not modeled | Explicit trust-based selection |
| Behavioral realism | Simple | Captures trust override behavior |
| Payoff modeling | Supported | Supported |
| Complexity | Lower | Higher |

---

## References

This model extends Bayesian inference principles to capture:
- **Source credibility effects** in information processing
- **Trust-based decision making** in hiring contexts
- **Multi-source information aggregation** with heterogeneous source reliability

The trust-override mechanism is consistent with research on:
- Social influence and source credibility
- Heuristic decision making
- Information cascades

---

## License

This project is provided as-is for educational and research purposes.
