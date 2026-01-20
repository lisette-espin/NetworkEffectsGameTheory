# Network Effects in Game Theory: Bayesian Hiring Models

Network Fairness Workshop Group 1

This repository contains implementations of Bayesian game-theoretic models for evaluating candidate fit in hiring scenarios. The models use observable signals (network effects, referrals, publication counts) to infer unobservable candidate quality through Bayesian inference.

---

## Models

This project includes two complementary Bayesian hiring models:

### 1. [Basic Bayesian Hiring Model](./README_chaperon.md) (Chaperon Effect)

**Key Features:**
- Uses **two observable signals**: network effect (binary: yes/no) and publication count (discrete: 1-10)
- **Simple structure**: Direct mapping from signals to candidate fit probability
- **Network effect**: Binary indicator of whether candidate worked with top researchers
- **Output**: Probability matrix showing Pr(F=good | G, N) for all combinations

**Use Case**: When you have straightforward signals about candidates and want a clean Bayesian framework to combine them.

**Files:**
- Model implementation: [`code/batch_simulation_chaperon.py`](./code/batch_simulation_chaperon.py)
- Presentation generator: [`code/helpers/create_presentation.py`](./code/helpers/create_presentation.py)

---

### 2. [Referral-Based Hiring Model](./README_referral.md)

**Key Features:**
- Uses **multiple referrals** (R1, R2) with **trust-based selection**
- **Behavioral insight**: Trust overrides referral quality - decision-makers prioritize trusted referrers even when untrusted sources provide better recommendations
- **Conservative approach**: When both referrers are trusted, uses the worst (lowest quality) referral
- **Network effect**: Directly equals the selected referral quality (5 levels: excellent → very bad)
- **Output**: Probability matrix showing Pr(F=good | G, N) where G is the selected referral quality

**Use Case**: When you receive multiple referrals from different sources and need to model how trust affects information weighting.

**Files:**
- Model implementation: [`code/batch_simulation_fof.py`](./code/batch_simulation_fof.py)
- Presentation generator: [`code/helpers/create_presentation_fof.py`](./code/helpers/create_presentation_fof.py)

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Simulations

**Basic Model:**
```bash

# Save outputs to specific directory
python code/batch_simulation_chaperon.py --dir results
```

**Referral-Based Model:**
```bash

# Save outputs to specific directory
python code/batch_simulation_fof.py --dir results
```

### Generating Presentations

**Basic Model:**
```bash
# Generate PowerPoint presentation
python code/helpers/create_presentation.py --dir pptx
```

**Referral-Based Model:**
```bash
# Generate PowerPoint presentation
python code/helpers/create_presentation_fof.py --dir pptx
```

---

## Key Differences

| Feature | Basic Model (Chaperon effect) | Referral Model (Friend-of-Friend advantage) |
|---------|-------------|----------------|
| **Network Signal** | Binary (yes/no) | 5 levels (excellent → very bad) |
| **Information Sources** | Single network indicator | Multiple referrals with trust |
| **Trust Mechanism** | Not modeled | Explicit trust-based selection |
| **Behavioral Realism** | Simple | Captures trust override behavior |
| **Complexity** | Lower | Higher |

---

## Output Files

Both models generate:
- **Probability Matrix CSV**: Posterior probabilities Pr(F=good | G, N)
- **Heatmap Visualization**: PNG image of the probability matrix
- **Simulated Candidates CSV**: Results from candidate simulations

---

## Documentation

- **[Basic Model Documentation](./README_chaperon.md)**: Complete mathematical formulation and usage guide
- **[Referral Model Documentation](./README_referral.md)**: Detailed explanation of trust-based selection and referral handling

---

## Requirements

See [`requirements.txt`](./requirements.txt) for full list of dependencies.

---

## License

This project is provided as-is for educational and research purposes.
