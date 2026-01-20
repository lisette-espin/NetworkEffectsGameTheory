"""
Bayesian Referral-Based Hiring Model Simulation
Models the posterior probability of candidate fit (F) given referrals, trust, and paper count (N).

Key feature: Decision-maker prioritizes referrals from trusted sources, even if untrusted sources
provide better recommendations.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ReferralBasedHiringModel:
    """
    Bayesian model for evaluating candidate fit based on referrals, trust, and publication count.
    Supports both probability-based and payoff-based decision making.
    
    The model captures how decision-makers weight referrals based on trust:
    - If you trust a referrer, you use their recommendation even if another (untrusted) 
      referrer gives a better recommendation
    - If both referrers are trusted, use the worst (lowest quality) referral (conservative)
    - Network effect (G) IS the selected referral quality (no separate mapping)
    """
    
    # Referral quality levels (ordered from best to worst)
    # These ARE the network effect levels G
    REFERRAL_LEVELS = ['excellent', 'good', 'average', 'bad', 'very bad']
    NETWORK_LEVELS = REFERRAL_LEVELS  # Network effect G = referral quality
    
    def __init__(self, p=0.5, lambda_good=5.0, lambda_bad=2.0, 
                 referral_accuracy=0.8, referral_bias=0.1, B=10.0, b=2.0, w=5.0):
        """
        Initialize the referral-based hiring model.
        
        Parameters:
        -----------
        p : float
            Prior probability that a candidate is good (Pr(F=good))
        lambda_good : float
            Poisson rate parameter for paper count given candidate is good
        lambda_bad : float
            Poisson rate parameter for paper count given candidate is bad
        referral_accuracy : float
            Probability that a referral correctly reflects candidate quality (0-1)
            Higher = referrers are more accurate
        referral_bias : float
            Probability of positive bias (referrers inflate quality) (0-1)
        B : float
            Employer payoff from hiring a good candidate (typically B > b)
        b : float
            Employer payoff from hiring a bad candidate (typically B > b)
        w : float
            Applicant payoff if hired (w > 0)
        """
        self.p = p
        self.lambda_good = lambda_good
        self.lambda_bad = lambda_bad
        self.referral_accuracy = referral_accuracy
        self.referral_bias = referral_bias
        self.B = B
        self.b = b
        self.w = w
        
        # Validate parameters
        assert 0 <= p <= 1, "p must be between 0 and 1"
        assert lambda_good > 0, "lambda_good must be positive"
        assert lambda_bad > 0, "lambda_bad must be positive"
        assert 0 <= referral_accuracy <= 1, "referral_accuracy must be between 0 and 1"
        assert 0 <= referral_bias <= 1, "referral_bias must be between 0 and 1"
        assert B > b, "B (payoff from good candidate) should be greater than b (payoff from bad candidate)"
        assert w > 0, "w (applicant payoff) must be positive"
        
        # Define referral quality probabilities conditional on fit
        self._setup_referral_distributions()
    
    def _setup_referral_distributions(self):
        """Set up probability distributions for referral quality given candidate fit."""
        # For good candidates: higher probability of excellent/good referrals
        # For bad candidates: higher probability of bad/very bad referrals
        
        # Good candidates: skewed toward positive referrals
        self.referral_probs_good = {
            'excellent': 0.4 * self.referral_accuracy + 0.1 * (1 - self.referral_accuracy),
            'good': 0.3 * self.referral_accuracy + 0.2 * (1 - self.referral_accuracy),
            'average': 0.15 * self.referral_accuracy + 0.3 * (1 - self.referral_accuracy),
            'bad': 0.1 * self.referral_accuracy + 0.2 * (1 - self.referral_accuracy),
            'very bad': 0.05 * self.referral_accuracy + 0.2 * (1 - self.referral_accuracy)
        }
        
        # Bad candidates: skewed toward negative referrals
        self.referral_probs_bad = {
            'excellent': 0.05 * (1 - self.referral_accuracy) + 0.1 * self.referral_bias,
            'good': 0.1 * (1 - self.referral_accuracy) + 0.15 * self.referral_bias,
            'average': 0.15 * (1 - self.referral_accuracy) + 0.2 * self.referral_bias,
            'bad': 0.3 * self.referral_accuracy + 0.25 * (1 - self.referral_accuracy),
            'very bad': 0.4 * self.referral_accuracy + 0.3 * (1 - self.referral_accuracy)
        }
        
        # Normalize to ensure they sum to 1
        total_good = sum(self.referral_probs_good.values())
        total_bad = sum(self.referral_probs_bad.values())
        self.referral_probs_good = {k: v/total_good for k, v in self.referral_probs_good.items()}
        self.referral_probs_bad = {k: v/total_bad for k, v in self.referral_probs_bad.items()}
    
    def likelihood_referral(self, referral_quality, f_good=True):
        """
        Compute likelihood of referral quality given fit F.
        
        Parameters:
        -----------
        referral_quality : str
            Quality level: 'excellent', 'good', 'average', 'bad', 'very bad'
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(R=r | F=f)
        """
        if f_good:
            return self.referral_probs_good.get(referral_quality, 0.0)
        else:
            return self.referral_probs_bad.get(referral_quality, 0.0)
    
    def likelihood_papers(self, n, f_good=True):
        """
        Compute likelihood of paper count N given fit F using Poisson distribution.
        
        Parameters:
        -----------
        n : int
            Number of papers (1 to K)
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(N=n | F=f)
        """
        lambda_param = self.lambda_good if f_good else self.lambda_bad
        return poisson.pmf(n, lambda_param)
    
    def select_referral(self, r1_quality, r2_quality, trust_r1, trust_r2):
        """
        Select which referral to use based on trust and quality.
        
        Decision rule:
        - If trust only R1: use R1
        - If trust only R2: use R2
        - If trust both: use the lowest (worst) quality (conservative approach)
        - If trust neither: use the worst one
        
        Parameters:
        -----------
        r1_quality : str
            Referral quality from referrer 1
        r2_quality : str
            Referral quality from referrer 2
        trust_r1 : bool
            Whether referrer 1 is trusted
        trust_r2 : bool
            Whether referrer 2 is trusted
        
        Returns:
        --------
        str : Selected referral quality
        """
        # Quality ordering (higher index = better quality)
        quality_order = {q: i for i, q in enumerate(self.REFERRAL_LEVELS)}
        
        # Case 1: Trust only R1
        if trust_r1 and not trust_r2:
            return r1_quality
        
        # Case 2: Trust only R2
        if trust_r2 and not trust_r1:
            return r2_quality
        
        # Case 3: Trust both - use the lowest (worst) quality (conservative)
        if trust_r1 and trust_r2:
            r1_score = quality_order.get(r1_quality, 0)
            r2_score = quality_order.get(r2_quality, 0)
            # Lower score = worse quality, so return the one with lower score
            return r1_quality if r1_score <= r2_score else r2_quality
        
        # Case 4: Trust neither - use the worst one
        r1_score = quality_order.get(r1_quality, 0)
        r2_score = quality_order.get(r2_quality, 0)
        return r1_quality if r1_score <= r2_score else r2_quality
    
    def referral_to_network(self, referral_quality):
        """
        Convert referral quality to network effect level.
        Since referral quality IS the network effect, this just returns the quality.
        
        Parameters:
        -----------
        referral_quality : str
            Referral quality level
        
        Returns:
        --------
        str : Network effect level (same as referral quality)
        """
        return referral_quality
    
    def likelihood_network(self, network_level, f_good=True):
        """
        Compute likelihood of network effect G given fit F.
        
        Since network effect G = referral quality, this is just the referral likelihood.
        
        Parameters:
        -----------
        network_level : str
            Network effect level: 'excellent', 'good', 'average', 'bad', 'very bad'
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(G=g | F=f)
        """
        # Network effect IS the referral quality, so use referral likelihood directly
        return self.likelihood_referral(network_level, f_good)
    
    def joint_likelihood(self, network_level, n, f_good=True):
        """
        Compute joint likelihood Pr(G=g, N=n | F=f) using conditional independence.
        
        Parameters:
        -----------
        network_level : str
            Network effect level ('excellent', 'good', 'average', 'bad', 'very bad')
        n : int
            Number of papers
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(G=g, N=n | F=f)
        """
        return self.likelihood_network(network_level, f_good) * self.likelihood_papers(n, f_good)
    
    def posterior_probability(self, network_level, n):
        """
        Compute posterior probability Pr(F=good | G=g, N=n) using Bayes' theorem.
        
        Parameters:
        -----------
        network_level : str
            Network effect level ('excellent', 'good', 'average', 'bad', 'very bad')
        n : int
            Number of papers
        
        Returns:
        --------
        tuple : (Pr(F=good | G, N), Pr(F=bad | G, N))
        """
        # Compute likelihoods
        likelihood_good = self.joint_likelihood(network_level, n, f_good=True)
        likelihood_bad = self.joint_likelihood(network_level, n, f_good=False)
        
        # Compute marginal probability (normalizing constant)
        marginal = likelihood_good * self.p + likelihood_bad * (1 - self.p)
        
        # Avoid division by zero
        if marginal == 0:
            return (0.5, 0.5)
        
        # Compute posterior probabilities
        posterior_good = (likelihood_good * self.p) / marginal
        posterior_bad = (likelihood_bad * (1 - self.p)) / marginal
        
        return (posterior_good, posterior_bad)
    
    def expected_payoff(self, network_level, n):
        """
        Compute expected employer payoff from hiring given signals.
        
        Expected payoff = B · Pr(F=good | G, N) + b · Pr(F=bad | G, N)
                        = b + (B - b) · Pr(F=good | G, N)
        
        Parameters:
        -----------
        network_level : str
            Network effect level ('excellent', 'good', 'average', 'bad', 'very bad')
        n : int
            Number of papers
        
        Returns:
        --------
        float : Expected employer payoff from hiring
        """
        posterior_good, posterior_bad = self.posterior_probability(network_level, n)
        expected_payoff = self.B * posterior_good + self.b * posterior_bad
        return expected_payoff
    
    def should_hire_payoff_based(self, network_level, n, threshold=0.0):
        """
        Determine whether to hire based on expected payoff.
        
        Decision rule: Hire if expected payoff > threshold
        
        Parameters:
        -----------
        network_level : str
            Network effect level ('excellent', 'good', 'average', 'bad', 'very bad')
        n : int
            Number of papers
        threshold : float
            Minimum expected payoff to hire (default: 0.0)
        
        Returns:
        --------
        bool : True if should hire, False otherwise
        """
        expected_payoff = self.expected_payoff(network_level, n)
        return expected_payoff > threshold
    
    def should_hire_probability_based(self, network_level, n, threshold=0.5):
        """
        Determine whether to hire based on posterior probability.
        
        Decision rule: Hire if Pr(F=good | G, N) > threshold
        
        Parameters:
        -----------
        network_level : str
            Network effect level ('excellent', 'good', 'average', 'bad', 'very bad')
        n : int
            Number of papers
        threshold : float
            Minimum probability threshold to hire (default: 0.5)
        
        Returns:
        --------
        bool : True if should hire, False otherwise
        """
        posterior_good, _ = self.posterior_probability(network_level, n)
        return posterior_good > threshold
    
    def compute_probability_matrix(self, n_max=10):
        """
        Compute the probability matrix where:
        - Rows: G (excellent, good, average, bad, very bad)
        - Columns: N (1, 2, ..., n_max)
        - Values: Pr(F=good | G, N)
        
        Parameters:
        -----------
        n_max : int
            Maximum number of papers to consider
        
        Returns:
        --------
        pd.DataFrame : Matrix with probabilities
        """
        # Initialize matrix
        matrix = np.zeros((len(self.NETWORK_LEVELS), n_max))
        row_labels = [f'G={level.title()}' for level in self.NETWORK_LEVELS]
        col_labels = [f'N={i}' for i in range(1, n_max + 1)]
        
        # Fill matrix
        for g_idx, network_level in enumerate(self.NETWORK_LEVELS):
            for n in range(1, n_max + 1):
                posterior_good, _ = self.posterior_probability(network_level, n)
                matrix[g_idx, n - 1] = posterior_good
        
        # Create DataFrame
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        return df
    
    def visualize_matrix(self, matrix, save_path=None):
        """
        Visualize the probability matrix as a heatmap.
        
        Parameters:
        -----------
        matrix : pd.DataFrame
            Probability matrix from compute_probability_matrix
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, cbar_kws={'label': 'Pr(F=good | G, N)'})
        plt.title('Posterior Probability of Being Good Given Network Effect and Paper Count', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Number of Papers (N)', fontsize=12)
        plt.ylabel('Network Effect (G)', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def simulate_candidates_with_referrals(model, n_candidates=1000, n_max=10, 
                                       prob_trust_r1=0.5, prob_trust_r2=0.5):
    """
    Simulate candidates with referrals and compute their posterior probabilities.
    
    Parameters:
    -----------
    model : ReferralBasedHiringModel
        The referral-based hiring model instance
    n_candidates : int
        Number of candidates to simulate
    n_max : int
        Maximum number of papers
    prob_trust_r1 : float
        Probability that referrer 1 is trusted
    prob_trust_r2 : float
        Probability that referrer 2 is trusted
    
    Returns:
    --------
    pd.DataFrame : Simulated candidates with posterior probabilities
    """
    candidates = []
    
    for _ in range(n_candidates):
        # Sample true fit (F)
        is_good = np.random.binomial(1, model.p) == 1
        
        # Sample referral qualities given fit
        referral_probs = model.referral_probs_good if is_good else model.referral_probs_bad
        referral_levels = list(referral_probs.keys())
        referral_probs_list = list(referral_probs.values())
        
        r1_quality = np.random.choice(referral_levels, p=referral_probs_list)
        r2_quality = np.random.choice(referral_levels, p=referral_probs_list)
        
        # Sample trust in referrers
        trust_r1 = np.random.binomial(1, prob_trust_r1) == 1
        trust_r2 = np.random.binomial(1, prob_trust_r2) == 1
        
        # Select which referral to use
        selected_referral = model.select_referral(r1_quality, r2_quality, trust_r1, trust_r2)
        
        # Convert to network effect
        network_level = model.referral_to_network(selected_referral)
        
        # Sample paper count (N) given fit
        lambda_param = model.lambda_good if is_good else model.lambda_bad
        n_papers = np.random.poisson(lambda_param)
        n_papers = max(1, min(n_papers, n_max))
        
        # Compute posterior probability
        posterior_good, posterior_bad = model.posterior_probability(network_level, n_papers)
        
        # Compute expected payoff
        expected_payoff = model.expected_payoff(network_level, n_papers)
        
        # Decision based on probability (threshold = 0.5)
        decision_prob = model.should_hire_probability_based(network_level, n_papers, threshold=0.5)
        
        # Decision based on payoff (threshold = 0.0)
        decision_payoff = model.should_hire_payoff_based(network_level, n_papers, threshold=0.0)
        
        candidates.append({
            'True_Fit': 'Good' if is_good else 'Bad',
            'R1_Quality': r1_quality,
            'R2_Quality': r2_quality,
            'Trust_R1': 'Yes' if trust_r1 else 'No',
            'Trust_R2': 'Yes' if trust_r2 else 'No',
            'Selected_Referral': selected_referral,
            'Network_Effect': network_level,
            'N_Papers': n_papers,
            'Posterior_Good': posterior_good,
            'Posterior_Bad': posterior_bad,
            'Expected_Payoff': expected_payoff,
            'Decision_Prob_Based': 'Hire' if decision_prob else 'Reject',
            'Decision_Payoff_Based': 'Hire' if decision_payoff else 'Reject',
            'Predicted_Fit': 'Good' if posterior_good > 0.5 else 'Bad'  # Keep for backward compatibility
        })
    
    return pd.DataFrame(candidates)


def analyze_referral_selection(candidates_df):
    """
    Analyze how trust affects referral selection.
    
    Parameters:
    -----------
    candidates_df : pd.DataFrame
        DataFrame from simulate_candidates_with_referrals
    """
    print("\n" + "=" * 80)
    print("Referral Selection Analysis")
    print("=" * 80)
    
    # Cases where trust matters
    trust_r1_only = candidates_df[
        (candidates_df['Trust_R1'] == 'Yes') & (candidates_df['Trust_R2'] == 'No')
    ]
    trust_r2_only = candidates_df[
        (candidates_df['Trust_R1'] == 'No') & (candidates_df['Trust_R2'] == 'Yes')
    ]
    trust_both = candidates_df[
        (candidates_df['Trust_R1'] == 'Yes') & (candidates_df['Trust_R2'] == 'Yes')
    ]
    trust_neither = candidates_df[
        (candidates_df['Trust_R1'] == 'No') & (candidates_df['Trust_R2'] == 'No')
    ]
    
    print(f"\nTrust R1 only: {len(trust_r1_only)} candidates")
    print(f"  Average posterior (good): {trust_r1_only['Posterior_Good'].mean():.3f}")
    
    print(f"\nTrust R2 only: {len(trust_r2_only)} candidates")
    print(f"  Average posterior (good): {trust_r2_only['Posterior_Good'].mean():.3f}")
    
    print(f"\nTrust both: {len(trust_both)} candidates")
    print(f"  Average posterior (good): {trust_both['Posterior_Good'].mean():.3f}")
    
    print(f"\nTrust neither: {len(trust_neither)} candidates")
    print(f"  Average posterior (good): {trust_neither['Posterior_Good'].mean():.3f}")
    
    # Analyze cases where trust overrides better referral
    print("\n" + "-" * 80)
    print("Cases where trust overrides better referral:")
    print("-" * 80)
    
    # Quality ordering
    quality_order = {'excellent': 4, 'good': 3, 'average': 2, 'bad': 1, 'very bad': 0}
    
    trust_override_cases = []
    for idx, row in candidates_df.iterrows():
        r1_score = quality_order.get(row['R1_Quality'], 0)
        r2_score = quality_order.get(row['R2_Quality'], 0)
        trust_r1 = row['Trust_R1'] == 'Yes'
        trust_r2 = row['Trust_R2'] == 'Yes'
        
        # Case: Trust R1 but R2 is better
        if trust_r1 and not trust_r2 and r2_score > r1_score:
            trust_override_cases.append({
                'Case': 'Trust R1, but R2 better',
                'R1': row['R1_Quality'],
                'R2': row['R2_Quality'],
                'Selected': row['Selected_Referral'],
                'Posterior': row['Posterior_Good']
            })
        
        # Case: Trust R2 but R1 is better
        if trust_r2 and not trust_r1 and r1_score > r2_score:
            trust_override_cases.append({
                'Case': 'Trust R2, but R1 better',
                'R1': row['R1_Quality'],
                'R2': row['R2_Quality'],
                'Selected': row['Selected_Referral'],
                'Posterior': row['Posterior_Good']
            })
    
    if trust_override_cases:
        override_df = pd.DataFrame(trust_override_cases)
        print(f"\nFound {len(override_df)} cases where trust overrides better referral:")
        print(override_df.head(10).to_string(index=False))
        print(f"\nAverage posterior in override cases: {override_df['Posterior'].mean():.3f}")
    else:
        print("\nNo cases found where trust overrides better referral.")


def main(output_dir=None, p=0.5, lambda_good=5.0, lambda_bad=2.0,
         referral_accuracy=0.8, referral_bias=0.1, n_candidates=1000, n_max=10,
         prob_trust_r1=0.5, prob_trust_r2=0.5, B=10.0, b=2.0, w=5.0):
    """
    Main function to run the referral-based simulation and generate results.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where to save output files. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    p : float, optional
        Prior probability that a candidate is good (Pr(F=good)). Default: 0.5
    lambda_good : float, optional
        Poisson rate parameter for paper count given candidate is good. Default: 5.0
    lambda_bad : float, optional
        Poisson rate parameter for paper count given candidate is bad. Default: 2.0
    referral_accuracy : float, optional
        Probability that a referral correctly reflects candidate quality. Default: 0.8
    referral_bias : float, optional
        Probability of positive bias (referrers inflate quality). Default: 0.1
    n_candidates : int, optional
        Number of candidates to simulate. Default: 1000
    n_max : int, optional
        Maximum number of papers to consider. Default: 10
    prob_trust_r1 : float, optional
        Probability that referrer 1 is trusted. Default: 0.5
    prob_trust_r2 : float, optional
        Probability that referrer 2 is trusted. Default: 0.5
    B : float, optional
        Employer payoff from hiring a good candidate. Default: 10.0
    b : float, optional
        Employer payoff from hiring a bad candidate. Default: 2.0
    w : float, optional
        Applicant payoff if hired. Default: 5.0
    """
    # Initialize model
    model = ReferralBasedHiringModel(
        p=p,
        lambda_good=lambda_good,
        lambda_bad=lambda_bad,
        referral_accuracy=referral_accuracy,
        referral_bias=referral_bias,
        B=B,
        b=b,
        w=w
    )
    
    print("=" * 80)
    print("Referral-Based Bayesian Hiring Model")
    print("=" * 80)
    print(f"\nModel Parameters:")
    print(f"  Prior Pr(F=good) = {model.p}")
    print(f"  E[N | F=good] = {model.lambda_good}")
    print(f"  E[N | F=bad] = {model.lambda_bad}")
    print(f"  Referral accuracy = {model.referral_accuracy}")
    print(f"  Referral bias = {model.referral_bias}")
    print(f"\nPayoff Parameters:")
    print(f"  B (payoff from good candidate) = {model.B}")
    print(f"  b (payoff from bad candidate) = {model.b}")
    print(f"  w (applicant payoff if hired) = {model.w}")
    
    print("\nReferral Quality Distributions:")
    print("\n  Given F=good:")
    for quality, prob in model.referral_probs_good.items():
        print(f"    {quality:12s}: {prob:.3f}")
    print("\n  Given F=bad:")
    for quality, prob in model.referral_probs_bad.items():
        print(f"    {quality:12s}: {prob:.3f}")
    
    print("\n" + "=" * 80)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compute probability matrix
    prob_matrix = model.compute_probability_matrix(n_max=n_max)
    print("\nProbability Matrix: Pr(F=good | G, N)")
    print(prob_matrix)
    print("\n" + "=" * 80)
    
    # Save matrix to CSV
    matrix_csv_path = 'referral_probability_matrix.csv'
    if output_dir:
        matrix_csv_path = os.path.join(output_dir, matrix_csv_path)
    prob_matrix.to_csv(matrix_csv_path)
    print(f"\nProbability matrix saved to '{matrix_csv_path}'")
    
    # Visualize matrix
    print("\nGenerating visualization...")
    heatmap_path = 'referral_probability_heatmap.png'
    if output_dir:
        heatmap_path = os.path.join(output_dir, heatmap_path)
    model.visualize_matrix(prob_matrix, save_path=heatmap_path)
    print(f"Heatmap saved to '{heatmap_path}'")
    
    # Run simulation
    print("\n" + "=" * 80)
    print(f"Simulating {n_candidates} candidates with referrals...")
    candidates_df = simulate_candidates_with_referrals(
        model, 
        n_candidates=n_candidates, 
        n_max=n_max,
        prob_trust_r1=prob_trust_r1,
        prob_trust_r2=prob_trust_r2
    )
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(candidates_df[['True_Fit', 'Network_Effect', 'N_Papers', 
                          'Posterior_Good', 'Expected_Payoff',
                          'Decision_Prob_Based', 'Decision_Payoff_Based']].describe())
    
    # Accuracy analysis - probability-based
    predicted_fit_prob = candidates_df['Decision_Prob_Based'].map({'Hire': 'Good', 'Reject': 'Bad'})
    accuracy_prob = (candidates_df['True_Fit'] == predicted_fit_prob).mean()
    print(f"\nPrediction Accuracy (Probability-based): {accuracy_prob:.2%}")
    
    # Accuracy analysis - payoff-based
    predicted_fit_payoff = candidates_df['Decision_Payoff_Based'].map({'Hire': 'Good', 'Reject': 'Bad'})
    accuracy_payoff = (candidates_df['True_Fit'] == predicted_fit_payoff).mean()
    print(f"Prediction Accuracy (Payoff-based): {accuracy_payoff:.2%}")
    
    # Compare decisions
    decisions_match = (candidates_df['Decision_Prob_Based'] == candidates_df['Decision_Payoff_Based']).mean()
    print(f"Decision Agreement: {decisions_match:.2%}")
    
    # Expected payoff analysis
    hired_payoff = candidates_df[candidates_df['Decision_Payoff_Based'] == 'Hire']['Expected_Payoff'].mean()
    print(f"\nAverage Expected Payoff (Payoff-based hires): {hired_payoff:.2f}")
    
    # Compare expected payoffs
    prob_hires = candidates_df[candidates_df['Decision_Prob_Based'] == 'Hire']
    if len(prob_hires) > 0:
        avg_payoff_prob = prob_hires['Expected_Payoff'].mean()
        print(f"Average Expected Payoff (Probability-based hires): {avg_payoff_prob:.2f}")
    
    # Analyze referral selection
    analyze_referral_selection(candidates_df)
    
    # Save simulation results
    candidates_csv_path = 'referral_simulated_candidates.csv'
    if output_dir:
        candidates_csv_path = os.path.join(output_dir, candidates_csv_path)
    candidates_df.to_csv(candidates_csv_path, index=False)
    print(f"\nSimulated candidates saved to '{candidates_csv_path}'")
    
    print("\n" + "=" * 80)
    print("Simulation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run referral-based Bayesian hiring model simulation'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=None,
        help='Directory where to save output files (default: current directory)'
    )
    parser.add_argument(
        '--p',
        type=float,
        default=0.5,
        help='Prior probability that a candidate is good (default: 0.5)'
    )
    parser.add_argument(
        '--lambda-good',
        type=float,
        default=5.0,
        dest='lambda_good',
        help='Poisson rate for paper count given candidate is good (default: 5.0)'
    )
    parser.add_argument(
        '--lambda-bad',
        type=float,
        default=2.0,
        dest='lambda_bad',
        help='Poisson rate for paper count given candidate is bad (default: 2.0)'
    )
    parser.add_argument(
        '--referral-accuracy',
        type=float,
        default=0.8,
        dest='referral_accuracy',
        help='Probability that a referral correctly reflects candidate quality (default: 0.8)'
    )
    parser.add_argument(
        '--referral-bias',
        type=float,
        default=0.1,
        dest='referral_bias',
        help='Probability of positive bias (referrers inflate quality) (default: 0.1)'
    )
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=1000,
        dest='n_candidates',
        help='Number of candidates to simulate (default: 1000)'
    )
    parser.add_argument(
        '--n-max',
        type=int,
        default=10,
        dest='n_max',
        help='Maximum number of papers to consider (default: 10)'
    )
    parser.add_argument(
        '--prob-trust-r1',
        type=float,
        default=0.5,
        dest='prob_trust_r1',
        help='Probability that referrer 1 is trusted (default: 0.5)'
    )
    parser.add_argument(
        '--prob-trust-r2',
        type=float,
        default=0.5,
        dest='prob_trust_r2',
        help='Probability that referrer 2 is trusted (default: 0.5)'
    )
    parser.add_argument(
        '--B',
        type=float,
        default=10.0,
        help='Employer payoff from hiring a good candidate (default: 10.0)'
    )
    parser.add_argument(
        '--b',
        type=float,
        default=-2.0,
        help='Employer payoff from hiring a bad candidate (default: 2.0)'
    )
    parser.add_argument(
        '--w',
        type=float,
        default=5.0,
        help='Applicant payoff if hired (default: 5.0)'
    )
    
    args = parser.parse_args()
    main(
        output_dir=args.dir,
        p=args.p,
        lambda_good=args.lambda_good,
        lambda_bad=args.lambda_bad,
        referral_accuracy=args.referral_accuracy,
        referral_bias=args.referral_bias,
        n_candidates=args.n_candidates,
        n_max=args.n_max,
        prob_trust_r1=args.prob_trust_r1,
        prob_trust_r2=args.prob_trust_r2,
        B=args.B,
        b=args.b,
        w=args.w
    )
