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
                 referral_accuracy=0.8, referral_bias=0.1):
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
        """
        self.p = p
        self.lambda_good = lambda_good
        self.lambda_bad = lambda_bad
        self.referral_accuracy = referral_accuracy
        self.referral_bias = referral_bias
        
        # Validate parameters
        assert 0 <= p <= 1, "p must be between 0 and 1"
        assert lambda_good > 0, "lambda_good must be positive"
        assert lambda_bad > 0, "lambda_bad must be positive"
        assert 0 <= referral_accuracy <= 1, "referral_accuracy must be between 0 and 1"
        assert 0 <= referral_bias <= 1, "referral_bias must be between 0 and 1"
        
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
            'Predicted_Fit': 'Good' if posterior_good > 0.5 else 'Bad'
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


def main(output_dir=None):
    """
    Main function to run the referral-based simulation and generate results.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where to save output files. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    # Initialize model
    model = ReferralBasedHiringModel(
        p=0.5,                    # Prior: 50% chance of being good
        lambda_good=5.0,          # Good candidates average 5 papers
        lambda_bad=2.0,           # Bad candidates average 2 papers
        referral_accuracy=0.8,    # 80% accuracy in referrals
        referral_bias=0.1         # 10% positive bias
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
    prob_matrix = model.compute_probability_matrix(n_max=10)
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
    print("Simulating 1000 candidates with referrals...")
    candidates_df = simulate_candidates_with_referrals(
        model, 
        n_candidates=1000, 
        n_max=10,
        prob_trust_r1=0.5,
        prob_trust_r2=0.5
    )
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(candidates_df[['True_Fit', 'Network_Effect', 'N_Papers', 
                          'Posterior_Good', 'Predicted_Fit']].describe())
    
    # Accuracy analysis
    accuracy = (candidates_df['True_Fit'] == candidates_df['Predicted_Fit']).mean()
    print(f"\nPrediction Accuracy: {accuracy:.2%}")
    
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
    
    args = parser.parse_args()
    main(output_dir=args.dir)
