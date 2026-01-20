"""
Bayesian Hiring Model Simulation
Models the posterior probability of candidate fit (F) given network effect (G) and paper count (N).
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import os


class BayesianHiringModel:
    """
    Bayesian model for evaluating candidate fit based on network effect and publication count.
    """
    
    def __init__(self, p=0.5, q_good=0.7, q_bad=0.3, lambda_good=5.0, lambda_bad=2.0):
        """
        Initialize the Bayesian hiring model.
        
        Parameters:
        -----------
        p : float
            Prior probability that a candidate is good (Pr(F=good))
        q_good : float
            Probability of having network effect given candidate is good (Pr(G=yes | F=good))
        q_bad : float
            Probability of having network effect given candidate is bad (Pr(G=yes | F=bad))
        lambda_good : float
            Poisson rate parameter for paper count given candidate is good
        lambda_bad : float
            Poisson rate parameter for paper count given candidate is bad
        """
        self.p = p
        self.q_good = q_good
        self.q_bad = q_bad
        self.lambda_good = lambda_good
        self.lambda_bad = lambda_bad
        
        # Validate parameters
        assert 0 <= p <= 1, "p must be between 0 and 1"
        assert 0 <= q_good <= 1, "q_good must be between 0 and 1"
        assert 0 <= q_bad <= 1, "q_bad must be between 0 and 1"
        assert q_good > q_bad, "q_good should be greater than q_bad"
        assert lambda_good > 0, "lambda_good must be positive"
        assert lambda_bad > 0, "lambda_bad must be positive"
    
    def likelihood_network(self, g, f_good=True):
        """
        Compute likelihood of network effect G given fit F.
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(G=g | F=f)
        """
        if f_good:
            return self.q_good if g else (1 - self.q_good)
        else:
            return self.q_bad if g else (1 - self.q_bad)
    
    def likelihood_papers(self, n, f_good=True):
        """
        Compute likelihood of paper count N given fit F using Poisson distribution.
        
        Parameters:
        -----------
        n : int
            Number of papers (1 to 10)
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(N=n | F=f)
        """
        lambda_param = self.lambda_good if f_good else self.lambda_bad
        # Poisson PMF: e^(-λ) * λ^n / n!
        return poisson.pmf(n, lambda_param)
    
    def joint_likelihood(self, g, n, f_good=True):
        """
        Compute joint likelihood Pr(G=g, N=n | F=f) using conditional independence.
        
        Under conditional independence: Pr(G, N | F) = Pr(G | F) * Pr(N | F)
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        n : int
            Number of papers
        f_good : bool
            Whether candidate is good (True) or bad (False)
        
        Returns:
        --------
        float : Pr(G=g, N=n | F=f)
        """
        return self.likelihood_network(g, f_good) * self.likelihood_papers(n, f_good)
    
    def posterior_probability(self, g, n):
        """
        Compute posterior probability Pr(F=good | G=g, N=n) using Bayes' theorem.
        
        Bayes' rule:
        Pr(F=good | G, N) = Pr(G, N | F=good) * Pr(F=good) / Pr(G, N)
        
        where Pr(G, N) = Pr(G, N | F=good) * Pr(F=good) + Pr(G, N | F=bad) * Pr(F=bad)
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        n : int
            Number of papers
        
        Returns:
        --------
        tuple : (Pr(F=good | G, N), Pr(F=bad | G, N))
        """
        # Compute likelihoods
        likelihood_good = self.joint_likelihood(g, n, f_good=True)
        likelihood_bad = self.joint_likelihood(g, n, f_good=False)
        
        # Compute marginal probability (normalizing constant)
        marginal = likelihood_good * self.p + likelihood_bad * (1 - self.p)
        
        # Avoid division by zero
        if marginal == 0:
            return (0.5, 0.5)  # Default to uniform if no information
        
        # Compute posterior probabilities
        posterior_good = (likelihood_good * self.p) / marginal
        posterior_bad = (likelihood_bad * (1 - self.p)) / marginal
        
        return (posterior_good, posterior_bad)
    
    def compute_probability_matrix(self, n_max=10):
        """
        Compute the probability matrix where:
        - Rows: G (yes, no)
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
        matrix = np.zeros((2, n_max))
        row_labels = ['Network: Yes', 'Network: No']
        col_labels = [f'N={i}' for i in range(1, n_max + 1)]
        
        # Fill matrix
        for g_idx, g in enumerate([True, False]):  # True = yes, False = no
            for n in range(1, n_max + 1):
                posterior_good, _ = self.posterior_probability(g, n)
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


def simulate_candidates(model, n_candidates=1000, n_max=10):
    """
    Simulate candidates and compute their posterior probabilities.
    
    Parameters:
    -----------
    model : BayesianHiringModel
        The Bayesian hiring model instance
    n_candidates : int
        Number of candidates to simulate
    n_max : int
        Maximum number of papers
    
    Returns:
    --------
    pd.DataFrame : Simulated candidates with posterior probabilities
    """
    candidates = []
    
    for _ in range(n_candidates):
        # Sample true fit (F)
        is_good = np.random.binomial(1, model.p) == 1
        
        # Sample network effect (G) given fit
        if is_good:
            has_network = np.random.binomial(1, model.q_good) == 1
        else:
            has_network = np.random.binomial(1, model.q_bad) == 1
        
        # Sample paper count (N) given fit
        lambda_param = model.lambda_good if is_good else model.lambda_bad
        n_papers = np.random.poisson(lambda_param)
        # Clip to valid range [1, n_max]
        n_papers = max(1, min(n_papers, n_max))
        
        # Compute posterior probability
        posterior_good, posterior_bad = model.posterior_probability(has_network, n_papers)
        
        candidates.append({
            'True_Fit': 'Good' if is_good else 'Bad',
            'Network_Effect': 'Yes' if has_network else 'No',
            'N_Papers': n_papers,
            'Posterior_Good': posterior_good,
            'Posterior_Bad': posterior_bad,
            'Predicted_Fit': 'Good' if posterior_good > 0.5 else 'Bad'
        })
    
    return pd.DataFrame(candidates)


def main(output_dir=None):
    """
    Main function to run the simulation and generate results.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where to save output files. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    # Initialize model with default parameters
    # You can adjust these parameters based on your domain knowledge
    model = BayesianHiringModel(
        p=0.5,           # Prior: 50% chance of being good
        q_good=0.7,      # 70% of good candidates have network effect
        q_bad=0.3,       # 30% of bad candidates have network effect
        lambda_good=5.0, # Good candidates average 5 papers
        lambda_bad=2.0   # Bad candidates average 2 papers
    )
    
    print("=" * 60)
    print("Bayesian Hiring Model - Probability Matrix")
    print("=" * 60)
    print(f"\nModel Parameters:")
    print(f"  Prior Pr(F=good) = {model.p}")
    print(f"  Pr(G=yes | F=good) = {model.q_good}")
    print(f"  Pr(G=yes | F=bad) = {model.q_bad}")
    print(f"  E[N | F=good] = {model.lambda_good}")
    print(f"  E[N | F=bad] = {model.lambda_bad}")
    print("\n" + "=" * 60)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compute probability matrix
    prob_matrix = model.compute_probability_matrix(n_max=10)
    print("\nProbability Matrix: Pr(F=good | G, N)")
    print(prob_matrix)
    print("\n" + "=" * 60)
    
    # Save matrix to CSV
    matrix_csv_path = 'probability_matrix.csv'
    if output_dir:
        matrix_csv_path = os.path.join(output_dir, matrix_csv_path)
    prob_matrix.to_csv(matrix_csv_path)
    print(f"\nProbability matrix saved to '{matrix_csv_path}'")
    
    # Visualize matrix
    print("\nGenerating visualization...")
    heatmap_path = 'probability_heatmap.png'
    if output_dir:
        heatmap_path = os.path.join(output_dir, heatmap_path)
    model.visualize_matrix(prob_matrix, save_path=heatmap_path)
    print(f"Heatmap saved to '{heatmap_path}'")
    
    # Run simulation
    print("\n" + "=" * 60)
    print("Simulating 1000 candidates...")
    candidates_df = simulate_candidates(model, n_candidates=1000, n_max=10)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(candidates_df[['True_Fit', 'Network_Effect', 'N_Papers', 
                         'Posterior_Good', 'Predicted_Fit']].describe())
    
    # Accuracy analysis
    accuracy = (candidates_df['True_Fit'] == candidates_df['Predicted_Fit']).mean()
    print(f"\nPrediction Accuracy: {accuracy:.2%}")
    
    # Save simulation results
    candidates_csv_path = 'simulated_candidates.csv'
    if output_dir:
        candidates_csv_path = os.path.join(output_dir, candidates_csv_path)
    candidates_df.to_csv(candidates_csv_path, index=False)
    print(f"\nSimulated candidates saved to '{candidates_csv_path}'")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Bayesian hiring model simulation'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=None,
        help='Directory where to save output files (default: current directory)'
    )
    
    args = parser.parse_args()
    main(output_dir=args.dir)
