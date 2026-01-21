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
    Supports both probability-based and payoff-based decision making.
    """
    
    def __init__(self, p=0.5, q_good=0.7, q_bad=0.3, lambda_good=5.0, lambda_bad=2.0,
                 B=10.0, b=-2.0, w=5.0):
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
        B : float
            Employer payoff from hiring a good candidate (typically B > b)
        b : float
            Employer payoff from hiring a bad candidate (typically B > b)
        w : float
            Applicant payoff if hired (w > 0)
        """
        self.p = p
        self.q_good = q_good
        self.q_bad = q_bad
        self.lambda_good = lambda_good
        self.lambda_bad = lambda_bad
        self.B = B
        self.b = b
        self.w = w
        
        # Validate parameters
        assert 0 <= p <= 1, "p must be between 0 and 1"
        assert 0 <= q_good <= 1, "q_good must be between 0 and 1"
        assert 0 <= q_bad <= 1, "q_bad must be between 0 and 1"
        assert q_good > q_bad, "q_good should be greater than q_bad"
        assert lambda_good > 0, "lambda_good must be positive"
        assert lambda_bad > 0, "lambda_bad must be positive"
        assert B > b, "B (payoff from good candidate) should be greater than b (payoff from bad candidate)"
        assert w > 0, "w (applicant payoff) must be positive"
    
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
    
    def expected_payoff(self, g, n):
        """
        Compute expected employer payoff from hiring given signals.
        
        Expected payoff = B · Pr(F=good | G, N) + b · Pr(F=bad | G, N)
                        = b + (B - b) · Pr(F=good | G, N)
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        n : int
            Number of papers
        
        Returns:
        --------
        float : Expected employer payoff from hiring
        """
        posterior_good, posterior_bad = self.posterior_probability(g, n)
        expected_payoff = self.B * posterior_good + self.b * posterior_bad
        return expected_payoff
    
    def should_hire_payoff_based(self, g, n, threshold=0.0):
        """
        Determine whether to hire based on expected payoff.
        
        Decision rule: Hire if expected payoff > threshold
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        n : int
            Number of papers
        threshold : float
            Minimum expected payoff to hire (default: 0.0)
        
        Returns:
        --------
        bool : True if should hire, False otherwise
        """
        expected_payoff = self.expected_payoff(g, n)
        return expected_payoff > threshold
    
    def should_hire_probability_based(self, g, n, threshold=0.5):
        """
        Determine whether to hire based on posterior probability.
        
        Decision rule: Hire if Pr(F=good | G, N) > threshold
        
        Parameters:
        -----------
        g : bool
            Network effect (True=yes, False=no)
        n : int
            Number of papers
        threshold : float
            Minimum probability threshold to hire (default: 0.5)
        
        Returns:
        --------
        bool : True if should hire, False otherwise
        """
        posterior_good, _ = self.posterior_probability(g, n)
        return posterior_good > threshold
    
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
        
        # Compute expected payoff
        expected_payoff = model.expected_payoff(has_network, n_papers)
        
        # Decision based on probability (threshold = 0.5)
        decision_prob = model.should_hire_probability_based(has_network, n_papers, threshold=0.5)
        
        # Decision based on payoff (threshold = 0.0)
        decision_payoff = model.should_hire_payoff_based(has_network, n_papers, threshold=0.0)
        
        candidates.append({
            'True_Fit': 'Good' if is_good else 'Bad',
            'Network_Effect': 'Yes' if has_network else 'No',
            'N_Papers': n_papers,
            'Posterior_Good': posterior_good,
            'Posterior_Bad': posterior_bad,
            'Expected_Payoff': expected_payoff,
            'Decision_Prob_Based': 'Hire' if decision_prob else 'Reject',
            'Decision_Payoff_Based': 'Hire' if decision_payoff else 'Reject',
            'Predicted_Fit': 'Good' if posterior_good > 0.5 else 'Bad'  # Keep for backward compatibility
        })
    
    return pd.DataFrame(candidates)


def main(output_dir=None, p=0.5, q_good=0.7, q_bad=0.3, lambda_good=5.0, 
         lambda_bad=2.0, n_candidates=1000, n_max=10, B=10.0, b=-2.0, w=5.0):
    """
    Main function to run the simulation and generate results.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where to save output files. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    p : float, optional
        Prior probability that a candidate is good (Pr(F=good)). Default: 0.5
    q_good : float, optional
        Probability of having network effect given candidate is good (Pr(G=yes | F=good)). Default: 0.7
    q_bad : float, optional
        Probability of having network effect given candidate is bad (Pr(G=yes | F=bad)). Default: 0.3
    lambda_good : float, optional
        Poisson rate parameter for paper count given candidate is good. Default: 5.0
    lambda_bad : float, optional
        Poisson rate parameter for paper count given candidate is bad. Default: 2.0
    n_candidates : int, optional
        Number of candidates to simulate. Default: 1000
    n_max : int, optional
        Maximum number of papers to consider. Default: 10
    B : float, optional
        Employer payoff from hiring a good candidate. Default: 10.0
    b : float, optional
        Employer payoff from hiring a bad candidate. Default: 2.0
    w : float, optional
        Applicant payoff if hired. Default: 5.0
    """
    # Initialize model with parameters
    model = BayesianHiringModel(
        p=p,
        q_good=q_good,
        q_bad=q_bad,
        lambda_good=lambda_good,
        lambda_bad=lambda_bad,
        B=B,
        b=b,
        w=w
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
    print(f"\nPayoff Parameters:")
    print(f"  B (payoff from good candidate) = {model.B}")
    print(f"  b (payoff from bad candidate) = {model.b}")
    print(f"  w (applicant payoff if hired) = {model.w}")
    print("\n" + "=" * 60)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compute probability matrix
    prob_matrix = model.compute_probability_matrix(n_max=n_max)
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
    heatmap_path = 'chaperone_probability_heatmap.png'
    if output_dir:
        heatmap_path = os.path.join(output_dir, heatmap_path)
    model.visualize_matrix(prob_matrix, save_path=heatmap_path)
    print(f"Heatmap saved to '{heatmap_path}'")
    
    # Run simulation
    print("\n" + "=" * 60)
    print(f"Simulating {n_candidates} candidates...")
    candidates_df = simulate_candidates(model, n_candidates=n_candidates, n_max=n_max)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(candidates_df[['True_Fit', 'Network_Effect', 'N_Papers', 
                         'Posterior_Good', 'Expected_Payoff', 
                         'Decision_Prob_Based', 'Decision_Payoff_Based']].describe())
    
    # Accuracy analysis - probability-based
    # We predict "Good" if we hire, "Bad" if we reject
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
    
    # Save simulation results
    candidates_csv_path = 'chaperone_simulated_candidates.csv'
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
    parser.add_argument(
        '--p',
        type=float,
        default=0.5,
        help='Prior probability that a candidate is good (default: 0.5)'
    )
    parser.add_argument(
        '--q-good',
        type=float,
        default=0.7,
        dest='q_good',
        help='Probability of network effect given candidate is good (default: 0.7)'
    )
    parser.add_argument(
        '--q-bad',
        type=float,
        default=0.3,
        dest='q_bad',
        help='Probability of network effect given candidate is bad (default: 0.3)'
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
        q_good=args.q_good,
        q_bad=args.q_bad,
        lambda_good=args.lambda_good,
        lambda_bad=args.lambda_bad,
        n_candidates=args.n_candidates,
        n_max=args.n_max,
        B=args.B,
        b=args.b,
        w=args.w
    )
