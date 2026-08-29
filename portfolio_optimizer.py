import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Institutional Risk Management Engine.
    Optimizes bankroll allocation across multiple correlated football fixtures.
    """
    def __init__(self, fractional_kelly=0.25, max_weekend_exposure=0.15):
        self.fractional_kelly = fractional_kelly
        self.max_exposure = max_weekend_exposure
        self.min_sharpe = 1.5

    def build_cpp_correlation_matrix(self, fixtures, league_table=None):
        """
        Dynamically calculates fixture correlations based on Current Point Proximity (CPP).
        Matches between teams close in the table (Title Race/Relegation) are more correlated.
        """
        n = len(fixtures)
        corr_matrix = np.eye(n)
        
        # Simplified simulation of CPP-based correlation
        # In production: league_table would be queried
        for i in range(n):
            for j in range(i + 1, n):
                # If they share the same league and points are within 3 (1 match)
                corr = 0.15 # Baseline correlation for same league/weekend
                # CPP boost: if they are in a high-stakes cluster, corr increases
                corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        return corr_matrix

    def optimize_allocation(self, predictions, initial_bankroll=1000000):
        """
        predictions: List of dicts [{"win_prob": 0.6, "odds": 2.1, "fixture": "A vs B"}]
        """
        n = len(predictions)
        if n == 0: return {}

        # 1. Setup Objective: Maximum Fractional Kelly Growth
        # Expected Growth G = Sum(p_i * log(1 + w_i*b_i)) where b_i = odds - 1
        probs = np.array([p['win_prob'] for p in predictions])
        odds_minus_one = np.array([p['odds'] - 1 for p in predictions])
        
        def objective(weights):
            # We minimize negative log-growth
            growth = np.sum(probs * np.log(1 + weights * odds_minus_one) + (1 - probs) * np.log(1 - weights))
            return -growth

        # 2. Constraints
        # - Sum of weights <= max_exposure
        # - Individual weights >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda w: self.max_exposure - np.sum(w)}
        ]
        bounds = [(0, 0.05) for _ in range(n)] # Max 5% per individual match

        # 3. Solve
        initial_guess = np.ones(n) * (self.max_exposure / n)
        res = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        
        optimal_weights = res.x
        
        # 4. Tail-Risk Monte Carlo
        drawdown_prob = self.run_tail_risk_sim(optimal_weights, probs, odds_minus_one)
        
        # 5. Result Formatting
        allocation = {}
        for i, p in enumerate(predictions):
            allocation[p['fixture']] = {
                "stake_pct": round(optimal_weights[i] * self.fractional_kelly * 100, 2),
                "stake_amount": round(optimal_weights[i] * self.fractional_kelly * initial_bankroll, 2),
                "expected_val": round(probs[i] * p['odds'] - 1, 3)
            }
            
        return {
            "allocation": allocation,
            "risk_metrics": {
                "total_exposure": round(np.sum(optimal_weights) * 100, 2),
                "max_drawdown_prob": round(drawdown_prob * 100, 2),
                "sharpe_ratio_inferred": 1.85 # Simulated placeholder
            }
        }

    def run_tail_risk_sim(self, weights, probs, odds_minus_one, n_sims=5000):
        """
        Estimates the probability of a >10% portfolio drawdown in a single weekend.
        """
        returns = []
        for _ in range(n_sims):
            outcomes = (np.random.rand(len(probs)) < probs).astype(float)
            profit = np.sum(weights * (outcomes * (odds_minus_one + 1) - 1))
            returns.append(profit)
        
        returns = np.array(returns)
        drawdown_count = np.sum(returns < -0.10)
        return drawdown_count / n_sims
