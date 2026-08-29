import numpy as np
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    def __init__(self, config=None):
        self.config = config or {}
        self.num_simulations = self.config.get('monte_carlo', {}).get('num_simulations', 10000)

    def simulate_match(self, home_lambda, away_lambda, corner_lambda=10.5, card_lambda=3.8):
        """
        Simulates 10,000 matches using Poisson distributions for goals, corners, and cards.
        """
        logger.debug(f"Running {self.num_simulations} MC simulations...")
        
        sim_home_goals = np.random.poisson(home_lambda, self.num_simulations)
        sim_away_goals = np.random.poisson(away_lambda, self.num_simulations)
        sim_corners = np.random.poisson(corner_lambda, self.num_simulations)
        sim_cards = np.random.poisson(card_lambda, self.num_simulations)
        
        # Calculate W/D/L
        home_wins = np.sum(sim_home_goals > sim_away_goals)
        draws = np.sum(sim_home_goals == sim_away_goals)
        away_wins = np.sum(sim_home_goals < sim_away_goals)
        
        total_goals = sim_home_goals + sim_away_goals
        
        results = {
            "probabilities": {
                "home": home_wins / self.num_simulations,
                "draw": draws / self.num_simulations,
                "away": away_wins / self.num_simulations
            },
            "goal_lines": {
                "over_0_5": np.sum(total_goals > 0.5) / self.num_simulations,
                "over_1_5": np.sum(total_goals > 1.5) / self.num_simulations,
                "over_2_5": np.sum(total_goals > 2.5) / self.num_simulations,
                "over_3_5": np.sum(total_goals > 3.5) / self.num_simulations
            },
            "corners": {
                "expected": float(np.mean(sim_corners)),
                "over_8_5": np.sum(sim_corners > 8.5) / self.num_simulations,
                "over_10_5": np.sum(sim_corners > 10.5) / self.num_simulations
            },
            "cards": {
                "expected": float(np.mean(sim_cards)),
                "over_3_5": np.sum(sim_cards > 3.5) / self.num_simulations,
                "over_4_5": np.sum(sim_cards > 4.5) / self.num_simulations
            },
            "expected_exact": (np.round(np.mean(sim_home_goals), 1), np.round(np.mean(sim_away_goals), 1))
        }
        return results
