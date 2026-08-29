import numpy as np
import logging

logger = logging.getLogger(__name__)

class AdaptiveStacker:
    def __init__(self):
        # Weights for the different model paradigms
        self.weights = {
            "poisson_mc": 0.35,  # Monte Carlo converted Poisson probs
            "lstm": 0.30,        # Temporal sequence outputs
            "xgboost": 0.35      # Contextual boosting outputs
        }
        
    def bayesian_update(self, prior_probs: np.ndarray, rumor_sentiment: float, manager_bounce: bool) -> np.ndarray:
        """
        Adjusts stacked probabilities based on late-breaking qualitative proxies (e.g. news).
        rumor_sentiment from NLP engine (-1 to 1).
        """
        # This is a conceptual implementation of the specified Bayesian updating layer
        posterior = prior_probs.copy()
        
        # If sentiment is highly negative towards home team, slightly shift prob distribution
        if rumor_sentiment < -0.5:
            posterior[0] *= 0.95 # reduce home win
            posterior[2] *= 1.05 # increase away win
            
        if manager_bounce:
            posterior[0] *= 1.10
        
        # Normalize
        posterior = posterior / np.sum(posterior)
        return posterior

    def blend(self, poisson_probs, lstm_probs, xgb_probs):
        """
        poisson_probs, lstm_probs, xgb_probs: Arrays representing [P(Home), P(Draw), P(Away)]
        """
        stacked = (
            poisson_probs * self.weights["poisson_mc"] +
            lstm_probs * self.weights["lstm"] +
            xgb_probs  * self.weights["xgboost"]
        )
        return stacked
