import numpy as np
import logging

logger = logging.getLogger(__name__)

class StakingEngine:
    def __init__(self, fractional_kelly=0.25):
        """
        fractional_kelly: Scalar to reduce volatility. 
        0.25 (Quarter-Kelly) is standard for 'Safety for Accuracy'.
        """
        self.fraction = fractional_kelly

    def calculate_kelly_stake(self, model_prob, decimal_odds):
        """
        k = (p*b - q) / b
        where:
        p = model probability
        q = (1-p)
        b = net odds (decimal_odds - 1)
        """
        b = decimal_odds - 1
        q = 1 - model_prob
        
        if b <= 0:
            return 0.0
            
        edge = (model_prob * b - q) / b
        
        # Safety Pruning: Never stake on negative edge or low certainty
        if edge <= 0:
            return 0.0
            
        # Apply fractional multiplier for billionaire-scale safety
        final_stake = edge * self.fraction
        
        # Max stake cap (e.g. 5% of bankroll per fixture)
        return min(final_stake, 0.05)

    def generate_trading_signal(self, prediction_dict, market_odds_dict):
        """
        Compares model intelligence with market price to generate a verified signal.
        """
        signals = {}
        for outcome in ['home', 'draw', 'away']:
            prob = prediction_dict['win_draw_loss'][outcome]
            odds = market_odds_dict.get(outcome, 1.0)
            stake = self.calculate_kelly_stake(prob, odds)
            
            if stake > 0.01: # Signal threshold
                signals[outcome] = {
                    "stake_pct": round(stake * 100, 2),
                    "confidence": "High" if stake > 0.03 else "Medium"
                }
        return signals
