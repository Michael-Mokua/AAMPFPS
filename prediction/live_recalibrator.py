import logging
import numpy as np

logger = logging.getLogger(__name__)

class LiveRecalibrator:
    """
    Handles real-time probability shifts based on live match events.
    Inputs: current_probs, event_stream, statistics.
    """
    def __init__(self):
        self.momentum_decay = 0.95 # Momentum fades over time
        self.red_card_tax = 0.15   # 15% reduction in win probability for affected team

    def calculate_momentum_index(self, window_events):
        """
        Aggregates recent shots, corners, and dangerous attacks into a single index.
        """
        # score = w1*shots + w2*corners + w3*dangerous_attacks
        weights = {"shot": 1.5, "corner": 0.5, "dangerous_attack": 0.2}
        score = sum(weights.get(e['type'], 0) for e in window_events)
        return min(score / 5.0, 1.0) # Normalized 0-1

    def recalibrate(self, base_probs, match_state):
        """
        base_probs: np.array([home, draw, away])
        match_state: dict containing score, minute, cards, and momentum_events.
        """
        new_probs = base_probs.copy()
        
        # 1. Handle Red Cards
        if match_state.get('home_red_cards', 0) > 0:
            new_probs[0] -= self.red_card_tax
            new_probs[2] += self.red_card_tax / 2
            new_probs[1] += self.red_card_tax / 2
            
        if match_state.get('away_red_cards', 0) > 0:
            new_probs[2] -= self.red_card_tax
            new_probs[0] += self.red_card_tax / 2
            new_probs[1] += self.red_card_tax / 2

        # 2. Handle Momentum (Attacking Pressure)
        momentum = self.calculate_momentum_index(match_state.get('recent_events', []))
        momentum_team = match_state.get('momentum_team', 'home')
        
        if momentum > 0.4:
            shift = (momentum - 0.4) * 0.1 # Max 6% shift
            if momentum_team == 'home':
                new_probs[0] += shift
                new_probs[2] -= shift
            else:
                new_probs[2] += shift
                new_probs[0] -= shift
        
        # 3. Handle Score-Line Recalibration (Simple Bayesian)
        # In a real app, we'd use a Live Poisson surface here.
        # For this module, we maintain consistency.
        
        # Normalize
        new_probs = np.clip(new_probs, 0.01, 0.98)
        return new_probs / new_probs.sum()
