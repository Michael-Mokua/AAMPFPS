import logging
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Institutional-grade source credibility map
SOURCE_CREDIBILITY = {
    "BBC Sport": 0.95,
    "The Athletic": 0.98,
    "Fabrizio Romano": 0.92,
    "Sky Sports": 0.88,
    "Transfermarkt": 0.90,
    "FBref": 0.95,
    "Daily Mail": 0.65,
    "Sports Forum X": 0.40,
    "Twitter/X Leak": 0.30
}

class IntelligenceFilter:
    def __init__(self, target_accuracy_bias=0.85):
        """
        target_accuracy_bias: Higher means 'Safety for Accuracy' (Requested).
        """
        self.bias = target_accuracy_bias

    def deduplicate_news(self, news_list):
        """
        Uses similarity filtering to collapse redundant headlines into a single signal.
        """
        unique_signals = []
        for news in news_list:
            is_duplicate = False
            for signal in unique_signals:
                if SequenceMatcher(None, news['title'], signal['title']).ratio() > 0.85:
                    # Keep the one with higher source credibility
                    if SOURCE_CREDIBILITY.get(news['source'], 0.5) > SOURCE_CREDIBILITY.get(signal['source'], 0.5):
                        signal['title'] = news['title']
                        signal['source'] = news['source']
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_signals.append(news)
        return unique_signals

    def score_qualitative_signal(self, text, source):
        """
        Calculates a weighted confidence score based on source prestige and accuracy bias.
        """
        base_score = SOURCE_CREDIBILITY.get(source, 0.5)
        # Apply the 'Safety' bias: prune low-confidence signals entirely if bias is high
        if base_score < (1.0 - self.bias):
            logger.warning(f"Safety Filter: Pruning low-credibility signal from {source}")
            return 0.0
            
        return base_score

    def reconcile_market_outliers(self, model_prob, market_prob):
        """
        Implements the 'Safety for Accuracy' rule when stats/market diverge.
        """
        divergence = abs(model_prob - market_prob)
        if divergence > 0.15: # 15% discrepancy
             logger.info(f"Anomaly Detected: {divergence:.2f} divergence. Reconciling towards safety...")
             # Lean towards the more conservative (closer to 0.33) probability to avoid traps
             return (model_prob + market_prob + 0.33) / 3
        return (model_prob + market_prob) / 2
