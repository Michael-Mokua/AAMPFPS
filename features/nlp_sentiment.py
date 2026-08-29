import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class NLPSentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_news_headline(self, text: str) -> float:
        """
        Analyzes a single piece of text (transfer rumour, press conference quote)
        Returns the compound polarity score (-1 to 1)
        """
        if not isinstance(text, str):
            return 0.0
            
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
        
    def process_team_sentiment(self, team_name: str, recent_news_list: list) -> dict:
        """
        Aggregates sentiments for a specific team over the recent news cycle.
        Returns average polarity, volume, and distraction index.
        """
        if not recent_news_list:
            return {"polarity": 0.0, "volume": 0, "distraction_index": 0.0}
            
        scores = [self.analyze_news_headline(news) for news in recent_news_list]
        avg_polarity = sum(scores) / len(scores)
        
        # Distraction index proxies how 'noisy' the news cycle is (many highly emotional reports)
        variance = sum([(s - avg_polarity)**2 for s in scores]) / len(scores)
        
        return {
            "polarity": avg_polarity,
            "volume": len(recent_news_list),
            "distraction_index": variance
        }
