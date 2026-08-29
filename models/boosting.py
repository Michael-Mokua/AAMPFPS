import xgboost as xgb
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BoostingEngine:
    def __init__(self, config=None):
        self.model = None
        self.config = config or {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits XGBoost on tabular interaction terms.
        Used for discrete modeling to capture non-linearities the Poisson misses.
        """
        logger.info("Training XGBoost Classifier...")
        
        n_estimators = self.config.get("n_estimators", 500)
        max_depth = self.config.get("max_depth", 6)
        learning_rate = self.config.get("learning_rate", 0.01)
        
        # Setup multi-class classification for Home Win, Draw, Away Win.
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False
        )
        
        self.model.fit(X, y)
        return self.model
        
    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError("XGBoost is not fitted yet.")
        return self.model.predict_proba(X)
