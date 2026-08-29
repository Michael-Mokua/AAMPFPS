import pandas as pd
import numpy as np
import logging
import yaml
import os

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path) and os.path.exists("../config.yaml"):
            config_path = "../config.yaml"
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.threshold = self.config.get('features', {}).get('shap_selection_threshold', 0.01)
        
    def filter_by_variance(self, df: pd.DataFrame, threshold=0.01):
        """
        Removes features with near-zero variance.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        keep_cols = variances[variances > threshold].index
        
        # Keep non-numeric ones too (like team names)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        
        return df[list(keep_cols) + list(non_numeric)]
        
    def calculate_shap_importance(self, model, X: pd.DataFrame):
        """
        Given a trained tree-based model (e.g. XGBoost), calculates SHAP values.
        Provides a hook to trim the dataset before neural network processing.
        """
        try:
            import shap
            # Fast tree explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP value per feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': mean_shap})
            
            # Filter features below config threshold
            retained = importance_df[importance_df['shap_importance'] > self.threshold]['feature'].tolist()
            return retained
        except ImportError:
            logger.warning("SHAP library not found. Returning top 50 features by standard feature_importances_.")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:50]
                return X.columns[indices].tolist()
            return X.columns.tolist()
