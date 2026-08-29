import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging

logger = logging.getLogger(__name__)

class PoissonGLM:
    def __init__(self):
        self.model = None
        
    def fit(self, df: pd.DataFrame):
        """
        Fits a bivariate-style Poisson model for Home and Away Goals.
        Uses team strengths, home advantage, and exogenous features.
        We stack the dataframe to predict 'goals' based on team, opponent and venue.
        """
        logger.info("Fitting Poisson GLM...")
        
        # Reshape data for GLM
        goal_data = pd.concat([
            df[['home_team', 'away_team', 'home_goals']].assign(home=1).rename(
                columns={'home_team': 'team', 'away_team': 'opponent', 'home_goals': 'goals'}
            ),
            df[['away_team', 'home_team', 'away_goals']].assign(home=0).rename(
                columns={'away_team': 'team', 'home_team': 'opponent', 'away_goals': 'goals'}
            )
        ])
        
        # Basic formula (could be expanded with xG differentials and ELOs)
        formula = "goals ~ home + team + opponent"
        self.model = smf.glm(formula=formula, data=goal_data, family=sm.families.Poisson()).fit()
        return self.model
        
    def predict_lambda(self, home_team: str, away_team: str):
        """
        Calculates expected goals (lambda) for both teams in a single fixture.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
            
        home_lambda = self.model.predict(pd.DataFrame(data={'team': [home_team], 'opponent': [away_team], 'home': [1]}))[0]
        away_lambda = self.model.predict(pd.DataFrame(data={'team': [away_team], 'opponent': [home_team], 'home': [0]}))[0]
        
        return home_lambda, away_lambda
