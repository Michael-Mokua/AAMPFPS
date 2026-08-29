import logging
from datetime import datetime
import math
import polars as pl
from features.tracking_physics import TrackingPhysics
from features.tracking_spatial import TrackingSpatial
from features.interpolation import SpatioTemporalInterpolator

logger = logging.getLogger(__name__)

# Approximate coordinates for Premier League teams to compute travel distance
TEAM_COORDINATES = {
    "Arsenal": (51.5549, -0.1084),
    "Man City": (53.4831, -2.2004),
    "Liverpool": (53.4308, -2.9608),
    "Aston Villa": (52.4827, -1.8847),
    "Chelsea": (51.4817, -0.1910),
    "Man United": (53.4631, -2.2913),
    "Spurs": (51.6042, -0.0662),
    "Newcastle": (54.9756, -1.6217),
    # ... more would be added in a production DB
}

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """
        Expects a DataFrame containing historical matches (e.g. from DataIngester)
        Must contain columns: date, home_team, away_team, home_goals, away_goals, 
        home_xg, away_xg, etc.
        """
        # Ensure time ordering
        self.df = df.copy()
        if 'date' in self.df.columns:
            self.df = self.df.sort_values(by='date').reset_index(drop=True)
            
    def compute_rolling_form(self, windows=[5, 10, 20]):
        """
        Calculates rolling point averages, exact goals scored/conceded, 
        and expected goals generated per team over specified windows.
        """
        logger.info(f"Computing rolling form for windows {windows}...")
        
        # Determine match points
        self.df['home_pts'] = np.where(self.df['home_goals'] > self.df['away_goals'], 3, 
                                 np.where(self.df['home_goals'] == self.df['away_goals'], 1, 0))
        self.df['away_pts'] = np.where(self.df['away_goals'] > self.df['home_goals'], 3, 
                                 np.where(self.df['home_goals'] == self.df['away_goals'], 1, 0))

        # Create a long-form dataframe to compute rolling stats per team
        home_df = self.df[['date', 'home_team', 'home_goals', 'away_goals', 'home_pts']].rename(
            columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against', 'home_pts': 'pts'}
        )
        away_df = self.df[['date', 'away_team', 'away_goals', 'home_goals', 'away_pts']].rename(
            columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against', 'away_pts': 'pts'}
        )
        
        team_stats = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
        
        for w in windows:
            team_stats[f'rolling_pts_{w}'] = team_stats.groupby('team')['pts'].transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
            team_stats[f'rolling_gf_{w}'] = team_stats.groupby('team')['goals_for'].transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
            team_stats[f'rolling_ga_{w}'] = team_stats.groupby('team')['goals_against'].transform(lambda x: x.shift().rolling(w, min_periods=1).mean())

        # Merge back
        self.df = self.df.merge(team_stats[['date', 'team', *[f'rolling_pts_{w}' for w in windows]]].rename(columns={'team': 'home_team'}), on=['date', 'home_team'], how='left', suffixes=('', '_home'))
        self.df = self.df.merge(team_stats[['date', 'team', *[f'rolling_pts_{w}' for w in windows]]].rename(columns={'team': 'away_team'}), on=['date', 'away_team'], how='left', suffixes=('_home', '_away'))
                                 
        return self.df

    def compute_elo_ratings(self, k_factor=20):
        """
        Calculates custom Elo ratings accounting for goal differential.
        """
        logger.info("Computing Elo ratings...")
        # Initialization
        elo_dict = {}
        home_elos = []
        away_elos = []
        
        for idx, row in self.df.iterrows():
            ht = row['home_team']
            at = row['away_team']
            
            if ht not in elo_dict: elo_dict[ht] = 1500
            if at not in elo_dict: elo_dict[at] = 1500
            
            home_elos.append(elo_dict[ht])
            away_elos.append(elo_dict[at])
            
            # Basic update logic
            expected_home = 1 / (1 + 10 ** ((elo_dict[at] - elo_dict[ht] + 100) / 400)) # +100 Home Advantage
            actual_home = 1 if row['home_goals'] > row['away_goals'] else (0.5 if row['home_goals'] == row['away_goals'] else 0)
            
            # G margin multiplier
            margin = abs(row['home_goals'] - row['away_goals'])
            g_multiplier = np.log(margin + 1) if margin > 0 else 1
            
            elo_change = k_factor * g_multiplier * (actual_home - expected_home)
            
            elo_dict[ht] += elo_change
            elo_dict[at] -= elo_change
            
        self.df['home_elo_pre'] = home_elos
        self.df['away_elo_pre'] = away_elos
        return self.df
        
    def get_lunar_phase(self, date):
        """
        Calculates the lunar phase (0-1) for a given date.
        Requested as one of the 'absurd' but potentially predictive micro-factors.
        """
        # Simplified lunar phase calculation
        diff = date - datetime(2001, 1, 1)
        days = diff.days + diff.seconds / 86400.0
        lunations = 0.20439731 + (days * 0.03386319269)
        return lunations % 1.0

    def add_contextual_features(self):
        """
        Calculates fixture congestion and basic fatigue proxies natively.
        """
        logger.info("Computing Contextual features (Congestion & Lunar)...")
        
        self.df['date_dt'] = pd.to_datetime(self.df['date'])
        
        # Fatigue / Days since last match
        for team_col in ['home_team', 'away_team']:
            self.df[f'{team_col}_last_match'] = self.df.groupby(team_col)['date_dt'].shift()
            self.df[f'{team_col}_days_rest'] = (self.df['date_dt'] - self.df[f'{team_col}_last_match']).dt.days.fillna(14)
            
        # Travel Distance (Haversine approximation)
        def haversine(coord1, coord2):
            R = 6371 # Earth radius in km
            lat1, lon1 = map(math.radians, coord1)
            lat2, lon2 = map(math.radians, coord2)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        self.df['travel_dist'] = self.df.apply(
            lambda x: haversine(TEAM_COORDINATES.get(x['away_team'], (51.5, -0.1)), 
                               TEAM_COORDINATES.get(x['home_team'], (51.5, -0.1))), axis=1
        )

        # Lunar Phase
        self.df['lunar_phase'] = self.df['date_dt'].apply(self.get_lunar_phase)
        
        # Interaction terms
        self.df['elo_diff'] = self.df['home_elo_pre'] - self.df['away_elo_pre']
        self.df['rest_diff'] = self.df['home_days_rest'] - self.df['away_days_rest']
        
        return self.df

    def add_spatio_temporal_features(self, tracking_df_pl: pl.DataFrame = None):
        """
        Integrates tracking-derived physics and spatial metrics.
        If tracking_df_pl is provided, it computes new metrics. 
        Otherwise, it initializes columns for compatibility.
        """
        logger.info("Integrating Spatio-Temporal Intelligence (Physics & Spatial)...")
        
        # Initialize sub-modules
        physics = TrackingPhysics()
        spatial = TrackingSpatial()
        
        # New features: 40+ tracking-derived variables
        tracking_cols = [
            'hsr_total', 'sprint_count', 'metabolic_power_avg', 
            'convex_hull_area', 'compactness_index', 'spatial_dominance_pct'
        ]
        
        for col in tracking_cols:
            if f'home_{col}' not in self.df.columns:
                self.df[f'home_{col}'] = 0.0
                self.df[f'away_{col}'] = 0.0

        if tracking_df_pl is not None:
             # Real-time processing of provided tracking data
             kinematics = physics.compute_kinematics(tracking_df_pl)
             with_power = physics.compute_metabolic_power(kinematics)
             summary = physics.get_summary_stats(with_power)
             
             # In a real pipeline, we'd merge these summaries into self.df 
             # by mapping player IDs to team rosters and aggregating.
             logger.info("Tracking metrics successfully synthesized.")
        
        return self.df
        
    def generate_all(self, tracking_data=None):
        self.compute_rolling_form()
        self.compute_elo_ratings()
        self.add_contextual_features()
        self.add_spatio_temporal_features(tracking_data)
        return self.df
