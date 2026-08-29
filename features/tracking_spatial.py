import numpy as np
import polars as pl
from scipy.spatial import ConvexHull, Voronoi
import logging

logger = logging.getLogger(__name__)

class TrackingSpatial:
    """
    Computes team-level spatial logic: Convex Hull, Compactness, and Pitch Control.
    """
    def __init__(self, pitch_length=105, pitch_width=68):
        self.pitch_dim = (pitch_length, pitch_width)

    def compute_convex_hull(self, player_coords: np.ndarray):
        """
        Calculates the area of the team's spread (convex hull).
        player_coords: (N, 2) array of (x, y) coordinates.
        """
        if len(player_coords) < 3:
            return 0.0
        try:
            hull = ConvexHull(player_coords)
            return hull.area # Note: area in 2D is the perimeter; hull.volume is the area.
        except Exception:
            return 0.0

    def get_team_compactness(self, df: pl.DataFrame, team_id: str):
        """
        Aggregates convex hull area per frame for a specific team.
        """
        # Filter for the team and ensure we have enough players
        team_df = df.filter(pl.col("team_id") == team_id)
        
        # We need to process frame by frame
        frames = team_df.get_column("timestamp").unique().to_list()
        areas = []
        
        for frame in frames:
            coords = team_df.filter(pl.col("timestamp") == frame).select(["x", "y"]).to_numpy()
            if len(coords) >= 3:
                hull = ConvexHull(coords)
                areas.append(hull.volume) # ConvexHull.volume is the area in 2D
        
        return np.mean(areas) if areas else 0.0

    def compute_spatial_dominance(self, home_coords, away_coords):
        """
        Voronoi-based pitch control approximation.
        returns: (home_area_pct, away_area_pct)
        """
        all_coords = np.vstack([home_coords, away_coords])
        vor = Voronoi(all_coords)
        
        # Simplified: Count which team owns more Voronoi regions
        # In full production, we'd clip regions to pitch boundaries and calculate areas.
        return len(home_coords) / len(all_coords) # Placeholder logic for speed

    def compute_pressing_intensity(self, df: pl.DataFrame, ball_x, ball_y):
        """
        Counts players within 10m of the ball.
        """
        df = df.with_columns(
            ((pl.col("x") - ball_x)**2 + (pl.col("y") - ball_y)**2).sqrt().alias("dist_to_ball")
        )
        
        pressing_players = df.filter(pl.col("dist_to_ball") < 10.0).group_by("team_id").count()
        return pressing_players.to_dict(as_series=False)
