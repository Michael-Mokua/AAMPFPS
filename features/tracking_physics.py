import polars as pl
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrackingPhysics:
    """
    Computes high-speed running, metabolic power, and fatigue metrics from coordinate data.
    """
    def __init__(self, sampling_rate=25):
        self.sampling_rate = sampling_rate # Hz
        self.dt = 1.0 / sampling_rate

    def compute_kinematics(self, df: pl.DataFrame):
        """
        Calculates velocity and acceleration for all players in the frame.
        Expects a Polars DataFrame with columns: timestamp, player_id, x, y.
        """
        logger.info("Computing kinematics (velocity/acceleration)...")
        
        # Sort by player and timestamp
        df = df.sort(["player_id", "timestamp"])
        
        # Velocity (v = delta_p / delta_t)
        df = df.with_columns([
            (pl.col("x").diff().over("player_id") / self.dt).alias("vx"),
            (pl.col("y").diff().over("player_id") / self.dt).alias("vy")
        ])
        
        df = df.with_columns(
            (pl.col("vx")**2 + pl.col("vy")**2).sqrt().alias("speed")
        )
        
        # Acceleration (a = delta_v / delta_t)
        df = df.with_columns(
            (pl.col("speed").diff().over("player_id") / self.dt).alias("acceleration")
        )
        
        return df

    def compute_metabolic_power(self, df: pl.DataFrame):
        """
        Di Prampero model approximation for energy expenditure.
        P = v * (1 + a^2/g^2)^0.5 * ESM
        """
        logger.info("Computing metabolic power indices...")
        g = 9.81 # gravity
        
        # Equivalent Slope (ES) calculation
        df = df.with_columns(
            (pl.col("acceleration") / g).alias("es")
        )
        
        # Energy Cost approximation
        df = df.with_columns(
            (pl.col("speed") * (1 + pl.col("es")**2).sqrt() * 1.29).alias("metabolic_power")
        )
        
        return df

    def get_summary_stats(self, df: pl.DataFrame):
        """
        Aggregation per player: distance, HSR, avg metabolic power.
        """
        summary = df.group_by("player_id").agg([
            (pl.col("speed").mean() * self.dt * df.height).alias("total_distance"),
            pl.col("speed").filter(pl.col("speed") > 7.0).count().alias("hsr_frames"), # ~25km/h
            pl.col("metabolic_power").mean().alias("avg_metabolic_power"),
            pl.col("acceleration").filter(pl.col("acceleration") > 2.0).count().alias("explosive_actions")
        ])
        
        # Convert frames to distance/time
        summary = summary.with_columns(
            (pl.col("hsr_frames") * self.dt).alias("hsr_duration_seconds")
        )
        
        return summary
