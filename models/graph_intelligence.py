import numpy as np
import logging

logger = logging.getLogger(__name__)

class GraphIntelligenceLayer:
    def __init__(self, teams: list):
        self.teams = teams
        self.num_teams = len(teams)
        self.team_to_idx = {team: i for i, team in enumerate(teams)}
        # Adjacency matrix representing the league graph
        self.adj = np.zeros((self.num_teams, self.num_teams))
        # Feature matrix for teams (Elo, Form, etc.)
        self.features = np.ones((self.num_teams, 5)) # Placeholder for 5 core team metrics

    def update_edge(self, team_a, team_b, result_score):
        """
        result_score: +1 for A win, 0 for draw, -1 for B win
        """
        idx_a = self.team_to_idx.get(team_a)
        idx_b = self.team_to_idx.get(team_b)
        
        if idx_a is not None and idx_b is not None:
             # Basic message passing: increase weight based on performance
             self.adj[idx_a, idx_b] += result_score
             self.adj[idx_b, idx_a] -= result_score
             logger.debug(f"Graph Intelligence: Updated edge {team_a} <-> {team_b}")

    def propagate_strength(self, iterations=3):
        """
        Simple graph propagation to capture transitive strength.
        """
        for _ in range(iterations):
            # Proximity-based strength update
            # New Strength = Base + alpha * (Average Strength of teams you beat)
            self.features = 0.8 * self.features + 0.2 * np.dot(self.adj, self.features)
            
    def get_relative_strength(self, team_a, team_b):
        """
        Returns the graph-inferred strength differential.
        """
        idx_a = self.team_to_idx.get(team_a)
        idx_b = self.team_to_idx.get(team_b)
        
        if idx_a is not None and idx_b is not None:
             strength_a = np.mean(self.features[idx_a])
             strength_b = np.mean(self.features[idx_b])
             return strength_a - strength_b
        return 0.0
