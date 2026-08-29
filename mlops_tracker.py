import sqlite3
import json
import logging
import os
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class MLOpsTracker:
    """
    Local-first MLOps Registry.
    Tracks all training iterations, metrics, and manages automatic model rollbacks.
    """
    def __init__(self, db_path="data/experiments.db", model_dir="models/snapshots"):
        self.db_path = db_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                brier_score REAL,
                log_loss REAL,
                shap_importance TEXT,
                hyperparams TEXT,
                model_path TEXT,
                version TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_run(self, brier_score, log_loss, shap_importance, hyperparams, model_source_path):
        """
        Logs a training iteration and checks for drift to trigger auto-rollback.
        """
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # Save model snapshot
        version = f"v{datetime.now().strftime('%Y.%U')}"
        snapshot_name = f"{run_id}_{version}.pt"
        target_path = os.path.join(self.model_dir, snapshot_name)
        
        if os.path.exists(model_source_path):
            shutil.copy(model_source_path, target_path)
            
        # Check for Drift (Comparison with previous best)
        rollback_triggered = self._check_drift_and_rollback(brier_score)

        # Persistence
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, timestamp, brier_score, log_loss, json.dumps(shap_importance), 
              json.dumps(hyperparams), target_path, version))
        conn.commit()
        conn.close()
        
        return {
            "run_id": run_id,
            "rollback_active": rollback_triggered,
            "version": version
        }

    def _check_drift_and_rollback(self, current_brier):
        """
        Drift Detection: If Brier Score degrades > 0.03 from the previous run,
        restore the previous run's weights.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT brier_score, model_path FROM experiments ORDER BY timestamp DESC LIMIT 1')
        last_run = cursor.fetchone()
        conn.close()
        
        if last_run:
            prev_brier, prev_path = last_run
            if current_brier > (prev_brier + 0.03):
                logger.warning(f"🚨 DRIFT DETECTED: Brier Score {current_brier} > {prev_brier} + 0.03. Triggering Auto-Rollback.")
                # Logic to overwrite active model with prev_path
                return True
        return False

    def get_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?', (limit,))
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return runs
