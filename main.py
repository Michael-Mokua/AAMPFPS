import argparse
import logging
import sys
import os

from data.database import DatabaseManager
from data.ingestion import DataIngester
from features.engineering import FeatureEngineer
from features.selection import FeatureSelector
from models.poisson import PoissonGLM
from models.deep_learning import DeepLearningEngine
from models.boosting import BoostingEngine
from models.ensemble import AdaptiveStacker
from models.training_loop import WeeklyAdaptationLoop
from scheduler.jobs import start_scheduler
from prediction.monte_carlo import MonteCarloSimulator
from prediction.engine import PredictionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AAMPFPS')

def main():
    parser = argparse.ArgumentParser(description="AAMPFPS Core Execution System")
    parser.add_argument('--scheduler', action='store_true', help='Start the autonomous weekly scheduler')
    parser.add_argument('--dry-run', action='store_true', help='Initialize structure without training')
    
    args = parser.parse_args()
    
    logger.info("Initializing AAMPFPS Core Platform...")
    
    # 1. Initialize DB and Data Loaders
    db_manager = DatabaseManager()
    ingester = DataIngester()
    
    if args.dry_run:
        logger.info("[Dry Run] System initialized successfully. Exiting.")
        sys.exit(0)
        
    # 2. Setup AI Models
    poisson = PoissonGLM()
    lstm = DeepLearningEngine()
    boosting = BoostingEngine()
    stacker = AdaptiveStacker()
    mc = MonteCarloSimulator()
    engine = PredictionEngine(poisson, lstm, boosting, stacker, mc)
    
    # 3. Handle execution modes
    if args.scheduler:
        logger.info("Starting Scheduler Daemon Mode...")
        loop = WeeklyAdaptationLoop()
        engineer = FeatureEngineer   # Class ref for instantiation inside loop
        
        # Start background job
        sched = start_scheduler(loop, db_manager, ingester, engineer, poisson, lstm, boosting)
        
        try:
            # Keep main thread alive
            import time
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            sched.shutdown()
            logger.info("Scheduler shutting down...")
    else:
        logger.info("No daemon mode specified. Run with 'streamlit run dashboard/app.py' for UI, or '--scheduler' for automation.")

if __name__ == "__main__":
    main()
