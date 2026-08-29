from apscheduler.schedulers.background import BackgroundScheduler
import logging
import time
from data.open_data import OpenDataInfrastructure

logger = logging.getLogger(__name__)

def start_scheduler(training_loop_instance, db_manager, ingester, engineer, poisson, lstm, boosting):
    """
    Initializes the APScheduler to run the weekly adaptation loop every Monday at 09:00 UTC.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    
    def weekly_job():
        logger.info("Scheduler triggered weekly adaptation sequence.")
        # 0. Sync Spatio-Temporal Open Data
        open_data = OpenDataInfrastructure()
        open_data.fetch_metrica_sample() # Simulated sync
        
        # 1. Standard adaptation
        training_loop_instance.execute(db_manager, ingester, engineer, poisson, lstm, boosting)
        
    # Schedule job for Monday (day_of_week=0) at 09:00
    scheduler.add_job(weekly_job, 'cron', day_of_week='mon', hour=9, minute=0)
    scheduler.start()
    
    logger.info("Scheduler started successfully. System is now autonomous.")
    
    return scheduler
