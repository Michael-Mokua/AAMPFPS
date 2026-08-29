import os
import yaml
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    id = Column(String, primary_key=True) # Usually constructed as Date_Home_Away
    date = Column(Date, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_yellows = Column(Integer)
    away_yellows = Column(Integer)
    home_reds = Column(Integer)
    away_reds = Column(Integer)
    referee = Column(String)
    
class TeamStats(Base):
    __tablename__ = 'team_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey('matches.id'))
    team = Column(String)
    possession = Column(Float)
    passes_completed = Column(Integer)
    pressing_intensity = Column(Float) # PPDA proxy
    formation = Column(String)

class Weather(Base):
    __tablename__ = 'weather'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey('matches.id'))
    temperature_c = Column(Float)
    precipitation_mm = Column(Float)
    wind_speed_kmh = Column(Float)

class DatabaseManager:
    def __init__(self, config_path="config.yaml"):
        # Go up one directory if called from within the package
        if not os.path.exists(config_path) and os.path.exists("../config.yaml"):
            config_path = "../config.yaml"
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        db_string = self.config.get('database', {}).get('connection_string', 'sqlite:///master_dataset.db')
        self.engine = create_engine(db_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()
