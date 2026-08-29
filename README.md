# AAMPFPS — Adaptive Multi-Model Football Prediction System

A football match prediction pipeline that combines statistical, deep learning, and ensemble methods to generate probability distributions for match outcomes, goal lines, corners, and cards — rather than a single point prediction.

## How it works

**Data ingestion** — Pulls historical and live match data via API integrations (API-Football, Sportmonks) and web scraping, with weather data (OpenWeather) as a supplementary feature. Falls back to cached/offline historical datasets when live sources are unavailable.

**Feature engineering** — Rolling-window team form, tactical and fatigue-related features, SHAP-based feature selection, and NLP sentiment extraction from football news (VADER sentiment).

**Modeling** — A hybrid ensemble rather than one model:
- **Poisson GLM** (`statsmodels`) — a bivariate-style Poisson regression modeling home/away goals as a function of team, opponent, and home advantage, the standard statistical baseline for football score prediction.
- **Deep learning (PyTorch LSTM)** — sequence modeling for time-dependent form/momentum.
- **Gradient boosting (XGBoost)** — tabular feature-based prediction.
- **Adaptive stacking ensemble** — combines the above model outputs.
- **Autoencoder** and **graph intelligence** modules for representation learning and team-relationship modeling.

**Prediction engine** — A Monte Carlo simulator runs 10,000 simulated matches per fixture using Poisson-distributed goals, corners, and cards, producing full probability distributions (win/draw/loss, over/under goal lines, corner and card totals) rather than single-point forecasts.

**Weekly adaptation loop** — An APScheduler-based job retrains models on fresh match data on a weekly cadence, with drift detection (Brier score, calibration slope) to trigger retraining when model performance degrades.

**Dashboard** — A Streamlit interface for visualizing predictions and model performance.

## Tech stack

Python · PyTorch · XGBoost · statsmodels · scikit-learn · SHAP · SQLAlchemy · APScheduler · Streamlit · Plotly

## Status

Core pipeline architecture is implemented end-to-end (ingestion → feature engineering → multi-model training → Monte Carlo simulation → dashboard), with the weekly adaptation loop and drift-based retraining scaffolded in. This is an active prototype, not a production betting system — model calibration and live data reliability are ongoing work.

## Setup

```bash
pip install -r requirements.txt
```

Configure API keys for API-Football, Sportmonks, and OpenWeather in `config.yaml` (the system degrades safely to offline historical data if keys aren't provided).

```bash
streamlit run main.py          # dashboard
python main.py --scheduler     # weekly adaptation daemon
```

## Structure
- `/data` — database management and data ingestion (API + scraping)
- `/features` — feature engineering, rolling windows, sentiment, SHAP selection
- `/models` — Poisson GLM, LSTM, XGBoost, autoencoder, graph intelligence, stacking ensemble
- `/prediction` — Monte Carlo simulation engine, live recalibration
- `/scheduler` — weekly retraining automation
- `/dashboard` — Streamlit interface
