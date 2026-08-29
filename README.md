# AAMPFPS: AI-Powered Adaptive Multi-Modal Football Prediction System

The definitive, production-ready football intelligence platform designed for a billion-dollar enterprise solution.

## Architecture

AAMPFPS utilizes a fully autonomous structure designed for extreme accuracy:
- **Exhaustive Feature Engineering**: Ingests everything from tactical line-ups and player fatigue to weather variations and team friction.
- **Hybrid AI Architecture**: Utilizes Poisson GLMs, deep Pytorch LSTMs, and XGBoost gradient boosting in a Bayseian-updated ensemble.
- **Adaptive Weekly Loop**: Re-trains and ingests fresh match data every Monday.
- **Monte Carlo Simulator**: 10,000+ simulations to project full probability distributions for win/draw/loss, goal lines, cards, and corners.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   Edit `config.yaml` to include any relevant API keys (API-Football, Sportmonks, Open-Meteo). The system works without them via safe degradation to offline historical datasets.

3. **Running the Dashboard**
   Launch the Streamlit interface:
   ```bash
   streamlit run main.py
   ```

4. **Running the Scheduler (Weekly Adaptation)**
   Initialize the system in auto-run daemon mode:
   ```bash
   python main.py --scheduler
   ```

## Folder Structure
- `/data`: Database management and API web scarper ingestion.
- `/features`: Deep feature engineering, rolling lags, specific sentiment extraction, and SHAP reduction.
- `/models`: Poisson, PyTorch LSTM, XGBoost, and Stacking structures.
- `/prediction`: Monte Carlo engine merging model outputs into exact betting distributions.
- `/scheduler`: APScheduler instance acting on weekly cadences.
- `/dashboard`: High-end Streamlit GUI.
