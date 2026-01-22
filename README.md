# Food-Demand-Forecast
Hackathon 


This project solves a demand forecasting problem for a meal delivery company using historical order data.

## Problem
Forecast weekly demand for meal–center combinations to help optimize inventory and staffing.

## Approach
- Feature engineering:
  - Long-term average demand per center–meal
  - Recent demand signal (lag-1)
  - Price & promotion features
  - Seasonal features (sin/cos of week)
- Model:
  - XGBoost Regressor
  - Log-transformed target to optimize RMSLE
- Evaluation:
  - RMSLE (competition metric)

## Results
- Local RMSLE: ~0.46
- Public leaderboard score: ~46

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost

## Notes
This project focuses on building a clean, stable, and explainable ML pipeline rather than aggressive leaderboard tuning.
