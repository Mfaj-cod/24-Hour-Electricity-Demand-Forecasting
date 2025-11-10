# Fast-Track 24h Electricity Demand Forecast

This project provides a lightweight, end-to-end forecasting pipeline for predicting the next 24 hours of electricity demand for Bareilly and Mathura, using historical smart meter data and optional weather forecasts.

## Why This Exists

Power utilities need quick, reasonably accurate demand forecasts for operational decisions like generation scheduling, demand response, and grid balancing. Traditional forecasting setups often require complex feature pipelines or proprietary software.

This script aims to bridge that gap by:
- Automating data cleaning & hourly aggregation from smart meter CSVs.
- Using seasonal naive and ridge regression methods for fast forecasts.
- Optionally enriching with weather data via Open-Meteo.
- Producing plots, metrics, and a PDF report within minutes — no complex MLOps stack needed.

## How It Works

1. Data ingestion:  
   Loads smart meter readings, cleans timestamps, finds the energy column, and resamples to hourly.

2. Preprocessing:  
   - Imputes short missing gaps (forward fill + interpolation).  
   - Caps extreme outliers at 1st/99th percentile.

3. Optional Weather: 
   Fetches hourly temperature forecasts for Bareilly and merges with demand data.

4. Feature Engineering: 
   Builds time features (hour, sin/cos, day of week), lagged values, and 24h rolling means.

5. Forecasting: 
   - Baseline: Seasonal naive = repeat yesterday’s hour values.  
   - Ridge Regression: Trains on last N days, autoregressively predicts next 24 hours.

6. Evaluation & Reporting:  
   If ground truth is available for the forecast horizon, computes MAE, WAPE, sMAPE and RMSE.  
   Generates plots and a 2-page PDF report with metrics and visualizations.

## Usage

# Run pipeline for Bareilly, 7 days history, without weather
python run_forecast.py --city Bareilly --data_path "data\CEEW - Smart meter data Bareilly 2020.csv" --history_window 7 --with_weather false --make_plots true --save_report true


# Run pipeline for Mathura, 7 days history, without weather
python run_forecast.py --city Mathura --data_path "data\CEEW - Smart meter data Mathura 2020.csv" "data\CEEW - Smart meter data Mathura 2020.csv" --history_window 7 --with_weather false --make_plots true --save_report true