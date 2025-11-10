import os
import argparse
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

def ensure_dirs():
    os.makedirs('artifacts/fast_track/plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)


def load_data(paths, city='Bareilly'):
    # ensuring paths is a list, even if a single string is passed
    if isinstance(paths, str):
        paths = [paths]
        
    all_data = []
    
    for path in paths:
        # Loading the data, checking for CSV or Parquet
        if path.endswith('.csv'):
            df = pd.read_csv(path, parse_dates=["x_Timestamp"])
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type for path: {path}")

        df.columns = [c.strip() for c in df.columns]

        # Finding t_kWh column
        possible_kwh = [c for c in df.columns if c.lower() in ('t_kwh')]
        if not possible_kwh:
            raise ValueError(f'Could not find energy column (t_kWh) in dataset: {path}')
        
        # renaming the column
        df.rename(columns={possible_kwh[0]: 'kwh'}, inplace=True)
        all_data.append(df)

    if not all_data:
        raise ValueError("No data loaded from paths.")
    
    # Concatenating
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Processing the combined DataFrame
    df_combined = df_combined.set_index('x_Timestamp').sort_index()

    # ensuring timezone (IST naive -> convert to UTC aware if needed)
    if df_combined.index.tz is None:
        try:
            df_combined.index = df_combined.index.tz_localize('Asia/Kolkata')
        except Exception:
            pass
            
    # Removing any duplicate index entries that might arise from concatenation
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

    return df_combined


def resample_to_hourly(df_hour3min):
    # sum over each hour
    df_h = df_hour3min['kwh'].resample('1h').sum().to_frame('kwh')
    # Ensuring continuous index
    idx = pd.date_range(df_h.index.min(), df_h.index.max(), freq='1h', tz=df_h.index.tz)
    df_h = df_h.reindex(idx)
    return df_h


def impute_and_cap(df_h):
    # Imputing small gaps: forward fill up to 3 hours, else interpolate
    nans = df_h['kwh'].isna().sum()
    if nans>0:
        df_h['kwh'] = df_h['kwh'].ffill(limit=3)
        df_h['kwh'] = df_h['kwh'].interpolate(limit_direction='both')
    # Capping outliers using percentiles
    low, high = df_h['kwh'].quantile([0.01, 0.99])
    df_h['kwh'] = df_h['kwh'].clip(lower=low, upper=high)
    return df_h



def fetch_open_meteo_forecast(lat, lon, start_iso, hours=24):
    """
    Fetch hourly temperature forecasts from Open-Meteo using lat/lon.
    Open-Meteo provides the next 48 hours of forecast data.
    """
    # Open-Meteo hourly temperature forecast. forecast_days=2 fetching 48 hours.
    url = (
        f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}' 
        f'&hourly=temperature_2m&forecast_days=2&timezone=Asia%2FKolkata'
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    
    # Extracting data using the correct Open-Meteo response structure
    times = pd.to_datetime(j['hourly']['time'])
    
    # Ensuring timezone localization
    if times.tz is None:
        times = times.tz_localize('Asia/Kolkata')
        
    temps = j['hourly']['temperature_2m']
    dfw = pd.DataFrame({'temperature': temps}, index=times)
    
    # Returning the full forecast, main function will handle time alignment.
    return dfw



def make_features(df, use_weather=False):
    df = df.copy()
    df['hour'] = df.index.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow'] = df.index.dayofweek
    df['lag1'] = df['kwh'].shift(1)
    df['lag2'] = df['kwh'].shift(2)
    df['lag3'] = df['kwh'].shift(3)
    df['roll_mean_24'] = df['kwh'].rolling(24, min_periods=1).mean()
    if use_weather and 'temperature' in df.columns:
        pass
    df = df.dropna(subset=['lag1', 'lag2', 'lag3', 'roll_mean_24'])
    return df


def seasonal_naive_forecast(df_h, origin_time):
    # origin_time is the last observed x_timestamp T
    # produce T+1..T+24 by copying same hours from previous day
    start = origin_time + pd.Timedelta(hours=1)
    idx = pd.date_range(start, periods=24, freq='1h', tz=origin_time.tz)
    yhat = []
    for t in idx:
        prev_time = t - pd.Timedelta(hours=24)
        if prev_time in df_h.index:
            yhat.append(df_h.loc[prev_time, 'kwh'])
        else:
            yhat.append(np.nan)

    return pd.DataFrame({'x_timestamp': idx, 'yhat': yhat}).set_index('x_timestamp')





def ridge_forecast(df_h, origin_time, use_weather=False):
    # Training on last 7 days ending at T-1 (history window will be handled outside)
    # preparing features, training on available rows prior to origin_time, then iteratively predicting next 24 hours
    df_feat = df_h.copy()
    if use_weather and 'temperature' not in df_feat.columns:
        raise ValueError('Weather requested but temperature column missing')
    dff = make_features(df_feat, use_weather=use_weather)

    # training: everything with index <= origin_time - 1 hour
    train_end = origin_time - pd.Timedelta(hours=1)
    train_start = origin_time - pd.Timedelta(days=7)
    train = dff.loc[train_start:train_end]

    X_cols = ['sin_hour','cos_hour','dow','lag1','lag2','lag3','roll_mean_24']
    if use_weather:
        X_cols = X_cols + ['temperature']
        # imputing NaNs in the temperature column for the training data.
        # mean of the future forecast
        fill_value = df_feat['temperature'].mean()
        
        # filling NaNs in the training data's temperature column
        train.loc[:, 'temperature'] = train['temperature'].fillna(fill_value)

    if len(train) == 0:
        raise ValueError("No training data available before origin_time")

    MIN_TRAINING_SAMPLES = 50 # minimum number of hourly samples
    
    if len(train) < MIN_TRAINING_SAMPLES:
        raise RuntimeError(
            f"Insufficient training data. Found only {len(train)} samples "
            f"after feature engineering and windowing (Target: >{MIN_TRAINING_SAMPLES} samples)."
        )

    X_train = train[X_cols]
    y_train = train['kwh'].values

    # model = Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='saga', positive=False, random_state=2)
    model = HistGradientBoostingRegressor(learning_rate=0.08)

    model.fit(X_train, y_train)

    # iterative forecasting: for h in 1..24
    preds = []
    df_sim = dff.copy()

    for h in range(1, 25):
        t = origin_time + pd.Timedelta(hours=h)
        # creating row for t using available history
        row = {}
        row['hour'] = t.hour
        row['sin_hour'] = np.sin(2 * np.pi * row['hour'] / 24)
        row['cos_hour'] = np.cos(2 * np.pi * row['hour'] / 24)
        row['dow'] = t.dayofweek
        # lags
        for lag in (1,2,3):
            lt = t - pd.Timedelta(hours=lag)
            lag_val = df_sim['kwh'].get(lt, np.nan) 
            row[f'lag{lag}'] = lag_val
           
        # roll mean 24
        window = [t - pd.Timedelta(hours=i) for i in range(1,25)]

        vals = [df_sim['kwh'].get(wt, np.nan) for wt in window]

        row['roll_mean_24'] = np.nanmean(vals)
        if use_weather:
            # safely retrieving temperature forecast
            temp_val = df_sim['temperature'].get(t, np.nan)
            row['temperature'] = temp_val

        Xpred = pd.DataFrame([row], index=[t])[X_cols].astype(float)
        if Xpred.isnull().values.any():
             # robust fallback for missing initial lags/roll_mean
             # Filling NaNs in Xpred using column means from training data for safety
             X_train_mean = X_train.mean()
             Xpred = Xpred.fillna(X_train_mean)

        ypred = model.predict(Xpred)[0]

        # Updating df_sim with the predicted value for the next iteration's lags
        df_sim.loc[t, 'kwh'] = ypred
        if use_weather and 'temperature' in df_sim.columns:
            # Ensuring temperature is also set, even if it's NaN for non-forecast times
            if t not in df_sim.index or 'temperature' not in df_sim.columns:
                 df_sim.loc[t, 'temperature'] = row.get('temperature')

        preds.append((t, ypred))

    idx = pd.DatetimeIndex([p[0] for p in preds])
    yhat = [p[1] for p in preds]
    return pd.DataFrame({'x_timestamp': idx, 'yhat': yhat}).set_index('x_timestamp')


def evaluate(y_true, y_pred):
    # y_true, y_pred are pandas Series aligned by index
    y_true, y_pred = y_true.align(y_pred, join='inner')
    mae = mean_absolute_error(y_true, y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
    rmse = root_mean_squared_error(y_true, y_pred)
    return {'MAE': mae, 'WAPE': wape, 'sMAPE': smape, 'RMSE': rmse}


def horizon_mae(actual, preds_df):
    # preds_df: index x_timestamp, column yhat; actual has true x_timestamps
    # returns list of MAE for horizons 1 to 24
    maes = []
    for t, row in preds_df.iterrows():
        maes.append(abs(actual.loc[t] - row['yhat']))

    return maes


def make_plots(df_h, forecast_df, baseline_df):
    # plot of last 3 days actual + forecast
    end = df_h.index.max()
    start = end - pd.Timedelta(days=3) + pd.Timedelta(hours=1)
    fig, ax = plt.subplots(figsize=(12,5))
    df_h.loc[start:end]['kwh'].plot(ax=ax, label='actual')
    forecast_df['yhat'].plot(ax=ax, label='HistGradientBoostingRegressor forecast', marker='o')
    baseline_df['yhat'].plot(ax=ax, label='seasonal naive', marker='x')
    ax.set_title('Last 3 days actuals + next 24h forecast')
    ax.legend()
    plt.tight_layout()
    p1 = 'artifacts/fast_track/plots/forecast_overview.png'
    fig.savefig(p1)
    plt.close(fig)


    # Horizon-wise MAE (requires truth for next 24h)
    maes = []
    for h in range(1,25):
        t = forecast_df.index[0] + pd.Timedelta(hours=h-1)
        if t in df_h.index:
            maes.append(abs(df_h.loc[t,'kwh'] - forecast_df.iloc[h-1]['yhat']))
        else:
            maes.append(np.nan)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(1,25), maes)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Abs Error')
    ax.set_title('Horizon-wise absolute error (1..24)')
    plt.tight_layout()
    p2 = 'artifacts/fast_track/plots/mae_plot.png'
    fig.savefig(p2)
    plt.close(fig)
    return p1, p2


def write_forecast_csv(df_forecast, tag='T_plus_24'):
    out = f'artifacts/fast_track/forecast_{tag}.csv'
    df_forecast.reset_index().rename(columns={'index':'x_timestamp'}).to_csv(out, index=False)
    return out


def write_metrics_csv(metrics_dict):
    dfm = pd.DataFrame(metrics_dict).T
    out = 'artifacts/fast_track/metrics.csv'
    dfm.to_csv(out)
    return out


def write_report_text(metrics, p1, p2):
    # Creating a simple 2-page PDF with text + plots using PdfPages
    pdf_path = 'reports/fast_track_report.pdf'
    with PdfPages(pdf_path) as pdf:
        # Page-1: text overview
        fig = plt.figure(figsize=(8.27, 11.69))
        txt = (
            'Objective: Predict 24-hour ahead electricity demand for the specified region (Bareilly/Mathura).\n\n'
            'Data Preparation:\n'
            '1. Data Coverage: Meter readings from 2020/2021 (Bareilly) and 2019/2020 (Mathura) were combined to ensure a \nrobust, multi-year training history.\n'
            '2. Resampling: Raw 3-minute readings were aggregated (summed) into 1-hour intervals, establishing the \n prediction frequency.\n'
            '3. Imputation: Missing values were handled by forward-filling up to 3 hours, followed by linear \n interpolation for remaining gaps.\n'
            '4. Outliers: Data stability was ensured by clipping extreme consumption values at the 1st and 99th \n historical percentiles.\n\n'
            'Modeling Approach:\n'
            '1. Baseline: Seasonal Naive, which predicts consumption for a future hour based on the consumption \n from the same hour on the previous day.\n'
            '2. HistGradientBoostingRegressor Model: A model trained on a 7-day rolling historical \n window (ending at origin\time T).\n'
            ' - Features: Lag consumption (1, 2, 3 hours), 24h rolling mean, day-of-week, and cyclical Sin/Cos \n of hour-of-day. Temperature is included if external API fetch is successful.\n'
            ' - Forecasting: Predictions were generated iteratively, where the forecast for $T+h$ is used to derive \n features (lags) needed for the $T+h+1$ prediction.\n\n'
            'Performance Metrics:\n\n'
        )
        # remove any stray tab characters that cause the Matplotlib warning
        txt = txt.replace('\t', ' ').replace('\xa0', ' ')

        # text at 5% from the left, 95% from the bottom (using va='top')
        fig.text(0.05, 0.95, txt, va='top', fontsize=10) 

        # the Metrics Table
        
        # data for the table
        headers = ['Model', 'MAE', 'WAPE', 'sMAPE', 'RMSE']
        data = []
        model_names = {'HistGradientBoosting': 'HGBoost', 'baseline': 'Seasonal Naive'}
        metric_keys = ['MAE', 'WAPE', 'sMAPE', 'RMSE']
        
        for key in ['HistGradientBoosting', 'baseline']:
            if key in metrics:
                row = [model_names[key]]
                for m_key in metric_keys:
                    val = metrics[key].get(m_key, np.nan)
                    # Format to 3 decimal places or display N/A if NaN
                    row.append(f"{val:.3f}" if not np.isnan(val) else 'N/A')
                data.append(row)

        # Defining an axis for the table (to help with positioning)
        ax_table = fig.add_subplot(111)
        ax_table.axis('off') # Hide axes lines

        # Positioning the table below the introductory text block
        table_y_position = 0.58 # Y-coordinate for the bottom of the table
        
        # Creating the table
        table = ax_table.table(
            cellText=data,
            colLabels=headers,
            loc='center', 
            cellLoc='center',
            bbox=[0.05, table_y_position, 0.9, 0.1] 
        )
        
        # style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5) # Scaled height slightly for readability

        pdf.savefig(fig)
        plt.close(fig)

        # Page-2: plots
        fig = plt.figure(figsize=(8.27, 11.69))
        ax1 = fig.add_subplot(211)
        img1 = plt.imread(p1)
        ax1.imshow(img1)
        ax1.axis('off')
        ax2 = fig.add_subplot(212)
        img2 = plt.imread(p2)
        ax2.imshow(img2)
        ax2.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def main(args):
    ensure_dirs()
    df = load_data(args.data_path, city=args.city)
    dfh = resample_to_hourly(df)
    dfh = impute_and_cap(dfh)

    # manually setting an earlier origin_time for backtesting.
    # pretend it's the end of Dec 30th and forecast the 31st. 
    origin_time = pd.Timestamp('2020-12-30 23:00:00', tz='Asia/Kolkata')

    # weather
    use_weather = args.with_weather
    if use_weather:
        # Bareilly approx coords
        lat, lon = 28.3670, 79.4304
        try:
            dfw = fetch_open_meteo_forecast(lat, lon, origin_time, hours=24)
            # merging into dfh (it will contain forecasts for next 48h)
            dfh = dfh.merge(dfw, left_index=True, right_index=True, how='left')
        except Exception as e:
            print('Weather fetch failed, proceeding without weather:', e)
            use_weather = False

    # Baseline forecast
    baseline = seasonal_naive_forecast(dfh, origin_time)
    # forecast
    try:
        # Pass the history_window argument to the forecast function
        ridge = ridge_forecast(dfh, origin_time, use_weather=use_weather)
    except Exception as e:
        print('forecast failed, falling back to baseline only:', e)
        ridge = baseline.copy()

    # Saving forecasts
    f1 = write_forecast_csv(ridge, tag='T_plus_24')
    f2 = write_forecast_csv(baseline, tag='baseline_T_plus_24')

    # Evaluating if actuals for next 24h exist in dfh
    metrics = {}
    if all(ts in dfh.index for ts in ridge.index):
        true = dfh.loc[ridge.index, 'kwh']
        metrics['HistGradientBoosting'] = evaluate(true, ridge['yhat'])
        metrics['baseline'] = evaluate(true, baseline['yhat'])
    else:
        metrics['HistGradientBoosting'] = {'MAE': np.nan, 'WAPE': np.nan, 'sMAPE': np.nan}
        metrics['baseline'] = {'MAE': np.nan, 'WAPE': np.nan, 'sMAPE': np.nan}

    mfile = write_metrics_csv(metrics)

    # Plots
    p1, p2 = make_plots(dfh, ridge, baseline)

    # Report
    rep = write_report_text(metrics, p1, p2)

    print('\nArtifacts written to:')
    print(' -', f1)
    print(' -', f2)
    print(' -', mfile)
    print(' -', p1)
    print(' -', p2)
    print(' -', rep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Bareilly')
    parser.add_argument('--data_path', required=True, nargs='+', help='Path(s) to smart meter CSV or parquet')
    parser.add_argument('--history_window', default='7', help='days of history to use (integer)')
    parser.add_argument('--with_weather', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)
    parser.add_argument('--make_plots', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)
    parser.add_argument('--save_report', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)
    args = parser.parse_args()

    try:
        args.history_window = int(str(args.history_window))
    except Exception:
        args.history_window = 7

    main(args)