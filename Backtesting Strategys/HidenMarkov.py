import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
FRED_API_KEY = '86ba6d681de05a0a677954b3e11743de'
ASSET_TICKER = 'ASML'     # Silver Futures
START_DATE = '2000-01-01'

def get_price_data():
    print(f"Downloading {ASSET_TICKER} data...")
    try:
        df = yf.download(ASSET_TICKER, start=START_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['Return'] = df[price_col].pct_change()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading price data: {e}")
        return pd.DataFrame()

def get_macro_data():
    print("Downloading Macro data from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    
    monthly_series = {'CPI': 'CPIAUCSL', 'UNRATE': 'UNRATE', 'PAYROLLS': 'PAYEMS'}
    daily_series = {'FEDFUNDS': 'FEDFUNDS', 'US10Y': 'DGS10'}
    
    macro_data = {}
    
    # Lag monthly data by 45 days to ensure publication delay is respected
    for name, sid in monthly_series.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE)
            s.index = s.index + pd.Timedelta(days=45)
            macro_data[name] = s
        except Exception as e:
            print(f"Error loading {name}: {e}")

    for name, sid in daily_series.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE)
            # Safe lag of 1 day even for daily data to avoid same-day bias
            s = s.shift(1)
            macro_data[name] = s
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    return pd.DataFrame(macro_data)

def merge_data(price_df, macro_df):
    print("Merging datasets (Strict Causality)...")
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None).floor('D')
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).floor('D')

    df = price_df.join(macro_df, how='left')
    df[macro_df.columns] = df[macro_df.columns].ffill()
    df.dropna(inplace=True)
    return df

def create_features(df):
    print("Engineering features (TOTAL QUARANTINE MODE)...")
    data = df.copy()
    
    # 1. Target: Volatilidade REALMENTE futura (T+2 a T+7)
    # Aumentamos o shift para garantir 0% de overlap com o fecho de hoje
    data['Target'] = data['Return'].rolling(window=5).std().shift(-7)
    
    # 2. Features Técnicas com LAG de 2 dias (Segurança máxima)
    for window in [5, 10, 20]:
        data[f'Vol_Lag{window}'] = data['Return'].rolling(window=window).std().shift(2)
        
    for lag in [2, 6, 11]: # Lags aumentados para evitar autocorrelação imediata
        data[f'Ret_Lag{lag}'] = data['Return'].shift(lag)
        
    # 3. Macro com Lag de 2 dias
    data['Fed_Diff'] = data['FEDFUNDS'].diff().shift(2)
    data['US10Y_Chg'] = data['US10Y'].diff().shift(2)

    data.dropna(inplace=True)
    
    # Guardamos os retornos para o backtest mas tiramos do treino
    data['Backtest_Return'] = data['Return'] 
    
    exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Backtest_Return']
    feature_cols = [c for c in data.columns if c not in exclude]
    
    return data, feature_cols

def walk_forward(df, features, train_years=5, test_months=6):
    print(f"Starting Walk-Forward | Train: {train_years}y, Test: {test_months}m")
    
    train_window = int(train_years * 252)
    test_window = int(test_months * 21)
    target_horizon = 5  # Purging gap
    
    def process_window(i):
        # Janelas com Embargo de 10 dias (Segurança Extra)
        train_start = i - train_window
        train_end = i - 10 
        test_start = i
        test_end = min(i + test_window, len(df))
        
        train_data = df.iloc[train_start:train_end].copy()
        test_data = df.iloc[test_start:test_end].copy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[features])
        X_test = scaler.transform(test_data[features])
        
        # HMM apenas com dados defasados
        try:
            hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            # Usamos o lag do retorno, não o retorno atual
            hmm_input = train_data[['Ret_Lag2']].values 
            hmm.fit(hmm_input)
            
            X_train = np.column_stack([X_train, hmm.predict_proba(train_data[['Ret_Lag2']].values)])
            X_test = np.column_stack([X_test, hmm.predict_proba(test_data[['Ret_Lag2']].values)])
        except: pass

        rf = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=50, random_state=42)
        rf.fit(X_train, train_data['Target'])
        
        threshold = np.percentile(rf.predict(X_train), 70)
        signals = np.where(rf.predict(X_test) < threshold, 1, 0)
        
        return list(signals), list(test_data['Backtest_Return']), list(test_data.index)

    window_indices = range(train_window, len(df) - test_window, test_window)
    results_list = Parallel(n_jobs=-1)(delayed(process_window)(i) for i in tqdm(window_indices))
    
    signals, returns, dates = [], [], []
    for res in results_list:
        if res:
            signals.extend(res[0]); returns.extend(res[1]); dates.extend(res[2])

    # --- Strategy Reconstruction ---
    res_df = pd.DataFrame({'Actual_Return': returns, 'Signal': signals}, index=dates)
    # Signal is applied to the NEXT day return
    res_df['Strategy_Return'] = (res_df['Signal'].shift(1) * res_df['Actual_Return']).fillna(0)
    # Trading Costs: 0.05% per switch
    res_df['Net_Return'] = res_df['Strategy_Return'] - (res_df['Signal'].diff().abs().fillna(0) * 0.0005)
    
    # Metrics
    sharpe = (res_df['Net_Return'].mean() / res_df['Net_Return'].std()) * np.sqrt(252)
    st_cum = (1 + res_df['Net_Return']).cumprod()
    bh_cum = (1 + res_df['Actual_Return']).cumprod()

    print("\n" + "="*40)
    print(f" BLINDED RESULTS: {ASSET_TICKER}")
    print(f" Sharpe Ratio: {sharpe:.4f}")
    print(f" Net Return:   {(st_cum.iloc[-1]-1):.2%}")
    print(f" Max Drawdown: {(st_cum / st_cum.cummax() - 1).min():.2%}")
    print("="*40)

    plt.figure(figsize=(12, 6))
    plt.plot(st_cum, label='Hidden Markov', color='dodgerblue')
    plt.plot(bh_cum, label='Buy & Hold', color='gray', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

    return res_df

def run_mcpt(target_results, iterations=1000):
    print(f"\n🎲 Running MCPT ({iterations} permutations)...")
    actual_rets = target_results['Actual_Return'].values
    signals = target_results['Signal'].values
    
    real_sharpe = (target_results['Net_Return'].mean() / target_results['Net_Return'].std()) * np.sqrt(252)
    
    def get_shuffled_sharpe():
        shuffled = np.random.permutation(signals)
        # Shifted returns logic
        s_rets = np.roll(shuffled, 1) * actual_rets
        s_rets[0] = 0
        costs = np.abs(np.diff(shuffled, prepend=shuffled[0])) * 0.0015
        net = s_rets - costs
        return (net.mean() / net.std()) * np.sqrt(252) if net.std() != 0 else 0

    null_dist = Parallel(n_jobs=-1)(delayed(get_shuffled_sharpe)() for _ in range(iterations))
    p_val = (sum(1 for s in null_dist if s >= real_sharpe) + 1) / (iterations + 1)
    print(f"MCPT P-Value: {p_val:.5f}")

if __name__ == '__main__':
    prices = get_price_data()
    macro = get_macro_data()
    
    if not prices.empty and not macro.empty:
        merged = merge_data(prices, macro)
        data, feature_cols = create_features(merged)
        results = walk_forward(data, feature_cols)
        if not results.empty:
            run_mcpt(results)