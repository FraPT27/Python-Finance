import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Configuration ---
DB_PATH = r"c:\Users\Utilizador\Desktop\Documentosestudos\Prog\sec_financial_data_20251208_115157.db"
BACKTEST_START_DATE = "2020-01-02"
SNAPSHOT_DATE = "2020-01-01" # Data known BY this date
SIMULATION_ITERATIONS = 50000 # Reduced slightly for speed, still robust
FORECAST_YEARS = 10
TERMINAL_GROWTH_RATE = 0.020 

# --- Monte Carlo Logic (Reused) ---
def calculate_dcf_simulation(fcf, iterations=1000):
    """
    Performs Monte Carlo simulation for DCF.
    Returns: Average Fair Value per Share (Equity Value)
    NOTE: This returns Total Value. Valid Shares must be divided later.
    """
    growth_rates = np.random.normal(0.05, 0.03, iterations) # 5% mean growth, 3% std
    discount_rates = np.random.normal(0.10, 0.015, iterations) # 10% WACC, 1.5% std
    
    # Ensure WACC > g_term
    discount_rates = np.maximum(discount_rates, TERMINAL_GROWTH_RATE + 0.01)

    factors = (1 + growth_rates) / (1 + discount_rates)
    
    # Explicit Period (Geometric Sum)
    geometric_sum = factors * (1 - factors**FORECAST_YEARS) / (1 - factors)
    pv_explicit = fcf * geometric_sum
    
    # Terminal Value
    fcf_final = fcf * ((1 + growth_rates) ** FORECAST_YEARS)
    tv = fcf_final * (1 + TERMINAL_GROWTH_RATE) / (discount_rates - TERMINAL_GROWTH_RATE)
    pv_tv = tv / ((1 + discount_rates) ** FORECAST_YEARS)
    
    total_value = pv_explicit + pv_tv
    return np.mean(total_value)

# --- Data Fetching ---
def get_financials_at_date(conn, snapshot_date):
    """
    Fetches the latest financial data available ON or BEFORE snapshot_date.
    """
    print(f"Fetching financials known as of {snapshot_date}...")
    
    query = """
    WITH RankedFilings AS (
        SELECT 
            cik,
            metric_name,
            value,
            filed_date,
            end_date,
            ROW_NUMBER() OVER(PARTITION BY cik, metric_name ORDER BY filed_date DESC) as rn
        FROM financial_statements
        WHERE filed_date <= ?
          AND metric_name IN ('Operating Cash Flow', 'Capital Expenditures', 'Net Income', 'EPS Basic')
          AND fiscal_period = 'FY' -- Using Annual data for stability
    )
    SELECT cik, metric_name, value
    FROM RankedFilings
    WHERE rn = 1
    """
    
    return pd.read_sql(query, conn, params=(snapshot_date,))

def get_ticker_map(conn):
    # Try to build map from existing data if possible, or use library
    # The DB doesn't strictly have a ticker table unless we look at a helper table
    # Let's use the valid tickers from the file system or library if available.
    # The previous script used sec_cik_mapper.
    try:
        from sec_cik_mapper import StockMapper
        mapper = StockMapper()
        cik_to_ticker = {str(v).zfill(10): k for k, v in mapper.ticker_to_cik.items()}
        return cik_to_ticker
    except:
        return {}

def main():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Get Historical Data
    df_metrics = get_financials_at_date(conn, SNAPSHOT_DATE)
    
    if df_metrics.empty:
        print("No data found before Jan 2020. Check database content.")
        conn.close()
        return

    # Pivot
    df = df_metrics.pivot(index='cik', columns='metric_name', values='value').reset_index()
    
    # Map CIK to Ticker
    cik_map = get_ticker_map(conn)
    df['cik_str'] = df['cik'].astype(str).str.zfill(10)
    df['Ticker'] = df['cik_str'].map(cik_map)
    
    # Filter only valid tickers
    df = df.dropna(subset=['Ticker'])
    
    # 2. Calculate TTM FCF and Shares (As of 2019 FY typically)
    # Ensure cols exist
    for col in ['Operating Cash Flow', 'Capital Expenditures', 'Net Income', 'EPS Basic']:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df.dropna(subset=['Operating Cash Flow', 'Capital Expenditures', 'EPS Basic'])
    
    candidates = []
    
    print(f"Running Valuations for {len(df)} companies...")
    
    for _, row in df.iterrows():
        try:
            ocf = row['Operating Cash Flow']
            capex = abs(row['Capital Expenditures'])
            fcf = ocf - capex
            
            eps = row['EPS Basic']
            net_income = row['Net Income']
            
            if eps == 0 or pd.isna(eps): 
                continue
                
            shares = net_income / eps
            
            if fcf <= 0 or shares <= 0:
                continue
                
            # Run Monte Carlo
            # We iterate fewer times for the backtest search to be faster, or full?
            # User wants "top 20", so accuracy matters.
            total_fv = calculate_dcf_simulation(fcf, iterations=10000)
            fv_per_share = total_fv / shares
            
            candidates.append({
                'Ticker': row['Ticker'],
                'FairValue': fv_per_share,
                'FCF': fcf
            })
        except Exception:
            continue
            
    df_vals = pd.DataFrame(candidates)
    
    if df_vals.empty:
        print("No valid candidates found.")
        conn.close()
        return
        
    # 3. Get Prices on Start Date (2020-01-02)
    print("Fetching 2020 Start Prices...")
    tickers = df_vals['Ticker'].unique().tolist()
    
    # chunking for yfinance
    chunk_size = 500
    start_prices = {}
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            # download 2020-01-02 to 2020-01-10 to ensure we find a trading day
            data = yf.download(chunk, start="2020-01-01", end="2020-01-08", progress=False)['Close']
            
            # If single ticker, it returns Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
                
            # Get first valid price
            for t in chunk:
                if t in data.columns:
                    first_valid = data[t].dropna().first_valid_index()
                    if first_valid:
                        start_prices[t] = data.loc[first_valid, t]
        except Exception as e:
            print(f"Error fetching prices for chunk: {e}")
            
    # Add Price to df
    df_vals['StartPrice'] = df_vals['Ticker'].map(start_prices)
    df_vals = df_vals.dropna(subset=['StartPrice'])
    
    # 4. Calculate Upside & Select Top 20
    df_vals['Upside'] = (df_vals['FairValue'] - df_vals['StartPrice']) / df_vals['StartPrice']
    
    # Filter for realistic upside? (e.g. remove > 500% as potential data errors?)
    # User just said "Top 20". Let's stick to that but maybe cap outliers?
    # Common issue: Low float or bad API data causing massive FCF/Price mismatch.
    # Let's remove Upside > 1000% (10x) to filter garbage data.
    df_vals = df_vals[df_vals['Upside'] < 10.0] 
    
    top_20 = df_vals.sort_values('Upside', ascending=False).head(20)
    
    print("\nTop 20 Picks (Jan 2020):")
    print(top_20[['Ticker', 'FairValue', 'StartPrice', 'Upside']].to_string(index=False))
    
    portfolio_tickers = top_20['Ticker'].tolist()
    
    if not portfolio_tickers:
        print("No portfolio generated.")
        return

    # 5. Backtest Performance (2020 - Present)
    print(f"\nBacktesting Portfolio: {portfolio_tickers}")
    
    start_date = "2020-01-02"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get Portfolio Data
    # auto_adjust=False ensures we get 'Adj Close' explicitly
    try:
        ptf_data_raw = yf.download(portfolio_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if 'Adj Close' in ptf_data_raw.columns:
            ptf_data = ptf_data_raw['Adj Close']
        elif 'Close' in ptf_data_raw.columns:
             ptf_data = ptf_data_raw['Close']
        else:
             print("Error: Could not find Close data in yfinance response.")
             return
             
    except Exception as e:
        print(f"Error downloading portfolio data: {e}")
        return
    
    # Bench Data
    try:
        spy_data_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if 'Adj Close' in spy_data_raw.columns:
            spy_data = spy_data_raw['Adj Close']
        else:
            spy_data = spy_data_raw['Close']
    except Exception:
        print("Error downloading SPY data")
        return
    
    # Normalize to 100
    # Portfolio: Equal Weight Rebalanced? Or Buy & Hold?
    # "back test it againts buy and hold... starting in 2020" usually implies Buy & Hold the basket.
    # So we invest $1000 in each of the 20 stocks.
    
    ptf_norm = ptf_data / ptf_data.iloc[0]
    # Handle NaNs (delistings)?
    # If a stock delists, yfinance usually stops updating. Value assumes 0 or last price?
    # Adj Close handles dividends. Limit NaNs.
    ptf_norm = ptf_norm.fillna(method='ffill') # Assume stuck at last price if delisted (optimistic if bankrupt, but simple)
    
    portfolio_curve = ptf_norm.mean(axis=1) # Equal weight average of returns
    
    if isinstance(spy_data, pd.DataFrame):
        spy_curve = spy_data.iloc[:, 0]
    else:
        spy_curve = spy_data
        
    spy_curve = spy_curve / spy_curve.iloc[0]
    
    # Align dates
    combined = pd.DataFrame({'Portfolio': portfolio_curve, 'SP500': spy_curve}).dropna()
    
    # Metrics
    tot_ret_ptf = combined['Portfolio'].iloc[-1] - 1
    tot_ret_spy = combined['SP500'].iloc[-1] - 1
    
    days = (combined.index[-1] - combined.index[0]).days
    years = days / 365.25
    cagr_ptf = (1 + tot_ret_ptf)**(1/years) - 1
    cagr_spy = (1 + tot_ret_spy)**(1/years) - 1
    
    print("\n" + "="*40)
    print(f"Results (Jan 2020 - {datetime.now().strftime('%b %Y')})")
    print("="*40)
    print(f"Portfolio Total Return: {tot_ret_ptf:.2%}")
    print(f"SP500 Total Return:     {tot_ret_spy:.2%}")
    print(f"Portfolio CAGR:         {cagr_ptf:.2%}")
    print(f"SP500 CAGR:             {cagr_spy:.2%}")
    print("="*40)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(combined.index, combined['Portfolio'], label=f'Top 20 Upside (CAGR {cagr_ptf:.1%})', linewidth=2)
    plt.plot(combined.index, combined['SP500'], label=f'S&P 500 (CAGR {cagr_spy:.1%})', color='gray', linestyle='--')
    plt.title("Valuation Strategy Backtest (Jan 2020 Top Picks)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    outfile = "Backtest_2020_Valuation.png"
    plt.savefig(outfile)
    print(f"Chart saved to {outfile}")
    
    # Save picks
    top_20.to_csv("top_20_picks_2020.csv", index=False)

if __name__ == "__main__":
    main()
