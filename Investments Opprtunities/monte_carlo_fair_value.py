import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from sec_cik_mapper import StockMapper
import os
import time
import threading
import concurrent.futures

# --- Configuration ---
DB_PATH = r"c:\Users\Utilizador\Desktop\Documentosestudos\Prog\sec_financial_data_20251208_115157.db"
RESULTS_FILE = r"monte_carlo_results.csv"
SIMULATION_ITERATIONS = 10000000 # 10M iterations Lower it if it takes too long 
FORECAST_YEARS = 10
TERMINAL_GROWTH_RATE = 0.020  # 2.0% terminal growth
CPU_THREADS = 8

# --- Helper Functions ---

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)

def get_ticker_cik_mapping():
    """Returns a dictionary mapping CIK (string, 10 digits) to Ticker."""
    try:
        mapper = StockMapper()
        # StockMapper gives ticker -> cik. We need CIK -> ticker.
        # Ensure CIK is 10 digits zero-padded string
        cik_to_ticker = {str(v).zfill(10): k for k, v in mapper.ticker_to_cik.items()}
        return cik_to_ticker
    except Exception as e:
        print(f"Warning: Could not use sec_cik_mapper ({e}). Falling back to manual method if needed.")
        return {}

def fetch_financial_data(conn):
    """Fetches relevant data for all companies from the database."""
    query = """
    SELECT 
        cik, 
        metric_name, 
        MAX(value) as value, -- Assuming we want the latest or aggregate, but better to filter by date
        MAX(end_date) as report_date
    FROM financial_statements 
    WHERE metric_name IN ('Operating Cash Flow', 'Capital Expenditures', 'Net Income', 'EPS Basic')
      AND fiscal_period = 'FY' -- Assuming we want full year data for consistency
    GROUP BY cik, metric_name, fiscal_period, fiscal_year
    ORDER BY cik, end_date DESC
    """
    
    # Updated query to just get the latest FY data for each metric for simplicity in this task
    # A more robust approach would be to get time series to calc CAGR
    query_latest = """
    WITH LatestData AS (
        SELECT 
            cik, 
            metric_name, 
            value,
            end_date,
            ROW_NUMBER() OVER(PARTITION BY cik, metric_name ORDER BY end_date DESC) as rn
        FROM financial_statements
        WHERE metric_name IN ('Operating Cash Flow', 'Capital Expenditures', 'Net Income', 'EPS Basic')
    )
    SELECT cik, metric_name, value 
    FROM LatestData 
    WHERE rn = 1
    """
    
    return pd.read_sql(query_latest, conn)

def calculate_dcf_simulation(fcf, iterations=10000):
    """
    Performs Monte Carlo simulation for DCF.
    Returns: Average Fair Value, 10th %ile, 90th %ile
    """
    # Assumptions for distributions
    # Growth Rate: Normal distribution. Mean=5% (conservative), StdDev=3%
    growth_rates = np.random.normal(0.05, 0.03, iterations)
    
    # Discount Rate (WACC): Normal distribution. Mean=10%, StdDev=1.5%
    discount_rates = np.random.normal(0.10, 0.015, iterations)
    
    # Terminal Multiple implication or Gordon Growth Model for Terminal Value
    # Using Gordon Growth: TV = FCF_n * (1 + g_term) / (WACC - g_term)
    # Warning: WACC must be > g_term. fix invalid rates.
    
    # Clean rates
    discount_rates = np.maximum(discount_rates, TERMINAL_GROWTH_RATE + 0.01) # Ensure WACC > g_term + 1%
    
    # Vectorized approach for speed
    # We simulate 'iterations' paths at once
    
    # Calculate factor: (1+g)/(1+r)
    factors = (1 + growth_rates) / (1 + discount_rates) # shape (iterations,)
    
    # Sum of geometric series formula: sum = a(1-r^n)/(1-r) where a = factor
    # Standard geometric sum: x + x^2 + ... + x^n = x(1-x^n)/(1-x)
    
    geometric_sum = factors * (1 - factors**FORECAST_YEARS) / (1 - factors)
    
    pv_explicit_period = fcf * geometric_sum
    
    # Terminal Value
    fcf_final = fcf * ((1 + growth_rates) ** FORECAST_YEARS)
    tv = fcf_final * (1 + TERMINAL_GROWTH_RATE) / (discount_rates - TERMINAL_GROWTH_RATE)
    
    pv_tv = tv / ((1 + discount_rates) ** FORECAST_YEARS)
    
    total_value = pv_explicit_period + pv_tv
    
    return total_value

def process_ticker_data(row_data):
    """
    Process analysis for a single ticker.
    This function must be at the top level for multiprocessing pickling.
    """
    ticker = row_data['Ticker']
    ocf = row_data['Operating Cash Flow']
    capex = row_data['Capital Expenditures']
    net_income = row_data['Net Income']
    eps = row_data['EPS Basic']
    
    # Calculate FCF
    # CapEx is usually negative in CF statement. If stored as positive, subtract. If negative, add.
    # To be safe: FCF = OCF - abs(CapEx)
    fcf = ocf - abs(capex)
    
    # Calculate Shares
    # Shares = Net Income / EPS
    if eps and eps != 0:
        shares = net_income / eps
    else:
        shares = np.nan
        
    # Validation
    if fcf > 0 and not pd.isna(shares) and shares > 0:
        sim_values = calculate_dcf_simulation(fcf, iterations=SIMULATION_ITERATIONS)
        equity_fair_values = sim_values / shares
        
        avg_fv = np.mean(equity_fair_values)
        p10_fv = np.percentile(equity_fair_values, 10)
        p90_fv = np.percentile(equity_fair_values, 90)
        
        return {
            'Ticker': ticker,
            'Average Fair Value': round(avg_fv, 2),
            '10th Percentile Value': round(p10_fv, 2),
            '90th Percentile Value': round(p90_fv, 2),
            'FCF': fcf,
            'Shares': shares
        }
    return None

def main():
    print("Connecting to database...")
    conn = get_db_connection()
    
    print("Fetching data...")
    df = fetch_financial_data(conn)
    conn.close()
    
    if df.empty:
        print("No data found in database.")
        return

    # Pivot data to have metrics as columns
    df_pivot = df.pivot(index='cik', columns='metric_name', values='value').reset_index()
    
    # Map CIK to Ticker
    print("Mapping CIKs to Tickers...")
    cik_map = get_ticker_cik_mapping()
    
    # Helper to clean CIKs (ensure 10 digits for mapping)
    df_pivot['cik_str'] = df_pivot['cik'].astype(str).str.zfill(10)
    df_pivot['Ticker'] = df_pivot['cik_str'].map(cik_map)
    
    # Fallback: if no ticker, use CIK (or skip, but requirement says "process ALL tickers... valid data")
    missing_tickers = df_pivot['Ticker'].isna().sum()
    if missing_tickers > 0:
        print(f"Warning: {missing_tickers} CIKs could not be mapped to Tickers. using CIK as Ticker for them.")
        df_pivot.loc[df_pivot['Ticker'].isna(), 'Ticker'] = df_pivot['cik_str']

    # Filter valid columns
    required_cols = ['Operating Cash Flow', 'Capital Expenditures', 'Net Income', 'EPS Basic']
    for col in required_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan
            
    # Drop rows with missing essential data for FCF calculation
    df_clean = df_pivot.dropna(subset=['Operating Cash Flow', 'Capital Expenditures'])
    
    companies_count = len(df_clean)
    print(f"Processing {companies_count} companies using 8 workers...")
    
    # Prepare data for parallel processing
    # Convert DataFrame to list of dicts for pickling
    rows_to_process = df_clean.to_dict('records')
    
    results = []
    
    # Start Timer
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound tasks (Monte Carlo)
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_ticker_data, row): row for row in rows_to_process}
        
        # Process results as they complete
        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{companies_count}...", end='\r')
                
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                ticker = futures[future].get('Ticker', 'Unknown')
                print(f"Error processing {ticker}: {e}")

    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds.")

    # Save Results
    results_df = pd.DataFrame(results)
    
    print(f"Generated estimates for {len(results_df)} tickers.")
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
