import pandas as pd
import yfinance as yf
import numpy as np
import time

# --- Configuration ---
INPUT_FILE = r"monte_carlo_results.csv"
OUTPUT_FILE = r"valuation_comparison.csv"

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run monte_carlo_fair_value.py first.")
        return

    print(f"Loaded {len(df)} records from {INPUT_FILE}.")
    
    # 2. Fetch live prices
    tickers = df['Ticker'].dropna().unique().tolist()
    
    # Batch download is faster
    print("Fetching live prices from Yahoo Finance...")
    
    # Use valid tickers only (basic check)
    valid_tickers = [t for t in tickers if isinstance(t, str) and t.isalpha()]
    
    # Split into chunks if too many tickers (yfinance can handle hundreds, but 390 is fine)
    try:
        data = yf.download(valid_tickers, period="1d", progress=True)['Close']
        if data.empty:
             print("No price data returned.")
             return
             
        # If single ticker, it returns Series, if multiple DataFrame
        # Get latest price (iloc[-1])
        current_prices = data.iloc[-1]
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return

    # Add Price column
    # Handle the fact that current_prices might be Series with Ticker index
    
    results = []
    
    print("Comparing Fair Value vs Market Price...")
    
    for index, row in df.iterrows():
        ticker = row['Ticker']
        avg_fv = row['Average Fair Value']
        p10_fv = row['10th Percentile Value']
        p90_fv = row['90th Percentile Value']
        
        # Get Price
        price = np.nan
        if ticker in current_prices.index:
            price = current_prices[ticker]
        else:
             # Try individual fetch if missed (rare if valid)
            pass
            
        # If price is valid
        if not pd.isna(price) and price > 0:
            margin_of_safety = (avg_fv - price) / price
            
            # Additional logic: Risk/Reward
            # Upside to 90th percentile vs Downside to 10th percentile?
            # Or just upside to avg.
            
            results.append({
                'Ticker': ticker,
                'Current Price': round(price, 2),
                'Fair Value (Avg)': avg_fv,
                'Fair Value (Conservative)': p10_fv,
                'Fair Value (Aggressive)': p90_fv,
                'Margin of Safety (%)': round(margin_of_safety * 100, 2),
                'Upside (%)': round(((avg_fv - price)/price)*100, 2)
            })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Sort by Margin of Safety (Best opportunities)
    comparison_df.sort_values(by='Margin of Safety (%)', ascending=False, inplace=True)
    
    # Save to CSV
    comparison_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Comparison saved to {OUTPUT_FILE}")
    
    # Print Top 10
    print("\nTop 10 Investment Opportunities (by Margin of Safety):")
    print(comparison_df[['Ticker', 'Current Price', 'Fair Value (Avg)', 'Margin of Safety (%)']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
