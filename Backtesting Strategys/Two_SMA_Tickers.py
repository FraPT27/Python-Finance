
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_strategy(tickers, start_date, end_date, period_name):
    print(f"\n{'='*60}")
    print(f"Running Backtest: {period_name} ({start_date} to {end_date})")
    print(f"{'='*60}")

    # Fetch Data
    # Fetching extra data for moving average calculation
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    fetch_start_date = (start_date_obj - timedelta(days=400)).strftime('%Y-%m-%d')
    
    print("Downloading data...")
    # Add SP500 benchmark
    all_tickers = tickers + ['^GSPC']
    data = yf.download(all_tickers, start=fetch_start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
    
    # Process each ticker
    results = {}
    
    for ticker in all_tickers:
        print(f"Processing {ticker}...")
        try:
            # Handle multi-index columns from yfinance if multiple tickers
            if len(all_tickers) > 1:
                df = data[ticker].copy()
            else:
                df = data.copy()

            if df.empty:
                print(f"  No data for {ticker}")
                continue

            # Check for Close column
            if 'Close' not in df.columns:
                 # Check if we have just a series if it's a single ticker download sometimes yf behaves differently
                 # but here with group_by='ticker' it should be a df
                 print(f"  No 'Close' column for {ticker}")
                 continue

            df = df[['Close']].copy()
            
            # 2. Calculate Indicators
            df['15d'] = df['Close'].rolling(window=15).mean()
            df['370d'] = df['Close'].rolling(window=370).mean()
            
            # 3. Calculate Spread and Signal
            # "the 15d trend is for the first time SD points above the 370d trend."
            # "Wait (park in cash) ... within a range of +/- SD points"
            # "Sell signal ... SD points below"
            
            df['15-370'] = df['15d'] - df['370d']
            SD = 50
            
            # Default to 0 (Cash)
            df['Regime'] = 0 
            # Long
            df['Regime'] = np.where(df['15-370'] > SD, 1, df['Regime'])
            # Short
            df['Regime'] = np.where(df['15-370'] < -SD, -1, df['Regime'])
            # Shift signal to avoid lookahead bias
            df['Position'] = df['Regime'].shift(1)
            
            # 4. Calculate Returns
            df['Market Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Strategy Returns'] = df['Position'] * df['Market Returns']
            
            # Filter for the specific backtest period
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]
            
            if df.empty:
                print(f"  No data in backtest period for {ticker}")
                continue

            df[['Market Returns', 'Strategy Returns']] = df[['Market Returns', 'Strategy Returns']].fillna(0)
            
            # Cumulative Returns
            df['Strategy Cumulative'] = df['Strategy Returns'].cumsum().apply(np.exp)
            df['Market Cumulative'] = df['Market Returns'].cumsum().apply(np.exp)
            
            results[ticker] = df
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

    # Display Metrics and Plot
    if not results:
        print("No valid results to display.")
        return

    # Metrics Summary
    metrics = []
    
    for ticker, df in results.items():
        total_return = df['Strategy Cumulative'].iloc[-1] - 1
        market_return = df['Market Cumulative'].iloc[-1] - 1
        
        # Drawdown
        cum_returns = df['Strategy Cumulative']
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe (Approximate annualized)
        daily_returns = df['Strategy Returns']
        sharpe = np.sqrt(370) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        metrics.append({
            'Ticker': ticker,
            'Strategy Return': f"{total_return:.2%}",
            'Market Return': f"{market_return:.2%}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}"
        })
        
    metrics_df = pd.DataFrame(metrics)
    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(14, 8))
    for ticker in tickers:
        if ticker in results:
             plt.plot(results[ticker].index, results[ticker]['Strategy Cumulative'], label=f'{ticker} Strategy')
    
    if '^GSPC' in results:
        plt.plot(results['^GSPC'].index, results['^GSPC']['Market Cumulative'], label='S&P 500 (Benchmark)', linewidth=2, color='black', linestyle='--')

    plt.title(f'Cumulative Returns - {period_name}')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Cumulative Returns')
    
    # Save plot
    filename = f'backtest_{period_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")
    plt.close()


def main():
    # User Input
    default_tickers = ['LVMHF']
    
    try:
        user_input = input(f"Enter tickers separated by comma (default: {', '.join(default_tickers)}): ").strip()
        if not user_input:
            tickers = default_tickers
        else:
            tickers = [t.strip().upper() for t in user_input.split(',')]
            
    except EOFError:
        # Fallback for non-interactive environments
        tickers = default_tickers
        print(f"Using default tickers: {', '.join(tickers)}")

    print(f"\nBacktesting Tickers: {tickers}")

    # Set Dates
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    
    date_5y = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    date_10y = (today - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
    # Run Validations
    print("\n" + "#"*50)
    print("BACKTEST 1: LAST 5 YEARS")
    print("#"*50)
    run_strategy(tickers, date_5y, end_date, "Last 5 Years")
    
    print("\n" + "#"*50)
    print("BACKTEST 2: LAST 10 YEARS")
    print("#"*50)
    run_strategy(tickers, date_10y, end_date, "Last 10 Years")

if __name__ == "__main__":
    main()
