import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_sp500_data(years=5):
    """
    Fetches S&P 500 (^GSPC) data for the specified number of years
    plus a buffer for moving average calculations.
    """
    print(f"Fetching S&P 500 data...")
    # Add buffer for the largest MA (252 days)
    start_date = datetime.now() - timedelta(days=years*365 + 400)
    
    # Fetch data
    ticker = "^GSPC"
    df = yf.download(ticker, start=start_date, progress=False)
    
    # Ensure simplified column names if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Close']].copy()
    
    return df

def calculate_metrics(daily_returns):
    """
    Calculates performance metrics from daily returns series.
    """
    if daily_returns.empty:
        return 0, 0, 0, 0
        
    # Total Return
    total_return = (1 + daily_returns).prod() - 1
    
    # CAGR
    days = len(daily_returns)
    years = days / 252.0
    if years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0
        
    # Annualized Volatility
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    if volatility > 0:
        sharpe = (daily_returns.mean() * 252) / volatility
    else:
        sharpe = 0
        
    # Max Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    return total_return, cagr, sharpe, max_drawdown

def backtest_dual_ma(df, sd_percent):
    """
    Backtests the Dual MA Strategy:
    Trend 1 (Short): 42 days
    Trend 2 (Long): 252 days
    
    Logic:
    Difference = MA_42 - MA_252
    Normalize Diff as % of MA_252 (or Price? Let's use MA_252 as baseline)
    
    Signal:
    - Long:  (MA_42 - MA_252) / MA_252 >  SD
    - Short: (MA_42 - MA_252) / MA_252 < -SD
    - Cash:  Otherwise
    """
    data = df.copy()
    data['MA_42'] = data['Close'].rolling(window=42).mean()
    data['MA_252'] = data['Close'].rolling(window=252).mean()
    
    # Calculate Normalized Difference
    # We normalized by MA_252 to make the SD "percentage points" comparable over time
    data['Diff_Pct'] = (data['MA_42'] - data['MA_252']) / data['MA_252']
    
    # Generate Signals
    # 1: Long
    # -1: Short
    # 0: Cash
    data['Signal'] = 0
    data.loc[data['Diff_Pct'] > sd_percent, 'Signal'] = 1
    data.loc[data['Diff_Pct'] < -sd_percent, 'Signal'] = -1
    
    # Calculate Returns
    data['Market_Return'] = data['Close'].pct_change()
    
    # Strategy Return = Signal(t-1) * Market_Return(t)
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Market_Return']
    
    data.dropna(inplace=True)
    
    # Filter for the last 5 years ONLY
    end_date = data.index.max()
    start_simulation = end_date - timedelta(days=5*365)
    data = data[data.index >= start_simulation]
    
    return data

def main():
    # 1. Get Data
    df_raw = get_sp500_data(years=5)
    if df_raw.empty:
        print("Error: No data fetched.")
        return

    # 2. Optimization Loop (SD from 0% to 5%)
    results = []
    # Test SDs: 0.00, 0.005 (0.5%), 0.01 (1%), ... 0.05
    sds = [x / 1000.0 for x in range(0, 55, 5)] # 0.0% to 5.0% in steps of 0.5%
    
    print(f"Optimizing Dual MA Strategy (SD 0% to 5%)...")
    
    best_sharpe = -np.inf # Optimize for Sharpe this time? Or Return? Let's check Return primarily.
    best_cagr = -np.inf
    best_sd = None
    best_data = None
    
    for sd in sds:
        bt_data = backtest_dual_ma(df_raw, sd)
        
        if bt_data.empty:
            continue
            
        strat_tot, strat_cagr, strat_sharpe, strat_dd = calculate_metrics(bt_data['Strategy_Return'])
        
        results.append({
            'SD_Percent': sd,
            'Total_Return': strat_tot,
            'CAGR': strat_cagr,
            'Sharpe': strat_sharpe,
            'Max_Drawdown': strat_dd
        })
        
        # Optimize for CAGR
        if strat_cagr > best_cagr:
            best_cagr = strat_cagr
            best_sharpe = strat_sharpe # Tracking
            best_sd = sd
            best_data = bt_data

    # 3. Benchmark (Buy & Hold)
    bh_ret = best_data['Market_Return']
    bh_tot, bh_cagr, bh_sharpe, bh_dd = calculate_metrics(bh_ret)

    # 4. Report Results to File
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total_Return', ascending=False)
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("DUAL MA STRATEGY (42d / 252d) - REGIME SWITCHING")
    report_lines.append("Optimization: Threshold (SD) for Long/Cash/Short")
    report_lines.append("="*60)
    report_lines.append(f"{'SD (%)':<10} {'Tot Ret':<10} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10}")
    report_lines.append("-" * 60)
    
    for _, row in results_df.iterrows():
        report_lines.append(f"{row['SD_Percent']:.1%}       {row['Total_Return']:.2%}    {row['CAGR']:.2%}    {row['Sharpe']:.2f}       {row['Max_Drawdown']:.2%}")
        
    report_lines.append("-" * 60)
    report_lines.append("BENCHMARK: BUY & HOLD")
    report_lines.append(f"Total Return: {bh_tot:.2%}")
    report_lines.append(f"CAGR:         {bh_cagr:.2%}")
    report_lines.append(f"Sharpe:       {bh_sharpe:.2f}")
    report_lines.append(f"Max DD:       {bh_dd:.2%}")
    
    report_lines.append("\n" + "="*60)
    report_lines.append(f"WINNER: SD = {best_sd:.1%}")
    report_lines.append(f"Return: {best_cagr:.2%} (Sharpe: {best_sharpe:.2f})")
    report_lines.append("="*60)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open("dual_ma_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # 5. Plotting
    if best_data is not None:
        best_data['Strategy_Eq'] = (1 + best_data['Strategy_Return']).cumprod()
        best_data['Benchmark_Eq'] = (1 + best_data['Market_Return']).cumprod()
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(best_data.index, best_data['Strategy_Eq'], label=f'Dual MA (SD {best_sd:.1%})', color='purple')
        plt.plot(best_data.index, best_data['Benchmark_Eq'], label='Buy & Hold', color='gray', linestyle='--')
        plt.title(f"Dual MA Strategy (42/252) vs S&P 500\nThreshold (SD): {best_sd:.1%} | CAGR: {best_cagr:.1%}")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Regime / Signal
        plt.subplot(2, 1, 2)
        # Plot the Difference %
        plt.plot(best_data.index, best_data['Diff_Pct'], label='MA Diff %', color='blue', linewidth=1)
        # Plot Thresholds
        plt.axhline(y=best_sd, color='green', linestyle='--', label=f'Long Threshold (+{best_sd:.1%})')
        plt.axhline(y=-best_sd, color='red', linestyle='--', label=f'Short Threshold (-{best_sd:.1%})')
        plt.fill_between(best_data.index, best_sd, 1, color='green', alpha=0.1)
        plt.fill_between(best_data.index, -1, -best_sd, color='red', alpha=0.1)
        
        plt.title("Regime Signals: MA Difference vs Thresholds")
        plt.ylabel("MA Diff %")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"Dual_MA_Result_SD{best_sd*100:.0f}.png"
        plt.savefig(filename)
        print(f"📈 Chart saved to: {filename}")
        plt.close()

if __name__ == "__main__":
    main()
