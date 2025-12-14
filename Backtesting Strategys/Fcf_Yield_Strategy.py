import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = r"c:\Users\Utilizador\Desktop\Documentosestudos\Prog\sec_financial_data_20251208_115157.db"
START_CAPITAL = 10000

# Strategy Parameters (Absolute Percentage Points for FCF Yield)
BUY_THRESHOLD = 0.005   # Buy when yield is Median + 0.5%
SELL_THRESHOLD = -0.0075 # Sell when yield is Median - 0.75%

# Ticker Mapping
COMPANY_TICKERS = {
    'Apple Inc.': 'AAPL',
    'AMAZON.COM, INC.': 'AMZN',
    'MICROSOFT CORPORATION': 'MSFT',
    'Alphabet Inc.': 'GOOGL',
    'META PLATFORMS, INC.': 'META'
}

def get_historical_financials(conn, company_name):
    """Reconstructs daily TTM Free Cash Flow history"""
    # Get Quarterly Data
    query = """
    SELECT metric_name, value, end_date, filed_date
    FROM financial_statements 
    WHERE company_name = ? 
    AND value_type = 'Quarterly'
    AND metric_name IN ('Operating Cash Flow', 'Capital Expenditures')
    ORDER BY filed_date ASC
    """
    df = pd.read_sql_query(query, conn, params=(company_name,))
    
    if df.empty:
        return None
        
    df['filed_date'] = pd.to_datetime(df['filed_date'])
    
    # Organize into TTM FCF series indexed by Filed Date
    dates = sorted(df['filed_date'].unique())
    history = []
    
    for date in dates:
        # Data known as of this filing date
        known_data = df[df['filed_date'] <= date]
        
        # Get last 4 quarters for OCF and CapEx
        ocf = known_data[known_data['metric_name'] == 'Operating Cash Flow'].sort_values('end_date').tail(4)
        capex = known_data[known_data['metric_name'] == 'Capital Expenditures'].sort_values('end_date').tail(4)
        
        if len(ocf) == 4 and len(capex) == 4:
            ttm_ocf = ocf['value'].sum()
            ttm_capex = abs(capex['value'].sum()) # Ensure positive for subtraction
            fcf = ttm_ocf - ttm_capex
            
            history.append({
                'date': date,
                'TTM_FCF': fcf
            })
            
    return pd.DataFrame(history).set_index('date')

def calculate_cagr(end_value, start_value, years):
    if years <= 0: return 0
    return (end_value / start_value) ** (1 / years) - 1

def run_backtest(conn, company_name, ticker):
    print(f"\n{'='*60}")
    print(f"Backtesting Strategy for {company_name} ({ticker})")
    print(f"{'='*60}")
    
    # 1. Get Financial Data
    df_fcf = get_historical_financials(conn, company_name)
    if df_fcf is None or df_fcf.empty:
        print("  ❌ Insufficient financial data")
        return None
        
    # 2. Get Market Data (Price + Shares)
    try:
        stock = yf.Ticker(ticker)
        start_date = df_fcf.index.min()
        df_price = stock.history(start=start_date)
        current_shares = stock.info.get('sharesOutstanding')
        
        if df_price.empty or not current_shares:
            print("  ❌ Error fetching market data")
            return None
            
        df_price.index = df_price.index.tz_localize(None)
    except Exception as e:
        print(f"  ❌ Data Error: {e}")
        return None

    # 3. Merge & Calculate Daily Metrics
    # Forward fill financial data (valid until next filing)
    df = pd.merge_asof(
        df_price[['Close']], 
        df_fcf, 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )
    df = df.dropna()
    
    # Calculate Market Cap & FCF Yield
    # Note: Using current split-adjusted shares is an approximation
    df['Market_Cap'] = df['Close'] * current_shares
    df['FCF_Yield'] = df['TTM_FCF'] / df['Market_Cap']
    
    # 4. Calculate Expanding Median (The "Fair Value" Anchor)
    # We use expanding window median to simulate "what we knew then"
    # Min periods = 252 (1 trading year) to return a signal
    df['Median_Yield'] = df['FCF_Yield'].expanding(min_periods=252).median()
    
    # 5. Execute Strategy
    capital = START_CAPITAL
    shares_held = 0
    in_market = False
    
    portfolio_values = []
    buy_signals = []
    sell_signals = []
    
    # For Buy & Hold Comparison
    bh_shares = START_CAPITAL / df['Close'].iloc[0]
    
    for date, row in df.iterrows():
        price = row['Close']
        yield_val = row['FCF_Yield']
        median_yield = row['Median_Yield']
        
        if pd.isna(median_yield):
            portfolio_values.append(capital)
            continue
            
        # Strategy Logic
        # Buy: Undervalued (Yield is High) -> Yield > Median + 0.5%
        # Sell: Overvalued (Yield is Low) -> Yield < Median - 0.75%
        
        buy_threshold = median_yield + BUY_THRESHOLD
        sell_threshold = median_yield + SELL_THRESHOLD 
        
        if not in_market and yield_val >= buy_threshold:
            # BUY
            shares_held = capital / price
            capital = 0
            in_market = True
            buy_signals.append((date, price))
            
        elif in_market and yield_val <= sell_threshold:
            # SELL
            capital = shares_held * price
            shares_held = 0
            in_market = False
            sell_signals.append((date, price))
            
        # Record Value
        current_val = capital + (shares_held * price)
        portfolio_values.append(current_val)
        
    df['Strategy_Value'] = portfolio_values
    df['Buy_Hold_Value'] = df['Close'] * bh_shares
    
    # 6. Calculate Performance Metrics
    years = (df.index[-1] - df.index[0]).days / 365.25
    final_val = df['Strategy_Value'].iloc[-1]
    bh_val = df['Buy_Hold_Value'].iloc[-1]
    
    strat_cagr = calculate_cagr(final_val, START_CAPITAL, years)
    bh_cagr = calculate_cagr(bh_val, START_CAPITAL, years)
    
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()} ({years:.1f} years)")
    print(f"  Initial Capital: ${START_CAPITAL:,.2f}")
    print(f"  Strategy Final:  ${final_val:,.2f} (CAGR: {strat_cagr*100:.2f}%)")
    print(f"  Buy & Hold Final: ${bh_val:,.2f} (CAGR: {bh_cagr*100:.2f}%)")
    print(f"  Total Trades: {len(buy_signals)} Buys, {len(sell_signals)} Sells")
    
    # 7. Plotting
    try:
        plot_backtest(df, company_name, ticker, buy_signals, sell_signals, strat_cagr, bh_cagr)
    except Exception as e:
        print(f"  ⚠️ Could not generate plot: {e}")

    return {
        'Company': company_name,
        'Strategy_CAGR': strat_cagr,
        'BuyHold_CAGR': bh_cagr
    }

def plot_backtest(df, company, ticker, buys, sells, strat_cagr, bh_cagr):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Portfolio Performance
    ax1.plot(df.index, df['Strategy_Value'], label=f'Strategy (CAGR {strat_cagr*100:.1f}%)', color='blue', linewidth=2)
    ax1.plot(df.index, df['Buy_Hold_Value'], label=f'Buy & Hold (CAGR {bh_cagr*100:.1f}%)', color='gray', linestyle='--', alpha=0.7)
    
    # Plot Buy/Sell Markers on Equity Curve
    # Extract dates/prices for cleaner plotting logic if needed, but plotting distinct points works
    # Actually, plotting markers on the price timeline is usually better visually
    # Let's verify: User wants to see "results", usually equity curve.
    
    ax1.set_title(f"{company} ({ticker}) - FCF Yield Strategy Backtest", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: FCF Yield vs Median (The Logic)
    ax2.plot(df.index, df['FCF_Yield'], label='FCF Yield', color='purple', alpha=0.8)
    ax2.plot(df.index, df['Median_Yield'], label='Median Yield (Fair)', color='black', linestyle='--')
    
    # Shade Buy/Sell Zones
    # Buy Zone: Yield > Median + 0.5%
    ax2.fill_between(df.index, df['FCF_Yield'], df['Median_Yield'] + BUY_THRESHOLD, 
                     where=(df['FCF_Yield'] >= df['Median_Yield'] + BUY_THRESHOLD), 
                     color='green', alpha=0.3, label='Undervalued Zone (Buy)')
                     
    # Sell Zone: Yield < Median - 0.75%
    ax2.fill_between(df.index, df['FCF_Yield'], df['Median_Yield'] + SELL_THRESHOLD, 
                     where=(df['FCF_Yield'] <= df['Median_Yield'] + SELL_THRESHOLD), 
                     color='red', alpha=0.3, label='Overvalued Zone (Sell)')
    
    # Plot Trade Markers
    if buys:
        buy_dates, _ = zip(*buys)
        # Find corresponding yields
        buy_yields = df.loc[list(buy_dates)]['FCF_Yield']
        ax2.scatter(buy_dates, buy_yields, marker='^', color='green', s=100, zorder=5)
        
    if sells:
        sell_dates, _ = zip(*sells)
        sell_yields = df.loc[list(sell_dates)]['FCF_Yield']
        ax2.scatter(sell_dates, sell_yields, marker='v', color='red', s=100, zorder=5)

    ax2.set_ylabel("FCF Yield")
    ax2.set_xlabel("Year")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format Y-axis as percentage
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Backtest_{ticker}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  📈 Chart saved to: {filename}")
    plt.close()

def main():
    conn = sqlite3.connect(DB_PATH)
    
    results = []
    
    for company, ticker in COMPANY_TICKERS.items():
        res = run_backtest(conn, company, ticker)
        if res:
            results.append(res)
            
    conn.close()
    
    if results:
        print("\n" + "="*60)
        print(f"{'Company':<20} {'Strategy CAGR':<15} {'Buy & Hold CAGR':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['Company'][:18]:<20} {r['Strategy_CAGR']*100:6.2f}%         {r['BuyHold_CAGR']*100:6.2f}%")
        print("="*60)

if __name__ == "__main__":
    main()
