
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import concurrent.futures
import itertools
import random
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1. STRATEGY LOGIC
# --------------------------------------------------------------------------------

class TrendStrategy:
    """
    Encapsulates the Dual SMA Trend strategy logic.
    """
    def __init__(self, short_window, long_window, threshold=50):
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold

    def generate_signals(self, prices):
        """
        Generates signals: 1 (Long), -1 (Short), 0 (Cash).
        prices: pd.Series of Close prices.
        """
        if len(prices) < self.long_window:
            return pd.Series(0, index=prices.index)

        sma_short = prices.rolling(window=self.short_window).mean()
        sma_long = prices.rolling(window=self.long_window).mean()
        spread = sma_short - sma_long
        
        # Vectorized signal generation
        # Initialize with 0
        regime = pd.Series(0, index=prices.index)
        
        # Long condition
        regime = np.where(spread > self.threshold, 1, regime)
        
        # Short condition
        regime = np.where(spread < -self.threshold, -1, regime)
        
        # The prompt implies: "Wait (park in cash) ... within a range of +/- SD".
        # So we don't hold position if spread drops back to neutral.
        # Strict interpretation:
        # > 50 -> 1
        # < -50 -> -1
        # else -> 0
        
        return pd.Series(regime, index=prices.index)

# --------------------------------------------------------------------------------
# 2. BACKTEST ENGINE
# --------------------------------------------------------------------------------

class BacktestEngine:
    @staticmethod
    def run_backtest(prices, signals):
        """
        Calculates returns and basic metrics.
        prices: pd.Series
        signals: pd.Series (1, -1, 0)
        """
        market_returns = np.log(prices / prices.shift(1)).fillna(0)
        
        # Shift signals to avoid lookahead bias
        positions = signals.shift(1).fillna(0)
        
        strategy_returns = positions * market_returns
        
        return strategy_returns, positions

    @staticmethod
    def calculate_metrics(daily_returns, positions=None):
        """
        Computes performance metrics.
        """
        if daily_returns.empty:
            return {}

        days = len(daily_returns)
        if days == 0:
            return {}
            
        # Cumulative Return
        cum_ret = np.exp(daily_returns.cumsum())
        total_ret = cum_ret.iloc[-1] - 1
        
        # CAGR
        years = days / 252
        cagr = (cum_ret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        vol = daily_returns.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if vol > 0 else 0
        
        # Drawdown
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # % Time in Cash
        if positions is not None:
            cash_pct = (positions == 0).sum() / days
        else:
            cash_pct = 0.0
            
        # Number of trades (changes in position)
        if positions is not None:
            trades = positions.diff().abs().sum()
        else:
            trades = 0

        return {
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Total_Return': total_ret,
            'Volatility': vol,
            'Cash_Pct': cash_pct,
            'Trades': trades
        }

# --------------------------------------------------------------------------------
# 3. OPTIMIZATION (GRID SEARCH & WFO)
# --------------------------------------------------------------------------------

def evaluate_params(args):
    """
    Worker function for parallel grid search.
    args: (short_w, long_w, prices_series)
    """
    short_w, long_w, prices = args
    
    # Sanity check
    if short_w >= long_w:
        return None

    strat = TrendStrategy(short_w, long_w)
    signals = strat.generate_signals(prices)
    returns, positions = BacktestEngine.run_backtest(prices, signals)
    metrics = BacktestEngine.calculate_metrics(returns, positions)
    
    metrics['Short'] = short_w
    metrics['Long'] = long_w
    
    return metrics

class GridSearchOptimizer:
    def __init__(self, price_data, short_range, long_range, max_workers=6):
        self.price_data = price_data # Should be a single Series (e.g. Portfolio or Index)
        self.short_range = short_range
        self.long_range = long_range
        self.max_workers = max_workers

    def run(self):
        """
        Systematic Grid Search.
        """
        combinations = []
        for s in self.short_range:
            for l in self.long_range:
                if s < l:
                    combinations.append((s, l, self.price_data))
        
        print(f"  > Evaluating {len(combinations)} parameter combinations...")
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Chunking to reduce overhead if many combinations
            chunk_size = max(1, len(combinations) // (self.max_workers * 4))
            for res in executor.map(evaluate_params, combinations, chunksize=chunk_size):
                if res:
                    results.append(res)
                    
        return pd.DataFrame(results)

class WalkForwardAnalyzer:
    def __init__(self, prices, train_years=4, test_years=1, step_years=1):
        self.prices = prices
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years

    def run(self):
        """
        Rolling Walk-Forward Analysis.
        """
        start_date = self.prices.index[0]
        end_date = self.prices.index[-1]
        
        current_start = start_date
        
        oos_results = []
        best_params_history = []
        
        print(f"\nStarting Walk-Forward Optimization ({self.train_years}y Train / {self.test_years}y Test)...")

        while True:
            # Define Windows
            train_end = current_start + timedelta(days=self.train_years*365)
            test_end = train_end + timedelta(days=self.test_years*365)
            
            if train_end > end_date:
                break
            
            # Slice Data
            train_data = self.prices.loc[current_start:train_end]
            test_data = self.prices.loc[train_end:test_end] # Overlap on boundary is minimal
            
            if test_data.empty:
                break
                
            print(f"  Window: Train {current_start.date()}->{train_end.date()} | Test ->{test_end.date()}")
            
            # 1. OPTIMIZE (In-Sample)
            # Reduced range for speed in demo, user specified 10-80, 150-400
            # We step by 5 to keep it reasonable but dense enough
            short_rng = range(10, 81, 5) 
            long_rng = range(150, 401, 10)
            
            optimizer = GridSearchOptimizer(train_data, short_rng, long_rng, max_workers=6)
            df_results = optimizer.run()
            
            if df_results.empty:
                print("    No valid parameters found.")
                current_start += timedelta(days=self.step_years*365)
                continue

            # Robustness Selection:
            # 1. Filter for positive Sharpe
            # 2. Rank by Sharpe
            # 3. Select top parameter set
            # (Ideally we'd look for parameter plateaus, but top Sharpe is a standard proxy)
            
            df_results = df_results[df_results['Trades'] > 5] # Min trades constraint
            if df_results.empty:
                 print("    No parameters with sufficient trades.")
                 current_start += timedelta(days=self.step_years*365)
                 continue

            best_param = df_results.sort_values('Sharpe', ascending=False).iloc[0]
            best_s, best_l = int(best_param['Short']), int(best_param['Long'])
            
            best_params_history.append({
                'Date': train_end,
                'Short': best_s,
                'Long': best_l,
                'IS_Sharpe': best_param['Sharpe']
            })
            
            print(f"    Best Params: Short={best_s}, Long={best_l} (Sharpe: {best_param['Sharpe']:.2f})")
            
            # 2. VALIDATE (Out-of-Sample)
            # Need some buffer before test data to calculate MA
            # Retrieve enough data prior to test_start
            buffer_size = best_l + 10
            # Get data slice starting from (train_end - buffer) to test_end
            start_idx = self.prices.index.get_loc(train_data.index[-1])
            # safe fetch
            adjusted_start_idx = max(0, start_idx - buffer_size)
            
            # We actually just need to run the strategy on the whole price series and slice the returns
            # This is safer for MA calculation
            
            strat = TrendStrategy(best_s, best_l)
            all_signals = strat.generate_signals(self.prices)
            all_returns, _ = BacktestEngine.run_backtest(self.prices, all_signals)
            
            oos_returns = all_returns.loc[train_end:test_end]
            oos_results.append(oos_returns)
            
            # Move forward
            current_start += timedelta(days=self.step_years*365)

        if not oos_results:
            return None, pd.DataFrame()
            
        full_oos_returns = pd.concat(oos_results)
        # Remove duplicates if any overlap
        full_oos_returns = full_oos_returns[~full_oos_returns.index.duplicated(keep='first')]
        
        return full_oos_returns, pd.DataFrame(best_params_history)

# --------------------------------------------------------------------------------
# 4. MONTE CARLO SIMULATION
# --------------------------------------------------------------------------------

class MonteCarloSimulator:
    def __init__(self, returns_series, num_sims=1000):
        self.returns = returns_series
        self.num_sims = num_sims

    def run(self):
        """
        Reshuffle trades/returns and add noise.
        """
        print(f"\nRunning {self.num_sims} Monte Carlo Simulations...")
        
        daily_rets = self.returns.dropna()
        if daily_rets.empty:
            return None

        sim_results = []
        
        # We simulate equity curves
        
        # Convert to roughly independent trade chunks or just shuffle daily returns?
        # Standard approach for simple backtest: Shuffle daily returns (ignores volatility clustering, but simple)
        # Better: Shuffle actual trades. But we have continuous returns.
        # Let's shuffle Daily Returns for simplicity in this context, 
        # OR if we had a trade list, shuffle that. Here we only passed returns.
        # We will keep it simple: Bootstrap daily returns with replacement.
        
        n_days = len(daily_rets)
        
        final_values = []
        max_drawdowns = []
        
        plt.figure(figsize=(10, 6))
        
        for i in range(self.num_sims):
            # Bootstrap with replacement for robustness check
            # This allows us to see the distribution of potential final outcomes (Final Values),
            # satisfying "Evaluate distribution of returns" and "Probability of Ruin".
            # (Strict reshuffling/permutation would result in identical Final Values for all paths).
            
            shuffled_rets = daily_rets.sample(n=n_days, replace=True).values
            
            # Add noise/slippage?
            # Random slippage: e.g., subtract a small random amount from non-zero returns?
            # Let's assume a small drag on every return to simulate exec cost variance
            # noise = np.random.normal(0, 0.0001, n_days) 
            # shuffled_rets += noise
            
            equity_curve = np.exp(np.cumsum(shuffled_rets))
            final_values.append(equity_curve[-1])
            
            # DD
            running_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - running_max) / running_max
            max_drawdowns.append(dd.min())
            
            # Plot first 50 paths
            if i < 50:
                plt.plot(equity_curve, color='gray', alpha=0.1)
                
        # Plot Original
        orig_curve = np.exp(np.cumsum(daily_rets.values))
        plt.plot(orig_curve, color='blue', label='Original OOS')
        
        plt.title('Monte Carlo: Permuted Returns (Ordering Risk)')
        plt.xlabel('Days')
        plt.ylabel('Equity Multiplier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('monte_carlo_paths.png')
        plt.close()
        
        return {
            'final_values': final_values,
            'max_drawdowns': max_drawdowns
        }

# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------

def main():
    # 1. SETUP
    print("="*60)
    print("SYSTEMATIC TREND STRATEGY OPTIMIZATION")
    print("="*60)
    
    # Universal Tickers or Portfolio? 
    # The strategy uses a fixed "SD=50 points" threshold which is calibrated for the S&P 500 Index value.
    # Using normalized prices (approx 1.0) or lower priced ETFs (SPY ~400) would break the logic.
    # We will optimize on the Benchmark (^GSPC) directly to find robust SMA parameters.
    
    tickers = ['LVMHF']
    print(f"Universe: {tickers}")
    
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading data...")
    # Add buffer for MA
    fetch_start = '2008-01-01'
    data = yf.download(tickers, start=fetch_start, end=end_date, progress=False, auto_adjust=True)
    
    # Process data
    portfolio_index = None
    if isinstance(data, pd.DataFrame):
        # Handle yfinance MultiIndex or standard columns
        if 'Close' in data.columns:
            portfolio_index = data['Close']
        elif isinstance(data.columns, pd.MultiIndex):
            # Try to locate Close in levels
            # Often yfinance returns (Price, Ticker) or (Ticker, Price)
            # We assume single ticker here so we can just grab the intersection or iloc
            portfolio_index = data.xs('Close', axis=1, level=-1, drop_level=True) 
        else:
             portfolio_index = data.iloc[:,0] 

    if portfolio_index is None:
        portfolio_index = data.iloc[:,0]

    # Ensure Series
    if isinstance(portfolio_index, pd.DataFrame):
        portfolio_index = portfolio_index.iloc[:,0] # Take first column if still DF
        
    portfolio_index = portfolio_index.dropna()
    portfolio_index.name = "^GSPC"
    
    print(f"Data prepared: {len(portfolio_index)} days (Price range: {portfolio_index.min():.0f}-{portfolio_index.max():.0f})")
    
    # 2. RUN WALK-FORWARD OPTIMIZATION
    # Train 4 years, Test 1 year, Step 1 year
    wfo = WalkForwardAnalyzer(portfolio_index, train_years=5, test_years=1, step_years=1)
    oos_returns, params_history = wfo.run()
    
    if oos_returns is None or oos_returns.empty:
        print("WFO Failed to generate returns.")
        return

    # 3. ANALYZE RESULTS
    print("\n" + "="*60)
    print("WALK-FORWARD RESULTS")
    print("="*60)
    
    metrics = BacktestEngine.calculate_metrics(oos_returns)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")
        
    print("\nParameter Stability:")
    print(params_history)
    
    # Plot WFO Equity Curve
    cumulative = np.exp(oos_returns.cumsum())
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, cumulative.values, label='Walk-Forward Strategy')
    plt.title('Walk-Forward Optimization Equity Curve (OOS)')
    plt.legend()
    plt.grid(True)
    plt.savefig('wfo_equity_curve.png')
    plt.close()
    
    # 4. MONTE CARLO
    mc = MonteCarloSimulator(oos_returns, num_sims=100000)
    mc_results = mc.run()
    
    # MC Stats
    final_vals = np.array(mc_results['final_values'])
    dds = np.array(mc_results['max_drawdowns'])
    
    print("\n" + "="*60)
    print("MONTE CARLO STATISTICS")
    print("="*60)
    print(f"Median Final Value: {np.median(final_vals):.2f}")
    print(f"5th %ile Final Val: {np.percentile(final_vals, 5):.2f} (VaR equivalent)")
    print(f"Median Max DD     : {np.median(dds):.2%}")
    print(f"95th %ile Max DD  : {np.percentile(dds, 5):.2%} (Worst case)") # DD is negative usually, here calculated as negative?
    # Actually my calc: (cum - max)/max is negative. 
    # So 5th percentile is the standard 'worst case' (most negative)
    
    # Dist Plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(final_vals, ax=ax[0], kde=True, color='green')
    ax[0].set_title('Distribution of Final Returns')
    
    sns.histplot(dds, ax=ax[1], kde=True, color='red')
    ax[1].set_title('Distribution of Max Drawdowns')
    
    plt.savefig('monte_carlo_distribution.png')
    plt.close()
    
    print("\nDone. Check valid optimization output files.")

if __name__ == "__main__":
    # Windows hack for multiprocessing
    # required for process pool
    main()
