import pandas as pd
import numpy as np
from ib_insync import *
import datetime
import pytz
import os

# =========================
# CONFIG
# =========================
SYMBOL = 'ASML'
CURRENCY = 'USD'
TIMEZONE = 'Europe/Amsterdam'

HOST = '127.0.0.1'
PORT = 7497          # Paper trading
CLIENT_ID = 7

CACHE_FILE = 'ASML_1min_data.csv'

# =========================
# IB DATA MANAGER
# =========================
class IBDataManager:
    def __init__(self):
        self.ib = IB()
        self.contract = Stock(
            symbol=SYMBOL,
            exchange='SMART',
            currency=CURRENCY,
            primaryExchange='NASDAQ'
        )

    def connect(self):
        self.ib.connect(HOST, PORT, clientId=CLIENT_ID)
        self.ib.qualifyContracts(self.contract)
        print("Connected to IB")

    def disconnect(self):
        self.ib.disconnect()

    def fetch_data(self, years=2):
        if os.path.exists(CACHE_FILE):
            print("Loading cached data...")
            df = pd.read_csv(CACHE_FILE, parse_dates=['date'], index_col='date')
            df.index = df.index.tz_convert(TIMEZONE)
            return df

        print("Fetching historical data from IB...")

        end_time = datetime.datetime.now(pytz.UTC)
        start_time = end_time - datetime.timedelta(days=365 * years)

        all_bars = []
        curr_end = end_time
        chunk_days = 2

        while curr_end > start_time:
            if curr_end.weekday() >= 5:
                curr_end -= datetime.timedelta(days=1)
                continue

            end_str = curr_end.strftime('%Y%m%d %H:%M:%S UTC')
            print(f"Requesting chunk ending {end_str}")

            try:
                bars = self.ib.reqHistoricalData(
                    self.contract,
                    endDateTime=end_str,
                    durationStr=f'{chunk_days} D',
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=False,
                    formatDate=1
                )
            except Exception as e:
                print("IB error:", e)
                curr_end -= datetime.timedelta(days=chunk_days)
                continue

            if not bars:
                curr_end -= datetime.timedelta(days=chunk_days)
                continue

            all_bars.extend(bars)
            curr_end = bars[0].date.replace(tzinfo=pytz.UTC)
            self.ib.sleep(2)

        if not all_bars:
            raise RuntimeError("No data retrieved from IB")

        df = util.df(all_bars)
        df.sort_values('date', inplace=True)
        df.drop_duplicates(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize('America/New_York')

        df.index = df.index.tz_convert(TIMEZONE)
        df.to_csv(CACHE_FILE)

        print(f"Saved {len(df)} bars to cache")
        return df

# =========================
# STRATEGY
# =========================
class StrategyLogic:
    def __init__(self, thresholds, slippage=0.0004):
        self.thresholds = thresholds
        self.slippage = slippage

    def run(self, df, entry_offset=0):
        results = []
        days = sorted(df.index.normalize().unique())

        for i in range(1, len(days)):
            curr_day = days[i]
            prev_day = days[i - 1]

            day_data = df[df.index.normalize() == curr_day]
            prev_data = df[df.index.normalize() == prev_day]

            if day_data.empty or prev_data.empty:
                continue

            prev_close = prev_data.iloc[-1]['close']

            ts_pre = pd.Timestamp(
                f"{curr_day.date()} 14:29",
                tz=TIMEZONE
            )

            if ts_pre not in day_data.index:
                continue

            p_pre = day_data.loc[ts_pre]['close']
            pre_return = (p_pre - prev_close) / prev_close

            ts_entry = pd.Timestamp(
                f"{curr_day.date()} 14:30",
                tz=TIMEZONE
            ) + pd.Timedelta(minutes=entry_offset)

            if ts_entry not in day_data.index:
                continue

            p_entry = day_data.loc[ts_entry]['open']

            for th in self.thresholds:
                if abs(pre_return) < th:
                    continue

                direction = 1 if pre_return > 0 else -1

                exits = {
                    '5m': '14:35',
                    '10m': '14:40',
                    '15m': '14:45'
                }

                for label, time_str in exits.items():
                    ts_exit = pd.Timestamp(
                        f"{curr_day.date()} {time_str}",
                        tz=TIMEZONE
                    )

                    if ts_exit not in day_data.index:
                        continue

                    p_exit = day_data.loc[ts_exit]['open']

                    gross = (
                        (p_exit - p_entry) / p_entry
                        if direction == 1
                        else (p_entry - p_exit) / p_entry
                    )

                    net = gross - self.slippage

                    results.append({
                        'date': curr_day.date(),
                        'threshold': th,
                        'exit': label,
                        'direction': 'Long' if direction == 1 else 'Short',
                        'gross_return': gross,
                        'net_return': net
                    })

        return pd.DataFrame(results)

# =========================
# MAIN
# =========================
def main():
    dm = IBDataManager()
    dm.connect()
    df = dm.fetch_data(years=0.05)
    dm.disconnect()

    strategy = StrategyLogic(
        thresholds=[0.002, 0.004, 0.006],
        slippage=0.0004
    )

    results = strategy.run(df)

    if results.empty:
        print("No trades generated.")
        return

    summary = results.groupby(['threshold', 'exit']).agg(
        Trades=('net_return', 'count'),
        WinRate=('net_return', lambda x: (x > 0).mean()),
        AvgReturn=('net_return', 'mean')
    )

    print("\n=== RESULTS ===")
    print(summary)

    results.to_csv('ASML_Backtest_Results.csv', index=False)
    print("\nSaved ASML_Backtest_Results.csv")

if __name__ == "__main__":
    main()
