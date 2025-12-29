# ============================================================
# data_handler.py — Final Multi-Region Version
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime
import pandas as pd
from alpaca_trade_api.rest import REST

# Imports from data_loader.py
from market_data_loader.data_loader import (
    get_market_data,
    get_macro_data,
    build_macro_variables
)
from market_data_loader.passwords import ALPACA_API_KEY, ALPACA_API_SECRET


# ============================================================
# 1. Abstract Base Class
# ============================================================

class DataHandler(ABC):
    """Abstract base class for data handling"""

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> pd.Series:
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        pass

    @abstractmethod
    def update_bars(self) -> None:
        pass


# ============================================================
# 2. Local Data: Historical CSV Files
# ============================================================

class HistoricCSVDataHandler(DataHandler):
    """Handles historical data stored in local CSV files"""

    def __init__(self, csv_dir: str, symbols: List[str]):
        self.csv_dir = csv_dir
        self.symbols = symbols
        self.data: Dict[str, pd.DataFrame] = {}
        self.latest_data: Dict[str, List[pd.Series]] = {s: [] for s in symbols}
        self._load_data()

    def _load_data(self) -> None:
        """Loads historical data from CSV files"""
        for symbol in self.symbols:
            file_path = f"{self.csv_dir}/{symbol}.csv"
            df = pd.read_csv(file_path, index_col="DATE", parse_dates=True)
            df.sort_index(inplace=True)
            self.data[symbol] = df
            self.latest_data[symbol] = [df.iloc[-1]]

    def get_latest_bar(self, symbol: str) -> pd.Series:
        return self.latest_data[symbol][-1]

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        return self.data[symbol].iloc[-N:].apply(lambda r: r, axis=1).tolist()

    def update_bars(self) -> None:
        """Simulates advancing one step in the local dataset"""
        for symbol in self.symbols:
            if not self.data[symbol].empty:
                latest_bar = self.data[symbol].iloc[-1]
                self.latest_data[symbol].append(latest_bar)
                self.data[symbol] = self.data[symbol].iloc[:-1]


# ============================================================
# 3. Online Data: Market + Macroeconomic (Multi-Region)
# ============================================================

class OnlineDataHandler(DataHandler):
    """Fetches market and macroeconomic data from Alpaca, Yahoo, and FRED"""

    def __init__(
        self,
        tickers: List[str],
        start: str = "2023-01-01",
        end: str = datetime.now().strftime("%Y-%m-%d"),
        timeframe: str = "1Day",
        macro_region: str = "us"  # 'us' | 'europe' | 'japan'
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.timeframe = timeframe
        self.macro_region = macro_region.lower()

        # Determine resampling rule
        self.resample = (
            "D" if timeframe == "1Day"
            else "W" if timeframe == "1Week"
            else "M" if timeframe == "1Month"
            else "Y" if timeframe == "1Year"
            else None
        )
        if self.resample is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Alpaca API client
        self.api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://data.alpaca.markets")

        # Data containers
        self.price_all_dict: Dict[str, pd.DataFrame] = {}
        self.price_close_df: pd.DataFrame | None = None
        self.return_close_df: pd.DataFrame | None = None
        self.macro_data: pd.DataFrame | None = None
        self.raw_macro: pd.DataFrame | None = None
        self.latest_data: Dict[str, List[pd.Series]] = {t: [] for t in tickers}

    # ---- Market Data ----
    def _load_market_data(self):
        """Fetches market data from Yahoo/Alpaca"""
        (
            self.return_close_df,
            self.price_close_df,
            self.price_all_dict
        ) = get_market_data(
            api=self.api,
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            timeframe=self.timeframe,
            feed="iex",
            verbose=True
        )

    # ---- Macro Data ----
    def _load_macro_data(self):
        """Fetches macroeconomic data according to the selected region"""
        self.raw_macro = get_macro_data(
            region=self.macro_region,
            start=self.start,
            end=self.end,
            resample_rule=self.resample,
            verbose=True
        )
        built_macro = build_macro_variables(
            self.raw_macro,
            resample_rule=self.resample,
            verbose=True
        )
        self.macro_data = pd.concat([self.raw_macro, built_macro], axis=1)
        self.macro_data = self.macro_data.dropna(axis=1)

    # ---- Public Interface ----
    def load_data(self):
        print(f"- Loading market data ({self.timeframe})...")
        self._load_market_data()
        print(f"- Loading macroeconomic data ({self.macro_region.upper()})...")
        self._load_macro_data()
        print("Data successfully loaded.\n")

    # ---- Interface from DataHandler ----
    def get_latest_bar(self, symbol: str) -> pd.Series:
        df = self.price_all_dict.get(symbol)
        if df is None or df.empty:
            raise KeyError(f"No market data for {symbol}")
        return df.iloc[-1]

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        df = self.price_all_dict.get(symbol)
        if df is None or df.empty:
            raise KeyError(f"No market data for {symbol}")
        return df.iloc[-N:].apply(lambda row: row, axis=1).tolist()

    def update_bars(self) -> None:
        """Simulates moving to the next market bar"""
        for symbol, df in self.price_all_dict.items():
            if not df.empty:
                latest_bar = df.iloc[-1]
                self.latest_data[symbol].append(latest_bar)
                self.price_all_dict[symbol] = df.iloc[:-1]

    # ---- Macroeconomic Snapshot ----
    def get_latest_macro_snapshot(self) -> pd.Series:
        """Returns the latest macroeconomic data snapshot"""
        return self.macro_data.iloc[-1]


# ============================================================
# 4. Example Usage
# ============================================================

if __name__ == "__main__":
    # --- Example: US Market + Macro Data ---
    tickers = ["AAPL", "SPY", "TLT"]
    handler_us = OnlineDataHandler(tickers, macro_region="us")
    handler_us.load_data()

    print("\nLatest bar for AAPL:")
    print(handler_us.get_latest_bar("AAPL"))

    print("\nLatest US macroeconomic variables:")
    print(handler_us.get_latest_macro_snapshot())

    # --- Example: Europe Macro Data ---
    handler_eu = OnlineDataHandler(tickers=["LVMH.PA", "BNP.PA"], macro_region="europe")
    handler_eu.load_data()

    print("\nLatest European macroeconomic variables:")
    print(handler_eu.get_latest_macro_snapshot())

    # --- Example: Local CSV Data ---
    csv_handler = HistoricCSVDataHandler("data/", ["AAPL"])
    print("\nLatest local bar for AAPL:")
    print(csv_handler.get_latest_bar("AAPL"))
