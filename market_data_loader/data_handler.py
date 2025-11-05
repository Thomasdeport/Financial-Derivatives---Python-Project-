# ============================================================
# data_handler.py — version finale adaptée
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime
import pandas as pd
from alpaca_trade_api.rest import REST

# Import des fonctions depuis ton module data_loader.py
from market_data_loader.data_loader import (
    get_market_data,
    get_macro_data,
    build_macro_variables
)
from market_data_loader.passwords import ALPACA_API_KEY, ALPACA_API_SECRET


# ============================================================
# 1. Classe abstraite : interface commune
# ============================================================

class DataHandler(ABC):
    """Abstract base class for data handling"""

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Returns the latest bar for a given symbol"""
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        """Returns the latest N bars for a given symbol"""
        pass

    @abstractmethod
    def update_bars(self) -> None:
        """Updates the bars to the next time step"""
        pass


# ============================================================
# 2. Données locales : CSV historiques
# ============================================================

class HistoricCSVDataHandler(DataHandler):
    """Handles data from local CSV files"""

    def __init__(self, csv_dir: str, symbols: List[str]):
        self.csv_dir = csv_dir
        self.symbols = symbols
        self.data: Dict[str, pd.DataFrame] = {}
        self.latest_data: Dict[str, List[pd.Series]] = {symbol: [] for symbol in symbols}
        self._load_data()

    def _load_data(self) -> None:
        """Loads data from CSV files into the data dictionary"""
        for symbol in self.symbols:
            file_path = f"{self.csv_dir}/{symbol}.csv"
            df = pd.read_csv(file_path, index_col="DATE", parse_dates=True)
            df.sort_index(inplace=True)
            self.data[symbol] = df
            self.latest_data[symbol] = [df.iloc[-1]]

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Returns the latest bar for a given symbol"""
        return self.latest_data[symbol][-1]

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        """Returns the latest N bars for a given symbol"""
        return self.data[symbol].iloc[-N:].apply(lambda row: row, axis=1).tolist()

    def update_bars(self) -> None:
        """Simulates moving one step forward in time"""
        for symbol in self.symbols:
            if not self.data[symbol].empty:
                latest_bar = self.data[symbol].iloc[-1]
                self.latest_data[symbol].append(latest_bar)
                self.data[symbol] = self.data[symbol].iloc[:-1]


# ============================================================
# 3. Données online : Alpaca + FRED
# ============================================================

class OnlineDataHandler(DataHandler):
    """Loads market and macroeconomic data from Alpaca and FRED APIs"""

    def __init__(
        self,
        tickers: List[str],
        start: str = "2023-01-01",
        end: str = datetime.now().strftime("%Y-%m-%d"),
        timeframe: str = "1Day"
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.timeframe = timeframe

        # Déterminer la règle de resampling à partir du timeframe
        if timeframe == "1Day":
            self.resample = "D"
        elif timeframe == "1Week":
            self.resample = "W"
        elif timeframe == "1Month":
            self.resample = "M"
        elif timeframe == "1Year":
            self.resample = "Y"
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Initialisation du client Alpaca
        self.api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://data.alpaca.markets")

        # Stockage des données
        self.price_all_dict: Dict[str, pd.DataFrame] = {}
        self.price_close_df: pd.DataFrame | None = None
        self.return_close_df: pd.DataFrame | None = None
        self.macro_data: pd.DataFrame | None = None
        self.fred: pd.DataFrame | None = None
        self.latest_data: Dict[str, List[pd.Series]] = {t: [] for t in tickers}

    # ---- Market data ----
    def _load_market_data(self):
        """Fetches market data from Alpaca via get_market_data()"""
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

    # ---- Macro data ----
    def _load_macro_data(self):
        """Fetches and constructs macroeconomic variables from FRED"""
        FRED_CODES = {
            "CPI": "CPIAUCSL",
            "INDPPI": "PPIACO",
            "M1SUPPLY": "M1SL",
            "CCREDIT": "TOTALSL",
            "BMINUSA": "BAA10Y",
            "AAA10Y": "AAA10Y",
            "TB3MS": "TB3MS"
        }

        self.fred = get_macro_data(
            fred_codes=FRED_CODES,
            start=self.start,
            end=self.end,
            resample_rule=self.resample,
            verbose=True
        )

        self.built_macro_data = build_macro_variables(
            self.fred,
            resample_rule=self.resample,
            verbose=True
        )
        self.macro_data = pd.concat([self.fred, self.built_macro_data], axis = 1)

    # ---- Public interface ----
    def load_data(self):
        print("Chargement des données de marché depuis Alpaca...")
        self._load_market_data()
        print("Chargement des données macroéconomiques depuis FRED...")
        self._load_macro_data()
        print("✓ Données chargées avec succès.")

    # ---- Interface héritée de DataHandler ----
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Returns the latest OHLCV bar for a given ticker"""
        df = self.price_all_dict[symbol]
        return df.iloc[-1]
    
    def get_latest_macro(self, symbol: str) -> pd.Series:
        """Returns the latest Macro for a given Symbol"""
        return self.macro_data[symbol].iloc[-1]

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[pd.Series]:
        """Returns the latest N OHLCV bars for a given ticker"""
        df = self.price_all_dict[symbol]
        return df.iloc[-N:].apply(lambda row: row, axis=1).tolist()

    def get_latest_macros(self, symbol: str) -> pd.Series:
        """Returns the latest Macro for a given Symbol"""
        return self.macro_data[symbol].iloc[-N:].apply(lambda row: row, axis=1).tolist()

    def update_bars(self) -> None:
        """Simulates advancing to the next time step"""
        for symbol in self.tickers:
            df = self.price_all_dict[symbol]
            if not df.empty:
                latest_bar = df.iloc[-1]
                self.latest_data[symbol].append(latest_bar)
                self.price_all_dict[symbol] = df.iloc[:-1]


# ============================================================
# 4. Exemple d’utilisation
# ============================================================

if __name__ == "__main__":
    # Exemple : données Alpaca + FRED
    tickers = ["AAPL", "SPY", "TLT"]
    handler = OnlineDataHandler(tickers)
    handler.load_data()

    print("\nDernier bar pour AAPL :")
    print(handler.get_latest_bar("AAPL"))

    print("\nVariables macroéconomiques :")
    print(handler.macro_data.tail())

    # Exemple : données locales
    csv_handler = HistoricCSVDataHandler("data/", ["AAPL"])
    print("\nDernier bar pour AAPL_D :")
    print(csv_handler.get_latest_bar("AAPL"))
