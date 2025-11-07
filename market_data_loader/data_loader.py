# ==========================================
# data_loader.py
# ==========================================
# Module : Data Loader
# Description : Fetches market and macroeconomic data from Yahoo Finance, Alpaca, and FRED.
# ==========================================

from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from alpaca_trade_api.rest import REST, TimeFrame

from market_data_loader.passwords import ALPACA_API_KEY, ALPACA_API_SECRET
from market_data_loader.utils import return_computation


# ==========================================
# 1. Global Parameters
# ==========================================
API_KEY = ALPACA_API_KEY
API_SECRET = ALPACA_API_SECRET
BASE_URL = "https://data.alpaca.markets"
API = REST(API_KEY, API_SECRET, base_url=BASE_URL)

UDL = "AAPL"
TICKERS = [
    UDL, "SPY", "QQQ", "XLK",     # Market and Tech
    "TLT", "VIXY", "GLD", "UUP"   # Macro proxies
]

START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME = "1Day"


# ==========================================
# 2. Alpaca Client
# ==========================================
def alpaca_client(api_key: str, api_secret: str, base_url: str = BASE_URL) -> REST:
    """
    Create an Alpaca REST client instance.
    """
    return REST(api_key, api_secret, base_url=base_url)


# ==========================================
# 3. Market Data Extraction (Yahoo + Alpaca Fallback)
# ==========================================
def _fetch_yfinance_data(ticker: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    interval_map = {"1Day": "1d", "1H": "1h", "1h": "1h", "1Min": "1m", "1min": "1m"}
    interval = interval_map.get(timeframe, "1d")

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("Empty dataframe returned by Yahoo Finance.")

    df.index.name = "DATE"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
        
    df = df.rename(columns=str.lower)
    df["timeframe"] = timeframe
    df["source"] = "Yahoo Finance"
    return df


def _fetch_alpaca_data(api: REST, ticker: str, start: str, end: str, timeframe: str, feed: str) -> pd.DataFrame:
    """Fetch OHLCV data from Alpaca."""
    tf = TimeFrame.Day if timeframe.lower() in ["1day", "day"] else TimeFrame.Hour
    bars = api.get_bars(ticker, timeframe=tf, start=start, end=end, feed=feed).df

    if bars.empty:
        raise ValueError("Empty dataframe returned by Alpaca.")

    df = bars.copy()
    df.index.name = "DATE"
    df = df.rename(columns=str.lower)
    df["timeframe"] = timeframe
    df["source"] = "Alpaca"
    return df


def get_market_data(
    api: REST,
    tickers: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",
    feed: str = "iex",
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for multiple tickers.
    Primary source: Yahoo Finance.
    Fallback: Alpaca API if Yahoo fails.

    Returns:
        retn (pd.DataFrame): computed returns
        px (pd.DataFrame): closing prices
        prices (dict[str, pd.DataFrame]): raw OHLCV per ticker
    """
    prices, close_series = {}, []

    for ticker in tickers:
        df, source = None, None

        # --- Try Yahoo first ---
        try:
            df = _fetch_yfinance_data(ticker, start, end, timeframe)
            source = "Yahoo Finance"
        except Exception as e:
            if verbose:
                print(f"[ WARN] {ticker}: Yahoo Finance failed ({e}). Trying Alpaca...")
            try:
                df = _fetch_alpaca_data(api, ticker, start, end, timeframe, feed)
                source = "Alpaca"
            except Exception as e2:
                print(f"[ ERROR] {ticker}: both sources failed ({e2}). Skipping.")
                continue

        # --- Clean + store ---
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "DATE"

        prices[ticker] = df
        s = df["close"].copy()
        s.name = ticker
        close_series.append(s)

        if verbose:
            print(f"{ticker}: {len(df)} obs from {source} "
                  f"({df.index.min().date()} → {df.index.max().date()}) [{timeframe}]")

    # --- Combine all closing prices ---
    if not close_series:
        raise ValueError("No valid data fetched for any ticker.")

    px = pd.concat(close_series, axis=1).sort_index().dropna(how="all")
    retn = return_computation(px, list(px.columns))

    if verbose:
        print("\n Market data successfully loaded")
        print(f"Tickers: {len(px.columns)} | Period: {px.index.min().date()} → {px.index.max().date()}")
        print(f"Observations: {px.shape[0]} | Timeframe: {timeframe}")

    return retn, px, prices


# ==========================================
# 4. Macro Data 
# ==========================================
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import numpy as np

def get_macro_data(
    region: str = "us",
    start: str = "2020-01-01",
    end: str | None = None,
    resample_rule: str = "D",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED for:
      - 'us': United States
      - 'europe': Euro Area
      - 'japan': Japan
    """

    region = region.lower()

    if region == "us":
        fred_codes = {
            "CPI": "CPIAUCSL",
            "INDPPI": "PPIACO",
            "M1SUPPLY": "M1SL",
            "CCREDIT": "TOTALSL",
            "AAA10Y": "AAA10Y",
            "TB3MS": "TB3MS"
        }

    elif region == "europe":
        fred_codes = {
            "CPI": "CP0000EZ19M086NEST",    # CPI Euro area (Eurostat)
            "INDPPI": "PIEAMP02EZM661N",    # Industrial production
            "M1SUPPLY": "MABMM201EZM189S",  # Broad Money (M3 proxy)
            "CCREDIT": "QEZLOCOODCSXDC",    # Credit to private sector (BIS)
            "AAA10Y": "IRLTLT01EZM156N",    # 10y gov bond yield
            "TB3MS": "IRSTCI01EZM156N"      # 3-month interbank rate
        }

    elif region == "japan":
        fred_codes = {
            "CPI": "JPNCPIALLMINMEI",
            "INDPPI": "JPNPROINDMISMEI",
            "M1SUPPLY": "MYAGM1JPM189S",
            "CCREDIT": "QJPNLOCOODCANQ",
            "AAA10Y": "IRLTLT01JPM156N",
            "TB3MS": "IRSTCI01JPM156N"
        }

    else:
        raise ValueError("Region must be one of: 'us', 'europe', 'japan'")

    # --- fetch each series safely ---
    data = {}
    failed = {}

    for name, code in fred_codes.items():
        try:
            s = web.DataReader(code, "fred", start, end)
            data[name] = s
        except Exception as e:
            msg = str(e).replace("\n", " ")[:100] + "\033[31m [WARN] This message is too long, cannot print it entirely\033[0m"
            print(f"[WARN] {name}: {msg}")


    fred = pd.concat(data.values(), axis=1) if data else pd.DataFrame()
    fred.columns = data.keys()

    # --- compute bond spread ---
    if "AAA10Y" in fred.columns and "TB3MS" in fred.columns:
        fred["BMINUSA"] = fred["AAA10Y"] - fred["TB3MS"]
    else:
        fred["BMINUSA"] = np.nan

    # --- resampling and cleaning ---
    fred = fred.resample(resample_rule).interpolate("time").bfill().ffill()

    if verbose:
        print("\n==============================")
        print(f"Macro data loaded from FRED ({region.upper()})")
        if not fred.empty:
            print(f"Period: {fred.index.min().date()} → {fred.index.max().date()}")
            print(f"{fred.shape[0]} obs, {fred.shape[1]} variables")
        else:
            print(" No macro data loaded.")

        # Summary of success/fail
        loaded = [k for k in fred_codes.keys() if k in fred.columns]
        missing = [k for k in fred_codes.keys() if k not in fred.columns]

        print("\n Successfully loaded:")
        for k in loaded:
            print(f"   - {k}")

        if missing:
            print("\n Failed to load:")
            for k in missing:
                print(f"   - {k}: {failed.get(k, 'unknown error')}")

        print("==============================\n")

    return fred


def build_macro_variables(macro_data: pd.DataFrame, resample_rule: str = "D", verbose: bool = True) -> pd.DataFrame:
    """Construct macroeconomic derived variables safely."""
    macro = pd.DataFrame(index=macro_data.index)

    def safe_logdiff(series):
        if isinstance(series, pd.Series) and series.notna().sum() > 2:
            return np.log(series / series.shift(1)) * 100
        else:
            return pd.Series(np.nan, index=macro_data.index)

    macro["INF"] = safe_logdiff(macro_data.get("CPI"))
    macro["DP"] = safe_logdiff(macro_data.get("INDPPI"))
    macro["DM"] = safe_logdiff(macro_data.get("M1SUPPLY"))
    macro["DC"] = safe_logdiff(macro_data.get("CCREDIT"))
    macro["DS"] = macro_data.get("BMINUSA", pd.Series(np.nan, index=macro_data.index)).diff()
    macro["TS"] = macro_data.get("AAA10Y", pd.Series(np.nan, index=macro_data.index)) - macro_data.get("TB3MS", pd.Series(np.nan, index=macro_data.index))
    macro["DT"] = macro["TS"].diff()

    tb3ms = macro_data.get("TB3MS", pd.Series(np.nan, index=macro.index))
    if resample_rule == "D":
        macro["RF"] = ((1 + tb3ms / 100) ** (1 / 252) - 1) * 100
    elif resample_rule == "W":
        macro["RF"] = ((1 + tb3ms / 100) ** (1 / 52) - 1) * 100
    elif resample_rule == "M":
        macro["RF"] = ((1 + tb3ms / 100) ** (1 / 12) - 1) * 100
    else:
        macro["RF"] = tb3ms / 100

    macro = macro.ffill().bfill()
    macro = macro.dropna(axis=1)
    if verbose:
        print("Macro variables built:")
        print(f"Period: {macro.index.min().date()} → {macro.index.max().date()}")
        print(f"{macro.shape[0]} observations, {macro.shape[1]} variables")

    return macro


# ==========================================
# 5. Main (for standalone testing)
# ==========================================
if __name__ == "__main__":
    api = alpaca_client(API_KEY, API_SECRET)
    returns, prices, raw_prices = get_market_data(api, TICKERS, START_DATE, END_DATE)
    fred_data = get_macro_data(region = "us", start = START_DATE, end = END_DATE, resample_rule = "D")
    macro_vars = build_macro_variables(fred_data)
    macro_vars.to_csv("market_data_macro_daily.csv")
    returns.to_csv("market_data_returns.csv")
