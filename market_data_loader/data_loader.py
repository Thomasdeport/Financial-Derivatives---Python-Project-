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
from binance.client import Client as BinanceClient

from market_data_loader.passwords import ALPACA_API_KEY, ALPACA_API_SECRET, BINANCE_API_KEY, BINANCE_API_SECRET
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
    UDL, "SPY", "QQQ", "XLK",
    "TLT", "VIXY", "GLD", "UUP",
    "BTCUSDT", "ETHUSDT"   # si tu veux suivre des crypto
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


def _fetch_binance_data(client: BinanceClient, symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance API (crypto only).
    """
    tf_map = {
        "1Day": BinanceClient.KLINE_INTERVAL_1DAY,
        "1H": BinanceClient.KLINE_INTERVAL_1HOUR,
        "1h": BinanceClient.KLINE_INTERVAL_1HOUR,
        "1Min": BinanceClient.KLINE_INTERVAL_1MINUTE,
        "1min": BinanceClient.KLINE_INTERVAL_1MINUTE,
    }
    interval = tf_map.get(timeframe, BinanceClient.KLINE_INTERVAL_1DAY)

    klines = client.get_historical_klines(symbol, interval, start, end)
    if len(klines) == 0:
        raise ValueError("Empty dataframe returned by Binance.")

    df = pd.DataFrame(klines, columns=[
        "DATE", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_base_volume", "taker_quote_volume", "ignore"
    ])

    df["DATE"] = pd.to_datetime(df["DATE"], unit="ms")
    df = df.set_index("DATE")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timeframe"] = timeframe
    df["source"] = "Binance"

    return df



def get_market_data(
    api: REST,
    tickers: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",
    feed: str = "iex",
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:

    prices, close_series = {}, []

    for ticker in tickers:

        df = None
        source = None

        # Primary source : Yahoo
        try:
            df = _fetch_yfinance_data(ticker, start, end, timeframe)
            source = "Yahoo Finance"

        # Fallback 1 : Alpaca
        except Exception as e:
            if verbose:
                print(f"[ WARN] {ticker}: Yahoo failed ({e}). Trying Alpaca...")

            try:
                df = _fetch_alpaca_data(api, ticker, start, end, timeframe, feed)
                source = "Alpaca"

            # Fallback 2 : Binance (crypto only)
            except Exception as e2:
                if verbose:
                    print(f"[ WARN] {ticker}: Alpaca failed ({e2}). Trying Binance...")

                try:
                    BINANCE = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
                    df = _fetch_binance_data(BINANCE, ticker, start, end, timeframe)
                    source = "Binance"

                except Exception as e3:
                    print(f"[ ERROR] {ticker}: all sources failed ({e3}). Skipping.")
                    continue

        df.index = pd.to_datetime(df.index)
        prices[ticker] = df

        s = df["close"].copy()
        s.name = ticker
        close_series.append(s)

        if verbose:
            print(f"{ticker}: {len(df)} obs from {source} "
                  f"({df.index.min().date()} → {df.index.max().date()}) [{timeframe}]")

    if not close_series:
        raise ValueError("No valid data fetched for any ticker.")

    px = pd.concat(close_series, axis=1).sort_index().dropna(how="all")
    retn = return_computation(px, TICKERS = list(px.columns))

    if verbose:
        print("\nMarket data successfully loaded")
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
import pandas as pd
import numpy as np
import pandas_datareader.data as web

def get_macro_data(
    region: str = "us",
    start: str = "2020-01-01",
    end: str | None = None,
    resample_rule: str = "D",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED for:
      - 'us': United States
      - 'europe': Euro Area
      - 'japan': Japan

    Notes:
      - TB3MS is intended as a *3-month money market* proxy for the region.
      - AAA10Y here is used as a *10Y sovereign yield* proxy (despite the name).
      - For rates, we forward-fill after resampling to avoid artificial intra-month wiggles.
    """
    region = region.lower()

    if region == "us":
        fred_codes = {
            "CPI": "CPIAUCSL",
            "INDPPI": "PPIACO",
            "M1SUPPLY": "M1SL",
            "CCREDIT": "TOTALSL",
            "AAA10Y": "DGS10",   # 10Y Treasury constant maturity (daily)
            "TB3MS": "TB3MS",    # 3-Month Treasury Bill (secondary market, monthly)
        }
        # Optional: include policy rate
        # fred_codes["POLICY"] = "FEDFUNDS"

    elif region == "europe":
        fred_codes = {
            "CPI": "CP0000EZ19M086NEST",   # CPI Euro area (Eurostat via FRED)
            "INDPPI": "PIEAMP02EZM661N",   # Industrial production (OECD)
            "M1SUPPLY": "MABMM201EZM189S", # Broad money (OECD)
            "CCREDIT": "QEZLOCOODCSXDC",   # Credit to private sector (BIS)
            "AAA10Y": "IRLTLT01EZM156N",   # 10Y gov bond yield, Euro Area (OECD)
            "TB3MS": "IR3TIB01EZM156N",    # 3M interbank rate, Euro Area (OECD)
            # Optional (overnight):
            # "ON": "ECBESTRVOLWGTTRMDMNRT", # €STR (daily, from 2019)
            # "ON_OLD": "EONIARATE",          # EONIA (discontinued)
        }

    elif region == "japan":
        fred_codes = {
            "CPI": "JPNCPIALLMINMEI",
            "INDPPI": "JPNPROINDMISMEI",
            "M1SUPPLY": "MYAGM1JPM189S",
            "CCREDIT": "QJPNLOCOODCANQ",
            "AAA10Y": "IRLTLT01JPM156N",   # 10Y gov bond yield, Japan (OECD)
            "TB3MS": "IR3TIB01JPM156N",    # 3M interbank rate, Japan (OECD)
        }

    else:
        raise ValueError("Region must be one of: 'us', 'europe', 'japan'")

    data = {}
    failed = {}

    for name, code in fred_codes.items():
        try:
            s = web.DataReader(code, "fred", start, end)
            # standardize to Series (1 col DF -> Series)
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:, 0]
            data[name] = s.rename(name)
        except Exception as e:
            raw = str(e).replace("\n", " ")
            clean = raw.split("Response Text:")[0].strip()  # coupe tout ce qui suit
            failed[name] = clean
            if verbose:
                print(f"[WARN] {name} ({code}): {clean[:160]}")


    if not data:
        if verbose:
            print("No macro series loaded.")
        return pd.DataFrame()

    fred = pd.concat(data.values(), axis=1).sort_index()
    fred.index = pd.to_datetime(fred.index)

    # Identify "rate-like" columns: we prefer ffill after resample
    rate_cols = [c for c in fred.columns if c in {"AAA10Y", "TB3MS", "ON", "ON_OLD", "POLICY"}]

    # Resample
    if resample_rule:
        fred_rs = fred.resample(resample_rule)

        # 1) rates: forward-fill to avoid interpolating within the month/day artificially
        fred_rates = fred[rate_cols].resample(resample_rule).ffill() if rate_cols else pd.DataFrame(index=fred_rs.mean().index)

        # 2) other macro levels: time interpolation (then ffill/bfill for edges)
        other_cols = [c for c in fred.columns if c not in rate_cols]
        fred_other = (
            fred[other_cols].resample(resample_rule).interpolate("time").ffill().bfill()
            if other_cols else pd.DataFrame(index=fred_rs.mean().index)
        )

        fred = pd.concat([fred_other, fred_rates], axis=1).sort_index()
    else:
        fred = fred.ffill().bfill()

    if verbose:
        print("\n==============================")
        print(f"Macro data loaded from FRED ({region.upper()})")
        print(f"Period: {fred.index.min().date()} → {fred.index.max().date()}")
        print(f"{fred.shape[0]} obs, {fred.shape[1]} variables")
        print("\n Successfully loaded:")
        for k in fred.columns:
            print(f"   - {k} ({fred_codes.get(k, 'n/a')})")
        missing = [k for k in fred_codes.keys() if k not in fred.columns]
        if missing:
            print("\n Failed to load:")
            for k in missing:
                print(f"   - {k} ({fred_codes[k]}): {failed.get(k, 'unknown error')}")
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
    macro["TS"] = macro_data.get("AAA10Y", pd.Series(np.nan, index=macro_data.index)) - macro_data.get("TB3MS", pd.Series(np.nan, index=macro_data.index))
    macro["DT"] = macro["TS"].diff()
    macro['DI'] = macro['INF'].diff()
    
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
