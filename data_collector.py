"""
=============================================================================
  ORDERFLOW BOT - DATA COLLECTOR v2
  ETHUSDT | Binance | 5 Dakikalik Mumlar
  
  v2 Yenilikler:
    - data.binance.vision'dan bulk aggTrades download (50-100x hizli)
    - 5dk mumlar
    - Gercek delta, min_delta, max_delta hesaplama
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import io
import zipfile
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL         = "ETHUSDT"
INTERVAL       = "5m"
DAYS_BACK      = 365
AGG_DAYS_BACK  = 30    # aggTrades sadece son 30 gun
OUTPUT_DIR     = "data"
AGG_DIR        = f"{OUTPUT_DIR}/agg_raw"       # indirilen zip/csv dosyalari
PARQUET_FILE   = f"{OUTPUT_DIR}/{SYMBOL}_orderflow_{DAYS_BACK}d_5m.parquet"
CSV_FILE       = f"{OUTPUT_DIR}/{SYMBOL}_orderflow_{DAYS_BACK}d_5m.csv"

BASE_URL       = "https://api.binance.com"
VISION_URL     = "https://data.binance.vision/data/spot"
KLINES_EP      = "/api/v3/klines"
LIMIT          = 1000
SLEEP_MS       = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def log(msg, color=Fore.WHITE):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{Fore.CYAN}[{ts}]{Style.RESET_ALL} {color}{msg}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. OHLCV (Klines) - REST API (hizli, sorun yok)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    all_klines = []
    current_start = start_ms

    log(f"OHLCV cekilmeye basliyor: {symbol} {interval}", Fore.YELLOW)

    total_est = (end_ms - start_ms) // (5 * 60 * 1000)
    with tqdm(total=total_est, desc="Mumlar", unit="bar") as pbar:
        while current_start < end_ms:
            params = {
                "symbol": symbol, "interval": interval,
                "startTime": current_start, "endTime": end_ms,
                "limit": LIMIT,
            }
            try:
                r = requests.get(BASE_URL + KLINES_EP, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log(f"HATA (klines): {e}", Fore.RED)
                time.sleep(2)
                continue

            if not data:
                break

            all_klines.extend(data)
            pbar.update(len(data))
            current_start = data[-1][0] + 5 * 60_000
            time.sleep(SLEEP_MS)

    log(f"Toplam {len(all_klines)} mum cekildi.", Fore.GREEN)
    return all_klines


def parse_klines(raw: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trade_count",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)

    df["trade_count"] = df["trade_count"].astype(int)
    df.drop(columns=["_ignore"], inplace=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2. aggTrades - BULK DOWNLOAD (data.binance.vision)
# ═══════════════════════════════════════════════════════════════════════════

def get_months_between(start_date, end_date):
    """Iki tarih arasindaki aylari listeler."""
    months = []
    current = start_date.replace(day=1)
    while current <= end_date:
        months.append(current)
        current += relativedelta(months=1)
    return months


def get_days_for_current_month(end_date):
    """Icinde bulundugumuz ay icin gunluk dosyalari listeler."""
    days = []
    current = end_date.replace(day=1)
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def download_agg_trades_bulk(symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    data.binance.vision'dan aylik aggTrades zip'lerini indirir.
    Son ay icin gunluk dosyalari kullanir (aylik henuz yayinlanmamis olabilir).
    """
    os.makedirs(AGG_DIR, exist_ok=True)

    all_dfs = []
    months  = get_months_between(start_date, end_date)

    # Son 2 ay gunluk, geri kalani aylik indir
    cutoff_month = (end_date - relativedelta(months=2)).replace(day=1)

    log(f"aggTrades indiriliyor: {symbol} ({len(months)} ay)", Fore.YELLOW)
    log(f"Kaynak: data.binance.vision (bulk download)", Fore.CYAN)

    for month in tqdm(months, desc="Aylar", unit="ay"):
        year  = month.strftime("%Y")
        mon   = month.strftime("%m")
        ymon  = month.strftime("%Y-%m")

        if month >= cutoff_month:
            # Son 2 ay: gunluk dosyalar
            days = get_days_for_current_month(min(
                month + relativedelta(months=1) - timedelta(days=1),
                end_date
            ))
            first_day = month
            for day in range(first_day.day, min(
                (month + relativedelta(months=1)).day if month.month != end_date.month else end_date.day + 1,
                32
            )):
                try:
                    d = month.replace(day=day)
                except ValueError:
                    break
                if d > end_date:
                    break

                date_str = d.strftime("%Y-%m-%d")
                fname    = f"{symbol}-aggTrades-{date_str}.zip"
                fpath    = os.path.join(AGG_DIR, fname)
                url      = f"{VISION_URL}/daily/aggTrades/{symbol}/{fname}"

                df_day = _download_and_parse_zip(url, fpath, date_str)
                if df_day is not None:
                    all_dfs.append(df_day)
        else:
            # Eski aylar: aylik dosya (cok daha hizli)
            fname = f"{symbol}-aggTrades-{ymon}.zip"
            fpath = os.path.join(AGG_DIR, fname)
            url   = f"{VISION_URL}/monthly/aggTrades/{symbol}/{fname}"

            df_month = _download_and_parse_zip(url, fpath, ymon)
            if df_month is not None:
                all_dfs.append(df_month)

    if not all_dfs:
        log("HATA: Hic aggTrades indirilemedi!", Fore.RED)
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Tarih filtresi
    start_ms = int(start_date.timestamp() * 1000)
    end_ms   = int(end_date.timestamp() * 1000)
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]

    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    log(f"Toplam {len(df):,} agg trade yuklendi.", Fore.GREEN)
    return df


def _download_and_parse_zip(url: str, fpath: str, label: str) -> pd.DataFrame:
    """Tek bir zip dosyasini indir, parse et, DataFrame don."""

    # Daha once indirilmis mi?
    csv_path = fpath.replace(".zip", ".csv")
    if os.path.exists(csv_path):
        try:
            return _read_agg_csv(csv_path)
        except Exception:
            pass

    if os.path.exists(fpath):
        try:
            return _extract_and_read(fpath)
        except Exception:
            pass

    # Indir
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 404:
            return None  # Dosya henuz yok (gelecekteki tarih vs)
        r.raise_for_status()

        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return _extract_and_read(fpath)

    except requests.exceptions.RequestException:
        return None


def _extract_and_read(zip_path: str) -> pd.DataFrame:
    """Zip'i ac, CSV'yi oku."""
    csv_path = zip_path.replace(".zip", ".csv")

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = z.namelist()[0]
        z.extract(csv_name, os.path.dirname(zip_path))

        extracted = os.path.join(os.path.dirname(zip_path), csv_name)
        if extracted != csv_path:
            os.rename(extracted, csv_path)

    # Zip artik gereksiz, sil (disk tasarrufu)
    try:
        os.remove(zip_path)
    except Exception:
        pass

    return _read_agg_csv(csv_path)


def _read_agg_csv(csv_path: str) -> pd.DataFrame:
    """Binance aggTrades CSV formatini oku."""
    # Binance CSV: agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match
    cols = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp_ms", "is_seller", "best_match"]
    df = pd.read_csv(csv_path, names=cols, header=None)

    # Bazen header satiri olabiliyor
    if df["agg_id"].dtype == object:
        df = df.iloc[1:]

    df = df[["timestamp_ms", "price", "qty", "is_seller"]].copy()
    df["price"]        = df["price"].astype(float)
    df["qty"]          = df["qty"].astype(float)
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    # is_seller: True = satici agresif (maker buy, taker sell)
    df["is_seller"]    = df["is_seller"].astype(str).str.lower().isin(["true", "1"])

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════

def compute_bar_delta_features(klines_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Her 5dk bar icin gercek orderflow feature'larini hesaplar."""
    log("Bar-bazli delta feature'lari hesaplaniyor (gercek aggTrades)...", Fore.YELLOW)

    # Trade'leri 5dk'lik barlara grupla
    trades_df = trades_df.copy()

    # Her trade'in hangi 5dk bara ait oldugunu bul
    trades_df["bar_time"] = trades_df.index.floor("5min")

    # Bar bazli aggregation (vektorize - cok hizli)
    buy_mask  = ~trades_df["is_seller"]
    sell_mask = trades_df["is_seller"]

    trades_df["signed_qty"] = np.where(buy_mask, trades_df["qty"], -trades_df["qty"])

    bar_stats = trades_df.groupby("bar_time").agg(
        buy_volume  = ("qty",        lambda x: x[buy_mask.loc[x.index]].sum()),
        sell_volume = ("qty",        lambda x: x[sell_mask.loc[x.index]].sum()),
        ask_trades  = ("is_seller",  lambda x: (~x).sum()),   # buyer aggressor count
        bid_trades  = ("is_seller",  lambda x: x.sum()),      # seller aggressor count
    )

    # Delta ve min/max delta icin bar icinde kumulatif hesap
    delta_stats = []
    for bar_time, group in tqdm(trades_df.groupby("bar_time"), desc="Delta hesap", unit="bar"):
        signed    = group["signed_qty"].values
        cum_delta = np.cumsum(signed)
        delta_stats.append({
            "bar_time":  bar_time,
            "delta":     cum_delta[-1] if len(cum_delta) > 0 else 0.0,
            "min_delta": cum_delta.min() if len(cum_delta) > 0 else 0.0,
            "max_delta": cum_delta.max() if len(cum_delta) > 0 else 0.0,
        })

    delta_df = pd.DataFrame(delta_stats).set_index("bar_time")

    # bar_stats ile birlestir
    bar_features = bar_stats.join(delta_df, how="outer")

    # Klines ile birlestir
    result = klines_df.join(bar_features, how="left")

    # Eksik barlari tahminle doldur
    missing = result["buy_volume"].isna()
    if missing.any():
        log(f"  {missing.sum()} barda aggTrades eksik, klines tahmini kullaniliyor.", Fore.YELLOW)
        result.loc[missing, "buy_volume"]  = result.loc[missing, "taker_buy_volume"]
        result.loc[missing, "sell_volume"] = result.loc[missing, "volume"] - result.loc[missing, "taker_buy_volume"]
        result.loc[missing, "delta"]       = result.loc[missing, "buy_volume"] - result.loc[missing, "sell_volume"]
        result.loc[missing, "min_delta"]   = result.loc[missing, "delta"]
        result.loc[missing, "max_delta"]   = result.loc[missing, "delta"]
        result.loc[missing, "bid_trades"]  = 0
        result.loc[missing, "ask_trades"]  = result.loc[missing, "trade_count"]

    # Imbalance ratio
    total_vol = result["buy_volume"] + result["sell_volume"]
    result["imbalance_ratio"] = result["buy_volume"] / (total_vol + 1e-9)

    return result


def compute_estimated_features(klines_df: pd.DataFrame) -> pd.DataFrame:
    """aggTrades olmadan klines bazli tahmin."""
    df = klines_df.copy()
    df["buy_volume"]      = df["taker_buy_volume"]
    df["sell_volume"]     = df["volume"] - df["taker_buy_volume"]
    df["delta"]           = df["buy_volume"] - df["sell_volume"]
    df["min_delta"]       = df["delta"]
    df["max_delta"]       = df["delta"]
    df["bid_trades"]      = 0
    df["ask_trades"]      = df["trade_count"]
    df["imbalance_ratio"] = df["buy_volume"] / (df["volume"] + 1e-9)
    return df


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]             = df.index.date
    df["session_delta"]    = df.groupby("date")["delta"].cumsum()
    df["session_volume"]   = df.groupby("date")["volume"].cumsum()
    df["cvd"]              = df["delta"].cumsum()
    df["volume_per_second"] = df["volume"] / 300.0  # 5dk = 300sn

    # VWAP
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_x_vol"]      = df["typical_price"] * df["volume"]
    df["cum_tp_vol"]    = df.groupby("date")["tp_x_vol"].cumsum()
    df["cum_vol"]       = df.groupby("date")["volume"].cumsum()
    df["vwap"]          = df["cum_tp_vol"] / df["cum_vol"]

    # POC
    def session_poc(group):
        poc_vals = []
        for i in range(len(group)):
            sub = group.iloc[: i + 1]
            poc_price = sub["close"].iloc[sub["volume"].values.argmax()]
            poc_vals.append(poc_price)
        return pd.Series(poc_vals, index=group.index)

    df["poc_price"] = df.groupby("date", group_keys=False).apply(session_poc)

    df.drop(columns=["date", "tp_x_vol", "cum_tp_vol", "cum_vol"], inplace=True)
    return df


def compute_imbalance_features(df: pd.DataFrame, threshold: float = 0.7, stack_count: int = 3) -> pd.DataFrame:
    df = df.copy()
    ratio = df["imbalance_ratio"]
    bull = (ratio > threshold).astype(int)
    bear = (ratio < (1 - threshold)).astype(int)
    df["stacked_imbalance_up"] = bull.rolling(stack_count).sum() == stack_count
    df["stacked_imbalance_dn"] = bear.rolling(stack_count).sum() == stack_count
    return df


def save_dataset(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(PARQUET_FILE, engine="pyarrow", compression="snappy")
    size_mb = os.path.getsize(PARQUET_FILE) / (1024 * 1024)
    log(f"Parquet kaydedildi: {PARQUET_FILE} ({size_mb:.1f} MB)", Fore.GREEN)

    df.to_csv(CSV_FILE)
    size_mb_csv = os.path.getsize(CSV_FILE) / (1024 * 1024)
    log(f"CSV kaydedildi:     {CSV_FILE} ({size_mb_csv:.1f} MB)", Fore.GREEN)


def print_dataset_summary(df: pd.DataFrame):
    print()
    print("=" * 60)
    print(f"  DATASET OZETI - {SYMBOL} ({INTERVAL})")
    print("=" * 60)
    print(f"  Satirlar         : {len(df):,}")
    print(f"  Kolonlar         : {len(df.columns)}")
    print(f"  Baslangic        : {df.index.min()}")
    print(f"  Bitis            : {df.index.max()}")
    print()
    print(f"  Ort. delta       : {df['delta'].mean():.2f}")
    print(f"  Ort. hacim/bar   : {df['volume'].mean():.2f}")
    print(f"  CVD toplam       : {df['cvd'].iloc[-1]:.2f}")
    print(f"  Stacked Up       : {df['stacked_imbalance_up'].sum():,} bar")
    print(f"  Stacked Dn       : {df['stacked_imbalance_dn'].sum():,} bar")
    print()
    if "buy_volume" in df.columns:
        has_real = (df["bid_trades"] > 0).sum()
        total    = len(df)
        pct      = has_real / total * 100
        print(f"  Gercek aggTrades : {has_real:,} / {total:,} bar (%{pct:.1f})")
    print("=" * 60)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + f"  ORDERFLOW DATA COLLECTOR v2.0")
    print(Fore.CYAN + f"  Sembol: {SYMBOL} | Aralik: {INTERVAL} | {DAYS_BACK} gun")
    print(Fore.CYAN + f"  aggTrades: data.binance.vision (bulk download)")
    print(Fore.CYAN + "=" * 60)
    print()

    now       = datetime.now(timezone.utc)
    start     = now - timedelta(days=DAYS_BACK)
    now_ms    = int(now.timestamp() * 1000)
    start_ms  = int(start.timestamp() * 1000)

    # ── 1. OHLCV ─────────────────────────────────────────────────
    log("ADIM 1/5: OHLCV cekilliyor...", Fore.MAGENTA)
    raw_klines = fetch_klines(SYMBOL, INTERVAL, start_ms, now_ms)
    df_klines  = parse_klines(raw_klines)

    # ── 2. aggTrades (bulk) ──────────────────────────────────────
    log("ADIM 2/5: aggTrades indiriliyor (data.binance.vision)...", Fore.MAGENTA)
    log("NOT: Aylik zip dosyalari indirilecek, ilk seferinde 5-10 dk surebilir.", Fore.YELLOW)
    log("     Sonraki calistirmalarda cache'den okunur (aninda).", Fore.YELLOW)

    try:
        agg_start = now - timedelta(days=AGG_DAYS_BACK)
        df_trades = download_agg_trades_bulk(SYMBOL, agg_start, now)
        use_agg = not df_trades.empty
    except KeyboardInterrupt:
        log("aggTrades atlandi (Ctrl+C). Tahmini delta kullanilacak.", Fore.YELLOW)
        df_trades = pd.DataFrame()
        use_agg = False
    except Exception as e:
        log(f"aggTrades indirme hatasi: {e}", Fore.RED)
        log("Tahmini delta kullanilacak.", Fore.YELLOW)
        df_trades = pd.DataFrame()
        use_agg = False

    # ── 3. Bar delta feature'lari ────────────────────────────────
    log("ADIM 3/5: Delta feature'lari hesaplaniyor...", Fore.MAGENTA)
    if use_agg:
        df = compute_bar_delta_features(df_klines, df_trades)
        log(f"Gercek aggTrades kullanildi ({len(df_trades):,} trade).", Fore.GREEN)
    else:
        df = compute_estimated_features(df_klines)
        log("Tahmini delta kullanildi (klines bazli).", Fore.YELLOW)

    # ── 4. Session + CVD + VWAP + POC ────────────────────────────
    log("ADIM 4/5: Session, CVD, VWAP, POC hesaplaniyor...", Fore.MAGENTA)
    df = compute_session_features(df)
    df = compute_imbalance_features(df)
    df.dropna(inplace=True)

    # ── 5. Kaydet ────────────────────────────────────────────────
    log("ADIM 5/5: Dataset kaydediliyor...", Fore.MAGENTA)
    save_dataset(df)
    print_dataset_summary(df)
    log("TAMAMLANDI! Signal engine icin hazir.", Fore.GREEN)

    return df


if __name__ == "__main__":
    df = main()