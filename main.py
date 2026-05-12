"""
=============================================================================
  ORDERFLOW BOT - SIGNAL ENGINE
  ETHUSDT | Claude 4.6 Opus Destekli Orderflow Sinyal Motoru

  Kullanim:
    python signal_engine.py                    # Canli mod (son bar)
    python signal_engine.py --backtest         # Tum dataset uzerinde test
    python signal_engine.py --last 100         # Son 100 bar backtest
=============================================================================
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
SYMBOL       = "ETHUSDT"
DAYS_BACK    = 180

# Sinyal parametreleri (bunlari optimize edebilirsin)
CFG = {
    # Delta divergence: fiyat yukari ama delta asagi = zayif yukselis
    "delta_threshold":          0.0,        # delta > 0 = net buy pressure

    # CVD momentum: son N barin CVD degisimi
    "cvd_lookback":             5,

    # Imbalance: buy_vol / total_vol
    "imbalance_bull":           0.55,       # >55% buy = bullish
    "imbalance_bear":           0.45,       # <45% buy = bearish

    # Session delta pozitif mi?
    "session_delta_weight":     True,

    # Stacked imbalance onay
    "stacked_confirm":          True,

    # VWAP: fiyat VWAP'in neresinde?
    "use_vwap":                 True,

    # Volume spike: ort hacmin kac kati?
    "volume_spike_mult":        1.5,

    # Sinyal icin minimum skor (0-10)
    "min_score_buy":            6,
    "min_score_sell":           6,
}
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset() -> pd.DataFrame:
    """Kaydedilmis parquet dataseti yukler."""
    parquet = f"{DATA_DIR}/{SYMBOL}_orderflow_{DAYS_BACK}d.parquet"
    csv     = f"{DATA_DIR}/{SYMBOL}_orderflow_{DAYS_BACK}d.csv"

    if os.path.exists(parquet):
        df = pd.read_parquet(parquet, engine="pyarrow")
        print(f"{Fore.GREEN}Dataset yuklendi: {parquet} ({len(df):,} bar){Style.RESET_ALL}")
    elif os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        print(f"{Fore.YELLOW}CSV yuklendi: {csv}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}HATA: Dataset bulunamadi. Once data_collector.py calistirin!{Style.RESET_ALL}")
        sys.exit(1)

    # Index timezone kontrolu
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Signal engine icin ek turevi feature'lar ekler."""
    df = df.copy()

    # CVD degisim momentum
    df["cvd_change"]     = df["cvd"].diff(CFG["cvd_lookback"])
    df["cvd_slope"]      = df["cvd"].diff(1)

    # Volume ortalamasi
    df["vol_ma20"]       = df["volume"].rolling(20).mean()
    df["vol_spike"]      = df["volume"] / df["vol_ma20"]

    # Delta momentum (son 3 bar ortalamasi)
    df["delta_ma3"]      = df["delta"].rolling(3).mean()
    df["delta_ma10"]     = df["delta"].rolling(10).mean()

    # Fiyat VWAP pozisyonu
    df["price_vs_vwap"]  = (df["close"] - df["vwap"]) / df["vwap"] * 100  # %

    # Kapanis: high'a mi low'a mi yakin?
    bar_range            = df["high"] - df["low"]
    df["close_position"] = np.where(
        bar_range > 0,
        (df["close"] - df["low"]) / bar_range,
        0.5
    )  # 0=low, 1=high

    # Delta / Volume orani (normalized delta)
    df["delta_pct"] = df["delta"] / (df["volume"] + 1e-9) * 100

    # Onceki bar delta
    df["prev_delta"]  = df["delta"].shift(1)
    df["delta_flip"]  = np.sign(df["delta"]) != np.sign(df["prev_delta"])

    return df


def compute_signal_score(row: pd.Series) -> dict:
    """
    Bir bar icin orderflow skorunu hesaplar.
    
    Her kural icin puan verir (0-1 arasi).
    Toplam skor 0-10 skalaya normalize edilir.
    
    Returns:
        dict: score_buy, score_sell, reasons_buy, reasons_sell
    """
    score_buy  = 0.0
    score_sell = 0.0
    reasons_b  = []
    reasons_s  = []

    # ── KURAL 1: Delta yonu ──────────────────────────────────────
    # Delta pozitif = alincilar agir basiyor
    if row["delta"] > 0:
        weight = min(abs(row["delta_pct"]) / 10, 1.5)
        score_buy += weight
        reasons_b.append(f"Delta pozitif ({row['delta']:+.1f}, {row['delta_pct']:+.1f}%)")
    else:
        weight = min(abs(row["delta_pct"]) / 10, 1.5)
        score_sell += weight
        reasons_s.append(f"Delta negatif ({row['delta']:+.1f}, {row['delta_pct']:+.1f}%)")

    # ── KURAL 2: CVD momentum ────────────────────────────────────
    if not pd.isna(row["cvd_change"]):
        if row["cvd_change"] > 0:
            score_buy += 1.0
            reasons_b.append(f"CVD yukseliyor (+{row['cvd_change']:.1f} son {CFG['cvd_lookback']} barda)")
        else:
            score_sell += 1.0
            reasons_s.append(f"CVD dusuyor ({row['cvd_change']:.1f} son {CFG['cvd_lookback']} barda)")

    # ── KURAL 3: Imbalance ratio ─────────────────────────────────
    ir = row["imbalance_ratio"]
    if ir > CFG["imbalance_bull"]:
        score_buy += 1.5
        reasons_b.append(f"Buy imbalance yuksek (%{ir*100:.1f})")
    elif ir < CFG["imbalance_bear"]:
        score_sell += 1.5
        reasons_s.append(f"Sell imbalance yuksek (%{(1-ir)*100:.1f})")

    # ── KURAL 4: Stacked imbalance ───────────────────────────────
    if CFG["stacked_confirm"]:
        if row.get("stacked_imbalance_up", False):
            score_buy += 2.0
            reasons_b.append("Stacked BULL imbalance (3+ bar)!")
        if row.get("stacked_imbalance_dn", False):
            score_sell += 2.0
            reasons_s.append("Stacked BEAR imbalance (3+ bar)!")

    # ── KURAL 5: Session delta yonu ──────────────────────────────
    if CFG["session_delta_weight"] and not pd.isna(row.get("session_delta", np.nan)):
        if row["session_delta"] > 0:
            score_buy += 0.75
            reasons_b.append(f"Session delta pozitif (+{row['session_delta']:.1f})")
        else:
            score_sell += 0.75
            reasons_s.append(f"Session delta negatif ({row['session_delta']:.1f})")

    # ── KURAL 6: Fiyat vs VWAP ───────────────────────────────────
    if CFG["use_vwap"] and not pd.isna(row.get("price_vs_vwap", np.nan)):
        pvw = row["price_vs_vwap"]
        if pvw > 0.05:   # VWAP'in %0.05 ustunde
            score_buy += 0.5
            reasons_b.append(f"Fiyat VWAP ustunde (+%{pvw:.3f})")
        elif pvw < -0.05:
            score_sell += 0.5
            reasons_s.append(f"Fiyat VWAP altinda (-%{abs(pvw):.3f})")

    # ── KURAL 7: Volume spike ────────────────────────────────────
    if not pd.isna(row.get("vol_spike", np.nan)):
        if row["vol_spike"] > CFG["volume_spike_mult"]:
            # Yuksek hacim: delta yonunde guclendirir
            if row["delta"] > 0:
                score_buy += 1.0
                reasons_b.append(f"Volume spike + buy delta ({row['vol_spike']:.1f}x ort)")
            else:
                score_sell += 1.0
                reasons_s.append(f"Volume spike + sell delta ({row['vol_spike']:.1f}x ort)")

    # ── KURAL 8: Bar kapanis pozisyonu ───────────────────────────
    cp = row.get("close_position", 0.5)
    if cp > 0.75:    # Bar'in ust %25'inde kapandi = guc
        score_buy += 0.5
        reasons_b.append(f"Bar yuksekte kapandi (%{cp*100:.0f})")
    elif cp < 0.25:  # Bar'in alt %25'inde kapandi = zayiflik
        score_sell += 0.5
        reasons_s.append(f"Bar alcakta kapandi (%{cp*100:.0f})")

    # ── KURAL 9: Delta MA crossover ──────────────────────────────
    if not pd.isna(row.get("delta_ma3", np.nan)) and not pd.isna(row.get("delta_ma10", np.nan)):
        if row["delta_ma3"] > row["delta_ma10"] and row["delta_ma3"] > 0:
            score_buy += 0.5
            reasons_b.append("Delta MA3 > MA10 (yukselis momentum)")
        elif row["delta_ma3"] < row["delta_ma10"] and row["delta_ma3"] < 0:
            score_sell += 0.5
            reasons_s.append("Delta MA3 < MA10 (dusus momentum)")

    # Normalize: 0-10
    max_possible = 10.0
    score_buy  = min(score_buy / max_possible * 10, 10)
    score_sell = min(score_sell / max_possible * 10, 10)

    return {
        "score_buy":   round(score_buy, 2),
        "score_sell":  round(score_sell, 2),
        "reasons_buy": reasons_b,
        "reasons_sell": reasons_s,
    }


def generate_signal(row: pd.Series) -> str:
    """
    Skor bazinda BUY / SELL / FLAT sinyali uretir.
    """
    result = compute_signal_score(row)
    sb = result["score_buy"]
    ss = result["score_sell"]

    if sb >= CFG["min_score_buy"] and sb > ss:
        return "BUY"
    elif ss >= CFG["min_score_sell"] and ss > sb:
        return "SELL"
    else:
        return "FLAT"


def print_signal_card(row: pd.Series, signal: str):
    """Terminal'a guzel bir sinyal karti yazdirir."""
    result = compute_signal_score(row)
    sb = result["score_buy"]
    ss = result["score_sell"]

    # Renk
    if signal == "BUY":
        color = Fore.GREEN
        symbol = "▲ BUY"
    elif signal == "SELL":
        color = Fore.RED
        symbol = "▼ SELL"
    else:
        color = Fore.YELLOW
        symbol = "— FLAT"

    ts = row.name if hasattr(row, "name") else "N/A"

    print()
    print(color + "=" * 60)
    print(color + f"  {symbol}  |  {SYMBOL}  |  {ts}")
    print(color + "=" * 60)
    print(f"  Fiyat          : {row['close']:.2f} USDT")
    print(f"  VWAP           : {row.get('vwap', 0):.2f} USDT")
    print(f"  POC            : {row.get('poc_price', 0):.2f} USDT")
    print()
    print(f"  Delta          : {row['delta']:+.2f}")
    print(f"  Session Delta  : {row.get('session_delta', 0):+.2f}")
    print(f"  CVD            : {row.get('cvd', 0):+.2f}")
    print(f"  Buy Vol        : {row.get('buy_volume', 0):.2f}")
    print(f"  Sell Vol       : {row.get('sell_volume', 0):.2f}")
    print(f"  Imbalance      : %{row.get('imbalance_ratio', 0)*100:.1f} buy")
    print(f"  Vol Spike      : {row.get('vol_spike', 1):.2f}x")
    print()
    print(f"  {Fore.GREEN}BUY  Skoru : {sb:.1f}/10")
    print(f"  {Fore.RED}SELL Skoru : {ss:.1f}/10")
    print()

    if result["reasons_buy"]:
        print(f"  {Fore.GREEN}BUY nedenleri:")
        for r in result["reasons_buy"]:
            print(f"    {Fore.GREEN}+ {r}")

    if result["reasons_sell"]:
        print(f"  {Fore.RED}SELL nedenleri:")
        for r in result["reasons_sell"]:
            print(f"    {Fore.RED}- {r}")

    print(color + "=" * 60)
    print()


def run_backtest(df: pd.DataFrame, last_n: int = None) -> pd.DataFrame:
    """
    Tum dataset uzerinde sinyal uretir, basit P&L hesaplar.
    """
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "  BACKTEST MODU")
    print(Fore.CYAN + "=" * 60)

    if last_n:
        df = df.tail(last_n).copy()
        print(f"Son {last_n} bar test ediliyor...")
    else:
        print(f"Tum {len(df):,} bar test ediliyor...")

    df = add_derived_features(df)
    df.dropna(inplace=True)

    signals = []
    buy_scores = []
    sell_scores = []

    for _, row in df.iterrows():
        sig = generate_signal(row)
        result = compute_signal_score(row)
        signals.append(sig)
        buy_scores.append(result["score_buy"])
        sell_scores.append(result["score_sell"])

    df["signal"]     = signals
    df["score_buy"]  = buy_scores
    df["score_sell"] = sell_scores

    # Basit P&L simulasyonu:
    # BUY = sonraki barda long gir, 3 bar sonra cik
    # SELL = sonraki barda short gir, 3 bar sonra cik
    HOLD_BARS = 3
    pnl_list  = []

    for i in range(len(df) - HOLD_BARS - 2):
        sig       = df["signal"].iloc[i]
        entry     = df["close"].iloc[i + 1]
        exit_     = df["close"].iloc[i + 1 + HOLD_BARS]
        pct_chg   = (exit_ - entry) / entry * 100

        if sig == "BUY":
            pnl_list.append(pct_chg)
        elif sig == "SELL":
            pnl_list.append(-pct_chg)
        else:
            pnl_list.append(0.0)

    df_pnl    = df.iloc[: len(pnl_list)].copy()
    df_pnl["pnl_pct"] = pnl_list

    # Istatistikler
    buys  = df_pnl[df_pnl["signal"] == "BUY"]
    sells = df_pnl[df_pnl["signal"] == "SELL"]
    flats = df_pnl[df_pnl["signal"] == "FLAT"]

    print()
    print(Fore.CYAN + "─" * 60)
    print("  BACKTEST SONUCLARI")
    print(Fore.CYAN + "─" * 60)
    print(f"  Toplam bar     : {len(df_pnl):,}")
    print(f"  BUY sinyal     : {len(buys):,}")
    print(f"  SELL sinyal    : {len(sells):,}")
    print(f"  FLAT           : {len(flats):,}")
    print()

    if len(buys) > 0:
        win_rate_b = (buys["pnl_pct"] > 0).mean() * 100
        avg_pnl_b  = buys["pnl_pct"].mean()
        print(f"  {Fore.GREEN}BUY  win rate  : %{win_rate_b:.1f}")
        print(f"  {Fore.GREEN}BUY  ort P&L   : %{avg_pnl_b:+.3f} per trade")

    if len(sells) > 0:
        win_rate_s = (sells["pnl_pct"] > 0).mean() * 100
        avg_pnl_s  = sells["pnl_pct"].mean()
        print(f"  {Fore.RED}SELL win rate  : %{win_rate_s:.1f}")
        print(f"  {Fore.RED}SELL ort P&L   : %{avg_pnl_s:+.3f} per trade")

    all_trades = df_pnl[df_pnl["signal"] != "FLAT"]
    if len(all_trades) > 0:
        total_pnl  = all_trades["pnl_pct"].sum()
        overall_wr = (all_trades["pnl_pct"] > 0).mean() * 100
        print()
        print(f"  Toplam P&L     : %{total_pnl:+.2f}")
        print(f"  Genel win rate : %{overall_wr:.1f}")

    print(Fore.CYAN + "─" * 60)
    print()

    # Son 20 sinyali goster
    recent_signals = df_pnl[df_pnl["signal"] != "FLAT"].tail(20)
    if not recent_signals.empty:
        print("  Son 20 sinyal:")
        for idx, row in recent_signals.iterrows():
            sig   = row["signal"]
            color = Fore.GREEN if sig == "BUY" else Fore.RED
            pnl_c = Fore.GREEN if row["pnl_pct"] > 0 else Fore.RED
            print(f"  {color}{sig:4s}{Style.RESET_ALL}  {idx}  "
                  f"close={row['close']:.2f}  "
                  f"delta={row['delta']:+.1f}  "
                  f"pnl={pnl_c}%{row['pnl_pct']:+.3f}{Style.RESET_ALL}")

    return df_pnl


def run_live(df: pd.DataFrame):
    """Son bari analiz eder ve sinyal uretir."""
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + f"  CANLI SINYAL MODU  |  {SYMBOL}")
    print(Fore.CYAN + "=" * 60)

    df = add_derived_features(df)
    df.dropna(inplace=True)

    # Son tamamlanan bar
    last_row = df.iloc[-1]
    signal   = generate_signal(last_row)
    print_signal_card(last_row, signal)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orderflow Signal Engine")
    parser.add_argument("--backtest", action="store_true", help="Tum dataset backtest")
    parser.add_argument("--last",     type=int,            help="Son N bar backtest")
    args = parser.parse_args()

    df = load_dataset()

    if args.backtest:
        run_backtest(df)
    elif args.last:
        run_backtest(df, last_n=args.last)
    else:
        run_live(df)


if __name__ == "__main__":
    main()