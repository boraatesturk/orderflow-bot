"""
backtest_mtf.py
===============
ORDERFLOW BOT — MTF Backtest (vectorbt)

Yenilikler:
  - Multi-Timeframe: 5M skor + 15M filtre + 1H bias
  - Absorption dedektörü bonus skoru
  - ATR bazlı SL/TP (sabit yüzde değil)
  - Confluence filtresi: 2/3 veya 3/3
  - Saat filtresi, spam filtresi
  - Detaylı istatistik + dashboard PNG
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import vectorbt as vbt
except ImportError:
    print("[HATA] vectorbt kurulu değil. Çalıştır: pip install vectorbt")
    raise

# ─── AYARLAR ─────────────────────────────────────────────────────────────────
PARQUET_PATH = Path("data/ETHUSDT_orderflow_365d_5m.parquet")

BALANCE       = 2500.0
LEVERAGE      = 10
COMMISSION    = 0.0006     # %0.06 taker (her iki taraf = %0.12 round trip)
SLIPPAGE      = 0.0002     # %0.02

ATR_PERIOD    = 14
ATR_SL_MULT   = 1.5        # SL = ATR * 1.5
ATR_TP1_MULT  = 2.25       # TP1 = ATR * 2.25 (1.5R)
ATR_TP2_MULT  = 4.5        # TP2 = ATR * 4.5  (3.0R)

# Sinyal eşikleri (5M skoru)
MIN_SCORE_LONG  = 6.0   # 5.0'dan
MIN_SCORE_SHORT = 6.0   # 5.0'dan
MIN_SCORE_15M   = 5.0   # 3.0'dan
MIN_SCORE_1H    = 4.0   # 2.0'dan
MIN_CONFLUENCE  = 3     # 2'den — sadece 3/3

BLOCKED_HOURS   = [2, 3, 4]   # UTC — düşük win rate

CFG = {
    "imbalance_bull":       0.58,
    "imbalance_bear":       0.42,
    "volume_spike_mult":    1.5,
    "cvd_lookback":         5,
    "delta_ma_fast":        3,
    "delta_ma_slow":        10,
    "absorption_vol_mult":  1.5,
    "absorption_body_atr":  1.0,
    "absorption_delta_pct": 0.30,
    "absorption_bars":      5,
}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet bulunamadı: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df.sort_index(inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    print(f"[+] Veri: {len(df):,} bar | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ATR HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════════
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high       = df["high"]
    low        = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 5M FEATURE HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════════
def add_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Temel orderflow (parquet'te zaten var, eksik olanları hesapla)
    if "delta" not in df.columns:
        df["delta"] = df.apply(
            lambda r: r["volume"] * 0.6 if r["close"] >= r["open"] else -r["volume"] * 0.6,
            axis=1
        )
    if "imbalance_ratio" not in df.columns:
        df["buy_volume"]      = df["volume"] * df["delta"].apply(lambda d: 0.7 if d > 0 else 0.3)
        df["imbalance_ratio"] = df["buy_volume"] / (df["volume"] + 1e-9)
    if "cvd" not in df.columns:
        df["cvd"] = df["delta"].cumsum()
    if "session_delta" not in df.columns:
        df["session_delta"] = df["delta"].rolling(12).sum()
    if "vwap" not in df.columns:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)
    if "stacked_imbalance_up" not in df.columns:
        bull_s = (df["imbalance_ratio"] > CFG["imbalance_bull"]).astype(int)
        bear_s = (df["imbalance_ratio"] < CFG["imbalance_bear"]).astype(int)
        df["stacked_imbalance_up"] = bull_s.rolling(3).sum() == 3
        df["stacked_imbalance_dn"] = bear_s.rolling(3).sum() == 3

    # Türev özellikler
    df["atr"]          = compute_atr(df)
    df["cvd_slope"]    = df["cvd"].diff(CFG["cvd_lookback"])
    df["vol_ma20"]     = df["volume"].rolling(20).mean()
    df["vol_ratio"]    = df["volume"] / (df["vol_ma20"] + 1e-9)
    df["delta_pct"]    = df["delta"] / (df["volume"] + 1e-9)
    df["delta_ma_fast"]= df["delta"].rolling(CFG["delta_ma_fast"]).mean()
    df["delta_ma_slow"]= df["delta"].rolling(CFG["delta_ma_slow"]).mean()
    bar_range          = df["high"] - df["low"]
    df["close_pos"]    = np.where(bar_range > 0, (df["close"] - df["low"]) / bar_range, 0.5)
    df["vwap_dist"]    = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-9)

    # Absorption
    df["body"]           = (df["open"] - df["close"]).abs()
    df["body_atr_ratio"] = df["body"] / (df["atr"] + 1e-9)
    df["delta_pressure"] = df["delta"].abs() / (df["volume"] + 1e-9)
    df["recent_low"]     = df["low"].rolling(CFG["absorption_bars"]).min().shift(1)
    df["recent_high"]    = df["high"].rolling(CFG["absorption_bars"]).max().shift(1)
    df["new_low"]        = df["low"] < df["recent_low"]
    df["new_high"]       = df["high"] > df["recent_high"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ÜSTÜ TF FEATURE HESAPLAMA (15M ve 1H)
# ═══════════════════════════════════════════════════════════════════════════════
def resample_tf(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """5M'den üst TF'ye resample et, orderflow özelliklerini hesapla."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    # delta ve diğerleri varsa topla
    for col in ["delta", "buy_volume", "sell_volume"]:
        if col in df_5m.columns:
            agg[col] = "sum"

    df_tf = df_5m.resample(rule).agg(agg).dropna()

    # Orderflow türevleri
    if "delta" not in df_tf.columns:
        df_tf["delta"] = df_tf.apply(
            lambda r: r["volume"] * 0.6 if r["close"] >= r["open"] else -r["volume"] * 0.6,
            axis=1
        )
    if "buy_volume" not in df_tf.columns:
        df_tf["buy_volume"] = df_tf["volume"] * df_tf["delta"].apply(lambda d: 0.7 if d > 0 else 0.3)
    df_tf["imbalance_ratio"]    = df_tf["buy_volume"] / (df_tf["volume"] + 1e-9)
    df_tf["cvd"]                = df_tf["delta"].cumsum()
    df_tf["cvd_slope"]          = df_tf["cvd"].diff(CFG["cvd_lookback"])
    df_tf["session_delta"]      = df_tf["delta"].rolling(12).sum()
    df_tf["vwap"]               = (df_tf["close"] * df_tf["volume"]).cumsum() / (df_tf["volume"].cumsum() + 1e-9)
    df_tf["vol_ma20"]           = df_tf["volume"].rolling(20).mean()
    df_tf["vol_ratio"]          = df_tf["volume"] / (df_tf["vol_ma20"] + 1e-9)
    df_tf["delta_pct"]          = df_tf["delta"] / (df_tf["volume"] + 1e-9)
    df_tf["delta_ma_fast"]      = df_tf["delta"].rolling(CFG["delta_ma_fast"]).mean()
    df_tf["delta_ma_slow"]      = df_tf["delta"].rolling(CFG["delta_ma_slow"]).mean()
    bar_range                   = df_tf["high"] - df_tf["low"]
    df_tf["close_pos"]          = np.where(bar_range > 0, (df_tf["close"] - df_tf["low"]) / bar_range, 0.5)
    bull_s = (df_tf["imbalance_ratio"] > CFG["imbalance_bull"]).astype(int)
    bear_s = (df_tf["imbalance_ratio"] < CFG["imbalance_bear"]).astype(int)
    df_tf["stacked_imbalance_up"] = bull_s.rolling(3).sum() == 3
    df_tf["stacked_imbalance_dn"] = bear_s.rolling(3).sum() == 3

    return df_tf


def align_to_5m(series_tf: pd.Series, idx_5m: pd.DatetimeIndex) -> pd.Series:
    """Üst TF sinyalini 5M index'ine forward-fill ile hizala."""
    return series_tf.reindex(idx_5m, method="ffill")


# ═══════════════════════════════════════════════════════════════════════════════
# SKOR HESAPLAMA (vektörize)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_score_vectorized(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Her bar için LONG ve SHORT skoru hesapla.
    Returns: (score_long, score_short) — her ikisi de pozitif sayı
    """
    sl = pd.Series(0.0, index=df.index)
    ss = pd.Series(0.0, index=df.index)

    # 1. Delta yönü (0-1.5)
    dw = (df["delta_pct"].abs() / 0.1).clip(0, 1.0) * 1.5
    sl += np.where(df["delta"] > 0, dw, 0)
    ss += np.where(df["delta"] < 0, dw, 0)

    # 2. CVD momentum (0-1.0)
    sl += np.where(df["cvd_slope"] > 0, 1.0, 0)
    ss += np.where(df["cvd_slope"] < 0, 1.0, 0)

    # 3. Imbalance ratio (0-1.5)
    sl += np.where(df["imbalance_ratio"] >= CFG["imbalance_bull"], 1.5, 0)
    ss += np.where(df["imbalance_ratio"] <= CFG["imbalance_bear"], 1.5, 0)

    # 4. Stacked imbalance (0-2.0)
    sl += np.where(df["stacked_imbalance_up"].astype(bool), 2.0, 0)
    ss += np.where(df["stacked_imbalance_dn"].astype(bool), 2.0, 0)

    # 5. Session delta (0-0.75)
    sl += np.where(df["session_delta"] > 0, 0.75, 0)
    ss += np.where(df["session_delta"] < 0, 0.75, 0)

    # 6. VWAP (0-0.5)
    vwap_band = 0.0005
    sl += np.where(df["vwap_dist"] >  vwap_band, 0.5, 0)
    ss += np.where(df["vwap_dist"] < -vwap_band, 0.5, 0)

    # 7. Volume spike + delta (0-1.0)
    spike = df["vol_ratio"] >= CFG["volume_spike_mult"]
    sl += np.where(spike & (df["delta"] > 0), 1.0, 0)
    ss += np.where(spike & (df["delta"] < 0), 1.0, 0)

    # 8. Bar kapanış pozisyonu (0-0.5)
    sl += np.where(df["close_pos"] >= 0.7, 0.5, 0)
    ss += np.where(df["close_pos"] <= 0.3, 0.5, 0)

    # 9. Delta MA (0-0.5)
    sl += np.where(df["delta_ma_fast"] > df["delta_ma_slow"], 0.5, 0)
    ss += np.where(df["delta_ma_fast"] < df["delta_ma_slow"], 0.5, 0)

    # 10. Absorption bonus (0-2.0)
    high_vol   = df["vol_ratio"]        >= CFG["absorption_vol_mult"]
    small_body = df["body_atr_ratio"]   <  CFG["absorption_body_atr"]
    strong_prs = df["delta_pressure"]   >= CFG["absorption_delta_pct"]

    bull_absorb = high_vol & small_body & strong_prs & (df["delta"] < 0) & ~df["new_low"]
    bear_absorb = high_vol & small_body & strong_prs & (df["delta"] > 0) & ~df["new_high"]

    sl += np.where(bull_absorb & (sl > ss), 2.0, 0)
    ss += np.where(bear_absorb & (ss > sl), 2.0, 0)

    return sl, ss


def compute_score_upper_tf(df_tf: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Üst TF için basitleştirilmiş skor (absorption yok)."""
    sl = pd.Series(0.0, index=df_tf.index)
    ss = pd.Series(0.0, index=df_tf.index)

    dw = (df_tf["delta_pct"].abs() / 0.1).clip(0, 1.0) * 1.5
    sl += np.where(df_tf["delta"] > 0, dw, 0)
    ss += np.where(df_tf["delta"] < 0, dw, 0)
    sl += np.where(df_tf["cvd_slope"] > 0, 1.0, 0)
    ss += np.where(df_tf["cvd_slope"] < 0, 1.0, 0)
    sl += np.where(df_tf["imbalance_ratio"] >= CFG["imbalance_bull"], 1.5, 0)
    ss += np.where(df_tf["imbalance_ratio"] <= CFG["imbalance_bear"], 1.5, 0)
    sl += np.where(df_tf["stacked_imbalance_up"].astype(bool), 2.0, 0)
    ss += np.where(df_tf["stacked_imbalance_dn"].astype(bool), 2.0, 0)
    sl += np.where(df_tf["session_delta"] > 0, 0.75, 0)
    ss += np.where(df_tf["session_delta"] < 0, 0.75, 0)
    sl += np.where(df_tf["close_pos"] >= 0.7, 0.5, 0)
    ss += np.where(df_tf["close_pos"] <= 0.3, 0.5, 0)
    sl += np.where(df_tf["delta_ma_fast"] > df_tf["delta_ma_slow"], 0.5, 0)
    ss += np.where(df_tf["delta_ma_fast"] < df_tf["delta_ma_slow"], 0.5, 0)

    return sl, ss


# ═══════════════════════════════════════════════════════════════════════════════
# MTF SİNYAL ÜRETİCİ
# ═══════════════════════════════════════════════════════════════════════════════
def generate_mtf_signals(df_5m: pd.DataFrame) -> tuple:
    """
    3 TF skor + confluence filtresi → giriş sinyalleri.
    Returns: (buy_signal, sell_signal, score_5m_long, score_5m_short, confluence_long, confluence_short)
    """
    print("[*] Özellikler hesaplanıyor...")

    # 5M özellikler
    df5 = add_features_5m(df_5m)
    df5.dropna(inplace=True)

    # 15M ve 1H resample
    df15 = resample_tf(df5, "15min")
    df1h = resample_tf(df5, "60min")

    # Skorlar
    sl5, ss5   = compute_score_vectorized(df5)
    sl15, ss15 = compute_score_upper_tf(df15)
    sl1h, ss1h = compute_score_upper_tf(df1h)

    # 15M ve 1H sinyallerini 5M'e hizala
    sl15_5m = align_to_5m(sl15,                   df5.index)
    ss15_5m = align_to_5m(ss15,                   df5.index)
    sl1h_5m = align_to_5m(sl1h,                   df5.index)
    ss1h_5m = align_to_5m(ss1h,                   df5.index)

    # TF bazlı yön
    long_5m  = sl5   >= MIN_SCORE_LONG
    short_5m = ss5   >= MIN_SCORE_SHORT
    long_15m = sl15_5m >= MIN_SCORE_15M
    short_15m= ss15_5m >= MIN_SCORE_15M
    long_1h  = sl1h_5m >= MIN_SCORE_1H
    short_1h = ss1h_5m >= MIN_SCORE_1H

    # Confluence sayısı
    conf_long  = long_5m.astype(int) + long_15m.astype(int) + long_1h.astype(int)
    conf_short = short_5m.astype(int) + short_15m.astype(int) + short_1h.astype(int)

    # Saat filtresi
    hour    = df5.index.hour
    hour_ok = ~pd.Series(hour, index=df5.index).isin(BLOCKED_HOURS)

    # Giriş sinyalleri
    buy_signal  = (conf_long  >= MIN_CONFLUENCE) & (conf_long  > conf_short) & hour_ok
    sell_signal = (conf_short >= MIN_CONFLUENCE) & (conf_short > conf_long)  & hour_ok

    # Çakışma önleme
    conflict    = buy_signal & sell_signal
    buy_signal  = buy_signal  & ~conflict
    sell_signal = sell_signal & ~conflict

    print(f"[+] 5M  LONG  sinyali : {long_5m.sum():,}")
    print(f"[+] 5M  SHORT sinyali : {short_5m.sum():,}")
    print(f"[+] MTF LONG  (≥{MIN_CONFLUENCE}/3): {buy_signal.sum():,}")
    print(f"[+] MTF SHORT (≥{MIN_CONFLUENCE}/3): {sell_signal.sum():,}")

    return buy_signal, sell_signal, sl5, ss5, conf_long, conf_short, df5


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(df_5m: pd.DataFrame):
    buy_sig, sell_sig, sl5, ss5, conf_l, conf_s, df5 = generate_mtf_signals(df_5m)

    close = df5["close"].astype(float)
    atr   = df5["atr"].astype(float)

    # ATR bazlı SL/TP (yüzde olarak)
    sl_pct_long   = (atr * ATR_SL_MULT  / close).clip(0.002, 0.05)
    tp1_pct_long  = (atr * ATR_TP1_MULT / close).clip(0.003, 0.10)
    sl_pct_short  = sl_pct_long.copy()
    tp1_pct_short = tp1_pct_long.copy()

    print("[*] LONG backtest çalışıyor...")
    pf_long = vbt.Portfolio.from_signals(
        close      = close,
        entries    = buy_sig,
        exits      = pd.Series(False, index=close.index),
        sl_stop    = sl_pct_long,
        tp_stop    = tp1_pct_long,
        fees       = COMMISSION,
        slippage   = SLIPPAGE,
        init_cash  = BALANCE,
        size       = BALANCE * LEVERAGE * 0.02 / close,
        size_type  = "amount",
        freq       = "5min",
    )

    print("[*] SHORT backtest çalışıyor...")
    pf_short = vbt.Portfolio.from_signals(
        close      = close,
        entries    = pd.Series(False, index=close.index),
        exits      = sell_sig,
        short_entries = sell_sig,
        short_exits   = pd.Series(False, index=close.index),
        sl_stop    = sl_pct_short,
        tp_stop    = tp1_pct_short,
        fees       = COMMISSION,
        slippage   = SLIPPAGE,
        init_cash  = BALANCE,
        size       = BALANCE * LEVERAGE * 0.02 / close,
        size_type  = "amount",
        freq       = "5min",
    )

    return pf_long, pf_short, df5, buy_sig, sell_sig, conf_l, conf_s


# ═══════════════════════════════════════════════════════════════════════════════
# İSTATİSTİKLER
# ═══════════════════════════════════════════════════════════════════════════════
def print_stats(pf_long, pf_short):
    print("\n" + "=" * 60)
    print("  BACKTEST SONUÇLARI")
    print("=" * 60)

    for name, pf in [("LONG", pf_long), ("SHORT", pf_short)]:
        try:
            s = pf.stats()
            print(f"\n  ── {name} ──")
            print(f"  Trade sayısı   : {s.get('Total Trades', 0):.0f}")
            print(f"  Win Rate       : %{s.get('Win Rate [%]', 0):.1f}")
            print(f"  Toplam Getiri  : %{s.get('Total Return [%]', 0):.2f}")
            print(f"  Sharpe Ratio   : {s.get('Sharpe Ratio', 0):.3f}")
            print(f"  Max Drawdown   : %{s.get('Max Drawdown [%]', 0):.2f}")
            print(f"  Son Bakiye     : ${BALANCE + BALANCE * s.get('Total Return [%]', 0) / 100:.2f}")
        except Exception as e:
            print(f"  {name} istatistik hatası: {e}")

    print("\n" + "=" * 60)


def print_monthly(pf_long, pf_short):
    print("\n  AYLIK GETİRİ (%):")
    print(f"  {'Ay':<10} {'LONG':>8} {'SHORT':>8} {'TOPLAM':>8}")
    print(f"  {'-'*38}")
    try:
        ml = pf_long.returns().resample("ME").sum() * 100
        ms = pf_short.returns().resample("ME").sum() * 100
        mt = ml.add(ms, fill_value=0)
        for dt, v in mt.items():
            l = ml.get(dt, 0)
            s = ms.get(dt, 0)
            sign = "✅" if v > 0 else "❌"
            print(f"  {dt.strftime('%Y-%m'):<10} {l:>7.1f}% {s:>7.1f}% {v:>7.1f}% {sign}")
    except Exception as e:
        print(f"  Aylık getiri hatası: {e}")
    print()


def print_confluence_stats(conf_l, conf_s, buy_sig, sell_sig):
    print("\n  CONFLUENCE DAĞILIMI:")
    for c in [3, 2, 1, 0]:
        bl = (conf_l == c).sum()
        bs = (conf_s == c).sum()
        print(f"  {c}/3 → LONG:{bl:,}  SHORT:{bs:,}")
    print(f"\n  Sinyal atılan (≥{MIN_CONFLUENCE}/3):")
    print(f"  LONG : {buy_sig.sum():,}")
    print(f"  SHORT: {sell_sig.sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def plot_results(pf_long, pf_short, df5, buy_sig, sell_sig, conf_l, conf_s):
    BG     = "#0d1117"
    PANEL  = "#161b22"
    TEXT   = "#e6edf3"
    MUTED  = "#8b949e"
    BORDER = "#30363d"
    GREEN  = "#3fb950"
    RED    = "#f85149"
    YELLOW = "#d29922"
    BLUE   = "#58a6ff"
    PURPLE = "#bc8cff"

    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "text.color":       TEXT,
        "axes.labelcolor":  MUTED,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax, title):
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontweight="bold")
        ax.grid(color=BORDER, lw=0.4, alpha=0.6)

    # ── 1. Equity Curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Equity Curve")
    try:
        eq_l     = pf_long.value()
        eq_s     = pf_short.value()
        eq_total = BALANCE + (eq_l - BALANCE) + (eq_s - BALANCE)
        ax1.plot(eq_l.index,     eq_l.values,     color=GREEN,  lw=1.2, alpha=0.7, label="LONG")
        ax1.plot(eq_s.index,     eq_s.values,     color=RED,    lw=1.2, alpha=0.7, label="SHORT")
        ax1.plot(eq_total.index, eq_total.values, color=YELLOW, lw=2,   label="Toplam")
        ax1.axhline(BALANCE, color=MUTED, ls="--", lw=0.8, alpha=0.6)
        ax1.fill_between(eq_total.index, BALANCE, eq_total.values,
                         where=(eq_total.values >= BALANCE), alpha=0.12, color=GREEN)
        ax1.fill_between(eq_total.index, BALANCE, eq_total.values,
                         where=(eq_total.values < BALANCE),  alpha=0.12, color=RED)
    except Exception as e:
        ax1.text(0.5, 0.5, str(e), transform=ax1.transAxes, color=RED, ha="center")
    ax1.set_ylabel("USDT")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9, loc="upper right", ncol=4)

    # ── 2. Özet stat kutusu ─────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.set_facecolor(PANEL)
    ax_s.axis("off")
    ax_s.set_title("Özet", color=TEXT, fontsize=10, pad=8, fontweight="bold")
    try:
        wr_l = pf_long.stats().get("Win Rate [%]", 0)
        wr_s = pf_short.stats().get("Win Rate [%]", 0)
        tr_l = pf_long.stats().get("Total Return [%]", 0)
        tr_s = pf_short.stats().get("Total Return [%]", 0)
        sh_l = pf_long.stats().get("Sharpe Ratio", 0)
        sh_s = pf_short.stats().get("Sharpe Ratio", 0)
        md_l = pf_long.stats().get("Max Drawdown [%]", 0)
        md_s = pf_short.stats().get("Max Drawdown [%]", 0)
        n_l  = pf_long.stats().get("Total Trades", 0)
        n_s  = pf_short.stats().get("Total Trades", 0)
    except:
        wr_l=wr_s=tr_l=tr_s=sh_l=sh_s=md_l=md_s=n_l=n_s=0

    rows = [
        ("",        "LONG",         "SHORT"),
        ("Trades",  f"{n_l:.0f}",   f"{n_s:.0f}"),
        ("WinRate", f"%{wr_l:.1f}", f"%{wr_s:.1f}"),
        ("Getiri",  f"%{tr_l:.1f}", f"%{tr_s:.1f}"),
        ("Sharpe",  f"{sh_l:.2f}",  f"{sh_s:.2f}"),
        ("MaxDD",   f"%{md_l:.1f}", f"%{md_s:.1f}"),
    ]
    for i, (lbl, vl, vs) in enumerate(rows):
        y = 0.92 - i * 0.155
        ax_s.text(0.05, y, lbl, transform=ax_s.transAxes, color=MUTED, fontsize=9)
        ax_s.text(0.45, y, vl,  transform=ax_s.transAxes, color=GREEN if i == 0 else TEXT, fontsize=9)
        ax_s.text(0.75, y, vs,  transform=ax_s.transAxes, color=RED   if i == 0 else TEXT, fontsize=9)
        if i == 0:
            ax_s.plot([0.03, 0.97], [y - 0.02, y - 0.02],
                      transform=ax_s.transAxes, color=BORDER, lw=0.6, clip_on=False)

    # ── 3. Drawdown ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    style_ax(ax2, "Drawdown (%)")
    try:
        dd_l = pf_long.drawdown()  * 100
        dd_s = pf_short.drawdown() * 100
        ax2.plot(dd_l.index, dd_l.values, color=GREEN, lw=1, alpha=0.8, label="LONG DD")
        ax2.plot(dd_s.index, dd_s.values, color=RED,   lw=1, alpha=0.8, label="SHORT DD")
        ax2.fill_between(dd_l.index, dd_l.values, 0, alpha=0.1, color=GREEN)
        ax2.fill_between(dd_s.index, dd_s.values, 0, alpha=0.1, color=RED)
        ax2.axhline(0, color=MUTED, lw=0.6)
    except Exception as e:
        ax2.text(0.5, 0.5, str(e), transform=ax2.transAxes, color=RED, ha="center")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)

    # ── 4. Aylık getiri ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    style_ax(ax3, "Aylık Getiri (Toplam %)")
    try:
        ml = pf_long.returns().resample("ME").sum()  * 100
        ms = pf_short.returns().resample("ME").sum() * 100
        mt = ml.add(ms, fill_value=0)
        bar_colors = [GREEN if v >= 0 else RED for v in mt.values]
        ax3.bar(range(len(mt)), mt.values, color=bar_colors, alpha=0.85, width=0.6)
        ax3.axhline(0, color=MUTED, lw=0.8)
        ax3.set_xticks(range(len(mt)))
        ax3.set_xticklabels([d.strftime("%m/%y") for d in mt.index],
                            rotation=45, ha="right", fontsize=7)
    except Exception as e:
        ax3.text(0.5, 0.5, str(e), transform=ax3.transAxes, color=RED, ha="center")
    ax3.set_ylabel("%")

    # ── 5. Fiyat + sinyaller (son 2000 bar) ─────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    style_ax(ax4, f"Fiyat + MTF Sinyaller — son 2000 bar  (Confluence≥{MIN_CONFLUENCE}/3)")
    sample  = df5.tail(2000)
    ax4.plot(range(len(sample)), sample["close"].values,
             color=MUTED, lw=0.7, alpha=0.8, label="Close")
    b_mask  = buy_sig.reindex(sample.index, fill_value=False)
    s_mask  = sell_sig.reindex(sample.index, fill_value=False)
    b_pos   = [i for i, v in enumerate(b_mask) if v]
    s_pos   = [i for i, v in enumerate(s_mask) if v]
    if b_pos:
        ax4.scatter(b_pos, sample["close"].values[b_pos],
                    color=GREEN, s=25, marker="^", zorder=5, label=f"BUY ({len(b_pos)})")
    if s_pos:
        ax4.scatter(s_pos, sample["close"].values[s_pos],
                    color=RED, s=25, marker="v", zorder=5, label=f"SELL ({len(s_pos)})")
    ax4.set_ylabel("USDT")
    ax4.set_xlabel("Bar (5dk)")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)

    # ── 6. Trade dağılımı ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(PANEL)
    ax5.axis("off")
    ax5.set_title("Trade Dağılımı", color=TEXT, fontsize=10, pad=8, fontweight="bold")
    for spine in ax5.spines.values():
        spine.set_color(BORDER)
    try:
        win_l  = round(wr_l / 100 * n_l)
        lose_l = int(n_l) - int(win_l)
        win_s  = round(wr_s / 100 * n_s)
        lose_s = int(n_s) - int(win_s)
        conf3_l = (conf_l == 3).reindex(buy_sig.index, fill_value=False) & buy_sig
        conf2_l = (conf_l == 2).reindex(buy_sig.index, fill_value=False) & buy_sig
        conf3_s = (conf_s == 3).reindex(sell_sig.index, fill_value=False) & sell_sig
        conf2_s = (conf_s == 2).reindex(sell_sig.index, fill_value=False) & sell_sig

        info = [
            ("LONG trade",    f"{int(n_l):,}",     GREEN),
            ("  ↳ Kazanan",   f"{int(win_l):,}",   GREEN),
            ("  ↳ Kaybeden",  f"{int(lose_l):,}",  RED),
            ("SHORT trade",   f"{int(n_s):,}",     RED),
            ("  ↳ Kazanan",   f"{int(win_s):,}",   GREEN),
            ("  ↳ Kaybeden",  f"{int(lose_s):,}",  RED),
            ("3/3 LONG",      f"{conf3_l.sum():,}", BLUE),
            ("3/3 SHORT",     f"{conf3_s.sum():,}", PURPLE),
            ("2/3 LONG",      f"{conf2_l.sum():,}", BLUE),
            ("2/3 SHORT",     f"{conf2_s.sum():,}", PURPLE),
        ]
        for i, (lbl, val, col) in enumerate(info):
            y = 0.93 - i * 0.093
            ax5.text(0.05, y, lbl, transform=ax5.transAxes, color=MUTED, fontsize=8.5)
            ax5.text(0.72, y, val, transform=ax5.transAxes, color=col,   fontsize=8.5, fontweight="bold")
    except Exception as e:
        ax5.text(0.5, 0.5, str(e), transform=ax5.transAxes, color=RED, ha="center")

    # ── Başlık ───────────────────────────────────────────────────────
    fig.suptitle(
        f"OrderFlow Bot — MTF Backtest  |  ETHUSDT 5m+15m+1H  |  "
        f"{df5.index[0].date()} → {df5.index[-1].date()}  |  "
        f"Confluence≥{MIN_CONFLUENCE}/3  |  ATR SL×{ATR_SL_MULT}  TP×{ATR_TP1_MULT}  |  "
        f"Kom %{COMMISSION*100:.2f}  Slip %{SLIPPAGE*100:.2f}",
        color=TEXT, fontsize=9, y=0.99
    )

    out = Path("data/backtest_mtf_dashboard.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[+] Dashboard kaydedildi: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  ORDERFLOW BOT — MTF BACKTEST (vectorbt)")
    print(f"  Confluence : ≥{MIN_CONFLUENCE}/3")
    print(f"  SL         : ATR × {ATR_SL_MULT}")
    print(f"  TP1        : ATR × {ATR_TP1_MULT}")
    print(f"  Komisyon   : %{COMMISSION*100:.2f}  Slippage: %{SLIPPAGE*100:.2f}")
    print("=" * 60)

    df = load_data()
    pf_long, pf_short, df5, buy_sig, sell_sig, conf_l, conf_s = run_backtest(df)

    print_stats(pf_long, pf_short)
    print_monthly(pf_long, pf_short)
    print_confluence_stats(conf_l, conf_s, buy_sig, sell_sig)
    plot_results(pf_long, pf_short, df5, buy_sig, sell_sig, conf_l, conf_s)

    # Trade CSV
    try:
        tl = pf_long.trades.records_readable
        ts = pf_short.trades.records_readable
        if len(tl): tl["direction"] = "LONG"
        if len(ts): ts["direction"] = "SHORT"
        all_t = pd.concat([tl, ts], ignore_index=True)
        if len(all_t):
            out = Path("data/backtest_mtf_trades.csv")
            all_t.to_csv(out, index=False)
            print(f"[+] Trade listesi: {out}")
    except Exception as e:
        print(f"Trade CSV hatası: {e}")

    print("[+] Tamamlandı!")


if __name__ == "__main__":
    main()
