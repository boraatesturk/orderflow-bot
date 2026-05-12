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
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from colorama import Fore, Style, init
from ict_features import ICTEngine
import time as time_mod

init(autoreset=True)

# ─── ML MODEL ────────────────────────────────────────────────────────────────
_BASE_DIR         = Path(__file__).parent          # signal_engine.py'nin bulundugu klasor
ML_MODEL_W_PATH   = _BASE_DIR / "data" / "xgb_weighted.pkl"
ML_MODEL_B_PATH   = _BASE_DIR / "data" / "xgb_binary.pkl"
ML_CONF_THRESHOLD = 0.60   # Bu guvenin altinda ML tahmini FLAT sayilir

_ml_model_w = None
_ml_model_b = None
_ml_le_w    = None
_ml_le_b    = None
_ml_feat_cols = None

def _load_ml_models():
    global _ml_model_w, _ml_model_b, _ml_le_w, _ml_le_b, _ml_feat_cols
    if _ml_model_w is not None or _ml_model_b is not None:
        return True
    if not ML_MODEL_B_PATH.exists():
        print(f"{Fore.YELLOW}[ML] Model bulunamadi: {ML_MODEL_B_PATH} -> Once 'egit' calistir{Style.RESET_ALL}")
        return False
    try:
        with open(ML_MODEL_B_PATH, "rb") as f:
            p = pickle.load(f)
        _ml_model_b   = p["model"]
        _ml_le_b      = p["le"]
        _ml_feat_cols = p["feature_cols"]
        if ML_MODEL_W_PATH.exists():
            with open(ML_MODEL_W_PATH, "rb") as f:
                p = pickle.load(f)
            _ml_model_w = p["model"]
            _ml_le_w    = p["le"]
        return True
    except Exception as e:
        print(f"{Fore.YELLOW}[ML] Model yuklenemedi: {e}{Style.RESET_ALL}")
        return False

def _build_ml_features(df):
    feat = df.copy()
    for col in ["delta", "imbalance_ratio", "cvd", "volume", "close"]:
        if col not in feat.columns:
            continue
        for lag in [1, 3, 6, 12]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)
    for window in [5, 10, 20]:
        feat[f"close_roc_{window}"]    = feat["close"].pct_change(window)
        feat[f"volume_mean_{window}"]  = feat["volume"].rolling(window).mean()
        feat[f"delta_mean_{window}"]   = feat["delta"].rolling(window).mean()
        feat[f"delta_std_{window}"]    = feat["delta"].rolling(window).std()
        feat[f"imbalance_ma_{window}"] = feat["imbalance_ratio"].rolling(window).mean()
    if "vwap" in feat.columns:
        feat["price_vs_vwap"] = (feat["close"] - feat["vwap"]) / feat["vwap"]
    if "volume" in feat.columns and "delta" in feat.columns:
        feat["delta_norm"] = feat["delta"] / (feat["volume"] + 1e-9)
    if "cvd" in feat.columns:
        feat["cvd_roc5"]  = feat["cvd"].diff(5)
        feat["cvd_roc10"] = feat["cvd"].diff(10)
    feat["bar_direction"] = np.sign(feat["close"] - feat["open"])
    feat["hl_range"]      = (feat["high"] - feat["low"]) / feat["close"]
    bool_cols = feat.select_dtypes(include="bool").columns
    feat[bool_cols] = feat[bool_cols].astype(int)
    return feat

_ML_LABEL = {-1: "DOWN", 0: "FLAT", 1: "UP"}

def get_ml_prediction(df):
    if not _load_ml_models():
        return {"available": False}
    try:
        feat = _build_ml_features(df)
        # Eksik kolonlari 0 ile doldur (fetch_latest_bars parquet'ten farkli olabilir)
        for col in _ml_feat_cols:
            if col not in feat.columns:
                feat[col] = 0
        last = feat[_ml_feat_cols].iloc[[-1]].fillna(0)

        prob_b     = _ml_model_b.predict_proba(last)[0]
        pred_b_enc = _ml_model_b.predict(last)[0]
        pred_b_raw = _ml_le_b.inverse_transform([pred_b_enc])[0]
        conf_b     = float(max(prob_b))   # np.float32 -> float
        binary_signal = ("BUY" if pred_b_raw == 1 else "SELL") if conf_b >= ML_CONF_THRESHOLD else "FLAT"

        result = {
            "available":     True,
            "binary_signal": binary_signal,
            "binary_conf":   round(conf_b, 3),
            "prob_down_b":   round(prob_b[0], 3),
            "prob_up_b":     round(prob_b[1], 3),
        }
        if _ml_model_w is not None:
            prob_w     = _ml_model_w.predict_proba(last)[0]
            pred_w_enc = _ml_model_w.predict(last)[0]
            pred_w_raw = _ml_le_w.inverse_transform([pred_w_enc])[0]
            result["weighted_signal"] = _ML_LABEL[pred_w_raw]
            result["prob_down_w"]     = round(prob_w[0], 3)
            result["prob_flat_w"]     = round(prob_w[1], 3)
            result["prob_up_w"]       = round(prob_w[2], 3)
        return result
    except Exception as e:
        print(f"{Fore.YELLOW}[ML] Tahmin hatasi: {e}{Style.RESET_ALL}")
        return {"available": False, "error": str(e)}
# ─────────────────────────────────────────────────────────────────────────────

TZ_TR = ZoneInfo("Europe/Istanbul")

def log(msg, color=Fore.WHITE):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{Fore.CYAN}[{ts}]{Style.RESET_ALL} {color}{msg}{Style.RESET_ALL}")

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

# Risk parametreleri
BALANCE      = 2500.0   # Baslangic bakiye (USDT)
MIN_LEV      = 2        # Minimum kaldirac
MAX_LEV      = 15       # Maksimum kaldirac
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


def compute_risk(signal: str, score: float, row: pd.Series, ict_setups: dict = None) -> dict:
    """
    Skor, ICT setup ve volatiliteye gore risk parametrelerini hesaplar.

    Kaldirac mantigi:
      - Skor 5.0 = MIN_LEV (2x)
      - Skor 10.0 = MAX_LEV (15x)
      - ICT confirm varsa +2x bonus (15x tavan)
      - Stacked imbalance varsa +1x bonus

    SL mantigi:
      - Temel SL: ATR bazli (son 14 barin ortalama range'i)
      - OB varsa OB sinirinin otesine koy
      - Min SL mesafesi: giris fiyatinin %0.3'u

    TP mantigi:
      - TP1: SL mesafesinin 1.5x'i (ilk hedef, pozisyonun %50'si)
      - TP2: SL mesafesinin 3.0x'i (ana hedef)
      - TP3: SL mesafesinin 5.0x'i (uzun hedef, sadece MSS/BOS varsa)
      - FVG veya VWAP yakinindaysa TP olarak kullan

    Returns:
        dict: leverage, sl_price, tp1, tp2, tp3, position_size, risk_usdt, reward_usdt
    """
    if signal == "FLAT":
        return {}

    if ict_setups is None:
        ict_setups = {"bull": [], "bear": []}

    close     = row["close"]
    direction = 1 if signal == "BUY" else -1

    # ── ATR hesabi (14 bar range ortalamasi) ─────────────────────
    bar_range = row.get("high", close) - row.get("low", close)
    # Tek bar range cok kucuk olabilir, vol_spike ile scale et
    vol_spike = row.get("vol_spike", 1.0)
    atr_est   = max(bar_range, close * 0.002)  # Min %0.2

    # ── ICT konfirmasyon sayisi ───────────────────────────────────
    bull_confirms = len(ict_setups.get("bull", []))
    bear_confirms = len(ict_setups.get("bear", []))
    ict_confirms  = bull_confirms if signal == "BUY" else bear_confirms
    has_mss       = any("MSS" in s for s in ict_setups.get("bull" if signal == "BUY" else "bear", []))
    has_ob        = any("Order Block" in s for s in ict_setups.get("bull" if signal == "BUY" else "bear", []))
    has_fvg       = any("FVG" in s for s in ict_setups.get("bull" if signal == "BUY" else "bear", []))
    has_bos       = any("BOS" in s for s in ict_setups.get("bull" if signal == "BUY" else "bear", []))
    has_stacked   = row.get("stacked_imbalance_up" if signal == "BUY" else "stacked_imbalance_dn", False)

    # ── Kaldirac hesabi ───────────────────────────────────────────
    # Baz kaldirac: skor 5=2x, skor 10=10x dogrusal
    score_norm  = max(0, score - 5.0) / 5.0       # 0.0 - 1.0
    base_lev    = MIN_LEV + score_norm * (10 - MIN_LEV)

    # ICT bonus
    ict_bonus = 0
    if has_mss:      ict_bonus += 2.5
    if has_ob:       ict_bonus += 1.5
    if has_bos:      ict_bonus += 1.0
    if has_fvg:      ict_bonus += 1.0
    if has_stacked:  ict_bonus += 1.0

    leverage = min(round(base_lev + ict_bonus), MAX_LEV)
    leverage = max(leverage, MIN_LEV)

    # ── SL hesabi ─────────────────────────────────────────────────
    # Baz SL: 1.5x ATR
    sl_distance = atr_est * 1.5

    # OB varsa SL'i OB sinirinin otesine koy
    if has_ob and signal == "BUY":
        ob_bot = row.get("ob_bull_bot", np.nan)
        if not np.isnan(ob_bot) and ob_bot < close:
            sl_distance = max(sl_distance, close - ob_bot + atr_est * 0.3)
    elif has_ob and signal == "SELL":
        ob_top = row.get("ob_bear_top", np.nan)
        if not np.isnan(ob_top) and ob_top > close:
            sl_distance = max(sl_distance, ob_top - close + atr_est * 0.3)

    # Minimum SL mesafesi
    sl_distance = max(sl_distance, close * 0.003)

    sl_price = round(close - direction * sl_distance, 2)

    # ── TP hesabi ─────────────────────────────────────────────────
    tp1 = round(close + direction * sl_distance * 1.5, 2)   # 1.5R
    tp2 = round(close + direction * sl_distance * 3.0, 2)   # 3.0R
    tp3 = round(close + direction * sl_distance * 5.0, 2) if (has_mss or has_bos) else None  # 5R sadece guclu setup

    # FVG varsa TP olarak kullan (daha anlamli hedef)
    if has_fvg and signal == "BUY":
        fvg_top = row.get("fvg_bull_top", np.nan)
        if not np.isnan(fvg_top) and fvg_top > close:
            tp1 = round(min(tp1, fvg_top), 2)
    elif has_fvg and signal == "SELL":
        fvg_bot = row.get("fvg_bear_bot", np.nan)
        if not np.isnan(fvg_bot) and fvg_bot < close:
            tp1 = round(max(tp1, fvg_bot), 2)

    # ── Pozisyon buyuklugu ────────────────────────────────────────
    # Izole mod: BALANCE * leverage = toplam pozisyon
    position_size = round(BALANCE * leverage, 2)

    # Risk / Reward hesabi (USDT cinsinden)
    sl_pct      = abs(close - sl_price) / close
    risk_usdt   = round(BALANCE * sl_pct * leverage, 2)       # Ne kadar kaybedebiliriz
    reward_usdt = round(BALANCE * (abs(tp2 - close) / close) * leverage, 2)  # TP2'de ne kazaniriz

    return {
        "leverage":      leverage,
        "sl_price":      sl_price,
        "tp1":           tp1,
        "tp2":           tp2,
        "tp3":           tp3,
        "sl_pct":        round(sl_pct * 100, 3),
        "position_size": position_size,
        "risk_usdt":     risk_usdt,
        "reward_usdt":   reward_usdt,
        "rr_ratio":      round(reward_usdt / max(risk_usdt, 0.01), 2),
        "ict_confirms":  ict_confirms,
        "has_mss":       has_mss,
    }


def print_risk_card(risk: dict, signal: str, color: str):
    """Risk / pozisyon bilgilerini yazdirir."""
    if not risk:
        return

    lev   = risk["leverage"]
    sl    = risk["sl_price"]
    tp1   = risk["tp1"]
    tp2   = risk["tp2"]
    tp3   = risk.get("tp3")
    rr    = risk["rr_ratio"]
    pos   = risk["position_size"]
    r_usd = risk["risk_usdt"]
    w_usd = risk["reward_usdt"]

    print(color + "-" * 60)
    print(color + "  POZISYON ONERILERI")
    print(color + "-" * 60)
    print(f"  Bakiye         : ${BALANCE:,.0f} USDT")
    print(f"  {Fore.YELLOW}Kaldirac       : {lev}x")
    print(f"  Pozisyon       : ${pos:,.0f} USDT (izole)")
    print()
    print(f"  {Fore.RED}Stop Loss      : {sl:.2f} USDT  (-%{risk['sl_pct']:.2f})")
    print(f"  {Fore.GREEN}TP1 (50%)      : {tp1:.2f} USDT  [1.5R]")
    print(f"  {Fore.GREEN}TP2 (ana)      : {tp2:.2f} USDT  [3.0R]")
    if tp3:
        print(f"  {Fore.GREEN}TP3 (uzun)     : {tp3:.2f} USDT  [5.0R]  (MSS/BOS konfirm)")
    print()
    print(f"  Maks. risk     : ${r_usd:.2f} USDT")
    print(f"  Beklenen kazan : ${w_usd:.2f} USDT (TP2)")
    print(f"  R/R orani      : {rr:.2f}x")
    if risk.get("has_mss"):
        print(f"  {Fore.YELLOW}  ** MSS konfirm - yuksek guven setup **")
    print(color + "-" * 60)
    print()


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


def print_signal_card(row: pd.Series, signal: str, ict_setups: dict = None, df: pd.DataFrame = None):
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

    # ── ML TAHMIN BLOGU ─────────────────────────────────────────────────
    ml = get_ml_prediction(df) if df is not None else {"available": False}
    if ml.get("available"):
        ml_sig  = ml["binary_signal"]
        ml_conf = ml["binary_conf"]
        ml_color = Fore.GREEN if ml_sig == "BUY" else Fore.RED if ml_sig == "SELL" else Fore.YELLOW
        conf_bar = "█" * int(ml_conf * 10) + "░" * (10 - int(ml_conf * 10))
        print(Fore.CYAN + "─" * 60)
        print(Fore.CYAN + "  ML TAHMIN (XGBoost Binary)")
        print(Fore.CYAN + "─" * 60)
        print(f"  Sinyal   : {ml_color}{ml_sig}{Style.RESET_ALL}  (guven: %{ml_conf*100:.0f}  [{conf_bar}])")
        print(f"  DOWN={ml['prob_down_b']:.1%}  UP={ml['prob_up_b']:.1%}  (esik: %{ML_CONF_THRESHOLD*100:.0f})")
        if ml.get("weighted_signal"):
            wc = Fore.GREEN if ml["weighted_signal"]=="UP" else Fore.RED if ml["weighted_signal"]=="DOWN" else Fore.YELLOW
            print(f"  Weighted : {wc}{ml['weighted_signal']}{Style.RESET_ALL}  "
                  f"(D={ml.get('prob_down_w',0):.1%} F={ml.get('prob_flat_w',0):.1%} U={ml.get('prob_up_w',0):.1%})")

        # Orderflow + ML uyumu
        of_dir  = "BUY" if signal == "BUY" else "SELL" if signal == "SELL" else "FLAT"
        ml_dir  = ml_sig
        if of_dir != "FLAT" and ml_dir != "FLAT":
            if of_dir == ml_dir:
                print(f"  {Fore.GREEN}[UYUM] Orderflow + ML ayni yonu gosteriyor -> Guclu sinyal!")
            else:
                print(f"  {Fore.RED}[CAKISMA] Orderflow {of_dir} ama ML {ml_dir} -> Dikkat!")
        elif of_dir == "FLAT" and ml_dir != "FLAT":
            print(f"  {Fore.YELLOW}[ML ONCU] Orderflow FLAT ama ML {ml_dir} diyor -> Izle")
        print(Fore.CYAN + "─" * 60)
        print()

    # Risk karti - FLAT'te bile en guclu tarafi goster
    if signal != "FLAT":
        active_score = sb if signal == "BUY" else ss
        risk = compute_risk(signal, active_score, row, ict_setups)
        if risk:
            print_risk_card(risk, signal, color)
    else:
        # FLAT sinyalinde en yuksek skoru olan tarafi goster (bilgi amacli)
        if sb > ss and sb > 3:
            print(f"  {Fore.YELLOW}[FLAT - Potansiyel BUY tarafina dikkat]")
            risk = compute_risk("BUY", sb, row, ict_setups)
            if risk:
                print_risk_card(risk, "BUY", Fore.YELLOW)
        elif ss > sb and ss > 3:
            print(f"  {Fore.YELLOW}[FLAT - Potansiyel SELL tarafina dikkat]")
            risk = compute_risk("SELL", ss, row, ict_setups)
            if risk:
                print_risk_card(risk, "SELL", Fore.YELLOW)


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

    for i in range(len(df) - HOLD_BARS - 1):
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

    # ICT feature'larini hesapla
    ict = ICTEngine(df, swing_left=10, swing_right=10, fvg_min_pct=0.05)
    df  = ict.compute_all()

    df = add_derived_features(df)

    # Sadece temel kolonlarda NaN temizle, ICT kolonlarini koru
    core_cols = ["close", "open", "high", "low", "volume", "delta",
                 "cvd", "vwap", "imbalance_ratio", "vol_ma20", "delta_ma3"]
    df.dropna(subset=[c for c in core_cols if c in df.columns], inplace=True)

    if df.empty:
        print(Fore.RED + "HATA: Veri islendikten sonra bos kaldi!")
        return

    # ICT setuplarini once hesapla (print_signal_card icin gerekli)
    setups = ict.get_active_setups(last_n=30)

    # Son tamamlanan bar
    last_row = df.iloc[-1]
    signal   = generate_signal(last_row)
    print_signal_card(last_row, signal, ict_setups=setups, df=df)

    # ICT aktif setuplarini goster
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "  ICT SETUP DURUMU (son 3 bar)")
    print(Fore.CYAN + "=" * 60)

    if not setups["bull"] and not setups["bear"]:
        print(f"  {Fore.YELLOW}Aktif ICT setup yok (FLAT bolge)")
    else:
        if setups["bull"]:
            print(f"  {Fore.GREEN}>> BULLISH SETUPLAR:")
            for s in setups["bull"]:
                print(f"     {Fore.GREEN}{s}")

        if setups["bull"] and setups["bear"]:
            print()

        if setups["bear"]:
            print(f"  {Fore.RED}>> BEARISH SETUPLAR:")
            for s in setups["bear"]:
                print(f"     {Fore.RED}{s}")

    print(Fore.CYAN + "=" * 60)
    print()



def fetch_latest_bars(symbol: str, interval: str = "5m", limit: int = 288) -> pd.DataFrame:
    """Binance REST API'den son N bari ceker - her cagrimda taze veri."""
    import requests as req
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = req.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trade_count","taker_buy_volume","taker_buy_quote_volume","_"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","taker_buy_volume"]:
        df[c] = df[c].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df.set_index("open_time", inplace=True)
    df.drop(columns=["_","close_time","quote_volume","taker_buy_quote_volume"], inplace=True, errors="ignore")

    df["buy_volume"]       = df["taker_buy_volume"]
    df["sell_volume"]      = df["volume"] - df["taker_buy_volume"]
    df["delta"]            = df["buy_volume"] - df["sell_volume"]
    df["min_delta"]        = df["delta"]
    df["max_delta"]        = df["delta"]
    df["bid_trades"]       = 0
    df["ask_trades"]       = df["trade_count"]
    df["imbalance_ratio"]  = df["buy_volume"] / (df["volume"] + 1e-9)
    df["date"]             = df.index.date
    df["session_delta"]    = df.groupby("date")["delta"].cumsum()
    df["session_volume"]   = df.groupby("date")["volume"].cumsum()
    df["cvd"]              = df["delta"].cumsum()
    df["volume_per_second"] = df["volume"] / 300.0
    df["typical_price"]    = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"]             = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["poc_price"]        = df["close"]
    df["stacked_imbalance_up"] = (df["imbalance_ratio"] > 0.7).rolling(3).sum() == 3
    df["stacked_imbalance_dn"] = (df["imbalance_ratio"] < 0.3).rolling(3).sum() == 3
    df.drop(columns=["date"], inplace=True, errors="ignore")
    return df


def run_realtime(once: bool = False):
    """Her 5 dakikada bir Binance'den TAZE veri ceker, sinyal uretir."""
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + f"  GERCEK ZAMANLI MOD  |  {SYMBOL}  |  5dk")
    print(Fore.CYAN + "  Durdurmak icin: Ctrl+C")
    print(Fore.CYAN + "=" * 60)

    while True:
        try:
            log("Binance'den taze veri cekiliyor...", Fore.YELLOW)
            df_live = fetch_latest_bars(SYMBOL, interval="5m", limit=300)

            # Henuz kapanmamis son bari cikar
            now_utc  = datetime.now(timezone.utc)
            current_bar_open = now_utc.replace(second=0, microsecond=0)
            minutes = current_bar_open.minute
            bar_minutes = (minutes // 5) * 5
            current_bar_open = current_bar_open.replace(minute=bar_minutes)
            df_live = df_live[df_live.index < current_bar_open]

            if df_live.empty:
                log("Yeterli veri yok, bekleniyor...", Fore.YELLOW)
                time_mod.sleep(10)
                continue

            # ICT analiz
            ict    = ICTEngine(df_live, swing_left=5, swing_right=5, fvg_min_pct=0.05)
            df_ict = ict.compute_all()
            df_ict = add_derived_features(df_ict)

            core = ["close","delta","cvd","imbalance_ratio","vol_ma20"]
            df_ict.dropna(subset=[c for c in core if c in df_ict.columns], inplace=True)

            if df_ict.empty:
                log("Veri isleme hatasi.", Fore.RED)
                time_mod.sleep(10)
                continue

            setups   = ict.get_active_setups(last_n=30)
            last_row = df_ict.iloc[-1]
            signal   = generate_signal(last_row)

            # Ekrani temizle ve yeni sinyali goster
            os.system("cls" if os.name == "nt" else "clear")
            print(Fore.CYAN + f"  Guncellendi: {datetime.now(TZ_TR).strftime('%d/%m/%Y %H:%M:%S')} (TR)  |  Bar: {last_row.name}")
            print_signal_card(last_row, signal, ict_setups=setups, df=df_ict)

            print(Fore.CYAN + "=" * 60)
            print(Fore.CYAN + "  ICT SETUP DURUMU (son 288 bar = 1 gün)")
            print(Fore.CYAN + "=" * 60)
            if not setups["bull"] and not setups["bear"]:
                print(f"  {Fore.YELLOW}Aktif ICT setup yok")
            if setups["bull"]:
                print(f"  {Fore.GREEN}>> BULLISH:")
                for s in setups["bull"]:
                    print(f"     {Fore.GREEN}{s}")
            if setups["bear"]:
                print(f"  {Fore.RED}>> BEARISH:")
                for s in setups["bear"]:
                    print(f"     {Fore.RED}{s}")
            print(Fore.CYAN + "=" * 60)

            # Sonraki 5dk bar kapanisina kadar bekle
            now_ts = int(datetime.now(timezone.utc).timestamp())
            seconds_to_next = 300 - (now_ts % 300)
            if once:
                break
            print(f"\n  {Fore.YELLOW}Sonraki guncelleme: {seconds_to_next} saniye sonra (bar kapanisi)")
            time_mod.sleep(seconds_to_next + 3)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Durduruldu.")
            break
        except Exception as e:
            log(f"HATA: {e}", Fore.RED)
            time_mod.sleep(15)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orderflow Signal Engine")
    parser.add_argument("--backtest", action="store_true", help="Tum dataset backtest")
    parser.add_argument("--last",     type=int,            help="Son N bar backtest")
    parser.add_argument("--live",     action="store_true", help="Gercek zamanli mod (5dk)")
    parser.add_argument("--once",     action="store_true", help="Tek seferlik sinyal al ve kapat")
    args = parser.parse_args()

    if args.live:
        run_realtime()
    elif args.once:
        run_realtime(once=True)
    else:
        df = load_dataset()
        if args.backtest:
            run_backtest(df)
        elif args.last:
            run_backtest(df, last_n=args.last)
        else:
            run_live(df)


if __name__ == "__main__":
    main()