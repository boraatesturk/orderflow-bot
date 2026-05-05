"""
=============================================================================
  SIGNAL LOGGER
  Her 5 dakikada GitHub Actions tarafindan calistirilir.
  Sinyali uretir, sonucu signals.json'a kaydeder.
  Onceki sinyalin kazanip kaybettigini de hesaplar.
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

TZ_TR  = ZoneInfo("Europe/Istanbul")
SYMBOL = "ETHUSDT"
SIGNALS_FILE = "signals.json"

# ─── VERİ ÇEK ────────────────────────────────────────────────────

def fetch_bars(limit=288):
    """
    Kraken API'den OHLCV ceker (GitHub Actions icin - Binance ABD'yi engelliyor)
    Kraken ETHUSDT = XETHZUSD pairi, 5dk = 5 (dakika cinsinden)
    """
    # Kraken 5dk mum: interval=5, since=simdi-limit*300sn
    since = int((datetime.now(timezone.utc).timestamp())) - (limit * 300)
    params = {"pair": "ETHUSD", "interval": 5, "since": since}
    r = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise Exception(f"Kraken hatasi: {data['error']}")

    # Kraken response: {"result": {"XETHZUSD": [[time, open, high, low, close, vwap, volume, count]]}}
    pair_key = list(data["result"].keys())[0]  # "XETHZUSD" veya "ETHUSD"
    ohlcv    = data["result"][pair_key]

    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","vwap","volume","count"])
    df["open_time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    for c in ["open","high","low","close","volume","vwap"]:
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype(int)
    df.set_index("open_time", inplace=True)
    df.drop(columns=["time","vwap"], inplace=True, errors="ignore")
    df.sort_index(inplace=True)

    # Kraken'de taker buy/sell ayirimi yok, volume'u %50/%50 tahmin et
    # Delta icin RSI-benzeri momentum kullanacagiz
    df["trade_count"]      = df["count"]
    df["taker_buy_volume"] = df["volume"] * 0.5  # tahmini
    df["buy_volume"]       = df["taker_buy_volume"]
    df["sell_volume"]      = df["volume"] - df["buy_volume"]

    # Fiyat hareketi ile delta tahmini (yukari bar = net buy, asagi bar = net sell)
    df["open"] = df["open"].astype(float)
    price_up   = df["close"] > df["open"]
    df["buy_volume"]  = np.where(price_up, df["volume"] * 0.65, df["volume"] * 0.35)
    df["sell_volume"] = df["volume"] - df["buy_volume"]
    df["delta"]       = df["buy_volume"] - df["sell_volume"]
    df["min_delta"]   = df["delta"]
    df["max_delta"]   = df["delta"]
    df["bid_trades"]  = 0
    df["ask_trades"]  = df["count"]
    df["imbalance_ratio"]  = df["buy_volume"] / (df["volume"] + 1e-9)

    df["date"]             = df.index.date
    df["session_delta"]    = df.groupby("date")["delta"].cumsum()
    df["session_volume"]   = df.groupby("date")["volume"].cumsum()
    df["cvd"]              = df["delta"].cumsum()
    df["volume_per_second"] = df["volume"] / 300.0
    df["typical_price"]    = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"]             = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["poc_price"]        = df["close"]
    df["stacked_imbalance_up"] = (df["imbalance_ratio"] > 0.65).rolling(3).sum() == 3
    df["stacked_imbalance_dn"] = (df["imbalance_ratio"] < 0.35).rolling(3).sum() == 3
    df.drop(columns=["date"], inplace=True, errors="ignore")

    # Kapanmamis son bari cikar
    now = datetime.now(timezone.utc)
    bar_min = (now.minute // 5) * 5
    current_bar = now.replace(minute=bar_min, second=0, microsecond=0)
    df = df[df.index < current_bar]

    return df


# ─── SİNYAL ÜRETİCİ (signal_engine'den kopyalanmis) ─────────────

CFG = {
    "cvd_lookback":    5,
    "imbalance_bull":  0.58,
    "imbalance_bear":  0.42,
    "volume_spike_mult": 1.5,
    "min_score_buy":   5,
    "min_score_sell":  7,
}

def add_features(df):
    df = df.copy()
    df["cvd_change"]     = df["cvd"].diff(CFG["cvd_lookback"])
    df["vol_ma20"]       = df["volume"].rolling(20).mean()
    df["vol_spike"]      = df["volume"] / df["vol_ma20"]
    df["delta_ma3"]      = df["delta"].rolling(3).mean()
    df["delta_ma10"]     = df["delta"].rolling(10).mean()
    df["price_vs_vwap"]  = (df["close"] - df["vwap"]) / df["vwap"] * 100
    bar_range            = df["high"] - df["low"]
    df["close_position"] = np.where(bar_range > 0, (df["close"] - df["low"]) / bar_range, 0.5)
    df["delta_pct"]      = df["delta"] / (df["volume"] + 1e-9) * 100
    df["prev_delta"]     = df["delta"].shift(1)
    return df


def score_bar(row):
    sb, ss = 0.0, 0.0

    # Delta
    if row["delta"] > 0:
        sb += min(abs(row["delta_pct"]) / 10, 1.5)
    else:
        ss += min(abs(row["delta_pct"]) / 10, 1.5)

    # CVD
    if not pd.isna(row.get("cvd_change")):
        if row["cvd_change"] > 0: sb += 1.0
        else: ss += 1.0

    # Imbalance
    ir = row["imbalance_ratio"]
    if ir > CFG["imbalance_bull"]:   sb += 1.5
    elif ir < CFG["imbalance_bear"]: ss += 1.5

    # Stacked imbalance
    if row.get("stacked_imbalance_up", False): sb += 2.0
    if row.get("stacked_imbalance_dn", False): ss += 2.0

    # Session delta
    if not pd.isna(row.get("session_delta")):
        if row["session_delta"] > 0: sb += 0.75
        else: ss += 0.75

    # VWAP
    pvw = row.get("price_vs_vwap", 0)
    if not pd.isna(pvw):
        if pvw > 0.05:   sb += 0.5
        elif pvw < -0.05: ss += 0.5

    # Volume spike
    vs = row.get("vol_spike", 1)
    if not pd.isna(vs) and vs > CFG["volume_spike_mult"]:
        if row["delta"] > 0: sb += 1.0
        else: ss += 1.0

    # Close position
    cp = row.get("close_position", 0.5)
    if cp > 0.75: sb += 0.5
    elif cp < 0.25: ss += 0.5

    # Delta MA
    dm3  = row.get("delta_ma3", 0)
    dm10 = row.get("delta_ma10", 0)
    if not pd.isna(dm3) and not pd.isna(dm10):
        if dm3 > dm10 and dm3 > 0: sb += 0.5
        elif dm3 < dm10 and dm3 < 0: ss += 0.5

    sb = min(sb / 10 * 10, 10)
    ss = min(ss / 10 * 10, 10)

    if sb >= CFG["min_score_buy"] and sb > ss:
        signal = "BUY"
    elif ss >= CFG["min_score_sell"] and ss > sb:
        signal = "SELL"
    else:
        signal = "FLAT"

    return signal, round(sb, 2), round(ss, 2)


# ─── KAZANDI / KAYBETTİ HESABI ───────────────────────────────────

def evaluate_previous(signals: list, current_price: float) -> list:
    """
    Son 3 bar once verilen sinyalin sonucunu hesapla.
    3 bar = 15 dakika sonraki fiyata bak.
    """
    updated = []
    for s in signals:
        if s.get("result") is not None:
            updated.append(s)
            continue

        # Sinyal zamanından 15 dakika gecti mi?
        signal_time = datetime.fromisoformat(s["time_utc"])
        elapsed     = (datetime.now(timezone.utc) - signal_time).total_seconds()

        if elapsed >= 900 and s["signal"] != "FLAT":  # 15 dakika = 3 bar
            entry = s["price"]
            pnl   = (current_price - entry) / entry * 100
            if s["signal"] == "SELL":
                pnl = -pnl

            s["exit_price"] = round(current_price, 2)
            s["pnl_pct"]    = round(pnl, 3)
            s["result"]     = "✅ KAZANDI" if pnl > 0 else "❌ KAYBETTİ"

        updated.append(s)
    return updated


# ─── TP/SL HESABI ────────────────────────────────────────────────

BALANCE  = 2500.0
MIN_LEV  = 2
MAX_LEV  = 15

def compute_risk(signal, score, row):
    if signal == "FLAT":
        return {}

    close     = float(row["close"])
    high      = float(row["high"])
    low       = float(row["low"])
    direction = 1 if signal == "BUY" else -1

    # ATR tahmini
    atr = max(high - low, close * 0.002)

    # Kaldirac: skor 5=2x, 10=10x
    score_norm = max(0, score - 5.0) / 5.0
    leverage   = min(round(MIN_LEV + score_norm * (10 - MIN_LEV)), MAX_LEV)
    leverage   = max(leverage, MIN_LEV)

    # SL/TP
    sl_dist = max(atr * 1.5, close * 0.003)
    sl      = round(close - direction * sl_dist, 2)
    tp1     = round(close + direction * sl_dist * 1.5, 2)
    tp2     = round(close + direction * sl_dist * 3.0, 2)
    tp3     = round(close + direction * sl_dist * 5.0, 2)

    sl_pct      = round(abs(close - sl) / close * 100, 2)
    position    = round(BALANCE * leverage, 2)
    risk_usd    = round(BALANCE * (sl_pct / 100) * leverage, 2)
    reward_usd  = round(BALANCE * (abs(tp2 - close) / close) * leverage, 2)

    return {
        "leverage":   leverage,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
        "tp3":        tp3,
        "sl_pct":     sl_pct,
        "position":   position,
        "risk_usd":   risk_usd,
        "reward_usd": reward_usd,
    }


# ─── SİNYAL KAYDET ───────────────────────────────────────────────

def load_signals() -> list:
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "r") as f:
            return json.load(f)
    return []


def save_signals(signals: list):
    # Sadece son 200 sinyali tut
    signals = signals[-200:]
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)


# ─── MAIN ────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now(TZ_TR).strftime('%H:%M:%S')} TR] Sinyal uretiliyor...")

    df = fetch_bars(limit=288)
    df = add_features(df)
    df.dropna(subset=["close","delta","cvd","vol_ma20"], inplace=True)

    if df.empty:
        print("Veri bos, cikiliyor.")
        return

    last      = df.iloc[-1]
    signal, sb, ss = score_bar(last)
    price     = round(float(last["close"]), 2)
    now_utc   = datetime.now(timezone.utc)
    now_tr    = datetime.now(TZ_TR)

    print(f"Sinyal: {signal} | BUY: {sb} | SELL: {ss} | Fiyat: {price}")
    print(f"DEBUG high={last.get('high')} low={last.get('low')} risk={risk}")

    # Onceki sinyalleri yukle ve sonuclari guncelle
    signals = load_signals()
    signals = evaluate_previous(signals, price)

    # Yeni sinyali ekle (FLAT da kaydet, rapor icin)
    active_score = sb if signal == "BUY" else ss
    risk = compute_risk(signal, active_score, last)
    print(f"DEBUG risk={risk}")

    new_entry = {
        "time_utc":   now_utc.isoformat(),
        "time_tr":    now_tr.strftime("%d/%m/%Y %H:%M"),
        "signal":     signal,
        "price":      price,
        "score_buy":  sb,
        "score_sell": ss,
        "delta":      round(float(last["delta"]), 2),
        "cvd":        round(float(last["cvd"]), 2),
        "imbalance":  round(float(last["imbalance_ratio"]) * 100, 1),
        "leverage":   risk.get("leverage"),
        "sl":         risk.get("sl"),
        "tp1":        risk.get("tp1"),
        "tp2":        risk.get("tp2"),
        "tp3":        risk.get("tp3"),
        "sl_pct":     risk.get("sl_pct"),
        "risk_usd":   risk.get("risk_usd"),
        "result":     None if signal != "FLAT" else "—",
        "exit_price": None,
        "pnl_pct":    None,
    }
    signals.append(new_entry)
    save_signals(signals)

    print(f"DEBUG risk={risk}")
    print(f"Kaydedildi. Toplam sinyal: {len(signals)}")
    return signal, sb, ss, price


if __name__ == "__main__":
    main()
