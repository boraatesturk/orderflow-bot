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
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

TZ_TR  = ZoneInfo("Europe/Istanbul")
SYMBOL = "ETHUSDT"
SIGNALS_FILE = "signals.json"

# ─── VERİ ÇEK ────────────────────────────────────────────────────

def fetch_bars(limit=300):
    """
    Bybit API'den OHLCV + taker buy/sell hacmi ceker.
    Bybit linear USDT perpetual, 5dk mumlar.
    Binance ve Kraken'in aksine gercek taker buy/sell verisi var.
    """
    # ── OHLCV ────────────────────────────────────────────────────
    url    = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": "ETHUSDT", "interval": "5", "limit": limit}
    r      = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data   = r.json()

    if data.get("retCode") != 0:
        raise Exception(f"Bybit OHLCV hatasi: {data.get('retMsg')}")

    # Bybit: [timestamp_ms, open, high, low, close, volume, turnover] — yeniden eskiye
    rows = data["result"]["list"]
    df   = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    df["open_time"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = df[c].astype(float)
    df.set_index("open_time", inplace=True)
    df.drop(columns=["ts"], inplace=True)
    df.sort_index(inplace=True)

    # ── TAKER BUY HACMI ──────────────────────────────────────────
    # Bybit /v5/market/recent-trade: son 1000 trade, side=Buy/Sell
    # Her 5dk bari icin taker buy oranini hesapla
    try:
        t_url    = "https://api.bybit.com/v5/market/recent-trade"
        t_params = {"category": "linear", "symbol": "ETHUSDT", "limit": 1000}
        t_r      = requests.get(t_url, params=t_params, timeout=10)
        t_data   = t_r.json()

        if t_data.get("retCode") == 0:
            trades = t_data["result"]["list"]
            tdf    = pd.DataFrame(trades)
            tdf["ts"]     = pd.to_datetime(tdf["time"].astype(np.int64), unit="ms", utc=True)
            tdf["volume"] = tdf["size"].astype(float)
            tdf["is_buy"] = tdf["side"] == "Buy"

            # 5dk bar'a yuvarla
            tdf["bar"] = tdf["ts"].dt.floor("5min")
            grp = tdf.groupby("bar").agg(
                buy_vol  = ("volume", lambda x: x[tdf.loc[x.index,"is_buy"]].sum()),
                sell_vol = ("volume", lambda x: x[~tdf.loc[x.index,"is_buy"]].sum()),
                count    = ("volume", "count")
            )
            df = df.join(grp, how="left")
            df["buy_volume"]  = df["buy_vol"].fillna(df["volume"] * 0.5)
            df["sell_volume"] = df["sell_vol"].fillna(df["volume"] * 0.5)
            df["trade_count"] = df["count"].fillna(0).astype(int)
            df.drop(columns=["buy_vol","sell_vol","count"], inplace=True, errors="ignore")
        else:
            raise Exception("trade verisi alinamadi")

    except Exception as e:
        print(f"[!] Taker trade fallback (fiyat tahmini): {e}")
        price_up          = df["close"] > df["open"]
        df["buy_volume"]  = np.where(price_up, df["volume"] * 0.65, df["volume"] * 0.35)
        df["sell_volume"] = df["volume"] - df["buy_volume"]
        df["trade_count"] = 0

    # ── TUREMIS KOLONLAR ─────────────────────────────────────────
    df["taker_buy_volume"]  = df["buy_volume"]
    df["delta"]             = df["buy_volume"] - df["sell_volume"]
    df["min_delta"]         = df["delta"]
    df["max_delta"]         = df["delta"]
    df["bid_trades"]        = 0
    df["ask_trades"]        = df["trade_count"]
    df["imbalance_ratio"]   = df["buy_volume"] / (df["volume"] + 1e-9)

    df["date"]              = df.index.date
    df["session_delta"]     = df.groupby("date")["delta"].cumsum()
    df["session_volume"]    = df.groupby("date")["volume"].cumsum()
    df["cvd"]               = df["delta"].cumsum()
    df["volume_per_second"] = df["volume"] / 300.0
    df["typical_price"]     = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"]              = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["poc_price"]         = df["close"]
    df["stacked_imbalance_up"] = (df["imbalance_ratio"] > 0.65).rolling(3).sum() == 3
    df["stacked_imbalance_dn"] = (df["imbalance_ratio"] < 0.35).rolling(3).sum() == 3
    df.drop(columns=["date","turnover"], inplace=True, errors="ignore")

    # Kapanmamis son bari cikar
    now         = datetime.now(timezone.utc)
    bar_min     = (now.minute // 5) * 5
    current_bar = now.replace(minute=bar_min, second=0, microsecond=0)
    df          = df[df.index < current_bar]

    print(f"[+] Bybit: {len(df)} bar | son fiyat: {df['close'].iloc[-1]:.2f}")
    return df


# ─── ML MODEL ───────────────────────────────────────────────────

_BASE_DIR         = Path(__file__).parent
ML_MODEL_B_PATH   = _BASE_DIR / "data" / "xgb_binary.pkl"
ML_MODEL_W_PATH   = _BASE_DIR / "data" / "xgb_weighted.pkl"
ML_CONF_THRESHOLD = 0.60
_ML_LABEL         = {-1: "DOWN", 0: "FLAT", 1: "UP"}

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

def get_ml_prediction(df):
    if not ML_MODEL_B_PATH.exists():
        return {"available": False}
    try:
        with open(ML_MODEL_B_PATH, "rb") as f:
            pb = pickle.load(f)
        model_b, le_b, feat_cols = pb["model"], pb["le"], pb["feature_cols"]

        feat = _build_ml_features(df)
        for col in feat_cols:
            if col not in feat.columns:
                feat[col] = 0
        last = feat[feat_cols].iloc[[-1]].fillna(0)

        prob_b     = model_b.predict_proba(last)[0]
        pred_b_raw = le_b.inverse_transform([model_b.predict(last)[0]])[0]
        conf_b     = float(max(prob_b))
        binary_signal = ("BUY" if pred_b_raw == 1 else "SELL") if conf_b >= ML_CONF_THRESHOLD else "FLAT"

        result = {
            "available":     True,
            "binary_signal": binary_signal,
            "binary_conf":   round(conf_b, 3),
            "prob_down":     round(float(prob_b[0]), 3),
            "prob_up":       round(float(prob_b[1]), 3),
        }

        if ML_MODEL_W_PATH.exists():
            with open(ML_MODEL_W_PATH, "rb") as f:
                pw = pickle.load(f)
            prob_w     = pw["model"].predict_proba(last)[0]
            pred_w_raw = pw["le"].inverse_transform([pw["model"].predict(last)[0]])[0]
            result["weighted_signal"] = _ML_LABEL[pred_w_raw]
            result["prob_down_w"]     = round(float(prob_w[0]), 3)
            result["prob_flat_w"]     = round(float(prob_w[1]), 3)
            result["prob_up_w"]       = round(float(prob_w[2]), 3)

        return result
    except Exception as e:
        print(f"[ML] Tahmin hatasi: {e}")
        return {"available": False}

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


# ─── TELEGRAM ───────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SPAM_MINUTES     = 30

def son_sinyal_ne_zaman(signals: list, direction: str) -> float:
    now = datetime.now(timezone.utc)
    for s in reversed(signals):
        if s.get("signal") == direction or s.get("ml_signal") == direction:
            try:
                t = datetime.fromisoformat(s["time_utc"])
                return (now - t).total_seconds() / 60
            except:
                pass
    return 9999

def send_telegram_signal(entry: dict) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Token eksik!")
        return False

    sig     = entry.get("signal", "FLAT")
    ml_sig  = entry.get("ml_signal", "N/A")
    ml_conf = entry.get("ml_conf", 0)
    uyum    = entry.get("ml_uyum", "—")

    # Gercek sinyal yonu: OF veya ML'den hangisi aktifse
    active_sig = sig if sig in ("BUY","SELL") else ml_sig
    emoji      = "🟢" if active_sig == "BUY" else "🔴"

    if uyum == "UYUM":       uyum_str = "🟰 UYUM (OF+ML)"
    elif uyum == "ML_ONCU":  uyum_str = "🤖 ML ÖNCÜ"
    else:                     uyum_str = "➖"

    lev  = entry.get("leverage", "—")
    sl   = entry.get("sl", "—")
    tp1  = entry.get("tp1", "—")
    tp2  = entry.get("tp2", "—")
    tp3  = entry.get("tp3")
    score = entry["score_buy"] if sig == "BUY" else entry["score_sell"]

    lines = [
        f"{emoji} <b>{active_sig} — ETHUSDT</b>  {uyum_str}",
        f"",
        f"💰 Giriş : <b>{entry['price']}</b> USDT",
        f"🛑 SL    : {sl}  (-%{entry.get('sl_pct','—')})",
        f"🎯 TP1   : {tp1}",
        f"🎯 TP2   : {tp2}",
    ]
    if tp3:
        lines.append(f"🎯 TP3   : {tp3}  (MSS/BOS)")
    lines += [
        f"",
        f"⚡ Kaldirac : {lev}x",
        f"📊 OF Skoru : {score:.1f}/10",
        f"🤖 ML       : {ml_sig} (%{ml_conf})",
        f"",
        f"🕐 {entry['time_tr']} (TR)",
    ]

    msg = "\n".join(lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r   = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    if r.status_code == 200:
        print("[TG] Sinyal gonderildi!")
        return True
    else:
        print(f"[TG] Hata: {r.text}")
        return False

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

    # ML tahmini
    ml = get_ml_prediction(df)
    ml_signal  = ml.get("binary_signal", "N/A") if ml.get("available") else "N/A"
    ml_conf    = ml.get("binary_conf", 0) if ml.get("available") else 0
    ml_w       = ml.get("weighted_signal", "N/A") if ml.get("available") else "N/A"

    # Uyum kontrolu
    if ml_signal != "N/A" and ml_signal != "FLAT" and signal != "FLAT":
        uyum = "UYUM" if signal == ml_signal else "CAKISMA"
    elif ml_signal != "FLAT" and ml_signal != "N/A" and signal == "FLAT":
        uyum = "ML_ONCU"
    else:
        uyum = "FLAT"

    print(f"Sinyal: {signal} | BUY: {sb} | SELL: {ss} | Fiyat: {price}")
    print(f"ML    : {ml_signal} (guven: %{ml_conf*100:.0f}) | Weighted: {ml_w} | Uyum: {uyum}")

    # Onceki sinyalleri yukle ve sonuclari guncelle
    signals = load_signals()
    signals = evaluate_previous(signals, price)

    # Yeni sinyali ekle (FLAT da kaydet, rapor icin)
    # ML ONCU durumunda da risk hesapla (OF=FLAT ama ML aktif)
    effective_signal = signal if signal in ("BUY","SELL") else ml_signal if ml_signal in ("BUY","SELL") else "FLAT"
    active_score = sb if effective_signal == "BUY" else ss if effective_signal == "SELL" else 0
    if active_score == 0 and effective_signal in ("BUY","SELL"):
        active_score = ml_conf * 10  # ML guvenini skor olarak kullan
    risk = compute_risk(effective_signal, active_score, last)

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
        "ml_signal":  ml_signal,
        "ml_conf":    round(ml_conf * 100, 1),
        "ml_weighted": ml_w,
        "ml_uyum":    uyum,
    }
    signals.append(new_entry)
    save_signals(signals)
    print(f"Kaydedildi. Toplam sinyal: {len(signals)}")

    # ── Telegram gonder ──────────────────────────────────────────
    # Kosul 1: Orderflow + ML ayni yonde (UYUM)
    # Kosul 2: ML tek basina %60+ guven veriyorsa (ML_ONCU dahil)
    of_dir = signal if signal in ("BUY","SELL") else None
    ml_dir = ml_signal if ml_signal in ("BUY","SELL") else None
    active_dir = of_dir or ml_dir

    should_send = (
        active_dir is not None and
        (uyum == "UYUM" or (ml_dir and ml_conf >= 0.60)) and
        son_sinyal_ne_zaman(signals[:-1], active_dir) >= SPAM_MINUTES
    )

    if should_send:
        print(f"[TG] Sinyal gonderiliyor: {active_dir} | uyum={uyum} | ml_conf=%{ml_conf*100:.0f}")
        send_telegram_signal(new_entry)
    else:
        reasons = []
        if not active_dir:                                    reasons.append("OF=FLAT ve ML=FLAT")
        elif uyum != "UYUM" and ml_conf < 0.60:             reasons.append(f"uyumsuz ve dusuk guven (%{ml_conf*100:.0f})")
        elif son_sinyal_ne_zaman(signals[:-1], active_dir) < SPAM_MINUTES:
                                                              reasons.append(f"spam koruma ({SPAM_MINUTES}dk)")
        print(f"[TG] Gonderilmedi: {', '.join(reasons)}")

    return signal, sb, ss, price


if __name__ == "__main__":
    main()