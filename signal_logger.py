"""
signal_logger.py — v2 Güncelleme Yaması
=========================================
Mevcut signal_logger.py'a bu değişiklikleri uygula.

YAPILAN DEĞİŞİKLİKLER:
1. signal_engine yerine signal_engine_v2 import ediliyor
2. Exit sinyali kontrolü eklendi
3. Telegram'a ayrı exit mesajı atılıyor
4. signals.json'a absorption + dynamic_exit bilgisi yazılıyor
"""

import json
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ─── YENİ IMPORT ───────────────────────────────────────────
from signal_engine_v2 import (
    analyze,
    format_telegram_signal,
    format_telegram_exit,
    SignalResult,
)

# ─── CONFIG ────────────────────────────────────────────────
SYMBOL          = "ETHUSDT"
SIGNALS_FILE    = "signals.json"
MAX_SIGNALS     = 200
SPAM_MINUTES    = 5          # aynı yönde tekrar atma süresi (dk)
TZ              = ZoneInfo("Europe/Istanbul")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── BYBIT VERİ ÇEKİCİ ────────────────────────────────────
def fetch_bybit_klines(symbol: str, interval: str = "5", limit: int = 300) -> pd.DataFrame:
    """Bybit'ten OHLCV + taker buy/sell verisi çek."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    rows = data["result"]["list"]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df = df.iloc[::-1].reset_index(drop=True)  # en eski → en yeni

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Orderflow türev kolonları (Bybit taker data yoksa tahmini hesapla)
    # Bybit v5 kline'da taker_buy_base_vol ayrı endpoint; burada delta'yı
    # (close > open → pozitif delta) ile tahmin ediyoruz
    df["delta"] = df.apply(
        lambda r: r["volume"] * 0.6 if r["close"] >= r["open"]
                  else -r["volume"] * 0.6,
        axis=1
    )
    df["buy_volume"]  = df["volume"] * df["delta"].apply(lambda d: 0.7 if d > 0 else 0.3)
    df["sell_volume"] = df["volume"] - df["buy_volume"]
    df["imbalance_ratio"] = df["buy_volume"] / df["volume"]
    df["cvd"] = df["delta"].cumsum()
    df["session_delta"] = df["delta"].rolling(12).sum()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # Stacked imbalance (3+ ardışık aynı yönlü imbalance)
    bull_streak = (df["imbalance_ratio"] > 0.58).astype(int)
    bear_streak = (df["imbalance_ratio"] < 0.42).astype(int)
    df["stacked_imbalance_up"] = bull_streak.rolling(3).sum() == 3
    df["stacked_imbalance_dn"] = bear_streak.rolling(3).sum() == 3

    return df


# ─── SIGNALS.JSON YARDIMCILARI ─────────────────────────────
def load_signals() -> list:
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_signals(signals: list) -> None:
    signals = signals[-MAX_SIGNALS:]
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, default=str)


def is_spam(signals: list, direction: str, minutes: int = SPAM_MINUTES) -> bool:
    """Son SPAM_MINUTES dk içinde aynı yönde sinyal var mı?"""
    now = datetime.now(timezone.utc)
    for s in reversed(signals):
        try:
            ts = datetime.fromisoformat(s["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff = (now - ts).total_seconds() / 60
            if diff <= minutes and s.get("direction") == direction:
                return True
        except Exception:
            pass
    return False


# ─── TELEGRAM GÖNDERİCİ ────────────────────────────────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram credentials eksik")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hata: {e}")
        return False


# ─── SİNYALİ JSON'A KAYDET ─────────────────────────────────
def result_to_dict(res: SignalResult) -> dict:
    """SignalResult → signals.json kaydı."""
    od  = res.orderflow
    ab  = res.absorption
    dex = res.dynamic_exit
    return {
        "timestamp":   str(datetime.now(TZ).isoformat()),
        "symbol":      res.symbol,
        "direction":   res.direction,
        "score":       round(res.score, 3),
        "entry":       res.entry,
        "sl":          res.sl,
        "tp1":         res.tp1,
        "tp2":         res.tp2,
        "tp3":         res.tp3,
        "leverage":    res.leverage,
        "atr":         res.atr,

        # Orderflow detayı
        "orderflow": {
            "delta":          round(od.delta, 2),
            "delta_pct":      round(od.delta_pct * 100, 2),
            "cvd":            round(od.cvd, 2),
            "cvd_slope":      round(od.cvd_slope, 2),
            "imbalance":      round(od.imbalance_ratio, 4),
            "stacked_up":     od.stacked_up,
            "stacked_dn":     od.stacked_dn,
            "vol_ratio":      round(od.volume_ratio, 3),
            "session_delta":  round(od.session_delta, 2),
            "price_vs_vwap":  od.price_vs_vwap,
            "bar_close_pos":  round(od.bar_close_pos, 3),
            "scores": {
                "delta":       od.score_delta,
                "cvd":         od.score_cvd,
                "imbalance":   od.score_imbalance,
                "stacked":     od.score_stacked,
                "session":     od.score_session,
                "vwap":        od.score_vwap,
                "vol_spike":   od.score_vol_spike,
                "bar_close":   od.score_bar_close,
                "delta_ma":    od.score_delta_ma,
                "absorption":  od.score_absorption,
            }
        },

        # Absorption
        "absorption": {
            "bullish":       ab.bullish,
            "bearish":       ab.bearish,
            "bonus":         ab.score_bonus,
            "vol_ratio":     ab.volume_ratio,
            "body_vs_atr":   ab.body_vs_atr,
            "delta_pressure":ab.delta_pressure,
            "detail":        ab.detail,
        },

        # Dynamic exit bilgisi
        "dynamic_exit": {
            "atr_value":      dex.atr_value,
            "atr_trail_long": dex.atr_trail_long,
            "atr_trail_short":dex.atr_trail_short,
            "cvd_divergence": dex.cvd_divergence,
            "cvd_div_type":   dex.cvd_div_type,
            "bpr_zone_hit":   dex.bpr_zone_hit,
            "bpr_zone_top":   dex.bpr_zone_top,
            "bpr_zone_bot":   dex.bpr_zone_bot,
        },

        # Çıkış sinyali
        "exit_signal": res.exit_signal,
        "exit_reason": res.exit_reason,

        # Sonuç (15dk sonra doldurulacak)
        "outcome":     None,
    }


# ─── ANA DÖNGÜ ─────────────────────────────────────────────
def run_once(open_position: str = None, entry_price: float = 0.0):
    """
    Tek çalışma:
    1. Veri çek
    2. Analiz et
    3. Sinyal varsa Telegram'a at + JSON'a kaydet
    4. Exit sinyali varsa ayrı Telegram mesajı at
    """
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] Çalışıyor...")

    # Veri çek
    try:
        df = fetch_bybit_klines(SYMBOL, interval="5", limit=300)
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return

    # Analiz
    from signal_engine_v2 import print_signal
    res = analyze(df, symbol=SYMBOL, position=open_position, entry_price=entry_price)
    print_signal(res)  # Detaylı konsol çıktısı

    signals = load_signals()

    # ── EXIT SİNYALİ ──────────────────────────────────────
    if res.exit_signal:
        exit_msg = format_telegram_exit(res)
        if exit_msg:
            ok = send_telegram(exit_msg)
            print(f"  ⚠️  EXIT sinyali Telegram: {'✅' if ok else '❌'}")

            # JSON'a kaydet
            signals.append({**result_to_dict(res), "type": "EXIT"})
            save_signals(signals)
        return  # exit varsa yeni sinyal üretme

    # ── GİRİŞ SİNYALİ ─────────────────────────────────────
    if res.direction == "FLAT":
        print("  FLAT — sinyal yok")
        return

    if is_spam(signals, res.direction, SPAM_MINUTES):
        print(f"  SPAM filtresi — {res.direction} son {SPAM_MINUTES}dk içinde atıldı")
        return

    # Telegram mesajı gönder
    msg = format_telegram_signal(res)
    ok  = send_telegram(msg)
    print(f"  📩 Telegram: {'✅' if ok else '❌'}")

    # JSON'a kaydet
    signals.append({**result_to_dict(res), "type": "SIGNAL"})
    save_signals(signals)
    print(f"  💾 signals.json güncellendi ({len(signals)} kayıt)")


# ─── ENTRY POINT ───────────────────────────────────────────
if __name__ == "__main__":
    """
    Cron job:
    */5 * * * * cd /opt/orderflow && /opt/orderflow/.venv/bin/python signal_logger.py >> bot.log 2>&1

    Manuel test:
    python signal_logger.py

    Açık pozisyonla test:
    POSITION=LONG ENTRY=3400 python signal_logger.py
    """
    pos   = os.environ.get("POSITION", None)
    entry = float(os.environ.get("ENTRY", "0"))
    run_once(open_position=pos, entry_price=entry)