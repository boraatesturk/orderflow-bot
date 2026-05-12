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
    analyze_mtf,
    format_telegram_signal,
    format_telegram_exit,
    format_telegram_mtf,
    print_mtf,
    SignalResult,
    MTFResult,
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


# ─── BYBIT FUNDING RATE ────────────────────────────────────
def fetch_funding_rate(symbol: str = "ETHUSDT") -> float:
    """Bybit'ten güncel funding rate çek."""
    try:
        url = "https://api.bybit.com/v5/market/funding/history"
        params = {"category": "linear", "symbol": symbol, "limit": 1}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        rows = data.get("result", {}).get("list", [])
        if rows:
            rate = float(rows[0].get("fundingRate", 0.0))
            print(f"  [FR] fundingRate: {rate}")
            return rate
        else:
            print(f"  [FR] Boş liste. API yanıt: {data}")
    except Exception as e:
        print(f"  Funding rate hata: {e}")
    return 0.0


# ─── BYBIT OPEN INTEREST ───────────────────────────────────
def fetch_open_interest(symbol: str = "ETHUSDT") -> tuple[float, float]:
    """Bybit'ten Open Interest çek."""
    try:
        url = "https://api.bybit.com/v5/market/open-interest"
        params = {"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 2}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        rows = data.get("result", {}).get("list", [])
        print(f"  [OI] {len(rows)} kayıt. İlk: {rows[0] if rows else 'YOK'}")
        if len(rows) >= 2:
            return float(rows[0].get("openInterest", 0.0)), float(rows[1].get("openInterest", 0.0))
        elif len(rows) == 1:
            oi = float(rows[0].get("openInterest", 0.0))
            return oi, oi
    except Exception as e:
        print(f"  OI hata: {e}")
    return 0.0, 0.0
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


def is_spam(signals: list, direction: str, price: float, minutes: int = SPAM_MINUTES) -> bool:
    """
    Son SPAM_MINUTES dk içinde aynı yönde sinyal var mı?
    Ayrıca fiyat ATR'nin 2x'inden yakınsa da spam say.
    """
    now = datetime.now(timezone.utc)
    for s in reversed(signals):
        if s.get("direction") != direction:
            continue
        try:
            ts = datetime.fromisoformat(s["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff_min = (now - ts).total_seconds() / 60

            # Zaman filtresi: son 5 dakika
            if diff_min <= minutes:
                return True

            # Fiyat filtresi: son 30 dakika içinde fiyat %0.3'den yakınsa
            if diff_min <= 30:
                prev_price = s.get("entry", 0)
                atr        = s.get("atr", 0)
                if prev_price > 0:
                    price_dist = abs(price - prev_price) / prev_price
                    min_dist   = max(atr * 2 / prev_price, 0.003)  # ATR x2 veya %0.3
                    if price_dist < min_dist:
                        print(f"  SPAM (fiyat yakın: %{price_dist*100:.2f} < %{min_dist*100:.2f})")
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

        # Funding Rate + OI
        "funding_oi": {
            "funding_rate":     res.funding_oi.funding_rate,
            "funding_rate_pct": res.funding_oi.funding_rate_pct,
            "funding_bias":     res.funding_oi.funding_bias,
            "oi_current":       res.funding_oi.oi_current,
            "oi_prev":          res.funding_oi.oi_prev,
            "oi_change_pct":    res.funding_oi.oi_change_pct,
            "oi_trend":         res.funding_oi.oi_trend,
            "score_funding":    res.funding_oi.score_funding,
            "score_oi":         res.funding_oi.score_oi,
        },

        # Sonuç (15dk sonra doldurulacak)
        "outcome":     None,
    }


# ─── OUTCOME TRACKER ───────────────────────────────────────
def fetch_current_price(symbol: str) -> float:
    """Bybit'ten anlık fiyat çek."""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        r = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=8)
        r.raise_for_status()
        data = r.json()
        items = data.get("result", {}).get("list", [])
        if items:
            return float(items[0].get("lastPrice", 0))
    except Exception as e:
        print(f"  Fiyat çekme hatası: {e}")
    return 0.0


def check_outcomes(signals: list, symbol: str) -> tuple[list, bool]:
    """
    outcome=None olan sinyalleri kontrol et.
    TP1/SL vurulduysa güncelle, Telegram bildirimi için liste döndür.
    Returns: (updated_signals, any_updated)
    """
    current_price = fetch_current_price(symbol)
    if current_price == 0:
        return signals, False

    any_updated  = False
    notifications = []

    for s in signals:
        if s.get("type") != "MTF_SIGNAL":
            continue
        if s.get("outcome") is not None:
            continue

        direction = s.get("direction")
        entry     = s.get("entry", 0)
        sl        = s.get("sl", 0)
        tp1       = s.get("tp1", 0)
        tp2       = s.get("tp2", 0)
        tp3       = s.get("tp3", 0)

        if not entry or not sl or not tp1:
            continue

        # Timestamp kontrolü — en az 15 dakika geçmiş mi?
        try:
            ts = datetime.fromisoformat(s["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - ts).total_seconds() / 60
            if elapsed < 15:
                continue
        except Exception:
            continue

        outcome = None
        hit     = None

        if direction == "LONG":
            if current_price <= sl:
                outcome = "SL"
                hit     = sl
            elif current_price >= tp3:
                outcome = "TP3"
                hit     = tp3
            elif current_price >= tp2:
                outcome = "TP2"
                hit     = tp2
            elif current_price >= tp1:
                outcome = "TP1"
                hit     = tp1

        elif direction == "SHORT":
            if current_price >= sl:
                outcome = "SL"
                hit     = sl
            elif current_price <= tp3:
                outcome = "TP3"
                hit     = tp3
            elif current_price <= tp2:
                outcome = "TP2"
                hit     = tp2
            elif current_price <= tp1:
                outcome = "TP1"
                hit     = tp1

        if outcome:
            s["outcome"]       = outcome
            s["outcome_price"] = current_price
            s["outcome_time"]  = datetime.now(TZ).isoformat()
            any_updated        = True
            print(f"  📊 Outcome güncellendi: {direction} {outcome} @ {current_price}")

    return signals, any_updated


# ─── ANA DÖNGÜ ─────────────────────────────────────────────
def run_once(open_position: str = None, entry_price: float = 0.0):
    """
    Tek çalışma:
    1. 3 timeframe veri çek (1H, 15M, 5M)
    2. FR + OI çek
    3. MTF analiz et
    4. Confluence 2/3 veya 3/3 ise Telegram'a at + JSON'a kaydet
    5. Exit sinyali varsa ayrı Telegram mesajı at
    """
    print(f"[{datetime.now(TZ).strftime('%H:%M:%S')}] Çalışıyor...")

    # ── OUTCOME TRACKER ────────────────────────────────────
    signals = load_signals()
    signals, updated = check_outcomes(signals, SYMBOL)
    if updated:
        save_signals(signals)

    # 3 → 5 timeframe veri çek
    try:
        df_5m  = fetch_bybit_klines(SYMBOL, interval="5",   limit=300)
        df_15m = fetch_bybit_klines(SYMBOL, interval="15",  limit=300)
        df_1h  = fetch_bybit_klines(SYMBOL, interval="60",  limit=200)
        df_4h  = fetch_bybit_klines(SYMBOL, interval="240", limit=200)
        df_1d  = fetch_bybit_klines(SYMBOL, interval="D",   limit=100)
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return

    # Funding Rate + OI çek
    funding_rate        = fetch_funding_rate(SYMBOL)
    oi_current, oi_prev = fetch_open_interest(SYMBOL)
    print(f"  Funding: {funding_rate*100:+.4f}%  OI: {oi_current:.0f}")

    # MTF Analiz
    mtf = analyze_mtf(
        df_1h        = df_1h,
        df_15m       = df_15m,
        df_5m        = df_5m,
        df_4h        = df_4h,
        df_1d        = df_1d,
        symbol       = SYMBOL,
        funding_rate = funding_rate,
        oi_current   = oi_current,
        oi_prev      = oi_prev,
        position     = open_position,
        entry_price  = entry_price,
    )
    print_mtf(mtf)  # Detaylı konsol çıktısı

    # signals zaten yüklendi (outcome check'te)

    # ── EXIT SİNYALİ ──────────────────────────────────────
    if mtf.res_5m.exit_signal:
        exit_msg = format_telegram_exit(mtf.res_5m)
        if exit_msg:
            ok = send_telegram(exit_msg)
            print(f"  ⚠️  EXIT sinyali Telegram: {'✅' if ok else '❌'}")
            signals.append({
                "type":      "EXIT",
                "timestamp": str(datetime.now(TZ).isoformat()),
                "symbol":    SYMBOL,
                "reason":    mtf.res_5m.exit_reason,
                "confluence": mtf.confluence,
            })
            save_signals(signals)
        return

    # ── GİRİŞ SİNYALİ ─────────────────────────────────────
    if not mtf.should_send:
        print(f"  Confluence {mtf.confluence}/3 → sinyal yok")
        return

    if is_spam(signals, mtf.direction, mtf.res_5m.entry, SPAM_MINUTES):
        print(f"  SPAM filtresi — {mtf.direction} son {SPAM_MINUTES}dk içinde atıldı")
        return

    # Telegram mesajı gönder
    msg = format_telegram_mtf(mtf)
    ok  = send_telegram(msg)
    print(f"  📩 Telegram: {'✅' if ok else '❌'}")

    # JSON'a kaydet (5M analizi baz alınır)
    res = mtf.res_5m
    rec = result_to_dict(res)
    rec["type"]       = "MTF_SIGNAL"
    rec["confluence"] = mtf.confluence
    rec["note"]       = mtf.note
    rec["tf_1d"]      = {"direction": mtf.res_1d.direction,  "score": mtf.res_1d.score}
    rec["tf_4h"]      = {"direction": mtf.res_4h.direction,  "score": mtf.res_4h.score}
    rec["tf_1h"]      = {"direction": mtf.res_1h.direction,  "score": mtf.res_1h.score}
    rec["tf_15m"]     = {"direction": mtf.res_15m.direction, "score": mtf.res_15m.score}
    rec["tf_5m"]      = {"direction": mtf.res_5m.direction,  "score": mtf.res_5m.score}
    signals.append(rec)
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