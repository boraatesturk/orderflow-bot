"""
telegram_listener.py — Telegram Bot Komut Dinleyici
=====================================================
Desteklenen komutlar:
  /sinyal  → Anlık signal_engine_v2 analizi
  /durum   → Açık pozisyon + son sinyal durumu

Çalıştırma (VPS'te systemd servisi olarak):
  python telegram_listener.py

Systemd servisi için: telegram-listener.service
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from signal_engine_v2 import (
    analyze,
    analyze_mtf,
    format_telegram_signal,
    format_telegram_mtf,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────
SYMBOL        = "ETHUSDT"
SIGNALS_FILE  = "signals.json"
TZ            = ZoneInfo("Europe/Istanbul")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

POLL_INTERVAL = 2   # saniye — Telegram'ı kaç saniyede bir kontrol et

# ─── TELEGRAM YARDIMCI FONKSİYONLARI ───────────────────────────────────────
def send_message(chat_id: str, text: str) -> bool:
    if not TELEGRAM_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "Markdown"
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [send_message hata] {e}")
        return False


def get_updates(offset: int) -> list:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        r = requests.get(url, params={
            "offset":  offset,
            "timeout": 30
        }, timeout=35)
        if r.status_code == 200:
            return r.json().get("result", [])
    except Exception as e:
        print(f"  [get_updates hata] {e}")
    return []


# ─── VERİ ÇEKİCİ ───────────────────────────────────────────────────────────
def fetch_live_data(symbol: str = "ETHUSDT", limit: int = 300) -> pd.DataFrame:
    """Binance'den gerçek taker data ile OHLCV çek."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "5m", "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_volume", "taker_buy_quote_volume", "_"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        df[c] = df[c].astype(float)
    df.set_index("open_time", inplace=True)
    df.drop(columns=["_", "close_time", "quote_volume",
                     "taker_buy_quote_volume"], inplace=True, errors="ignore")

    df["buy_volume"]      = df["taker_buy_volume"]
    df["sell_volume"]     = df["volume"] - df["buy_volume"]
    df["delta"]           = df["buy_volume"] - df["sell_volume"]
    df["imbalance_ratio"] = df["buy_volume"] / (df["volume"] + 1e-9)

    # taker_data.json varsa WebSocket verisiyle güncelle
    taker_path = Path("taker_data.json")
    if taker_path.exists():
        try:
            with open(taker_path) as f:
                taker_raw = json.load(f)
            taker_df = pd.DataFrame(taker_raw)
            taker_df["timestamp"] = pd.to_datetime(taker_df["timestamp"], utc=True)
            taker_df.set_index("timestamp", inplace=True)
            common = df.index.intersection(taker_df.index)
            if len(common) > 0:
                df.loc[common, "buy_volume"]      = taker_df.loc[common, "buy_volume"]
                df.loc[common, "sell_volume"]     = taker_df.loc[common, "sell_volume"]
                df.loc[common, "delta"]           = taker_df.loc[common, "buy_volume"] - taker_df.loc[common, "sell_volume"]
                df.loc[common, "imbalance_ratio"] = taker_df.loc[common, "buy_volume"] / (df.loc[common, "volume"] + 1e-9)
        except Exception:
            pass

    df["cvd"]           = df["delta"].cumsum()
    df["session_delta"] = df["delta"].rolling(12).sum()
    df["vwap"]          = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["poc_price"]     = df["close"]

    from signal_engine_v2 import CFG
    bull_streak = (df["imbalance_ratio"] > CFG["imbalance_bull"]).astype(int)
    bear_streak = (df["imbalance_ratio"] < CFG["imbalance_bear"]).astype(int)
    df["stacked_imbalance_up"] = bull_streak.rolling(3).sum() == 3
    df["stacked_imbalance_dn"] = bear_streak.rolling(3).sum() == 3

    return df


# ─── KOMUT İŞLEYİCİLER ─────────────────────────────────────────────────────
def handle_sinyal(chat_id: str):
    """Anlık sinyal analizi üret ve gönder."""
    send_message(chat_id, "⏳ Analiz yapılıyor, bekle...")
    try:
        df = fetch_live_data()
        result = analyze(df, symbol=SYMBOL)
        msg = format_telegram_signal(result)
        if not msg:
            msg = f"📊 *{SYMBOL}* — Sinyal yok (FLAT)\nSkor: `{result.score:.2f}`"
        send_message(chat_id, msg)
        print(f"  [/sinyal] Gönderildi → {result.direction} {result.score:.2f}")
    except Exception as e:
        send_message(chat_id, f"❌ Analiz hatası: `{e}`")
        print(f"  [/sinyal HATA] {e}")


def handle_durum(chat_id: str):
    """Açık pozisyon ve son sinyal durumunu gönder."""
    try:
        signals_path = Path(SIGNALS_FILE)
        if not signals_path.exists():
            send_message(chat_id, "📭 Henüz sinyal yok.")
            return

        with open(signals_path) as f:
            signals = json.load(f)

        if not signals:
            send_message(chat_id, "📭 signals.json boş.")
            return

        # Açık pozisyon var mı?
        open_pos = None
        for s in reversed(signals):
            if s.get("type") == "MTF_SIGNAL" and s.get("outcome") is None:
                open_pos = s
                break

        now_tr = datetime.now(TZ).strftime("%H:%M %d/%m")
        lines  = [f"📊 *{SYMBOL} Durum* — `{now_tr}`\n"]

        if open_pos:
            direction = open_pos.get("direction", "?")
            entry     = open_pos.get("entry", 0)
            sl        = open_pos.get("sl", 0)
            tp1       = open_pos.get("tp1", 0)
            conf      = open_pos.get("confluence", 0)
            ts        = open_pos.get("timestamp", "")[:16]
            emoji     = "🟢" if direction == "LONG" else "🔴"

            # Anlık fiyat çek
            try:
                r = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": SYMBOL}, timeout=5)
                current = float(r.json()["price"])
                if direction == "LONG":
                    pnl_pct = (current - entry) / entry * 100
                else:
                    pnl_pct = (entry - current) / entry * 100
                pnl_emoji = "✅" if pnl_pct > 0 else "❌"
                pnl_str   = f"{pnl_emoji} PnL: `%{pnl_pct:+.2f}` (anlık `{current:.2f}`)"
            except Exception:
                pnl_str = ""

            lines.append(f"{emoji} *Açık Pozisyon:* {direction} ({conf}/5)")
            lines.append(f"💰 Giriş: `{entry:.2f}`")
            lines.append(f"🛑 SL: `{sl:.2f}`  🎯 TP1: `{tp1:.2f}`")
            if pnl_str:
                lines.append(pnl_str)
            lines.append(f"🕐 Açılış: `{ts}`")
        else:
            lines.append("⏸️ Açık pozisyon yok")

        # Son 3 sinyal
        recent = [s for s in signals if s.get("type") == "MTF_SIGNAL"][-3:]
        if recent:
            lines.append("\n📋 *Son Sinyaller:*")
            for s in reversed(recent):
                d       = s.get("direction", "?")
                outcome = s.get("outcome") or "⏳"
                entry   = s.get("entry", 0)
                conf    = s.get("confluence", 0)
                ts      = s.get("timestamp", "")[:16]
                emoji   = "🟢" if d == "LONG" else "🔴"
                outcome_emoji = {
                    "TP1": "✅TP1", "TP2": "✅✅TP2", "TP3": "🏆TP3",
                    "SL":  "❌SL",  "TRAIL_EXIT": "🏁Trail"
                }.get(outcome, outcome)
                lines.append(f"  {emoji} {d} `{entry:.2f}` {conf}/5 → {outcome_emoji} `{ts}`")

        send_message(chat_id, "\n".join(lines))
        print(f"  [/durum] Gönderildi")

    except Exception as e:
        send_message(chat_id, f"❌ Durum hatası: `{e}`")
        print(f"  [/durum HATA] {e}")


# ─── ANA DÖNGÜ ──────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        print("[HATA] TELEGRAM_TOKEN bulunamadı. /etc/environment kontrol et.")
        return

    print(f"[telegram_listener] Başlatıldı — {SYMBOL}")
    print(f"  Komutlar: /sinyal  /durum")
    print(f"  Poll interval: {POLL_INTERVAL}s")

    offset = 0

    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1

                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue

                chat_id = str(msg["chat"]["id"])
                text    = msg.get("text", "").strip().lower()
                user    = msg.get("from", {}).get("username", "?")

                # Sadece kendi chat'inden gelen komutları işle
                if chat_id != str(TELEGRAM_CHAT_ID):
                    print(f"  [güvenlik] Bilinmeyen chat: {chat_id} ({user})")
                    continue

                print(f"  [{user}] {text}")

                if text in ["/sinyal", "/signal"]:
                    handle_sinyal(chat_id)
                elif text in ["/durum", "/status"]:
                    handle_durum(chat_id)
                elif text in ["/yardim", "/help", "/start"]:
                    send_message(chat_id, (
                        "🤖 *OrderFlow Bot Komutları*\n\n"
                        "/sinyal — Anlık orderflow analizi\n"
                        "/durum  — Açık pozisyon + son sinyaller"
                    ))

        except Exception as e:
            print(f"  [ana döngü hata] {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
