"""
trade_collector.py
==================
Bybit WebSocket'ten gerçek taker trade verisi toplar.
Her trade'i 5 dakikalık mumlara gruplar, taker_data.json'a yazar.
Systemd servisi olarak sürekli çalışır.

Kullanım:
  python trade_collector.py

Systemd:
  /etc/systemd/system/trade-collector.service
"""

import json
import time
import signal
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

try:
    import websocket
except ImportError:
    print("websocket-client kurulu değil. Çalıştır: pip install websocket-client")
    raise

OUTPUT_FILE  = Path("/opt/orderflow/taker_data.json")
LOG_FILE     = Path("/opt/orderflow/collector.log")
SYMBOL       = "ETHUSDT"
INTERVAL_MS  = 5 * 60 * 1000   # 5 dakika
MAX_CANDLES  = 2000             # Maksimum kaç mum tutulsun (~7 gün)
SAVE_EVERY   = 10               # Her kaç trade'de bir kaydet

# Logging
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─── VERİ YAPISI ─────────────────────────────────────────────
# candle_data: {candle_ts_ms: {"buy": float, "sell": float, "count": int}}
candle_data = defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "count": 0})
data_lock   = threading.Lock()
trade_count = 0
running     = True


def get_candle_ts(trade_ts_ms: int) -> int:
    """Trade timestamp'inden mum başlangıç zamanını hesapla."""
    return (trade_ts_ms // INTERVAL_MS) * INTERVAL_MS


def save_data():
    """candle_data'yı taker_data.json'a kaydet."""
    try:
        with data_lock:
            # Son MAX_CANDLES mumu tut
            sorted_keys = sorted(candle_data.keys())[-MAX_CANDLES:]
            output = {
                str(k): {
                    "buy":   round(candle_data[k]["buy"],   4),
                    "sell":  round(candle_data[k]["sell"],  4),
                    "delta": round(candle_data[k]["buy"] - candle_data[k]["sell"], 4),
                    "count": candle_data[k]["count"],
                    "ts_ms": k,
                    "time":  datetime.fromtimestamp(k/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                for k in sorted_keys
            }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f)

    except Exception as e:
        log.error(f"Kaydetme hatası: {e}")


def on_message(ws, message):
    """Her WebSocket mesajında çağrılır."""
    global trade_count, running

    try:
        data = json.loads(message)

        # Pong / subscription confirmation
        if "op" in data:
            return

        topic = data.get("topic", "")
        if "publicTrade" not in topic:
            return

        trades = data.get("data", [])
        with data_lock:
            for t in trades:
                ts_ms     = int(t["T"])        # trade timestamp
                size      = float(t["v"])      # volume
                side      = t["S"]             # "Buy" veya "Sell"
                candle_ts = get_candle_ts(ts_ms)

                if side == "Buy":
                    candle_data[candle_ts]["buy"]   += size
                else:
                    candle_data[candle_ts]["sell"]  += size
                candle_data[candle_ts]["count"] += 1

        trade_count += len(trades)

        # Her SAVE_EVERY trade'de bir kaydet
        if trade_count % SAVE_EVERY == 0:
            save_data()

        # Her 1000 trade'de bir log
        if trade_count % 1000 == 0:
            with data_lock:
                candle_count = len(candle_data)
            log.info(f"Toplam trade: {trade_count:,} | Mum sayısı: {candle_count}")

    except Exception as e:
        log.error(f"Mesaj işleme hatası: {e}")


def on_error(ws, error):
    log.error(f"WebSocket hata: {error}")


def on_close(ws, close_status_code, close_msg):
    log.warning(f"WebSocket kapandı: {close_status_code} {close_msg}")


def on_open(ws):
    """Bağlantı açılınca subscribe ol."""
    log.info(f"WebSocket bağlandı — {SYMBOL} trade stream'e subscribe olunuyor...")
    subscribe_msg = {
        "op":   "subscribe",
        "args": [f"publicTrade.{SYMBOL}"]
    }
    ws.send(json.dumps(subscribe_msg))


def run_websocket():
    """WebSocket bağlantısını başlat ve yönet."""
    global running

    url = "wss://stream.bybit.com/v5/public/linear"
    websocket.enableTrace(False)

    while running:
        try:
            log.info("WebSocket bağlantısı kuruluyor...")
            ws = websocket.WebSocketApp(
                url,
                on_open    = on_open,
                on_message = on_message,
                on_error   = on_error,
                on_close   = on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.error(f"WebSocket çökme: {e}")

        if running:
            log.info("5 saniye sonra yeniden bağlanılıyor...")
            time.sleep(5)


def signal_handler(signum, frame):
    """SIGTERM/SIGINT ile düzgün kapat."""
    global running
    log.info("Kapatma sinyali alındı...")
    running = False
    save_data()
    log.info(f"Son kayıt yapıldı. Toplam trade: {trade_count:,}")


def main():
    global running

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)

    log.info("=" * 50)
    log.info(f"Trade Collector başlatılıyor — {SYMBOL}")
    log.info(f"Çıktı: {OUTPUT_FILE}")
    log.info("=" * 50)

    # Mevcut veriyi yükle
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                existing = json.load(f)
            with data_lock:
                for k, v in existing.items():
                    candle_data[int(k)]["buy"]   = v["buy"]
                    candle_data[int(k)]["sell"]  = v["sell"]
                    candle_data[int(k)]["count"] = v.get("count", 0)
            log.info(f"Mevcut veri yüklendi: {len(existing)} mum")
        except Exception as e:
            log.warning(f"Mevcut veri yüklenemedi: {e}")

    run_websocket()
    log.info("Trade Collector durduruldu.")


if __name__ == "__main__":
    main()