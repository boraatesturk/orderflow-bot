"""
=============================================================================
  TELEGRAM GUNLUK RAPOR
  Her gun oglen 12:00 TR saatinde GitHub Actions tarafindan calistirilir.
  signals.json'dan gunun sinyallerini okur, Telegram'a gonderir.
=============================================================================
"""

import json
import os
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ_TR = ZoneInfo("Europe/Istanbul")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SIGNALS_FILE     = "signals.json"
SYMBOL           = "ETHUSDT"


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("HATA: TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID eksik!")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    r = requests.post(url, data=data, timeout=10)
    if r.status_code == 200:
        print("Telegram mesaji gonderildi!")
        return True
    else:
        print(f"Telegram hatasi: {r.text}")
        return False


def load_signals() -> list:
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r") as f:
        return json.load(f)


def build_daily_report(signals: list, target_date: str = None) -> str:
    """
    Gunluk raporu olusturur.
    target_date: "DD/MM/YYYY" formatinda, None ise bugun
    """
    if target_date is None:
        target_date = datetime.now(TZ_TR).strftime("%d/%m/%Y")

    # Bugune ait sinyalleri filtrele
    day_signals = [s for s in signals if s.get("time_tr", "").startswith(target_date)]

    if not day_signals:
        return f"📊 <b>{SYMBOL} | {target_date}</b>\n\nBugün hiç sinyal üretilmedi."

    # Sadece BUY/SELL sinyalleri
    trade_signals = [s for s in day_signals if s["signal"] != "FLAT"]
    flat_count    = sum(1 for s in day_signals if s["signal"] == "FLAT")

    # İstatistikler
    wins   = [s for s in trade_signals if s.get("result") == "✅ KAZANDI"]
    losses = [s for s in trade_signals if s.get("result") == "❌ KAYBETTİ"]
    open_  = [s for s in trade_signals if s.get("result") is None]

    total_evaluated = len(wins) + len(losses)
    win_rate = (len(wins) / total_evaluated * 100) if total_evaluated > 0 else 0

    total_pnl = sum(s.get("pnl_pct", 0) or 0 for s in trade_signals)

    # Başlık
    lines = [
        f"📊 <b>{SYMBOL} Günlük Sinyal Raporu</b>",
        f"📅 <b>{target_date}</b>",
        f"",
        f"📈 Toplam sinyal  : {len(trade_signals)} (BUY/SELL)",
        f"⬜ FLAT           : {flat_count}",
        f"✅ Kazandı        : {len(wins)}",
        f"❌ Kaybetti       : {len(losses)}",
        f"⏳ Bekliyor       : {len(open_)}",
        f"🎯 Win Rate       : %{win_rate:.1f}",
        f"💰 Toplam P&L     : %{total_pnl:+.2f}",
        f"",
        f"─────────────────────────",
        f"<b>SİNYAL DETAYLARI</b>",
        f"─────────────────────────",
    ]

    # Her sinyali listele
    for s in trade_signals:
        sig    = s["signal"]
        emoji  = "🟢" if sig == "BUY" else "🔴"
        result = s.get("result") or "⏳"
        time   = s["time_tr"].split(" ")[1]  # Sadece saat

        score  = s["score_buy"] if sig == "BUY" else s["score_sell"]
        pnl    = f"%{s['pnl_pct']:+.3f}" if s.get("pnl_pct") is not None else "—"

        exit_p = f"→ {s['exit_price']}" if s.get("exit_price") else ""

        lines.append(
            f"{emoji} <b>{sig}</b> | {time} | {s['price']} {exit_p}"
        )
        lines.append(
            f"   Skor: {score:.1f}/10 | P&L: {pnl} | {result}"
        )

    # BUY/SELL dagılımı
    buy_count  = sum(1 for s in trade_signals if s["signal"] == "BUY")
    sell_count = sum(1 for s in trade_signals if s["signal"] == "SELL")

    lines += [
        f"",
        f"─────────────────────────",
        f"🟢 BUY  sinyali : {buy_count}",
        f"🔴 SELL sinyali : {sell_count}",
        f"",
        f"🤖 OrderFlow Bot — ETHUSDT 5m",
    ]

    return "\n".join(lines)


def main():
    now_tr = datetime.now(TZ_TR)
    print(f"[{now_tr.strftime('%H:%M:%S')} TR] Gunluk rapor gonderiliyor...")

    signals = load_signals()

    # Bugunun raporu
    report = build_daily_report(signals)
    print(report)
    print()

    send_telegram(report)


if __name__ == "__main__":
    main()
