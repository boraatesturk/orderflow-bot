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

    day_signals   = [s for s in signals if s.get("time_tr", "").startswith(target_date)]

    if not day_signals:
        return f"📊 <b>{SYMBOL} | {target_date}</b>\n\nBugün hiç sinyal üretilmedi."

    trade_signals = [s for s in day_signals if s["signal"] != "FLAT"]
    flat_count    = sum(1 for s in day_signals if s["signal"] == "FLAT")

    wins   = [s for s in trade_signals if s.get("result") == "✅ KAZANDI"]
    losses = [s for s in trade_signals if s.get("result") == "❌ KAYBETTİ"]
    open_  = [s for s in trade_signals if s.get("result") is None]

    total_evaluated = len(wins) + len(losses)
    win_rate  = (len(wins) / total_evaluated * 100) if total_evaluated > 0 else 0
    total_pnl = sum(s.get("pnl_pct", 0) or 0 for s in trade_signals)

    # ML istatistikleri
    ml_signals    = [s for s in day_signals if s.get("ml_signal") not in (None, "N/A")]
    uyum_count    = sum(1 for s in ml_signals if s.get("ml_uyum") == "UYUM")
    cakisma_count = sum(1 for s in ml_signals if s.get("ml_uyum") == "CAKISMA")
    ml_oncu_count = sum(1 for s in ml_signals if s.get("ml_uyum") == "ML_ONCU")
    # UYUM sinyallerinin win rate'i
    uyum_signals  = [s for s in trade_signals if s.get("ml_uyum") == "UYUM" and s.get("result") in ("✅ KAZANDI","❌ KAYBETTİ")]
    uyum_wr       = (sum(1 for s in uyum_signals if s.get("result") == "✅ KAZANDI") / len(uyum_signals) * 100) if uyum_signals else 0

    buy_count  = sum(1 for s in trade_signals if s["signal"] == "BUY")
    sell_count = sum(1 for s in trade_signals if s["signal"] == "SELL")

    lines = [
        f"📊 <b>{SYMBOL} Günlük Sinyal Raporu</b>",
        f"📅 <b>{target_date}</b>",
        f"",
        f"📈 BUY: {buy_count}  🔴 SELL: {sell_count}  ⬜ FLAT: {flat_count}",
        f"✅ Kazandı: {len(wins)}  ❌ Kaybetti: {len(losses)}  ⏳ Bekliyor: {len(open_)}",
        f"🎯 Win Rate: %{win_rate:.1f}  💰 Toplam P&L: %{total_pnl:+.2f}",
        f"",
        f"🤖 <b>ML Analizi</b>",
        f"🟰 Uyum     : {uyum_count}  ⚡ Çakışma: {cakisma_count}  👀 ML Öncü: {ml_oncu_count}",
        f"🎯 Uyum Win Rate: %{uyum_wr:.1f}" if uyum_signals else f"🎯 Uyum Win Rate: —",
        f"",
        f"─────────────────────────",
        f"<b>SİNYAL DETAYLARI</b>",
        f"─────────────────────────",
    ]

    for s in trade_signals:
        sig    = s["signal"]
        emoji  = "🟢" if sig == "BUY" else "🔴"
        result = s.get("result") or "⏳"
        time   = s["time_tr"].split(" ")[1]
        score  = s["score_buy"] if sig == "BUY" else s["score_sell"]
        pnl    = f"%{s['pnl_pct']:+.3f}" if s.get("pnl_pct") is not None else "—"
        exit_p = f"→ {s['exit_price']}" if s.get("exit_price") else ""
        lev    = f"{s['leverage']}x" if s.get("leverage") else "—"

        # ML bilgisi
        ml_sig  = s.get("ml_signal", "N/A")
        ml_conf = s.get("ml_conf", 0)
        ml_uyum = s.get("ml_uyum", "—")
        if ml_uyum == "UYUM":       ml_emoji = "🟰"
        elif ml_uyum == "CAKISMA":  ml_emoji = "⚡"
        elif ml_uyum == "ML_ONCU":  ml_emoji = "👀"
        else:                        ml_emoji = "➖"

        lines.append(f"{emoji} <b>{sig}</b> | {time} | {s['price']} {exit_p} | {result}")
        lines.append(f"   Skor: {score:.1f}/10 | Kaldirac: {lev} | P&L: {pnl}")
        lines.append(f"   {ml_emoji} ML: {ml_sig} (%{ml_conf}) | {ml_uyum}")

    lines += [
        f"",
        f"🤖 OrderFlow Bot — ETHUSDT 5m",
    ]

    return "\n".join(lines)


ARCHIVE_FILE = "signals_archive.json"

def archive_and_reset(signals: list, target_date: str) -> list:
    """
    Hedef tarihin sinyallerini arsive tasir, signals.json'dan siler.
    Kalan (yeni gun) sinyalleri dondurur.
    """
    import os, json

    day_sigs  = [s for s in signals if s.get("time_tr", "").startswith(target_date)]
    rest_sigs = [s for s in signals if not s.get("time_tr", "").startswith(target_date)]

    if not day_sigs:
        return rest_sigs

    # Arsivi yukle veya bos baslat
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r") as f:
            archive = json.load(f)
    else:
        archive = []

    archive.extend(day_sigs)

    # Arsivi kaydet (max 2000 kayit)
    archive = archive[-2000:]
    with open(ARCHIVE_FILE, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    print(f"[+] {len(day_sigs)} sinyal arsive tasindu: {ARCHIVE_FILE}")
    return rest_sigs


def save_signals(signals: list):
    signals = signals[-200:]
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)


def main():
    now_tr      = datetime.now(TZ_TR)
    target_date = now_tr.strftime("%d/%m/%Y")
    print(f"[{now_tr.strftime('%H:%M:%S')} TR] Gunluk rapor gonderiliyor...")

    signals = load_signals()

    # Once raporu olustur (henuz arsivelenmemis verilerle)
    report = build_daily_report(signals, target_date)
    print(report)
    print()

    # Telegram'a gonder
    ok = send_telegram(report)

    if ok:
        # Basarili gonderim sonrasi: bugunun sinyallerini arsive tas
        remaining = archive_and_reset(signals, target_date)
        save_signals(remaining)
        print(f"[+] signals.json sifirlandi. Kalan (yeni gun): {len(remaining)}")
    else:
        print("[!] Telegram gonderilemedi, signals.json dokunulmadi.")


if __name__ == "__main__":
    main()
