"""
=============================================================================
  TELEGRAM GUNLUK RAPOR — MTF v2
  Her gun saat 23:58 TR'de cron job tarafindan calistirilir.
  signals.json'dan gunun MTF sinyallerini okur, Telegram'a gonderir.
=============================================================================
"""

import json
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

TZ_TR = ZoneInfo("Europe/Istanbul")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SIGNALS_FILE     = "signals.json"
ARCHIVE_FILE     = "signals_archive.json"
SYMBOL           = "ETHUSDT"


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("HATA: TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID eksik!")
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, data=data, timeout=10)
    if r.status_code == 200:
        print("Telegram mesaji gonderildi!")
        return True
    print(f"Telegram hatasi: {r.text}")
    return False


def load_signals() -> list:
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_signals(signals: list):
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(signals[-200:], f, indent=2, ensure_ascii=False)


def archive_and_reset(signals: list, target_date: str) -> list:
    """Bugünün sinyallerini arşive taşı, kalanları döndür."""
    day_sigs  = [s for s in signals if s.get("timestamp", "").startswith(target_date)]
    rest_sigs = [s for s in signals if not s.get("timestamp", "").startswith(target_date)]

    if day_sigs:
        archive = []
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                try:
                    archive = json.load(f)
                except Exception:
                    archive = []
        archive.extend(day_sigs)
        with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(archive[-2000:], f, indent=2, ensure_ascii=False)
        print(f"[+] {len(day_sigs)} sinyal arsive tasindu.")

    return rest_sigs


def build_daily_report(signals: list, target_date: str = None) -> str:
    """
    MTF formatındaki signals.json'dan günlük rapor oluştur.
    target_date: "YYYY-MM-DD" formatında
    """
    if target_date is None:
        target_date = datetime.now(TZ_TR).strftime("%Y-%m-%d")

    display_date = datetime.now(TZ_TR).strftime("%d/%m/%Y")

    # Bugünün sinyallerini filtrele
    day_signals = [s for s in signals if s.get("timestamp", "").startswith(target_date)]

    if not day_signals:
        return f"📊 <b>{SYMBOL} | {display_date}</b>\n\nBugün hiç sinyal üretilmedi."

    # Sinyal tipine göre ayır
    trade_signals = [s for s in day_signals if s.get("type") == "MTF_SIGNAL"]
    exit_signals  = [s for s in day_signals if s.get("type") == "EXIT"]

    if not trade_signals:
        return f"📊 <b>{SYMBOL} | {display_date}</b>\n\nBugün sinyal eşiği geçilemedi."

    long_sigs  = [s for s in trade_signals if s.get("direction") == "LONG"]
    short_sigs = [s for s in trade_signals if s.get("direction") == "SHORT"]

    # Confluence dağılımı
    conf_5 = sum(1 for s in trade_signals if s.get("confluence", 0) == 5)
    conf_4 = sum(1 for s in trade_signals if s.get("confluence", 0) == 4)
    conf_3 = sum(1 for s in trade_signals if s.get("confluence", 0) == 3)

    # Absorption olan sinyaller
    ab_count = sum(1 for s in trade_signals
                   if s.get("absorption", {}).get("bullish") or s.get("absorption", {}).get("bearish"))

    # Ortalama skor
    scores     = [abs(s.get("score", 0)) for s in trade_signals]
    avg_score  = sum(scores) / len(scores) if scores else 0

    # Funding & OI özeti
    funding_vals = [s.get("funding_oi", {}).get("funding_rate_pct", 0) for s in trade_signals]
    avg_funding  = sum(funding_vals) / len(funding_vals) if funding_vals else 0
    last_oi_trend = trade_signals[-1].get("funding_oi", {}).get("oi_trend", "—") if trade_signals else "—"

    lines = [
        f"📊 <b>{SYMBOL} MTF Günlük Rapor</b>",
        f"📅 <b>{display_date}</b>",
        f"",
        f"🟢 LONG: {len(long_sigs)}  🔴 SHORT: {len(short_sigs)}  ⚠️ EXIT: {len(exit_signals)}",
        f"",
        f"🔗 <b>Confluence Dağılımı</b>",
        f"   5/5 → {conf_5} sinyal",
        f"   4/5 → {conf_4} sinyal",
        f"   3/5 → {conf_3} sinyal",
        f"",
        f"📦 Absorption sinyali: {ab_count}",
        f"⭐ Ort. Skor: {avg_score:.2f}",
        f"📈 Ort. Funding: %{avg_funding:+.4f}",
        f"📊 Son OI Trend: {last_oi_trend}",
        f"",
        f"─────────────────────────",
        f"<b>SİNYAL DETAYLARI</b>",
        f"─────────────────────────",
    ]

    for s in trade_signals[-10:]:  # Son 10 sinyal
        direction = s.get("direction", "—")
        emoji     = "🟢" if direction == "LONG" else "🔴"
        ts        = s.get("timestamp", "")
        try:
            t = datetime.fromisoformat(ts).strftime("%H:%M")
        except Exception:
            t = ts[:16]
        score    = s.get("score", 0)
        conf     = s.get("confluence", 0)
        entry    = s.get("entry", 0)
        sl       = s.get("sl", 0)
        tp1      = s.get("tp1", 0)
        lev      = s.get("leverage", "—")
        note     = s.get("note", "")
        note_e   = {"GÜÇLÜ": "🔥", "ORTA": "⚡"}.get(note, "")

        # TF breakdown
        tf_1d  = s.get("tf_1d",  {}).get("direction", "—")
        tf_4h  = s.get("tf_4h",  {}).get("direction", "—")
        tf_1h  = s.get("tf_1h",  {}).get("direction", "—")
        tf_15m = s.get("tf_15m", {}).get("direction", "—")
        tf_5m  = s.get("tf_5m",  {}).get("direction", "—")

        def d_emoji(d):
            return "🟢" if d == "LONG" else ("🔴" if d == "SHORT" else "🟡")

        ab = s.get("absorption", {})
        ab_tag = " 📦" if ab.get("bullish") or ab.get("bearish") else ""

        lines += [
            f"",
            f"{emoji} <b>{direction}</b> | {t} | Skor:{score:+.1f} | {conf}/5 {note_e}{ab_tag}",
            f"   {d_emoji(tf_1d)}1D {d_emoji(tf_4h)}4H {d_emoji(tf_1h)}1H {d_emoji(tf_15m)}15M {d_emoji(tf_5m)}5M",
            f"   💰{entry} 🛑{sl} 🎯{tp1} ⚡{lev}x",
        ]

    if exit_signals:
        lines += ["", f"─────────────────────────", f"<b>EXIT SİNYALLERİ</b>"]
        for s in exit_signals:
            ts = s.get("timestamp", "")
            try:
                t = datetime.fromisoformat(ts).strftime("%H:%M")
            except Exception:
                t = ts[:16]
            lines.append(f"⚠️ {t} | {s.get('reason', '—')}")

    lines += ["", f"🤖 OrderFlow Bot — ETHUSDT MTF"]
    return "\n".join(lines)


def main():
    now_tr      = datetime.now(TZ_TR)
    target_date = now_tr.strftime("%Y-%m-%d")
    print(f"[{now_tr.strftime('%H:%M:%S')} TR] Gunluk rapor gonderiliyor...")

    signals = load_signals()
    report  = build_daily_report(signals, target_date)
    print(report)
    print()

    ok = send_telegram(report)
    if ok:
        remaining = archive_and_reset(signals, target_date)
        save_signals(remaining)
        print(f"[+] signals.json sifirlandi. Kalan (yeni gun): {len(remaining)}")
    else:
        print("[!] Telegram gonderilemedi, signals.json dokunulmadi.")


if __name__ == "__main__":
    main()