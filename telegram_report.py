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


def ts_to_date(ts: str) -> str:
    """Timestamp'ten YYYY-MM-DD tarihi çıkar (timezone'dan bağımsız)."""
    try:
        return datetime.fromisoformat(ts).astimezone(TZ_TR).strftime("%Y-%m-%d")
    except Exception:
        return ts[:10]


def archive_and_reset(signals: list, target_date: str) -> list:
    """Bugünün sinyallerini arşive taşı, kalanları döndür."""
    day_sigs  = [s for s in signals if ts_to_date(s.get("timestamp", "")) == target_date]
    rest_sigs = [s for s in signals if ts_to_date(s.get("timestamp", "")) != target_date]

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
    target_date: YYYY-MM-DD formatında
    """
    if target_date is None:
        target_date = datetime.now(TZ_TR).strftime("%Y-%m-%d")

    display_date = datetime.now(TZ_TR).strftime("%d/%m/%Y")

    # Bugünün sinyallerini filtrele (timezone-safe)
    day_signals = [s for s in signals if ts_to_date(s.get("timestamp", "")) == target_date]

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

    # Outcome istatistikleri
    evaluated = [s for s in trade_signals if s.get("outcome") is not None]
    open_sigs  = [s for s in trade_signals if s.get("outcome") is None]
    wins       = [s for s in evaluated if s.get("outcome") != "SL"]
    losses     = [s for s in evaluated if s.get("outcome") == "SL"]
    win_rate   = len(wins) / len(evaluated) * 100 if evaluated else 0

    tp1_hits = sum(1 for s in evaluated if s.get("outcome") == "TP1")
    tp2_hits = sum(1 for s in evaluated if s.get("outcome") == "TP2")
    tp3_hits = sum(1 for s in evaluated if s.get("outcome") == "TP3")
    sl_hits  = len(losses)

    lines = [
        f"📊 <b>{SYMBOL} MTF Günlük Rapor</b>",
        f"📅 <b>{display_date}</b>",
        f"",
        f"🟢 LONG: {len(long_sigs)}  🔴 SHORT: {len(short_sigs)}  ⚠️ EXIT: {len(exit_signals)}",
        f"",
        f"📈 <b>Sonuçlar</b>",
        f"✅ Kazandı: {len(wins)}  ❌ Kaybetti: {len(losses)}  ⏳ Açık: {len(open_sigs)}",
    ]

    if evaluated:
        lines += [
            f"🎯 Win Rate: %{win_rate:.1f}",
            f"🏆 TP1:{tp1_hits}  TP2:{tp2_hits}  TP3:{tp3_hits}  SL:{sl_hits}",
        ]

    lines += [
        f"",
        f"🔗 <b>Confluence Dağılımı</b>",
        f"   5/5 → {conf_5} sinyal",
        f"   4/5 → {conf_4} sinyal",
        f"   3/5 → {conf_3} sinyal",
        f"",
        f"📦 Absorption: {ab_count}  ⭐ Ort. Skor: {avg_score:.2f}",
        f"📈 Ort. Funding: %{avg_funding:+.4f}  📊 Son OI: {last_oi_trend}",
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
        conf    = s.get("confluence", 0)
        entry   = s.get("entry", 0)
        sl      = s.get("sl", 0)
        tp1     = s.get("tp1", 0)
        lev     = s.get("leverage", "—")
        note_e  = {"GÜÇLÜ": "🔥", "ORTA": "⚡"}.get(s.get("note", ""), "")
        ab      = s.get("absorption", {})
        ab_tag  = "📦" if ab.get("bullish") or ab.get("bearish") else "  "

        outcome = s.get("outcome")
        if outcome is None:
            out_e = "⏳"
        elif outcome == "SL":
            out_e = "❌"
        else:
            out_e = "✅"
        out_str = f"{out_e}{outcome or 'Açık'}"

        tf_1d  = s.get("tf_1d",  {}).get("direction", "—")
        tf_4h  = s.get("tf_4h",  {}).get("direction", "—")
        tf_1h  = s.get("tf_1h",  {}).get("direction", "—")
        tf_15m = s.get("tf_15m", {}).get("direction", "—")
        tf_5m  = s.get("tf_5m",  {}).get("direction", "—")
        de = lambda d: "🟢" if d=="LONG" else ("🔴" if d=="SHORT" else "🟡")

        lines += [
            f"{emoji}{t} {conf}/5{note_e} {ab_tag} <code>{entry:.1f}</code> TP<code>{tp1:.1f}</code> SL<code>{sl:.1f}</code> {lev}x → {out_str}",
            f"   {de(tf_1d)}1D {de(tf_4h)}4H {de(tf_1h)}1H {de(tf_15m)}15M {de(tf_5m)}5M",
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