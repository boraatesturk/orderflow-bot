"""
gex_calculator.py — Gamma Exposure (GEX) Hesaplayıcı
=====================================================
Deribit ETH options verisiyle GEX seviyeleri hesaplar.

Kullanım:
    from gex_calculator import fetch_gex
    gex_levels = fetch_gex(symbol="ETH")

    # gex_levels listesi:
    # [{"strike": 2000, "gex": 1234567, "type": "call_wall"}, ...]
"""

import requests
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
from typing import Optional

# ─── CONFIG ────────────────────────────────────────────────────────────────
DERIBIT_BASE  = "https://www.deribit.com/api/v2/public"
RISK_FREE_RATE = 0.05   # %5 risksiz faiz (yaklaşık)
TOP_N_LEVELS   = 8      # Kaç seviye gösterilsin

# ─── BLACK-SCHOLES GAMMA ───────────────────────────────────────────────────
def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Gamma hesapla.
    S: spot fiyat
    K: strike
    T: vadeye kalan süre (yıl cinsinden)
    r: risksiz faiz
    sigma: implied volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception:
        return 0.0

# ─── DERİBİT API ───────────────────────────────────────────────────────────
def fetch_spot_price(symbol: str = "ETH") -> float:
    """Spot fiyatı çek."""
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/get_index_price",
            params={"index_name": f"{symbol.lower()}_usd"},
            timeout=8
        )
        return float(r.json()["result"]["index_price"])
    except Exception as e:
        print(f"  [GEX] Spot fiyat hatası: {e}")
        return 0.0


def fetch_instruments(symbol: str = "ETH") -> list:
    """Aktif option kontratlarını çek."""
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/get_instruments",
            params={"currency": symbol, "kind": "option"},
            timeout=10
        )
        return r.json().get("result", [])
    except Exception as e:
        print(f"  [GEX] Instrument hatası: {e}")
        return []


def fetch_ticker(instrument_name: str) -> Optional[dict]:
    """Tek bir option için ticker verisi çek (IV, OI, gamma)."""
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/get_ticker",
            params={"instrument_name": instrument_name},
            timeout=5
        )
        return r.json().get("result", None)
    except Exception:
        return None


def fetch_book_summary(symbol: str = "ETH") -> list:
    """
    Tüm option'lar için özet veri — tek API çağrısıyla.
    Daha hızlı ama gamma yok, sadece OI ve IV var.
    """
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/get_book_summary_by_currency",
            params={"currency": symbol, "kind": "option"},
            timeout=15
        )
        return r.json().get("result", [])
    except Exception as e:
        print(f"  [GEX] Book summary hatası: {e}")
        return []

# ─── GEX HESAPLAMA ─────────────────────────────────────────────────────────
def fetch_gex(symbol: str = "ETH", max_dte: int = 45) -> list:
    """
    Ana fonksiyon — GEX seviyelerini hesapla ve döndür.

    Returns:
        list of dict: [
            {"strike": 2000, "gex": 1234567, "net_gex": 500000,
             "call_gex": 800000, "put_gex": -300000,
             "type": "call_wall", "dte": 7},
            ...
        ]
    """
    print(f"  [GEX] Deribit'ten {symbol} option verisi çekiliyor...")

    # Spot fiyat
    spot = fetch_spot_price(symbol)
    if spot == 0:
        print("  [GEX] Spot fiyat alınamadı, GEX hesaplanamıyor.")
        return []

    print(f"  [GEX] Spot: ${spot:,.2f}")

    # Tüm option'lar için özet veri
    summaries = fetch_book_summary(symbol)
    if not summaries:
        print("  [GEX] Veri alınamadı.")
        return []

    now_utc = datetime.now(timezone.utc)
    gex_by_strike = {}  # strike → {call_gex, put_gex}

    processed = 0
    skipped   = 0

    for s in summaries:
        name = s.get("instrument_name", "")
        if not name:
            continue

        # Instrument adını parse et: ETH-29MAY26-2000-C
        parts = name.split("-")
        if len(parts) != 4:
            continue

        try:
            strike     = float(parts[2])
            option_type = parts[3]  # C veya P
        except Exception:
            continue

        # Vadeye kalan gün
        try:
            exp_ts = s.get("creation_timestamp", 0)  # ms
            # Deribit'te expiration_timestamp instrument'ta var
            # book_summary'de underlying_index'ten alıyoruz
            # Alternatif: instrument adından tarih parse et
            exp_str = parts[1]  # 29MAY26
            exp_date = datetime.strptime(exp_str, "%d%b%y").replace(tzinfo=timezone.utc)
            dte = (exp_date - now_utc).days
        except Exception:
            dte = 0

        # Çok uzak vadeleri atla
        if dte < 0 or dte > max_dte:
            skipped += 1
            continue

        # T (yıl cinsinden)
        T = max(dte / 365.0, 1 / 365.0)

        # IV ve OI
        iv  = s.get("mark_iv", 0) / 100.0  # % → decimal
        oi  = s.get("open_interest", 0)     # ETH cinsinden

        if iv <= 0 or oi <= 0:
            skipped += 1
            continue

        # Gamma hesapla
        gamma = bs_gamma(S=spot, K=strike, T=T, r=RISK_FREE_RATE, sigma=iv)

        # GEX = Gamma × OI × Spot² × 100 (kontrat büyüklüğü)
        # ETH options: 1 kontrat = 1 ETH
        gex = gamma * oi * spot * spot

        if strike not in gex_by_strike:
            gex_by_strike[strike] = {"call_gex": 0.0, "put_gex": 0.0, "dte": dte}

        if option_type == "C":
            gex_by_strike[strike]["call_gex"] += gex
        else:
            gex_by_strike[strike]["put_gex"] -= gex  # Put GEX negatif

        processed += 1

    print(f"  [GEX] İşlenen: {processed} option | Atlanan: {skipped}")

    if not gex_by_strike:
        return []

    # Net GEX hesapla ve sırala
    results = []
    for strike, data in gex_by_strike.items():
        net_gex    = data["call_gex"] + data["put_gex"]
        abs_gex    = abs(data["call_gex"]) + abs(data["put_gex"])
        results.append({
            "strike":   strike,
            "call_gex": data["call_gex"],
            "put_gex":  data["put_gex"],
            "net_gex":  net_gex,
            "abs_gex":  abs_gex,
            "dte":      data["dte"],
        })

    # Toplam GEX
    total_gex = sum(r["net_gex"] for r in results)
    print(f"  [GEX] Toplam Net GEX: {total_gex:,.0f}")
    print(f"  [GEX] Piyasa durumu: {'POZİTİF (dar bant)' if total_gex > 0 else 'NEGATİF (volatil)'}")

    # En büyük abs GEX seviyelerine göre sırala
    results.sort(key=lambda x: x["abs_gex"], reverse=True)
    top = results[:TOP_N_LEVELS]

    # Seviye tipi belirle
    for r in top:
        if r["call_gex"] > abs(r["put_gex"]) * 1.5:
            r["type"] = "call_wall"    # Güçlü direnç
        elif abs(r["put_gex"]) > r["call_gex"] * 1.5:
            r["type"] = "put_wall"     # Güçlü destek
        else:
            r["type"] = "gex_level"    # Nötr seviye

    # Strike'a göre sırala (grafik için)
    top.sort(key=lambda x: x["strike"])

    # Özet yazdır
    print(f"\n  {'Strike':>8}  {'Net GEX':>12}  {'Tip':>12}  {'DTE':>5}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*5}")
    for r in top:
        arrow = "▲" if r["net_gex"] > 0 else "▼"
        print(f"  {r['strike']:>8.0f}  {r['net_gex']:>+12,.0f}  {r['type']:>12}  {r['dte']:>5}d  {arrow}")

    return top


# ─── STANDALONE TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    levels = fetch_gex("ETH", max_dte=45)
    print(f"\nToplam {len(levels)} GEX seviyesi bulundu.")