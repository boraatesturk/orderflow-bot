"""
gex_calculator.py — Gamma/Vanna/Charm Exposure Hesaplayıcı
===========================================================
Deribit ETH options verisiyle GEX, VEX, CEX seviyeleri hesaplar.

Kullanım:
    from gex_calculator import fetch_greeks
    levels = fetch_greeks(symbol="ETH")
"""

import requests
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
from typing import Optional

# ─── CONFIG ────────────────────────────────────────────────────────────────
DERIBIT_BASE   = "https://www.deribit.com/api/v2/public"
RISK_FREE_RATE = 0.05
TOP_N_LEVELS   = 8

# ─── BLACK-SCHOLES GREEKS ──────────────────────────────────────────────────
def bs_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Black-Scholes ile Gamma, Vanna, Charm hesapla.
    S: spot, K: strike, T: süre (yıl), r: faiz, sigma: IV
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"gamma": 0.0, "vanna": 0.0, "charm": 0.0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(T)

        # Gamma — delta'nın fiyata göre değişimi
        gamma = pdf_d1 / (S * sigma * sqrt_T)

        # Vanna — delta'nın IV'e göre değişimi
        vanna = -pdf_d1 * d2 / sigma

        # Charm — delta'nın zamana göre değişimi
        charm = -pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)

        return {"gamma": gamma, "vanna": vanna, "charm": charm}
    except Exception:
        return {"gamma": 0.0, "vanna": 0.0, "charm": 0.0}


# ─── DERİBİT API ───────────────────────────────────────────────────────────
def fetch_spot_price(symbol: str = "ETH") -> float:
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


def fetch_book_summary(symbol: str = "ETH") -> list:
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


# ─── ANA HESAPLAMA ─────────────────────────────────────────────────────────
def fetch_greeks(symbol: str = "ETH", max_dte: int = 45) -> list:
    """
    GEX + VEX + CEX seviyelerini hesapla.

    Returns:
        list of dict — her strike için GEX, Vanna, Charm exposure
    """
    print(f"  [GEX] Deribit'ten {symbol} option verisi çekiliyor...")

    spot = fetch_spot_price(symbol)
    if spot == 0:
        print("  [GEX] Spot fiyat alınamadı.")
        return []

    print(f"  [GEX] Spot: ${spot:,.2f}")

    summaries = fetch_book_summary(symbol)
    if not summaries:
        return []

    now_utc     = datetime.now(timezone.utc)
    by_strike   = {}
    processed   = 0
    skipped     = 0

    for s in summaries:
        name = s.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4:
            continue

        try:
            strike      = float(parts[2])
            option_type = parts[3]  # C veya P
            exp_str     = parts[1]  # 29MAY26
            exp_date    = datetime.strptime(exp_str, "%d%b%y").replace(tzinfo=timezone.utc)
            dte         = (exp_date - now_utc).days
        except Exception:
            skipped += 1
            continue

        if dte < 0 or dte > max_dte:
            skipped += 1
            continue

        T     = max(dte / 365.0, 1 / 365.0)
        iv    = s.get("mark_iv", 0) / 100.0
        oi    = s.get("open_interest", 0)

        if iv <= 0 or oi <= 0:
            skipped += 1
            continue

        g = bs_greeks(S=spot, K=strike, T=T, r=RISK_FREE_RATE, sigma=iv)

        # Exposure = greek × OI × spot²
        sign = 1 if option_type == "C" else -1

        gex   = g["gamma"] * oi * spot * spot * sign
        vex   = g["vanna"] * oi * spot          * sign  # Vanna Exposure
        cex   = g["charm"] * oi                 * sign  # Charm Exposure

        if strike not in by_strike:
            by_strike[strike] = {
                "call_gex": 0.0, "put_gex": 0.0,
                "vex": 0.0, "cex": 0.0, "dte": dte
            }

        if option_type == "C":
            by_strike[strike]["call_gex"] += gex
            by_strike[strike]["vex"]      += vex
            by_strike[strike]["cex"]      += cex
        else:
            by_strike[strike]["put_gex"]  += gex  # zaten negatif (sign=-1)
            by_strike[strike]["vex"]      += vex
            by_strike[strike]["cex"]      += cex

        processed += 1

    print(f"  [GEX] İşlenen: {processed} | Atlanan: {skipped}")

    # Sonuçları derle
    results = []
    for strike, data in by_strike.items():
        net_gex = data["call_gex"] + data["put_gex"]
        abs_gex = abs(data["call_gex"]) + abs(data["put_gex"])
        results.append({
            "strike":   strike,
            "call_gex": data["call_gex"],
            "put_gex":  data["put_gex"],
            "net_gex":  net_gex,
            "abs_gex":  abs_gex,
            "vex":      data["vex"],
            "cex":      data["cex"],
            "dte":      data["dte"],
        })

    total_gex = sum(r["net_gex"] for r in results)
    total_vex = sum(r["vex"]     for r in results)
    total_cex = sum(r["cex"]     for r in results)

    print(f"  [GEX] Net GEX: {total_gex/1e6:.1f}M  |  VEX: {total_vex/1e6:.1f}M  |  CEX: {total_cex:.0f}")
    print(f"  [GEX] Piyasa: {'POZ (dar bant)' if total_gex > 0 else 'NEG (volatil)'}")

    # En büyük GEX seviyelerini seç
    results.sort(key=lambda x: x["abs_gex"], reverse=True)
    top = results[:TOP_N_LEVELS]

    # Tip belirle
    for r in top:
        if r["call_gex"] > abs(r["put_gex"]) * 1.5:
            r["type"] = "call_wall"
        elif abs(r["put_gex"]) > r["call_gex"] * 1.5:
            r["type"] = "put_wall"
        else:
            r["type"] = "gex_level"

    top.sort(key=lambda x: x["strike"])

    # Özet
    print(f"\n  {'Strike':>8}  {'Net GEX':>10}  {'VEX':>8}  {'CEX':>8}  {'Tip':>12}  DTE")
    print(f"  {'-'*65}")
    for r in top:
        arrow = "▲" if r["net_gex"] > 0 else "▼"
        print(f"  {r['strike']:>8.0f}  {r['net_gex']/1e6:>+9.1f}M  "
              f"{r['vex']/1e6:>+7.1f}M  {r['cex']:>+8.1f}  "
              f"{r['type']:>12}  {r['dte']}d {arrow}")

    return top


# ─── GERİYE DÖNÜK UYUMLULUK ────────────────────────────────────────────────
def fetch_gex(symbol: str = "ETH", max_dte: int = 45) -> list:
    """Eski kod için — fetch_greeks'i çağırır."""
    return fetch_greeks(symbol, max_dte)


# ─── TEST ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    levels = fetch_greeks("ETH", max_dte=45)
    print(f"\nToplam {len(levels)} seviye.")