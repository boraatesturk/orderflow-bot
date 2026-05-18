"""
=============================================================================
  S/R BOLGELER + LIKIDITE HAVUZLARI ANALIZI
  API gerektirmez - OHLCV + Volume Profile bazli
  
  Kullanim:
    python sr_liquidity.py              # Son 150 bar analiz
    python sr_liquidity.py --bars 500   # Son 500 bar
=============================================================================
"""

import pandas as pd
import numpy as np
import requests
import argparse
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

init(autoreset=True)

SYMBOL   = "ETHUSDT"
INTERVAL = "5m"
TZ_TR    = ZoneInfo("Europe/Istanbul")


# ═══════════════════════════════════════════════════════════════════
#  VERİ ÇEK
# ═══════════════════════════════════════════════════════════════════

def fetch_bars(symbol: str, interval: str = "5m", limit: int = 300) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trade_count","taker_buy_volume","taker_buy_quote_volume","_"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","taker_buy_volume"]:
        df[c] = df[c].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df.set_index("open_time", inplace=True)
    df.drop(columns=["_","close_time","quote_volume","taker_buy_quote_volume"], inplace=True, errors="ignore")

    # Delta tahmini
    df["buy_volume"]  = df["taker_buy_volume"]
    df["sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df["delta"]       = df["buy_volume"] - df["sell_volume"]

    return df


# ═══════════════════════════════════════════════════════════════════
#  S/R SEVİYELERİ
# ═══════════════════════════════════════════════════════════════════

def find_sr_levels(df: pd.DataFrame, swing_left: int = 8, swing_right: int = 8,
                   cluster_pct: float = 0.002) -> dict:
    """
    Swing high/low + volume profile ile S/R seviyeleri bulur.
    
    cluster_pct: Birbirine bu kadar yakin seviyeleri birlestir (%0.2)
    
    Returns:
        dict: support[], resistance[], volume_poc, vwap
    """
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    vo = df["volume"].values
    n  = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(swing_left, n - swing_right):
        if hi[i] == max(hi[i - swing_left : i + swing_right + 1]):
            swing_highs.append({"price": hi[i], "idx": i, "volume": vo[i]})
        if lo[i] == min(lo[i - swing_left : i + swing_right + 1]):
            swing_lows.append({"price": lo[i], "idx": i, "volume": vo[i]})

    current_price = cl[-1]

    # Volume Profile - fiyat bolgelerine gore hacim dagitimi
    price_min = df["low"].min()
    price_max = df["high"].max()
    n_bins    = 50
    bins      = np.linspace(price_min, price_max, n_bins + 1)
    vol_profile = np.zeros(n_bins)

    for i in range(n):
        bar_lo  = lo[i]
        bar_hi  = hi[i]
        bar_vol = vo[i]
        for b in range(n_bins):
            bin_lo = bins[b]
            bin_hi = bins[b + 1]
            overlap = max(0, min(bar_hi, bin_hi) - max(bar_lo, bin_lo))
            bar_range = max(bar_hi - bar_lo, 1e-9)
            vol_profile[b] += bar_vol * (overlap / bar_range)

    poc_bin   = np.argmax(vol_profile)
    poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

    # VWAP
    typical  = (df["high"] + df["low"] + df["close"]) / 3
    vwap     = (typical * df["volume"]).sum() / df["volume"].sum()

    # Swing seviyeleri cluster'la (yakin seviyeleri birlestir)
    def cluster_levels(levels, key="price"):
        if not levels:
            return []
        sorted_lvls = sorted(levels, key=lambda x: x[key])
        clusters = []
        current  = [sorted_lvls[0]]

        for lvl in sorted_lvls[1:]:
            ref = current[-1][key]
            if abs(lvl[key] - ref) / ref < cluster_pct:
                current.append(lvl)
            else:
                # Cluster'i volume ile agirlikli ortalama al
                prices = [l[key] for l in current]
                vols   = [l["volume"] for l in current]
                avg_p  = np.average(prices, weights=vols)
                total_v = sum(vols)
                touches = len(current)
                clusters.append({"price": avg_p, "volume": total_v, "touches": touches})
                current = [lvl]

        if current:
            prices = [l[key] for l in current]
            vols   = [l["volume"] for l in current]
            avg_p  = np.average(prices, weights=vols)
            clusters.append({"price": avg_p, "volume": sum(vols), "touches": len(current)})

        return clusters

    sup_clusters = cluster_levels(swing_lows)
    res_clusters = cluster_levels(swing_highs)

    # Fiyatin altindakiler support, ustundekiler resistance
    supports    = sorted([s for s in sup_clusters if s["price"] < current_price],
                         key=lambda x: x["price"], reverse=True)
    resistances = sorted([r for r in res_clusters if r["price"] > current_price],
                         key=lambda x: x["price"])

    return {
        "supports":      supports[:6],     # En yakin 6 support
        "resistances":   resistances[:6],  # En yakin 6 resistance
        "poc":           poc_price,
        "vwap":          vwap,
        "vol_profile":   (bins, vol_profile),
        "current_price": current_price,
    }


# ═══════════════════════════════════════════════════════════════════
#  LİKİDİTE HAVUZLARI
# ═══════════════════════════════════════════════════════════════════

def find_liquidity_pools(df: pd.DataFrame, sr: dict,
                          swing_left: int = 8, swing_right: int = 8) -> dict:
    """
    Tahmini likidite havuzlari:
    
    1. Equal Highs/Lows (EQH/EQL): Ayni seviyede birden fazla swing = stop birikimi
    2. Swing High altindaki buy stop'lar  -> yukari likidite
    3. Swing Low ustundeki sell stop'lar  -> asagi likidite
    4. Hacim agirlikli buyuk swing'ler    -> kurumsal stop avlama bolgesi
    
    Her havuz icin:
        - Fiyat araligi
        - Tahmini yogunluk (touches * volume)
        - Yon (yukari / asagi)
    """
    hi = df["high"].values
    lo = df["low"].values
    vo = df["volume"].values
    n  = len(df)

    current = sr["current_price"]
    tolerance = current * 0.001  # %0.1 tolerans

    swing_highs = []
    swing_lows  = []

    for i in range(swing_left, n - swing_right):
        if hi[i] == max(hi[i - swing_left : i + swing_right + 1]):
            swing_highs.append({"price": hi[i], "idx": i, "volume": vo[i]})
        if lo[i] == min(lo[i - swing_left : i + swing_right + 1]):
            swing_lows.append({"price": lo[i], "idx": i, "volume": vo[i]})

    # Equal Highs (EQH) - Ayni seviyede 2+ swing high = yukari likidite havuzu
    pools_bull = []  # Yukari likidite (buy stop'lar)
    pools_bear = []  # Asagi likidite (sell stop'lar)

    # Swing high'lari grupla
    sh_prices = [s["price"] for s in swing_highs]
    sl_prices = [s["price"] for s in swing_lows]

    def find_equal_levels(levels_list, tol):
        groups = []
        used   = set()
        for i, lvl in enumerate(levels_list):
            if i in used:
                continue
            group = [i]
            for j, other in enumerate(levels_list):
                if j != i and j not in used:
                    if abs(lvl - other) <= tol:
                        group.append(j)
                        used.add(j)
            if len(group) >= 2:
                groups.append(group)
                used.add(i)
        return groups

    # Equal Highs -> yukari likidite (stop buy order'lar burada birikir)
    eq_highs = find_equal_levels(sh_prices, tolerance * 2)
    for grp in eq_highs:
        prices = [sh_prices[i] for i in grp]
        vols   = [swing_highs[i]["volume"] for i in grp if i < len(swing_highs)]
        avg_p  = np.mean(prices)
        density = sum(vols) * len(grp)
        if avg_p > current:
            pools_bull.append({
                "price":    avg_p,
                "range_lo": avg_p - tolerance,
                "range_hi": avg_p + tolerance * 3,
                "density":  density,
                "touches":  len(grp),
                "type":     "EQH (Equal High)",
            })

    # Equal Lows -> asagi likidite (stop sell order'lar burada birikir)
    eq_lows = find_equal_levels(sl_prices, tolerance * 2)
    for grp in eq_lows:
        prices = [sl_prices[i] for i in grp]
        vols   = [swing_lows[i]["volume"] for i in grp if i < len(swing_lows)]
        avg_p  = np.mean(prices)
        density = sum(vols) * len(grp)
        if avg_p < current:
            pools_bear.append({
                "price":    avg_p,
                "range_lo": avg_p - tolerance * 3,
                "range_hi": avg_p + tolerance,
                "density":  density,
                "touches":  len(grp),
                "type":     "EQL (Equal Low)",
            })

    # Hacim bazli buyuk swing'ler (kurumsal likidite bolgesi)
    if swing_highs:
        vol_threshold = np.percentile([s["volume"] for s in swing_highs], 75)
        for s in swing_highs:
            if s["volume"] >= vol_threshold and s["price"] > current:
                pools_bull.append({
                    "price":    s["price"],
                    "range_lo": s["price"] * 0.999,
                    "range_hi": s["price"] * 1.002,
                    "density":  s["volume"],
                    "touches":  1,
                    "type":     "Yuksek Hacimli Swing High",
                })

    if swing_lows:
        vol_threshold = np.percentile([s["volume"] for s in swing_lows], 75)
        for s in swing_lows:
            if s["volume"] >= vol_threshold and s["price"] < current:
                pools_bear.append({
                    "price":    s["price"],
                    "range_lo": s["price"] * 0.998,
                    "range_hi": s["price"] * 1.001,
                    "density":  s["volume"],
                    "touches":  1,
                    "type":     "Yuksek Hacimli Swing Low",
                })

    # Density'e gore sirala, en yakin 5 tane
    pools_bull = sorted(pools_bull, key=lambda x: x["price"])[:5]
    pools_bear = sorted(pools_bear, key=lambda x: x["price"], reverse=True)[:5]

    return {"bull": pools_bull, "bear": pools_bear}


# ═══════════════════════════════════════════════════════════════════
#  TERMİNAL ÇIKTISI
# ═══════════════════════════════════════════════════════════════════

def print_terminal_report(sr: dict, pools: dict):
    current = sr["current_price"]
    now_tr  = datetime.now(TZ_TR).strftime("%d/%m/%Y %H:%M:%S")

    print()
    print(Fore.CYAN + "=" * 62)
    print(Fore.CYAN + f"  S/R + LİKİDİTE ANALİZİ  |  {SYMBOL}  |  {now_tr}")
    print(Fore.CYAN + "=" * 62)
    print(f"  Mevcut Fiyat : {Fore.WHITE}{current:.2f} USDT")
    print(f"  VWAP         : {Fore.YELLOW}{sr['vwap']:.2f} USDT")
    print(f"  POC          : {Fore.YELLOW}{sr['poc']:.2f} USDT")
    print()

    # RESISTANCE
    print(Fore.RED + "  ── RESISTANCE (Direnc) ─────────────────────────────")
    if sr["resistances"]:
        for r in reversed(sr["resistances"]):
            dist = (r["price"] - current) / current * 100
            strength = "★★★" if r["touches"] >= 3 else "★★ " if r["touches"] == 2 else "★  "
            print(f"  {Fore.RED}{r['price']:>10.2f} USDT  +{dist:.2f}%  {strength}  ({r['touches']} dokunma)")
    else:
        print(f"  {Fore.YELLOW}  Resistance bulunamadi")

    print()
    print(f"  {'─'*20} {Fore.WHITE}{current:.2f} ◄ FIYAT {'─'*20}")
    print()

    # SUPPORT
    print(Fore.GREEN + "  ── SUPPORT (Destek) ────────────────────────────────")
    if sr["supports"]:
        for s in sr["supports"]:
            dist = (current - s["price"]) / current * 100
            strength = "★★★" if s["touches"] >= 3 else "★★ " if s["touches"] == 2 else "★  "
            print(f"  {Fore.GREEN}{s['price']:>10.2f} USDT  -{dist:.2f}%  {strength}  ({s['touches']} dokunma)")
    else:
        print(f"  {Fore.YELLOW}  Support bulunamadi")

    print()
    print(Fore.CYAN + "=" * 62)
    print(Fore.CYAN + "  LİKİDİTE HAVUZLARI")
    print(Fore.CYAN + "=" * 62)

    # YUKARI LİKİDİTE
    print(Fore.GREEN + "  ▲ YUKARI LİKİDİTE (Buy Stop Bolgesi)")
    if pools["bull"]:
        for p in reversed(pools["bull"]):
            dist   = (p["price"] - current) / current * 100
            yog    = "YUKSEK" if p["touches"] >= 2 else "NORMAL"
            print(f"  {Fore.GREEN}  {p['price']:>10.2f} USDT  +{dist:.2f}%  [{p['range_lo']:.2f} - {p['range_hi']:.2f}]")
            print(f"  {Fore.GREEN}  {'':>10}        Tip: {p['type']}  |  Yogunluk: {yog}")
    else:
        print(f"  {Fore.YELLOW}  Yukari likidite havuzu bulunamadi")

    print()
    print(f"  {'─'*20} {Fore.WHITE}{current:.2f} ◄ FIYAT {'─'*20}")
    print()

    # ASAGI LİKİDİTE
    print(Fore.RED + "  ▼ ASAGI LİKİDİTE (Sell Stop Bolgesi)")
    if pools["bear"]:
        for p in pools["bear"]:
            dist = (current - p["price"]) / current * 100
            yog  = "YUKSEK" if p["touches"] >= 2 else "NORMAL"
            print(f"  {Fore.RED}  {p['price']:>10.2f} USDT  -{dist:.2f}%  [{p['range_lo']:.2f} - {p['range_hi']:.2f}]")
            print(f"  {Fore.RED}  {'':>10}        Tip: {p['type']}  |  Yogunluk: {yog}")
    else:
        print(f"  {Fore.YELLOW}  Asagi likidite havuzu bulunamadi")

    print(Fore.CYAN + "=" * 62)
    print()


# ═══════════════════════════════════════════════════════════════════
#  MATPLOTLİB CHART
# ═══════════════════════════════════════════════════════════════════

def plot_chart(df: pd.DataFrame, sr: dict, pools: dict):
    fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
    gs  = GridSpec(1, 5, figure=fig)

    ax_candle = fig.add_subplot(gs[0, :4])  # Mum grafigi (genis)
    ax_vp     = fig.add_subplot(gs[0, 4])   # Volume profile (dar)

    ax_candle.set_facecolor("#0d1117")
    ax_vp.set_facecolor("#0d1117")

    current = sr["current_price"]
    bins, vol_profile = sr["vol_profile"]

    # ── MUM GRAFİGİ ───────────────────────────────────────────────
    for i, (idx, row) in enumerate(df.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        # Gövde
        ax_candle.add_patch(mpatches.Rectangle(
            (i - 0.3, min(row["open"], row["close"])),
            0.6, abs(row["close"] - row["open"]),
            color=color, zorder=2
        ))
        # Fitil
        ax_candle.plot([i, i], [row["low"], row["high"]], color=color, lw=0.8, zorder=1)

    n = len(df)

    # ── S/R ÇİZGİLERİ ────────────────────────────────────────────
    for r in sr["resistances"]:
        ax_candle.axhline(r["price"], color="#ef5350", lw=1.2, ls="--", alpha=0.8, zorder=3)
        ax_candle.text(n + 0.5, r["price"], f"R {r['price']:.1f}", color="#ef5350",
                       fontsize=7.5, va="center", fontweight="bold")

    for s in sr["supports"]:
        ax_candle.axhline(s["price"], color="#26a69a", lw=1.2, ls="--", alpha=0.8, zorder=3)
        ax_candle.text(n + 0.5, s["price"], f"S {s['price']:.1f}", color="#26a69a",
                       fontsize=7.5, va="center", fontweight="bold")

    # VWAP
    ax_candle.axhline(sr["vwap"], color="#ffeb3b", lw=1.5, ls="-", alpha=0.9, zorder=3)
    ax_candle.text(n + 0.5, sr["vwap"], f"VWAP {sr['vwap']:.1f}",
                   color="#ffeb3b", fontsize=7.5, va="center")

    # POC
    ax_candle.axhline(sr["poc"], color="#ff9800", lw=1.5, ls="-.", alpha=0.9, zorder=3)
    ax_candle.text(n + 0.5, sr["poc"], f"POC {sr['poc']:.1f}",
                   color="#ff9800", fontsize=7.5, va="center")

    # ── LİKİDİTE HAVUZLARI ───────────────────────────────────────
    for p in pools["bull"]:
        ax_candle.axhspan(p["range_lo"], p["range_hi"],
                          alpha=0.15, color="#4caf50", zorder=0)
        ax_candle.axhline(p["price"], color="#4caf50", lw=1, ls=":", alpha=0.6)
        ax_candle.text(1, p["price"], f"⚡ LIQ {p['price']:.1f}",
                       color="#4caf50", fontsize=7, va="center")

    for p in pools["bear"]:
        ax_candle.axhspan(p["range_lo"], p["range_hi"],
                          alpha=0.15, color="#f44336", zorder=0)
        ax_candle.axhline(p["price"], color="#f44336", lw=1, ls=":", alpha=0.6)
        ax_candle.text(1, p["price"], f"⚡ LIQ {p['price']:.1f}",
                       color="#f44336", fontsize=7, va="center")

    # Mevcut fiyat
    ax_candle.axhline(current, color="#ffffff", lw=1.5, ls="-", alpha=1, zorder=5)
    ax_candle.text(n + 0.5, current, f"◄ {current:.2f}",
                   color="#ffffff", fontsize=8, va="center", fontweight="bold")

    # X ekseni - zaman etiketleri
    step   = max(1, n // 10)
    xticks = list(range(0, n, step))
    xlabels = [df.index[i].astimezone(TZ_TR).strftime("%H:%M") for i in xticks]
    ax_candle.set_xticks(xticks)
    ax_candle.set_xticklabels(xlabels, color="#aaaaaa", fontsize=7)
    ax_candle.set_xlim(-1, n + 12)
    ax_candle.tick_params(axis="y", colors="#aaaaaa", labelsize=7)
    ax_candle.spines[:].set_color("#333333")
    ax_candle.set_title(f"{SYMBOL} — S/R & Likidite Haritasi  ({datetime.now(TZ_TR).strftime('%d/%m/%Y %H:%M')} TR)",
                        color="#ffffff", fontsize=11, pad=10)

    # ── VOLUME PROFILE ────────────────────────────────────────────
    bin_mids = (bins[:-1] + bins[1:]) / 2
    max_vol  = vol_profile.max()

    colors_vp = []
    for mid in bin_mids:
        if mid > current:
            colors_vp.append("#ef5350")
        else:
            colors_vp.append("#26a69a")

    ax_vp.barh(bin_mids, vol_profile, height=(bins[1] - bins[0]) * 0.9,
               color=colors_vp, alpha=0.8)

    # POC vurgula
    poc_bin = np.argmax(vol_profile)
    ax_vp.barh(bin_mids[poc_bin], vol_profile[poc_bin],
               height=(bins[1] - bins[0]) * 0.9, color="#ff9800", alpha=1.0,
               label="POC")

    ax_vp.axhline(current, color="#ffffff", lw=1.2)
    ax_vp.set_xlim(0, max_vol * 1.1)
    ax_vp.set_ylim(ax_candle.get_ylim())
    ax_vp.tick_params(axis="y", colors="#aaaaaa", labelsize=6)
    ax_vp.tick_params(axis="x", colors="#555555", labelsize=5)
    ax_vp.set_title("Vol Profile", color="#aaaaaa", fontsize=8)
    ax_vp.spines[:].set_color("#333333")

    plt.tight_layout()

    # Telegram için dosyaya kaydet
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="S/R + Likidite Analizi")
    parser.add_argument("--bars", type=int, default=150, help="Kac bar analiz edilsin (varsayilan: 150)")
    parser.add_argument("--no-chart", action="store_true", help="Sadece terminal ciktisi")
    args = parser.parse_args()

    print(Fore.CYAN + f"\n  {SYMBOL} verisi cekiliyor ({args.bars} bar)...", end=" ")
    df = fetch_bars(SYMBOL, interval=INTERVAL, limit=args.bars)
    
    # Henuz kapanmamis son bari cikar
    now_utc = datetime.now(timezone.utc)
    minutes = now_utc.replace(second=0, microsecond=0).minute
    bar_min = (minutes // 5) * 5
    current_bar = now_utc.replace(minute=bar_min, second=0, microsecond=0)
    df = df[df.index < current_bar]

    print(Fore.GREEN + "OK")

    sr    = find_sr_levels(df)
    pools = find_liquidity_pools(df, sr)

    print_terminal_report(sr, pools)

    if not args.no_chart:
        path = plot_chart(df, sr, pools)
        if path:
            import subprocess, sys, shutil
            # Kalici konuma kopyala
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_liquidity_chart.png")
            shutil.copy(path, out)
            os.unlink(path)
            print(Fore.GREEN + f"  Grafik kaydedildi: {out}")
            # Ekranda ac
            if sys.platform == "win32":
                os.startfile(out)
            elif sys.platform == "darwin":
                subprocess.run(["open", out])
            else:
                subprocess.run(["xdg-open", out])


if __name__ == "__main__":
    main()