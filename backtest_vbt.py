"""
ORDERFLOW BOT - Gerçekçi Backtest (vectorbt)
=============================================
Kullanim:
    pip install vectorbt
    python backtest_vbt.py

Mevcut backtest'ten farklari:
    - Komisyon dahil (%0.05 taker, her iki taraf)
    - Slippage dahil (%0.02)
    - TP1 / SL ile cikis (sabit 3 bar degil)
    - Kaldiraca gore pozisyon boyutu
    - Equity curve, drawdown, Sharpe ratio
    - Her ay ayri performans ozeti
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import vectorbt as vbt
except ImportError:
    print("[HATA] vectorbt kurulu degil.")
    print("Calistir: pip install vectorbt")
    raise

# ─── AYARLAR ────────────────────────────────────────────────────────────────
PARQUET_PATH = Path("data/ETHUSDT_orderflow_365d_5m.parquet")

BALANCE       = 2500.0    # Baslangic bakiye (USDT)
LEVERAGE      = 10        # Sabit kaldirac (basitlik icin)
RISK_PCT      = 0.02      # Her trade icin bakilacak maksimum risk (%2)

# Bybit taker fee %0.06, maker %0.01 — taker kullaniyoruz
COMMISSION    = 0.0006    # %0.06 (her iki taraf = %0.12 round trip)
SLIPPAGE      = 0.0002    # %0.02 slippage

# TP / SL (ATR bazli degil, basit yuzde)
SL_PCT        = 0.005     # %0.5 stop loss
TP1_PCT       = 0.009     # %0.9 take profit 1 (1.8R)
TP2_PCT       = 0.015     # %1.5 take profit 2 (3R)

# Sinyal parametreleri
MIN_SCORE_BUY  = 7.0
MIN_SCORE_SELL = 8.5      # SHORT icin daha yuksek esik

# Saat filtresi (UTC) — bu saatler arasi sinyal uretme
BLOCKED_HOURS = [2, 3, 4]  # Asya gece, dusuk win rate

CFG = {
    "cvd_lookback":      5,
    "imbalance_bull":    0.55,
    "imbalance_bear":    0.45,
    "volume_spike_mult": 1.5,
}
# ────────────────────────────────────────────────────────────────────────────


def load_data():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet bulunamadi: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df.sort_index(inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    print(f"[+] Veri: {len(df):,} bar | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def add_features(df):
    df = df.copy()
    df["cvd_change"]    = df["cvd"].diff(CFG["cvd_lookback"])
    df["vol_ma20"]      = df["volume"].rolling(20).mean()
    df["vol_spike"]     = df["volume"] / (df["vol_ma20"] + 1e-9)
    df["delta_ma3"]     = df["delta"].rolling(3).mean()
    df["delta_ma10"]    = df["delta"].rolling(10).mean()
    df["delta_pct"]     = df["delta"] / (df["volume"] + 1e-9) * 100
    bar_range           = df["high"] - df["low"]
    df["close_pos"]     = np.where(bar_range > 0, (df["close"] - df["low"]) / bar_range, 0.5)
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-9) * 100
    return df


def compute_score(df):
    """Vektorize sinyal skoru hesapla — her bar icin."""
    sb = pd.Series(0.0, index=df.index)
    ss = pd.Series(0.0, index=df.index)

    # Kural 1: Delta yonu
    delta_weight = (df["delta_pct"].abs() / 10).clip(0, 1.5)
    sb += np.where(df["delta"] > 0, delta_weight, 0)
    ss += np.where(df["delta"] < 0, delta_weight, 0)

    # Kural 2: CVD momentum
    sb += np.where(df["cvd_change"] > 0, 1.0, 0)
    ss += np.where(df["cvd_change"] < 0, 1.0, 0)

    # Kural 3: Imbalance
    sb += np.where(df["imbalance_ratio"] > CFG["imbalance_bull"], 1.5, 0)
    ss += np.where(df["imbalance_ratio"] < CFG["imbalance_bear"], 1.5, 0)

    # Kural 4: Stacked imbalance
    sb += np.where(df["stacked_imbalance_up"].astype(bool), 2.0, 0)
    ss += np.where(df["stacked_imbalance_dn"].astype(bool), 2.0, 0)

    # Kural 5: Session delta
    sb += np.where(df["session_delta"] > 0, 0.75, 0)
    ss += np.where(df["session_delta"] < 0, 0.75, 0)

    # Kural 6: VWAP
    sb += np.where(df["price_vs_vwap"] > 0.05, 0.5, 0)
    ss += np.where(df["price_vs_vwap"] < -0.05, 0.5, 0)

    # Kural 7: Volume spike
    spike = df["vol_spike"] > CFG["volume_spike_mult"]
    sb += np.where(spike & (df["delta"] > 0), 1.0, 0)
    ss += np.where(spike & (df["delta"] < 0), 1.0, 0)

    # Kural 8: Bar kapanis pozisyonu
    sb += np.where(df["close_pos"] > 0.75, 0.5, 0)
    ss += np.where(df["close_pos"] < 0.25, 0.5, 0)

    # Kural 9: Delta MA crossover
    sb += np.where((df["delta_ma3"] > df["delta_ma10"]) & (df["delta_ma3"] > 0), 0.5, 0)
    ss += np.where((df["delta_ma3"] < df["delta_ma10"]) & (df["delta_ma3"] < 0), 0.5, 0)

    # 0-10 normalize
    sb = (sb / 10 * 10).clip(0, 10)
    ss = (ss / 10 * 10).clip(0, 10)

    return sb, ss


def generate_signals(df):
    sb, ss = compute_score(df)

    # Saat filtresi — dusuk win rate saatlerinde sinyal kapatiliyor
    hour = df.index.hour
    hour_ok = ~pd.Series(hour, index=df.index).isin(BLOCKED_HOURS)

    buy_signal  = (sb >= MIN_SCORE_BUY)  & (sb > ss) & hour_ok
    sell_signal = (ss >= MIN_SCORE_SELL) & (ss > sb) & hour_ok

    # Ayni barda ikisi de tetiklenmesin
    conflict    = buy_signal & sell_signal
    buy_signal  = buy_signal  & ~conflict
    sell_signal = sell_signal & ~conflict

    return buy_signal, sell_signal, sb, ss


def run_backtest_vbt(df):
    print("[*] Sinyaller hesaplaniyor...")
    df = add_features(df)
    df.dropna(inplace=True)

    buy_sig, sell_sig, sb, ss = generate_signals(df)

    print(f"[+] BUY sinyali  : {buy_sig.sum():,} bar")
    print(f"[+] SELL sinyali : {sell_sig.sum():,} bar")
    print(f"[+] Toplam sinyal: {buy_sig.sum() + sell_sig.sum():,} bar")

    close = df["close"].astype(float)

    # ── TP/SL seviyeleri ─────────────────────────────────────────
    # Longlarda: SL asagi, TP yukarı
    # Shortlarda: SL yukari, TP asagi
    sl_long  = close * (1 - SL_PCT)
    tp_long  = close * (1 + TP1_PCT)
    sl_short = close * (1 + SL_PCT)
    tp_short = close * (1 - TP1_PCT)

    # ── vectorbt portfolio ───────────────────────────────────────
    print("[*] Backtest calistiriliyor...")

    # LONG portfolio — TP veya SL vurana kadar bekle
    pf_long = vbt.Portfolio.from_signals(
        close        = close,
        entries      = buy_sig,
        exits        = pd.Series(False, index=close.index),  # manuel cikis yok
        sl_stop      = SL_PCT,
        tp_stop      = TP1_PCT,
        fees         = COMMISSION,
        slippage     = SLIPPAGE,
        init_cash    = BALANCE,
        size         = BALANCE * LEVERAGE * RISK_PCT / close,
        size_type    = "amount",
        freq         = "5min",
    )

    # SHORT portfolio — TP veya SL vurana kadar bekle
    pf_short = vbt.Portfolio.from_signals(
        close         = close,
        entries       = pd.Series(False, index=close.index),
        exits         = pd.Series(False, index=close.index),
        short_entries = sell_sig,
        short_exits   = pd.Series(False, index=close.index),
        sl_stop       = SL_PCT,
        tp_stop       = TP1_PCT,
        fees          = COMMISSION,
        slippage      = SLIPPAGE,
        init_cash     = BALANCE,
        size          = BALANCE * LEVERAGE * RISK_PCT / close,
        size_type     = "amount",
        freq          = "5min",
    )

    return pf_long, pf_short, df, buy_sig, sell_sig


def print_stats(pf_long, pf_short):
    print(f"\n{'='*60}")
    print(f"  BACKTEST SONUCLARI — {BALANCE:.0f}$ bakiye, {LEVERAGE}x kaldirac")
    print(f"  Komisyon: %{COMMISSION*100:.2f} | Slippage: %{SLIPPAGE*100:.2f}")
    print(f"  SL: %{SL_PCT*100:.1f} | TP: %{TP1_PCT*100:.2f}")
    print(f"{'='*60}")

    for name, pf in [("LONG (BUY)", pf_long), ("SHORT (SELL)", pf_short)]:
        stats = pf.stats()
        trades = pf.trades.records_readable

        print(f"\n  ── {name} ──")
        print(f"  Trade sayisi     : {stats.get('Total Trades', 0):.0f}")
        print(f"  Win rate         : %{stats.get('Win Rate [%]', 0):.1f}")
        print(f"  Toplam getiri    : %{stats.get('Total Return [%]', 0):.2f}")
        print(f"  Sharpe ratio     : {stats.get('Sharpe Ratio', 0):.2f}")
        print(f"  Max drawdown     : %{stats.get('Max Drawdown [%]', 0):.2f}")
        print(f"  Avg trade P&L    : %{stats.get('Avg Winning Trade [%]', 0):.2f} / %{stats.get('Avg Losing Trade [%]', 0):.2f} (W/L)")
        print(f"  Son bakiye       : {BALANCE + pf.final_value() - BALANCE:.2f}$")

        if len(trades) > 0:
            print(f"\n  Son 5 trade:")
            cols = ["Entry Timestamp", "Exit Timestamp", "PnL", "Return [%]", "Status"]
            available = [c for c in cols if c in trades.columns]
            print(trades[available].tail(5).to_string(index=False))

    print(f"\n{'='*60}")


def print_monthly(pf_long, pf_short):
    print(f"\n  AYLIK PERFORMANS:")
    print(f"  {'Ay':<12} {'LONG':>10} {'SHORT':>10} {'Toplam':>10}")
    print(f"  {'-'*44}")

    try:
        monthly_long  = pf_long.returns().resample("ME").sum() * 100
        monthly_short = pf_short.returns().resample("ME").sum() * 100
        monthly_total = monthly_long + monthly_short

        for idx in monthly_long.index:
            l = monthly_long.get(idx, 0)
            s = monthly_short.get(idx, 0)
            t = l + s
            color_t = "+" if t >= 0 else ""
            print(f"  {str(idx.date()):<12} {l:>+9.2f}% {s:>+9.2f}% {t:>+9.2f}%")
    except Exception as e:
        print(f"  Aylik analiz hatasi: {e}")

    print(f"  {'-'*44}")


def plot_results(pf_long, pf_short, df, buy_sig, sell_sig):
    BG      = "#0d1117"
    PANEL   = "#161b22"
    BORDER  = "#2d3748"
    TEXT    = "#c9d1d9"
    MUTED   = "#8b9cad"
    GREEN   = "#00d4aa"
    RED     = "#ff6b6b"
    YELLOW  = "#ffd700"
    BLUE    = "#60a5fa"
    PURPLE  = "#a78bfa"

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.canvas.manager.set_window_title("OrderFlow Bot — Backtest Dashboard")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.07
    )

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontweight="bold")
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.grid(color=BORDER, lw=0.4, alpha=0.6)

    # ── 1. Equity Curve (geniş, üst sol+orta) ──────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Equity Curve")
    try:
        eq_l = pf_long.value()
        eq_s = pf_short.value()
        eq_total = BALANCE + (eq_l - BALANCE) + (eq_s - BALANCE)
        ax1.plot(eq_l.index,     eq_l.values,     color=GREEN,  lw=1.2, alpha=0.7, label="LONG")
        ax1.plot(eq_s.index,     eq_s.values,     color=RED,    lw=1.2, alpha=0.7, label="SHORT")
        ax1.plot(eq_total.index, eq_total.values, color=YELLOW, lw=2,   label="Toplam")
        ax1.axhline(BALANCE, color=MUTED, ls="--", lw=0.8, alpha=0.6, label=f"Başlangıç ${BALANCE:.0f}")
        ax1.fill_between(eq_total.index, BALANCE, eq_total.values,
                         where=(eq_total.values < BALANCE), alpha=0.12, color=RED)
        ax1.fill_between(eq_total.index, BALANCE, eq_total.values,
                         where=(eq_total.values >= BALANCE), alpha=0.12, color=GREEN)
    except Exception as e:
        ax1.text(0.5, 0.5, str(e), transform=ax1.transAxes, color=RED, ha="center")
    ax1.set_ylabel("USDT")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9,
               loc="upper right", ncol=4)

    # ── 2. Özet stat kutuları (üst sağ) ────────────────────────────
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.set_facecolor(PANEL)
    ax_s.axis("off")
    for spine in ax_s.spines.values():
        spine.set_color(BORDER)
    ax_s.set_title("Özet", color=TEXT, fontsize=10, pad=8, fontweight="bold")

    try:
        wr_l = pf_long.stats().get("Win Rate [%]", 0)
        wr_s = pf_short.stats().get("Win Rate [%]", 0)
        tr_l = pf_long.stats().get("Total Return [%]", 0)
        tr_s = pf_short.stats().get("Total Return [%]", 0)
        sh_l = pf_long.stats().get("Sharpe Ratio", 0)
        sh_s = pf_short.stats().get("Sharpe Ratio", 0)
        md_l = pf_long.stats().get("Max Drawdown [%]", 0)
        md_s = pf_short.stats().get("Max Drawdown [%]", 0)
        n_l  = pf_long.stats().get("Total Trades", 0)
        n_s  = pf_short.stats().get("Total Trades", 0)
    except:
        wr_l=wr_s=tr_l=tr_s=sh_l=sh_s=md_l=md_s=n_l=n_s=0

    rows = [
        ("",          "LONG",              "SHORT"),
        ("Trades",    f"{n_l:.0f}",        f"{n_s:.0f}"),
        ("Win Rate",  f"%{wr_l:.1f}",      f"%{wr_s:.1f}"),
        ("Getiri",    f"%{tr_l:.1f}",      f"%{tr_s:.1f}"),
        ("Sharpe",    f"{sh_l:.2f}",       f"{sh_s:.2f}"),
        ("Max DD",    f"%{md_l:.1f}",      f"%{md_s:.1f}"),
    ]
    for i, (lbl, vl, vs) in enumerate(rows):
        y = 0.92 - i * 0.155
        cl = GREEN if i == 0 else (GREEN if i > 1 and float(vl.replace('%','')) > 0 else RED if i > 1 else TEXT)
        cs = GREEN if i == 0 else (GREEN if i > 1 and float(vs.replace('%','')) > 0 else RED if i > 1 else TEXT)
        ax_s.text(0.05, y, lbl, transform=ax_s.transAxes, color=MUTED, fontsize=9)
        ax_s.text(0.45, y, vl,  transform=ax_s.transAxes, color=GREEN if i==0 else TEXT, fontsize=9, fontweight="bold" if i==0 else "normal")
        ax_s.text(0.75, y, vs,  transform=ax_s.transAxes, color=RED   if i==0 else TEXT, fontsize=9, fontweight="bold" if i==0 else "normal")
        if i == 0:
            ax_s.plot([0.03, 0.97], [y-0.02, y-0.02],
                      transform=ax_s.transAxes, color=BORDER, lw=0.6, clip_on=False)

    # ── 3. Drawdown (orta sol+orta) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    style_ax(ax2, "Drawdown (%)")
    try:
        dd_l = pf_long.drawdown() * 100
        dd_s = pf_short.drawdown() * 100
        ax2.plot(dd_l.index, dd_l.values, color=GREEN, lw=1, alpha=0.8, label="LONG DD")
        ax2.plot(dd_s.index, dd_s.values, color=RED,   lw=1, alpha=0.8, label="SHORT DD")
        ax2.fill_between(dd_l.index, dd_l.values, 0, alpha=0.1, color=GREEN)
        ax2.fill_between(dd_s.index, dd_s.values, 0, alpha=0.1, color=RED)
        ax2.axhline(0, color=MUTED, lw=0.6)
    except Exception as e:
        ax2.text(0.5, 0.5, str(e), transform=ax2.transAxes, color=RED, ha="center")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)

    # ── 4. Aylık getiri bar chart (orta sağ) ────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    style_ax(ax3, "Aylık Getiri (Toplam %)")
    try:
        ml = pf_long.returns().resample("ME").sum() * 100
        ms = pf_short.returns().resample("ME").sum() * 100
        mt = ml + ms
        bar_colors = [GREEN if v >= 0 else RED for v in mt.values]
        ax3.bar(range(len(mt)), mt.values, color=bar_colors, alpha=0.85, width=0.6)
        ax3.axhline(0, color=MUTED, lw=0.8)
        ax3.set_xticks(range(len(mt)))
        ax3.set_xticklabels([d.strftime("%m/%y") for d in mt.index],
                            rotation=45, ha="right", fontsize=7)
    except Exception as e:
        ax3.text(0.5, 0.5, str(e), transform=ax3.transAxes, color=RED, ha="center")
    ax3.set_ylabel("%")

    # ── 5. Fiyat + sinyaller (alt sol+orta, son 1500 bar) ───────────
    ax4 = fig.add_subplot(gs[2, :2])
    style_ax(ax4, f"Fiyat + Sinyaller — son 1500 bar  (MIN_SCORE≥{MIN_SCORE_BUY})")
    sample = df.tail(1500)
    ax4.plot(range(len(sample)), sample["close"].values,
             color=MUTED, lw=0.7, alpha=0.8, label="Close")

    b_mask = buy_sig.reindex(sample.index, fill_value=False)
    s_mask = sell_sig.reindex(sample.index, fill_value=False)
    b_pos  = [i for i, v in enumerate(b_mask) if v]
    s_pos  = [i for i, v in enumerate(s_mask) if v]
    b_prices = sample["close"].values[b_pos]
    s_prices = sample["close"].values[s_pos]

    if len(b_pos):
        ax4.scatter(b_pos, b_prices, color=GREEN, s=18, zorder=5,
                    marker="^", label=f"BUY ({len(b_pos)})")
    if len(s_pos):
        ax4.scatter(s_pos, s_prices, color=RED, s=18, zorder=5,
                    marker="v", label=f"SELL ({len(s_pos)})")

    ax4.set_ylabel("USDT")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    ax4.set_xlabel("Bar (5dk)")

    # ── 6. Win rate pasta (alt sağ) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    style_ax(ax5, "Trade Dağılımı")
    ax5.set_facecolor(PANEL)
    ax5.axis("off")
    try:
        total_buy  = int(buy_sig.sum())
        total_sell = int(sell_sig.sum())
        win_l  = round(wr_l / 100 * n_l)
        lose_l = int(n_l) - int(win_l)
        win_s  = round(wr_s / 100 * n_s)
        lose_s = int(n_s) - int(win_s)

        info = [
            ("BUY sinyali",    f"{total_buy:,}",  BLUE),
            ("SELL sinyali",   f"{total_sell:,}",  PURPLE),
            ("LONG trade",     f"{int(n_l):,}",    GREEN),
            ("  ↳ Kazanan",    f"{int(win_l):,}",  GREEN),
            ("  ↳ Kaybeden",   f"{int(lose_l):,}", RED),
            ("SHORT trade",    f"{int(n_s):,}",    RED),
            ("  ↳ Kazanan",    f"{int(win_s):,}",  GREEN),
            ("  ↳ Kaybeden",   f"{int(lose_s):,}", RED),
        ]
        for i, (lbl, val, col) in enumerate(info):
            y = 0.90 - i * 0.115
            ax5.text(0.05, y, lbl, transform=ax5.transAxes, color=MUTED, fontsize=9)
            ax5.text(0.75, y, val, transform=ax5.transAxes, color=col,   fontsize=9, fontweight="bold")
    except Exception as e:
        ax5.text(0.5, 0.5, str(e), transform=ax5.transAxes, color=RED, ha="center")

    # ── Başlık ──────────────────────────────────────────────────────
    fig.suptitle(
        f"OrderFlow Bot — Backtest Dashboard  |  ETHUSDT 5m  |  "
        f"{df.index[0].date()} → {df.index[-1].date()}  |  "
        f"{BALANCE:.0f}$ / {LEVERAGE}x Kaldıraç  |  Komisyon %{COMMISSION*100:.2f}  |  SL %{SL_PCT*100:.1f}  TP %{TP1_PCT*100:.2f}",
        color=TEXT, fontsize=10, y=0.98
    )

    out = Path("data/backtest_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[+] PNG kaydedildi: {out}")
    plt.show()   # ← ekranda açılır
    plt.close()


def main():
    print("=" * 60)
    print("  ORDERFLOW BOT — GERÇEKÇI BACKTEST (vectorbt)")
    print(f"  Bakiye: {BALANCE}$ | Kaldirac: {LEVERAGE}x | Risk/trade: %{RISK_PCT*100:.0f}")
    print("=" * 60)

    df = load_data()
    pf_long, pf_short, df_feat, buy_sig, sell_sig = run_backtest_vbt(df)

    print_stats(pf_long, pf_short)
    print_monthly(pf_long, pf_short)
    plot_results(pf_long, pf_short, df_feat, buy_sig, sell_sig)

    # CSV kaydet
    trades_l = pf_long.trades.records_readable
    trades_s = pf_short.trades.records_readable
    if len(trades_l) > 0:
        trades_l["direction"] = "LONG"
    if len(trades_s) > 0:
        trades_s["direction"] = "SHORT"
    all_trades = pd.concat([trades_l, trades_s], ignore_index=True)
    if len(all_trades) > 0:
        all_trades.to_csv("data/backtest_trades.csv", index=False)
        print(f"[+] Trade listesi: data/backtest_trades.csv")

    print(f"\n[+] Tamamlandi!")


if __name__ == "__main__":
    main()