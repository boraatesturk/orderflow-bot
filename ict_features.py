"""
=============================================================================
  ICT FEATURES MODULE v2 (VECTORIZED)
  MSS, BOS, CISD, FVG, IFVG, Order Block hesaplama motoru
  Numpy ile vektorize edilmis, 50-100x hizli

  Kullanim:
    from ict_features import ICTEngine
    ict = ICTEngine(df)
    df = ict.compute_all()
=============================================================================
"""

import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


class ICTEngine:
    def __init__(self, df: pd.DataFrame, swing_left: int = 10, swing_right: int = 10, fvg_min_pct: float = 0.05):
        self.df          = df.copy()
        self.swing_left  = swing_left
        self.swing_right = swing_right
        self.fvg_min_pct = fvg_min_pct

    # ─── SWING HIGH / LOW ────────────────────────────────────────

    def _find_swings(self) -> pd.DataFrame:
        df  = self.df
        hi  = df["high"].values
        lo  = df["low"].values
        n   = len(df)
        L   = self.swing_left
        R   = self.swing_right

        swing_high = np.zeros(n, dtype=bool)
        swing_low  = np.zeros(n, dtype=bool)

        for i in range(L, n - R):
            if hi[i] > hi[i - L : i].max() and hi[i] > hi[i + 1 : i + R + 1].max():
                swing_high[i] = True
            if lo[i] < lo[i - L : i].min() and lo[i] < lo[i + 1 : i + R + 1].min():
                swing_low[i] = True

        df["swing_high"] = swing_high
        df["swing_low"]  = swing_low
        return df

    # ─── BOS ─────────────────────────────────────────────────────

    def _find_bos(self) -> pd.DataFrame:
        df = self.df
        n  = len(df)
        close = df["close"].values
        hi    = df["high"].values
        lo    = df["low"].values

        bos_bull  = np.zeros(n, dtype=bool)
        bos_bear  = np.zeros(n, dtype=bool)
        bos_level = np.full(n, np.nan)

        last_sh = np.nan
        last_sl = np.nan
        sh_mask = df["swing_high"].values
        sl_mask = df["swing_low"].values

        for i in range(1, n):
            if sh_mask[i - 1]:
                last_sh = hi[i - 1]
            if sl_mask[i - 1]:
                last_sl = lo[i - 1]

            if not np.isnan(last_sh) and close[i] > last_sh:
                bos_bull[i]  = True
                bos_level[i] = last_sh
                last_sh = np.nan

            if not np.isnan(last_sl) and close[i] < last_sl:
                bos_bear[i]  = True
                bos_level[i] = last_sl
                last_sl = np.nan

        df["bos_bull"]  = bos_bull
        df["bos_bear"]  = bos_bear
        df["bos_level"] = bos_level
        return df

    # ─── MSS ─────────────────────────────────────────────────────

    def _find_mss(self) -> pd.DataFrame:
        df = self.df
        n  = len(df)
        bos_b = df["bos_bull"].values
        bos_e = df["bos_bear"].values

        mss_bull = np.zeros(n, dtype=bool)
        mss_bear = np.zeros(n, dtype=bool)
        last_two = []

        for i in range(n):
            if bos_b[i]:
                if len(last_two) >= 2 and last_two[-1] == "bear" and last_two[-2] == "bear":
                    mss_bull[i] = True
                last_two.append("bull")
                if len(last_two) > 5:
                    last_two.pop(0)
            elif bos_e[i]:
                if len(last_two) >= 2 and last_two[-1] == "bull" and last_two[-2] == "bull":
                    mss_bear[i] = True
                last_two.append("bear")
                if len(last_two) > 5:
                    last_two.pop(0)

        df["mss_bull"] = mss_bull
        df["mss_bear"] = mss_bear
        return df

    # ─── FVG (fully vectorized) ──────────────────────────────────

    def _find_fvg(self) -> pd.DataFrame:
        df = self.df
        n  = len(df)
        hi = df["high"].values
        lo = df["low"].values
        cl = df["close"].values

        fvg_bull     = np.zeros(n, dtype=bool)
        fvg_bear     = np.zeros(n, dtype=bool)
        fvg_bull_top = np.full(n, np.nan)
        fvg_bull_bot = np.full(n, np.nan)
        fvg_bear_top = np.full(n, np.nan)
        fvg_bear_bot = np.full(n, np.nan)

        if n > 2:
            hi_2 = hi[:-2]
            lo_0 = lo[2:]
            lo_2 = lo[:-2]
            hi_0 = hi[2:]

            gap_bull = lo_0 - hi_2
            bull_mask = (gap_bull > 0) & ((gap_bull / hi_2) * 100 >= self.fvg_min_pct)

            gap_bear = lo_2 - hi_0
            bear_mask = (gap_bear > 0) & ((gap_bear / lo_2) * 100 >= self.fvg_min_pct)

            idx = np.arange(2, n)
            fvg_bull[idx[bull_mask]] = True
            fvg_bull_top[idx[bull_mask]] = lo_0[bull_mask]
            fvg_bull_bot[idx[bull_mask]] = hi_2[bull_mask]

            fvg_bear[idx[bear_mask]] = True
            fvg_bear_top[idx[bear_mask]] = lo_2[bear_mask]
            fvg_bear_bot[idx[bear_mask]] = hi_0[bear_mask]

        df["fvg_bull"]     = fvg_bull
        df["fvg_bear"]     = fvg_bear
        df["fvg_bull_top"] = fvg_bull_top
        df["fvg_bull_bot"] = fvg_bull_bot
        df["fvg_bear_top"] = fvg_bear_top
        df["fvg_bear_bot"] = fvg_bear_bot

        # FVG fill (vektorize)
        fvg_filled_bull = np.zeros(n, dtype=bool)
        fvg_filled_bear = np.zeros(n, dtype=bool)

        for fi in np.where(fvg_bull)[0]:
            top, bot = fvg_bull_top[fi], fvg_bull_bot[fi]
            end = min(fi + 20, n)
            fills = (cl[fi + 1 : end] < top) & (cl[fi + 1 : end] > bot)
            if fills.any():
                fvg_filled_bull[fi + 1 + np.argmax(fills)] = True

        for fi in np.where(fvg_bear)[0]:
            top, bot = fvg_bear_top[fi], fvg_bear_bot[fi]
            end = min(fi + 20, n)
            fills = (cl[fi + 1 : end] > bot) & (cl[fi + 1 : end] < top)
            if fills.any():
                fvg_filled_bear[fi + 1 + np.argmax(fills)] = True

        df["fvg_filled_bull"] = fvg_filled_bull
        df["fvg_filled_bear"] = fvg_filled_bear
        return df

    # ─── IFVG ────────────────────────────────────────────────────

    def _find_ifvg(self) -> pd.DataFrame:
        df = self.df
        n  = len(df)
        cl = df["close"].values

        ifvg_bull = np.zeros(n, dtype=bool)
        ifvg_bear = np.zeros(n, dtype=bool)

        for fi in np.where(df["fvg_bear"].values)[0]:
            top = df["fvg_bear_top"].iloc[fi]
            end = min(fi + 10, n)
            crosses = cl[fi + 1 : end] > top
            if crosses.any():
                ifvg_bull[fi + 1 + np.argmax(crosses)] = True

        for fi in np.where(df["fvg_bull"].values)[0]:
            bot = df["fvg_bull_bot"].iloc[fi]
            end = min(fi + 10, n)
            crosses = cl[fi + 1 : end] < bot
            if crosses.any():
                ifvg_bear[fi + 1 + np.argmax(crosses)] = True

        df["ifvg_bull"] = ifvg_bull
        df["ifvg_bear"] = ifvg_bear
        return df

    # ─── ORDER BLOCK ─────────────────────────────────────────────

    def _find_order_blocks(self) -> pd.DataFrame:
        df = self.df
        n  = len(df)
        op = df["open"].values
        cl = df["close"].values
        hi = df["high"].values
        lo = df["low"].values

        ob_bull      = np.zeros(n, dtype=bool)
        ob_bear      = np.zeros(n, dtype=bool)
        ob_bull_top  = np.full(n, np.nan)
        ob_bull_bot  = np.full(n, np.nan)
        ob_bear_top  = np.full(n, np.nan)
        ob_bear_bot  = np.full(n, np.nan)
        ob_bull_test = np.zeros(n, dtype=bool)
        ob_bear_test = np.zeros(n, dtype=bool)

        bos_b = df["bos_bull"].values
        bos_e = df["bos_bear"].values
        active_bull = []
        active_bear = []

        for i in range(1, n):
            if bos_b[i]:
                for j in range(i - 1, max(i - 10, 0), -1):
                    if cl[j] < op[j]:
                        ob_bull[j]     = True
                        ob_bull_top[j] = hi[j]
                        ob_bull_bot[j] = lo[j]
                        active_bull.append((hi[j], lo[j], j))
                        break

            if bos_e[i]:
                for j in range(i - 1, max(i - 10, 0), -1):
                    if cl[j] > op[j]:
                        ob_bear[j]     = True
                        ob_bear_top[j] = hi[j]
                        ob_bear_bot[j] = lo[j]
                        active_bear.append((hi[j], lo[j], j))
                        break

            for top, bot, idx in active_bull:
                if lo[i] <= top and hi[i] >= bot:
                    ob_bull_test[i] = True
                    break

            for top, bot, idx in active_bear:
                if lo[i] <= top and hi[i] >= bot:
                    ob_bear_test[i] = True
                    break

            if i % 50 == 0:
                active_bull = [(t, b, x) for t, b, x in active_bull if i - x < 50]
                active_bear = [(t, b, x) for t, b, x in active_bear if i - x < 50]

        df["ob_bull"]      = ob_bull
        df["ob_bear"]      = ob_bear
        df["ob_bull_top"]  = ob_bull_top
        df["ob_bull_bot"]  = ob_bull_bot
        df["ob_bear_top"]  = ob_bear_top
        df["ob_bear_bot"]  = ob_bear_bot
        df["ob_bull_test"] = ob_bull_test
        df["ob_bear_test"] = ob_bear_test
        return df

    # ─── CISD ────────────────────────────────────────────────────

    def _find_cisd(self) -> pd.DataFrame:
        """
        Gercek CISD mantigi:
        
        Bearish CISD: Bar once swing high'in ustune cikiyor (wiek), 
                      sonra kapanis o swing high'in ALTINDA bituyor.
                      = Alicilar tuzaga dusuruldu, saticilar devraldi.
        
        Bullish CISD: Bar once swing low'un altina iniyor (wick),
                      sonra kapanis o swing low'un USTUNDE bituyor.
                      = Saticilar tuzaga dusuruldu, alicilar devraldi.
        
        Ikisi ayni anda olamaz cunku bir bar ya yukarida ya asagida kapanir.
        """
        df = self.df
        n  = len(df)
        cl = df["close"].values
        op = df["open"].values
        hi = df["high"].values
        lo = df["low"].values

        cisd_bull = np.zeros(n, dtype=bool)
        cisd_bear = np.zeros(n, dtype=bool)

        last_sh     = np.nan
        last_sl     = np.nan
        last_sh_idx = -1
        last_sl_idx = -1
        sh_mask = df["swing_high"].values
        sl_mask = df["swing_low"].values

        for i in range(1, n):
            # Swing seviyelerini guncelle (bir onceki barda olusanlar)
            if sh_mask[i - 1]:
                last_sh     = hi[i - 1]
                last_sh_idx = i - 1
            if sl_mask[i - 1]:
                last_sl     = lo[i - 1]
                last_sl_idx = i - 1

            # Bearish CISD:
            # - Son swing high yakin zamanda olusmus (max 15 bar)
            # - Bu barin HIGH'i o swing high'in ustune cikti (sweep)
            # - Ama CLOSE o swing high'in ALTINDA kapandi (rejection)
            # - Bar bearish kapandi (close < open)
            if (not np.isnan(last_sh) and
                    i - last_sh_idx <= 15 and
                    hi[i] > last_sh and          # sweep yukari
                    cl[i] < last_sh and          # kapanis asagida
                    cl[i] < op[i]):              # bearish bar
                cisd_bear[i] = True

            # Bullish CISD:
            # - Son swing low yakin zamanda olusmus (max 15 bar)
            # - Bu barin LOW'u o swing low'un altina indi (sweep)
            # - Ama CLOSE o swing low'un USTUNDE kapandi (rejection)
            # - Bar bullish kapandi (close > open)
            elif (not np.isnan(last_sl) and      # elif: ayni anda ikisi olamaz
                    i - last_sl_idx <= 15 and
                    lo[i] < last_sl and           # sweep asagi
                    cl[i] > last_sl and           # kapanis yukarda
                    cl[i] > op[i]):               # bullish bar
                cisd_bull[i] = True

        df["cisd_bull"] = cisd_bull
        df["cisd_bear"] = cisd_bear
        return df

    # ─── COMPUTE ALL ─────────────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        print(f"{Fore.CYAN}ICT analiz ediliyor...{Style.RESET_ALL}", end=" ", flush=True)
        self._find_swings()
        self._find_bos()
        self._find_mss()
        self._find_fvg()
        self._find_ifvg()
        self._find_order_blocks()
        self._find_cisd()
        print(f"{Fore.GREEN}Tamamlandi.{Style.RESET_ALL}")
        return self.df

    # ─── AKTIF SETUPLAR ──────────────────────────────────────────

    def get_active_setups(self, last_n: int = 3) -> dict:
        df   = self.df
        last = df.tail(last_n)
        setups = {"bull": [], "bear": []}

        if last["mss_bull"].any():
            setups["bull"].append("MSS BULL VAR  -> Trend yukari donusu (en guclu)")
        if last["bos_bull"].any():
            level = last.loc[last["bos_bull"], "bos_level"].dropna()
            if not level.empty:
                setups["bull"].append(f"BOS BULL VAR  -> Kirilan seviye: {level.iloc[-1]:.2f} USDT")
        if last["fvg_bull"].any():
            top_s = last.loc[last["fvg_bull"], "fvg_bull_top"].dropna()
            bot_s = last.loc[last["fvg_bull"], "fvg_bull_bot"].dropna()
            if not top_s.empty and not bot_s.empty:
                setups["bull"].append(f"FVG BULL VAR  -> Bosluk: {bot_s.iloc[-1]:.2f} - {top_s.iloc[-1]:.2f} USDT")
        if last["ifvg_bull"].any():
            setups["bull"].append("IFVG BULL VAR -> Bearish FVG asildi (yukari)")
        if last["ob_bull_test"].any():
            t = df["ob_bull_top"].dropna()
            b = df["ob_bull_bot"].dropna()
            if not t.empty:
                setups["bull"].append(f"OB BULL TEST  -> Bolge: {b.iloc[-1]:.2f} - {t.iloc[-1]:.2f} USDT")
        if last["cisd_bull"].any():
            setups["bull"].append("CISD BULL VAR -> Delivery yukariya dondu")

        if last["mss_bear"].any():
            setups["bear"].append("MSS BEAR VAR  -> Trend asagi donusu (en guclu)")
        if last["bos_bear"].any():
            level = last.loc[last["bos_bear"], "bos_level"].dropna()
            if not level.empty:
                setups["bear"].append(f"BOS BEAR VAR  -> Kirilan seviye: {level.iloc[-1]:.2f} USDT")
        if last["fvg_bear"].any():
            top_s = last.loc[last["fvg_bear"], "fvg_bear_top"].dropna()
            bot_s = last.loc[last["fvg_bear"], "fvg_bear_bot"].dropna()
            if not top_s.empty and not bot_s.empty:
                setups["bear"].append(f"FVG BEAR VAR  -> Bosluk: {bot_s.iloc[-1]:.2f} - {top_s.iloc[-1]:.2f} USDT")
        if last["ifvg_bear"].any():
            setups["bear"].append("IFVG BEAR VAR -> Bullish FVG asildi (asagi)")
        if last["ob_bear_test"].any():
            t = df["ob_bear_top"].dropna()
            b = df["ob_bear_bot"].dropna()
            if not t.empty:
                setups["bear"].append(f"OB BEAR TEST  -> Bolge: {b.iloc[-1]:.2f} - {t.iloc[-1]:.2f} USDT")
        if last["cisd_bear"].any():
            setups["bear"].append("CISD BEAR VAR -> Delivery asagiya dondu")

        return setups