"""Causal, optional market-context features for M5/M15 FX bars.

The module observes completed OHLCV bars only. It does not gate trades, fit a
model, download data, or infer institutional intent from unavailable order flow.
FVG/OB/MSS are deterministic price-action proxies and should be treated as
features, not proof of institutional activity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "fvg_bull_distance_pips", "fvg_bull_width_pips", "fvg_bull_present",
    "fvg_bear_distance_pips", "fvg_bear_width_pips", "fvg_bear_present",
    "ob_bull_distance_pips", "ob_bull_present", "ob_bull_mitigated",
    "ob_bear_distance_pips", "ob_bear_present", "ob_bear_mitigated",
    "liquidity_asia_high_distance_pips", "liquidity_asia_low_distance_pips",
    "liquidity_session_high_distance_pips", "liquidity_session_low_distance_pips",
    "sweep_asia_high", "sweep_asia_low", "sweep_session_high", "sweep_session_low",
    "mss_direction",
)

# Fixed scales are a unit conversion, not a fitted normalizer. Raw pips stay
# available in the Data Farm output; the genetic input uses bounded values.
FEATURE_SCALES = {
    name: 50.0 for name in (
        "fvg_bull_distance_pips", "fvg_bear_distance_pips",
        "ob_bull_distance_pips", "ob_bear_distance_pips",
        "liquidity_asia_high_distance_pips", "liquidity_asia_low_distance_pips",
        "liquidity_session_high_distance_pips", "liquidity_session_low_distance_pips",
    )
}
FEATURE_SCALES.update({
    "fvg_bull_width_pips": 25.0, "fvg_bear_width_pips": 25.0,
})


@dataclass(frozen=True)
class DataFarmConfig:
    pip_size: float = 0.0001
    fvg_max_age_bars: int = 500
    ob_max_age_bars: int = 96
    ob_break_lookback: int = 5
    ob_atr_period: int = 14
    ob_impulse_atr: float = 1.2
    pivot_left: int = 2
    pivot_right: int = 2
    asia_start_hour_utc: int = 0
    asia_end_hour_utc: int = 7
    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 13
    new_york_start_hour_utc: int = 13
    new_york_end_hour_utc: int = 21

    def __post_init__(self):
        if not np.isfinite(self.pip_size) or self.pip_size <= 0:
            raise ValueError("pip_size debe ser positivo y finito")
        for name in ("fvg_max_age_bars", "ob_max_age_bars", "ob_break_lookback",
                     "ob_atr_period", "pivot_left", "pivot_right"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser entero positivo")
        if not np.isfinite(self.ob_impulse_atr) or self.ob_impulse_atr <= 0:
            raise ValueError("ob_impulse_atr debe ser positivo")
        windows = (
            (self.asia_start_hour_utc, self.asia_end_hour_utc),
            (self.london_start_hour_utc, self.london_end_hour_utc),
            (self.new_york_start_hour_utc, self.new_york_end_hour_utc),
        )
        if any(type(a) is not int or type(b) is not int or not 0 <= a < b <= 24
               for a, b in windows):
            raise ValueError("Las ventanas UTC requieren 0 <= inicio < fin <= 24")
        if self.asia_end_hour_utc != self.london_start_hour_utc or (
            self.london_end_hour_utc != self.new_york_start_hour_utc
        ):
            raise ValueError("Asia, Londres y Nueva York deben ser ventanas contiguas")


def _bar_minutes(timeframe: str) -> int:
    if timeframe not in ("M5", "M15"):
        raise ValueError("La Data Farm solo acepta M5 o M15")
    return 5 if timeframe == "M5" else 15


def validate_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return clean, sorted bars; timestamps are bar-open times in UTC."""
    minutes = _bar_minutes(timeframe)
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas OHLC: {sorted(required - set(frame.columns))}")
    out = frame.copy()
    if "volume" not in out:
        out["volume"] = 0.0
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    for name in ("open", "high", "low", "close", "volume"):
        out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
    values = out[["open", "high", "low", "close", "volume"]].to_numpy()
    if out.empty or not np.isfinite(values).all():
        raise ValueError("OHLCV vacío o con nulos/infinitos")
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out.volume < 0).any():
        raise ValueError("Precios deben ser positivos y volumen no negativo")
    if (out.high < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high incoherente")
    if (out.low > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low incoherente")
    if out.timestamp.isna().any() or out.timestamp.duplicated().any():
        raise ValueError("Timestamps vacíos o duplicados")
    out = out.sort_values("timestamp").reset_index(drop=True)
    step = pd.Timedelta(minutes=minutes)
    if (out.timestamp != out.timestamp.dt.floor(f"{minutes}min")).any():
        raise ValueError(f"Los timestamps deben estar alineados a {timeframe}")
    out.attrs["timeframe"] = timeframe
    out.attrs["bar_minutes"] = minutes
    return out


def _signed_zone_distance(price: float, lower: float, upper: float, pip: float) -> float:
    if price > upper:
        return (price - upper) / pip
    if price < lower:
        return (price - lower) / pip
    return 0.0


def _nearest_zone(zones, price, pip):
    if not zones:
        return 0.0, 0.0, 0.0
    # Tie-break toward the most recently formed zone.
    chosen = min(zones, key=lambda z: (abs(_signed_zone_distance(price, z[0], z[1], pip)), -z[2]))
    return (_signed_zone_distance(price, chosen[0], chosen[1], pip),
            (chosen[1] - chosen[0]) / pip, 1.0)


def _session(hour: int, cfg: DataFarmConfig):
    if cfg.asia_start_hour_utc <= hour < cfg.asia_end_hour_utc:
        return "asia"
    if cfg.london_start_hour_utc <= hour < cfg.london_end_hour_utc:
        return "london"
    if cfg.new_york_start_hour_utc <= hour < cfg.new_york_end_hour_utc:
        return "new_york"
    return None


def build_micro_event_features(
    frame: pd.DataFrame, *, timeframe: str, config: DataFarmConfig | None = None,
) -> pd.DataFrame:
    """Build 21 causal context features aligned to the input bar-open timestamps.

    Each row is available only after that bar closes. FVG/OB are wick-zone
    proxies; a liquidity sweep is a wick beyond a previously known level with
    a close back through it; MSS is HH+HL / LH+LL using confirmed pivots only.
    Missing reference levels are encoded as zero distance + zero presence/flag.
    """
    cfg = config or DataFarmConfig()
    bars = validate_ohlcv(frame, timeframe)
    step = pd.Timedelta(minutes=_bar_minutes(timeframe))
    ts = pd.DatetimeIndex(bars.timestamp)
    op = bars.open.to_numpy()
    hi = bars.high.to_numpy()
    lo = bars.low.to_numpy()
    cl = bars.close.to_numpy()
    n = len(bars)
    consecutive = np.zeros(n, dtype=bool)
    if n > 1:
        consecutive[1:] = np.diff(ts.asi8) == step.value

    # Causal ATR proxy for OB displacement. A gap breaks its rolling warmup.
    prev_close = np.r_[np.nan, cl[:-1]]
    broken_rows = np.flatnonzero(~consecutive[1:]) + 1
    prev_close[broken_rows] = np.nan
    true_range = np.maximum.reduce((hi - lo, np.abs(hi - prev_close), np.abs(lo - prev_close)))
    atr = pd.Series(true_range).rolling(cfg.ob_atr_period, min_periods=cfg.ob_atr_period).mean().to_numpy()

    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    bull_fvgs, bear_fvgs = [], []
    bull_ob = None
    bear_ob = None
    swing_highs, swing_lows = [], []
    mss = 0

    asia_reference = None
    previous_session_reference = None
    current_session = None
    current_session_high = current_session_low = None
    current_session_count = 0
    current_session_expected = 0
    current_session_incomplete = False
    previous_time = None

    for i in range(n):
        stamp = ts[i]
        hour = stamp.hour
        day = stamp.date()
        session = _session(hour, cfg)
        gap = previous_time is not None and stamp - previous_time != step

        # Close the prior session only if every expected bar was observed.
        if session != current_session or gap:
            if current_session is not None and current_session_count == current_session_expected and not current_session_incomplete:
                ref = (current_session_high, current_session_low)
                previous_session_reference = ref
                if current_session == "asia":
                    asia_reference = ref
            current_session = session
            current_session_high = current_session_low = None
            current_session_count = 0
            current_session_expected = 0
            current_session_incomplete = False
            if session == "asia":
                current_session_expected = int(
                    (cfg.asia_end_hour_utc - cfg.asia_start_hour_utc) * 60 / _bar_minutes(timeframe)
                )
            elif session == "london":
                current_session_expected = int(
                    (cfg.london_end_hour_utc - cfg.london_start_hour_utc) * 60 / _bar_minutes(timeframe)
                )
            elif session == "new_york":
                current_session_expected = int(
                    (cfg.new_york_end_hour_utc - cfg.new_york_start_hour_utc) * 60 / _bar_minutes(timeframe)
                )

        # Distances and sweep flags are computed against levels known BEFORE this bar.
        if asia_reference is not None:
            ah, al = asia_reference
            out[i, col["liquidity_asia_high_distance_pips"]] = (cl[i] - ah) / cfg.pip_size
            out[i, col["liquidity_asia_low_distance_pips"]] = (cl[i] - al) / cfg.pip_size
            out[i, col["sweep_asia_high"]] = float(hi[i] > ah and cl[i] < ah)
            out[i, col["sweep_asia_low"]] = float(lo[i] < al and cl[i] > al)
        if previous_session_reference is not None:
            sh, sl = previous_session_reference
            out[i, col["liquidity_session_high_distance_pips"]] = (cl[i] - sh) / cfg.pip_size
            out[i, col["liquidity_session_low_distance_pips"]] = (cl[i] - sl) / cfg.pip_size
            out[i, col["sweep_session_high"]] = float(hi[i] > sh and cl[i] < sh)
            out[i, col["sweep_session_low"]] = float(lo[i] < sl and cl[i] > sl)

        # Existing FVGs are invalidated only after a complete traverse through the zone.
        bull_fvgs = [z for z in bull_fvgs
                     if i - z[2] <= cfg.fvg_max_age_bars and lo[i] > z[0]]
        bear_fvgs = [z for z in bear_fvgs
                     if i - z[2] <= cfg.fvg_max_age_bars and hi[i] < z[1]]
        bd, bw, bp = _nearest_zone(bull_fvgs, cl[i], cfg.pip_size)
        sd, sw, sp = _nearest_zone(bear_fvgs, cl[i], cfg.pip_size)
        out[i, col["fvg_bull_distance_pips"]] = bd
        out[i, col["fvg_bull_width_pips"]] = bw
        out[i, col["fvg_bull_present"]] = bp
        out[i, col["fvg_bear_distance_pips"]] = sd
        out[i, col["fvg_bear_width_pips"]] = sw
        out[i, col["fvg_bear_present"]] = sp

        # Latest recent OB is retained after first touch, with mitigation state.
        if bull_ob is not None and i - bull_ob[2] > cfg.ob_max_age_bars:
            bull_ob = None
        if bear_ob is not None and i - bear_ob[2] > cfg.ob_max_age_bars:
            bear_ob = None
        if bull_ob is not None and not bull_ob[3] and hi[i] >= bull_ob[0] and lo[i] <= bull_ob[1]:
            bull_ob[3] = True
        if bear_ob is not None and not bear_ob[3] and hi[i] >= bear_ob[0] and lo[i] <= bear_ob[1]:
            bear_ob[3] = True
        for zone, prefix in ((bull_ob, "ob_bull"), (bear_ob, "ob_bear")):
            if zone is not None:
                out[i, col[prefix + "_distance_pips"]] = _signed_zone_distance(
                    cl[i], zone[0], zone[1], cfg.pip_size
                )
                out[i, col[prefix + "_present"]] = 1.0
                out[i, col[prefix + "_mitigated"]] = float(zone[3])

        # Three-candle FVG formation: candle i is complete; no future candle is read.
        if i >= 2 and consecutive[i] and consecutive[i - 1]:
            if lo[i] > hi[i - 2]:
                bull_fvgs.append((hi[i - 2], lo[i], i))
            if hi[i] < lo[i - 2]:
                bear_fvgs.append((hi[i], lo[i - 2], i))

        # OB proxy: confirmed displacement close through prior range, then last opposite candle.
        lookback_start = max(0, i - cfg.ob_break_lookback)
        no_gap_window = (i >= cfg.ob_break_lookback and consecutive[lookback_start + 1:i + 1].all())
        if no_gap_window and np.isfinite(atr[i]) and atr[i] > 0:
            prior_high = float(np.max(hi[lookback_start:i]))
            prior_low = float(np.min(lo[lookback_start:i]))
            body = abs(cl[i] - op[i])
            if cl[i] > op[i] and cl[i] > prior_high and body >= cfg.ob_impulse_atr * atr[i]:
                candidates = [j for j in range(lookback_start, i) if cl[j] < op[j]]
                if candidates:
                    j = candidates[-1]
                    bull_ob = [lo[j], hi[j], i, False]
            elif cl[i] < op[i] and cl[i] < prior_low and body >= cfg.ob_impulse_atr * atr[i]:
                candidates = [j for j in range(lookback_start, i) if cl[j] > op[j]]
                if candidates:
                    j = candidates[-1]
                    bear_ob = [lo[j], hi[j], i, False]

        # A pivot at k=i-right becomes knowable at close[i], never at close[k].
        k = i - cfg.pivot_right
        left = k - cfg.pivot_left
        if left >= 0 and consecutive[left + 1:i + 1].all():
            if hi[k] > np.max(hi[left:k]) and hi[k] >= np.max(hi[k + 1:i + 1]):
                swing_highs.append(float(hi[k]))
                swing_highs = swing_highs[-2:]
            if lo[k] < np.min(lo[left:k]) and lo[k] <= np.min(lo[k + 1:i + 1]):
                swing_lows.append(float(lo[k]))
                swing_lows = swing_lows[-2:]
            if len(swing_highs) == 2 and len(swing_lows) == 2:
                hh = swing_highs[-1] > swing_highs[-2]
                hl = swing_lows[-1] > swing_lows[-2]
                lh = swing_highs[-1] < swing_highs[-2]
                ll = swing_lows[-1] < swing_lows[-2]
                mss = 1 if hh and hl else -1 if lh and ll else 0
        out[i, col["mss_direction"]] = float(mss)

        # Update current session range after all decisions for this completed bar.
        if session is not None:
            current_session_high = hi[i] if current_session_high is None else max(current_session_high, hi[i])
            current_session_low = lo[i] if current_session_low is None else min(current_session_low, lo[i])
            current_session_count += 1
            if current_session_count > current_session_expected:
                current_session_incomplete = True
        previous_time = stamp

    result = pd.DataFrame(out, columns=FEATURE_NAMES, index=bars.index)
    result.insert(0, "timestamp", bars.timestamp)
    result.attrs.update(
        timeframe=timeframe,
        available_at="bar close; timestamps represent bar opens",
        pip_size=cfg.pip_size,
        feature_names=list(FEATURE_NAMES),
        feature_scales=FEATURE_SCALES,
        config=cfg.__dict__,
        causal=True,
    )
    if not np.isfinite(result[list(FEATURE_NAMES)].to_numpy()).all():
        raise ValueError("La Data Farm produjo features no finitas")
    return result


def normalized_context_matrix(context: pd.DataFrame) -> np.ndarray:
    """Map raw pips to fixed bounded model inputs; preserve raw columns separately."""
    missing = set(FEATURE_NAMES) - set(context.columns)
    if missing:
        raise ValueError(f"Faltan features de contexto: {sorted(missing)}")
    matrix = context.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("La matriz de contexto contiene nulos/infinitos")
    for name, scale in FEATURE_SCALES.items():
        matrix[:, FEATURE_NAMES.index(name)] = np.clip(matrix[:, FEATURE_NAMES.index(name)] / scale, -1., 1.)
    for name in FEATURE_NAMES:
        if name.endswith("_present") or name.endswith("_mitigated") or name.startswith("sweep_"):
            matrix[:, FEATURE_NAMES.index(name)] = np.clip(matrix[:, FEATURE_NAMES.index(name)], 0., 1.)
    matrix[:, FEATURE_NAMES.index("mss_direction")] = np.clip(
        matrix[:, FEATURE_NAMES.index("mss_direction")], -1., 1.
    )
    return matrix
