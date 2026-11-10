"""Build immutable Data Farm M5/M15 context arrays from local OHLCV CSVs.

Accepted input layout: INPUT_ROOT/PAIR/M1.csv, or PAIR/M5.csv + PAIR/M15.csv.
CSV columns: timestamp,open,high,low,close[,volume]; timestamps are UTC bar
opens. M1 aggregation keeps only complete consecutive groups. No API/network
calls, interpolation, backfill, trade filters, or model fitting.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_farm import (
    DataFarmConfig, FEATURE_NAMES, FEATURE_SCALES,
    build_micro_event_features, validate_ohlcv,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "timestamp" not in frame:
        raise ValueError(f"{path}: se requiere la columna timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame


def aggregate_m1(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """OHLC aggregation from exact consecutive M1 opens; incomplete bins vanish."""
    if minutes not in (5, 15):
        raise ValueError("Solo se agregan barras M5/M15")
    source = frame.copy()
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(source.columns):
        raise ValueError(f"M1 requiere columnas {sorted(required)}")
    if "volume" not in source:
        source["volume"] = 0.0
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="raise")
    for name in ("open", "high", "low", "close", "volume"):
        source[name] = pd.to_numeric(source[name], errors="raise").astype(float)
    source = source.sort_values("timestamp").reset_index(drop=True)
    if source.timestamp.isna().any() or source.timestamp.duplicated().any():
        raise ValueError("M1 tiene timestamps vacíos o duplicados")
    if (source.timestamp != source.timestamp.dt.floor("1min")).any():
        raise ValueError("M1 no está alineado al minuto UTC")
    if not np.isfinite(source[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("M1 contiene nulos o infinitos")
    prices = source[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (source.volume < 0).any():
        raise ValueError("M1 OHLCV inválido")
    if (source.high < prices.max(axis=1)).any() or (source.low > prices.min(axis=1)).any():
        raise ValueError("M1 OHLC incoherente")

    source["bucket"] = source.timestamp.dt.floor(f"{minutes}min")
    rows = []
    for bucket, group in source.groupby("bucket", sort=True):
        expected = pd.date_range(bucket, periods=minutes, freq="1min")
        if len(group) != minutes or not pd.DatetimeIndex(group.timestamp).equals(expected):
            continue
        rows.append(dict(
            timestamp=bucket,
            open=float(group.open.iloc[0]),
            high=float(group.high.max()),
            low=float(group.low.min()),
            close=float(group.close.iloc[-1]),
            volume=float(group.volume.sum()),
        ))
    if not rows:
        raise ValueError(f"No hay velas M{minutes} completas")
    return pd.DataFrame(rows)


def save_asset(frame: pd.DataFrame, *, pair: str, timeframe: str,
               output_path: Path, config: DataFarmConfig, source_path: Path):
    context = build_micro_event_features(frame, timeframe=timeframe, config=config)
    minutes = 5 if timeframe == "M5" else 15
    raw_open_ns = pd.DatetimeIndex(context.timestamp).asi8.astype(np.int64)
    decision_ns = raw_open_ns + minutes * 60 * 1_000_000_000
    matrix = context.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        row_times_ns=raw_open_ns,
        decision_times_ns=decision_ns,
        open=frame.open.to_numpy(dtype=np.float64),
        high=frame.high.to_numpy(dtype=np.float64),
        low=frame.low.to_numpy(dtype=np.float64),
        close=frame.close.to_numpy(dtype=np.float64),
        volume=frame.volume.to_numpy(dtype=np.float64),
        context_features=matrix,
        context_feature_names=np.asarray(json.dumps(FEATURE_NAMES)),
    )
    return dict(
        pair=pair, timeframe=timeframe, rows=len(frame),
        first_open_utc=context.timestamp.iloc[0].isoformat(),
        last_open_utc=context.timestamp.iloc[-1].isoformat(),
        available_at="bar close",
        source=str(source_path),
        source_sha256=file_sha256(source_path),
        output=str(output_path),
        output_sha256=file_sha256(output_path),
        context_feature_count=len(FEATURE_NAMES),
        context_feature_names=list(FEATURE_NAMES),
        fixed_input_scales=FEATURE_SCALES,
    )


def build_data_farm(input_root: Path, output_root: Path, pairs=(
    "EURUSD", "GBPUSD", "AUDUSD", "USDCAD"
), config: DataFarmConfig | None = None):
    cfg = config or DataFarmConfig()
    report = dict(
        schema="FOREX_DATA_FARM_CONTEXT_v1",
        causal=True,
        no_trade_gates=True,
        no_model_fit=True,
        timeframe_timestamp_convention="UTC bar-open; decision features available at bar close",
        session_windows_utc={
            "asia": [cfg.asia_start_hour_utc, cfg.asia_end_hour_utc],
            "london": [cfg.london_start_hour_utc, cfg.london_end_hour_utc],
            "new_york": [cfg.new_york_start_hour_utc, cfg.new_york_end_hour_utc],
        },
        config=cfg.__dict__,
        assets=[],
    )
    for pair in pairs:
        folder = input_root / pair
        m1 = folder / "M1.csv"
        if m1.exists():
            source = read_ohlcv(m1)
            frames = {"M5": aggregate_m1(source, 5), "M15": aggregate_m1(source, 15)}
            source_paths = {"M5": m1, "M15": m1}
        else:
            m5_path, m15_path = folder / "M5.csv", folder / "M15.csv"
            if not m5_path.exists() or not m15_path.exists():
                raise FileNotFoundError(
                    f"{pair}: aportar {m1}, o ambos {m5_path} y {m15_path}"
                )
            frames = {
                "M5": validate_ohlcv(read_ohlcv(m5_path), "M5"),
                "M15": validate_ohlcv(read_ohlcv(m15_path), "M15"),
            }
            source_paths = {"M5": m5_path, "M15": m15_path}
        for timeframe, frame in frames.items():
            path = output_root / pair / f"{timeframe}.npz"
            report["assets"].append(save_asset(
                frame, pair=pair, timeframe=timeframe,
                output_path=path, config=cfg, source_path=source_paths[timeframe],
            ))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/data_farm"))
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"],
                        choices=["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"])
    parser.add_argument("--pip-size", type=float, default=0.0001)
    args = parser.parse_args()
    config = DataFarmConfig(pip_size=args.pip_size)
    report = build_data_farm(args.input_root, args.output_root, args.pairs, config)
    print(json.dumps({
        "assets": len(report["assets"]),
        "rows": {f"{item['pair']}_{item['timeframe']}": item["rows"] for item in report["assets"]},
        "manifest": str(args.output_root / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
