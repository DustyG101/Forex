from __future__ import annotations

from pathlib import Path
import ast

import numpy as np
import pandas as pd

from src.context_features import (
    ContextConfig, FEATURE_NAMES, build_micro_event_features,
    normalized_context_matrix,
)
from src.multi_asset_evolution import (
    EvolutionConfig, PAIRS, _align_m15_to_m5, assert_zero_context_equivalence,
    expand_generation_zero, load_generation_zero,
    multi_asset_fitness, portfolio_kill_switch,
)


ROOT = Path(__file__).resolve().parents[1]


def _ohlcv_bars(count=130, start="2026-11-09T00:00:00Z", minutes=5):
    timestamps = pd.date_range(start, periods=count, freq=f"{minutes}min", tz="UTC")
    close = 1.1000 + np.arange(count) * 0.00001
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.00010
    low = np.minimum(open_, close) - 0.00010
    # Create a confirmed bullish FVG on bar 3: low[3] > high[1].
    # Bar 4 stays above the lower edge, so the zone remains active.
    high[1] = 1.10002
    open_[3], close[3], high[3], low[3] = 1.10090, 1.10100, 1.10110, 1.10080
    open_[4], close[4], high[4], low[4] = 1.10100, 1.10102, 1.10112, 1.10090
    # At 07:00 UTC the first London bar sweeps the completed Asia high
    # and closes back below it.
    high[84] = 1.10300
    close[84] = 1.10090
    open_[84] = 1.10092
    low[84] = 1.10070
    return pd.DataFrame(dict(
        timestamp=timestamps, open=open_, high=high, low=low,
        close=close, volume=np.ones(count),
    ))


def test_context_inputs_emit_fvg_and_sweep_as_context_not_gates():
    bars = _ohlcv_bars()
    features = build_micro_event_features(bars, timeframe="M5")
    assert tuple(n for n in features.columns if n in FEATURE_NAMES) == FEATURE_NAMES
    # FVG formed at bar 3; is exposed only to the next completed decision.
    assert features.loc[3, "fvg_bull_present"] == 0.0
    assert features.loc[4, "fvg_bull_present"] == 1.0
    assert features.loc[4, "fvg_bull_width_pips"] > 0.0
    assert features.loc[84, "sweep_asia_high"] == 1.0
    assert features.loc[84, "ob_bull_present"] in (0.0, 1.0)
    assert set(features.mss_direction.unique()).issubset({-1.0, 0.0, 1.0})
    model_inputs = normalized_context_matrix(features)
    assert model_inputs.shape == (len(bars), len(FEATURE_NAMES))
    assert np.isfinite(model_inputs).all()
    assert np.max(np.abs(model_inputs)) <= 1.0


def test_future_bars_do_not_change_past_context_features():
    bars = _ohlcv_bars()
    first = build_micro_event_features(bars, timeframe="M5")
    altered = bars.copy()
    altered.loc[60:, ["open", "high", "low", "close"]] *= 1.07
    second = build_micro_event_features(altered, timeframe="M5")
    np.testing.assert_array_equal(
        first.loc[:59, FEATURE_NAMES].to_numpy(),
        second.loc[:59, FEATURE_NAMES].to_numpy(),
    )


def test_generation_zero_is_exactly_the_six_verified_pr44_pass_genomes():
    original, metadata, payload = load_generation_zero(
        ROOT / "research/multi_currency/generation_0_seeds.json"
    )
    assert original.shape == (6, 22)
    assert len(metadata) == 6
    assert payload["source_q3_verdict_sha256"]
    extended = expand_generation_zero(original)
    assert extended.shape[0] == 6
    assert extended.shape[1] == 22 + 2 * 42
    assert np.all(extended[:, 9:51] == 0.0)
    assert np.all(extended[:, 60:102] == 0.0)
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(128, 8))
    for genes in original:
        assert_zero_context_equivalence(genes, sample)


def test_multi_asset_fitness_uses_minimum_pf_and_maximum_drawdown():
    metrics = {
        "EURUSD": dict(profit_factor=1.50, profit_factor_status="finite", max_drawdown_fraction=0.10, trades=50, account_stop=False),
        "GBPUSD": dict(profit_factor=1.80, profit_factor_status="finite", max_drawdown_fraction=0.05, trades=40, account_stop=False),
        "AUDUSD": dict(profit_factor=1.20, profit_factor_status="finite", max_drawdown_fraction=0.12, trades=60, account_stop=False),
        "USDCAD": dict(profit_factor=1.70, profit_factor_status="finite", max_drawdown_fraction=0.08, trades=45, account_stop=False),
    }
    result = multi_asset_fitness(metrics)
    assert result["fitness"] == 1.20 * (1.0 - 0.12)
    assert result["eligible"] is True
    assert result["formula"] == "min(PF_pair) * (1 - max(DD_pair_fraction))"


def test_missing_pair_or_no_trades_cannot_receive_positive_fitness():
    metrics = {
        pair: dict(profit_factor=1.5, max_drawdown_fraction=0.1, trades=10, account_stop=False)
        for pair in PAIRS
    }
    del metrics["USDCAD"]
    try:
        multi_asset_fitness(metrics)
        assert False, "must reject incomplete four-pair evaluation"
    except ValueError:
        pass
    metrics["USDCAD"] = dict(
        profit_factor=None, max_drawdown_fraction=0.0, trades=0, account_stop=False
    )
    result = multi_asset_fitness(metrics)
    assert result["fitness"] == 0.0
    assert result["eligible"] is False


def test_profit_factor_infinite_is_distinct_from_undefined_no_trades():
    metrics = {
        pair: dict(
            profit_factor=None,
            profit_factor_status="infinite_no_losses",
            max_drawdown_fraction=0.05,
            trades=10,
            wins=10,
            losses=0,
            account_stop=False,
        )
        for pair in PAIRS
    }
    result = multi_asset_fitness(metrics)
    assert result["eligible"] is True
    assert result["fitness_status"] == "unbounded_infinite"
    assert result["min_profit_factor_status"] == "infinite_no_losses"


def test_correlated_losses_and_combined_drawdown_trigger_global_kill_switch():
    returns = np.zeros((120, 4), dtype=float)
    returns[40:60, :] = -0.012
    result = portfolio_kill_switch(returns)
    assert result["triggered"] is True
    assert result["reason"] in (
        "combined_drawdown_limit", "correlated_losses_with_drawdown"
    )
    assert result["combined_drawdown"] >= 0.15



def test_m15_alignment_rejects_context_at_exactly_15_minute_age():
    minute = 60 * 1_000_000_000
    m5 = np.asarray([5, 10, 15, 20, 25, 30], dtype=np.int64) * minute
    m15 = np.asarray([15], dtype=np.int64) * minute
    valid, indices = _align_m15_to_m5(m5, m15)
    np.testing.assert_array_equal(valid, [False, False, True, True, True, False])
    np.testing.assert_array_equal(indices, [-1, -1, 0, 0, 0, 0])


def test_multi_currency_delegates_to_the_existing_evolution_engine():
    source = (ROOT / "src/multi_asset_evolution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "evolution_multiscale"
    ]
    assert any(
        {alias.name for alias in node.names} >= {"POPULATION", "evolve_population"}
        for node in imports
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "breed_generation"
        for node in tree.body
    )
    config = EvolutionConfig()
    assert config.generations == 20
    assert config.seed == 20261110
