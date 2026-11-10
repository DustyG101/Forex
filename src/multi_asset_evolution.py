"""Lightweight multi-currency genetic evolution from the exact PR44 PASS seeds.

No neural-network/ML framework is imported. Each candidate is replayed in four
isolated accounts. The specified multi-asset fitness is min(PF) * (1-max DD);
a separate portfolio kill-switch blocks candidates with unsafe account stops
or correlated combined drawdown. Q3 is provenance only, never an input dataset.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_gate import AdaptiveGate
from .context_features import FEATURE_NAMES, normalized_context_matrix
PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "USDCAD")
BASE_FEATURE_COUNT = 8
ORIGINAL_GENOME_LENGTH = 22
CONTEXT_PER_TIMEFRAME = len(FEATURE_NAMES)
CONTEXT_TIMEFRAMES = 2
CONTEXT_COUNT = CONTEXT_PER_TIMEFRAME * CONTEXT_TIMEFRAMES
EXTENDED_GENOME_LENGTH = ORIGINAL_GENOME_LENGTH + 2 * CONTEXT_COUNT
GENERATION_ZERO_SCHEMA = "PR44_MULTI_CURRENCY_GENERATION_ZERO_v1"
SEED_VERDICT_SHA = "f6b7ceeeb3a042b698420215c0224faed9cc73d7c0e592d5288e4ab1e607ec83"

INITIAL_BALANCE = 1000.0
STAKE = 10.0
PAYOUT = 0.80
ACCOUNT_DRAWDOWN_LIMIT = 0.20
HOURLY_TRADE_CAP = 2
GATE_PERCENTILE = 0.85

# Layout preserves the original PR44 gene positions/meaning. New M5/M15
# context weights are inserted adjacent to their respective linear heads.
ENTRY_BASE = slice(0, 8)
ENTRY_BIAS = 8
ENTRY_CONTEXT = slice(9, 9 + CONTEXT_COUNT)
DIRECTION_BASE = slice(9 + CONTEXT_COUNT, 17 + CONTEXT_COUNT)
DIRECTION_BIAS = 17 + CONTEXT_COUNT
DIRECTION_CONTEXT = slice(18 + CONTEXT_COUNT, 18 + 2 * CONTEXT_COUNT)
GATE_START = 18 + 2 * CONTEXT_COUNT


@dataclass(frozen=True)
class EvolutionConfig:
    """Solo parámetros de la corrida; población y operadores vienen del motor original."""
    generations: int = 20
    seed: int = 20261110

    def __post_init__(self):
        if type(self.generations) is not int or not 1 <= self.generations <= 100:
            raise ValueError("generations debe estar entre 1 y 100")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed debe ser un entero entre 0 y 2**32-1")


def _genome_digest(genome) -> str:
    array = np.asarray(genome, dtype="<f8")
    return sha256(array.tobytes()).hexdigest()


def load_generation_zero(path: str | Path, *, expected_verdict_sha: str = SEED_VERDICT_SHA):
    """Load and verify all six Q3 PASS genomes; never infer weights from metrics."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != GENERATION_ZERO_SCHEMA:
        raise ValueError("Esquema de población semilla incompatible")
    if payload.get("source_protocol") != "PR44_Q3_24_frozen_genomes_single_pass_v1":
        raise ValueError("El origen no es la auditoría PR44 Q3")
    if payload.get("source_q3_verdict_sha256") != expected_verdict_sha:
        raise ValueError("SHA256 del veredicto Q3 no coincide")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 6 or payload.get("seed_count") != 6:
        raise ValueError("Generación 0 requiere exactamente seis semillas PASS")
    ids, population, metadata = set(), [], []
    for seed in seeds:
        if seed.get("generation") != 0 or seed.get("q3_metrics", {}).get("qualifies") is not True:
            raise ValueError("La semilla no es PASS de Generación 0")
        if seed.get("group") not in ("PR42_CONTROL", "PR43_EVOLVED"):
            raise ValueError("Grupo de semilla desconocido")
        if seed.get("origin_pair") not in PAIRS:
            raise ValueError("Par de origen desconocido")
        genes = np.asarray(seed.get("genome"), dtype="<f8")
        if genes.shape != (ORIGINAL_GENOME_LENGTH,) or not np.isfinite(genes).all():
            raise ValueError("Cada genoma original debe tener exactamente 22 genes finitos")
        digest = _genome_digest(genes)
        if digest != seed.get("genome_sha256"):
            raise ValueError(f"Hash de genoma incorrecto: {seed.get('id')}")
        if seed.get("id") in ids:
            raise ValueError("ID de semilla duplicado")
        ids.add(seed["id"])
        population.append(genes.copy())
        metadata.append({k: seed.get(k) for k in (
            "id", "group", "origin_pair", "origin_seed", "genome_sha256",
            "source_protocol", "source_generation", "source_artifact_id",
            "source_artifact_sha256", "q3_metrics",
        )})
    if len(ids) != 6:
        raise ValueError("Población G0 incompleta")
    return np.stack(population), metadata, payload


def expand_seed_genome(original: np.ndarray) -> np.ndarray:
    """Zero-extend context weights while preserving the exact original response."""
    old = np.asarray(original, dtype="<f8")
    if old.shape != (ORIGINAL_GENOME_LENGTH,) or not np.isfinite(old).all():
        raise ValueError("El genoma PR44 debe tener 22 genes finitos")
    extended = np.zeros(EXTENDED_GENOME_LENGTH, dtype=np.float64)
    extended[ENTRY_BASE] = old[0:8]
    extended[ENTRY_BIAS] = old[8]
    extended[ENTRY_CONTEXT] = 0.0
    extended[DIRECTION_BASE] = old[9:17]
    extended[DIRECTION_BIAS] = old[17]
    extended[DIRECTION_CONTEXT] = 0.0
    extended[GATE_START:GATE_START + 4] = old[18:22]
    return extended


def expand_generation_zero(original_population: np.ndarray) -> np.ndarray:
    values = np.asarray(original_population, dtype=np.float64)
    if values.shape != (6, ORIGINAL_GENOME_LENGTH):
        raise ValueError("La población original debe ser 6 x 22")
    return np.stack([expand_seed_genome(row) for row in values])


def genome_outputs(genome: np.ndarray, base_features: np.ndarray, context_features: np.ndarray):
    """Apply the PR44 local head with optional learned M5/M15 context weights."""
    g = np.asarray(genome, dtype=np.float64)
    base = np.asarray(base_features, dtype=np.float64)
    context = np.asarray(context_features, dtype=np.float64)
    if g.shape != (EXTENDED_GENOME_LENGTH,):
        raise ValueError("Genoma extendido con longitud incorrecta")
    if base.ndim != 2 or base.shape[1] != BASE_FEATURE_COUNT:
        raise ValueError("La matriz base PR44 debe tener ocho columnas")
    if context.shape != (len(base), CONTEXT_COUNT):
        raise ValueError("La matriz de contexto debe contener 21 features M5 + 21 M15")
    if not np.isfinite(base).all() or not np.isfinite(context).all() or not np.isfinite(g).all():
        raise ValueError("Features y genes deben ser finitos")
    entry = base @ g[ENTRY_BASE] + g[ENTRY_BIAS] + context @ g[ENTRY_CONTEXT]
    raw_direction = base @ g[DIRECTION_BASE] + g[DIRECTION_BIAS] + context @ g[DIRECTION_CONTEXT]
    scale = 4.5 + 2.0 * np.tanh(g[GATE_START])
    score = scale * np.tanh(raw_direction)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    margin_floor = 0.3 + 2.0 * sigmoid(g[GATE_START + 3])
    garch_low = 0.35 + sigmoid(g[GATE_START + 1])
    garch_high = 1.2 + 2.0 * sigmoid(g[GATE_START + 2])
    return entry, score, margin_floor, garch_low, garch_high


def assert_zero_context_equivalence(original: np.ndarray, base_features: np.ndarray):
    """Regression guard: zero-padded G0 must score exactly like original PR44."""
    old = np.asarray(original, dtype=np.float64)
    new = expand_seed_genome(old)
    base = np.asarray(base_features, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 8 or not np.isfinite(base).all():
        raise ValueError("base_features debe ser una matriz finita N x 8")
    old_entry = base @ old[:8] + old[8]
    old_score = (4.5 + 2.0 * np.tanh(old[18])) * np.tanh(base @ old[9:17] + old[17])
    ctx = np.zeros((len(base), CONTEXT_COUNT), dtype=np.float64)
    entry, score, _, _, _ = genome_outputs(new, base, ctx)
    if not np.array_equal(old_entry, entry) or not np.array_equal(old_score, score):
        raise AssertionError("El padding de contexto alteró la respuesta de Generación 0")
    return True


def garch_ratio(volatility: np.ndarray, prior_tail: np.ndarray | None = None):
    """PR44-style causal volatility ratio: sigma[t] / median(sigma[t-48:t])."""
    sigma = np.asarray(volatility, dtype=np.float64)
    if sigma.ndim != 1 or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Volatilidad inválida")
    current = np.maximum(sigma, 1e-12)
    offset = 0
    if prior_tail is not None:
        tail = np.asarray(prior_tail, dtype=np.float64)
        if tail.shape != (48,) or not np.isfinite(tail).all() or (tail < 0).any():
            raise ValueError("La cola GARCH previa debe contener 48 valores finitos no negativos")
        current = np.r_[np.maximum(tail, 1e-12), current]
        offset = 48
    past = pd.Series(current).shift(1).rolling(48, min_periods=24).median().to_numpy()[offset:]
    ratio = sigma / np.where(np.isfinite(past), past, np.nan)
    return ratio


def _causal_gate_reasons(decision_times: pd.DatetimeIndex, scores: np.ndarray,
                         outcomes: np.ndarray, proposed: np.ndarray, step: pd.Timedelta):
    """p85 gate: an outcome enters its state only after the corresponding expiry."""
    gate = AdaptiveGate(percentile=GATE_PERCENTILE, elastic=True)
    pending = deque()
    reasons = np.full(len(decision_times), "not_proposed", dtype="<U24")
    for idx in proposed:
        now = decision_times[idx]
        while pending and pending[0][0] <= now:
            _, won = pending.popleft()
            gate.observe_settlement(won)
        decision = gate.decide(float(scores[idx]))
        reasons[idx] = decision.reason
        side = 1 if scores[idx] >= 0 else -1
        outcome = int(outcomes[idx])
        won = None if outcome == 0 else bool(side == outcome)
        pending.append((now + step, won))
    return reasons


def _evaluate_pair(genome, data: dict, pair: str, *, portfolio_stop_time_ns: int | None = None):
    times = pd.DatetimeIndex(pd.to_datetime(data["decision_times_ns"], unit="ns", utc=True))
    base = np.asarray(data["base_features"], dtype=np.float64)
    context = np.asarray(data["context_features"], dtype=np.float64)
    volatility = np.asarray(data["volatility"], dtype=np.float64)
    outcomes = np.asarray(data["outcomes"], dtype=np.int8)
    prior_tail = data.get("q2_eligible_volatility_tail")
    if (len(times) != len(base) or len(times) != len(context) or len(times) != len(volatility)
            or len(times) != len(outcomes) or not times.is_monotonic_increasing
            or not times.is_unique or not np.isfinite(base).all()
            or not np.isfinite(context).all() or not np.isfinite(volatility).all()
            or not np.isin(outcomes, (-1, 0, 1)).all()):
        raise ValueError(f"Dataset {pair} desalineado o inválido")
    if len(times) < 100:
        raise ValueError(f"Dataset {pair} requiere al menos 100 barras")
    step = pd.Timedelta(minutes=5)
    if np.any(np.diff(times.asi8) < step.value):
        raise ValueError(f"Dataset {pair} contiene barras solapadas o desordenadas")

    entry, score, margin_floor, low, high = genome_outputs(genome, base, context)
    ratio = garch_ratio(volatility, prior_tail)
    eligible = (entry > 0.0) & (np.abs(score) > margin_floor) & (ratio >= low) & (ratio <= high)
    # A signal is only evaluable if the next M5 close exists and is contiguous.
    contiguous_next = np.zeros(len(times), dtype=bool)
    if len(times) > 1:
        contiguous_next[:-1] = np.diff(times.asi8) == step.value
    valid = eligible & contiguous_next
    proposed = np.flatnonzero(valid)
    reasons = _causal_gate_reasons(times, score, outcomes, proposed, step)

    # Rolling one-hour cap, applied after the adaptive gate.
    accepted = []
    recent = deque()
    for idx in proposed:
        if reasons[idx] != "accepted":
            continue
        now = times[idx]
        while recent and now - recent[0] >= pd.Timedelta(hours=1):
            recent.popleft()
        if len(recent) >= HOURLY_TRADE_CAP:
            reasons[idx] = "hourly_cap"
        else:
            recent.append(now)
            accepted.append(idx)

    balance = peak = INITIAL_BALANCE
    max_dd = 0.0
    stopped = False
    stop_reason = None
    stop_time = None
    gross_win = gross_loss = 0.0
    pnl_total = 0.0
    wins = losses = ties = executed = 0
    step_returns = {}
    ledger = []
    accepted_set = set(accepted)
    for idx in proposed:
        side = "CALL" if score[idx] >= 0 else "PUT"
        item = dict(
            pair=pair,
            decision_utc=times[idx].isoformat(),
            expiry_utc=(times[idx] + step).isoformat(),
            side=side,
            score=float(score[idx]),
            gate_reason=str(reasons[idx]),
            executed=False,
            pnl_usd=None,
            exit_reason="not_entered",
        )
        if idx not in accepted_set:
            ledger.append(item)
            continue
        if portfolio_stop_time_ns is not None and times[idx].value >= portfolio_stop_time_ns:
            item["gate_reason"] = "portfolio_kill_switch"
            item["exit_reason"] = "portfolio_kill_switch"
            ledger.append(item)
            continue
        if stopped:
            item["exit_reason"] = "account_stopped"
            ledger.append(item)
            continue
        outcome = int(outcomes[idx])
        if outcome == 0:
            pnl = 0.0
            ties += 1
        elif (1 if score[idx] >= 0 else -1) == outcome:
            pnl = PAYOUT * STAKE
            gross_win += pnl
            wins += 1
        else:
            pnl = -STAKE
            gross_loss += -pnl
            losses += 1
        before = balance
        balance += pnl
        peak = max(peak, balance)
        dd = max(0.0, 1.0 - balance / peak)
        max_dd = max(max_dd, dd)
        pnl_total += pnl
        executed += 1
        settle_time = times[idx] + step
        step_returns[settle_time.value] = step_returns.get(settle_time.value, 0.0) + pnl / before
        item.update(executed=True, pnl_usd=float(pnl), balance_usd=float(balance),
                    drawdown_fraction=float(dd), exit_reason="expiry")
        if balance < STAKE or dd + 1e-12 >= ACCOUNT_DRAWDOWN_LIMIT:
            stopped = True
            stop_reason = "insufficient_balance" if balance < STAKE else "drawdown_limit"
            stop_time = settle_time.isoformat()
            item["account_stop"] = stop_reason
        ledger.append(item)

    pf = gross_win / gross_loss if gross_loss > 0 else None
    if gross_loss > 0:
        pf_status = "finite"
    elif executed == 0:
        pf_status = "undefined_no_trades"
    elif gross_win > 0:
        pf_status = "infinite_no_losses"
    else:
        pf_status = "undefined_no_wins_or_losses"
    observed_hours = len(times) * 5.0 / 60.0
    metrics = dict(
        pair=pair,
        profit_factor=float(pf) if pf is not None else None,
        profit_factor_status=pf_status,
        max_drawdown_fraction=float(max_dd),
        max_drawdown_pct=float(max_dd * 100.0),
        trades=int(executed), wins=int(wins), losses=int(losses), ties=int(ties),
        pnl_usd=float(pnl_total), account_stop=bool(stopped), stop_reason=stop_reason,
        account_stop_time_utc=stop_time,
        observed_hours=float(observed_hours),
        trades_per_hour=float(executed / observed_hours) if observed_hours else 0.0,
        proposed=int(len(proposed)), accepted=int(len(accepted)),
    )
    return dict(metrics=metrics, ledger=ledger, step_returns=step_returns,
                decision_times_ns=times.asi8.copy())


def _max_pairwise_loss_correlation(window: np.ndarray) -> float | None:
    """Correlation of simultaneous downside returns; constant pairs are ignored."""
    losses = np.minimum(np.asarray(window, dtype=np.float64), 0.0)
    if losses.ndim != 2 or losses.shape[1] != 4:
        raise ValueError("La correlación requiere cuatro series de pares")
    # Constant downside series carry no correlation information. Exclude them
    # before corrcoef to avoid invalid-divide warnings and undefined pairs.
    variable = np.std(losses, axis=0) > 0.0
    if np.count_nonzero(variable) < 2:
        return None
    corr = np.corrcoef(losses[:, variable], rowvar=False)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    finite = upper[np.isfinite(upper)]
    return float(np.max(finite)) if len(finite) else None


def portfolio_kill_switch(
    returns_matrix: np.ndarray, *, event_times_ns: np.ndarray | None = None,
    drawdown_limit: float = 0.15,
    correlation_drawdown_trigger: float = 0.08,
    loss_correlation_limit: float = 0.85,
    correlation_window: int = 96,
    min_observations: int = 24,
):
    """Find the FIRST combined-risk stop on synchronized four-pair equity.

    Each row is one settlement timestamp; missing pair activity is a zero
    return because each account is independently held in cash otherwise.
    Correlation is the rolling correlation of downside returns, a measurable
    proxy for synchronized drawdown pressure. Correlation alone never stops:
    it must coincide with the configured combined portfolio drawdown.
    """
    values = np.asarray(returns_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("El kill-switch requiere retornos finitos T x 4")
    if not (0 < drawdown_limit <= 1 and 0 <= correlation_drawdown_trigger <= drawdown_limit
            and -1 <= loss_correlation_limit <= 1 and correlation_window >= 2
            and min_observations >= 2):
        raise ValueError("Umbrales del kill-switch inválidos")
    event_times = None
    if event_times_ns is not None:
        event_times = np.asarray(event_times_ns, dtype=np.int64)
        if (event_times.shape != (len(values),) or
                (len(event_times) > 1 and np.any(np.diff(event_times) <= 0))):
            raise ValueError("Tiempos del portafolio desalineados")
    if len(values) == 0:
        return dict(triggered=False, reason="no_observations", combined_drawdown=0.0,
                    max_loss_correlation=None, observations=0, trigger_index=None,
                    triggered_at_ns=None, thresholds=dict(
                        drawdown_limit=drawdown_limit,
                        correlation_drawdown_trigger=correlation_drawdown_trigger,
                        loss_correlation_limit=loss_correlation_limit,
                        correlation_window=correlation_window,
                        min_observations=min_observations,
                    ))

    combined = np.mean(values, axis=1)
    combined = np.maximum(combined, -0.999999)
    equity = np.cumprod(1.0 + combined)
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / peaks
    max_dd = float(np.max(drawdowns))
    trigger_index = None
    trigger_reason = None
    max_corr_seen = None

    for i in range(len(values)):
        if drawdowns[i] >= drawdown_limit:
            trigger_index = i
            trigger_reason = "combined_drawdown_limit"
            break
        begin = max(0, i - correlation_window + 1)
        window = values[begin:i + 1]
        if len(window) >= min_observations:
            corr = _max_pairwise_loss_correlation(window)
            if corr is not None:
                max_corr_seen = corr if max_corr_seen is None else max(max_corr_seen, corr)
            if (corr is not None and corr >= loss_correlation_limit and
                    drawdowns[i] >= correlation_drawdown_trigger):
                trigger_index = i
                trigger_reason = "correlated_losses_with_drawdown"
                break

    if max_corr_seen is None:
        begin = max(0, len(values) - correlation_window)
        tail_corr = _max_pairwise_loss_correlation(values[begin:])
        max_corr_seen = tail_corr

    triggered_at_ns = (
        int(event_times[trigger_index])
        if trigger_index is not None and event_times is not None else None
    )
    return dict(
        triggered=trigger_index is not None,
        reason=trigger_reason,
        combined_drawdown=max_dd,
        max_loss_correlation=max_corr_seen,
        observations=len(values),
        trigger_index=trigger_index,
        triggered_at_ns=triggered_at_ns,
        thresholds=dict(
            drawdown_limit=drawdown_limit,
            correlation_drawdown_trigger=correlation_drawdown_trigger,
            loss_correlation_limit=loss_correlation_limit,
            correlation_window=correlation_window,
            min_observations=min_observations,
        ),
    )


def multi_asset_fitness(pair_metrics: dict, kill_switch: dict | None = None):
    """Exact requested fitness: min(PF_i) * (1 - max(DD_i)).

    PF is mathematically +infinity when a pair has wins but no losses.
    A pair with no trades, or only refunds, has undefined PF and cannot qualify.
    JSON reports encode an all-pairs infinite result as null + status, never NaN.
    """
    if set(pair_metrics) != set(PAIRS):
        raise ValueError("El fitness exige resultados completos de los cuatro pares")
    pfs, drawdowns = [], []
    reasons = []
    for pair in PAIRS:
        metrics = pair_metrics[pair]
        pf = metrics.get("profit_factor")
        status = metrics.get("profit_factor_status")
        dd = metrics.get("max_drawdown_fraction")
        trades = int(metrics.get("trades", 0))
        wins = int(metrics.get("wins", 0))
        if trades <= 0:
            reasons.append(f"{pair}:no_trades")
        if status == "finite" and pf is not None and np.isfinite(pf) and pf >= 0:
            pfs.append(float(pf))
        elif status == "infinite_no_losses" and trades > 0 and wins > 0:
            pfs.append(math.inf)
        else:
            reasons.append(f"{pair}:undefined_profit_factor")
        if dd is None or not np.isfinite(dd) or not 0 <= dd <= 1:
            raise ValueError(f"Drawdown inválido en {pair}")
        if metrics.get("account_stop"):
            reasons.append(f"{pair}:account_stop")
        drawdowns.append(float(dd))
    min_pf = min(pfs) if len(pfs) == 4 else None
    max_dd = max(drawdowns)
    raw_value = min_pf * (1.0 - max_dd) if min_pf is not None else 0.0
    unbounded = bool(min_pf is not None and math.isinf(raw_value))
    value = None if unbounded else float(raw_value)
    kill = kill_switch or {"triggered": False, "reason": None}
    if kill.get("triggered"):
        reasons.append("portfolio_kill_switch:" + str(kill.get("reason")))
    return dict(
        fitness=value,
        fitness_status="unbounded_infinite" if unbounded else "finite",
        formula="min(PF_pair) * (1 - max(DD_pair_fraction))",
        min_profit_factor=None if min_pf is None or math.isinf(min_pf) else float(min_pf),
        min_profit_factor_status="infinite_no_losses" if min_pf is not None and math.isinf(min_pf) else (
            "finite" if min_pf is not None else "undefined"
        ),
        max_drawdown_fraction=float(max_dd),
        eligible=not reasons,
        rejection_reasons=reasons,
        portfolio_kill_switch=kill,
    )


def _fitness_rank_value(fitness: dict) -> float:
    """Finite comparable rank value without serializing JSON Infinity."""
    if fitness.get("fitness_status") == "unbounded_infinite":
        return 1e12
    value = fitness.get("fitness")
    if value is None or not np.isfinite(value):
        return -1e12
    return float(np.clip(value, -1e12, 1e12))


def _portfolio_returns(pair_results: dict):
    events = sorted({stamp for result in pair_results.values()
                     for stamp in result["step_returns"]})
    matrix = np.zeros((len(events), 4), dtype=np.float64)
    positions = {stamp: i for i, stamp in enumerate(events)}
    for j, pair in enumerate(PAIRS):
        for stamp, value in pair_results[pair]["step_returns"].items():
            matrix[positions[stamp], j] = value
    return matrix, np.asarray(events, dtype=np.int64)


def evaluate_multicurrency_genome(genome: np.ndarray, assets: dict):
    """Evaluate isolated accounts and enforce the earliest synchronized portfolio stop."""
    if set(assets) != set(PAIRS):
        raise ValueError("Se requieren datasets para EURUSD, GBPUSD, AUDUSD y USDCAD")

    # First pass discovers the earliest portfolio or single-account stop.
    preliminary = {pair: _evaluate_pair(genome, assets[pair], pair) for pair in PAIRS}
    initial_returns, initial_times = _portfolio_returns(preliminary)
    kill = portfolio_kill_switch(initial_returns, event_times_ns=initial_times)

    account_stops = []
    for pair in PAIRS:
        metric = preliminary[pair]["metrics"]
        if metric.get("account_stop") and metric.get("account_stop_time_utc"):
            timestamp = pd.Timestamp(metric["account_stop_time_utc"])
            account_stops.append((int(timestamp.value), pair, metric.get("stop_reason")))

    stop_candidates = []
    if kill.get("triggered") and kill.get("triggered_at_ns") is not None:
        stop_candidates.append((
            int(kill["triggered_at_ns"]),
            str(kill.get("reason")),
            None,
        ))
    if account_stops:
        earliest_account = min(account_stops, key=lambda item: item[0])
        stop_candidates.append((
            earliest_account[0],
            "individual_account_stop",
            {"pair": earliest_account[1], "reason": earliest_account[2]},
        ))

    if stop_candidates:
        stop_ns, stop_reason, account_stop_detail = min(stop_candidates, key=lambda item: item[0])
        # Stop all four account streams before decisions at/after the trigger.
        pair_results = {
            pair: _evaluate_pair(
                genome, assets[pair], pair, portfolio_stop_time_ns=stop_ns
            ) for pair in PAIRS
        }
        kill = dict(kill)
        kill.update(
            triggered=True,
            reason=stop_reason,
            triggered_at_ns=int(stop_ns),
            triggered_at_utc=pd.Timestamp(stop_ns, tz="UTC").isoformat(),
            account_stop_detail=account_stop_detail,
        )
    else:
        pair_results = preliminary
        kill = dict(kill, triggered_at_utc=None, account_stop_detail=None)

    metrics = {pair: pair_results[pair]["metrics"] for pair in PAIRS}
    final_returns, final_times = _portfolio_returns(pair_results)
    fitness = multi_asset_fitness(metrics, kill)
    return dict(
        pair_metrics=metrics,
        fitness=fitness,
        ledgers={pair: pair_results[pair]["ledger"] for pair in PAIRS},
        portfolio_returns=final_returns,
        portfolio_event_times_ns=final_times,
    )


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _verify_context_feature_names(asset: dict, label: str):
    encoded = asset.get("context_feature_names")
    if encoded is None:
        raise ValueError(f"{label}: falta el contrato ordenado de nombres de features")
    values = np.asarray(encoded)
    if values.size != 1:
        raise ValueError(f"{label}: context_feature_names debe ser un escalar JSON")
    try:
        names = json.loads(str(values.reshape(-1)[0]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: context_feature_names inválido") from exc
    if tuple(names) != tuple(FEATURE_NAMES):
        raise ValueError(f"{label}: orden/versión de features no coincide con context features")


def _align_m15_to_m5(m5_decision_times_ns: np.ndarray, m15_decision_times_ns: np.ndarray):
    """Return M5 rows with a closed M15 context younger than one full M15 bar."""
    t5 = np.asarray(m5_decision_times_ns, dtype=np.int64)
    t15 = np.asarray(m15_decision_times_ns, dtype=np.int64)
    if (t5.ndim != 1 or t15.ndim != 1 or
            (len(t5) > 1 and np.any(np.diff(t5) <= 0)) or
            (len(t15) > 1 and np.any(np.diff(t15) <= 0))):
        raise ValueError("Tiempos M5/M15 deben ser vectores estrictamente crecientes")
    idx15 = np.searchsorted(t15, t5, side="right") - 1
    valid = idx15 >= 0
    valid_indices = np.flatnonzero(valid)
    # At exactly 15 minutes the prior M15 context is stale: the next M15
    # close should already exist, otherwise the feed has a gap.
    valid[valid_indices] = (
        t5[valid_indices] - t15[idx15[valid_indices]] < 15 * 60 * 1_000_000_000
    )
    return valid, idx15


def load_multicurrency_datasets(data_root: str | Path):
    """Join context features M5/M15 context to the existing PR44 eight-column base.

    Layout:
      DATA_ROOT/PPAIR/M5.npz       and M15.npz  (built by scripts.build_context_features)
      DATA_ROOT/PPAIR/base_m5.npz  (canonical PR41/PR44 base features + volatility)
    base_m5.npz must contain decision_times_ns, timeframe (N x 8), volatility.
    No nearest-neighbour interpolation, forward-fill across gaps, or date edits.
    """
    root = Path(data_root)
    result = {}
    step5 = 5 * 60 * 1_000_000_000
    step15 = 15 * 60 * 1_000_000_000
    for pair in PAIRS:
        folder = root / pair
        m5 = _load_npz(folder / "M5.npz")
        m15 = _load_npz(folder / "M15.npz")
        base = _load_npz(folder / "base_m5.npz")
        required = ("decision_times_ns", "context_features", "close")
        for name, obj in (("M5", m5), ("M15", m15)):
            if any(key not in obj for key in required):
                raise ValueError(f"{pair}/{name}.npz carece de {required}")
            _verify_context_feature_names(obj, f"{pair}/{name}.npz")
        base_times = base.get("decision_times_ns")
        if base_times is None or "timeframe" not in base or "volatility" not in base:
            raise ValueError(f"{pair}/base_m5.npz requiere decision_times_ns, timeframe y volatility")
        t5 = np.asarray(m5["decision_times_ns"], dtype=np.int64)
        t15 = np.asarray(m15["decision_times_ns"], dtype=np.int64)
        tb = np.asarray(base_times, dtype=np.int64)
        base_matrix_all = np.asarray(base["timeframe"], dtype=np.float64)
        vol_all = np.asarray(base["volatility"], dtype=np.float64)
        c5 = np.asarray(m5["context_features"], dtype=np.float64)
        c15 = np.asarray(m15["context_features"], dtype=np.float64)
        close_all = np.asarray(m5["close"], dtype=np.float64)
        if not np.array_equal(t5, tb):
            raise ValueError(f"{pair}: la matriz base M5 no coincide exactamente con context features M5")
        if (t5.ndim != 1 or t15.ndim != 1 or len(t5) < 100 or len(t15) < 40
                or np.any(np.diff(t5) <= 0) or np.any(np.diff(t15) <= 0)):
            raise ValueError(f"{pair}: timestamps insuficientes o no crecientes")
        train_start_ns = pd.Timestamp("2026-01-01T00:00:00Z").value
        train_end_ns = pd.Timestamp("2026-07-01T00:00:00Z").value
        if t5[-1] > train_end_ns or t15[-1] > train_end_ns:
            raise ValueError(f"{pair}: el conjunto contiene datos posteriores al IS H1 2026; Q3 no se carga")
        if (base_matrix_all.shape != (len(t5), BASE_FEATURE_COUNT)
                or vol_all.shape != (len(t5),) or close_all.shape != (len(t5),)
                or c5.shape != (len(t5), CONTEXT_PER_TIMEFRAME)
                or c15.shape != (len(t15), CONTEXT_PER_TIMEFRAME)):
            raise ValueError(f"{pair}: tamaños incompatibles entre precios, features y GARCH")
        if (not np.isfinite(base_matrix_all).all() or not np.isfinite(vol_all).all()
                or not np.isfinite(close_all).all() or not np.isfinite(c5).all()
                or not np.isfinite(c15).all()):
            raise ValueError(f"{pair}: NaN/Inf en la matriz base, volatilidad o contexto")
        if not np.array_equal(np.asarray(m5["row_times_ns"], dtype=np.int64) + step5, t5):
            raise ValueError(f"{pair}: timestamp M5 no corresponde a su cierre")
        if not np.array_equal(np.asarray(m15["row_times_ns"], dtype=np.int64) + step15, t15):
            raise ValueError(f"{pair}: timestamp M15 no corresponde a su cierre")
        # At each M5 decision, use only a recently closed M15 candle.
        valid, idx15 = _align_m15_to_m5(t5, t15)
        if valid.sum() < 100:
            raise ValueError(f"{pair}: cobertura M15 cerrada y reciente insuficiente para alinear M5")
        t5 = t5[valid]
        idx15 = idx15[valid]
        base_matrix = base_matrix_all[valid]
        vol = vol_all[valid]
        m5_frame = pd.DataFrame(c5[valid], columns=FEATURE_NAMES)
        m15_frame = pd.DataFrame(c15[idx15], columns=FEATURE_NAMES)
        context = np.column_stack((
            normalized_context_matrix(m5_frame),
            normalized_context_matrix(m15_frame),
        ))
        close = close_all[valid]
        outcomes = np.zeros(len(t5), dtype=np.int8)
        contiguous = np.zeros(len(t5), dtype=bool)
        if len(t5) > 1:
            contiguous[:-1] = np.diff(t5) == step5
            outcomes[:-1][contiguous[:-1]] = np.sign(np.diff(close)[contiguous[:-1]]).astype(np.int8)
        # Se conservan barras previas a enero para calcular indicadores causales,
        # pero el evolucionador solo recibe decisiones de H1 2026 (Q1/Q2).
        usable = contiguous & (t5 >= train_start_ns) & (t5 < train_end_ns)
        if int(usable.sum()) < 100:
            raise ValueError(f"{pair}: menos de 100 decisiones M5 dentro de H1 2026")
        tail = base.get("q2_eligible_volatility_tail")
        result[pair] = dict(
            decision_times_ns=t5[usable],
            base_features=base_matrix[usable],
            context_features=context[usable],
            volatility=vol[usable],
            outcomes=outcomes[usable],
            q2_eligible_volatility_tail=tail,
        )
        if not np.isfinite(result[pair]["base_features"]).all() or not np.isfinite(result[pair]["volatility"]).all():
            raise ValueError(f"{pair}: features base/volatilidad no finitos")
    return result


def _rank_values(fitness: dict):
    """Orden de selección compartido por el lote y el campeón individual."""
    status = fitness.get("min_profit_factor_status")
    if status == "infinite_no_losses":
        min_pf = 1e12
    elif status == "finite" and fitness.get("min_profit_factor") is not None:
        min_pf = float(fitness["min_profit_factor"])
    else:
        min_pf = -1e12
    return (
        bool(fitness["eligible"]),
        _fitness_rank_value(fitness),
        min_pf,
        -float(fitness["max_drawdown_fraction"]),
    )


def _multi_asset_rank(values: dict):
    return (
        values["qualifies"],
        values["score"],
        values["min_profit_factor"],
        values["negative_max_drawdown"],
    )


def _evaluate_population(population: np.ndarray, assets: dict):
    """Evalúa cada genoma en los cuatro pares y devuelve vectores al motor original."""
    ranked = [_rank_values(evaluate_multicurrency_genome(genome, assets)["fitness"])
              for genome in population]
    return dict(
        qualifies=np.asarray([row[0] for row in ranked], dtype=bool),
        score=np.asarray([row[1] for row in ranked], dtype=np.float64),
        min_profit_factor=np.asarray([row[2] for row in ranked], dtype=np.float64),
        negative_max_drawdown=np.asarray([row[3] for row in ranked], dtype=np.float64),
    )


def run_evolution(seed_manifest: str | Path, data_root: str | Path,
                  output_dir: str | Path, config: EvolutionConfig | None = None):
    """Continúa los seis fundadores con el evolucionador original y fitness conjunto."""
    from .evolution_multiscale import POPULATION, evolve_population

    cfg = config or EvolutionConfig()
    seed_path = Path(seed_manifest)
    original, seed_metadata, _ = load_generation_zero(seed_path)
    founders = expand_generation_zero(original)
    assets = load_multicurrency_datasets(data_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report = dict(
        protocol="PR45_four_pair_context_inputs_on_original_evolution_v2",
        seed_manifest=str(seed_path),
        seed_manifest_sha256=sha256(seed_path.read_bytes()).hexdigest(),
        generation_zero=[dict(meta, context_weights="zero_padded") for meta in seed_metadata],
        config=dict(generations=cfg.generations, seed=cfg.seed, population=POPULATION),
        evolution_operator="src.evolution_multiscale.evolve_population -> src.evolution_engine.seeded_population/breed",
        pairs=list(PAIRS),
        context_features=list(FEATURE_NAMES),
        fitness_definition="min(PF_pair) * (1 - max(DD_pair_fraction))",
        risk_model="four isolated simulated accounts; per-pair stake, payout, 20% account stop and 2 trades/hour",
        training_period="[2026-01-01T00:00:00Z, 2026-07-01T00:00:00Z)",
        q3_loaded=False,
        seed_runs=[],
    )

    for index, (founder, metadata) in enumerate(zip(founders, seed_metadata, strict=True)):
        run_seed = (cfg.seed + index * 1009) % (2**32)
        evolve = evolve_population(
            lambda population: _evaluate_population(population, assets),
            founder.copy(),
            f"pr45_{metadata['id']}_four_pair",
            run_seed,
            first_generation=1,
            last_generation=cfg.generations,
            ranking=_multi_asset_rank,
        )
        genome = np.asarray(evolve["genome"], dtype="<f8")
        if genome.shape != (EXTENDED_GENOME_LENGTH,) or not np.isfinite(genome).all():
            raise ValueError(f"{metadata['id']}: el motor original devolvió un genoma inválido")

        winner = evaluate_multicurrency_genome(genome, assets)
        rank = _rank_values(winner["fitness"])
        selected = _multi_asset_rank(evolve["champion"])
        if (rank[0] != bool(selected[0]) or
                not np.allclose(np.asarray(rank[1:], dtype=float),
                                np.asarray(selected[1:], dtype=float), atol=1e-10, rtol=0)):
            raise ValueError(f"{metadata['id']}: el campeón no reproduce su fitness conjunto")

        ledger_paths = {}
        for pair, trades in winner["ledgers"].items():
            ledger_path = output / f"{metadata['id']}-{pair.lower()}-ledger.json"
            ledger_path.write_text(
                json.dumps(trades, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            ledger_paths[pair] = str(ledger_path)

        genome_path = output / f"{metadata['id']}-genome.npz"
        np.savez_compressed(
            genome_path,
            genome=genome,
            source_genome=np.asarray(founder, dtype="<f8"),
            source_id=np.asarray(metadata["id"]),
        )
        seed_run = dict(
            source_seed_id=metadata["id"],
            source_pair=metadata["origin_pair"],
            source_genome_sha256=metadata["genome_sha256"],
            expanded_founder_sha256=_genome_digest(founder),
            evolved_genome_sha256=_genome_digest(genome),
            evolved_genome=genome.tolist(),
            champion_generation=int(evolve["champion"]["generation"]),
            population=POPULATION,
            generations=cfg.generations,
            evaluations=int(evolve["evaluations"]),
            rank=dict(
                qualifies=rank[0],
                score=rank[1],
                min_profit_factor=rank[2],
                negative_max_drawdown=rank[3],
            ),
            fitness=winner["fitness"],
            pair_metrics=winner["pair_metrics"],
            generation_history=evolve["generations"],
            genome_artifact=str(genome_path),
            ledger_artifacts=ledger_paths,
        )
        report["seed_runs"].append(seed_run)

    best = max(report["seed_runs"], key=lambda row: _multi_asset_rank(row["rank"]))
    report["best_final_seed_id"] = best["source_seed_id"]
    report["best_final_fitness"] = best["fitness"]
    report_path = output / "evolution-report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", type=Path,
                        default=Path("research/multi_currency/generation_0_seeds.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/multi_currency"))
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    run_evolution(
        args.seed_manifest, args.data_root, args.output,
        EvolutionConfig(generations=args.generations, seed=args.seed),
    )


if __name__ == "__main__":
    main()
