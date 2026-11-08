"""Ablación causal del gate elástico con el genoma sellado Gen 354/43."""

from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import load_minutes, m5_features
from scripts.run_m5_direction_benchmark import admission_blocks, load_references
from scripts.run_m5_gen333 import ARM, GEN333_SHA256, sealed_gen333
from scripts.run_m5_multiscale import (
    APRIL_SIGNATURE, APRIL_START, MARCH_SIGNATURE, MARCH_START, Q1, genome_hash, prepare_window,
)
from src.adaptive_gate import (
    CONVICTION_WINDOW, MASTERY_ACCURACY, MASTERY_BLOCKS, MASTERY_CONVICTION,
    MASTERY_STEP_SETTLEMENTS, PERCENTILE_STAGES, AdaptiveGate,
)
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.evolution_context import native_direction_context
from src.evolution_lineage import MARKET
from src.evolution_multiscale import FEATURE_NAMES, M5_TACTICAL, arm_inputs, direction_scores, scale_inputs
from src.market import PROTECTED_HOLDOUT_START
from src.replay_engine import replay_contracts
from src.train import write_json


SOURCE = Path('data/adaptive-source')
RAW = Path('data/histdata_scalping')
OUTPUT = Path('reports/m5-adaptive-gate')
PERIODS = (*Q1, '202604')
GEN354_SHA256 = 'fdbba07e3400d466b8a3e5d8cd98de159808e8603b301f64d69122b5e2cf3bcb'
PROBABILITY_NAMES = ('mc_prob_up', 'mc_prob_down', 'mc_prob_tie')
VARIANTS = dict(conviction=False, elastic=True)


def archived_march(seal, reference, mean, std, paths):
    snapshot = SOURCE/'march-inputs.npz'
    if file_sha256(snapshot) != reference['march_snapshot_sha256']:
        raise ValueError('El snapshot debe coincidir con el replay canónico')
    with np.load(snapshot, allow_pickle=False) as data:
        inputs, opens, outcome, volatility = (data[k] for k in ('inputs', 'opens_ns', 'outcome', 'volatility'))
    if (sha256(inputs.tobytes()).hexdigest() != seal['training_input_sha256']
            or sha256(opens.tobytes()).hexdigest() != MARCH_SIGNATURE[2]):
        raise ValueError('Matriz o máscara de marzo distinta de la original')
    cols = [FEATURE_NAMES.index(name) for name in PROBABILITY_NAMES]
    if (np.abs(inputs[:, cols]) >= seal['normalization']['clip']).any():
        raise ValueError('Probabilidades archivadas recortadas')
    raw = inputs[:, cols] * std[cols] + mean[cols]
    probs = np.rint(raw * paths) / paths
    np.testing.assert_allclose(probs, raw, rtol=0., atol=1e-12)
    entries = pd.to_datetime(opens, utc=True, unit='ns') + MARKET.step
    rows = pd.DataFrame(probs, index=entries, columns=PROBABILITY_NAMES)
    return rows.assign(outcome=outcome, cond_volatility=volatility), inputs


def february_inputs(minutes, founder, quant, mean, std):
    previous = minutes.loc[minutes.timestamp < MARCH_START]
    data = m5_features(previous, quant, end=MARCH_START)
    start = pd.Timestamp('2026-02-01T00:00:00Z')
    blocks, _ = admission_blocks(data, founder, [('february', start, MARCH_START)])
    rows = blocks['february'].rows.tail(CONVICTION_WINDOW)
    if len(rows) != CONVICTION_WINDOW:
        raise ValueError('Faltan propuestas anteriores para inicializar la convicción')
    native = native_direction_context(previous, rows.index, end=MARCH_START)
    raw = rows.loc[:, list(M5_TACTICAL)].join(native).loc[:, list(FEATURE_NAMES)]
    return scale_inputs(raw.to_numpy(dtype=float), mean, std)


def simulate_admission(rows, scores, warmup, *, percentile=PERCENTILE_STAGES[0], elastic=True):
    gate = AdaptiveGate(percentile=percentile, elastic=elastic, warmup_scores=warmup)
    pending = deque()
    decisions = []
    for time, score, outcome in zip(rows.index, scores, rows.outcome, strict=True):
        while pending and pending[0][0] <= time:
            _, won = pending.popleft()
            gate.observe_settlement(won)
        decision = gate.decide(score)
        won = None if outcome == 0 else bool((1 if score >= 0. else -1) == outcome)
        pending.append((time + MARKET.step, won))
        decisions.append(asdict(decision))
    return pd.DataFrame(decisions, index=rows.index)


def main():
    seal, _, mean, std = sealed_gen333(SOURCE/'champions.json')
    reference = json.loads((SOURCE/'replay.json').read_text())
    extension = json.loads((SOURCE/'extension.json').read_text())
    historical = json.loads(Path('data/adaptive-static/replay.json').read_text())
    winner = next(r for r in extension['replicas'] if r['seed'] == 43)
    genome = winner['genome']
    if (winner['champion']['generation'] != 354 or genome_hash(genome) != GEN354_SHA256
            or reference['genome_sha256'] != GEN333_SHA256
            or historical['gen354_sha256'] != GEN354_SHA256 or historical['gate'] != reference['gate']
            or historical['march_snapshot_sha256'] != reference['march_snapshot_sha256']
            or extension['training_input_sha256'] != seal['training_input_sha256']
            or extension['normalization'] != seal['normalization']):
        raise ValueError('Las referencias deben pertenecer a la Gen 354/43 y al gate original')
    founder, _, quant = load_references()
    gate = reference['gate']
    risk = RiskConfig(min_confidence=gate['confidence'], min_expected_edge=gate['min_expected_edge'],
                      mc_z=gate['wilson_z'], drawdown_limit=gate['drawdown_stop_fraction'])
    march, train = archived_march(seal, reference, mean, std, gate['n_paths'])
    april, raw = prepare_window(RAW, founder, quant, PERIODS, APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    windows = dict(march=(march, train), april=(april.rows, scale_inputs(raw, mean, std)))
    rows = pd.concat((march, april.rows))
    scores = direction_scores(arm_inputs(np.vstack((train, windows['april'][1])), ARM), [genome])[0]
    minutes = load_minutes(RAW, PERIODS, end=PROTECTED_HOLDOUT_START)
    warmup = direction_scores(arm_inputs(february_inputs(minutes, founder, quant, mean, std), ARM), [genome])[0]
    adaptive = {
        name: simulate_admission(rows, scores, warmup, elastic=elastic)
        for name, elastic in VARIANTS.items()
    }
    side = pd.Series(np.where(scores >= 0., 1, -1), index=rows.index)
    reports, traces = {}, []
    for period, (window, _) in windows.items():
        settings = dict(observed_hours=reference['windows'][period]['observed_hours'], hourly_cap=gate['hourly_cap'])
        directions = side.loc[window.index].to_numpy()
        control, trace = replay_contracts(window, directions, founder.account, risk, n_paths=gate['n_paths'],
                                          max_volatility=gate['max_volatility'], **settings)
        for key in ('filters', 'all_at_expiry', 'operational'):
            if control[key] != historical['windows'][period]['gen354'][key]:
                raise ValueError(f'El control no reproduce {key} en {period}')
        reports[period] = dict(static=control)
        traces.append(trace.assign(window=period, policy='static'))
        for name, decisions in adaptive.items():
            state = decisions.loc[window.index]
            result, log = replay_contracts(window, directions, founder.account, risk,
                                           candidate_reasons=state.reason.to_numpy(), **settings)
            result['elasticity'] = dict(
                mastery_evidence=int(state.mastery_evidence.sum()),
                proposals_over_conviction_bar=int((state.mean_conviction > MASTERY_CONVICTION).sum()),
                max_mean_conviction=float(state.mean_conviction.max()),
                minimum_percentile=float(state.percentile.min()),
                fully_open=int((state.percentile == 0.).sum()),
            )
            reports[period][name] = result
            traces.append(pd.concat((log, state.reset_index(drop=True)), axis=1).assign(window=period, policy=name))
    report = dict(
        genome_sha256=GEN354_SHA256, warmup_proposals=len(warmup), windows=reports,
        adaptive_settings=dict(conviction_window=CONVICTION_WINDOW, percentile_stages=PERCENTILE_STAGES,
                               mastery_conviction=MASTERY_CONVICTION, mastery_accuracy=MASTERY_ACCURACY,
                               mastery_blocks=MASTERY_BLOCKS,
                               maturity_step_settlements=MASTERY_STEP_SETTLEMENTS),
        hourly_cap=gate['hourly_cap'], drawdown_stop_fraction=gate['drawdown_stop_fraction'],
        protected_holdout_evaluated=False,
    )
    write_json(OUTPUT/'adaptive-gate.json', report)
    pd.concat(traces, ignore_index=True).to_csv(OUTPUT/'admissions.csv', index=False)
    print('ADAPTIVE_GATE ' + json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
