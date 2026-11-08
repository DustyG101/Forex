"""Auditoría de abril con los genomas sellados de la PR 28."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import load_minutes
from scripts.run_m5_direction_benchmark import load_references
from scripts.run_m5_multiscale import APRIL_SIGNATURE, APRIL_START, Q1, prepare_window, read_seal
from src.artifacts import file_sha256
from src.evolution_context import H1_STRUCTURAL, M15_TACTICAL
from src.evolution_direction import unit_payoff
from src.evolution_lineage import MARKET
from src.evolution_multiscale import (
    FEATURE_NAMES, HIDDEN, arm_inputs, directional_metrics, directions, scale_inputs,
)
from src.market import MarketConfig, PROTECTED_HOLDOUT_START
from src.resampling import aggregate
from src.train import write_json


ROOT = Path('data/m5-april-audit')
RAW = Path('data/histdata_scalping')
OUTPUT = Path('reports/m5-april-audit')
PERIODS = (*Q1, '202604')
SEAL_SHA256 = 'e201a87c12131af0441b03855175e43afccc4f79252587efdfadea7e8010612c'
TZ = 'America/Lima'


def layer(genome):
    g = np.asarray(genome, dtype=float)
    return g[:-2 * HIDDEN - 1].reshape(len(FEATURE_NAMES), HIDDEN), g[-2 * HIDDEN - 1:-HIDDEN - 1], g[-HIDDEN - 1:-1], g[-1]


def predict_scores(inputs, genome):
    w, bias, output, intercept = layer(genome)
    return np.tanh(inputs @ w + bias) @ output + intercept


def sign(scores):
    return np.where(scores >= 0., 1, -1)


def block_effect(inputs, genome, full_scores, block, outcome, mc, account):
    indices = [FEATURE_NAMES.index(feature) for feature in block]
    weights, _, output, _ = layer(genome)
    neutral = inputs.copy()
    neutral[:, indices] = 0.
    scores = predict_scores(neutral, genome)
    full_side, neutral_side = sign(full_scores), sign(scores)
    full_payoff = unit_payoff(full_side, outcome, account)
    neutral_payoff = unit_payoff(neutral_side, outcome, account)
    change = full_payoff - neutral_payoff
    summary = dict(
        weights_by_feature={f: weights[i].tolist() for f, i in zip(block, indices, strict=True)},
        input_l2=float(np.linalg.norm(weights[indices])),
        input_rms=float(np.sqrt(np.mean(weights[indices] ** 2))),
        output_weighted_l2=float(np.linalg.norm(weights[indices] * output)),
        mean_absolute_score_change=float(np.abs(full_scores - scores).mean()),
        score_change_p95=float(np.quantile(np.abs(full_scores - scores), .95)),
        flipped=int(np.count_nonzero(full_side != neutral_side)),
        wins_added=int((change > 0.).sum()), wins_removed=int((change < 0.).sum()),
        refund_flips=int(np.count_nonzero((full_side != neutral_side) & (outcome == 0))),
        full_minus_neutral_usd=float(change.sum() * account.stake),
        neutral=directional_metrics(neutral_side, outcome, mc, account),
    )
    return scores, neutral_payoff, summary


def native_reversals(entries, raw_inputs):
    minutes = load_minutes(RAW, PERIODS, end=PROTECTED_HOLDOUT_START)
    market = MarketConfig('M15')
    bars = aggregate(minutes, 1, 'M15', now=PROTECTED_HOLDOUT_START)
    continuous = bars.timestamp.diff().eq(market.step)
    returns = np.log(bars.close).diff().where(continuous)
    latest = pd.DatetimeIndex(bars.timestamp + market.step).get_indexer(
        entries, method='pad', tolerance=market.step - MARKET.step,
    )
    ready = latest >= 0
    current, prior = np.full(len(entries), np.nan), np.full(len(entries), np.nan)
    current[ready] = returns.to_numpy()[latest[ready]]
    prior[ready] = returns.shift().to_numpy()[latest[ready]]
    slope = raw_inputs[:, FEATURE_NAMES.index('h1_ema_slope')]
    atr = raw_inputs[:, FEATURE_NAMES.index('m15_atr_fraction')]
    defined = np.isfinite(current) & np.isfinite(slope) & (current != 0.) & (slope != 0.)
    opposed = defined & (np.sign(current) != np.sign(slope))
    sharp = (current * prior < 0.) & (np.abs(current) >= atr)
    return current, prior, slope, atr, defined, opposed, sharp


def totals(frame):
    return dict(
        admissions=len(frame), m15_usd=round(float(frame.m15_usd.sum()), 6),
        h1_usd=round(float(frame.h1_usd.sum()), 6),
        h1_minus_m15_usd=round(float(frame.h1_minus_m15_usd.sum()), 6),
        h1_neutral_usd=round(float(frame.h1_neutral_usd.sum()), 6),
        direct_h1_usd=round(float(frame.direct_h1_usd.sum()), 6),
        direction_disagreements=int((frame.side_h1 != frame.side_m15).sum()),
        opposite=int(frame.h1_m15_opposed.sum()), sharp=int(frame.sharp_reversal.sum()),
    )


def main():
    seal_path = ROOT/'champions.json'
    if file_sha256(seal_path) != SEAL_SHA256:
        raise ValueError('El sello no corresponde a la corrida 42507157819')
    seal, mean, std = read_seal(seal_path)
    saved = json.loads((ROOT/'april.json').read_text())
    if saved['seal_sha256'] != SEAL_SHA256 or saved['admissions_sha256'] != APRIL_SIGNATURE[2]:
        raise ValueError('El diagnóstico de referencia no coincide con el sello')
    founder, _, quant = load_references()
    window, raw = prepare_window(RAW, founder, quant, PERIODS, APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    entries = window.rows.index
    outcome, mc = window.rows.outcome.to_numpy(), window.rows.mc_side.to_numpy()
    account = founder.account
    inputs = scale_inputs(raw, mean, std)
    models = {c['arm']: c for c in seal['champions'] if c['arm'] in ('m15', 'm15_h1')}
    scores = {arm: predict_scores(arm_inputs(inputs, arm), model['genome']) for arm, model in models.items()}
    for arm, model in models.items():
        side = sign(scores[arm])
        np.testing.assert_array_equal(side, directions(arm_inputs(inputs, arm), [model['genome']])[0])
        high = window.rows.cond_volatility.to_numpy() > seal['garch_p67']
        for regime, mask in (('all', np.ones(len(side), dtype=bool)), ('high', high), ('moderate', ~high)):
            metrics = directional_metrics(side[mask], outcome[mask], mc[mask], account)
            expected = saved['policies'][arm][regime]
            for key in ('admissions', 'wins', 'losses', 'refunds', 'pnl', 'disagreements_vs_mc'):
                if not np.isclose(metrics[key], expected[key], rtol=0., atol=1e-9):
                    raise ValueError(f'El replay difiere del original en {arm}, {regime}, {key}')

    genome = models['m15_h1']['genome']
    h1_zero_score, h1_zero_payoff, h1_effect = block_effect(
        inputs, genome, scores['m15_h1'], H1_STRUCTURAL, outcome, mc, account,
    )
    m15_zero_score, _, m15_effect = block_effect(
        inputs, genome, scores['m15_h1'], M15_TACTICAL, outcome, mc, account,
    )
    current, prior, slope, atr, defined, opposed, sharp = native_reversals(entries, raw)
    m15_side, h1_side = sign(scores['m15']), sign(scores['m15_h1'])
    m15_pnl = unit_payoff(m15_side, outcome, account) * account.stake
    h1_pnl = unit_payoff(h1_side, outcome, account) * account.stake
    h1_zero_pnl = h1_zero_payoff * account.stake
    local = entries.tz_convert(TZ)
    trace = pd.DataFrame(dict(
        decision_utc=entries, decision_lima=local, expiry_utc=entries + MARKET.step,
        outcome=outcome, side_mc=mc, side_m15=m15_side, side_h1=h1_side,
        side_h1_neutral=sign(h1_zero_score), m15_usd=m15_pnl, h1_usd=h1_pnl,
        h1_minus_m15_usd=h1_pnl-m15_pnl, h1_neutral_usd=h1_zero_pnl, direct_h1_usd=h1_pnl-h1_zero_pnl,
        score_m15=scores['m15'], score_h1=scores['m15_h1'], score_h1_neutral=h1_zero_score,
        score_m15_neutral=m15_zero_score, last_m15_log_return=current, prior_m15_log_return=prior,
        h1_ema_slope=slope, m15_atr_relative=atr, h1_m15_defined=defined, h1_m15_opposed=opposed,
        sharp_reversal=sharp, h1_prediction_matches_slope=(h1_side == np.sign(slope)),
    ))
    by_day = [dict(start=t.isoformat(), **totals(g)) for t, g in trace.groupby(local.floor('D'))]
    by_hour = [dict(start=t.isoformat(), **totals(g)) for t, g in trace.groupby(local.floor('h'))]
    conditions = {
        'opposed': opposed, 'aligned': defined & ~opposed, 'missing_or_neutral': ~defined,
        'sharp': sharp, 'other': ~sharp, 'sharp_and_opposed': sharp & opposed,
    }
    report = dict(
        seal_sha256=SEAL_SHA256, admission_sha256=window.mask_sha256, timezone=TZ,
        last_expiry=(entries[-1] + MARKET.step).isoformat(), replay_matches_original=True,
        models={arm: dict(seed=c['seed'], generation=c['selection']['generation'],
                          genome_sha256=c['genome_sha256']) for arm, c in models.items()},
        all=totals(trace), by_day=by_day, by_hour=by_hour,
        conditions={key: totals(trace.loc[mask]) for key, mask in conditions.items()},
        weights=dict(h1=h1_effect, m15=m15_effect, output=layer(genome)[2].tolist()),
    )
    write_json(OUTPUT/'april-audit.json', report)
    trace.to_csv(OUTPUT/'april-admissions.csv', index=False)
    compact = {key: value for key, value in report.items() if key != 'by_hour'}
    compact['worst_hours'] = sorted(by_hour, key=lambda row: row['h1_minus_m15_usd'])[:8]
    print('APRIL_AUDIT ' + json.dumps(compact, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
