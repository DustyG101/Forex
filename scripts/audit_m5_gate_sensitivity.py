"""Sensibilidad retrospectiva del gate con la dirección congelada de Gen 333."""

from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_benchmark import load_references
from scripts.run_m5_gen333 import ARM, HOURLY_CAP, sealed_gen333
from scripts.run_m5_multiscale import APRIL_SIGNATURE, APRIL_START, MARCH_SIGNATURE, Q1, prepare_window
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.evolution_direction import unit_payoff
from src.evolution_lineage import MARKET
from src.evolution_multiscale import FEATURE_NAMES, arm_inputs, directions, scale_inputs
from src.market import PROTECTED_HOLDOUT_START
from src.replay_engine import payoff_summary, replay_contracts
from src.train import write_json


ROOT = Path('data/gate-sensitivity')
OUTPUT = Path('reports/gate-sensitivity')
BASE = (.57, 2.24, 'p67')
PROBABILITY_NAMES = ('mc_prob_up', 'mc_prob_down', 'mc_prob_tie')


def archived_march(seal, reference, mean, std, paths):
    snapshot = ROOT/'march-inputs.npz'
    if file_sha256(snapshot) != reference['march_snapshot_sha256']:
        raise ValueError('El snapshot debe coincidir con el intento de referencia')
    with np.load(snapshot, allow_pickle=False) as data:
        inputs, opens, outcome, sigma = (data[k] for k in ('inputs', 'opens_ns', 'outcome', 'volatility'))
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
    return rows.assign(outcome=outcome, cond_volatility=sigma), inputs


def assess(rows, inputs, genome, account, risk, paths, hours, ceilings, reference):
    side = directions(arm_inputs(inputs, ARM), [genome])[0]
    payoff = unit_payoff(side, rows.outcome.to_numpy(), account) * account.stake
    settings = dict(observed_hours=hours, n_paths=paths, hourly_cap=HOURLY_CAP)
    baseline, trace = replay_contracts(rows, side, account, risk, max_volatility=ceilings['p67'], **settings)
    for key in ('trades', 'pnl_usd', 'profit_factor'):
        np.testing.assert_allclose(baseline['operational'][key], reference['operational'][key], rtol=0., atol=1e-10)
        np.testing.assert_allclose(baseline['all_at_expiry'][key], reference['all_at_expiry'][key], rtol=0., atol=1e-10)
    if baseline['filters'] != reference['filters']:
        raise ValueError('Los descartes del gate de referencia cambiaron')
    reasons = trace.gate_reason.to_numpy()
    accepted = reasons == 'accepted'
    by_reason = {name: payoff_summary(payoff[reasons == name], account, hours)
                 for name in ('accepted', 'confidence', 'expected_edge', 'volatility', 'hourly_cap')}
    probability = np.where(side > 0, rows.mc_prob_up, rows.mc_prob_down)
    bins = pd.cut(probability, [-np.inf, .50, .54, .57, .60, .63, np.inf], right=True)
    by_probability = {str(band): payoff_summary(payoff[bins == band], account, hours) for band in bins.categories}
    scenarios = []
    for threshold, z, cap in product((.54, .57, .60, .63), (0., 1.96, 2.24), ceilings):
        result, alternative = replay_contracts(
            rows, side, account, replace(risk, min_confidence=threshold, mc_z=z),
            max_volatility=ceilings[cap], **settings,
        )
        candidate = alternative.gate_reason.to_numpy() == 'accepted'
        scenarios.append(dict(
            threshold=threshold, wilson_z=z, volatility=cap, **result['operational'],
            rejection_pct=result['filters']['rejected_pct'],
            gate_only_pnl=result['gate_only_at_expiry']['pnl_usd'],
            added=payoff_summary(payoff[candidate & ~accepted], account, hours),
            removed=payoff_summary(payoff[accepted & ~candidate], account, hours),
        ))
    return dict(baseline=baseline, by_reason=by_reason, by_probability=by_probability, scenarios=scenarios)


def main():
    seal, model, mean, std = sealed_gen333(ROOT/'champions.json')
    reference = json.loads((ROOT/'replay.json').read_text())
    founder, _, quant = load_references()
    risk = RiskConfig(min_confidence=BASE[0], min_expected_edge=.01, mc_z=BASE[1],
                      drawdown_limit=founder.drawdown_limit)
    march, train = archived_march(seal, reference, mean, std, quant.n_paths)
    ceilings = dict(p50=float(march.cond_volatility.quantile(.50)), p67=seal['garch_p67'],
                    p90=float(march.cond_volatility.quantile(.90)), open=np.finfo(float).max)
    block, raw = prepare_window(Path('data/histdata_scalping'), founder, quant, (*Q1, '202604'),
                                APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    windows = {'march': (march, train, reference['windows']['march']['observed_hours']),
               'april': (block.rows, scale_inputs(raw, mean, std), block.observed_hours)}
    reports = {name: assess(rows, inputs, model['genome'], founder.account, risk, quant.n_paths,
                            hours, ceilings, reference['windows'][name])
               for name, (rows, inputs, hours) in windows.items()}
    write_json(OUTPUT/'sensitivity.json', dict(windows=reports, volatility_caps=ceilings,
                                              original_march_sha256=seal['training_input_sha256']))
    pd.json_normalize([dict(window=name, **s) for name, r in reports.items() for s in r['scenarios']]).to_csv(
        OUTPUT/'scenarios.csv', index=False,
    )
    for name, report in reports.items():
        fields = ('threshold', 'wilson_z', 'volatility', 'trades', 'pnl_usd', 'profit_factor',
                  'return_on_staked_pct', 'contracts_per_observed_hour', 'account_stop', 'max_drawdown_pct')
        summary = dict(report, scenarios=[{**{key: s[key] for key in fields},
                                           'added': [s['added']['trades'], s['added']['pnl_usd']],
                                           'removed': [s['removed']['trades'], s['removed']['pnl_usd']]}
                                          for s in report['scenarios']])
        print(f'GATE_SENSITIVITY_{name.upper()} ' + json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
