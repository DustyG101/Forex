"""Compara Gen 333 y 354 con el gate histórico y la matriz canónica."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_m5_gate_sensitivity import ROOT, archived_march
from scripts.run_m5_direction_benchmark import load_references
from scripts.run_m5_gen333 import ARM, GEN333_SHA256, sealed_gen333
from scripts.run_m5_multiscale import APRIL_SIGNATURE, APRIL_START, Q1, genome_hash, prepare_window
from src.decision import RiskConfig
from src.evolution_multiscale import arm_inputs, directions, scale_inputs
from src.market import PROTECTED_HOLDOUT_START
from src.replay_engine import replay_contracts
from src.train import write_json


OUTPUT = Path('reports/gen354-reference-gate')
GEN354_SHA256 = 'fdbba07e3400d466b8a3e5d8cd98de159808e8603b301f64d69122b5e2cf3bcb'


def main():
    seal, gen333, mean, std = sealed_gen333(ROOT/'champions.json')
    reference = json.loads((ROOT/'replay.json').read_text())
    extension = json.loads((ROOT/'extension.json').read_text())
    gen354 = next(r for r in extension['replicas'] if r['seed'] == 43)
    if (reference['genome_sha256'] != GEN333_SHA256 or reference['early_exits']['mode'] != 'unavailable'
            or extension['founder_genome_sha256'] != GEN333_SHA256 or extension['arm'] != ARM
            or extension['training_input_sha256'] != seal['training_input_sha256']
            or extension['normalization'] != seal['normalization']
            or extension['march_snapshot_sha256'] != reference['march_snapshot_sha256']
            or gen354['champion']['generation'] != 354 or gen354['genome_sha256'] != GEN354_SHA256
            or genome_hash(gen354['genome']) != GEN354_SHA256):
        raise ValueError('La extensión debe ser la Gen 354/43 del snapshot original')
    founder, _, quant = load_references()
    gate = reference['gate']
    if gate['n_paths'] != quant.n_paths or gate['drawdown_stop_fraction'] != founder.drawdown_limit:
        raise ValueError('El gate debe usar el contrato de referencia')
    risk = RiskConfig(min_confidence=gate['confidence'], min_expected_edge=gate['min_expected_edge'],
                      mc_z=gate['wilson_z'], drawdown_limit=gate['drawdown_stop_fraction'])
    march, train = archived_march(seal, reference, mean, std, quant.n_paths)
    april, raw = prepare_window(Path('data/histdata_scalping'), founder, quant, (*Q1, '202604'),
                                APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    windows = dict(march=(march, train), april=(april.rows, scale_inputs(raw, mean, std)))
    reports, traces = {}, []
    for name, (rows, inputs) in windows.items():
        expected = reference['windows'][name]
        results, orders = {}, {}
        for label, model in (('gen333', gen333), ('gen354', gen354)):
            sides = directions(arm_inputs(inputs, ARM), [model['genome']])[0]
            results[label], orders[label] = replay_contracts(
                rows, sides, founder.account, risk, observed_hours=expected['observed_hours'],
                n_paths=gate['n_paths'], max_volatility=gate['max_volatility'], hourly_cap=gate['hourly_cap'],
            )
        for key in ('filters', 'all_at_expiry', 'operational'):
            if results['gen333'][key] != expected[key]:
                raise ValueError(f'El control Gen 333 no reprodujo {key} en {name}')
        if name == 'march':
            np.testing.assert_allclose(results['gen354']['all_at_expiry']['pnl_usd'],
                                       gen354['march']['pnl'], rtol=0., atol=1e-9)
        a, b = orders['gen333'].pnl_usd, orders['gen354'].pnl_usd
        both = a.notna() & b.notna()
        only333 = a.notna() & b.isna()
        only354 = b.notna() & a.isna()
        results['execution_comparison'] = dict(
            shared=int(both.sum()), shared_pnl_change_usd=float((b[both] - a[both]).sum()),
            only_gen333=int(only333.sum()), only_gen333_pnl_usd=float(a[only333].sum()),
            only_gen354=int(only354.sum()), only_gen354_pnl_usd=float(b[only354].sum()),
        )
        reports[name] = results
        traces.extend(trace.assign(window=name, model=label) for label, trace in orders.items())
    report = dict(gate=gate, windows=reports, gen354_sha256=GEN354_SHA256,
                  march_snapshot_sha256=reference['march_snapshot_sha256'],
                  protected_holdout_evaluated=False)
    write_json(OUTPUT/'replay.json', report)
    pd.concat(traces, ignore_index=True).to_csv(OUTPUT/'admissions.csv', index=False)
    print('GEN354_GATE ' + json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
