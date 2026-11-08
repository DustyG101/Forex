"""Evoluciona la densidad del gate puro en marzo y diagnostica abril tras el sello."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_adaptive_gate import (
    GEN354_SHA256, RAW, SOURCE, archived_march, february_inputs, simulate_admission,
)
from scripts.run_m5_direction_ablation import load_minutes
from scripts.run_m5_direction_benchmark import load_references
from scripts.run_m5_gen333 import ARM, GEN333_SHA256, sealed_gen333
from scripts.run_m5_multiscale import APRIL_SIGNATURE, APRIL_START, MARCH_START, Q1, genome_hash, prepare_window
from src.adaptive_gate import CONVICTION_WINDOW, MASTERY_ACCURACY, rolling_conviction_admissions
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.evolution_engine import MAX_TRADES_HOUR, MIN_PROFIT_FACTOR, MIN_TRADES_HOUR
from src.evolution_multiscale import POPULATION, SEEDS, arm_inputs, direction_scores, evolve_population, scale_inputs
from src.market import PROTECTED_HOLDOUT_START
from src.replay_engine import apply_hourly_cap, replay_contracts
from src.train import write_json


OUTPUT = Path('reports/m5-conviction-evolution')
PERCENTILES = (.85, .80, .75)
FIRST, LAST = 355, 375


def label(percentile):
    return f'p{percentile * 100:.0f}'


def setup():
    seal, _, mean, std = sealed_gen333(SOURCE/'champions.json')
    reference = json.loads((SOURCE/'replay.json').read_text())
    extension = json.loads((SOURCE/'extension.json').read_text())
    parent = next(r for r in extension['replicas'] if r['seed'] == 43)
    if (reference['genome_sha256'] != GEN333_SHA256 or parent['champion']['generation'] != 354
            or genome_hash(parent['genome']) != GEN354_SHA256
            or extension['training_input_sha256'] != seal['training_input_sha256']
            or extension['normalization'] != seal['normalization']
            or extension['march_snapshot_sha256'] != reference['march_snapshot_sha256']):
        raise ValueError('Se requiere el genoma Gen 354/43 y el snapshot canónico')
    founder, _, quant = load_references()
    gate = reference['gate']
    if gate['n_paths'] != quant.n_paths or gate['drawdown_stop_fraction'] != founder.drawdown_limit:
        raise ValueError('La capa de riesgo debe coincidir con el replay de referencia')
    march, inputs = archived_march(seal, reference, mean, std, gate['n_paths'])
    risk = RiskConfig(drawdown_limit=gate['drawdown_stop_fraction'])
    return reference, founder, quant, risk, parent['genome'], march, inputs, mean, std


def measured(result, account, admissions):
    pnl, rate = result['pnl_usd'], result['contracts_per_observed_hour']
    wins, losses = result['wins'], result['losses']
    margin = account.payout * wins - MIN_PROFIT_FACTOR * losses
    pf_pass = margin > 1e-12
    return dict(
        pf_pass=pf_pass, density_at_pf=min(rate, MIN_TRADES_HOUR) if pf_pass else 0.,
        meets_targets=pf_pass and MIN_TRADES_HOUR <= rate <= MAX_TRADES_HOUR,
        score=margin * account.stake, pnl_usd=pnl, trades=result['trades'], frequency=rate,
        return_per_admission=pnl / (account.stake * admissions),
    )


def ranking(fitness):
    return (fitness['pf_pass'], fitness['density_at_pf'], fitness['score'], fitness['return_per_admission'])


def candidate_reasons(rows, scores, side, percentile):
    # La maestría por tres bloques implica primero el acierto agregado de 500 liquidadas.
    outcomes = rows.outcome.to_numpy()
    resolved = outcomes != 0
    correct = np.r_[0, np.cumsum(side[resolved] == outcomes[resolved])]
    if ((correct[CONVICTION_WINDOW:] - correct[:-CONVICTION_WINDOW])
            >= CONVICTION_WINDOW * MASTERY_ACCURACY).any():
        return simulate_admission(rows, scores[CONVICTION_WINDOW:], scores[:CONVICTION_WINDOW],
                                  percentile=percentile).reason.to_numpy()
    admitted = rolling_conviction_admissions(scores, percentile)[CONVICTION_WINDOW:]
    return np.where(admitted, 'accepted', 'conviction')


def evolve_one(rows, inputs, parent, account, risk, settings, percentile, seed):
    counts = dict(pf_pass=0, meets_targets=0)

    def evaluate(population):
        records = []
        for scores in direction_scores(inputs, population):
            side = np.where(scores[CONVICTION_WINDOW:] >= 0., 1, -1)
            reasons = candidate_reasons(rows, scores, side, percentile)
            result, _ = replay_contracts(rows, side, account, risk, candidate_reasons=reasons, **settings)
            records.append(measured(result['operational'], account, len(rows)))
        for key in counts:
            counts[key] += sum(record[key] for record in records)
        return {key: np.array([r[key] for r in records]) for key in records[0]}

    result = evolve_population(evaluate, np.asarray(parent), ARM, seed, first_generation=FIRST,
                               last_generation=LAST, ranking=ranking)
    result.update(percentile=percentile, genome_sha256=genome_hash(result['genome']), candidates_passing=counts)
    print(f'Completed {label(percentile)} seed {seed} with champion {result["champion"]["generation"]}', flush=True)
    return result


def evolve():
    path = OUTPUT/'evolution.json'
    if path.exists():
        raise ValueError('Los campeones de marzo ya están sellados')
    reference, founder, quant, risk, parent, march, inputs, mean, std = setup()
    past = load_minutes(RAW, Q1[:-1], end=MARCH_START)
    warmup = february_inputs(past, founder, quant, mean, std)
    settings = dict(observed_hours=reference['windows']['march']['observed_hours'],
                    hourly_cap=reference['gate']['hourly_cap'])
    inputs = arm_inputs(np.vstack((warmup, inputs)), ARM)
    with ThreadPoolExecutor(max_workers=len(PERCENTILES)) as executor:
        futures = [
            executor.submit(evolve_one, march, inputs, parent, founder.account, risk, settings, percentile, seed)
            for seed in SEEDS for percentile in PERCENTILES
        ]
        replicas = [future.result() for future in futures]
    winners = {
        label(p): max((r for r in replicas if r['percentile'] == p), key=lambda r: ranking(r['champion']))['seed']
        for p in PERCENTILES
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    warmup_path = OUTPUT/'warmup.npy'
    np.save(warmup_path, warmup, allow_pickle=False)
    report = dict(
        first_generation=FIRST, last_generation=LAST, population=POPULATION, seeds=SEEDS,
        evaluations=sum(r['evaluations'] for r in replicas), percentiles=PERCENTILES, winners=winners,
        parent_sha256=GEN354_SHA256, warmup_sha256=file_sha256(warmup_path), replicas=replicas,
        march_snapshot_sha256=reference['march_snapshot_sha256'], protected_holdout_evaluated=False,
    )
    write_json(path, report)
    print(f'Sealed March champions SHA256 {file_sha256(path)}', flush=True)


def replay():
    reference, founder, quant, risk, parent, march, inputs, mean, std = setup()
    path = OUTPUT/'evolution.json'
    training = json.loads(path.read_text())
    warmup_path = OUTPUT/'warmup.npy'
    if (training['parent_sha256'] != GEN354_SHA256
            or training['march_snapshot_sha256'] != reference['march_snapshot_sha256']
            or training['warmup_sha256'] != file_sha256(warmup_path)):
        raise ValueError('Los campeones deben pertenecer al snapshot sellado')
    warmup = np.load(warmup_path, allow_pickle=False)
    champions = {'gen354': dict(genome=parent, generation=354, seed=43)}
    for percentile in PERCENTILES:
        key = label(percentile)
        item = next(r for r in training['replicas'] if r['percentile'] == percentile
                    and r['seed'] == training['winners'][key])
        if genome_hash(item['genome']) != item['genome_sha256']:
            raise ValueError('Hash del campeón incompatible')
        champions[key] = dict(genome=item['genome'], generation=item['champion']['generation'],
                              seed=item['seed'], expected=item['champion'])
    april, raw = prepare_window(RAW, founder, quant, (*Q1, '202604'),
                                APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    windows = dict(march=march, april=april.rows)
    rows = pd.concat((march, april.rows))
    inputs = arm_inputs(np.vstack((warmup, inputs, scale_inputs(raw, mean, std))), ARM)
    hourly_cap = reference['gate']['hourly_cap']
    comparisons = {}
    for key, model in champions.items():
        scores = direction_scores(inputs, [model['genome']])[0]
        direction = pd.Series(np.where(scores[CONVICTION_WINDOW:] >= 0., 1, -1), index=rows.index)
        cases = {}
        for percentile in PERCENTILES:
            state = simulate_admission(rows, scores[CONVICTION_WINDOW:], scores[:CONVICTION_WINDOW],
                                       percentile=percentile)
            result = {}
            for period, window in windows.items():
                decisions = state.loc[window.index]
                replayed, _ = replay_contracts(
                    window, direction.loc[window.index].to_numpy(), founder.account, risk, hourly_cap=hourly_cap,
                    observed_hours=reference['windows'][period]['observed_hours'],
                    candidate_reasons=decisions.reason.to_numpy(),
                )
                operational = replayed['operational']
                target = measured(operational, founder.account, len(window))
                if key == label(percentile) and period == 'march':
                    for field in ('pnl_usd', 'trades', 'frequency', 'pf_pass', 'meets_targets'):
                        if target[field] != model['expected'][field]:
                            raise ValueError(f'El replay debe reproducir la selección de {key} en marzo')
                result[period] = dict(
                    **operational, pf_pass=target['pf_pass'], meets_targets=target['meets_targets'],
                    adaptive=dict(
                        distinct_cutoffs=int(decisions.cutoff.nunique()),
                        min_cutoff=float(decisions.cutoff.min()), max_cutoff=float(decisions.cutoff.max()),
                        percentile_stages=sorted(decisions.percentile.unique().tolist(), reverse=True),
                        fully_open=int((decisions.percentile == 0.).sum()),
                    ),
                )
            cases[label(percentile)] = result
        comparisons[key] = dict(generation=model['generation'], seed=model['seed'], cases=cases)
    capacity = {}
    for period, window in windows.items():
        accepted = apply_hourly_cap(window.index, np.repeat('accepted', len(window)), hourly_cap) == 'accepted'
        capacity[period] = dict(contracts=int(accepted.sum()),
                                per_hour=float(accepted.sum() / reference['windows'][period]['observed_hours']))
    report = dict(comparisons=comparisons, capacity=capacity, evolution_sha256=file_sha256(path),
                  protected_holdout_evaluated=False)
    write_json(OUTPUT/'replay.json', report)
    print('CONVICTION_EVOLUTION ' + json.dumps({
        **report, 'evaluations': training['evaluations'],
        'replicas': [{k: r[k] for k in ('seed', 'percentile', 'champion', 'candidates_passing')}
                     for r in training['replicas']],
    }, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evolve', 'replay'))
    if parser.parse_args().phase == 'evolve':
        evolve()
    else:
        replay()
