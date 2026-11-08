"""One-off comparison of frozen M5 genomes and the repository's MC entry rule."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scripts.acquire_histdata import acquire
from scripts.build_histdata_dataset import read_native_minutes
from src.decision import RiskConfig, mc_probability_lower
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_context import build_observations
from src.evolution_engine import M5Simulation, assess_fitness
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.market import PROTECTED_HOLDOUT_START
from src.quant_engine import FEATURES, QuantConfig, enrich_features
from src.resampling import aggregate
from src.trading_profiles import resolve_profile
from src.train import chronological_market_split

ROOT = Path('data/mc-comparison')
RAW = Path('data/histdata_scalping')
PRESET = resolve_profile('scalping', 'M5')
GENOMES = {239: ROOT/'gen239/lineage.zip', 268: ROOT/'gen268/lineage.zip'}
SOURCES = json.loads(Path('datasets/eurusd/manifest.json').read_text())['sources']


def read_manifest(bundle):
    with ZipFile(bundle) as archive:
        return json.loads(archive.read('manifest.json'))


def prepared_data(periods, quant):
    entries = [entry for entry in SOURCES if entry['period'] in periods]
    if {entry['period'] for entry in entries} != set(periods):
        raise ValueError('Missing pinned source')
    for entry in entries:
        period = entry['period']
        if not (RAW / entry['zip_file']).is_file():
            acquire(RAW, int(period[:4]), int(period[4:]) if len(period) > 4 else None)
    minutes, _ = read_native_minutes(RAW, entries)
    bars = aggregate(minutes, 1, 'M5')
    enriched = enrich_features(bars, quant, timeframe='M5', expiry_candles=1)
    return prepare_market_data(enriched.dropna(subset=FEATURES).reset_index(drop=True), market=MARKET)


def summarize(sim, genome):
    metrics = sim.evaluate(genome)
    keys = ('total_trades', 'win_rate_excluding_refunds', 'profit_factor',
            'trades_per_observed_hour', 'max_drawdown_pct', 'net_profit', 'risk_stop',
            'signals_after_activation', 'signals_unfilled_due_to_stop')
    return {key: metrics[key] for key in keys} | {'checks': assess_fitness(metrics, payout=sim.account.payout)['checks']}


def compare(window, context, founders, quant):
    reference = founders[239]
    normalizer = reference.normalizer
    width = len(MARKET_FEATURES)
    market_normalizer = {k: normalizer[k][:width] for k in ('features', 'mean', 'std')}
    market_normalizer['clip'] = normalizer['clip']
    context = context.loc[window.timestamp].copy()
    values = (context.to_numpy() - np.asarray(normalizer['mean'][width:])) / np.asarray(normalizer['std'][width:])
    context.iloc[:, :] = np.clip(values, -normalizer['clip'], normalizer['clip'])
    env = BinaryOptionsEnv(window, market=MARKET, prepared=True, account=reference.account,
                           risk=RiskConfig(enabled=False, drawdown_limit=reference.drawdown_limit),
                           normalizer=market_normalizer)
    sim = M5Simulation(env, context)
    results = {f'Gen {gen}': summarize(sim, founder.genome) for gen, founder in founders.items()}
    mc = M5Simulation(env)
    up, down, _ = env._probabilities[mc.entries].T
    confidence = np.maximum(up, down)
    lower = mc_probability_lower(confidence, quant.n_paths, PRESET['FOREX_MC_Z'])
    accepted = ((confidence > PRESET['FOREX_MIN_CONFIDENCE'])
                & ((reference.account.payout + 1.) * lower - 1. > PRESET['FOREX_MIN_EXPECTED_EDGE']))
    mc.features = accepted.astype(np.float32)[:, None]
    results['Monte Carlo'] = summarize(mc, np.array([1., -.5, 0., 0., 0.]))
    return dict(start=window.timestamp.iloc[0].isoformat(), end=window.timestamp.iloc[-1].isoformat(),
                observed_hours=sim.observed_hours, policies=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('period', choices=['historical', 'fresh'])
    args = parser.parse_args()
    manifests = {gen: read_manifest(bundle) for gen, bundle in GENOMES.items()}
    assert all(gen == m['selection']['generation'] for gen, m in manifests.items())
    assert manifests[239]['quant_config'] == manifests[268]['quant_config']
    quant = QuantConfig.from_metadata(manifests[239]['quant_config'])
    historic = args.period == 'historical'
    periods = ['2025'] if historic else ['202601', '202602', '202603']
    data = prepared_data(periods, quant)
    if historic:
        splits = chronological_market_split(data, MARKET)
        validation = splits['validation']
        cut = 3 * len(validation) // 4
        train_end = splits['train'].timestamp.iloc[-1]
        search_end = validation.timestamp.iloc[cut-1]
        windows = {'Confirmación histórica': validation.iloc[cut+1:]}
        expected = manifests[239]['splits']['confirmation']
        assert windows['Confirmación histórica'].timestamp.iloc[0] == pd.Timestamp(expected['start'])
        assert windows['Confirmación histórica'].timestamp.iloc[-1] == pd.Timestamp(expected['end'])
    else:
        train_end, search_end = (pd.Timestamp(manifests[239]['splits'][name]['end'])
                                 for name in ('train', 'search'))
        data = data.loc[data.timestamp + MARKET.step <= pd.Timestamp('2026-04-01T00:00:00Z')]
        windows = {'Enero–marzo continuo': data}
        windows.update({label: data.loc[data.timestamp.dt.month.eq(month)]
                        for month, label in enumerate(('Enero tras warmup', 'Febrero', 'Marzo'), start=1)})

    assert all(window.timestamp.iloc[-1] + MARKET.step <= PROTECTED_HOLDOUT_START for window in windows.values())
    context = build_observations(data.loc[data.timestamp <= max(w.timestamp.iloc[-1] for w in windows.values())])
    founders = {gen: load_founder(bundle, None, train_end, search_end) for gen, bundle in GENOMES.items()}
    assert founders[239].normalizer == founders[268].normalizer
    assert founders[239].account == founders[268].account
    assert founders[239].drawdown_limit == founders[268].drawdown_limit
    comparison = {name: compare(window, context, founders, quant) for name, window in windows.items()}
    if historic:
        observed = comparison['Confirmación histórica']['policies']
        expected = {'Gen 239': (289, -194.), 'Gen 268': (300, -186.)}
        replay_matches = all((observed[name]['total_trades'], observed[name]['net_profit']) == pair
                             for name, pair in expected.items())
    else:
        replay_matches = None
    payload = dict(period=args.period, sources=periods, account=asdict(founders[239].account),
                   stop=founders[239].drawdown_limit, replay_matches=replay_matches, comparison=comparison)
    (ROOT/f'{args.period}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print('RESULT ' + json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
