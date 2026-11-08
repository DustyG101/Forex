"""Replay March with frozen Gen 268, M15/GARCH vetoes and matched MC controls."""
import json

import numpy as np
import pandas as pd

from scripts.audit_m5_mc import audit_window, grouped
from scripts.compare_m5_mc import GENOMES, PRESET, RAW, ROOT, prepared_data, read_manifest
from src.decision import RiskConfig, mc_probability_lower
from src.environment import BinaryOptionsEnv
from src.evolution_context import build_observations, closed_trend_bias
from src.evolution_engine import M5Simulation, assess_fitness
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.market import PROTECTED_HOLDOUT_START
from src.quant_engine import QuantConfig


def outcome_table(future, direction, mask, account):
    future, direction = future[mask], direction[mask]
    profits = np.where(future == direction, account.stake * account.payout,
                       np.where((future == 0) & (account.tie_policy == 'refund'), 0., -account.stake))
    return grouped(pd.DataFrame({'profit': profits}))


def replay(window, rows, founder, mask, direction):
    env = BinaryOptionsEnv(window, market=MARKET, prepared=True, account=founder.account,
                           risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit))
    sim = M5Simulation(env)
    assert pd.Index(env._times[sim.entries]).equals(rows.index)
    active = mask & (direction != 0)
    sim.features = np.column_stack((active.astype(float), direction))
    sim.mc_margin = np.zeros(len(rows))
    metrics = sim.evaluate(np.array([1., 0., -.5, 0., 1., 0., 0.]))
    assert metrics['signals_after_activation'] == int(active.sum())
    count = metrics['total_trades']
    last = np.flatnonzero(active)[count - 1] if count else None
    keys = ('total_trades', 'win_rate_excluding_refunds', 'profit_factor', 'net_profit',
            'max_drawdown_pct', 'risk_stop', 'trades_per_observed_hour', 'signals_after_activation',
            'signals_unfilled_due_to_stop')
    return {k: metrics[k] for k in keys} | dict(
        last_expiry=rows.expiry_utc.iloc[last].isoformat() if last is not None else None,
        checks=assess_fitness(metrics, payout=founder.account.payout)['checks'],
    )


def main():
    manifest = read_manifest(GENOMES[268])
    assert manifest['selection']['generation'] == 268
    quant = QuantConfig.from_metadata(manifest['quant_config'])
    train_end, search_end = (pd.Timestamp(manifest['splits'][name]['end']) for name in ('train', 'search'))
    founder = load_founder(GENOMES[268], None, train_end, search_end)
    RAW.mkdir(parents=True, exist_ok=True)
    data = prepared_data(['202601','202602','202603'], quant)
    data = data.loc[data.timestamp + MARKET.step <= pd.Timestamp('2026-04-01T00:00:00Z')]
    assert data.timestamp.iloc[-1] + MARKET.step <= PROTECTED_HOLDOUT_START
    ctx_raw = build_observations(data)
    norm, width = founder.normalizer, len(MARKET_FEATURES)
    ctx = ((ctx_raw - np.asarray(norm['mean'][width:])) / np.asarray(norm['std'][width:])).clip(-norm['clip'], norm['clip'])
    m15 = pd.Series(closed_trend_bias(data, 'M15'), index=data.timestamp)
    _, _, feb_raw, _, _, _ = audit_window(data.loc[data.timestamp.dt.month.eq(2)], ctx_raw, ctx, m15, founder)
    low_edge, high_edge = np.quantile(feb_raw.cond_volatility, [1/3, 2/3])
    window = data.loc[data.timestamp.dt.month.eq(3)]
    summary, rows, raw, _, _, traded = audit_window(window, ctx_raw, ctx, m15, founder)
    env_hours = summary['funnel']['hours']
    assert len(traded) == 101 and traded.profit.sum() == -198.
    high_vol = raw.cond_volatility.to_numpy() > high_edge
    m15_proxy = raw.m15_ema_gap_proxy.to_numpy()
    genotype = rows.direction.to_numpy()
    gen_mask = rows.accepted.to_numpy()
    original = rows.executed.to_numpy()
    future = rows.future_direction.to_numpy()
    countertrend = m15_proxy * genotype < 0.

    rules = {'Base Gen 268': np.ones(len(rows), dtype=bool),
             'Veto M15': ~countertrend,
             'Veto volatilidad': ~high_vol,
             'Ambos vetos': ~(countertrend | high_vol)}
    original_retained = {name:outcome_table(future, genotype, original & allowed, founder.account)
                         for name,allowed in rules.items()}
    gen_replays = {name:replay(window, rows, founder, gen_mask & allowed, genotype)
                   for name,allowed in rules.items()}
    assert gen_replays['Base Gen 268']['total_trades'] == 101
    assert gen_replays['Base Gen 268']['net_profit'] == -198.

    up, down = raw.mc_prob_up.to_numpy(), raw.mc_prob_down.to_numpy()
    mc_dir = np.sign(up - down)
    confidence = np.maximum(up, down)
    lower = mc_probability_lower(confidence, quant.n_paths, PRESET['FOREX_MC_Z'])
    mc_mask = ((confidence > PRESET['FOREX_MIN_CONFIDENCE'])
               & ((founder.account.payout+1.)*lower-1. > PRESET['FOREX_MIN_EXPECTED_EDGE'])
               & (mc_dir != 0.))
    paired_replays = {
        'Entradas Gen / dirección Monte Carlo': replay(window, rows, founder, gen_mask, mc_dir),
        'Entradas Monte Carlo / dirección Gen': replay(window, rows, founder, mc_mask, genotype),
        'Monte Carlo': replay(window, rows, founder, mc_mask, mc_dir),
        'Monte Carlo con veto volatilidad': replay(window, rows, founder, mc_mask & ~high_vol, mc_dir),
        'Entradas Monte Carlo / siempre CALL': replay(window, rows, founder, mc_mask, np.ones(len(rows))),
        'Entradas Monte Carlo / siempre PUT': replay(window, rows, founder, mc_mask, -np.ones(len(rows))),
    }
    assert paired_replays['Monte Carlo']['total_trades'] == 503
    assert paired_replays['Monte Carlo']['net_profit'] == -88.
    bands = {'baja':raw.cond_volatility.to_numpy() <= low_edge,
             'media':(raw.cond_volatility.to_numpy() > low_edge) & ~high_vol, 'alta':high_vol}
    proposed = {}
    for name, mask, side in [('Gen 268', gen_mask, genotype), ('Monte Carlo', mc_mask, mc_dir)]:
        proposed[name] = dict(
            n=int(mask.sum()),
            calls=int((mask & (side > 0)).sum()),
            puts=int((mask & (side < 0)).sum()),
            high_vol_share=float((mask & high_vol).sum()/mask.sum()),
            countertrend=int((mask & (m15_proxy*side < 0.)).sum()),
            outcomes=outcome_table(future, side, mask, founder.account),
            by_volatility={name:outcome_table(future, side, mask & band, founder.account)
                           for name,band in bands.items()},
        )
    common = gen_mask & mc_mask
    disagree = common & (genotype != mc_dir)
    payload = dict(
        dates=[window.timestamp.iloc[0].isoformat(),window.timestamp.iloc[-1].isoformat()],
        observed_hours=env_hours, threshold=float(founder.genome[-1]),
        volatility_cutoff_february=float(high_edge),
        original_veto_counts=dict(m15=int((original & countertrend).sum()),
                                  volatility=int((original & high_vol).sum()),
                                  overlap=int((original & countertrend & high_vol).sum())),
        original_retained=original_retained, gen_replays=gen_replays, paired_replays=paired_replays,
        realized_directions=dict(up=int((future>0).sum()), down=int((future<0).sum()), ties=int((future==0).sum())),
        proposed_before_stop=proposed,
        matched=dict(n=int(common.sum()), disagreements=int(disagree.sum()),
                     all_gen=outcome_table(future, genotype, common, founder.account),
                     all_mc=outcome_table(future, mc_dir, common, founder.account),
                     disagreements_gen=outcome_table(future, genotype, disagree, founder.account),
                     disagreements_mc=outcome_table(future, mc_dir, disagree, founder.account)),
    )
    (ROOT/'veto.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print('VETO ' + json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
