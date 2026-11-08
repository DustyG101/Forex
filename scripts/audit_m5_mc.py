"""Audit the Gen 268 frozen policy before and after its March stop."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from scripts.compare_m5_mc import GENOMES, ROOT, prepared_data, read_manifest
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv
from src.evolution_context import build_observations, closed_trend_bias
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.quant_engine import QuantConfig


def rho(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3 or x.min() == x.max() or y.min() == y.max():
        return None
    return float(spearmanr(x, y).statistic)


def describe(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    q = np.quantile(x, [.1, .5, .9])
    return dict(n=len(x), mean=float(np.mean(x)), std=float(np.std(x)), p10=float(q[0]),
                median=float(q[1]), p90=float(q[2]))


def grouped(df):
    wins = int((df.profit > 0).sum())
    losses = int((df.profit < 0).sum())
    gross = -float(df.loc[df.profit < 0, 'profit'].sum())
    return dict(n=len(df), wins=wins, losses=losses, refunds=len(df)-wins-losses,
                win_rate=wins/(wins+losses) if wins+losses else None,
                pf=float(df.loc[df.profit > 0, 'profit'].sum())/gross if gross else None,
                pnl=float(df.profit.sum()))


def funnel(rows, hours):
    stats = {name: int(rows[name].sum()) for name in ('entry_positive', 'direction_nonzero', 'accepted', 'executed')}
    return dict(eligible=len(rows), hours=hours, **stats, entry_rejected=len(rows)-stats['entry_positive'],
                direction_ties=stats['entry_positive']-stats['direction_nonzero'],
                threshold_rejected=stats['direction_nonzero']-stats['accepted'],
                threshold_rejection_rate=1-stats['accepted']/max(1, stats['direction_nonzero']),
                unfilled_stop=stats['accepted']-stats['executed'],
                accepted_per_hour=stats['accepted']/hours,
                executed_per_hour=stats['executed']/hours)


def audit_window(window, context_raw, context, m15, founder):
    normalizer = founder.normalizer
    width = len(MARKET_FEATURES)
    market_normalizer = {key: normalizer[key][:width] for key in ('features', 'mean', 'std')}
    market_normalizer['clip'] = normalizer['clip']
    env = BinaryOptionsEnv(window, market=MARKET, prepared=True, account=founder.account,
                           risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit),
                           normalizer=market_normalizer)
    sim = M5Simulation(env, context.loc[window.timestamp])
    frozen = sim.evaluate(founder.genome)
    heads = founder.genome[:-1].reshape(2, sim.features.shape[1]+1)
    threshold = float(founder.genome[-1])
    contributions = sim.features[:, None, :] * heads[None, :, :-1]
    scores = np.einsum('ij,kj->ik', sim.features, heads[:, :-1]) + heads[:, -1]
    direction = sim.mc_margin + scores[:, 1]
    margin = np.abs(direction)
    mask_entry = scores[:, 0] > 0.
    mask_direction = mask_entry & (margin > 0.)
    accepted = mask_direction & (margin > threshold)
    accepted_idx = np.flatnonzero(accepted)
    executed_idx = accepted_idx[:frozen['total_trades']]
    executed = np.zeros(len(accepted), dtype=bool)
    executed[executed_idx] = True
    candidate_side = np.sign(direction)
    tied = sim.change == 0.
    payoff = founder.account
    profit = np.where(sim.change == candidate_side, payoff.stake*payoff.payout,
                      np.where(tied & (payoff.tie_policy == 'refund'), 0., -payoff.stake))
    opens = env._times[sim.entries]
    raw = pd.concat([window.set_index('timestamp').loc[opens, list(MARKET_FEATURES)], context_raw.loc[opens]], axis=1)
    raw['m15_ema_gap_proxy'] = m15.loc[opens]
    raw['entry_score'] = scores[:, 0]
    raw['direction_score'] = direction
    raw['direction_margin'] = margin
    rows = pd.DataFrame(dict(entry_positive=mask_entry, direction_nonzero=mask_direction, accepted=accepted,
                             executed=executed, direction=candidate_side, profit=profit,
                             future_direction=sim.change, entry_utc=sim.timestamps[sim.entries],
                             expiry_utc=sim.timestamps[sim.entries+1]), index=opens)
    assert int(accepted.sum()) == frozen['signals_after_activation']
    traded = rows.loc[rows.executed].copy()
    assert float(traded.profit.sum()) == frozen['net_profit']
    traded['balance'] = payoff.initial_balance + traded.profit.cumsum()
    traded['peak'] = traded.balance.cummax().clip(lower=payoff.initial_balance)
    traded['drawdown_pct'] = 100*(1-traded.balance/traded.peak)
    stop = traded.iloc[-1] if frozen['risk_stop'] else None
    stop_info = None if stop is None else dict(expiry_utc=stop.expiry_utc.isoformat(), executed=len(traded),
        pnl=float(traded.profit.sum()), peak=float(stop.peak), balance=float(stop.balance),
        drawdown_pct=float(stop.drawdown_pct), eligible_through_last_entry=int(executed_idx[-1]+1),
        accepted_through_last_entry=int(rows.accepted.iloc[:executed_idx[-1]+1].sum()))
    prefix = traded.iloc[:101]
    summary = dict(funnel=funnel(rows, sim.observed_hours), trade_outcomes=grouped(traded), stop=stop_info,
                   first_101=grouped(prefix), sides={name:grouped(traded.loc[traded.direction==side])
                       for side,name in [(-1,'PUT'),(1,'CALL')]},
                   rank_correlation={name:rho(raw[name], rows.future_direction)
                                     for name in raw.columns},
                   margin_vs_win={name:rho(raw.loc[selection, 'direction_margin'], rows.loc[selection, 'profit']>0)
                                  for name,selection in [('accepted_all', rows.accepted & (rows.profit!=0)),
                                                         ('executed', rows.executed & (rows.profit!=0))]})
    return summary, rows, raw, sim.features, contributions, traded


def main():
    manifest = read_manifest(GENOMES[268])
    quant = QuantConfig.from_metadata(manifest['quant_config'])
    train_end, search_end = (pd.Timestamp(manifest['splits'][name]['end']) for name in ('train', 'search'))
    founder = load_founder(GENOMES[268], None, train_end, search_end)
    data = prepared_data(['202601','202602','202603'], quant)
    data = data.loc[data.timestamp + MARKET.step <= pd.Timestamp('2026-04-01T00:00:00Z')]
    ctx_raw = build_observations(data)
    norm = founder.normalizer
    width = len(MARKET_FEATURES)
    ctx = (ctx_raw - np.asarray(norm['mean'][width:])) / np.asarray(norm['std'][width:])
    ctx = ctx.clip(-norm['clip'], norm['clip'])
    m15 = pd.Series(closed_trend_bias(data, 'M15'), index=data.timestamp)
    audit = {month: audit_window(data.loc[data.timestamp.dt.month.eq(m)], ctx_raw, ctx, m15, founder)
             for m,month in ((2,'Febrero'),(3,'Marzo'))}
    feb, mar = audit['Febrero'], audit['Marzo']
    features = {}
    for i, name in enumerate(feb[2].columns):
        a, b = feb[2][name].dropna(), mar[2][name].dropna()
        row = dict(feb=describe(a), mar=describe(b), ks=float(ks_2samp(a,b).statistic))
        if i < len(norm['features']):
            assert name == norm['features'][i]
            delta = mar[4][:,:,i].mean(axis=0)-feb[4][:,:,i].mean(axis=0)
            row.update(normalized_mean_shift=float(mar[3][:,i].mean()-feb[3][:,i].mean()),
                       entry_contribution_change=float(delta[0]), direction_contribution_change=float(delta[1]))
        features[name] = row

    vol_edges = np.quantile(feb[2].cond_volatility, [1/3, 2/3])
    for month, (summary, rows, raw, normalized, _, traded) in audit.items():
        vol = raw.loc[traded.index, 'cond_volatility']
        volatility = pd.cut(vol, [-np.inf,*vol_edges,np.inf], labels=['baja','media','alta'])
        summary['volatility_tertiles'] = {label:grouped(traded.loc[volatility==label])
                                         for label in ('baja','media','alta')}
        m15_trade = raw.loc[traded.index, 'm15_ema_gap_proxy']
        agreement = np.sign(m15_trade) == traded.direction
        available = m15_trade.notna() & (m15_trade!=0.)
        summary['m15_agreement'] = {'agrees':grouped(traded.loc[available & agreement]),
                                    'disagrees':grouped(traded.loc[available & ~agreement]),
                                    'missing_or_flat':grouped(traded.loc[~available])}
        used = pd.DataFrame(normalized[rows.executed.to_numpy()], index=traded.index,
                            columns=[f'input_{name}' for name in norm['features']])
        traded.join(raw).join(used).to_csv(ROOT/f'{month.lower()}-contratos-gen268.csv', index=False)
    payload = dict(threshold=float(founder.genome[-1]), founder=founder.provenance,
                   volatility_edges=vol_edges.tolist(), months={month:v[0] for month,v in audit.items()},
                   feature_distributions=features, excluded_from_model=['m15_ema_gap_proxy'])
    (ROOT/'audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    top = sorted(features, key=lambda k:features[k]['ks'], reverse=True)[:12]
    print('AUDIT ' + json.dumps(payload | {'feature_distributions':{k:features[k] for k in top}},
                                ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
