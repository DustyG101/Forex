"""Replay de contratos binarios con admisión causal y liquidación documentada."""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decision import resolve_action
from .evolution_direction import unit_payoff
from .evolution_lineage import MARKET


@dataclass(frozen=True)
class ContractExits:
    quotes: dict
    stop_loss: float
    take_profit: float

    @classmethod
    def from_csv(cls, path, *, stop_loss, take_profit):
        if not (np.isfinite([stop_loss, take_profit]).all() and 0. < stop_loss <= 1. and take_profit > 0.):
            raise ValueError('SL y TP requieren fracciones de stake positivas, con SL como máximo 1')
        timestamps = ('entry_utc', 'expiry_utc', 'quote_utc')
        columns = (*timestamps, 'side', 'sell_proceeds_usd')
        quotes = pd.read_csv(path, usecols=list(columns))
        for name in timestamps:
            quotes[name] = pd.to_datetime(quotes[name], utc=True)
        sides = quotes.side.map({'CALL': 1, 'PUT': -1})
        proceeds = pd.to_numeric(quotes.sell_proceeds_usd)
        if (quotes.empty or quotes[list(timestamps)].isna().any().any() or sides.isna().any()
                or not np.isfinite(proceeds).all() or (proceeds < 0.).any()
                or (quotes.expiry_utc != quotes.entry_utc + MARKET.step).any()
                or ((quotes.quote_utc <= quotes.entry_utc) | (quotes.quote_utc >= quotes.expiry_utc)).any()
                or quotes.duplicated(['entry_utc', 'expiry_utc', 'side', 'quote_utc']).any()):
            raise ValueError('Cotizaciones de recompra inválidas o fuera de la vida del contrato')
        ordered = quotes.assign(side=sides, sell_proceeds_usd=proceeds).sort_values('quote_utc')
        by_contract = {
            key: tuple(zip(rows.quote_utc, rows.sell_proceeds_usd, strict=True))
            for key, rows in ordered.groupby(['entry_utc', 'expiry_utc', 'side'])
        }
        return cls(by_contract, stop_loss, take_profit)

    def settle(self, quotes, *, stake, expiry, payoff_at_expiry):
        for time, proceeds in quotes:
            pnl = proceeds - stake
            if pnl <= -self.stop_loss * stake:
                return time, pnl, 'stop_loss'
            if pnl >= self.take_profit * stake:
                return time, pnl, 'take_profit'
        return expiry, payoff_at_expiry, 'expiry'


def apply_hourly_cap(entries, reasons, hourly_cap):
    """Cuenta entradas aceptadas en la hora móvil anterior a cada decisión."""
    if type(hourly_cap) is not int or hourly_cap < 1:
        raise ValueError('Cupo horario inválido')
    accepted = deque()
    hour = pd.Timedelta(hours=1).value
    times = entries.as_unit('ns').asi8
    decisions = np.asarray(reasons, dtype='<U24').copy()
    for i in np.flatnonzero(decisions == 'accepted'):
        time = times[i]
        while accepted and accepted[0] <= time - hour:
            accepted.popleft()
        if len(accepted) >= hourly_cap:
            decisions[i] = 'hourly_cap'
        else:
            accepted.append(time)
    return decisions


def gate_reasons(rows, sides, account, risk, *, n_paths, max_volatility, hourly_cap):
    if max_volatility is None or not np.isfinite(max_volatility) or max_volatility <= 0.:
        raise ValueError('Techo de volatilidad inválido')
    reasons = []
    for side, up, down, tie, sigma in zip(
        sides, rows.mc_prob_up, rows.mc_prob_down, rows.mc_prob_tie, rows.cond_volatility,
        strict=True,
    ):
        decision = resolve_action(2 if side > 0 else 0, up, down, tie, payout=account.payout,
                                  tie_policy=account.tie_policy, risk=risk, n_paths=n_paths)
        if decision['blocked']:
            reasons.append(decision['filter_reason'])
        elif sigma > max_volatility:
            reasons.append('volatility')
        else:
            reasons.append('accepted')
    return apply_hourly_cap(rows.index, reasons, hourly_cap)


def payoff_summary(profits, account, observed_hours):
    wins, losses = profits > 0., profits < 0.
    gross_win, gross_loss = float(profits[wins].sum()), float(-profits[losses].sum())
    pnl = gross_win - gross_loss
    trades = len(profits)
    balance = np.r_[account.initial_balance, account.initial_balance + np.cumsum(profits)]
    peaks = np.maximum.accumulate(balance)
    return dict(
        trades=trades, pnl_usd=pnl, pnl_in_stakes=pnl / account.stake,
        return_on_staked_pct=100 * pnl / (trades * account.stake) if trades else None,
        profit_factor=gross_win / gross_loss if gross_loss else None,
        profit_factor_status=('finite' if gross_loss else 'infinite_no_losses' if gross_win
                              else 'no_trades' if not trades else 'undefined_no_profit_or_loss'),
        wins=int(wins.sum()), losses=int(losses.sum()), refunds=int((profits == 0.).sum()),
        max_drawdown_pct=float(np.max(1 - balance / peaks) * 100),
        contracts_per_observed_hour=trades / observed_hours,
    )


def replay_contracts(rows, sides, account, risk, *, observed_hours, hourly_cap, n_paths=None,
                     max_volatility=None, exits=None, candidate_reasons=None):
    entries = rows.index
    sides = np.asarray(sides)
    if (rows.empty or sides.shape != (len(rows),) or not np.isin(sides, [-1, 1]).all()
            or not isinstance(entries, pd.DatetimeIndex)
            or entries.tz is None or entries.hasnans or (np.diff(entries.as_unit('ns').asi8) < MARKET.step.value).any()
            or not np.isfinite(observed_hours) or observed_hours <= 0.
            or not rows.outcome.isin([-1, 0, 1]).all()):
        raise ValueError('El replay requiere decisiones M5 válidas, cronológicas y sin solapamiento')
    if candidate_reasons is None:
        if not np.isfinite(rows.cond_volatility).all() or (rows.cond_volatility < 0.).any():
            raise ValueError('La volatilidad de las decisiones debe ser válida')
        reasons = gate_reasons(rows, sides, account, risk, n_paths=n_paths,
                               max_volatility=max_volatility, hourly_cap=hourly_cap)
        signal_rejections = ('confidence', 'expected_edge', 'volatility')
        rejected_key = 'rejected_probability_or_volatility_pct'
    else:
        signal_rejections = ('conviction_warmup', 'conviction')
        if (np.shape(candidate_reasons) != (len(entries),)
                or not np.isin(candidate_reasons, ('accepted', *signal_rejections)).all()):
            raise ValueError('El gate adaptativo debe decidir cada propuesta')
        reasons = apply_hourly_cap(entries, candidate_reasons, hourly_cap)
        rejected_key = 'rejected_signal_pct'
    accepted = reasons == 'accepted'
    expiry = entries + MARKET.step
    expiry_pnl = unit_payoff(sides, rows.outcome.to_numpy(), account) * account.stake
    profits = np.full(len(entries), np.nan)
    settlement = [pd.NaT] * len(entries)
    exit_reason = np.full(len(entries), 'not_entered', dtype='<U20')
    balance = peak = account.initial_balance
    stop = None
    limit = min(account.max_drawdown, risk.drawdown_limit)
    covered = 0
    for i in np.flatnonzero(accepted):
        if stop is not None:
            exit_reason[i] = 'account_stopped'
            continue
        time, pnl, reason = expiry[i], expiry_pnl[i], 'expiry'
        if exits is not None:
            quotes = exits.quotes.get((entries[i], expiry[i], sides[i]), ())
            covered += bool(quotes)
            time, pnl, reason = exits.settle(quotes, stake=account.stake, expiry=time, payoff_at_expiry=pnl)
        profits[i], settlement[i], exit_reason[i] = pnl, time, reason
        balance += pnl
        peak = max(peak, balance)
        drawdown = 1 - balance / peak
        if balance < account.stake or drawdown + 1e-12 >= limit:
            stop = dict(time=time.isoformat(), reason='insufficient_balance' if balance < account.stake else 'drawdown',
                        drawdown_pct=100 * drawdown, balance_usd=float(balance))

    counts = Counter(reasons)
    filters = dict(
        rejected_by_reason={reason: counts[reason] for reason in (*signal_rejections, 'hourly_cap')},
        accepted=counts['accepted'], rejected_pct=100 * (1 - counts['accepted'] / len(entries)),
    )
    filters[rejected_key] = 100 * sum(counts[r] for r in signal_rejections) / len(entries)
    executed = np.isfinite(profits)
    operational = payoff_summary(profits[executed], account, observed_hours)
    operational.update(account_stop=stop, unfilled_after_stop=int((accepted & ~executed).sum()),
                       quoted_trades=covered, stop_loss_exits=int((exit_reason == 'stop_loss').sum()),
                       take_profit_exits=int((exit_reason == 'take_profit').sum()))
    result = dict(
        admissions=len(entries), observed_hours=observed_hours, filters=filters,
        all_at_expiry=payoff_summary(expiry_pnl, account, observed_hours),
        gate_only_at_expiry=payoff_summary(expiry_pnl[accepted], account, observed_hours),
        operational=operational,
    )
    trace = pd.DataFrame(dict(entry_utc=entries, expiry_utc=expiry, side=np.where(sides > 0, 'CALL', 'PUT'),
                              gate_reason=reasons, exit_reason=exit_reason, settlement_utc=settlement,
                              pnl_usd=profits))
    return result, trace
