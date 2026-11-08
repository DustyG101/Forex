"""Ablación evolutiva sobre la dirección y el payoff de admisiones fijas."""

import numpy as np

from .evolution_context import CONTEXT_FEATURES, H1_STRUCTURAL, M15_TACTICAL
from .evolution_direction import direction_genes, unit_payoff
from .evolution_engine import (
    ELITE_FRACTION, WIDE_MUTATION_FRACTION, breed, seeded_population,
)
from .evolution_lineage import MARKET_FEATURES, OBSERVATION_FEATURES


M5_TACTICAL = (*MARKET_FEATURES, *(name for name in CONTEXT_FEATURES if name.startswith('m5_')))
FEATURE_NAMES = (*M5_TACTICAL, *M15_TACTICAL, *H1_STRUCTURAL)
ARM_WIDTHS = dict(m5=len(M5_TACTICAL), m15=len(M5_TACTICAL) + len(M15_TACTICAL),
                  m15_h1=len(FEATURE_NAMES))
HIDDEN = 8
GENOME_WIDTH = (len(FEATURE_NAMES) + 2) * HIDDEN + 1
POPULATION = 128
FIRST_GENERATION, LAST_GENERATION = 315, 335
SEEDS = (42, 43, 44)
SCALE_CLIP = 6.
PF_PRIOR = 5.
WORST_WEIGHT = .55
PF_DISPERSION_WEIGHT = .20
MIN_SUBBLOCK = 64


def fit_scale(inputs):
    if np.isinf(inputs).any() or not np.isfinite(inputs).any(axis=0).all():
        raise ValueError('Cada feature requiere observaciones finitas en marzo')
    mean = np.nanmean(inputs, axis=0)
    std = np.nanstd(inputs, axis=0)
    return mean, np.where(std > 1e-12, std, 1.)


def scale_inputs(inputs, mean, std):
    scaled = (inputs - mean) / std
    return np.nan_to_num(scaled, nan=0.).clip(-SCALE_CLIP, SCALE_CLIP)


def project_founder(checkpoint, mean, std):
    """Proyecta el residuo táctico del control a la primera neurona tanh."""
    inherited = direction_genes(checkpoint.genome)
    old = checkpoint.normalizer
    indices = [OBSERVATION_FEATURES.index(name) for name in M5_TACTICAL]
    size = len(indices)
    raw_weights = inherited[indices] / np.asarray(old['std'])[indices]
    bias = inherited[-1] + raw_weights @ (mean[:size] - np.asarray(old['mean'])[indices])
    # El margen MC de la política anterior se incorpora a los pesos iniciales.
    up, down = (M5_TACTICAL.index(name) for name in ('mc_prob_up', 'mc_prob_down'))
    raw_weights[up] += 1.
    raw_weights[down] -= 1.
    bias += mean[up] - mean[down]
    weights = np.zeros((len(FEATURE_NAMES), HIDDEN))
    weights[:size, 0] = raw_weights * std[:size]
    hidden_bias, output = np.zeros(HIDDEN), np.zeros(HIDDEN)
    hidden_bias[0], output[0] = bias, 1.
    return np.r_[weights.ravel(), hidden_bias, output, 0.]


def arm_inputs(inputs, arm):
    values = inputs.copy()
    values[:, ARM_WIDTHS[arm]:] = 0.
    return values


def direction_scores(inputs, genomes):
    """Puntúa la población en lote con la misma topología en cada rama."""
    genomes = np.asarray(genomes, dtype=float)
    count, width = len(genomes), inputs.shape[1]
    weights = genomes[:, :width * HIDDEN].reshape(count, width, HIDDEN)
    hidden = (inputs @ weights.transpose(1, 0, 2).reshape(width, count * HIDDEN)).reshape(
        len(inputs), count, HIDDEN,
    )
    hidden += genomes[:, -2 * HIDDEN - 1:-HIDDEN - 1]
    np.tanh(hidden, out=hidden)
    scores = np.einsum('nph,ph->pn', hidden, genomes[:, -HIDDEN - 1:-1])
    scores += genomes[:, -1, None]
    return scores


def directions(inputs, genomes):
    scores = direction_scores(inputs, genomes)
    return np.where(scores >= 0., 1, -1).astype(np.int8)


def volatility_groups(volatility):
    """Separa regímenes dentro de cuatro bloques contiguos de admisiones."""
    cutoff = float(np.quantile(volatility, 2/3))
    groups = np.empty(len(volatility), dtype=np.int8)
    names = []
    for number, block in enumerate(np.array_split(np.arange(len(groups)), 4), start=1):
        high = volatility[block] > cutoff
        strata = [(f'{number}-moderada', block[~high]), (f'{number}-alta', block[high])]
        if min(len(indices) for _, indices in strata) < MIN_SUBBLOCK:
            strata = [(f'{number}-mixta', block)]
        for name, indices in strata:
            groups[indices] = len(names)
            names.append(name)
    return groups, names, cutoff


def directional_metrics(side, outcome, mc_side, account):
    payoff = unit_payoff(side, outcome, account)
    baseline = unit_payoff(mc_side, outcome, account)
    wins, losses = int((payoff > 0.).sum()), int((payoff < 0.).sum())
    return dict(
        admissions=len(outcome), wins=wins, losses=losses, refunds=len(outcome) - wins - losses,
        win_rate_excluding_refunds=wins / (wins + losses) if wins + losses else None,
        profit_factor=account.payout * wins / losses if losses else None,
        return_per_admission=float(payoff.mean()), pnl=float(payoff.sum() * account.stake),
        delta_vs_mc=float((payoff - baseline).mean()), disagreements_vs_mc=int((side != mc_side).sum()),
    )


class MultiregimeFitness:
    """Payoff unitario antes del stop, con bloques y denominadores compartidos."""

    def __init__(self, outcome, mc_side, groups, account):
        self.outcome = outcome
        self.account = account
        self.membership = np.eye(int(groups.max()) + 1, dtype=np.float32)[groups]
        self.admissions = self.membership.sum(axis=0, dtype=float)
        resolved = (outcome != 0) | (account.tie_policy != 'refund')
        self.resolved = resolved @ self.membership
        mc = unit_payoff(mc_side, outcome, account)
        self.mc_return = float(mc.mean())
        self.mc_blocks = (mc @ self.membership) / self.admissions

    def evaluate(self, sides):
        wins = ((sides == self.outcome).astype(np.float32) @ self.membership).astype(float)
        losses = self.resolved - wins
        gross_profit = self.account.payout * wins
        block_returns = (gross_profit - losses) / self.admissions
        total_return = (gross_profit - losses).sum(axis=1) / self.admissions.sum()
        block_delta = block_returns - self.mc_blocks
        regularized_pf = (gross_profit + PF_PRIOR) / (losses + PF_PRIOR)
        dispersion = np.log(regularized_pf).std(axis=1)
        worst = block_returns.min(axis=1)
        score = (1 - WORST_WEIGHT) * total_return + WORST_WEIGHT * worst - PF_DISPERSION_WEIGHT * dispersion
        return dict(
            score=score, return_per_admission=total_return, delta_vs_mc=total_return - self.mc_return,
            worst_return=worst, worst_delta_vs_mc=block_delta.min(axis=1), pf_log_std=dispersion,
            block_returns=block_returns, block_deltas_vs_mc=block_delta,
            block_regularized_pf=regularized_pf,
            qualifies=(total_return > 0.) & (block_delta > 1e-12).all(axis=1),
        )


def champion_key(result):
    return (result['qualifies'], result['score'], result['return_per_admission'],
            result['worst_delta_vs_mc'], -result['pf_log_std'])


def evolve_replica(inputs, founder, objective, arm, seed, *,
                   first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION):
    """Mantiene el mejor individuo y reutiliza cruce y mutación del motor M5."""
    inputs = arm_inputs(inputs, arm)
    return evolve_population(lambda population: objective.evaluate(directions(inputs, population)), founder,
                             arm, seed, first_generation=first_generation, last_generation=last_generation,
                             ranking=champion_key)


def evolve_population(evaluate, founder, arm, seed, *, first_generation, last_generation, ranking):
    if not 0 <= first_generation <= last_generation:
        raise ValueError('Intervalo de generaciones inválido')
    rng = np.random.default_rng([seed, first_generation])
    sigma = .12
    wide_count = round(WIDE_MUTATION_FRACTION * POPULATION)
    parents_count = int(np.ceil(ELITE_FRACTION * POPULATION))
    population = seeded_population(rng, founder, POPULATION, sigma, wide_count=wide_count)
    history = []
    winner = genome = founder_fitness = None
    for generation in range(first_generation, last_generation + 1):
        fitness = evaluate(population)
        parents = np.lexsort((-np.arange(POPULATION), *reversed(ranking(fitness))))[-parents_count:][::-1]
        best = parents[0]
        if generation == first_generation:
            founder_fitness = {name: values[0].tolist() for name, values in fitness.items()}
        record = dict(generation=generation, candidate=int(best),
                      **{name: values[best].tolist() for name, values in fitness.items()})
        improved = winner is None or ranking(record) > ranking(winner)
        if improved:
            winner, genome = record, population[best].copy()
        returns = fitness['return_per_admission']
        history.append(dict(record, median_score=float(np.median(fitness['score'])), mutation_sigma=sigma,
                            nonnegative_candidates=int((returns >= 0.).sum()),
                            max_candidate_return=float(returns.max())))
        if generation < last_generation:
            sigma = max(.025, sigma * .85) if improved else min(.35, sigma * 1.25)
            population = breed(rng, population, [best], parents, sigma, wide_count=wide_count)
    return dict(arm=arm, seed=seed, founder=founder_fitness, champion=winner, genome=genome.tolist(),
                generations=history, evaluations=POPULATION * len(history))
