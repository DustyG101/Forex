"""Admisión causal por convicción relativa y madurez observada."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

CONVICTION_WINDOW = 500
PERCENTILE_STAGES = (.85, .60, .30, 0.)
MASTERY_CONVICTION = .80
MASTERY_ACCURACY = .95
MASTERY_BLOCKS = 3
MASTERY_STEP_SETTLEMENTS = 100


@dataclass(frozen=True)
class AdaptiveDecision:
    reason: str
    conviction: float
    mean_conviction: float | None
    cutoff: float | None
    percentile: float
    mastery_evidence: bool


def conviction(score):
    if not np.isfinite(score):
        raise ValueError('El puntaje de dirección debe ser finito')
    return float(abs(np.tanh(score)))


class AdaptiveGate:
    """Cada decisión consulta solo propuestas y liquidaciones anteriores."""

    def __init__(self, *, percentile=PERCENTILE_STAGES[0], elastic=True, warmup_scores=()):
        if not PERCENTILE_STAGES[1] < percentile < 1.:
            raise ValueError('Percentil inicial fuera de las etapas de elasticidad')
        self._stages = (percentile, *PERCENTILE_STAGES[1:])
        self.elastic = elastic
        self._convictions = deque((conviction(s) for s in warmup_scores), maxlen=CONVICTION_WINDOW)
        self._settled = deque(maxlen=CONVICTION_WINDOW)
        self._resolved = 0
        self._mastery_since = None

    def observe_settlement(self, won):
        """Registra una propuesta ya vencida, incluso si el gate la rechazó."""
        if won is not None:
            self._settled.append(bool(won))
            self._resolved += 1

    def _mastered(self, mean_conviction):
        if (not self.elastic or mean_conviction is None or mean_conviction <= MASTERY_CONVICTION
                or len(self._settled) < CONVICTION_WINDOW):
            return False
        hits = np.fromiter(self._settled, dtype=bool)
        return all(block.mean() >= MASTERY_ACCURACY for block in np.array_split(hits, MASTERY_BLOCKS))

    def decide(self, score):
        current = conviction(score)
        ready = len(self._convictions) == CONVICTION_WINDOW
        mean = float(np.mean(self._convictions)) if ready else None
        mastered = self._mastered(mean)
        if mastered and self._mastery_since is None:
            self._mastery_since = self._resolved
        elif not mastered:
            self._mastery_since = None
        confirmed = 0 if self._mastery_since is None else self._resolved - self._mastery_since
        stage = min(confirmed // MASTERY_STEP_SETTLEMENTS, len(self._stages) - 1)
        percentile = self._stages[stage]
        cutoff = (float(np.quantile(self._convictions, percentile)) if percentile else 0.) if ready else None
        reason = 'accepted'
        if not ready:
            reason = 'conviction_warmup'
        elif percentile and current <= cutoff:
            reason = 'conviction'
        self._convictions.append(current)
        return AdaptiveDecision(reason, current, mean, cutoff, percentile, mastered)


def rolling_conviction_admissions(scores, percentile):
    """Evaluación por lote de la etapa restrictiva sobre propuestas ordenadas."""
    values = np.abs(np.tanh(scores))
    cuts = pd.Series(values).rolling(CONVICTION_WINDOW).quantile(percentile).shift()
    return values > cuts.to_numpy()
