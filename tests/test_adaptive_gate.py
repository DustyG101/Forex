import numpy as np
import pandas as pd
import pytest

from src.adaptive_gate import AdaptiveGate, rolling_conviction_admissions
from src.decision import RiskConfig
from src.environment import AccountConfig
from src.replay_engine import replay_contracts


def test_cutoff_excludes_current_proposal_and_keeps_rejected_proposals():
    prior = np.linspace(0., 2., 500)
    gate = AdaptiveGate(warmup_scores=prior)
    first = gate.decide(.1)
    assert first.cutoff == pytest.approx(np.quantile(np.tanh(prior), .85))
    assert first.reason == 'conviction'
    following = gate.decide(3.)
    assert following.cutoff == pytest.approx(np.quantile(np.tanh(np.r_[prior[1:], .1]), .85))
    assert following.reason == 'accepted'
    tied = AdaptiveGate(warmup_scores=np.full(500, 3.))
    for i in range(500):
        tied.observe_settlement(i % 4 != 0)
    decision = tied.decide(3.)
    assert decision.mean_conviction > .80
    assert decision.percentile == .85 and not decision.mastery_evidence
    assert decision.reason == 'conviction'


def test_mastery_needs_settled_blocks_and_reopens_in_confirmed_steps():
    gate = AdaptiveGate(warmup_scores=np.full(500, 2.))
    for _ in range(499):
        gate.observe_settlement(True)
    assert not gate.decide(3.).mastery_evidence
    gate.observe_settlement(True)
    first = gate.decide(3.)
    assert first.mastery_evidence and first.percentile == .85
    for expected in (.60, .30, 0.):
        for _ in range(100):
            gate.observe_settlement(True)
        decision = gate.decide(3.)
        assert decision.percentile == expected
    assert gate.decide(.01).reason == 'accepted' and decision.cutoff == 0.
    for _ in range(12):
        gate.observe_settlement(False)
    deteriorated = gate.decide(.01)
    assert deteriorated.percentile == .85 and deteriorated.reason == 'conviction'


@pytest.mark.parametrize('percentile', [.85, .80, .75])
def test_rolling_cutoffs_move_autonomously_and_match_batch(percentile):
    scores = np.r_[np.linspace(-2., 2., 500), np.linspace(-3., 3., 128)]
    gate = AdaptiveGate(percentile=percentile, warmup_scores=scores[:500])
    decisions = [gate.decide(score) for score in scores[500:]]
    np.testing.assert_array_equal([d.reason == 'accepted' for d in decisions],
                                  rolling_conviction_admissions(scores, percentile)[500:])
    assert len({d.cutoff for d in decisions}) > 1
    assert {d.percentile for d in decisions} == {percentile}


def test_adaptive_admission_keeps_risk_cap_and_account_stop():
    times = pd.date_range('2026-03-01T00:00Z', periods=3, freq='5min')
    rows = pd.DataFrame({'outcome': [-1, 1, 1]}, index=times)
    result, trace = replay_contracts(rows, [1, 1, 1], AccountConfig(initial_balance=50.), RiskConfig(),
                                     observed_hours=1., hourly_cap=2, candidate_reasons=['accepted'] * 3)
    assert trace.gate_reason.tolist() == ['accepted', 'accepted', 'hourly_cap']
    assert trace.exit_reason.tolist() == ['expiry', 'account_stopped', 'not_entered']
    assert result['operational']['account_stop']['drawdown_pct'] == pytest.approx(20.)
