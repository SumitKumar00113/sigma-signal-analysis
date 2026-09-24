"""Combine the rule-based classifier with the learned model.

``hybrid`` (the default when a model is available) keeps the explainable
rule-based decision unless the evidence says the model knows better:

* the rules found an unmodulated carrier → keep that (it is measured
  directly);
* the rules found nothing (UNKNOWN) and the model is reasonably sure →
  take the model's answer;
* rules and model disagree, the model is very sure, and it gives the
  rules' answer almost no probability → take the model's answer;
* otherwise keep the rules, listing the model's view as evidence.

The thresholds were checked on independently generated signals with
``scripts/eval_classifier.py``, which reports rules / model / hybrid
accuracy side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.enums import ModulationType
from src.dsp.classification import ClassificationResult
from src.ml.dataset import canonical_label
from src.ml.model import Prediction

MODES = ("rules", "ml", "hybrid")

TAKE_OVER_UNKNOWN = 0.6      # model probability needed when the rules found nothing
OVERRIDE_TOP = 0.85          # model probability needed to overrule the rules …
OVERRIDE_RULE_MAX = 0.05     # … when the model gives the rules' answer at most this


@dataclass
class Decision:
    modulation: ModulationType
    confidence: float
    candidates: list[tuple[ModulationType, float]]
    source: str                              # "rules", "model" or "rules+model"
    evidence: list[str] = field(default_factory=list)


def _model_text(pred: Prediction) -> str:
    top = ", ".join(f"{m.value} {p:.0%}" for m, p in pred.ranked[:3])
    return f"Learned model: {top}."


def combine(rule: ClassificationResult, pred: Prediction | None, mode: str) -> Decision:
    if mode not in MODES:
        raise ValueError(f"Unknown classifier mode {mode!r}")
    rules_decision = Decision(rule.modulation, rule.confidence, list(rule.candidates), "rules")
    if pred is None or mode == "rules":
        return rules_decision

    model_decision = Decision(pred.modulation, pred.probability, list(pred.ranked), "model",
                              [_model_text(pred)])
    if mode == "ml":
        return model_decision

    probs = dict(pred.ranked)
    rule_mod = canonical_label(rule.modulation)
    unmodulated = any("Unmodulated carrier" in e for e in rule.evidence)

    if unmodulated:
        # Measured directly (one carrier line, flat envelope): keep it
        rules_decision.evidence.append(_model_text(pred))
        return rules_decision

    if rule.modulation == ModulationType.UNKNOWN:
        if pred.modulation != ModulationType.UNKNOWN and pred.probability >= TAKE_OVER_UNKNOWN:
            model_decision.evidence.insert(0, "Rule-based features were inconclusive; "
                                              "using the learned model.")
            return model_decision
        rules_decision.evidence.append(_model_text(pred))
        return rules_decision

    if canonical_label(pred.modulation) == rule_mod:
        rules_decision.confidence = max(rule.confidence, pred.probability)
        rules_decision.source = "rules+model"
        rules_decision.evidence.append(_model_text(pred) + " Agrees with the rules.")
        return rules_decision

    if pred.probability >= OVERRIDE_TOP and probs.get(rule_mod, 0.0) <= OVERRIDE_RULE_MAX:
        model_decision.evidence.insert(
            0, f"The learned model overrides the rule-based answer ({rule.modulation.value}, "
               f"which it rates {probs.get(rule_mod, 0.0):.0%}).")
        return model_decision

    rules_decision.evidence.append(_model_text(pred) + " (kept the rule-based answer)")
    return rules_decision
