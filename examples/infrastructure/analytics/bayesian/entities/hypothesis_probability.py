"""Hypothesis probability data entity."""

from dataclasses import dataclass


@dataclass
class HypothesisProbability:
    """Probability data for a single hypothesis in Bayesian analysis."""

    hypothesis_id: str
    hypothesis_name: str
    prior_probability: float
    likelihood: float
    posterior_probability: float
    confidence_interval_lower: float
    confidence_interval_upper: float
