"""Bayesian probability calculator service."""

import numpy as np
from infrastructure.analytics.bayesian.entities.bayesian_analysis_result import (
    BayesianAnalysisResult,
)
from infrastructure.analytics.bayesian.entities.confidence_interval import (
    ConfidenceInterval,
)
from infrastructure.analytics.bayesian.entities.evidence_likelihood_mapping import (
    EvidenceLikelihoodMapping,
)
from infrastructure.analytics.bayesian.entities.evidence_update import EvidenceUpdate
from infrastructure.analytics.bayesian.entities.hypothesis_mapping import (
    HypothesisMapping,
)
from infrastructure.analytics.bayesian.entities.hypothesis_name_mapping import (
    HypothesisNameMapping,
)
from infrastructure.analytics.bayesian.entities.hypothesis_probability import (
    HypothesisProbability,
)
from infrastructure.analytics.bayesian.entities.probability_entry import (
    ProbabilityEntry,
)
from infrastructure.analytics.bayesian.entities.probability_mapping import (
    ProbabilityMapping,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from scipy import stats


class BayesianCalculator:
    """Bayesian probability calculator for intelligence analysis."""

    def __init__(self):
        pass  # Stateless service

    def calculate_posterior_probabilities(
        self,
        prior_probabilities: list[HypothesisMapping],
        likelihood_matrix: list[EvidenceLikelihoodMapping],
        hypothesis_names: list[HypothesisNameMapping],
        confidence_level: float = 0.95,
        default_likelihood: float = 0.5,
        min_marginal_likelihood: float = 1e-10,
        prior_sum_tolerance: float = 0.05,
    ) -> list[HypothesisProbability]:
        """Calculate posterior probabilities using proper Bayes' theorem."""
        # Validate inputs
        self._validate_inputs(
            prior_probabilities,
            likelihood_matrix,
            hypothesis_names,
            prior_sum_tolerance,
        )

        # Build working structures efficiently
        hypothesis_lookup = ProtocolSerializer.serialize_hypothesis_mapping(
            {mapping.hypothesis_id: mapping for mapping in prior_probabilities}
        )
        name_lookup = ProtocolSerializer.serialize_name_mapping(
            {
                mapping.hypothesis_id: mapping.hypothesis_name
                for mapping in hypothesis_names
            }
        )

        # Initialize current probabilities with priors
        current_probabilities = ProtocolSerializer.serialize_probability_mapping(
            {mapping.hypothesis_id: mapping.value for mapping in prior_probabilities}
        )

        # Apply evidence sequentially using proper Bayes' theorem
        for evidence_mapping in likelihood_matrix:
            current_probabilities = self._apply_bayes_update(
                current_probabilities,
                evidence_mapping,
                default_likelihood,
                min_marginal_likelihood,
            )

        # Build results with confidence intervals
        results = []
        for hypothesis_id, posterior_prob in current_probabilities.items():
            prior_prob = hypothesis_lookup[hypothesis_id].value
            hypothesis_name = name_lookup.get(hypothesis_id, hypothesis_id)

            # Calculate confidence intervals using Beta distribution
            ci_lower, ci_upper = self._calculate_confidence_interval(
                posterior_prob, confidence_level
            )

            # Calculate combined likelihood for this hypothesis
            combined_likelihood = self._calculate_combined_likelihood(
                hypothesis_id, likelihood_matrix, default_likelihood
            )

            results.append(
                HypothesisProbability(
                    hypothesis_id=hypothesis_id,
                    hypothesis_name=hypothesis_name,
                    prior_probability=prior_prob,
                    likelihood=combined_likelihood,
                    posterior_probability=posterior_prob,
                    confidence_interval_lower=ci_lower,
                    confidence_interval_upper=ci_upper,
                )
            )

        # Sort by posterior probability (descending)
        return sorted(results, key=lambda h: h.posterior_probability, reverse=True)

    def update_with_new_evidence(
        self, current_analysis: BayesianAnalysisResult, new_evidence: EvidenceUpdate
    ) -> BayesianAnalysisResult:
        """Update existing Bayesian analysis with new evidence."""
        # Combine all evidence
        all_evidence = current_analysis.evidence_used + [new_evidence]

        # Convert to likelihood matrix format
        likelihood_matrix = [
            EvidenceLikelihoodMapping(
                evidence_id=evidence.evidence_id,
                hypothesis_likelihoods=evidence.likelihood_mapping,
            )
            for evidence in all_evidence
        ]

        # Extract hypothesis data
        hypothesis_names = [
            HypothesisNameMapping(
                hypothesis_id=h.hypothesis_id, hypothesis_name=h.hypothesis_name
            )
            for h in current_analysis.hypotheses
        ]

        # Use original priors for full recalculation
        original_priors = [
            HypothesisMapping(hypothesis_id=h.hypothesis_id, value=h.prior_probability)
            for h in current_analysis.hypotheses
        ]

        # Recalculate with all evidence
        updated_hypotheses = self.calculate_posterior_probabilities(
            original_priors, likelihood_matrix, hypothesis_names
        )

        return BayesianAnalysisResult(
            analysis_id=current_analysis.analysis_id,
            hypotheses=updated_hypotheses,
            evidence_used=all_evidence,
            total_evidence_count=len(all_evidence),
            most_probable_hypothesis=updated_hypotheses[0],
            probability_distribution_entropy=self._calculate_entropy(
                updated_hypotheses
            ),
            convergence_status=self._assess_convergence(
                current_analysis.hypotheses, updated_hypotheses
            ),
        )

    def _apply_bayes_update(
        self,
        prior_probabilities: ProbabilityMapping,
        evidence_mapping: EvidenceLikelihoodMapping,
        default_likelihood: float,
        min_marginal_likelihood: float,
    ) -> ProbabilityMapping:
        """Apply single evidence update using proper Bayes' theorem."""
        # Build likelihood lookup for this evidence
        likelihood_lookup = {
            mapping.hypothesis_id: mapping.value
            for mapping in evidence_mapping.hypothesis_likelihoods
        }

        # Calculate marginal likelihood P(E) = Σ P(E|Hi) * P(Hi)
        marginal_likelihood = sum(
            likelihood_lookup.get(hypothesis_id, default_likelihood) * prior_prob
            for hypothesis_id, prior_prob in prior_probabilities.items()
        )

        # Avoid division by zero
        if marginal_likelihood <= 0:
            marginal_likelihood = min_marginal_likelihood

        # Calculate posteriors: P(Hi|E) = P(E|Hi) * P(Hi) / P(E)
        posterior_entries = []
        for hypothesis_id, prior_prob in prior_probabilities.items():
            likelihood = likelihood_lookup.get(hypothesis_id, default_likelihood)
            posterior = (likelihood * prior_prob) / marginal_likelihood
            posterior_entries.append(
                ProbabilityEntry(hypothesis_id=hypothesis_id, probability=posterior)
            )

        # Normalize to ensure probabilities sum to exactly 1.0
        total = sum(entry.probability for entry in posterior_entries)
        if total > 0:
            posterior_entries = [
                ProbabilityEntry(
                    hypothesis_id=entry.hypothesis_id,
                    probability=entry.probability / total,
                )
                for entry in posterior_entries
            ]

        return ProbabilityMapping(entries=posterior_entries)

    def _calculate_confidence_interval(
        self, probability: float, confidence_level: float, n_equiv: int = 100
    ) -> ConfidenceInterval:
        """Calculate confidence interval for probability using Beta distribution."""
        # Use Beta distribution with method of moments
        alpha = probability * n_equiv + 1
        beta = (1 - probability) * n_equiv + 1

        lower, upper = stats.beta.interval(confidence_level, alpha, beta)
        return ConfidenceInterval(lower_bound=lower, upper_bound=upper)

    def _calculate_combined_likelihood(
        self,
        hypothesis_id: str,
        likelihood_matrix: list[EvidenceLikelihoodMapping],
        default_likelihood: float,
    ) -> float:
        """Calculate combined likelihood for hypothesis across all evidence."""
        likelihoods = []

        for evidence_mapping in likelihood_matrix:
            # Find likelihood for this hypothesis in this evidence
            likelihood = default_likelihood
            for mapping in evidence_mapping.hypothesis_likelihoods:
                if mapping.hypothesis_id == hypothesis_id:
                    likelihood = mapping.value
                    break
            likelihoods.append(likelihood)

        if not likelihoods:
            return default_likelihood

        # Use geometric mean for combined likelihood
        return float(np.power(np.prod(likelihoods), 1.0 / len(likelihoods)))

    def _calculate_entropy(self, hypotheses: list[HypothesisProbability]) -> float:
        """Calculate Shannon entropy of probability distribution."""
        probabilities = [
            h.posterior_probability for h in hypotheses if h.posterior_probability > 0
        ]

        if not probabilities:
            return 0.0

        return float(-np.sum([p * np.log2(p) for p in probabilities]))

    def _assess_convergence(
        self,
        previous_hypotheses: list[HypothesisProbability],
        current_hypotheses: list[HypothesisProbability],
        converged_threshold: float = 0.01,
        converging_threshold: float = 0.05,
    ) -> str:
        """Assess whether probabilities are converging."""
        if len(previous_hypotheses) != len(current_hypotheses):
            return "inconsistent"

        # Build lookup for efficient comparison
        prev_lookup = {
            h.hypothesis_id: h.posterior_probability for h in previous_hypotheses
        }

        changes = []
        for curr_h in current_hypotheses:
            if curr_h.hypothesis_id in prev_lookup:
                prev_prob = prev_lookup[curr_h.hypothesis_id]
                change = abs(curr_h.posterior_probability - prev_prob)
                changes.append(change)

        if not changes:
            return "unknown"

        max_change = max(changes)
        if max_change < converged_threshold:
            return "converged"
        elif max_change < converging_threshold:
            return "converging"
        else:
            return "changing"

    def _validate_inputs(
        self,
        prior_probabilities: list[HypothesisMapping],
        likelihood_matrix: list[EvidenceLikelihoodMapping],
        hypothesis_names: list[HypothesisNameMapping],
        prior_sum_tolerance: float,
    ) -> None:
        """Validate input parameters."""
        if not prior_probabilities:
            raise ValueError("Prior probabilities cannot be empty")

        if not hypothesis_names:
            raise ValueError("Hypothesis names cannot be empty")

        # Check that priors sum to approximately 1.0
        total_prior = sum(mapping.value for mapping in prior_probabilities)
        lower_bound = 1.0 - prior_sum_tolerance
        upper_bound = 1.0 + prior_sum_tolerance
        if not lower_bound <= total_prior <= upper_bound:
            raise ValueError(
                f"Prior probabilities must sum to 1.0 (±{prior_sum_tolerance}), got: {total_prior}"
            )

        # Check that all probabilities are valid
        for mapping in prior_probabilities:
            if not 0 <= mapping.value <= 1:
                raise ValueError(f"Invalid prior probability: {mapping.value}")

        # Check that hypothesis sets match
        prior_ids = {mapping.hypothesis_id for mapping in prior_probabilities}
        name_ids = {mapping.hypothesis_id for mapping in hypothesis_names}

        if prior_ids != name_ids:
            raise ValueError("Hypothesis IDs must match between priors and names")

        # Validate likelihood matrix if provided
        if likelihood_matrix:
            for evidence_mapping in likelihood_matrix:
                if not evidence_mapping.evidence_id:
                    raise ValueError("Evidence ID cannot be empty")

                if not evidence_mapping.hypothesis_likelihoods:
                    raise ValueError(
                        f"Evidence {evidence_mapping.evidence_id} must have hypothesis likelihoods"
                    )

                # Check likelihood values are valid probabilities
                for likelihood_mapping in evidence_mapping.hypothesis_likelihoods:
                    if not 0 <= likelihood_mapping.value <= 1:
                        raise ValueError(
                            f"Invalid likelihood value: {likelihood_mapping.value}"
                        )
