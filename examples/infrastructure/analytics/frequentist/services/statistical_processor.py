"""Statistical analysis processor service."""

import numpy as np
from infrastructure.analytics.frequentist.entities.confidence_interval_result import (
    ConfidenceIntervalResult,
)
from infrastructure.analytics.frequentist.entities.correlation_analysis_result import (
    CorrelationAnalysisResult,
)
from infrastructure.analytics.frequentist.entities.correlation_pair_result import (
    CorrelationPairResult,
)
from infrastructure.analytics.frequentist.entities.descriptive_statistics_result import (
    DescriptiveStatisticsResult,
)
from infrastructure.analytics.frequentist.entities.hypothesis_test_result import (
    HypothesisTestResult,
)
from infrastructure.analytics.frequentist.entities.statistical_data_input import (
    StatisticalDataInput,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr


class StatisticalProcessor:
    """Statistical analysis processor for intelligence analysis."""

    def __init__(self):
        pass  # Stateless service

    def compute_descriptive_statistics(
        self, data_input: StatisticalDataInput
    ) -> DescriptiveStatisticsResult:
        """Compute comprehensive descriptive statistics for a variable."""
        data = np.array(data_input.data)

        # Remove NaN values for calculations
        clean_data = data[~np.isnan(data)]

        if len(clean_data) == 0:
            raise ValueError("No valid data points after removing NaN values")

        count = len(clean_data)
        mean_val = float(np.mean(clean_data))
        median_val = float(np.median(clean_data))
        std_val = float(np.std(clean_data, ddof=1))  # Sample standard deviation
        var_val = float(np.var(clean_data, ddof=1))  # Sample variance
        min_val = float(np.min(clean_data))
        max_val = float(np.max(clean_data))
        range_val = max_val - min_val

        # Quartiles
        q1 = float(np.percentile(clean_data, 25))
        q3 = float(np.percentile(clean_data, 75))
        iqr = q3 - q1

        # Shape statistics
        skewness = float(stats.skew(clean_data))
        kurt = float(stats.kurtosis(clean_data))

        # Coefficient of variation
        cv = std_val / mean_val if mean_val != 0 else float("inf")

        return DescriptiveStatisticsResult(
            variable_name=data_input.variable_name,
            count=count,
            mean=mean_val,
            median=median_val,
            std_deviation=std_val,
            variance=var_val,
            minimum=min_val,
            maximum=max_val,
            range_value=range_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurt,
            coefficient_of_variation=cv,
        )

    def compute_correlation_analysis(
        self,
        data_inputs: list[StatisticalDataInput],
        method: str = "pearson",
        significance_threshold: float = 0.05,
        confidence_level: float = 0.95,
    ) -> CorrelationAnalysisResult:
        """Compute correlation analysis between multiple variables."""
        if len(data_inputs) < 2:
            raise ValueError("At least 2 variables required for correlation analysis")

        valid_methods = ProtocolSerializer.serialize_method_set(
            {"pearson", "spearman", "kendall"}
        )
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        correlation_pairs = []
        significant_count = 0

        # Compute all pairwise correlations
        for i, var_x in enumerate(data_inputs):
            for j in range(i + 1, len(data_inputs)):
                var_y = data_inputs[j]

                # Align data and remove NaN pairs
                x_data = np.array(var_x.data)
                y_data = np.array(var_y.data)

                if len(x_data) != len(y_data):
                    raise ValueError(
                        f"Variables {var_x.variable_name} and {var_y.variable_name} have different lengths"
                    )

                # Remove NaN pairs
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]

                if len(x_clean) < 3:
                    raise ValueError(
                        f"Insufficient valid data points for correlation between {var_x.variable_name} and {var_y.variable_name}"
                    )

                # Compute correlation based on method
                if method == "pearson":
                    corr_coef, p_val = pearsonr(x_clean, y_clean)
                elif method == "spearman":
                    corr_coef, p_val = spearmanr(x_clean, y_clean)
                elif method == "kendall":
                    corr_coef, p_val = kendalltau(x_clean, y_clean)

                # Compute confidence interval for correlation
                ci_result = self._compute_correlation_confidence_interval(
                    corr_coef, len(x_clean), confidence_level
                )

                is_significant = p_val < significance_threshold
                if is_significant:
                    significant_count += 1

                correlation_pairs.append(
                    CorrelationPairResult(
                        variable_x=var_x.variable_name,
                        variable_y=var_y.variable_name,
                        correlation_coefficient=float(corr_coef),
                        p_value=float(p_val),
                        is_significant=is_significant,
                        confidence_interval_lower=ci_result.lower_bound,
                        confidence_interval_upper=ci_result.upper_bound,
                    )
                )

        return CorrelationAnalysisResult(
            analysis_method=method,
            correlation_pairs=correlation_pairs,
            total_variables=len(data_inputs),
            significant_correlations_count=significant_count,
            significance_threshold=significance_threshold,
        )

    def perform_t_test(
        self,
        sample1: StatisticalDataInput,
        sample2: StatisticalDataInput | None = None,
        population_mean: float | None = None,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> HypothesisTestResult:
        """Perform t-test (one-sample or two-sample)."""
        valid_alternatives = {"two-sided", "less", "greater"}
        if alternative not in valid_alternatives:
            raise ValueError(f"Alternative must be one of {valid_alternatives}")

        alpha = 1 - confidence_level

        # Clean data
        data1 = np.array(sample1.data)
        data1 = data1[~np.isnan(data1)]

        if sample2 is None and population_mean is None:
            raise ValueError("Either sample2 or population_mean must be provided")

        if sample2 is not None and population_mean is not None:
            raise ValueError("Provide either sample2 or population_mean, not both")

        # One-sample t-test
        if population_mean is not None:
            test_stat, p_val = stats.ttest_1samp(
                data1, population_mean, alternative=alternative
            )
            df = len(data1) - 1
            critical_val = (
                stats.t.ppf(1 - alpha / 2, df)
                if alternative == "two-sided"
                else stats.t.ppf(1 - alpha, df)
            )

            test_name = "One-sample t-test"
            null_hyp = f"Population mean = {population_mean}"
            alt_hyp = f"Population mean {self._get_alternative_symbol(alternative)} {population_mean}"

            # Effect size (Cohen's d)
            effect_size = abs(np.mean(data1) - population_mean) / np.std(data1, ddof=1)

        # Two-sample t-test
        else:
            data2 = np.array(sample2.data)
            data2 = data2[~np.isnan(data2)]
            test_stat, p_val = stats.ttest_ind(data1, data2, alternative=alternative)
            df = len(data1) + len(data2) - 2
            critical_val = (
                stats.t.ppf(1 - alpha / 2, df)
                if alternative == "two-sided"
                else stats.t.ppf(1 - alpha, df)
            )

            test_name = "Two-sample t-test"
            null_hyp = "Mean1 = Mean2"
            alt_hyp = f"Mean1 {self._get_alternative_symbol(alternative)} Mean2"

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(data1) - 1) * np.var(data1, ddof=1)
                    + (len(data2) - 1) * np.var(data2, ddof=1)
                )
                / df
            )
            effect_size = abs(np.mean(data1) - np.mean(data2)) / pooled_std

        is_significant = p_val < alpha
        reject_null = is_significant

        return HypothesisTestResult(
            test_name=test_name,
            null_hypothesis=null_hyp,
            alternative_hypothesis=alt_hyp,
            test_statistic=float(test_stat),
            p_value=float(p_val),
            critical_value=float(critical_val),
            confidence_level=confidence_level,
            is_significant=is_significant,
            reject_null=reject_null,
            effect_size=float(effect_size),
        )

    def perform_chi_square_test(
        self,
        observed_frequencies: np.ndarray,
        expected_frequencies: np.ndarray | None = None,
        confidence_level: float = 0.95,
    ) -> HypothesisTestResult:
        """Perform chi-square goodness of fit test."""
        if expected_frequencies is None:
            # Equal expected frequencies
            expected_frequencies = np.full_like(
                observed_frequencies, np.mean(observed_frequencies)
            )

        if len(observed_frequencies) != len(expected_frequencies):
            raise ValueError("Observed and expected frequencies must have same length")

        test_stat, p_val = stats.chisquare(observed_frequencies, expected_frequencies)
        df = len(observed_frequencies) - 1
        alpha = 1 - confidence_level
        critical_val = stats.chi2.ppf(1 - alpha, df)

        is_significant = p_val < alpha
        reject_null = is_significant

        # Effect size (Cramér's V)
        n = np.sum(observed_frequencies)
        effect_size = np.sqrt(test_stat / (n * (min(len(observed_frequencies)) - 1)))

        return HypothesisTestResult(
            test_name="Chi-square goodness of fit test",
            null_hypothesis="Observed frequencies match expected frequencies",
            alternative_hypothesis="Observed frequencies differ from expected frequencies",
            test_statistic=float(test_stat),
            p_value=float(p_val),
            critical_value=float(critical_val),
            confidence_level=confidence_level,
            is_significant=is_significant,
            reject_null=reject_null,
            effect_size=float(effect_size),
        )

    def _compute_correlation_confidence_interval(
        self, correlation: float, n: int, confidence_level: float
    ) -> ConfidenceIntervalResult:
        """Compute confidence interval for correlation coefficient using Fisher z-transformation."""
        if abs(correlation) >= 1.0:
            return ConfidenceIntervalResult(
                lower_bound=correlation, upper_bound=correlation
            )

        # Fisher z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))

        # Standard error
        se = 1 / np.sqrt(n - 3)

        # Z critical value
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Confidence interval in z space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se

        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        return ConfidenceIntervalResult(
            lower_bound=float(r_lower), upper_bound=float(r_upper)
        )

    def _get_alternative_symbol(self, alternative: str) -> str:
        """Get symbol for alternative hypothesis."""
        symbols = {
            "two-sided": "≠",
            "less": "<",
            "greater": ">",
        }
        return symbols[alternative]
