"""Time series analysis service."""

import numpy as np
from infrastructure.analytics.time_series.entities.adf_test_result import AdfTestResult
from infrastructure.analytics.time_series.entities.ar_model_fit_result import (
    ArModelFitResult,
)
from infrastructure.analytics.time_series.entities.arima_model_result import (
    ARIMAModelResult,
)
from infrastructure.analytics.time_series.entities.arima_order import (
    ARIMAOrder,
)
from infrastructure.analytics.time_series.entities.forecast_result import ForecastResult
from infrastructure.analytics.time_series.entities.stationarity_test_result import (
    StationarityTestResult,
)
from infrastructure.analytics.time_series.entities.time_series_data_input import (
    TimeSeriesDataInput,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from scipy import stats


class TimeSeriesAnalyzer:
    """Time series analysis service for intelligence analysis."""

    def __init__(self):
        pass  # Stateless service

    def test_stationarity(
        self,
        data_input: TimeSeriesDataInput,
        test_type: str = "adf",
        significance_level: float = 0.05,
        max_lags: int | None = None,
        trend: str = "c",
    ) -> StationarityTestResult:
        """Test time series for stationarity using simplified Augmented Dickey-Fuller test."""
        valid_tests = ProtocolSerializer.serialize_test_set({"adf"})
        if test_type not in valid_tests:
            raise ValueError(f"Test type must be one of {valid_tests}")

        valid_trends = ProtocolSerializer.serialize_trend_set({"c", "n"})
        if trend not in valid_trends:
            raise ValueError(f"Trend must be one of {valid_trends}")

        data = np.array(data_input.values)

        if max_lags is None:
            max_lags = int(12 * (len(data) / 100) ** 0.25)  # Rule of thumb

        # Simplified ADF test implementation
        adf_result = self._simplified_adf_test(data, max_lags, trend)

        # Critical values (approximate)
        critical_values = ProtocolSerializer.serialize_critical_values(
            {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        )

        # ADF: H0 = non-stationary, reject if test_stat < critical_value
        is_stationary = adf_result.test_statistic < critical_values["5%"]

        return StationarityTestResult(
            test_name="Simplified Augmented Dickey-Fuller Test",
            test_statistic=adf_result.test_statistic,
            p_value=adf_result.p_value,
            critical_values=ProtocolSerializer.serialize_float_mapping(critical_values),
            is_stationary=is_stationary,
            confidence_level=1 - significance_level,
            number_of_lags=adf_result.number_of_lags,
            trend_component=trend,
        )

    def fit_arima_model(
        self,
        data_input: TimeSeriesDataInput,
        order: ARIMAOrder,
        include_constant: bool = True,
        method: str = "lbfgs",
        maxiter: int = 1000,
    ) -> ARIMAModelResult:
        """Fit simplified ARIMA model to time series data."""
        p, d, q = order.p, order.d, order.q

        if p < 0 or d < 0 or q < 0:
            raise ValueError("ARIMA order parameters must be non-negative")

        if d > 2:
            raise ValueError("Differencing order (d) should not exceed 2")

        data = np.array(data_input.values)

        # Apply differencing
        diff_data = self._difference_series(data, d)

        # Fit simplified AR model (simplified ARIMA)
        if p > 0:
            ar_fit_result = self._fit_ar_model(diff_data, p, include_constant)
            ar_params = np.array(ar_fit_result.parameters)
            fitted_values = np.array(ar_fit_result.fitted_values)
            residuals = np.array(ar_fit_result.residuals)
        else:
            # Simple mean model
            ar_params = np.array([np.mean(diff_data)] if include_constant else [])
            fitted_values = np.full_like(diff_data, np.mean(diff_data))
            residuals = diff_data - fitted_values

        # Calculate information criteria
        n = len(residuals)
        mse = np.mean(residuals**2)
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(mse) + 1)

        k = len(ar_params)  # number of parameters
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)

        # Diagnostic tests
        is_stationary = self._test_residual_stationarity(residuals)
        ljung_box_p = self._ljung_box_test(residuals)
        jb_p_value = self._jarque_bera_test(residuals)
        dw_stat = self._durbin_watson_statistic(residuals)

        # Parameter names
        param_names = []
        if include_constant:
            param_names.append("const")
        for i in range(p):
            param_names.append(f"ar.L{i+1}")

        return ARIMAModelResult(
            model_order_p=order[0],
            model_order_d=order[1],
            model_order_q=order[2],
            aic_score=float(aic),
            bic_score=float(bic),
            log_likelihood=float(log_likelihood),
            parameters=ar_params.tolist()
            if hasattr(ar_params, "tolist")
            else list(ar_params),
            parameter_names=param_names,
            residuals=residuals.tolist()
            if hasattr(residuals, "tolist")
            else list(residuals),
            fitted_values=fitted_values.tolist()
            if hasattr(fitted_values, "tolist")
            else list(fitted_values),
            is_stationary=is_stationary,
            ljung_box_p_value=ljung_box_p,
            jarque_bera_p_value=jb_p_value,
            durbin_watson_statistic=float(dw_stat),
        )

    def auto_arima(
        self,
        data_input: TimeSeriesDataInput,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        information_criterion: str = "aic",
        stepwise: bool = True,
    ) -> ARIMAModelResult:
        """Automatically select best ARIMA model parameters."""
        valid_criteria = {"aic", "bic"}
        if information_criterion not in valid_criteria:
            raise ValueError(f"Information criterion must be one of {valid_criteria}")

        best_score = float("inf")
        best_model = None

        # Search ranges
        p_range = range(0, max_p + 1)
        d_range = range(0, max_d + 1)
        q_range = range(0, max_q + 1)

        if stepwise:
            # Stepwise search (simplified heuristic)
            candidate_orders = [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 1),
                (2, 1, 0),
                (0, 1, 2),
                (2, 1, 2),
            ]
            # Filter valid orders
            candidate_orders = [
                (p, d, q)
                for p, d, q in candidate_orders
                if p <= max_p and d <= max_d and q <= max_q
            ]
        else:
            # Grid search
            candidate_orders = [
                (p, d, q) for p in p_range for d in d_range for q in q_range
            ]

        for order in candidate_orders:
            try:
                model_result = self.fit_arima_model(data_input, order)

                score = (
                    model_result.aic_score
                    if information_criterion == "aic"
                    else model_result.bic_score
                )

                if score < best_score:
                    best_score = score
                    best_model = model_result

            except Exception:
                # Skip problematic model configurations
                continue

        if best_model is None:
            raise ValueError("Could not find suitable ARIMA model")

        return best_model

    def forecast(
        self,
        arima_result: ARIMAModelResult,
        data_input: TimeSeriesDataInput,
        forecast_horizon: int,
        confidence_level: float = 0.95,
        include_prediction_intervals: bool = False,
    ) -> ForecastResult:
        """Generate simple forecasts using trend extrapolation."""
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        data = np.array(data_input.values)

        # Simple forecasting: use last few values to estimate trend
        window_size = min(10, len(data))
        recent_data = data[-window_size:]

        # Linear trend estimation
        x = np.arange(len(recent_data))
        coeffs = np.polyfit(x, recent_data, 1)
        trend_slope = coeffs[0]
        last_value = data[-1]

        # Generate simple trend-based forecasts
        forecast_values = []
        for i in range(1, forecast_horizon + 1):
            forecast_val = last_value + trend_slope * i
            forecast_values.append(float(forecast_val))

        # Simple confidence intervals based on recent volatility
        residuals = np.array(arima_result.residuals)
        residual_std = np.std(residuals)
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        conf_lower = []
        conf_upper = []
        for i, forecast_val in enumerate(forecast_values):
            margin = z_score * residual_std * np.sqrt(i + 1)
            conf_lower.append(float(forecast_val - margin))
            conf_upper.append(float(forecast_val + margin))

        # Generate forecast dates if timestamps are available
        forecast_dates = None
        if data_input.timestamps is not None:
            forecast_dates = []
            # Simple date progression (assuming daily data)
            for i in range(1, forecast_horizon + 1):
                forecast_dates.append(f"T+{i}")

        # Prediction intervals (wider than confidence intervals)
        pred_lower = None
        pred_upper = None
        if include_prediction_intervals:
            pred_lower = []
            pred_upper = []
            for i, forecast_val in enumerate(forecast_values):
                pred_margin = z_score * residual_std * np.sqrt((i + 1) * 1.5)  # Wider
                pred_lower.append(float(forecast_val - pred_margin))
                pred_upper.append(float(forecast_val + pred_margin))

        method_name = f"Simplified_ARIMA({arima_result.model_order_p},{arima_result.model_order_d},{arima_result.model_order_q})"

        return ForecastResult(
            forecast_values=forecast_values,
            confidence_intervals_lower=conf_lower,
            confidence_intervals_upper=conf_upper,
            forecast_horizon=forecast_horizon,
            confidence_level=confidence_level,
            forecast_method=method_name,
            forecast_dates=forecast_dates,
            prediction_intervals_lower=pred_lower,
            prediction_intervals_upper=pred_upper,
        )

    def _durbin_watson_statistic(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        return float(dw_stat)

    def _simplified_adf_test(
        self, data: np.ndarray, max_lags: int, trend: str
    ) -> AdfTestResult:
        """Simplified Augmented Dickey-Fuller test implementation."""

        # First difference
        diff_data = np.diff(data)
        lagged_data = data[:-1]

        # Build regression matrix
        if trend == "c":
            # Include constant
            X = np.column_stack([np.ones(len(diff_data)), lagged_data])
        else:
            # No constant
            X = lagged_data.reshape(-1, 1)

        # Add lagged differences (simplified to 1 lag)
        if max_lags > 0 and len(diff_data) > 1:
            lag_diff = np.concatenate([[0], diff_data[:-1]])  # One lag
            X = np.column_stack([X, lag_diff])
            n_lags = 1
        else:
            n_lags = 0

        # OLS regression
        try:
            beta = np.linalg.lstsq(X, diff_data, rcond=None)[0]
            y_pred = X @ beta
            residuals = diff_data - y_pred

            # Test statistic (t-statistic for coefficient of lagged level)
            mse = np.mean(residuals**2)
            cov_matrix = mse * np.linalg.inv(X.T @ X)

            # Test statistic for lagged level coefficient
            coef_idx = 1 if trend == "c" else 0
            test_stat = beta[coef_idx] / np.sqrt(cov_matrix[coef_idx, coef_idx])

            # Approximate p-value (very rough approximation)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

        except np.linalg.LinAlgError:
            test_stat = 0.0
            p_value = 1.0

        return AdfTestResult(
            test_statistic=float(test_stat),
            p_value=float(p_value),
            number_of_lags=int(n_lags),
        )

    def _difference_series(self, data: np.ndarray, d: int) -> np.ndarray:
        """Apply differencing to time series."""
        if d == 0:
            return data

        result = data.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _fit_ar_model(
        self, data: np.ndarray, p: int, include_constant: bool
    ) -> ArModelFitResult:
        """Fit AR(p) model using OLS."""
        n = len(data)

        # Build lagged matrix
        X = np.zeros((n - p, p + (1 if include_constant else 0)))
        y = data[p:]

        col_idx = 0
        if include_constant:
            X[:, col_idx] = 1
            col_idx += 1

        for i in range(p):
            X[:, col_idx + i] = data[p - 1 - i : n - 1 - i]

        # OLS estimation
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            fitted = X @ beta
            residuals = y - fitted
        except np.linalg.LinAlgError:
            # Fallback to simple mean
            beta = np.array([np.mean(y)] if include_constant else [])
            fitted = np.full_like(y, np.mean(y))
            residuals = y - fitted

        return ArModelFitResult(
            parameters=beta.tolist() if hasattr(beta, "tolist") else list(beta),
            fitted_values=fitted.tolist()
            if hasattr(fitted, "tolist")
            else list(fitted),
            residuals=residuals.tolist()
            if hasattr(residuals, "tolist")
            else list(residuals),
        )

    def _test_residual_stationarity(self, residuals: np.ndarray) -> bool:
        """Simple test for residual stationarity."""
        # Use variance ratio test (simplified)
        n = len(residuals)
        if n < 20:
            return True  # Too few observations

        mid = n // 2
        var1 = np.var(residuals[:mid])
        var2 = np.var(residuals[mid:])

        # F-test for equal variances
        f_stat = var1 / var2 if var1 > var2 else var2 / var1
        return f_stat < 2.0  # Rough threshold

    def _ljung_box_test(self, residuals: np.ndarray) -> float:
        """Simplified Ljung-Box test for autocorrelation."""
        n = len(residuals)
        lags = min(10, n // 4)

        if lags < 1:
            return 1.0

        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, lags + 1):
            if lag >= n:
                break
            corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(corr**2)

        if not autocorrs:
            return 1.0

        # Ljung-Box statistic (simplified)
        lb_stat = n * (n + 2) * sum(ac / (n - i - 1) for i, ac in enumerate(autocorrs))

        # Approximate p-value using chi-square distribution
        p_value = 1 - stats.chi2.cdf(lb_stat, df=len(autocorrs))
        return float(p_value)

    def _jarque_bera_test(self, residuals: np.ndarray) -> float:
        """Jarque-Bera test for normality."""
        n = len(residuals)

        if n < 4:
            return 1.0

        # Calculate skewness and kurtosis
        mean_res = np.mean(residuals)
        std_res = np.std(residuals, ddof=1)

        skewness = np.mean(((residuals - mean_res) / std_res) ** 3)
        kurtosis = np.mean(((residuals - mean_res) / std_res) ** 4) - 3

        # JB statistic
        jb_stat = n * (skewness**2 / 6 + kurtosis**2 / 24)

        # p-value using chi-square distribution with 2 df
        p_value = 1 - stats.chi2.cdf(jb_stat, df=2)
        return float(p_value)
