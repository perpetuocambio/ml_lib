"""ARIMA model result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ARIMAModelResult:
    """Result of ARIMA model fitting."""

    model_order_p: int
    model_order_d: int
    model_order_q: int
    aic_score: float
    bic_score: float
    log_likelihood: float
    parameters: list[float]
    parameter_names: list[str]
    residuals: list[float]
    fitted_values: list[float]
    is_stationary: bool
    ljung_box_p_value: float
    jarque_bera_p_value: float
    durbin_watson_statistic: float
