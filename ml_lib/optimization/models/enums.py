"""
Enumeraciones para el módulo de optimización de ml_lib.
"""

from enum import Enum, auto


class OptimizerType(Enum):
    """Tipos de optimizadores."""
    # First-order methods
    SGD = "sgd"
    MOMENTUM = "momentum"
    NESTEROV = "nesterov"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"
    ADAMAX = "adamax"
    NADAM = "nadam"
    ADADELTA = "adadelta"
    ADAMOD = "adamod"

    # Second-order methods
    NEWTON = "newton"
    BFGS = "bfgs"
    LBFGS = "lbfgs"
    CONJUGATE_GRADIENT = "cg"
    TRUST_REGION = "trust_region"

    # Specialized
    RPROP = "rprop"
    ASGD = "asgd"  # Averaged SGD


class SchedulerType(Enum):
    """Tipos de schedulers para learning rate."""
    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    CYCLIC = "cyclic"
    WARMUP = "warmup"
    LINEAR = "linear"


class ConvergenceCriterion(Enum):
    """Criterios de convergencia."""
    GRADIENT_NORM = "gradient_norm"
    LOSS_CHANGE = "loss_change"
    PARAMETER_CHANGE = "parameter_change"
    MAX_ITERATIONS = "max_iterations"
    RELATIVE_LOSS_CHANGE = "relative_loss_change"
    FUNCTION_TOLERANCE = "function_tolerance"
    GRADIENT_TOLERANCE = "gradient_tolerance"


class LineSearchMethod(Enum):
    """Métodos de búsqueda de línea."""
    ARMIJO = "armijo"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    BACKTRACKING = "backtracking"
    EXACT = "exact"
    NONE = "none"


class OptimizationStatus(Enum):
    """Estado de la optimización."""
    NOT_STARTED = auto()
    RUNNING = auto()
    CONVERGED = auto()
    MAX_ITERATIONS_REACHED = auto()
    DIVERGED = auto()
    STALLED = auto()
    FAILED = auto()
    STOPPED_BY_USER = auto()


class GradientEstimationMethod(Enum):
    """Métodos de estimación de gradientes."""
    EXACT = "exact"
    FINITE_DIFFERENCES = "finite_differences"
    FORWARD_MODE_AD = "forward_ad"
    REVERSE_MODE_AD = "reverse_ad"
    NUMERICAL = "numerical"


class ConstraintType(Enum):
    """Tipos de restricciones."""
    NONE = "none"
    BOX = "box"  # Lower and upper bounds
    LINEAR_EQUALITY = "linear_equality"
    LINEAR_INEQUALITY = "linear_inequality"
    NONLINEAR_EQUALITY = "nonlinear_equality"
    NONLINEAR_INEQUALITY = "nonlinear_inequality"


class RegularizationType(Enum):
    """Tipos de regularización."""
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"
    DROPOUT = "dropout"
    WEIGHT_DECAY = "weight_decay"


class UpdateRule(Enum):
    """Reglas de actualización de parámetros."""
    ADDITIVE = "additive"  # theta = theta - alpha * gradient
    MULTIPLICATIVE = "multiplicative"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    MOMENTUM_BASED = "momentum"
