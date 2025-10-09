# Product Backlog - ML Library

## Índice de User Stories

### 🎯 Épica 0: Code Quality & Foundation (NUEVA - ALTA PRIORIDAD)

Mejoras de calidad de código que deben completarse antes de continuar con nuevos módulos.

#### User Stories:

- **[US 0.1: Refactorización a Clases con Tipado Fuerte](00_code_quality/US_0.1_refactor_to_classes.md)** ⚡ CRÍTICO
  - Eliminar uso excesivo de diccionarios
  - Convertir strings mágicos en Enums
  - Crear clases dataclass bien tipadas
  - **Estimación:** 28 horas

- **[US 0.2: Seguridad de Tipos Completa](00_code_quality/US_0.2_type_safety.md)** ⚡ CRÍTICO
  - Eliminar uso innecesario de `Any`
  - Configurar mypy en modo strict
  - Implementar Generics correctamente
  - Usar numpy.typing
  - **Estimación:** 20 horas

- **[US 0.3: Validación y Robustez](00_code_quality/US_0.3_validation_and_robustness.md)** ⚡ CRÍTICO
  - Crear jerarquía de excepciones
  - Validación en todas las dataclasses
  - Decoradores de validación
  - ArrayValidator completo
  - **Estimación:** 18 horas

**Total Épica 0:** 66 horas (~8 días)

---

### ✅ Épica 1: Core Infrastructure (COMPLETADO)

Componentes fundamentales de la biblioteca.

#### User Stories:

- **[US 1.1: Interfaces Consistentes](01_core_infrastructure/US_1.1_consistent_interfaces.md)** ✅ COMPLETADO
  - EstimatorInterface, TransformerInterface
  - MetricInterface, OptimizerInterface
  - Uso de typing.Protocol

- **[US 1.2: Validación Automática de Entradas](01_core_infrastructure/US_1.2_input_validation.md)** ✅ COMPLETADO
  - ValidationService
  - Excepciones personalizadas
  - Integración en métodos fit/transform

- **[US 1.3: Manejo de Errores y Logging](01_core_infrastructure/US_1.3_error_handling_and_logging.md)** ✅ COMPLETADO
  - ErrorHandler con decoradores
  - LoggingService configurable
  - Logging estructurado

- **[US 1.4: Modelo Base](01_core_infrastructure/US_1.4_base_model.md)** ✅ COMPLETADO
  - BaseModel con metadatos
  - Tracking de estado
  - Serialización

---

### ✅ Épica 2: Linear Algebra (COMPLETADO)

Operaciones de álgebra lineal optimizadas.

#### User Stories:

- **[US 2.1: Operaciones Optimizadas](02_linear_algebra/US_2.1_optimized_operations.md)** ✅ COMPLETADO
  - Integración con BLAS/LAPACK
  - Operaciones básicas optimizadas
  - Cache inteligente

- **[US 2.2: Descomposiciones Matriciales](02_linear_algebra/US_2.2_matrix_decompositions.md)** ✅ COMPLETADO
  - LU, QR, SVD, Cholesky
  - Eigenvalue decomposition
  - Optimización de memoria

- **[US 2.3: Matrices Dispersas](02_linear_algebra/US_2.3_sparse_matrices.md)** ✅ COMPLETADO
  - Formatos CSR, CSC, COO
  - Operaciones optimizadas
  - Conversión entre formatos

---

### 🚧 Épica 3: Automatic Differentiation (EN DESARROLLO)

Sistema de diferenciación automática.

#### User Stories:

- **[US 3.1: Grafo Computacional](03_automatic_differentiation/US_3.1_computational_graph.md)** 🚧 EN PROGRESO
  - Construcción de grafo
  - Nodos y operaciones
  - Tape-based recording

- **[US 3.2: Backpropagation](03_automatic_differentiation/US_3.2_backpropagation.md)** 📋 PENDIENTE
  - Algoritmo de backprop
  - Cálculo eficiente de gradientes
  - Optimización de memoria

---

### 🚧 Épica 4: Numerical Optimization (EN DESARROLLO)

Algoritmos de optimización numérica.

#### User Stories:

- **[US 4.1: Optimizadores de Primer Orden](04_numerical_optimization/US_4.1_first_order_optimizers.md)** 🚧 EN PROGRESO
  - SGD, Momentum, Nesterov
  - Adam, RMSprop, AdaGrad

- **[US 4.2: Optimizadores de Segundo Orden](04_numerical_optimization/US_4.2_second_order_optimizers.md)** 📋 PENDIENTE
  - BFGS, L-BFGS
  - Newton methods
  - Conjugate gradient

- **[US 4.3: Schedulers de Learning Rate](04_numerical_optimization/US_4.3_lr_schedulers.md)** 📋 PENDIENTE
  - Step decay, exponential
  - Cosine annealing
  - Reduce on plateau

---

### 📋 Épica 5: Kernel Methods (PLANIFICADO)

Métodos de kernel y SVM.

#### User Stories:

- **[US 5.1: Funciones Kernel](05_kernel_methods/US_5.1_kernel_functions.md)** 📋 PENDIENTE
  - RBF, Polynomial, Linear
  - Custom kernels
  - Kernel composition

- **[US 5.2: Implementación SVM](05_kernel_methods/US_5.2_svm_implementation.md)** 📋 PENDIENTE
  - SVC, SVR
  - SMO algorithm
  - Kernel trick

- **[US 5.3: Gaussian Processes](05_kernel_methods/US_5.3_gaussian_processes.md)** 📋 PENDIENTE
  - GP regression
  - GP classification
  - Hyperparameter optimization

---

### 📋 Épica 6: Probabilistic Models (PLANIFICADO)

Modelos probabilísticos y bayesianos.

#### User Stories:

- **[US 6.1: Modelos Gráficos](06_probabilistic_models/US_6.1_graphical_models.md)** 📋 PENDIENTE
  - Bayesian networks
  - Markov chains
  - Factor graphs

- **[US 6.2: Métodos Monte Carlo](06_probabilistic_models/US_6.2_monte_carlo_methods.md)** 📋 PENDIENTE
  - MCMC
  - Gibbs sampling
  - Metropolis-Hastings

- **[US 6.3: Modelos de Variable Latente](06_probabilistic_models/US_6.3_latent_variable_models.md)** 📋 PENDIENTE
  - EM algorithm
  - Variational inference
  - VAE

---

### 📋 Épica 7: Deep Learning (PLANIFICADO)

Componentes de deep learning.

#### User Stories:

- **[US 7.1: Arquitecturas Core](07_deep_learning/US_7.1_core_architectures.md)** 📋 PENDIENTE
  - Dense, Conv, RNN layers
  - Activations
  - Loss functions

- **[US 7.2: Regularización Avanzada](07_deep_learning/US_7.2_advanced_regularization.md)** 📋 PENDIENTE
  - Dropout, BatchNorm
  - Weight decay
  - Early stopping

- **[US 7.3: Inicialización y Normalización](07_deep_learning/US_7.3_initialization_and_normalization.md)** 📋 PENDIENTE
  - Xavier, He initialization
  - LayerNorm, GroupNorm
  - Weight normalization

---

### 📋 Épica 8: Ensemble Learning (PLANIFICADO)

Métodos de ensemble.

#### User Stories:

- **[US 8.1: Algoritmos de Boosting](08_ensemble_learning/US_8.1_boosting_algorithms.md)** 📋 PENDIENTE
- **[US 8.2: Stacking y Meta-Learning](08_ensemble_learning/US_8.2_stacking_and_meta_learning.md)** 📋 PENDIENTE
- **[US 8.3: Random Forests](08_ensemble_learning/US_8.3_random_forests.md)** 📋 PENDIENTE

---

### 📋 Épica 9: Feature Engineering (PLANIFICADO)

Ingeniería de características.

#### User Stories:

- **[US 9.1: Selección de Features](09_feature_engineering/US_9.1_feature_selection.md)** 📋 PENDIENTE
- **[US 9.2: Extracción de Features](09_feature_engineering/US_9.2_feature_extraction.md)** 📋 PENDIENTE
- **[US 9.3: Síntesis Automática](09_feature_engineering/US_9.3_automated_feature_synthesis.md)** 📋 PENDIENTE

---

### 📋 Épica 10: Data Handling at Scale (PLANIFICADO)

Procesamiento de datos a gran escala.

#### User Stories:

- **[US 10.1: Procesamiento Out-of-Core](10_data_handling_at_scale/US_10.1_out_of_core_processing.md)** 📋 PENDIENTE
- **[US 10.2: Procesamiento Distribuido](10_data_handling_at_scale/US_10.2_distributed_processing.md)** 📋 PENDIENTE
- **[US 10.3: Optimización de Memoria](10_data_handling_at_scale/US_10.3_memory_optimization.md)** 📋 PENDIENTE

---

### 📋 Épica 11: Uncertainty Quantification (PLANIFICADO)

Cuantificación de incertidumbre.

#### User Stories:

- **[US 11.1: Incertidumbre Predictiva](11_uncertainty_quantification/US_11.1_predictive_uncertainty.md)** 📋 PENDIENTE
- **[US 11.2: Incertidumbre Ensemble](11_uncertainty_quantification/US_11.2_ensemble_based_uncertainty.md)** 📋 PENDIENTE
- **[US 11.3: Bayesian Deep Learning](11_uncertainty_quantification/US_11.3_bayesian_deep_learning.md)** 📋 PENDIENTE

---

### 📋 Épica 12: Time Series (PLANIFICADO)

Modelado de series temporales.

#### User Stories:

- **[US 12.1: Modelos Clásicos](12_time_series/US_12.1_classical_models.md)** 📋 PENDIENTE
- **[US 12.2: Deep Learning para TS](12_time_series/US_12.2_deep_learning_for_ts.md)** 📋 PENDIENTE
- **[US 12.3: Forecasting Avanzado](12_time_series/US_12.3_advanced_forecasting.md)** 📋 PENDIENTE

---

### 📋 Épica 13: Reinforcement Learning (PLANIFICADO)

Aprendizaje por refuerzo.

#### User Stories:

- **[US 13.1: Entorno RL](13_reinforcement_learning/US_13.1_rl_environment.md)** 📋 PENDIENTE

---

## Priorización y Roadmap

### Sprint 0: Code Quality Foundation (PRÓXIMO) ⚡

**Duración:** 2 semanas
**Objetivo:** Establecer estándares de calidad antes de continuar

- US 0.1: Refactorización a Clases (28h)
- US 0.2: Seguridad de Tipos (20h)
- US 0.3: Validación y Robustez (18h)

**Total:** 66 horas

### Sprint 1: Automatic Differentiation

**Duración:** 2 semanas
**Objetivo:** Completar autograd

- US 3.1: Grafo Computacional
- US 3.2: Backpropagation

### Sprint 2: Numerical Optimization

**Duración:** 2 semanas
**Objetivo:** Completar optimizadores

- US 4.1: Optimizadores de Primer Orden
- US 4.2: Optimizadores de Segundo Orden
- US 4.3: Schedulers

### Sprints Futuros

Los sprints posteriores se planificarán según prioridades del negocio y dependencies técnicas.

---

## Leyenda

- ⚡ **CRÍTICO**: Debe completarse inmediatamente
- ✅ **COMPLETADO**: Implementado y testeado
- 🚧 **EN PROGRESO**: Actualmente en desarrollo
- 📋 **PENDIENTE**: Planificado pero no iniciado
- 🔴 **BLOQUEADO**: Bloqueado por dependencias

---

## Métricas de Progreso

### Por Épica

| Épica | User Stories | Completadas | En Progreso | Pendientes | % Completado |
|-------|--------------|-------------|-------------|------------|--------------|
| 0: Code Quality | 3 | 0 | 0 | 3 | 0% |
| 1: Core | 4 | 4 | 0 | 0 | 100% |
| 2: Linalg | 3 | 3 | 0 | 0 | 100% |
| 3: Autograd | 2 | 0 | 1 | 1 | 0% |
| 4: Optimization | 3 | 0 | 1 | 2 | 0% |
| 5: Kernels | 3 | 0 | 0 | 3 | 0% |
| 6: Probabilistic | 3 | 0 | 0 | 3 | 0% |
| 7: Deep Learning | 3 | 0 | 0 | 3 | 0% |
| 8: Ensemble | 3 | 0 | 0 | 3 | 0% |
| 9: Feature Eng | 3 | 0 | 0 | 3 | 0% |
| 10: Data Scale | 3 | 0 | 0 | 3 | 0% |
| 11: Uncertainty | 3 | 0 | 0 | 3 | 0% |
| 12: Time Series | 3 | 0 | 0 | 3 | 0% |
| 13: RL | 1 | 0 | 0 | 1 | 0% |

### Total

- **Total User Stories:** 41
- **Completadas:** 7 (17%)
- **En Progreso:** 2 (5%)
- **Pendientes:** 32 (78%)

---

## Notas

- **Épica 0 (Code Quality)** debe completarse ANTES de continuar con nuevas features
- Las estimaciones son aproximadas y pueden ajustarse
- Cada US debe tener tests, documentación y code review antes de considerarse completa
- Se priorizan épicas según valor de negocio y dependencias técnicas

---

**Última actualización:** 2025-10-09
