
# User Story 5.3: Modelado con Procesos Gaussianos

**Como investigador,** quiero poder modelar la incertidumbre en mis predicciones utilizando Procesos Gaussianos (GP) para regresión, con la capacidad de definir kernels de covarianza personalizados.

## Tareas:

- **Task 5.3.1:** Crear una clase `GaussianProcessRegressor` que implemente la `EstimatorInterface`.
- **Task 5.3.2:** Permitir que el `GaussianProcessRegressor` acepte cualquier objeto que cumpla la `KernelInterface` como función de covarianza.
- **Task 5.3.3:** Implementar la lógica de inferencia para predecir la media y la varianza en nuevos puntos.
- **Task 5.3.4:** Implementar la optimización de los hiperparámetros del kernel maximizando la verosimilitud marginal (marginal likelihood).
- **Task 5.3.5:** Investigar e implementar aproximaciones dispersas (Sparse GP) para mejorar la escalabilidad a conjuntos de datos más grandes.
