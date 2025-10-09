
# User Story 6.1: Modelos Gráficos Probabilísticos

**Como científico de datos,** quiero poder definir, entrenar y realizar inferencia sobre modelos gráficos probabilísticos, como las Redes Bayesianas, para modelar relaciones de dependencia condicional en mis datos.

## Tareas:

- **Task 6.1.1:** Definir una `DistributionInterface` para representar distribuciones de probabilidad.
- **Task 6.1.2:** Crear una clase `BayesianNetwork` que implemente la `GraphicalModelInterface` y permita definir un grafo acíclico dirigido (DAG) de variables aleatorias.
- **Task 6.1.3:** Implementar un `InferenceService` para realizar inferencia exacta en modelos simples (ej. eliminación de variables).
- **Task 6.1.4:** Implementar el aprendizaje de parámetros a partir de datos (maximum likelihood estimation).
