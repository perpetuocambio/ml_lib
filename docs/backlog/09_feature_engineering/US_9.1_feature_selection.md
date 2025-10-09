
# User Story 9.1: Selección Automática de Características

**Como científico de datos,** quiero un conjunto de herramientas automáticas para la selección de características que me ayuden a reducir la dimensionalidad, mejorar el rendimiento del modelo y reducir el tiempo de entrenamiento.

## Tareas:

- **Task 9.1.1:** Implementar una `SelectorInterface` que herede de `TransformerInterface`.
- **Task 9.1.2:** Implementar métodos de filtro: `SelectKBest` con diferentes funciones de puntuación (ej. chi-cuadrado, información mutua).
- **Task 9.1.3:** Implementar métodos de wrapper: `RecursiveFeatureElimination` (RFE).
- **Task 9.1.4:** Implementar métodos embebidos: `SelectFromModel` que utilice la `feature_importance_` de un modelo ya entrenado.
- **Task 9.1.5:** Investigar e implementar el algoritmo `Boruta` para una selección de características más robusta.
