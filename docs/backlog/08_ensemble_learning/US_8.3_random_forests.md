
# User Story 8.3: Random Forests Optimizados

**Como practicante de Machine Learning,** quiero una implementación eficiente y paralelizable de Random Forests para tareas de clasificación y regresión, que sea rápida y escale a grandes conjuntos de datos.

## Tareas:

- **Task 8.3.1:** Implementar un `DecisionTree` eficiente, con optimizaciones en la búsqueda de los mejores splits.
- **Task 8.3.2:** Crear los estimadores `RandomForestClassifier` y `RandomForestRegressor`.
- **Task 8.3.3:** Implementar el `bagging` (Bootstrap Aggregating) de los árboles.
- **Task 8.3.4:** Utilizar el `ParallelService` del módulo `utils` para construir los árboles en paralelo.
- **Task 8.3.5:** Implementar el cálculo de la importancia de características (feature importance) basado en la impureza de Gini o la permutación.
- **Task 8.3.6:** Implementar `ExtremelyRandomizedTrees` (ExtraTrees) como una variante.
