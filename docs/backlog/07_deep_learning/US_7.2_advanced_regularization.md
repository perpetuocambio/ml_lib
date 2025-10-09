
# User Story 7.2: Técnicas de Regularización Avanzadas

**Como ingeniero de Deep Learning,** necesito un conjunto de técnicas de regularización avanzadas para mejorar la generalización de mis modelos y prevenir el sobreajuste en conjuntos de datos complejos.

## Tareas:

- **Task 7.2.1:** Implementar `Dropout` como una capa que desactiva neuronas aleatoriamente durante el entrenamiento.
- **Task 7.2.2:** Implementar `BatchNormalization` como una capa que normaliza las activaciones dentro de un mini-batch.
- **Task 7.2.3:** Implementar `LayerNormalization` y `GroupNormalization` como alternativas a `BatchNorm`.
- **Task 7.2.4:** Integrar la regularización `L1/L2` (weight decay) directamente en los optimizadores o como una penalización en la función de pérdida.
- **Task 7.2.5:** Crear un `Callback` de `EarlyStopping` para detener el entrenamiento cuando una métrica de validación deja de mejorar.
