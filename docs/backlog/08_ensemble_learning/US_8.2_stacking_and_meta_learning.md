
# User Story 8.2: Stacking y Meta-Aprendizaje

**Como competidor de Kaggle,** quiero poder combinar las predicciones de múltiples modelos diversos (stacking) utilizando un meta-modelo para mejorar la precisión y robustez de mis resultados.

## Tareas:

- **Task 8.2.1:** Crear un estimador `StackingEnsemble` que tome una lista de modelos base y un meta-modelo final.
- **Task 8.2.2:** Implementar la lógica de entrenamiento con predicciones out-of-fold para generar el conjunto de datos de entrenamiento para el meta-modelo y evitar fugas de datos.
- **Task 8.2.3:** Permitir diferentes estrategias para las predicciones, como `predict` y `predict_proba`.
- **Task 8.2.4:** Implementar `Blending`, una variante más simple de stacking que utiliza un set de validación separado.
