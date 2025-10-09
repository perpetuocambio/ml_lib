
# User Story 4.3: Planificadores de Tasa de Aprendizaje (Learning Rate Schedulers)

**Como practicante de Deep Learning,** quiero poder ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento para mejorar la convergencia y evitar mínimos locales subóptimos.

## Tareas:

- **Task 4.3.1:** Diseñar una `SchedulerInterface` que pueda ser asociada a un optimizador.
- **Task 4.3.2:** Implementar un `LearningRateHandler` que aplique la lógica del scheduler al optimizador en cada época o iteración.
- **Task 4.3.3:** Implementar schedulers comunes: `StepLR` (decae en épocas específicas), `ExponentialLR` (decae exponencialmente), y `ReduceLROnPlateau` (decae cuando una métrica se estanca).
- **Task 4.3.4:** Implementar schedulers más avanzados como `CosineAnnealingLR`.
