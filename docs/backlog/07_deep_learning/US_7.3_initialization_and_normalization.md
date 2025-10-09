
# User Story 7.3: Inicialización y Normalización de Pesos

**Como investigador,** quiero tener control total sobre las estrategias de inicialización de pesos y los mecanismos de normalización para poder experimentar y asegurar un flujo de gradientes estable durante el entrenamiento de redes profundas.

## Tareas:

- **Task 7.3.1:** Crear un `WeightInitializationHandler` con una `InitializerInterface`.
- **Task 7.3.2:** Implementar estrategias de inicialización estándar: `Xavier/Glorot` (uniforme y normal) y `He` (uniforme y normal).
- **Task 7.3.3:** Implementar `OrthogonalInitialization`.
- **Task 7.3.4:** Permitir que las capas (`DenseLayer`, `Conv2DLayer`) acepten un objeto inicializador en su constructor.
- **Task 7.3.5:** Implementar conexiones residuales (`Residual Connections`) como un componente o capa para facilitar el entrenamiento de redes muy profundas.
