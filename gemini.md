# Gemini Project: ml_lib

Este documento contiene las reglas del proyecto y el backlog de desarrollo para la biblioteca `ml_lib`.

## Reglas del Proyecto

1.  **Uso de Diccionarios Justificado:** El uso de `dict` debe estar muy justificado. Se prefiere el uso de `dataclasses` o estructuras de datos fuertemente tipadas para la configuración y el estado de los modelos.
2.  **Tipado Estricto:** Todo el código base debe estar completamente tipado (`Type Hinting`). No se aceptará código sin anotaciones de tipo.
3.  **Foco en `ml_lib`:** La aplicación principal y el objetivo de este proyecto es la biblioteca `ml_lib`. La calidad, el rendimiento y la robustez de la biblioteca son la máxima prioridad.
4.  **Rol de `ecoml_analyzer`:** La aplicación `ecoml_analyzer` es un consumidor de la biblioteca `ml_lib`. Su propósito es servir como un caso de uso práctico, para pruebas de integración y para demostrar las capacidades de la biblioteca. Su desarrollo es secundario al de `ml_lib`.
5.  **Documentación:** La documentación oficial del proyecto se encuentra en el directorio `@docs/`.

---

## Backlog de Desarrollo: ml_lib

### Épica 1: Infraestructura del Núcleo (Core)

> Establecer la base arquitectónica de la biblioteca, asegurando que sea robusta, extensible y fácil de usar.

**User Story 1.1:** Como desarrollador de la biblioteca, quiero interfaces base consistentes para todos los componentes (estimadores, transformadores, etc.) para garantizar la interoperabilidad y un diseño predecible.

- **Task 1.1.1:** Definir la `EstimatorInterface` (ABC) con métodos `fit`, `predict`, `get_params`, `set_params`.
- **Task 1.1.2:** Definir la `TransformerInterface` (ABC) con métodos `fit`, `transform`, `fit_transform`.
- **Task 1.1.3:** Definir la `MetricInterface` (ABC) para métricas de evaluación.
- **Task 1.1.4:** Definir la `OptimizerInterface` (ABC) para los algoritmos de optimización.
- **Task 1.1.5:** Usar `typing.Protocol` donde el subtipado estructural sea más apropiado que las ABC.

**User Story 1.2:** Como usuario de la biblioteca, quiero que los modelos y transformadores validen automáticamente los datos de entrada para poder detectar errores de formato o dimensionalidad de manera temprana.

- **Task 1.2.1:** Implementar un `ValidationService` para validaciones comunes (ej. forma de `ndarray`, consistencia de longitud entre X e y).
- **Task 1.2.2:** Integrar `ValidationService` en los métodos `fit` y `transform` de las implementaciones base.
- **Task 1.2.3:** Crear excepciones personalizadas para errores de validación (`InvalidInputError`, `DimensionMismatchError`).

**User Story 1.3:** Como desarrollador de la biblioteca, necesito un sistema centralizado de manejo de errores y logging para diagnosticar problemas de manera eficiente.

- **Task 1.3.1:** Implementar un `ErrorHandler` con decoradores para capturar y registrar excepciones en puntos críticos.
- **Task 1.3.2:** Configurar un `LoggingService` que pueda ser inyectado en diferentes componentes.
- **Task 1.3.3:** Definir una jerarquía de excepciones personalizadas para la biblioteca.

**User Story 1.4:** Como desarrollador de la biblioteca, quiero un modelo base que gestione el estado y los metadatos comunes a todos los estimadores.

- **Task 1.4.1:** Crear un `BaseModel` (posiblemente un `dataclass`) que incluya metadatos como `name`, `version` y estado como `is_fitted`.
- **Task 1.4.2:** Implementar el método `check_is_fitted` para ser usado antes de `predict` o `transform`.

### Épica 2: Álgebra Lineal de Alto Rendimiento (linalg)

> Construir un módulo de álgebra lineal que sea rápido, eficiente en el uso de memoria y que sirva como base para todos los algoritmos numéricos.

**User Story 2.1:** Como científico de datos, quiero realizar operaciones matriciales básicas (multiplicación, transposición, inversión) de forma optimizada para no tener cuellos de botella en mis algoritmos.

- **Task 2.1.1:** Implementar una clase `Matrix` que encapsule `numpy.ndarray` pero con una API extendida.
- **Task 2.1.2:** Crear un `BLASService` que actúe como wrapper de operaciones BLAS/LAPACK (vía NumPy/SciPy) para garantizar el uso de implementaciones optimizadas.
- **Task 2.1.3:** Implementar algoritmos de multiplicación de matrices cache-friendly (ej. `blocking`).
- **Task 2.1.4:** Añadir soporte para diferentes `memory layouts` (C-contiguous vs F-contiguous) a través de un `MemoryLayoutHandler`.

**User Story 2.2:** Como investigador de ML, necesito acceso a descomposiciones matriciales fundamentales (SVD, QR, Cholesky) para implementar algoritmos avanzados.

- **Task 2.2.1:** Implementar `QRDecompositionService` utilizando Householder o Givens.
- **Task 2.2.2:** Implementar `SVDService`, incluyendo variantes como `randomized SVD` para matrices grandes.
- **Task 2.2.3:** Implementar `CholeskyDecompositionService` para matrices simétricas definidas positivas.
- **Task 2.2.4:** Crear interfaces claras (`DecompositionInterface`) para cada tipo de descomposición.

**User Story 2.3:** Como usuario de la biblioteca, quiero poder trabajar con datos dispersos de manera eficiente para ahorrar memoria y cómputo.

- **Task 2.3.1:** Diseñar una `SparseMatrix` class compatible con los formatos de SciPy (CSR, CSC).
- **Task 2.3.2:** Implementar un `SparseService` que adapte las operaciones del `linalg` para matrices dispersas.

### Épica 3: Diferenciación Automática (autograd)

> Desarrollar un motor de diferenciación automática desde cero para entender y controlar el proceso de cálculo de gradientes.

**User Story 3.1:** Como desarrollador de modelos de DL, quiero definir operaciones matemáticas sobre variables y que el sistema construya automáticamente un grafo computacional.

- **Task 3.1.1:** Crear una clase `Variable` que almacene un valor y su gradiente.
- **Task 3.1.2:** Implementar un `ComputationalGraph` que registre las operaciones y dependencias.
- **Task 3.1.3:** Diseñar una `OperationInterface` para todas las operaciones diferenciables (ej. `Add`, `Mul`, `Sin`).

**User Story 3.2:** Como desarrollador de modelos de DL, quiero calcular gradientes de una variable de salida con respecto a cualquier variable de entrada utilizando el modo reverso (backpropagation).

- **Task 3.2.1:** Implementar el método `backward()` en la clase `Variable`.
- **Task 3.2.2:** Implementar la lógica de `backward_service` que recorra el grafo computacional en orden topológico inverso.
- **Task 3.2.3:** Para cada operación, implementar su `_backward` pass correspondiente que propague el gradiente.
- **Task 3.2.4:** Escribir tests unitarios para verificar la correctitud de los gradientes con diferencias finitas.

### Épica 4: Optimización Numérica (optimization)

> Implementar una suite de optimizadores numéricos clásicos y modernos para entrenar los modelos de la biblioteca.

**User Story 4.1:** Como científico de datos, quiero poder entrenar mis modelos utilizando optimizadores de primer orden estándar como SGD, Adam y sus variantes.

- **Task 4.1.1:** Implementar `SGD` con soporte para momento y Nesterov.
- **Task 4.1.2:** Implementar `Adam` y `RMSprop` desde cero, gestionando sus estados internos (`optimizer_state`).
- **Task 4.1.3:** Crear un `LearningRateHandler` para implementar schedulers (ej. step decay, exponential decay).
- **Task 4.1.4:** Todos los optimizadores deben implementar la `OptimizerInterface` del core.

**User Story 4.2:** Como investigador, necesito acceso a optimizadores de segundo orden para problemas que se beneficien de información de la curvatura.

- **Task 4.2.1:** Implementar el método de Newton-Raphson.
- **Task 4.2.2:** Implementar una versión de L-BFGS, almacenando un histórico limitado de gradientes.
- **Task 4.2.3:** Crear una interfaz `SecondOrderOptimizerInterface` que requiera el Hessiano o una aproximación.

### Épica 5: Modelos de Redes Neuronales (neural)

> Construir un framework para crear y entrenar redes neuronales, utilizando los módulos `linalg`, `autograd` y `optimization`.

**User Story 5.1:** Como practicante de ML, quiero poder definir arquitecturas de redes neuronales apilando capas de forma secuencial.

- **Task 5.1.1:** Crear una `LayerInterface` abstracta.
- **Task 5.1.2:** Implementar capas densas (`DenseLayer`), de activación (`ActivationLayer`), y de regularización (`Dropout`, `BatchNormalization`).
- **Task 5.1.3:** Implementar una clase `NeuralNetwork` o `Sequential` que gestione una lista de capas.
- **Task 5.1.4:** Implementar el `forward_pass_handler` que propague la entrada a través de las capas.

**User Story 5.2:** Como practicante de ML, quiero entrenar mis redes neuronales utilizando el `fit` API estándar, especificando un optimizador y una función de pérdida.

- **Task 5.2.1:** Implementar una `LossInterface` y clases para `MeanSquaredError` y `CrossEntropyLoss`.
- **Task 5.2.2:** Modificar el método `fit` del modelo de red neuronal para que ejecute el bucle de entrenamiento (forward, loss, backward, optimizer step).
- **Task 5.2.3:** Integrar el `BackpropagationService` que utiliza el motor de `autograd`.
- **Task 5.2.4:** Implementar un `WeightInitializationHandler` con estrategias como Xavier/Glorot y He.
