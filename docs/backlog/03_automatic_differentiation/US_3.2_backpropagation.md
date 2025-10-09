
# User Story 3.2: Cálculo de Gradientes con Backpropagation

**Como desarrollador de modelos de Deep Learning,** quiero poder calcular los gradientes de una variable de salida (ej. la pérdida) con respecto a cualquier otra variable de entrada (ej. los pesos del modelo) de forma automática y eficiente utilizando el modo reverso (backpropagation).

## Tareas:

- **Task 3.2.1:** Implementar un método `backward()` en la clase `Variable` que inicie el proceso de retropropagación.
- **Task 3.2.2:** Implementar un `BackwardService` que realice un recorrido topológico inverso del grafo computacional, empezando desde la variable de salida.
- **Task 3.2.3:** Durante el recorrido, invocar el método `_backward()` de cada `Operation` para calcular y propagar los gradientes a las variables de entrada correspondientes (regla de la cadena).
- **Task 3.2.4:** Escribir tests unitarios robustos que verifiquen la correctitud de los gradientes calculados comparándolos con una aproximación por diferencias finitas.
