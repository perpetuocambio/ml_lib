
# User Story 4.1: Optimizadores de Primer Orden

**Como científico de datos,** quiero poder entrenar mis modelos utilizando una variedad de optimizadores de primer orden estándar como SGD, Adam y sus variantes, para encontrar los que mejor funcionen para mi problema específico.

## Tareas:

- **Task 4.1.1:** Implementar un optimizador `SGD` desde cero, con soporte para momento y momento de Nesterov.
- **Task 4.1.2:** Implementar `Adam`, gestionando eficientemente sus estados internos (estimaciones de primer y segundo momento).
- **Task 4.1.3:** Implementar `AdaGrad` y `RMSprop`.
- **Task 4.1.4:** Asegurar que todos los optimizadores implementen la `OptimizerInterface` del core y gestionen un `dataclass` para su estado (`OptimizerState`).
- **Task 4.1.5:** Implementar `AdamW` con desacoplamiento de la regularización de pesos.
