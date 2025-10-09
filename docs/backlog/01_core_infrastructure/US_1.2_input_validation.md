
# User Story 1.2: Validación Automática de Entradas

**Como usuario de la biblioteca,** quiero que los modelos y transformadores validen automáticamente los datos de entrada para poder detectar errores de formato o dimensionalidad de manera temprana.

## Tareas:

- **Task 1.2.1:** Implementar un `ValidationService` para validaciones comunes (ej. forma de `ndarray`, consistencia de longitud entre X e y, número de características esperadas).
- **Task 1.2.2:** Integrar `ValidationService` en los métodos `fit` y `transform` de las implementaciones base para que se ejecute automáticamente.
- **Task 1.2.3:** Crear excepciones personalizadas y claras para errores de validación (ej. `InvalidInputError`, `DimensionMismatchError`).
