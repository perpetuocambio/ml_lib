
# User Story 1.4: Modelo Base para Estado y Metadatos

**Como desarrollador de la biblioteca,** quiero un modelo base que gestione el estado y los metadatos comunes a todos los estimadores para evitar la duplicación de código.

## Tareas:

- **Task 1.4.1:** Crear un `BaseModel` (usando un `dataclass`) que incluya metadatos comunes como `name`, `version` y el estado de entrenamiento `is_fitted`.
- **Task 1.4.2:** Implementar un método `check_is_fitted()` que lance una excepción (`NotFittedError`) si se intenta predecir o transformar con un modelo no entrenado.
- **Task 1.4.3:** Asegurar que todas las clases de estimadores hereden de este `BaseModel`.
