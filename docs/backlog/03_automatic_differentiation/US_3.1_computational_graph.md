
# User Story 3.1: Construcción de Grafo Computacional Dinámico

**Como desarrollador de modelos de Deep Learning,** quiero definir operaciones matemáticas sobre variables y que el sistema construya automáticamente un grafo computacional dinámico que registre estas operaciones y sus dependencias.

## Tareas:

- **Task 3.1.1:** Crear una clase `Variable` que encapsule un tensor (de `linalg`) y almacene su gradiente y la operación que la creó.
- **Task 3.1.2:** Diseñar una `OperationInterface` para todas las operaciones diferenciables (ej. `Add`, `Mul`, `MatMul`, `ReLU`). Cada operación debe conocer su `forward` y `backward` pass.
- **Task 3.1.3:** Sobrecargar los operadores de Python (`+`, `*`, `@`) en la clase `Variable` para que creen nodos de operación y los añadan al grafo de forma transparente.
- **Task 3.1.4:** Implementar un `GraphBuilderService` que gestione la creación y conexión de nodos en el grafo computacional.
