
# User Story 2.1: Operaciones Matriciales Optimizadas

**Como científico de datos,** quiero realizar operaciones matriciales básicas (multiplicación, transposición, inversión) de forma optimizada para no tener cuellos de botella computacionales en mis algoritmos de ML.

## Tareas:

- **Task 2.1.1:** Implementar una clase `Matrix` que encapsule `numpy.ndarray` y ofrezca una API extendida y fluida.
- **Task 2.1.2:** Crear un `BLASService` que actúe como wrapper de operaciones BLAS/LAPACK (vía NumPy/SciPy) para garantizar el uso de las implementaciones más rápidas disponibles.
- **Task 2.1.3:** Investigar e implementar algoritmos de multiplicación de matrices cache-friendly (ej. `blocking` o `tiling`).
- **Task 2.1.4:** Añadir un `MemoryLayoutHandler` para gestionar y optimizar el acceso a datos con `memory layouts` C-contiguous vs F-contiguous.
