
# User Story 2.3: Soporte para Matrices Dispersas

**Como usuario de la biblioteca,** quiero poder trabajar con datos dispersos de manera eficiente para ahorrar memoria y acelerar el cómputo en problemas con gran cantidad de características vacías (ej. NLP o sistemas de recomendación).

## Tareas:

- **Task 2.3.1:** Diseñar una clase `SparseMatrix` que sea compatible con los formatos estándar de SciPy (CSR, CSC, COO).
- **Task 2.3.2:** Implementar un `SparseService` que adapte las operaciones del `linalg` (multiplicación, etc.) para que operen eficientemente sobre matrices dispersas.
- **Task 2.3.3:** Asegurar que las interfaces de los estimadores y transformadores puedan aceptar tanto matrices densas como dispersas de forma transparente.
