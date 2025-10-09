
# User Story 4.2: Optimizadores de Segundo Orden

**Como investigador,** necesito acceso a optimizadores de segundo orden para problemas complejos donde la información de la curvatura (Hessiano) es crucial para una convergencia más rápida y precisa.

## Tareas:

- **Task 4.2.1:** Implementar el método de Newton-Raphson, asumiendo que el Hessiano puede ser calculado.
- **Task 4.2.2:** Implementar una versión de L-BFGS (Limited-memory BFGS), que aproxime el Hessiano inverso almacenando un histórico limitado de actualizaciones de pesos y gradientes.
- **Task 4.2.3:** Crear una `SecondOrderOptimizerInterface` que herede de la `OptimizerInterface` base pero que pueda requerir el Hessiano o una función que calcule el producto Hessiano-vector.
- **Task 4.2.4:** Implementar el método de Gradiente Conjugado.
