
# User Story 1.1: Interfaces Base Consistentes

**Como desarrollador de la biblioteca,** quiero interfaces base consistentes para todos los componentes (estimadores, transformadores, etc.) para garantizar la interoperabilidad y un diseño predecible.

## Tareas:

- **Task 1.1.1:** Definir la `EstimatorInterface` (ABC) con métodos `fit`, `predict`, `get_params`, `set_params`.
- **Task 1.1.2:** Definir la `TransformerInterface` (ABC) con métodos `fit`, `transform`, `fit_transform`.
- **Task 1.1.3:** Definir la `MetricInterface` (ABC) para métricas de evaluación.
- **Task 1.1.4:** Definir la `OptimizerInterface` (ABC) para los algoritmos de optimización.
- **Task 1.1.5:** Investigar y usar `typing.Protocol` donde el subtipado estructural sea más apropiado que las ABC para mayor flexibilidad.
