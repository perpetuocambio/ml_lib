
# User Story 6.2: Métodos de Monte Carlo por Cadenas de Markov (MCMC)

**Como investigador,** necesito poder muestrear de distribuciones de probabilidad complejas y posteriores para las cuales no hay una forma analítica cerrada, utilizando algoritmos MCMC.

## Tareas:

- **Task 6.2.1:** Implementar un `SamplingService` con una `SamplerInterface`.
- **Task 6.2.2:** Implementar el algoritmo de `Metropolis-Hastings` para muestrear de una distribución objetivo dada una función de propuesta.
- **Task 6.2.3:** Implementar el `Gibbs Sampling`, un caso especial de M-H para problemas multivariados donde las distribuciones condicionales son conocidas.
- **Task 6.2.4:** Investigar la implementación de `Hamiltonian Monte Carlo` (HMC) para un muestreo más eficiente en espacios de alta dimensión, utilizando información del gradiente.
