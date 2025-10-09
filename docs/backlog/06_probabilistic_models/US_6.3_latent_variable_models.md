
# User Story 6.3: Modelos de Variables Latentes

**Como analista de datos,** quiero descubrir estructuras y patrones ocultos en mis datos utilizando modelos de variables latentes como Modelos de Mezcla Gaussiana (GMM) para clustering o Modelos Ocultos de Markov (HMM) para datos secuenciales.

## Tareas:

- **Task 6.3.1:** Implementar el algoritmo `Expectation-Maximization` (EM) como un servicio (`EMService`) general para ajustar modelos con variables latentes.
- **Task 6.3.2:** Crear una clase `GaussianMixtureModel` que utilice el `EMService` para encontrar los parámetros de las componentes gaussianas.
- **Task 6.3.3:** Implementar los algoritmos `forward-backward` y `Viterbi` para Modelos Ocultos de Markov (HMM).
- **Task 6.3.4:** Crear una clase `HiddenMarkovModel` para ajustar y predecir sobre secuencias de datos.
