
# User Story 5.1: Funciones Kernel Flexibles

**Como científico de datos,** quiero tener acceso a una variedad de funciones kernel (Lineal, Polinómico, RBF, Sigmoide) y la capacidad de crear mis propias funciones personalizadas para usarlas en algoritmos kernelizados como SVM o Kernel PCA.

## Tareas:

- **Task 5.1.1:** Definir una `KernelInterface` que estandarice el cálculo de la matriz de Gram.
- **Task 5.1.2:** Implementar las funciones kernel estándar: `LinearKernel`, `PolynomialKernel`, `RBFKernel`, `SigmoidKernel`.
- **Task 5.1.3:** Crear un `KernelComputationService` para calcular eficientemente la matriz de Gram (Kernel Matrix).
- **Task 5.1.4:** Implementar un `KernelCacheHandler` para almacenar en caché los resultados de las evaluaciones del kernel y acelerar el entrenamiento.
