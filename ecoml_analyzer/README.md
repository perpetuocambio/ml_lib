# EcoML Analyzer

Una aplicación de demostración que muestra la funcionalidad completa de nuestra biblioteca de Machine Learning en el contexto de la ecología, específicamente para el análisis de datos ecológicos como abundancia de especies, diversidad y comunidades.

## Descripción

EcoML Analyzer es una aplicación que demuestra cómo nuestra biblioteca de ML personalizada puede ser utilizada para tareas de análisis ecológico. La aplicación incluye:

- **Lectura y preprocesamiento** de datos ecológicos (abundancia de especies, datos ambientales, etc.)
- **Análisis de diversidad ecológica** (índices de Shannon, Simpson, riqueza)
- **Análisis de comunidades** usando PCA, PCoA y clustering
- **Identificación de comunidades ecológicas** basadas en composición de especies
- **Modelos de distribución de especies** usando regresión logística
- **Análisis multi-especie** para identificar guildas o grupos funcionales
- **Visualización interactiva** de resultados ecológicos

## Características principales

### 1. Preprocesamiento de datos ecológicos
- Lectura de datos de abundancia de especies desde archivos CSV
- Normalización de datos (abundancia relativa, presencia/ausencia, transformaciones)
- Filtrado de especies por criterios de ocurrencia y abundancia
- Cálculo de índices de diversidad ecológica

### 2. Análisis ecológico avanzado
- Cálculo de índices de diversidad (Shannon, Simpson, riqueza)
- Análisis de componentes principales (PCA) y coordenadas principales (PCoA)
- Clustering de sitios basado en composición de especies
- Identificación de comunidades ecológicas
- Modelos de distribución de especies
- Análisis multi-especie y agrupamiento en guildas

### 3. Integración con nuestra biblioteca ML
- Uso de interfaces de estimadores personalizados
- Cálculos de álgebra lineal optimizados con nuestra biblioteca `linalg`
- Modelos personalizados que extienden nuestra arquitectura base
- Validación y manejo de errores consistentes

### 4. Visualización de resultados
- Heatmaps de abundancia de especies
- Gráficos de índices de diversidad
- Análisis de comunidades (PCA, PCoA)
- Curvas de rango-abundancia
- Resultados de clustering ecológico
- Identificación visual de comunidades

## Uso

### Instalación

La aplicación forma parte del paquete `ml-lib`:

```bash
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
uv pip install -e .
```

### Ejecución

Para ejecutar la demostración completa:

```bash
cd /src/perpetuocambio/ml_lib
PYTHONPATH=. python ecoml_analyzer/main.py
```

### Ejemplo simple

Para una ejecución rápida:

```bash
cd /src/perpetuocambio/ml_lib
PYTHONPATH=. python -c "
from ecoml_analyzer.main import run_simple_example
run_simple_example()
"
```

## Estructura del código

```
ecoml_analyzer/
├── __init__.py
├── data/                 # Lectura y preprocesamiento de datos ecológicos
│   └── __init__.py
├── analysis/             # Algoritmos de análisis ecológico
│   └── __init__.py
├── visualization/        # Visualización de resultados ecológicos
│   └── __init__.py
├── models/               # Modelos específicos (vacío por ahora)
│   └── __init__.py
├── utils/                # Utilidades (vacío por ahora)
│   └── __init__.py
└── main.py              # Punto de entrada de la aplicación
```

## Caso de uso demostrado

La aplicación demuestra cómo:

1. **Leer datos ecológicos** de diferentes formatos
2. **Preprocesar** datos usando técnicas ecológicas estándar
3. **Aplicar técnicas de ML** para análisis ecológicos
4. **Visualizar resultados** de manera efectiva
5. **Integrar** nuestra biblioteca personalizada con herramientas científicas

Específicamente, la demostración:
- Genera datos sintéticos de abundancia de especies con estructura de hábitat
- Calcula índices de diversidad ecológica
- Realiza análisis de comunidades usando PCA y PCoA
- Clustering de sitios para identificar agrupaciones ecológicas
- Identifica comunidades ecológicas basadas en similitud de especies
- Ajusta modelos de distribución de especies
- Realiza análisis multi-especie para identificar guildas
- Visualiza todos los resultados con gráficos ecológicos especializados

## Componentes de la biblioteca demostrados

- **Core**: Interfaces, modelos base, servicios de validación y logging
- **Linalg**: Operaciones de álgebra lineal y descomposiciones matriciales
- **Arquitectura modular**: Separación clara de responsabilidades
- **Tipado estricto**: Uso completo de type hints para seguridad
- **Manejo de errores**: Sistemas robustos de validación y manejo de errores

## Resultados esperados

La ejecución de la demostración completa:

1. Procesa una matriz de abundancia de 50 especies x 20 sitios
2. Calcula índices de diversidad ecológica
3. Realiza análisis de comunidades ecológicas (PCA, PCoA)
4. Clasifica sitios en comunidades ecológicas
5. Genera 6 visualizaciones ecológicas de alta calidad
6. Demuestra el uso de modelos personalizados con nuestra biblioteca

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias de mejora, especialmente en:

- Nuevos tipos de análisis ecológicos
- Soporte para más formatos de datos ecológicos
- Algoritmos de análisis más avanzados
- Mejores visualizaciones ecológicas
- Integración con paquetes ecológicos especializados (como vegan, AICcmodavg, etc.)

## Licencia

MIT