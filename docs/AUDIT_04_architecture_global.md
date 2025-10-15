# AUDITORÍA: Arquitectura Global de ml_lib

**Fecha:** 2025-10-15
**Alcance:** Estructura completa del proyecto
**Enfoque:** Patrones arquitecturales, capas, dependencias

---

## RESUMEN EJECUTIVO

El proyecto `ml_lib` tiene **28 módulos** organizados por dominio técnico (diffusion, llm, linalg, etc.). Aunque la separación modular es buena, hay **problemas arquitecturales serios** que afectan mantenibilidad y escalabilidad.

### Hallazgos Críticos

1. **Sin arquitectura de capas clara** - Mezcla de dominio, aplicación e infraestructura
2. **Acoplamiento cruzado** - Módulos importando directamente de otros sin abstracciones
3. **Falta de bounded contexts** - No hay límites claros entre dominios
4. **Sin Dependency Injection** - Construcción manual de dependencias en código
5. **Código procedural con fachada OOP** - Servicios + entidades anémicas

---

## ESTRUCTURA ACTUAL

```
ml_lib/
├── autograd/
├── automl/
├── core/
│   ├── entities/
│   ├── ports/
│   ├── services/
│   └── use_cases/
├── database/
├── deployment/
├── diffusion/              ← ANALIZADO EN AUDIT_01, AUDIT_02
│   ├── config/
│   ├── docs/
│   ├── handlers/
│   ├── interfaces/
│   ├── models/
│   ├── services/           ← 24 servicios, GOD CLASSES
│   └── facade.py
├── ensemble/
├── fairness/
├── feature_engineering/
├── interpretability/
├── kernels/
├── linalg/
├── llm/
│   ├── clients/
│   ├── config/
│   ├── entities/
│   ├── providers/
│   └── services/
├── neural/
├── optimization/
├── optimization_numérica_avanzada/
├── performance/
├── plugin_system/
├── probabilistic/
├── reinforcement/
├── storage/
├── system/                 ← ANALIZADO EN AUDIT_03 (MAL UBICADO)
├── time_series/
├── uncertainty/
├── utils/
└── visualization/
```

---

## PROBLEMAS ARQUITECTURALES GLOBALES

### 1. SIN SEPARACIÓN DE CAPAS

**Problema:** Todo mezclado en cada módulo

**Lo que hay:**
```
diffusion/
  ├── services/       ← Mezcla Application + Domain + Infrastructure
  ├── models/         ← Solo Domain (anémicas)
  ├── handlers/       ← ¿Application? ¿Infrastructure?
  └── interfaces/     ← Algunos protocols
```

**Lo que debería ser:**
```
diffusion/
  ├── domain/              ← Entidades ricas, lógica de negocio
  │   ├── entities/
  │   ├── value_objects/
  │   ├── aggregates/
  │   └── domain_services/
  ├── application/         ← Use cases, orquestación
  │   ├── use_cases/
  │   └── dto/
  ├── infrastructure/      ← DB, APIs, filesystem
  │   ├── persistence/
  │   ├── external_apis/
  │   └── monitoring/
  └── interfaces/          ← Ports & Adapters
      ├── protocols/
      └── contracts/
```

**Tareas CRÍTICAS:**
1. Reorganizar cada módulo grande (diffusion, llm) en capas claras
2. Domain no debe importar de infrastructure
3. Application orquesta, no ejecuta lógica de negocio
4. Infrastructure implementa interfaces del domain

### 2. ACOPLAMIENTO DIRECTO ENTRE MÓDULOS

**Problema:** Importaciones directas sin abstracciones

**Ejemplos encontrados:**
```python
# diffusion → system (VIOLACIÓN)
from ml_lib.system.resource_monitor import ResourceMonitor

# diffusion → llm (acoplamiento concreto)
from ml_lib.llm.providers import OllamaProvider
from ml_lib.llm.clients import LLMClient

# Probablemente más...
```

**Consecuencias:**
- Imposible testear aisladamente
- Cambios en un módulo rompen otros
- No se pueden reemplazar implementaciones
- Módulos no reutilizables

**Solución:**
```python
# diffusion define interface
from diffusion.interfaces import IResourceMonitor, ILLMProvider

# infrastructure implementa
class SystemResourceMonitor(IResourceMonitor): ...
class OllamaLLMProvider(ILLMProvider): ...

# DI container conecta
container.register(IResourceMonitor, SystemResourceMonitor)
```

**Tareas:**
1. Auditar TODAS las importaciones cross-module
2. Identificar acoplamiento concreto
3. Definir interfaces en módulo consumidor
4. Implementar en módulo proveedor
5. Wire con DI container

### 3. FALTA DE BOUNDED CONTEXTS

**Problema:** No hay límites claros entre dominios

**Ejemplo:** `diffusion` debería ser un bounded context completo:
- Tiene su propio language ubiquitous
- Conceptos: Pipeline, LoRA, Prompt, Generation
- No debería "saber" de detalles de `llm` o `system`

**Lo que pasa ahora:**
- `diffusion` conoce implementación de `llm.providers.OllamaProvider`
- `diffusion` conoce implementación de `system.ResourceMonitor`
- Sin anti-corruption layer entre contextos

**Solución - Context Mapping:**
```
Diffusion Context
  ↓ (Anticorruption Layer)
  ↓ Define: ILLMAnalyzer
  ↓
LLM Context
  ↑ Implementa: OllamaLLMAnalyzer
  ↑ Adapta su modelo al contrato de Diffusion
```

**Tareas:**
1. Identificar bounded contexts naturales (diffusion, llm, visualization, etc.)
2. Definir context map (relaciones entre contextos)
3. Implementar anticorruption layers
4. Published Language para APIs entre contextos

### 4. SERVICIOS PROCEDURALES CON ENTIDADES ANÉMICAS

**Problema:** Arquitectura procedural disfrazada de OOP

**Patrón actual:**
```python
# Entidad anémica
@dataclass
class LoRAInfo:
    name: str
    alpha: float
    # Sin comportamiento

# Servicio con toda la lógica
class LoRARecommender:
    def recommend(self, analysis) -> list[LoRAInfo]:
        # Toda la lógica aquí
        loras = self.registry.get_loras()
        scored = self._score_loras(loras, analysis)
        filtered = self._filter_by_confidence(scored)
        return filtered
```

**Debería ser:**
```python
# Entidad rica
class LoRA:
    name: str
    alpha: Alpha  # Value Object validado

    def matches_analysis(self, analysis: PromptAnalysis) -> bool:
        # Lógica de matching en la entidad

    def calculate_relevance_for(self, prompt: Prompt) -> Confidence:
        # Scoring interno

# Servicio delgado (solo coordinación)
class LoRARecommender:
    def recommend(self, analysis) -> list[LoRA]:
        candidates = self.repository.get_compatible_loras(analysis.base_model)
        # Entidades tienen el comportamiento
        return [l for l in candidates if l.matches_analysis(analysis)]
```

**Tareas:**
1. Ver AUDIT_02_diffusion_models.md para detalles
2. Migrar lógica de servicios a entidades donde corresponda
3. Servicios solo para coordinación cross-aggregate
4. Domain services solo para operaciones multi-entity

### 5. SIN DEPENDENCY INJECTION

**Problema:** Construcción manual de dependencias

**Código actual:**
```python
class IntelligentGenerationPipeline:
    def __init__(self, config=None):
        self.registry = ModelRegistry()  # ❌ Construcción manual
        self.analyzer = PromptAnalyzer()  # ❌
        self.recommender = LoRARecommender(registry=self.registry)  # ❌
        self.optimizer = ParameterOptimizer()  # ❌
        # ...
```

**Problemas:**
- Testing requiere herencia o mocks complejos
- No se pueden reemplazar implementaciones
- Configuración hardcodeada
- Difícil cambiar comportamiento

**Solución DI:**
```python
class IntelligentGenerationPipeline:
    def __init__(
        self,
        registry: IModelRegistry,
        analyzer: IPromptAnalyzer,
        recommender: ILoRARecommender,
        optimizer: IParameterOptimizer,
    ):
        self.registry = registry
        self.analyzer = analyzer
        self.recommender = recommender
        self.optimizer = optimizer

# Container configura todo
container = Container()
container.register(IModelRegistry, SQLiteModelRegistry)
container.register(IPromptAnalyzer, OllamaPromptAnalyzer)
# ...
pipeline = container.resolve(IntelligentGenerationPipeline)
```

**Tareas:**
1. Implementar lightweight DI container (o usar dependency-injector)
2. Refactorizar constructores para recibir interfaces
3. Configurar container en application layer
4. Eliminar construcción manual

---

## PATRONES ARQUITECTURALES RECOMENDADOS

### Para ml_lib global:

1. **Hexagonal Architecture (Ports & Adapters)**
   - Dominio en el centro (entidades ricas, lógica de negocio)
   - Ports (interfaces) definen contratos
   - Adapters (infrastructure) implementan

2. **Domain-Driven Design (DDD)**
   - Bounded Contexts claros
   - Ubiquitous Language por contexto
   - Agregados bien definidos
   - Domain Events para comunicación

3. **CQRS (donde aplique)**
   - Separar comandos de queries
   - Especialmente útil en `learning_engine`, `model_registry`

4. **Event-Driven donde sea necesario**
   - Ej: Generation completed → trigger feedback collection
   - Memory threshold reached → trigger cleanup

---

## ESTRUCTURA PROPUESTA (REFACTOR COMPLETO)

```
ml_lib/
├── shared/                          ← Shared Kernel
│   ├── domain/
│   │   └── value_objects/          ← Compartidos entre contextos
│   ├── interfaces/
│   └── utils/
│
├── diffusion/                       ← Bounded Context
│   ├── domain/
│   │   ├── entities/               ← LoRA, Pipeline, Generation (ricas)
│   │   ├── value_objects/          ← Alpha, PromptWeight, etc.
│   │   ├── aggregates/             ← GenerationSession, etc.
│   │   ├── repositories/           ← Interfaces
│   │   └── services/               ← Domain services
│   ├── application/
│   │   ├── use_cases/              ← GenerateImageUseCase
│   │   ├── dto/
│   │   └── services/               ← Application services (orquestación)
│   ├── infrastructure/
│   │   ├── persistence/            ← SQLite, file system
│   │   ├── external_apis/          ← CivitAI, HuggingFace
│   │   └── diffusion_backends/     ← ComfyUI, diffusers
│   └── interfaces/
│       ├── api/                    ← REST, GraphQL (futuro)
│       └── cli/                    ← CLI commands
│
├── llm/                             ← Bounded Context
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   │   └── providers/              ← Ollama, OpenAI, etc.
│   └── interfaces/
│
├── infrastructure/                  ← Infraestructura compartida
│   ├── monitoring/                 ← ResourceMonitor (era system/)
│   ├── logging/
│   ├── configuration/
│   └── di/                         ← Dependency Injection
│
└── application/                     ← Application layer global
    ├── composition_root.py         ← DI Container setup
    └── facade.py                   ← API principal simplificada
```

---

## MÉTRICAS GLOBALES

```
Módulos totales: 28
Módulos grandes analizados: 3 (diffusion, llm, system)
Líneas estimadas total: ~50,000+
God classes identificadas: 3+ (IntelligentPipeline, ModelOrchestrator, etc.)
Violaciones arquitectura: MÚLTIPLES
Acoplamiento cross-module: ALTO
Testabilidad: BAJA (requiere mocks masivos)
Mantenibilidad: MEDIA-BAJA
```

---

## RECOMENDACIONES PRIORITARIAS

### CRÍTICO (Bloquea evolución)
1. **Establecer arquitectura de capas**
   - Empezar con diffusion/ (más grande)
   - Separar domain, application, infrastructure

2. **Eliminar acoplamiento directo cross-module**
   - Definir interfaces en módulo consumidor
   - Implementar DI básico

3. **Refactorizar god classes**
   - IntelligentGenerationPipeline
   - ModelOrchestrator

### IMPORTANTE (Facilita desarrollo)
4. **Definir bounded contexts**
   - Context map
   - Anticorruption layers

5. **Implementar DI completo**
   - Container configuration
   - Constructor injection

### MEJORA CONTINUA
6. **Enriquecer domain model**
   - Migrar lógica a entidades
   - Value Objects tipados

7. **Event-driven donde aplique**
   - Domain events
   - Async processing

---

## RUTA DE MIGRACIÓN SUGERIDA

### Fase 1: Preparación (No rompe nada)
1. Crear estructura nueva en paralelo
2. Definir interfaces/protocols
3. Implementar DI container
4. Crear tests para código crítico

### Fase 2: Migración Gradual (Por módulo)
1. Empezar con diffusion (más grande, más problemas)
2. Reorganizar en capas
3. Refactorizar god classes
4. Migrar código módulo a módulo

### Fase 3: Limpieza
1. Eliminar código viejo
2. Reorganizar imports
3. Actualizar docs

### Fase 4: Optimización
1. Event-driven donde aplique
2. CQRS donde corresponda
3. Performance tuning

---

## CONCLUSIÓN GLOBAL

ml_lib tiene **buena funcionalidad** pero **arquitectura débil**. Los principales problemas son:

1. **Arquitectura procedural** con fachada OOP
2. **Acoplamiento alto** entre módulos
3. **Sin separación de capas** clara
4. **Entidades anémicas** con servicios god-class
5. **Sin DI** - construcción manual

**IMPACTO:**
- Testing difícil
- Evolución lenta
- Refactors riesgosos
- Onboarding complicado

**PRIORIDAD:** Refactorizar arquitectura ANTES de agregar más features.

**TIEMPO ESTIMADO:**
- Fase 1: 2-3 semanas
- Fase 2: 4-6 semanas (gradual)
- Fase 3-4: 2-3 semanas

**BENEFICIO:** Código mantenible, testeable, escalable y comprensible.
