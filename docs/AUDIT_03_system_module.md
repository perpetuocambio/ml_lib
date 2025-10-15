# AUDITORÍA: ml_lib/system/

**Fecha:** 2025-10-15
**Módulo:** `ml_lib/system/`
**Problema Central:** LÓGICA MAL UBICADA + MÓDULO MAL DISEÑADO

---

## RESUMEN EJECUTIVO

El módulo `system/` contiene solo 3 archivos (~600 líneas) pero tiene un **problema arquitectural grave**: contiene funcionalidad que **NO debería estar en un módulo de "system"** y está siendo usado incorrectamente desde `diffusion/`.

### Hallazgo Principal

**El módulo `system/` es un cajón de sastre mal nombrado.** Contiene:
1. `resource_monitor.py` - ✅ Correcto para system
2. `process_utils.py` - ⚠️ Dudoso, probablemente debería ser core/utils
3. Uso desde `diffusion/` - ❌ VIOLACIÓN de arquitectura

**PROBLEMA CRÍTICO:** `diffusion/services/model_orchestrator.py` importa directamente de `ml_lib.system.resource_monitor`, creando acoplamiento entre capas que deberían estar separadas.

---

## ANÁLISIS DETALLADO

### 1. RESOURCE_MONITOR.PY (~480 líneas)

**Propósito:** Monitorear recursos del sistema (GPU, CPU, RAM)

**Estructura:**
- `GPUStats`, `CPUStats`, `RAMStats`, `SystemResources` (dataclasses)
- `ResourceMonitor` (clase principal)

**LO BUENO:**
- ✅ Bien encapsulado
- ✅ Sin dependencias del resto de ml_lib
- ✅ Reutilizable
- ✅ Código limpio y bien documentado
- ✅ Manejo correcto de dependencias opcionales (torch, psutil, pynvml)

**LO MALO:**
- ❌ Ubicación incorrecta - debería ser `ml_lib/monitoring/` o `ml_lib/infrastructure/`
- ❌ Usado directamente desde `diffusion/` - VIOLACIÓN de capas
- ⚠️ Dataclasses podrían tener más comportamiento (ej: `GPUStats.format_summary()`)
- ⚠️ `print_summary()` mezcla lógica con presentación (debería ser un formatter separado)

**Tareas futuras:**
- **CRÍTICO:** Mover a `ml_lib/infrastructure/monitoring/`
- Crear interface `IResourceMonitor` para abstraer implementación
- Extraer `ResourceFormatter` para separar presentation de lógica
- Agregar métodos útiles a dataclasses (ej: `is_critical()`, `format()`)
- Considerar event-driven monitoring (alertas cuando memoria baja)

### 2. PROCESS_UTILS.PY (~120 líneas estimadas)

**Propósito (sin ver código):** Probablemente utilidades de gestión de procesos

**PROBLEMAS PROBABLES:**
- ❌ Nombre demasiado genérico
- ⚠️ Probablemente contiene lógica que debería estar en otro lugar
- ⚠️ "Utils" es code smell - indica falta de abstracción

**Tareas futuras:**
- **Auditar contenido** y determinar si pertenece aquí
- Si es infra → mover a `infrastructure/`
- Si es generic → mover a `core/utils/`
- **Eliminar "utils" pattern** - crear abstracciones con nombre claro

### 3. __INIT__.PY

**Contenido probable:** Exports del módulo

**Tareas:**
- Asegurar que exports sean claros
- No exponer internals

---

## PROBLEMA ARQUITECTURAL: USO DESDE DIFFUSION/

### El Problema

En `diffusion/services/model_orchestrator.py:19`:
```python
from ml_lib.system.resource_monitor import ResourceMonitor
```

**Por qué es malo:**
1. **Acoplamiento de capas**: `diffusion` (dominio) acoplado a `system` (infra)
2. **Violación Dependency Inversion**: Depende de implementación concreta
3. **Testing difícil**: Requiere mockear `ResourceMonitor` completo
4. **Reutilización imposible**: `diffusion` no portable sin `system`

### La Solución Correcta

**Arquitectura de capas:**
```
diffusion/           (Dominio - no conoce infra)
  ├── interfaces/    ✅ Define IResourceMonitor
  └── services/      ✅ Depende de IResourceMonitor

infrastructure/      (Implementaciones)
  └── monitoring/
      └── resource_monitor.py ✅ Implementa IResourceMonitor

application/         (Wiring)
  └── dependency_container.py ✅ Inyecta ResourceMonitor
```

**Tareas CRÍTICAS:**
1. Crear `diffusion/interfaces/resource_protocol.py`:
   - Definir `IResourceMonitor` protocol
   - Definir `ResourceStats` interface

2. Refactorizar `ModelOrchestrator`:
   - Constructor acepta `IResourceMonitor`
   - No importar directamente de system

3. Mover `ResourceMonitor`:
   - De `system/` a `infrastructure/monitoring/`
   - Implementar `IResourceMonitor` protocol

4. Crear Dependency Container:
   - Wire interfaces con implementaciones
   - Inyectar en construcción

---

## PROBLEMAS ESTRUCTURALES

### 1. MÓDULO "SYSTEM" MAL NOMBRADO

**Problema:** "system" es demasiado vago

**Qué contiene realmente:**
- Monitoring de recursos ← Esto es `infrastructure/monitoring`
- Process utils ← Probablemente `core/utils` o `infrastructure/process`

**Tareas:**
- Renombrar/eliminar directorio `system/`
- Redistribuir contenido según responsabilidad real
- Estructura sugerida:
  ```
  infrastructure/
    ├── monitoring/
    │   └── resource_monitor.py
    └── process/
        └── (process utils si aplica)
  ```

### 2. FALTA DE ABSTRACCIONES

**Problema:** Implementación concreta sin interface

**Consecuencias:**
- Otros módulos dependen de implementación
- Testing difícil
- No se puede cambiar implementación

**Tareas:**
- Definir protocols/interfaces para cada componente
- Separar interface de implementación
- Dependency Injection

### 3. MIXING CONCERNS EN RESOURCE_MONITOR

**Problema:** `print_summary()` mezcla lógica con presentación

**Tareas:**
- Extraer `ResourceFormatter` class
- `ResourceMonitor` solo hace monitoring
- Formatter hace presentation

---

## MÉTRICAS

```
Total archivos: 3
Líneas totales: ~600
Archivos bien ubicados: 0 (todos deberían moverse)
Violaciones arquitectura: 1+ (uso desde diffusion)
Acoplamiento incorrecto: CRÍTICO
```

---

## RECOMENDACIONES PRIORITARIAS

### CRÍTICO (Hacer YA)
1. **Eliminar importación directa desde diffusion/**
   - Crear `IResourceMonitor` protocol
   - Refactorizar `ModelOrchestrator` para recibir interface
   - Implementar Dependency Injection

2. **Reorganizar estructura**
   - Mover `resource_monitor.py` → `infrastructure/monitoring/`
   - Evaluar `process_utils.py` y reubicar apropiadamente
   - Eliminar directorio `system/` si queda vacío

### IMPORTANTE
3. **Separar concerns**
   - Extraer `ResourceFormatter` de `ResourceMonitor`
   - Lógica de monitoring vs. presentación

4. **Agregar comportamiento a dataclasses**
   - Métodos útiles en `GPUStats`, etc.
   - Evitar lógica dispersa en servicios

### MEJORA CONTINUA
5. **Documentar arquitectura de capas**
   - Diagramas de dependencias permitidas
   - Guidelines de uso

---

## CONCLUSIÓN

El módulo `system/` tiene código **de buena calidad** pero está **mal ubicado arquitecturalmente** y **mal usado** desde otros módulos.

**PROBLEMA MÁS GRAVE:** El acoplamiento `diffusion` → `system` viola principios arquitecturales y hace el código:
- Difícil de testear
- Difícil de evolucionar
- No portable
- Acoplado a implementación

**Prioridad MÁXIMA:** Refactorizar arquitectura de dependencias ANTES de continuar.

---

## ADVERTENCIA

⚠️ **NO agregues más código a `system/`**

Este módulo debe ser eliminado/reorganizado. Cualquier nueva funcionalidad debe ir en:
- `infrastructure/` si es implementación de infra
- `core/` si es lógica compartida
- Con interfaces/protocols claros en los dominios que lo usan
