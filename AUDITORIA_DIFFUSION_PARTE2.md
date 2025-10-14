# AUDITORÍA DIFFUSION - PARTE 2: Problemas Funcionales y Técnicos

## 4. PROBLEMAS FUNCIONALES (continuación)

### 4.1 Soluciones Propuestas para Precisión del Prompt

#### Solución 1: Pipeline de Compactación Obligatorio

```python
class PromptProcessingPipeline:
    """Pipeline unificado de procesamiento de prompts."""

    def __init__(
        self,
        analyzer: PromptAnalyzer,
        compactor: PromptCompactor,
        max_tokens: int = 77
    ):
        self.analyzer = analyzer
        self.compactor = compactor
        self.max_tokens = max_tokens

    def process(
        self,
        prompt: str,
        base_model_architecture: str,
        quality: str = "balanced"
    ) -> ProcessedPrompt:
        """
        Procesa prompt con garantía de < 77 tokens.

        Returns:
            ProcessedPrompt con warnings si contenido fue removido
        """
        # 1. Optimizar para arquitectura
        optimized_positive, optimized_negative = self.analyzer.optimize_for_model(
            prompt=prompt,
            base_model_architecture=base_model_architecture,
            quality=quality,
            enable_compaction=False  # No compactar aún
        )

        # 2. Verificar tokens
        token_count = self.compactor.count_tokens(optimized_positive)

        # 3. Compactar SI NECESARIO
        if token_count > self.max_tokens:
            result = self.compactor.compact(
                optimized_positive,
                preserve_nsfw=True,
                min_quality_tags=2
            )

            return ProcessedPrompt(
                original=prompt,
                final=result.compacted_prompt,
                was_compacted=True,
                removed_content=result.removed_tokens,
                warnings=result.warnings
            )

        return ProcessedPrompt(
            original=prompt,
            final=optimized_positive,
            was_compacted=False
        )
```

**Beneficios:**
- ✅ Garantiza ≤77 tokens
- ✅ Usuario informado de cambios
- ✅ Contenido NSFW preservado prioritariamente
- ✅ Un solo punto de procesamiento

#### Solución 2: Sistema de Feedback al Usuario

```python
@dataclass
class ProcessedPrompt:
    """Resultado de procesamiento de prompt."""

    original: str
    final: str
    was_compacted: bool
    removed_content: list[TokenClassification] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_critical_removals(self) -> bool:
        """Verifica si se removió contenido crítico."""
        return any(
            token.priority == PromptTokenPriority.CRITICAL
            for token in self.removed_content
        )

    @property
    def user_notification(self) -> str:
        """Genera notificación para el usuario."""
        if not self.was_compacted:
            return "Prompt procesado sin cambios."

        msg = f"Prompt compactado: {len(self.removed_content)} elementos removidos.\n"

        if self.has_critical_removals:
            msg += "⚠️ ADVERTENCIA: Contenido crítico fue removido!\n"
            critical = [t.token for t in self.removed_content
                       if t.priority == PromptTokenPriority.CRITICAL]
            msg += f"Removido: {', '.join(critical)}\n"

        return msg
```

### 4.2 Falta de Integración entre Componentes

**Problema:** Componentes aislados sin comunicación

#### A. UserPreferencesDB no se usa en ningún sitio

**Debería integrarse en:**

```python
# intelligent_builder.py
class IntelligentPipelineBuilder:
    def __init__(
        self,
        # ... otros parámetros
        user_preferences_db: Optional[UserPreferencesDB] = None,
        user_id: Optional[str] = None
    ):
        self.user_prefs_db = user_preferences_db
        self.user_id = user_id

    def _select_models(self, config: GenerationConfig) -> SelectedModels:
        # INTEGRAR: Consultar preferencias de usuario
        if self.user_prefs_db and self.user_id:
            prefs = self.user_prefs_db.get_or_create_preferences(self.user_id)

            # Aplicar favoritos
            if prefs.favorite_base_models:
                # Priorizar modelos favoritos
                pass

            # Filtrar bloqueados
            if prefs.blocked_models:
                # Excluir modelos bloqueados
                pass
```

**Cambios necesarios:**
1. Añadir parámetros user_id en facade.py
2. Pasar user_prefs_db a IntelligentPipelineBuilder
3. Consultar preferencias en selección de modelos
4. Registrar generaciones en historial

#### B. PromptCompactor no se usa en PromptAnalyzer

**Situación actual:**
- `PromptAnalyzer` tiene su propia implementación de compactación
- `PromptCompactor` es un servicio dedicado más robusto
- **Código duplicado y lógica inconsistente**

**Solución:**

```python
# prompt_analyzer.py
class PromptAnalyzer:
    def __init__(
        self,
        # ... otros parámetros
        prompt_compactor: Optional[PromptCompactor] = None
    ):
        self.prompt_compactor = prompt_compactor or PromptCompactor(max_tokens=77)

    def compact_prompt(
        self,
        prompt: str,
        max_tokens: int = 77,
        preserve_nsfw: bool = True
    ) -> tuple[str, dict]:
        """
        DEPRECATED: Usar self.prompt_compactor directamente.

        Este método se mantiene por compatibilidad pero delega
        al PromptCompactor dedicado.
        """
        result = self.prompt_compactor.compact(prompt, preserve_nsfw)

        # Convertir a formato antiguo para compatibilidad
        return result.compacted_prompt, {
            "original_tokens": result.original_token_count,
            "final_tokens": result.compacted_token_count,
            "removed_parts": [t.token for t in result.removed_tokens],
            "compaction_needed": result.was_compacted
        }
```

### 4.3 Extensibilidad para Otros Tipos de Contenido

**Situación actual:**
- Sistema optimizado SOLO para contenido NSFW
- `content_tags.py` tiene hardcoded categorías NSFW
- No hay abstracciones para otros dominios

**Problema para el futuro:**
- ¿Cómo añadir contenido de fantasía?
- ¿Cómo gestionar diseños de camisetas?
- ¿Cómo manejar arte conceptual?

#### Solución: Sistema de Categorización Extensible

```python
# models/content_categories.py
class ContentDomain(Enum):
    """Dominios de contenido soportados."""
    NSFW = "nsfw"
    FANTASY = "fantasy"
    DESIGN = "design"
    CONCEPT_ART = "concept_art"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class ContentCategoryRegistry:
    """
    Registro extensible de categorías de contenido.

    Permite añadir nuevos dominios sin modificar código existente.
    """

    def __init__(self):
        self._categories: dict[ContentDomain, ContentCategoryDefinition] = {}
        self._register_default_categories()

    def register_category(
        self,
        domain: ContentDomain,
        definition: ContentCategoryDefinition
    ) -> None:
        """Registra nueva categoría de contenido."""
        self._categories[domain] = definition

    def get_keywords(self, domain: ContentDomain) -> KeywordSet:
        """Obtiene keywords para un dominio."""
        if domain not in self._categories:
            raise ValueError(f"Unknown domain: {domain}")
        return self._categories[domain].keywords

    def classify_token(
        self,
        token: str,
        domain: ContentDomain
    ) -> TokenClassification:
        """Clasifica token según dominio específico."""
        definition = self._categories.get(domain)
        if not definition:
            return TokenClassification(
                token=token,
                priority=PromptTokenPriority.MEDIUM,
                category=None
            )

        return definition.classify_token(token)


@dataclass
class ContentCategoryDefinition:
    """Definición de una categoría de contenido."""

    domain: ContentDomain
    keywords: KeywordSet
    priorities: PriorityRules

    def classify_token(self, token: str) -> TokenClassification:
        """Clasifica token según esta categoría."""
        # Implementación específica por dominio
        pass


@dataclass
class KeywordSet:
    """Conjunto de keywords para un dominio."""

    critical: list[str]  # CRITICAL priority
    high: list[str]      # HIGH priority
    medium: list[str]    # MEDIUM priority
    low: list[str]       # LOW priority

    @classmethod
    def from_dict(cls, data: dict) -> "KeywordSet":
        """Carga desde diccionario (config file)."""
        return cls(
            critical=data.get("critical", []),
            high=data.get("high", []),
            medium=data.get("medium", []),
            low=data.get("low", [])
        )


# Uso:
registry = ContentCategoryRegistry()

# Para NSFW (actual)
registry.register_category(
    ContentDomain.NSFW,
    ContentCategoryDefinition.from_config("config/nsfw_categories.yaml")
)

# Para FANTASY (futuro)
registry.register_category(
    ContentDomain.FANTASY,
    ContentCategoryDefinition.from_config("config/fantasy_categories.yaml")
)

# Clasificar
token_class = registry.classify_token("dragon", ContentDomain.FANTASY)
```

**Beneficios:**
- ✅ Extensible sin modificar código existente
- ✅ Configuración externa (YAML/JSON)
- ✅ Un sistema para todos los dominios
- ✅ Fácil añadir nuevos tipos de contenido

#### Estructura de Config Externa

```yaml
# config/nsfw_categories.yaml
domain: nsfw
keywords:
  critical:
    - "girl"
    - "boy"
    - "woman"
    - "man"
  high:
    - "fellatio"
    - "blowjob"
    - "anal"
    - "vaginal"
  medium:
    - "lingerie"
    - "nude"
  low:
    - "breasts"
    - "nipples"

# config/fantasy_categories.yaml
domain: fantasy
keywords:
  critical:
    - "dragon"
    - "wizard"
    - "knight"
    - "elf"
  high:
    - "magic"
    - "spell"
    - "sword"
    - "castle"
  medium:
    - "forest"
    - "mountain"
  low:
    - "mystical"
    - "enchanted"
```

---

## 5. PROBLEMAS TÉCNICOS

### 5.1 Gestión de Dependencias

#### A. Dependencias Circulares Potenciales

**Detectado:**
```python
# services/intelligent_builder.py
from ml_lib.diffusion.services.model_orchestrator import ModelOrchestrator
from ml_lib.diffusion.services.ollama_selector import OllamaModelSelector

# services/model_orchestrator.py
from ml_lib.diffusion.storage.metadata_db import MetadataDatabase

# storage/metadata_db.py
from ml_lib.diffusion.model_enums import ModelType, BaseModel
# ✅ OK - no circular

# services/ollama_selector.py
from ml_lib.llm.providers.ollama_provider import OllamaProvider
# ✅ OK - dependencia externa
```

**Evaluación:** ✅ No hay circulares actualmente, pero estructura frágil

**Recomendación:** Implementar Dependency Injection

```python
# services/service_container.py
class ServiceContainer:
    """Container IoC para servicios."""

    def __init__(self):
        self._services: dict[type, object] = {}

    def register(self, service_type: type, instance: object) -> None:
        """Registra servicio."""
        self._services[service_type] = instance

    def get(self, service_type: type) -> object:
        """Obtiene servicio."""
        if service_type not in self._services:
            raise ValueError(f"Service not registered: {service_type}")
        return self._services[service_type]

# Uso en builder
class IntelligentPipelineBuilder:
    def __init__(
        self,
        service_container: Optional[ServiceContainer] = None
    ):
        self.container = service_container or self._create_default_container()

        # Obtener servicios del container
        self.orchestrator = self.container.get(ModelOrchestrator)
        self.prompt_analyzer = self.container.get(PromptAnalyzer)
```

#### B. Imports Inline Excesivos

**Categorización:**

| Tipo | Cantidad | Estado |
|------|----------|--------|
| Lazy loading (libs pesadas) | 18 | ✅ Justificado |
| Imports de stdlib | 5 | ❌ Mover a top |
| Type checking (TYPE_CHECKING) | 12 | ✅ Correcto |

**Acción requerida:**
```python
# ❌ MAL - user_preferences_db.py:294
def record_generation(self, ...):
    import hashlib  # ← MOVER A TOP
    import json     # ← MOVER A TOP

# ✅ BIEN - intelligent_builder.py:734
def _configure_scheduler(self, ...):
    from diffusers import (  # ← OK: lazy loading
        DPMSolverMultistepScheduler,
    )
```

### 5.2 Manejo de Errores

**Problema:** Manejo inconsistente de errores

**Ejemplos:**

```python
# prompt_compactor.py:84
try:
    tokens = tokenizer.encode(text)
    return len(tokens)
except Exception as e:  # ❌ Demasiado genérico
    logger.debug(f"Tokenizer encoding failed: {e}, using fallback")

# intelligent_builder.py:728
except Exception as e:  # ❌ Captura todo
    logger.error(f"Failed to load pipeline: {e}")
    raise
```

**Solución:**

```python
# Excepciones específicas
class DiffusionError(Exception):
    """Base exception for diffusion module."""
    pass

class TokenizerError(DiffusionError):
    """Tokenizer-related errors."""
    pass

class ModelLoadError(DiffusionError):
    """Model loading errors."""
    pass

# Uso
try:
    tokens = tokenizer.encode(text)
except (RuntimeError, ValueError) as e:
    raise TokenizerError(f"Failed to tokenize: {e}") from e
```

### 5.3 Testing

**Estado actual:** ⚠️ **INSUFICIENTE**

**Tests encontrados:**
```bash
tests/
├── test_compaction.py          # ❌ Nuevo, sin ejecutar
├── test_nsfw_lora_selection.py # ❌ Nuevo, sin ejecutar
└── test_prompt_compaction.py   # ❌ Nuevo, sin ejecutar
```

**Cobertura estimada:** < 20%

**Tests faltantes críticos:**
1. ❌ `test_user_preferences_integration.py`
2. ❌ `test_prompt_processing_pipeline.py`
3. ❌ `test_model_selection_with_preferences.py`
4. ❌ `test_lora_recommendation_accuracy.py`

**Recomendación:** Crear suite completa de tests de integración

---

## 6. PROBLEMAS DE PERFORMANCE

### 6.1 Ollama Server Management

**Problema detectado:**
```python
# intelligent_builder.py:403-413
if selector.stop_server(force=True):
    logger.info("Stopped Ollama server...")
    time.sleep(2)  # ❌ Sleep blocking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Issues:**
1. `sleep(2)` bloquea el hilo
2. No hay verificación de que el servidor realmente paró
3. Force=True puede matar procesos de otros usuarios

**Solución:**
```python
async def _stop_ollama_gracefully(self, selector: OllamaModelSelector) -> bool:
    """Para Ollama de forma asíncrona y verificada."""
    if not await selector.stop_server_async(force=False, timeout=5.0):
        logger.warning("Ollama didn't stop gracefully, forcing...")
        return await selector.stop_server_async(force=True, timeout=2.0)
    return True
```

### 6.2 Carga de Tokenizer

**Problema:**
```python
# prompt_compactor.py:58
from transformers import CLIPTokenizer
self._tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
# Primera llamada descarga ~500MB
```

**Solución:** Caché compartido

```python
# Singleton tokenizer cache
class TokenizerCache:
    _instance = None
    _tokenizers: dict[str, CLIPTokenizer] = {}

    @classmethod
    def get_tokenizer(cls, model_name: str) -> CLIPTokenizer:
        if model_name not in cls._tokenizers:
            cls._tokenizers[model_name] = CLIPTokenizer.from_pretrained(
                model_name,
                cache_dir="/shared/cache/tokenizers"
            )
        return cls._tokenizers[model_name]
```

---

## 7. PRIORIZACIÓN DE CORRECCIONES

### 7.1 CRÍTICO (Hacer AHORA)

1. **Integrar PromptCompactor en pipeline**
   - Esfuerzo: 2-3 horas
   - Impacto: Alto - Mejora precisión prompts

2. **Consolidar lógica de compactación**
   - Eliminar código duplicado en PromptAnalyzer
   - Esfuerzo: 1-2 horas

3. **Integrar content_tags en models/__init__.py**
   - Esfuerzo: 15 minutos
   - Impacto: Medio - Permite usar las clases

4. **Mover imports inline de stdlib a top**
   - Esfuerzo: 30 minutos
   - Impacto: Bajo - Limpieza de código

### 7.2 ALTO (Hacer ESTA SEMANA)

5. **Integrar UserPreferencesDB**
   - En IntelligentPipelineBuilder
   - En ModelOrchestrator
   - En LoRARecommender
   - Esfuerzo: 4-6 horas
   - Impacto: Alto - Personalización

6. **Refactorizar dicts a clases**
   - NSFW_KEYWORDS → NSFWKeywordRegistry
   - detected_acts → DetectedActs
   - Esfuerzo: 3-4 horas
   - Impacto: Medio - Cumplimiento normas

7. **Implementar ContentCategoryRegistry**
   - Sistema extensible para nuevos dominios
   - Esfuerzo: 6-8 horas
   - Impacto: Alto - Escalabilidad futura

### 7.3 MEDIO (Hacer PRÓXIMAS 2 SEMANAS)

8. **Crear suite de tests**
   - Tests de integración
   - Tests de precisión de prompts
   - Esfuerzo: 8-12 horas

9. **Implementar sistema de feedback al usuario**
   - ProcessedPrompt con warnings
   - Notificaciones de cambios
   - Esfuerzo: 4-5 horas

10. **Optimizar performance**
    - Caché de tokenizers
    - Async para Ollama
    - Esfuerzo: 3-4 horas

---

## 8. PLAN DE IMPLEMENTACIÓN

### Fase 1: Correcciones Críticas (DÍA 1)
- [ ] Integrar PromptCompactor
- [ ] Consolidar compactación
- [ ] Fix imports inline
- [ ] Añadir content_tags a __init__

### Fase 2: Integración Core (DÍA 2-3)
- [ ] Integrar UserPreferencesDB
- [ ] Implementar ProcessedPrompt
- [ ] Sistema de warnings

### Fase 3: Refactoring (DÍA 4-5)
- [ ] Dicts → Clases
- [ ] Tuples → Value Objects
- [ ] Excepciones específicas

### Fase 4: Extensibilidad (SEMANA 2)
- [ ] ContentCategoryRegistry
- [ ] Config externa (YAML)
- [ ] Tests extensibilidad

### Fase 5: Testing & Performance (SEMANA 2)
- [ ] Suite tests integración
- [ ] Performance optimizations
- [ ] Documentación actualizada

---

**FIN DE AUDITORÍA - PARTE 2**
