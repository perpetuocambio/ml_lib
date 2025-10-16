# CQRS Architecture Guide

## Overview

**CQRS (Command Query Responsibility Segregation)** separates read and write operations into distinct models:

- **Commands** (`application/commands/`): Write operations that modify state
- **Queries** (`application/queries/`): Read operations that retrieve data

## Architecture

```
┌─────────────────────────────────────────────┐
│              Application Layer              │
│                                             │
│  ┌─────────────────┐  ┌──────────────────┐ │
│  │   Commands      │  │     Queries      │ │
│  │  (Write Model)  │  │  (Read Model)    │ │
│  └─────────────────┘  └──────────────────┘ │
│         │                      │            │
│         ▼                      ▼            │
│  ┌─────────────────┐  ┌──────────────────┐ │
│  │  CommandBus     │  │   QueryBus       │ │
│  └─────────────────┘  └──────────────────┘ │
│         │                      │            │
└─────────┼──────────────────────┼────────────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────────────┐
│              Domain Layer                   │
│                                             │
│  Domain Services, Entities, Repositories    │
└─────────────────────────────────────────────┘
```

## Commands vs Queries

### Commands (Write Operations)

**Purpose:** Modify system state

**Characteristics:**
- Return status/result (success/failure)
- Have side effects
- Require validation
- Can fail
- Not cacheable
- Transactional

**Examples:**
```python
RecommendLoRAsCommand
GenerateImageCommand
CreateUserCommand
UpdateSettingsCommand
```

**Handler Pattern:**
```python
class MyCommandHandler(ICommandHandler[MyCommand]):
    def handle(self, command: MyCommand) -> CommandResult:
        # Validate
        if not command.param:
            return CommandResult.validation_error("param required")

        # Execute (modify state)
        result = self.service.do_something(command.param)

        # Return result
        return CommandResult.success(data=result)
```

### Queries (Read Operations)

**Purpose:** Retrieve data without modification

**Characteristics:**
- Return data directly
- No side effects
- No validation needed
- Should not fail (exception if error)
- Cacheable
- Optimizable with read models
- Can be denormalized

**Examples:**
```python
GetAllLoRAsQuery
GetLoRAsByBaseModelQuery
SearchLoRAsByPromptQuery
GetUserByIdQuery
```

**Handler Pattern:**
```python
class MyQueryHandler(IQueryHandler[MyQuery]):
    def handle(self, query: MyQuery) -> QueryResult:
        # Fetch data (no modification)
        data = self.repository.get(query.id)

        # Return data
        return QueryResult.success(data=data)
```

## Key Differences

| Aspect | Commands | Queries |
|--------|----------|---------|
| Purpose | Modify state | Retrieve data |
| Side Effects | Yes | No |
| Return | Status + optional data | Data always |
| Validation | Required | Not needed |
| Caching | No | Yes |
| Failure | Return error status | Raise exception |
| Performance | Write-optimized | Read-optimized |

## Usage Examples

### Command Usage

```python
from ml_lib.diffusion.application.commands import (
    CommandBus,
    RecommendLoRAsCommand,
    RecommendLoRAsHandler,
)

# Setup
bus = CommandBus()
handler = RecommendLoRAsHandler(lora_service)
bus.register(RecommendLoRAsCommand, handler)

# Execute write operation
command = RecommendLoRAsCommand(
    prompt="anime girl",
    base_model="SDXL",
    max_loras=3,
)
result = bus.dispatch(command)

if result.is_success:
    recommendations = result.data
else:
    print(f"Error: {result.error}")
```

### Query Usage

```python
from ml_lib.diffusion.application.queries import (
    QueryBus,
    GetAllLoRAsQuery,
    GetAllLoRAsHandler,
)

# Setup
bus = QueryBus()
handler = GetAllLoRAsHandler(lora_service)
bus.register(GetAllLoRAsQuery, handler)

# Execute read operation
query = GetAllLoRAsQuery()
result = bus.dispatch(query)

# Queries always succeed or raise
loras = result.data
print(f"Found {len(loras)} LoRAs")
```

## Benefits of CQRS

### 1. Separation of Concerns
- Write logic separate from read logic
- Different optimization strategies
- Independent scaling

### 2. Performance Optimization
- Reads can use denormalized data
- Writes focus on consistency
- Caching for queries
- Eventual consistency possible

### 3. Flexibility
- Different models for different needs
- Read models can be materialized views
- Write models enforce business rules

### 4. Scalability
- Read and write databases can be separate
- Scale reads independently from writes
- Read replicas for queries

### 5. Security
- Clear separation of read/write permissions
- Easy to enforce read-only access
- Audit trail for writes only

## Implementation Patterns

### Pattern 1: Shared Domain

Both Commands and Queries use same domain layer:

```python
# Commands modify through domain
class CreateLoRAHandler:
    def handle(self, command):
        lora = LoRA.create(...)  # Domain entity
        self.repository.save(lora)
        return CommandResult.success()

# Queries read from domain
class GetLoRAHandler:
    def handle(self, query):
        lora = self.repository.get(query.id)  # Same entity
        return QueryResult.success(lora)
```

### Pattern 2: Separate Read Models

Queries use optimized read models:

```python
# Write: Full domain model
class CreateLoRAHandler:
    def handle(self, command):
        lora = LoRA.create(...)  # Rich domain model
        self.repository.save(lora)
        # Update read model asynchronously
        return CommandResult.success()

# Read: Lightweight DTO
@dataclass
class LoRAReadModel:
    id: str
    name: str
    tags: list[str]
    # Only fields needed for display

class GetLoRAHandler:
    def handle(self, query):
        # Read from optimized read model
        lora_dto = self.read_model.get(query.id)
        return QueryResult.success(lora_dto)
```

## Current Implementation

### Available Commands

**LoRA Commands:**
- `RecommendLoRAsCommand` - Get LoRA recommendations
- `RecommendTopLoRACommand` - Get best single LoRA
- `FilterConfidentRecommendationsCommand` - Filter by confidence

**Image Generation Commands:**
- `GenerateImageCommand` - Full image generation
- `QuickGenerateCommand` - Simplified generation

### Available Queries

**LoRA Queries:**
- `GetAllLoRAsQuery` - Browse all LoRAs
- `GetLoRAsByBaseModelQuery` - Filter by model
- `SearchLoRAsByPromptQuery` - Search by keywords

## Best Practices

### 1. Keep Queries Pure

```python
# ✅ Good: No side effects
class GetLoRAHandler:
    def handle(self, query):
        return QueryResult.success(self.repo.get(query.id))

# ❌ Bad: Has side effects
class GetLoRAHandler:
    def handle(self, query):
        self.logger.increment_view_count(query.id)  # Side effect!
        return QueryResult.success(self.repo.get(query.id))
```

### 2. Commands Should Not Return Large Data

```python
# ✅ Good: Return ID or confirmation
class CreateLoRAHandler:
    def handle(self, command):
        lora = self.service.create(command.params)
        return CommandResult.success(data={"id": lora.id})

# ❌ Bad: Returning full object (use query instead)
class CreateLoRAHandler:
    def handle(self, command):
        lora = self.service.create(command.params)
        return CommandResult.success(data=lora)  # Too much data
```

### 3. Use Queries for Data Retrieval

```python
# ✅ Good: Separate command and query
# 1. Create resource
result = command_bus.dispatch(CreateLoRACommand(...))
lora_id = result.data["id"]

# 2. Fetch resource
result = query_bus.dispatch(GetLoRAByIdQuery(lora_id))
lora = result.data

# ❌ Bad: Command returns data
result = command_bus.dispatch(CreateLoRACommand(...))
lora = result.data  # Don't mix concerns
```

### 4. Make Queries Cacheable

```python
@dataclass(frozen=True)  # Immutable for caching
class GetLoRAQuery(IQuery):
    lora_id: str

    def cache_key(self) -> str:
        """Generate cache key for this query."""
        return f"lora:{self.lora_id}"
```

## Future Enhancements

### Event Sourcing
- Commands generate events
- Events update read models
- Complete audit trail

### Async Read Models
- Commands publish events
- Subscribers update read models
- Eventual consistency

### Caching Layer
- Query results cached
- Invalidation on commands
- Distributed cache support

### Multiple Read Models
- Different projections for different use cases
- Optimized for specific queries
- Materialized views

## Example: Full CQRS Flow

```python
# Setup buses
command_bus = CommandBus()
query_bus = QueryBus()

# Register handlers
command_bus.register(RecommendLoRAsCommand, RecommendLoRAsHandler(service))
query_bus.register(GetLoRAsByBaseModelQuery, GetLoRAsByBaseModelHandler(service))

# Application flow

# 1. User wants recommendations (write operation)
command = RecommendLoRAsCommand(
    prompt="anime girl",
    base_model="SDXL",
)
cmd_result = command_bus.dispatch(command)

if cmd_result.is_success:
    recommendations = cmd_result.data

# 2. User wants to browse all LoRAs for a model (read operation)
query = GetLoRAsByBaseModelQuery(base_model="SDXL")
query_result = query_bus.dispatch(query)
all_loras = query_result.data

# 3. User generates image (write operation with data)
gen_command = GenerateImageCommand(
    prompt="anime girl",
    base_model="SDXL",
)
gen_result = command_bus.dispatch(gen_command)

if gen_result.is_success:
    image_result = gen_result.data
    image_result.image.save("output.png")
```

## Summary

CQRS provides:
- ✅ Clear separation of reads and writes
- ✅ Independent optimization
- ✅ Scalability
- ✅ Flexibility
- ✅ Maintainability

Use **Commands** for operations that change state.
Use **Queries** for operations that retrieve data.
