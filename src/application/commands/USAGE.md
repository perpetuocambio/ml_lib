# Command Pattern Usage Guide

## Overview

The Command Pattern implementation provides a clean way to encapsulate requests as objects, allowing for:
- Decoupling between request and execution
- Centralized command dispatching
- Easy logging and auditing
- Queue-based execution (future)
- Transaction management (future)

## Architecture

```
Client → Command → CommandBus → Handler → Use Case/Service → Domain
```

## Basic Usage

### 1. Setup CommandBus

```python
from ml_lib.diffusion.application.commands import (
    CommandBus,
    RecommendLoRAsCommand,
    RecommendLoRAsHandler,
)

# Create bus
bus = CommandBus()

# Register handlers
lora_handler = RecommendLoRAsHandler(lora_service)
bus.register(RecommendLoRAsCommand, lora_handler)
```

### 2. Create and Dispatch Commands

```python
# Create command
command = RecommendLoRAsCommand(
    prompt="anime girl with sword",
    base_model="SDXL",
    max_loras=3,
    min_confidence=0.5,
)

# Dispatch
result = bus.dispatch(command)

# Check result
if result.is_success:
    recommendations = result.data
    print(f"Found {len(recommendations)} LoRAs")
else:
    print(f"Error: {result.error}")
```

## Available Commands

### LoRA Commands

#### RecommendLoRAsCommand
Get multiple LoRA recommendations.

```python
command = RecommendLoRAsCommand(
    prompt="beautiful anime girl",
    base_model="SDXL",
    max_loras=3,
    min_confidence=0.5,
)
result = bus.dispatch(command)
```

#### RecommendTopLoRACommand
Get single best LoRA.

```python
command = RecommendTopLoRACommand(
    prompt="photorealistic portrait",
    base_model="SDXL",
)
result = bus.dispatch(command)
```

#### FilterConfidentRecommendationsCommand
Filter to confident recommendations only.

```python
command = FilterConfidentRecommendationsCommand(
    recommendations=all_recommendations,
)
result = bus.dispatch(command)
```

### Image Generation Commands

#### GenerateImageCommand
Full-featured image generation.

```python
command = GenerateImageCommand(
    prompt="masterpiece, anime girl",
    negative_prompt="blurry, low quality",
    base_model="SDXL",
    seed=42,
    num_steps=30,
    cfg_scale=7.0,
    width=1024,
    height=1024,
    max_loras=3,
    min_lora_confidence=0.5,
)
result = bus.dispatch(command)

if result.is_success:
    image_result = result.data
    image_result.image.save("output.png")
    print(f"Generation time: {image_result.generation_time_seconds}s")
    print(f"LoRAs used: {image_result.loras_applied}")
    print(f"Explanation: {image_result.explanation}")
```

#### QuickGenerateCommand
Simplified generation with defaults.

```python
command = QuickGenerateCommand(
    prompt="beautiful landscape",
    base_model="SDXL",
)
result = bus.dispatch(command)
```

## Command Results

All commands return `CommandResult` with:

```python
class CommandResult:
    status: CommandStatus  # SUCCESS, FAILED, VALIDATION_ERROR, etc.
    data: any              # Result data (if successful)
    error: str | None      # Error message (if failed)
    metadata: dict | None  # Additional info

    # Convenience properties
    is_success: bool
    is_failure: bool
```

### Status Types

- `SUCCESS`: Command executed successfully
- `FAILED`: Command failed during execution
- `VALIDATION_ERROR`: Invalid command parameters
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Conflict with existing state

### Result Patterns

```python
result = bus.dispatch(command)

# Pattern 1: Check success flag
if result.is_success:
    data = result.data
else:
    print(f"Error: {result.error}")

# Pattern 2: Match on status
match result.status:
    case CommandStatus.SUCCESS:
        handle_success(result.data)
    case CommandStatus.VALIDATION_ERROR:
        handle_validation_error(result.error)
    case CommandStatus.NOT_FOUND:
        handle_not_found(result.error)
    case _:
        handle_generic_error(result.error)
```

## Creating New Commands

### 1. Define Command

```python
from dataclasses import dataclass
from ml_lib.diffusion.application.commands import ICommand

@dataclass(frozen=True)
class MyCommand(ICommand):
    """My custom command."""
    param1: str
    param2: int = 10  # Optional with default
```

### 2. Create Handler

```python
from ml_lib.diffusion.application.commands import (
    ICommandHandler,
    CommandResult,
)

class MyCommandHandler(ICommandHandler[MyCommand]):
    """Handler for MyCommand."""

    def __init__(self, my_service: MyService):
        self.service = my_service

    def handle(self, command: MyCommand) -> CommandResult:
        try:
            # Validate
            if not command.param1:
                return CommandResult.validation_error("param1 required")

            # Execute
            result = self.service.do_something(command.param1, command.param2)

            # Return success
            return CommandResult.success(data=result)

        except Exception as e:
            return CommandResult.failure(str(e))
```

### 3. Register and Use

```python
handler = MyCommandHandler(my_service)
bus.register(MyCommand, handler)

command = MyCommand(param1="value", param2=20)
result = bus.dispatch(command)
```

## Best Practices

### 1. Immutable Commands
Always use `@dataclass(frozen=True)` for commands.

```python
@dataclass(frozen=True)  # ← frozen=True
class MyCommand(ICommand):
    ...
```

### 2. Validation in Handlers
Validate in handler, not in command.

```python
def handle(self, command: MyCommand) -> CommandResult:
    # ✅ Validate here
    if not command.param:
        return CommandResult.validation_error("param required")
```

### 3. Single Responsibility
One command = one business operation.

```python
# ✅ Good: Specific command
class RecommendLoRAsCommand(ICommand):
    ...

# ❌ Bad: Generic command
class DoLoRAStuffCommand(ICommand):
    operation: str  # "recommend", "filter", "apply"...
```

### 4. Rich Metadata
Include useful metadata in results.

```python
return CommandResult.success(
    data=result,
    metadata={
        "execution_time_ms": elapsed_ms,
        "items_processed": count,
        "cache_hit": was_cached,
    },
)
```

### 5. Handler Dependencies
Inject services, don't create them.

```python
# ✅ Good: Dependency injection
class MyHandler(ICommandHandler[MyCommand]):
    def __init__(self, service: MyService):
        self.service = service

# ❌ Bad: Creating dependencies
class MyHandler(ICommandHandler[MyCommand]):
    def __init__(self):
        self.service = MyService()  # Hard to test!
```

## Testing Commands

Commands and handlers are easy to test in isolation:

```python
def test_recommend_loras_command():
    # Arrange
    mock_service = Mock()
    mock_service.recommend.return_value = [mock_recommendation]

    handler = RecommendLoRAsHandler(mock_service)
    command = RecommendLoRAsCommand(
        prompt="test",
        base_model="SDXL",
    )

    # Act
    result = handler.handle(command)

    # Assert
    assert result.is_success
    assert len(result.data) == 1
    mock_service.recommend.assert_called_once()
```

## Future Enhancements

The Command Pattern enables future features:

- **Command Queue**: Async execution with message queue
- **Command Logging**: Audit trail of all operations
- **Command Retry**: Automatic retry on failures
- **Command Undo**: Reversible operations
- **Middleware**: Cross-cutting concerns (auth, logging, timing)
- **CQRS**: Separate Commands (writes) from Queries (reads)

## Example: Full Application Setup

```python
from ml_lib.diffusion.application.commands import (
    CommandBus,
    RecommendLoRAsCommand,
    RecommendLoRAsHandler,
    GenerateImageCommand,
    GenerateImageHandler,
)

# Setup services
lora_service = LoRARecommendationService(repository)
generate_use_case = GenerateImageUseCase(lora_service, analyzer, monitor)

# Setup command bus
bus = CommandBus()

# Register all handlers
bus.register(RecommendLoRAsCommand, RecommendLoRAsHandler(lora_service))
bus.register(GenerateImageCommand, GenerateImageHandler(generate_use_case))

# Use from application
def api_endpoint(request_data):
    command = GenerateImageCommand(**request_data)
    result = bus.dispatch(command)

    if result.is_success:
        return {"image": result.data, "metadata": result.metadata}
    else:
        return {"error": result.error}, 400
```
