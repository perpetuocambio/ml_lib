# Módulo de Servicios de Carga de Datos (`data_loaders/services`)

Este módulo proporciona servicios para cargar datos desde diferentes fuentes y formatos, convirtiéndolos en objetos de dominio tipados y validados. El objetivo es asegurar que los datos que entran al sistema sean consistentes y cumplan con los contratos definidos en el dominio.

## `TypedYamlLoader`

El componente principal de este módulo es `TypedYamlLoader`. Esta clase se especializa en cargar archivos YAML y convertirlos de manera inteligente y recursiva en instancias de `dataclasses` de Python. Esto elimina la necesidad de manejar diccionarios genéricos y asegura que los datos se cargan con los tipos correctos desde el principio.

### Características

-   **Casting Inteligente**: Utiliza la introspección de tipos de Python para convertir los datos del YAML a los tipos definidos en los `dataclasses`.
-   **Soporte para Tipos Anidados**: Maneja `dataclasses` anidados y listas de `dataclasses` (`List[MyDataclass]`) de forma recursiva.
-   **Validación Estricta**: No permite el uso de `dict`, `Any` o `tuple` en las firmas, forzando el uso de tipos concretos.
-   **Cero Dependencias del Dominio**: Es una utilidad de infraestructura pura que no tiene conocimiento del dominio de negocio.

### `load()`

El método `load()` es el punto de entrada principal. Orquesta la lectura del archivo YAML y el proceso de conversión al `dataclass` especificado.

## Diagrama de Clases

```mermaid
classDiagram
    class TypedYamlLoader {
        +file_path: str
        +target_type: type
        +load(): T
        -_cast_to_dataclass(yaml_data): T
        -_cast_value_to_type(value, target_type): object
        -_create_nested_dataclass(yaml_data, dataclass_type): object
    }

    class Path
    class yaml

    TypedYamlLoader ..> Path : uses
    TypedYamlLoader ..> yaml : uses
```

## Diagrama de Secuencia

Este diagrama ilustra el proceso de carga de un archivo YAML y su conversión a un `dataclass` tipado.

```mermaid
sequenceDiagram
    participant Client
    participant TypedYamlLoader
    participant FileSystem
    participant yaml_parser

    Client->>TypedYamlLoader: __init__(file_path, MyDataclass)
    Client->>TypedYamlLoader: load()
    activate TypedYamlLoader

    TypedYamlLoader->>FileSystem: read(file_path)
    FileSystem-->>TypedYamlLoader: yaml_content

    TypedYamlLoader->>yaml_parser: safe_load(yaml_content)
    yaml_parser-->>TypedYamlLoader: raw_yaml_data

    loop for each field in MyDataclass
        TypedYamlLoader->>TypedYamlLoader: _cast_value_to_type(value, field_type)
    end

    TypedYamlLoader-->>Client: instance_of_MyDataclass
    deactivate TypedYamlLoader
```
