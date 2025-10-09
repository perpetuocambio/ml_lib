# Módulo de Interfaces (`pyintelcivil.src.infrastructure.extractors.interfaces`)

Este módulo define las interfaces y clases base abstractas para los extractores de documentos. Establece los contratos que deben cumplir los extractores concretos, promoviendo la extensibilidad y la interoperabilidad dentro del sistema de extracción.

## Clases

-   `BaseDocumentExtractor`: Clase base abstracta para extractores de documentos.
-   `DocumentExtractor`: Protocolo que define la interfaz para extractores de documentos.

## Diagrama de Clases

```mermaid
classDiagram
    direction LR

    class BaseDocumentExtractor {
        <<abstract>>
        +name: str
        +can_extract(file_path, document_type): bool
        +extract(file_path, config): ExtractedContent
        +get_capabilities(): ExtractionCapabilities
        +get_supported_types(): list~DocumentType~
    }

    class DocumentExtractor {
        <<protocol>>
        +can_extract(file_path, document_type): bool
        +extract(file_path, config): ExtractedContent
        +get_capabilities(): ExtractionCapabilities
    }

    DocumentExtractor <|.. BaseDocumentExtractor
```
