# Módulo de Servicios (`pyintelcivil.src.infrastructure.extractors.services`)

Este módulo contiene la lógica de orquestación y gestión de los extractores de documentos. La clase principal aquí es el `DocumentExtractorFactory`, que se encarga de seleccionar el extractor adecuado y coordinar el proceso de extracción, incluyendo estrategias de fallback.

## Clases

-   `DocumentExtractorFactory`: Factory para crear y gestionar extractores de documentos.

## Diagrama de Clases

```mermaid
classDiagram
    direction LR

    class DocumentExtractorFactory {
        +registry: ExtractorRegistry
        +extract_document(file_path, config): ExtractedContent
        +get_supported_types(): list~ExtractorRegistration~
        +register_custom_extractor(strategy, extractor)
        +set_extraction_priority(order)
    }

    DocumentExtractorFactory "1" *-- "1" ExtractorRegistry
    DocumentExtractorFactory "1" *-- "1" DocumentTypeDetector
    DocumentExtractorFactory "1" *-- "1" FileValidator

    ExtractorRegistry <.. DocumentExtractorFactory
    DocumentTypeDetector <.. DocumentExtractorFactory
    FileValidator <.. DocumentExtractorFactory
    ExtractedContent <.. DocumentExtractorFactory
    ExtractionConfig <.. DocumentExtractorFactory
    ExtractorRegistration <.. DocumentExtractorFactory
```

## Diagrama de Secuencia

```mermaid
sequenceDiagram
    participant Client
    participant DocumentExtractorFactory
    participant ExtractorRegistry
    participant DocumentTypeDetector
    participant FileValidator
    participant BaseDocumentExtractor

    Client->>DocumentExtractorFactory: extract_document(file_path, config)
    activate DocumentExtractorFactory

    DocumentExtractorFactory->>FileValidator: validate_file(file_path, max_size_mb)
    activate FileValidator
    FileValidator-->>DocumentExtractorFactory: validation_result
    deactivate FileValidator

    DocumentExtractorFactory->>DocumentTypeDetector: detect_type(file_path)
    activate DocumentTypeDetector
    DocumentTypeDetector-->>DocumentExtractorFactory: document_type
    deactivate DocumentTypeDetector

    DocumentExtractorFactory->>ExtractorRegistry: get_best_extractor_for_type(document_type)
    activate ExtractorRegistry
    ExtractorRegistry-->>DocumentExtractorFactory: selected_extractor
    deactivate ExtractorRegistry

    DocumentExtractorFactory->>selected_extractor: extract(file_path, config)
    activate selected_extractor
    selected_extractor-->>DocumentExtractorFactory: extracted_content
    deactivate selected_extractor

    DocumentExtractorFactory-->>Client: ExtractedContent
    deactivate DocumentExtractorFactory
```
