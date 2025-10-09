# Módulo de Handlers (`pyintelcivil.src.infrastructure.extractors.handlers`)

Este módulo contiene las implementaciones concretas de los extractores de documentos, así como utilidades y lógicas de procesamiento que soportan el proceso de extracción. Aquí se encuentran las clases que interactúan directamente con librerías externas y realizan las tareas de limpieza, detección y registro.

## Clases

-   `ContentCleaner`: Limpia y procesa contenido extraído.
-   `DoclingExtractor`: Extractor de documentos usando la librería Docling.
-   `DoclingFormatMapper`: Mapea tipos de documento a formatos de Docling.
-   `DocumentTypeDetector`: Detecta tipos de documento basado en extensión y contenido.
-   `ExtractorRegistry`: Registro de extractores disponibles.
-   `FileValidator`: Valida archivos para extracción.
-   `MarkdownParser`: Parser para extraer información de texto markdown.
-   `MarkItDownExtractor`: Extractor de documentos usando la librería MarkItDown.
-   `TextCleaner`: Limpia texto extraído.

## Diagrama de Clases

```mermaid
classDiagram
    direction LR

    class ContentCleaner {
        +text_cleaner: TextCleaner
        +markdown_parser: MarkdownParser
        +clean_text(text): str
        +extract_structure(text): DocumentStructure
        +markdown_to_text(markdown_text): str
    }

    class DoclingExtractor {
        +FORMAT_MAP: dict~DocumentType, InputFormat~
    }

    class DoclingFormatMapper {
        +mappings: list~DocumentFormatMapping~
    }

    class DocumentTypeDetector {
        +extension_mapping: ExtensionMapping
        +mime_mapping: MimeTypeMapping
        +detect_type(file_path): DocumentType
    }

    class ExtractorRegistry {
        +extractors: list~ExtractorRegistration~
        +fallback_order: list~ExtractionStrategy~
        +register_extractor(strategy, extractor)
        +get_extractor(strategy): BaseDocumentExtractor
        +get_available_strategies(): list~ExtractionStrategy~
        +get_best_extractor_for_type(document_type): BaseDocumentExtractor
        +set_fallback_order(order)
    }

    class FileValidator {
        +validate_file(file_path, max_size_mb)
    }

    class MarkdownParser {
        +extract_headers(text): list~StructuralElement~
        +extract_lists(text): list~StructuralElement~
        +count_markdown_elements(text): DocumentStructure
    }

    class MarkItDownExtractor {
        +SUPPORTED_TYPES: list~DocumentType~
    }

    class TextCleaner {
        +clean_whitespace(text): str
        +remove_empty_lines(text): str
        +clean_text(text): str
    }

    ContentCleaner "1" *-- "1" TextCleaner
    ContentCleaner "1" *-- "1" MarkdownParser
    DocumentTypeDetector "1" *-- "1" ExtensionMapping
    DocumentTypeDetector "1" *-- "1" MimeTypeMapping
    ExtractorRegistry "1" *-- "*" ExtractorRegistration

    DoclingExtractor --|> BaseDocumentExtractor
    MarkItDownExtractor --|> BaseDocumentExtractor

    DoclingFormatMapper "1" *-- "*" DocumentFormatMapping

    DocumentStructure <.. MarkdownParser
    StructuralElement <.. MarkdownParser
    DocumentType <.. DoclingExtractor
    DocumentType <.. MarkItDownExtractor
    DocumentType <.. DoclingFormatMapper
    DocumentType <.. DocumentTypeDetector
    DocumentType <.. ExtractorRegistry
    ExtractionStrategy <.. ExtractorRegistry
    BaseDocumentExtractor <.. ExtractorRegistry
    ExtensionMapping <.. DocumentTypeDetector
    MimeTypeMapping <.. DocumentTypeDetector
    DocumentFormatMapping <.. DoclingFormatMapper
    ExtractorRegistration <.. ExtractorRegistry
```

## Diagrama de Secuencia (ContentCleaner)

```mermaid
sequenceDiagram
    participant Client
    participant ContentCleaner
    participant TextCleaner
    participant MarkdownParser

    Client->>ContentCleaner: clean_text(raw_text)
    activate ContentCleaner
    ContentCleaner->>TextCleaner: clean_text(raw_text)
    activate TextCleaner
    TextCleaner-->>ContentCleaner: cleaned_text
    deactivate TextCleaner
    ContentCleaner-->>Client: cleaned_text
    deactivate ContentCleaner

    Client->>ContentCleaner: extract_structure(text)
    activate ContentCleaner
    ContentCleaner->>MarkdownParser: count_markdown_elements(text)
    activate MarkdownParser
    MarkdownParser-->>ContentCleaner: document_structure
    deactivate MarkdownParser
    ContentCleaner-->>Client: document_structure
    deactivate ContentCleaner

    Client->>ContentCleaner: markdown_to_text(markdown_content)
    activate ContentCleaner
    ContentCleaner->>TextCleaner: clean_text(processed_text)
    activate TextCleaner
    TextCleaner-->>ContentCleaner: final_text
    deactivate TextCleaner
    ContentCleaner-->>Client: final_text
    deactivate ContentCleaner
```
