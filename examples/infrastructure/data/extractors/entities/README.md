# Módulo de Entidades (`pyintelcivil.src.infrastructure.extractors.entities`)

Este módulo define las estructuras de datos (dataclasses) utilizadas en el proceso de extracción de documentos. Estas entidades aseguran un tipado estricto y una representación clara de la información extraída y la configuración del extractor.

## Clases

-   `CustomParameters`: Parámetros personalizados para extractores.
-   `DocumentFormatMapping`: Mapea un tipo de documento a un formato de entrada.
-   `DocumentMetadata`: Metadatos extraídos del documento.
-   `DocumentStructure`: Estructura del documento extraído.
-   `ExtensionMapEntry`: Mapea una extensión de archivo a un tipo de documento.
-   `ExtensionMapping`: Mapea extensiones de archivo a tipos de documento.
-   `ExtractedContent`: Contenido extraído de un documento.
-   `ExtractionCapabilities`: Capacidades de un extractor.
-   `ExtractionConfig`: Configuración para la extracción de documentos.
-   `ExtractionError`: Excepción personalizada para errores de extracción.
-   `ExtractorRegistration`: Registra un extractor con su estrategia.
-   `ImageInfo`: Información sobre una imagen extraída.
-   `MimeTypeMapEntry`: Mapea un tipo MIME a un tipo de documento.
-   `MimeTypeMapping`: Mapea tipos MIME a tipos de documento.
-   `StructuralElement`: Elemento estructural del documento.
-   `TableInfo`: Información sobre una tabla extraída.

## Diagrama de Clases

```mermaid
classDiagram
    direction LR

    class CustomParameters {
        +docling_extract_links: bool
        +docling_process_images: bool
        +docling_table_extraction: bool
        +markitdown_preserve_links: bool
        +markitdown_clean_html: bool
        +additional_options: str
    }

    class DocumentFormatMapping {
        +document_type: DocumentType
        +input_format: InputFormat
    }

    class DocumentMetadata {
        +file_path: Path
        +file_size: int
        +document_type: DocumentType
        +title: str
        +author: str
        +creator: str
        +subject: str
        +keywords: list~str~
        +creation_date: datetime
        +modification_date: datetime
        +page_count: int
        +word_count: int
        +language: str
        +has_custom_metadata: bool
        +custom_metadata_preview: str
    }

    class DocumentStructure {
        +line_count: int
        +word_count: int
        +character_count: int
        +paragraph_count: int
        +header_count: int
        +list_count: int
        +code_block_count: int
        +link_count: int
        +bold_text_count: int
        +italic_text_count: int
        +elements: list~StructuralElement~
    }

    class ExtensionMapEntry {
        +extension: str
        +document_type: DocumentType
    }

    class ExtensionMapping {
        +mappings: list~ExtensionMapEntry~
    }

    class ExtractedContent {
        +text: str
        +metadata: DocumentMetadata
        +images: list~ImageInfo~
        +tables: list~TableInfo~
        +structure: DocumentStructure
        +extraction_time: float
        +extraction_strategy: ExtractionStrategy
        +success: bool
        +error_message: str
    }

    class ExtractionCapabilities {
        +supported_types: list~DocumentType~
        +can_extract_images: bool
        +can_extract_tables: bool
        +can_extract_metadata: bool
        +can_preserve_formatting: bool
        +supports_ocr: bool
        +max_file_size_mb: int
    }

    class ExtractionConfig {
        +strategy: ExtractionStrategy
        +preserve_formatting: bool
        +extract_images: bool
        +extract_tables: bool
        +extract_metadata: bool
        +output_format: OutputFormat
        +max_file_size_mb: int
        +timeout_seconds: int
        +custom_params: CustomParameters
    }

    class ExtractionError {
        +message: str
        +file_path: Path
        +strategy: ExtractionStrategy
    }

    class ExtractorRegistration {
        +strategy: ExtractionStrategy
        +extractor: BaseDocumentExtractor
    }

    class ImageInfo {
        +index: int
        +image_type: str
        +width: int
        +height: int
        +format_type: str
        +alt_text: str
        +url: str
        +position: int
        +data_available: bool
    }

    class MimeTypeMapEntry {
        +mime_type: str
        +document_type: DocumentType
    }

    class MimeTypeMapping {
        +mappings: list~MimeTypeMapEntry~
    }

    class StructuralElement {
        +element_type: str
        +text_content: str
        +page_number: int
        +position: int
        +level: int
    }

    class TableInfo {
        +index: int
        +table_type: str
        +rows: int
        +columns: int
        +start_line: int
        +headers: list~str~
        +has_data: bool
        +data_preview: str
    }

    DocumentMetadata "1" *-- "1" DocumentType
    DocumentStructure "1" *-- "*" StructuralElement
    ExtractedContent "1" *-- "1" DocumentMetadata
    ExtractedContent "1" *-- "*" ImageInfo
    ExtractedContent "1" *-- "*" TableInfo
    ExtractedContent "1" *-- "1" DocumentStructure
    ExtractedContent "1" *-- "1" ExtractionStrategy
    ExtractionConfig "1" *-- "1" ExtractionStrategy
    ExtractionConfig "1" *-- "1" OutputFormat
    ExtractionConfig "1" *-- "1" CustomParameters
    ExtractionCapabilities "1" *-- "*" DocumentType
    ExtensionMapping "1" *-- "*" ExtensionMapEntry
    MimeTypeMapping "1" *-- "*" MimeTypeMapEntry
    DocumentFormatMapping "1" *-- "1" DocumentType
    ExtractorRegistration "1" *-- "1" ExtractionStrategy
    ExtractorRegistration "1" *-- "1" BaseDocumentExtractor
    ExtractionError "1" *-- "1" ExtractionStrategy

    DocumentType <.. DocumentMetadata
    DocumentType <.. ExtensionMapEntry
    DocumentType <.. MimeTypeMapEntry
    DocumentType <.. ExtractionCapabilities
    DocumentType <.. DocumentFormatMapping

    ExtractionStrategy <.. ExtractedContent
    ExtractionStrategy <.. ExtractionConfig
    ExtractionStrategy <.. ExtractorRegistration
    ExtractionStrategy <.. ExtractionError

    OutputFormat <.. ExtractionConfig

    BaseDocumentExtractor <.. ExtractorRegistration
```
