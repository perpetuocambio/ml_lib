# Capa de Infraestructura (`infrastructure`)

La capa de infraestructura de `pyintelcivil` es responsable de manejar los detalles técnicos y externos de la aplicación. Actúa como un adaptador entre la lógica de negocio del dominio y los servicios externos, como sistemas de archivos, bases de datos, servicios de LLM y configuraciones.

## Submódulos

-   **[config](./config/README.md)**: Contiene la configuración de la aplicación y de los servicios de infraestructura.
-   **[data_loaders](./data_loaders/README.md)**: Proporciona servicios para cargar datos desde diversas fuentes y formatos, transformándolos en entidades tipadas del dominio.
-   **[fact_types](./fact_types/README.md)**: Gestiona la carga y definición de los tipos de hechos (`fact types`) utilizados en el sistema.
-   **[llm](./llm/README.md)**: Proporciona la infraestructura necesaria para interactuar con Modelos de Lenguaje Grandes (LLMs).

## Arquitectura General

La capa de infraestructura sigue los principios de una arquitectura limpia (Clean Architecture) o hexagonal, donde las dependencias fluyen hacia adentro, pero esta capa es la más externa. Esto significa que la infraestructura depende del dominio, pero el dominio no depende de la infraestructura. Esto permite que los detalles de implementación técnica cambien sin afectar la lógica de negocio central.

## Principios Clave

-   **Adaptadores**: Actúa como un conjunto de adaptadores para servicios externos.
-   **Desacoplamiento**: Mantiene la lógica de negocio desacoplada de los detalles técnicos.
-   **Intercambiabilidad**: Permite cambiar implementaciones de infraestructura (e.g., diferentes bases de datos, diferentes proveedores de LLM) con un impacto mínimo.
