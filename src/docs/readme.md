# Módulo de Diffusion Models con Soporte Completo para LoRA

Arquitectura para Stable Diffusion, Pony Diffusion, SDXL, y otros modelos

```text
ml_library/diffusion/
│
├── services/
│ ├── pipeline_service.py # Gestión de pipelines de difusión
│ ├── lora_service.py # Carga, merge y gestión de LoRAs
│ ├── checkpoint_service.py # Gestión de checkpoints base
│ ├── sampling_service.py # Schedulers y métodos de sampling
│ ├── conditioning_service.py # Text encoding y embeddings
│ ├── unet_service.py # Gestión del UNet y capas
│ └── vae_service.py # VAE encoding/decoding
│
├── handlers/
│ ├── lora_merge_handler.py # Fusión de múltiples LoRAs
│ ├── weight_injection_handler.py # Inyección de pesos en capas
│ ├── cross_attention_handler.py # Gestión de cross-attention LoRA
│ ├── safetensors_handler.py # Carga de archivos safetensors
│ ├── prompt_handler.py # Procesamiento de prompts
│ ├── latent_handler.py # Manipulación de espacio latente
│ └── controlnet_handler.py # Integración de ControlNet
│
├── interfaces/
│ ├── diffusion_model_interface.py # Contrato para modelos de difusión
│ ├── lora_interface.py # Contrato para LoRAs
│ ├── scheduler_interface.py # Interface para schedulers
│ ├── encoder_interface.py # Text/CLIP encoders
│ ├── pipeline_interface.py # Pipeline genérico
│ └── adapter_interface.py # Adaptadores genéricos
│
├── models/
│ ├── diffusion_checkpoint.py # Modelo checkpoint completo
│ ├── lora_weights.py # Pesos LoRA estructurados
│ ├── pipeline_config.py # Configuración de pipeline
│ ├── generation_params.py # Parámetros de generación
│ ├── latent_tensor.py # Representación latente
│ ├── attention_weights.py # Pesos de atención
│ └── adapter_config.py # Configuración de adaptadores
│
└── **init**.py
```
