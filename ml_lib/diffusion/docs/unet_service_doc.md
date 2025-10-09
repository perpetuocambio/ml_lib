# UNet Service - Documentación Técnica Detallada

## Tabla de Contenidos
1. [Visión General](#vision-general)
2. [Arquitectura del Servicio](#arquitectura)
3. [Estructura de Capas](#capas)
4. [Flujos de Predicción](#flujos)
5. [Gestión de Condicionamiento](#condicionamiento)
6. [Optimizaciones](#optimizaciones)

---

## 1. Visión General {#vision-general}

### Propósito
El UNet es el componente central del proceso de difusión, responsable de predecir el ruido en cada paso del proceso iterativo. Opera en el espacio latente y utiliza múltiples formas de condicionamiento para guiar la generación.

### Responsabilidades
- **Predicción de Ruido**: Estimar ruido en latents para cada timestep
- **Condicionamiento Multi-modal**: Integrar texto, imágenes, máscaras
- **Cross-Attention**: Fusionar embeddings de texto con features visuales
- **Gestión de Adaptadores**: Inyectar LoRA, ControlNet, IP-Adapter
- **Self-Attention**: Capturar relaciones espaciales en la imagen
- **Residual Connections**: Mantener información entre bloques

### Relación con Otros Servicios

```mermaid
graph TB
    UNet[UNet Service<br/>Noise Predictor]
    
    subgraph "Input Conditioning"
        Latents[Latent Tensor<br/>From VAE/Scheduler]
        TextEmb[Text Embeddings<br/>From Text Encoder]
        Timestep[Timestep<br/>From Scheduler]
        ControlHint[Control Hints<br/>From ControlNet]
        IPEmb[Image Embeddings<br/>From IP-Adapter]
    end
    
    subgraph "Adapter Injections"
        LoRA[LoRA Weights]
        CN[ControlNet]
        IPA[IP-Adapter]
    end
    
    subgraph "Outputs"
        NoisePred[Noise Prediction<br/>To Scheduler]
    end
    
    Latents --> UNet
    TextEmb --> UNet
    Timestep --> UNet
    ControlHint --> UNet
    IPEmb --> UNet
    
    LoRA -.->|Inject| UNet
    CN -.->|Inject| UNet
    IPA -.->|Inject| UNet
    
    UNet --> NoisePred
    
    style UNet fill:#FFD700
    style NoisePred fill:#90EE90
```

---

## 2. Arquitectura del Servicio {#arquitectura}

### 2.1 Estructura General

```mermaid
graph TB
    subgraph "UNet Service Architecture"
        Core[UNet Core]
        
        subgraph "Model Management"
            Loader[Model Loader]
            StateManager[State Manager]
            WeightManager[Weight Manager]
        end
        
        subgraph "Layer Management"
            DownBlocks[Down Blocks]
            MidBlock[Middle Block]
            UpBlocks[Up Blocks]
            AttentionLayers[Attention Layers]
        end
        
        subgraph "Conditioning System"
            TextCond[Text Conditioning]
            TimeCond[Time Embedding]
            ControlCond[Control Conditioning]
            IPCond[IP-Adapter Conditioning]
        end
        
        subgraph "Adapter Integration"
            LoRAInjector[LoRA Injector]
            CNInjector[ControlNet Injector]
            IPAInjector[IP-Adapter Injector]
            AdapterStack[Adapter Stack Manager]
        end
        
        Core --> Loader
        Core --> StateManager
        Core --> WeightManager
        
        Core --> DownBlocks
        Core --> MidBlock
        Core --> UpBlocks
        Core --> AttentionLayers
        
        Core --> TextCond
        Core --> TimeCond
        Core --> ControlCond
        Core --> IPCond
        
        Core --> LoRAInjector
        Core --> CNInjector
        Core --> IPAInjector
        Core --> AdapterStack
    end
```

### 2.2 Modelo de Datos

```mermaid
classDiagram
    class UNetConfig {
        +str model_type
        +int in_channels
        +int out_channels
        +list~int~ block_out_channels
        +int layers_per_block
        +int attention_head_dim
        +bool use_linear_projection
        +str cross_attention_dim
        
        +validate() bool
        +get_layer_names() list
    }
    
    class UNetState {
        +dict layer_weights
        +dict active_adapters
        +dict attention_masks
        +list injection_points
        +datetime last_update
        
        +snapshot() StateSnapshot
        +restore(snapshot) void
    }
    
    class PredictionContext {
        +ndarray latents
        +ndarray text_embeddings
        +int timestep
        +float guidance_scale
        +dict control_hints
        +dict adapter_scales
        
        +prepare_inputs() dict
        +apply_guidance() ndarray
    }
    
    class AttentionConfig {
        +int num_heads
        +int head_dim
        +float dropout
        +bool use_flash_attention
        +bool use_xformers
        
        +calculate_attention() ndarray
    }
    
    UNetConfig --> UNetState
    UNetState --> PredictionContext
    AttentionConfig --> UNetConfig
```

---

## 3. Estructura de Capas {#capas}

### 3.1 Arquitectura Completa del UNet

```mermaid
graph TB
    Input[Latent Input<br/>+ Timestep + Text Emb]
    
    subgraph "Input Processing"
        TimeEmb[Time Embedding<br/>MLP]
        TextProj[Text Projection]
    end
    
    Input --> TimeEmb
    Input --> TextProj
    
    subgraph "Encoder Path - Down Blocks"
        Down1[Down Block 1<br/>ResNet + Attn<br/>320 channels]
        Down2[Down Block 2<br/>ResNet + Attn<br/>640 channels]
        Down3[Down Block 3<br/>ResNet + Attn<br/>1280 channels]
        Down4[Down Block 4<br/>ResNet only<br/>1280 channels]
        
        Down1 --> DS1[Downsample 1<br/>2x reduction]
        DS1 --> Down2
        Down2 --> DS2[Downsample 2<br/>2x reduction]
        DS2 --> Down3
        Down3 --> DS3[Downsample 3<br/>2x reduction]
        DS3 --> Down4
    end
    
    TimeEmb --> Down1
    TextProj --> Down1
    
    subgraph "Middle Block"
        Mid[Mid Block<br/>ResNet + Attn + ResNet<br/>1280 channels]
    end
    
    Down4 --> Mid
    
    subgraph "Decoder Path - Up Blocks"
        Up1[Up Block 1<br/>ResNet + Attn<br/>1280 channels]
        Up2[Up Block 2<br/>ResNet + Attn<br/>1280 channels]
        Up3[Up Block 3<br/>ResNet + Attn<br/>640 channels]
        Up4[Up Block 4<br/>ResNet + Attn<br/>320 channels]
        
        US1[Upsample 1<br/>2x increase]
        US2[Upsample 2<br/>2x increase]
        US3[Upsample 3<br/>2x increase]
        
        Mid --> Up1
        Up1 --> US1
        US1 --> Up2
        Up2 --> US2
        US2 --> Up3
        Up3 --> US3
        US3 --> Up4
    end
    
    subgraph "Skip Connections"
        Down1 -.->|Concat| Up4
        Down2 -.->|Concat| Up3
        Down3 -.->|Concat| Up2
        Down4 -.->|Concat| Up1
    end
    
    subgraph "Output Processing"
        OutConv[Output Conv<br/>Group Norm + SiLU]
        Up4 --> OutConv
    end
    
    OutConv --> Output[Noise Prediction]
    
    style Input fill:#E6E6FA
    style Output fill:#90EE90
    style Mid fill:#FFB6C1
```

### 3.2 Bloque Residual (ResNet)

```mermaid
graph LR
    Input[Input Features]
    
    Input --> GN1[Group Norm 1]
    GN1 --> Act1[SiLU Activation]
    Act1 --> Conv1[Conv 3×3]
    
    Conv1 --> TimeAdd[+ Time Embedding]
    TimeAdd --> GN2[Group Norm 2]
    GN2 --> Act2[SiLU Activation]
    Act2 --> Drop[Dropout]
    Drop --> Conv2[Conv 3×3]
    
    Input --> Skip{Channel<br/>Match?}
    Skip -->|No| SkipConv[Conv 1×1]
    Skip -->|Yes| Add
    SkipConv --> Add
    
    Conv2 --> Add[+ Residual]
    Add --> Output[Output Features]
    
    style Input fill:#E6E6FA
    style Output fill:#90EE90
```

### 3.3 Bloque de Atención

```mermaid
graph TB
    Input[Input Features]
    
    subgraph "Self-Attention"
        Input --> Norm1[Layer Norm]
        Norm1 --> QKV1[Linear Q, K, V<br/>Spatial attention]
        QKV1 --> SelfAttn[Multi-Head Self-Attention<br/>Relate spatial positions]
        SelfAttn --> Proj1[Output Projection]
        Proj1 --> Res1[+ Residual 1]
    end
    
    subgraph "Cross-Attention"
        Res1 --> Norm2[Layer Norm]
        Norm2 --> Q[Linear Q<br/>from features]
        
        TextEmb[Text Embeddings] --> KV[Linear K, V<br/>from text]
        
        Q --> CrossAttn[Multi-Head Cross-Attention<br/>Condition on text]
        KV --> CrossAttn
        
        CrossAttn --> Proj2[Output Projection]
        Proj2 --> Res2[+ Residual 2]
    end
    
    subgraph "Feed-Forward"
        Res2 --> Norm3[Layer Norm]
        Norm3 --> FF1[Linear + GELU]
        FF1 --> FF2[Linear]
        FF2 --> Res3[+ Residual 3]
    end
    
    Res3 --> Output[Output Features]
    
    style Input fill:#E6E6FA
    style Output fill:#90EE90
    style TextEmb fill:#FFE4B5
```

---

## 4. Flujos de Predicción {#flujos}

### 4.1 Flujo Completo de Predicción de Ruido

```mermaid
sequenceDiagram
    participant Scheduler
    participant UNet
    participant Encoder
    participant Middle
    participant Decoder
    participant Adapters
    
    Scheduler->>UNet: predict_noise(latents, t, text_emb)
    
    Note over UNet: Preparación de Inputs
    UNet->>UNet: embed_timestep(t)
    UNet->>UNet: prepare_text_conditioning(text_emb)
    
    Note over UNet: Encoder Path
    loop Down Blocks (1-4)
        UNet->>Encoder: process_down_block()
        
        alt Attention Block
            Encoder->>Encoder: self_attention()
            Encoder->>Encoder: cross_attention(text_emb)
        end
        
        alt LoRA Active
            Encoder->>Adapters: apply_lora_weights()
        end
        
        Encoder->>UNet: down_features + skip_connection
        UNet->>UNet: store_skip_connection()
    end
    
    Note over UNet: Middle Block
    UNet->>Middle: process_middle_block()
    Middle->>Middle: self_attention()
    Middle->>Middle: cross_attention(text_emb)
    
    alt ControlNet Active
        Middle->>Adapters: add_controlnet_residuals()
    end
    
    Middle->>UNet: middle_features
    
    Note over UNet: Decoder Path
    loop Up Blocks (1-4)
        UNet->>UNet: retrieve_skip_connection()
        UNet->>Decoder: process_up_block(concat_skip)
        
        alt Attention Block
            Decoder->>Decoder: self_attention()
            Decoder->>Decoder: cross_attention(text_emb)
        end
        
        alt IP-Adapter Active
            Decoder->>Adapters: apply_ip_adapter()
        end
        
        Decoder->>UNet: up_features
    end
    
    Note over UNet: Output
    UNet->>UNet: final_conv()
    UNet-->>Scheduler: noise_prediction
```

### 4.2 Classifier-Free Guidance

```mermaid
graph TB
    Input[Latents + Timestep]
    
    Input --> Duplicate[Duplicate Batch<br/>2× size]
    
    Duplicate --> Cond[Conditional Path<br/>With text embeddings]
    Duplicate --> Uncond[Unconditional Path<br/>Empty/negative prompt]
    
    Cond --> UNetCond[UNet Forward<br/>Conditional]
    Uncond --> UNetUncond[UNet Forward<br/>Unconditional]
    
    UNetCond --> NoiseCond[Noise Pred<br/>Conditional]
    UNetUncond --> NoiseUncond[Noise Pred<br/>Unconditional]
    
    NoiseCond --> Guidance[Apply Guidance Scale<br/>noise = uncond + scale × (cond - uncond)]
    NoiseUncond --> Guidance
    
    Guidance --> Output[Guided Noise Prediction]
    
    style Input fill:#E6E6FA
    style Output fill:#90EE90
    style Guidance fill:#FFE4B5
```

**Fórmula del Guidance:**
```
noise_pred = noise_uncond + guidance_scale × (noise_cond - noise_uncond)
```

### 4.3 Inyección de Adaptadores en Forward Pass

```mermaid
sequenceDiagram
    participant Layer
    participant BaseWeights
    participant LoRA
    participant ControlNet
    participant IPAdapter
    
    Note over Layer: Inicio de Layer Forward
    
    Layer->>BaseWeights: get_base_weights()
    BaseWeights-->>Layer: base_W
    
    alt LoRA Active
        Layer->>LoRA: get_lora_delta(layer_name)
        LoRA-->>Layer: lora_delta
        Layer->>Layer: W = base_W + alpha × lora_delta
    else No LoRA
        Layer->>Layer: W = base_W
    end
    
    Layer->>Layer: compute_output = W @ input
    
    alt ControlNet Active
        Layer->>ControlNet: get_control_residual(layer_idx)
        ControlNet-->>Layer: control_residual
        Layer->>Layer: output += control_residual
    end
    
    alt IP-Adapter Active
        Layer->>IPAdapter: get_image_conditioning()
        IPAdapter-->>Layer: image_features
        Layer->>Layer: apply_cross_attention(image_features)
    end
    
    Layer-->>Layer: return modified_output
```

---

## 5. Gestión de Condicionamiento {#condicionamiento}

### 5.1 Sistema de Condicionamiento Multi-Modal

```mermaid
graph TB
    subgraph "Conditioning System"
        CM[Conditioning Manager]
        
        subgraph "Text Conditioning"
            PromptEmb[Prompt Embeddings<br/>77×768 or 77×1024]
            NegEmb[Negative Embeddings]
            PooledEmb[Pooled Embeddings<br/>SDXL]
        end
        
        subgraph "Time Conditioning"
            TimeEmb[Timestep Embedding<br/>Sinusoidal]
            TimeProj[Time MLP<br/>Project to model dim]
        end
        
        subgraph "Spatial Conditioning"
            ControlHints[Control Hints<br/>ControlNet]
            Masks[Masks<br/>Inpainting]
            DepthMaps[Depth Maps]
        end
        
        subgraph "Image Conditioning"
            IPEmbeddings[IP-Adapter<br/>Image embeddings]
            RefImages[Reference Images]
        end
        
        CM --> PromptEmb
        CM --> NegEmb
        CM --> PooledEmb
        
        CM --> TimeEmb
        CM --> TimeProj
        
        CM --> ControlHints
        CM --> Masks
        CM --> DepthMaps
        
        CM --> IPEmbeddings
        CM --> RefImages
    end
    
    style CM fill:#FFD700
```

### 5.2 Fusión de Condicionamientos

```mermaid
graph LR
    subgraph "Conditioning Fusion"
        Text[Text Conditioning]
        Time[Time Conditioning]
        Control[Control Conditioning]
        Image[Image Conditioning]
        
        Text --> CrossAttn[Cross-Attention Layers<br/>Text → Features]
        
        Time --> AddEmbed[Add to Features<br/>Time → ResNet blocks]
        
        Control --> AddResidual[Add Residuals<br/>Control → Skip connections]
        
        Image --> CrossAttnImg[Cross-Attention Layers<br/>Image → Features]
        
        CrossAttn --> Fused[Fused Features]
        AddEmbed --> Fused
        AddResidual --> Fused
        CrossAttnImg --> Fused
        
        Fused --> Output[Conditioned Output]
    end
    
    style Text fill:#E6E6FA
    style Output fill:#90EE90
```

### 5.3 Attention Masking

```mermaid
graph TB
    Query[Query Tensor<br/>Spatial features]
    Key[Key Tensor<br/>Text tokens]
    
    Query --> QProj[Q Projection]
    Key --> KProj[K Projection]
    
    QProj --> Scores[Compute Scores<br/>Q @ K^T / √d]
    KProj --> Scores
    
    Scores --> Mask{Apply<br/>Mask?}
    
    Mask -->|Yes| ApplyMask[Mask Invalid Positions<br/>Set to -inf]
    Mask -->|No| Softmax
    
    ApplyMask --> Softmax[Softmax<br/>Normalize scores]
    
    Value[Value Tensor] --> VProj[V Projection]
    
    Softmax --> AttnWeights[Attention Weights]
    VProj --> Multiply[Weighted Sum<br/>Weights @ V]
    AttnWeights --> Multiply
    
    Multiply --> Output[Attention Output]
    
    style Query fill:#E6E6FA
    style Output fill:#90EE90
```

---

## 6. Optimizaciones {#optimizaciones}

### 6.1 Estrategias de Optimización

```mermaid
mindmap
    root((UNet<br/>Optimizations))
        Memory
            Gradient Checkpointing
            Attention Slicing
            VAE Slicing
            CPU Offloading
        Attention
            Flash Attention
            xFormers
            Memory Efficient Attention
            Sparse Attention
        Computation
            Torch Compile
            CUDA Graphs
            Mixed Precision FP16
            Operator Fusion
        Model
            Pruning
            Quantization
            Knowledge Distillation
            Layer Freezing
```

### 6.2 Attention Optimization

```mermaid
graph TB
    subgraph "Attention Optimization Strategies"
        Standard[Standard Attention<br/>O(N²) memory]
        
        Standard --> Choose{Choose<br/>Strategy}
        
        Choose -->|Memory Critical| Flash[Flash Attention<br/>O(N) memory<br/>Tiling + recompute]
        
        Choose -->|Speed Critical| xFormers[xFormers Attention<br/>Block-sparse<br/>Optimized kernels]
        
        Choose -->|Balanced| MemEfficient[Memory Efficient<br/>Chunked processing<br/>Lower peak memory]
        
        Flash --> Benchmark[Benchmark Performance]
        xFormers --> Benchmark
        MemEfficient --> Benchmark
        
        Benchmark --> Select[Select Best for Hardware]
    end
    
    style Standard fill:#FFB6C1
    style Select fill:#90EE90
```

### 6.3 Gradient Checkpointing

```mermaid
sequenceDiagram
    participant Forward
    participant Checkpoints
    participant Backward
    
    Note over Forward: Forward Pass
    
    Forward->>Forward: Layer 1
    Forward->>Checkpoints: Save checkpoint 1
    Forward->>Forward: Layer 2
    Forward->>Forward: Layer 3
    Forward->>Checkpoints: Save checkpoint 2
    Forward->>Forward: Layer 4
    
    Note over Forward: Only checkpoints kept,<br/>intermediate activations freed
    
    Note over Backward: Backward Pass
    
    Backward->>Backward: Compute grad Layer 4
    Backward->>Checkpoints: Retrieve checkpoint 2
    Backward->>Backward: Recompute Layer 3
    Backward->>Backward: Compute grad Layer 3
    Backward->>Backward: Recompute Layer 2
    Backward->>Backward: Compute grad Layer 2
    Backward->>Checkpoints: Retrieve checkpoint 1
    Backward->>Backward: Compute grad Layer 1
    
    Note over Backward: Trade computation for memory<br/>~2× slower, ~2× less memory
```

### 6.4 Gestión de Memoria por Layers

```mermaid
graph LR
    subgraph "Layer Memory Management"
        Input[Layer Input]
        
        Input --> Check{Memory<br/>Available?}
        
        Check -->|High| FullPrecision[Full Precision FP32<br/>Best quality]
        Check -->|Medium| MixedPrecision[Mixed Precision FP16<br/>Balance]
        Check -->|Low| Optimized[Aggressive Optimization<br/>Slicing + Checkpointing]
        
        FullPrecision --> Process[Process Layer]
        MixedPrecision --> Process
        Optimized --> Process
        
        Process --> Monitor[Monitor Peak Memory]
        Monitor --> Adjust{Adjust<br/>Needed?}
        
        Adjust -->|Yes| Strategy[Change Strategy]
        Adjust -->|No| Continue[Continue]
        
        Strategy --> Check
    end
    
    style Input fill:#E6E6FA
    style Continue fill:#90EE90
```

---

## 7. Integración con Adaptadores

### 7.1 Stack de Adaptadores en UNet

```mermaid
graph TB
    BaseUNet[Base UNet Weights]
    
    subgraph "Adapter Stack - Applied in Order"
        L1[Layer 1: LoRA<br/>Style modification<br/>Priority: 1]
        L2[Layer 2: LoRA<br/>Character<br/>Priority: 2]
        L3[Layer 3: ControlNet<br/>Pose control<br/>Priority: 3]
        L4[Layer 4: IP-Adapter<br/>Image style<br/>Priority: 4]
    end
    
    BaseUNet --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> FinalWeights[Modified UNet]
    
    style BaseUNet fill:#E6E6FA
    style FinalWeights fill:#90EE90
```

### 7.2 Resolución de Conflictos entre Adaptadores

```mermaid
graph TB
    Conflict[Adapter Conflict Detected]
    
    Conflict --> Type{Conflict<br/>Type}
    
    Type -->|Same Layer| LayerConflict[Multiple adapters<br/>on same layer]
    Type -->|Resource| ResourceConflict[Memory/compute<br/>constraint]
    Type -->|Incompatible| IncompatConflict[Incompatible<br/>modifications]
    
    LayerConflict --> Merge[Merge Strategy<br/>Weighted combination]
    ResourceConflict --> Prioritize[Prioritize<br/>by importance]
    IncompatConflict --> Disable[Disable<br/>conflicting adapter]
    
    Merge --> Resolve[Conflict Resolved]
    Prioritize --> Resolve
    Disable --> Resolve
    
    style Conflict fill:#FFB6C1
    style Resolve fill:#90EE90
```

### 7.3 Puntos de Inyección de Adaptadores

```mermaid
graph LR
    subgraph "UNet Injection Points"
        Down[Down Blocks]
        Mid[Middle Block]
        Up[Up Blocks]
        
        subgraph "Down Block Injections"
            D1[Down 1<br/>- LoRA: Q,K,V,Out<br/>- ControlNet: Add residual]
            D2[Down 2<br/>- LoRA: Q,K,V,Out<br/>- ControlNet: Add residual]
            D3[Down 3<br/>- LoRA: Q,K,V,Out<br/>- ControlNet: Add residual]
        end
        
        subgraph "Middle Block Injections"
            M1[Middle<br/>- LoRA: All attention<br/>- IP-Adapter: Cross-attn<br/>- ControlNet: Strong residual]
        end
        
        subgraph "Up Block Injections"
            U1[Up 1<br/>- LoRA: Q,K,V,Out<br/>- IP-Adapter: Cross-attn]
            U2[Up 2<br/>- LoRA: Q,K,V,Out<br/>- IP-Adapter: Cross-attn]
            U3[Up 3<br/>- LoRA: Q,K,V,Out]
        end
        
        Down --> D1
        Down --> D2
        Down --> D3
        
        Mid --> M1
        
        Up --> U1
        Up --> U2
        Up --> U3
    end
```

---

## 8. Monitoreo y Debugging

### 8.1 Métricas del UNet

```mermaid
graph TB
    subgraph "UNet Metrics Dashboard"
        Perf[Performance]
        Quality[Quality]
        Resource[Resources]
        Adapters[Adapters]
        
        Perf --> PerfMetrics[- Forward pass time<br/>- Attention compute time<br/>- Layer-wise latency<br/>- Throughput]
        
        Quality --> QualMetrics[- Noise prediction accuracy<br/>- Feature statistics<br/>- Gradient norms<br/>- Loss values]
        
        Resource --> ResMetrics[- Memory per layer<br/>- GPU utilization<br/>- Activation memory<br/>- Parameter count]
        
        Adapters --> AdapterMetrics[- Active adapters<br/>- Injection overhead<br/>- Adapter impact<br/>- Conflict status]
    end
    
    style Perf fill:#E6E6FA
    style Quality fill:#FFE4B5
```

### 8.2 Debugging Flow

```mermaid
sequenceDiagram
    participant User
    participant UNet
    participant Debugger
    participant Visualizer
    
    User->>UNet: generate() with debug=True
    
    UNet->>Debugger: enable_layer_hooks()
    
    loop Each Layer
        UNet->>Debugger: log_layer_input(layer_name)
        UNet->>UNet: process_layer()
        UNet->>Debugger: log_layer_output(layer_name)
        UNet->>Debugger: log_attention_maps()
    end
    
    UNet->>Debugger: log_final_prediction()
    
    User->>Debugger: get_debug_info()
    Debugger-->>User: layer_statistics
    
    User->>Visualizer: visualize_attention()
    Visualizer-->>User: attention_heatmaps
    
    User->>Visualizer: visualize_activations()
    Visualizer-->>User: activation_plots
```

### 8.3 Troubleshooting Common Issues

```mermaid
graph TB
    Issue[UNet Issue]
    
    Issue --> Type{Issue Type}
    
    Type -->|NaN Values| NaN[NaN in Output]
    Type -->|Poor Quality| Quality[Bad Generation]
    Type -->|Slow| Performance[Slow Inference]
    Type -->|Memory| Memory[OOM Error]
    
    NaN --> NaNSol[- Check gradient clipping<br/>- Reduce learning rate<br/>- Check adapter weights<br/>- Validate input ranges]
    
    Quality --> QualSol[- Verify text embeddings<br/>- Check adapter scales<br/>- Validate timesteps<br/>- Review conditioning]
    
    Performance --> PerfSol[- Enable flash attention<br/>- Use torch.compile<br/>- Reduce attention heads<br/>- Enable gradient checkpointing]
    
    Memory --> MemSol[- Enable attention slicing<br/>- Use FP16<br/>- Reduce batch size<br/>- Offload to CPU]
    
    style Issue fill:#FFB6C1
    style NaNSol fill:#90EE90
    style QualSol fill:#90EE90
    style PerfSol fill:#90EE90
    style MemSol fill:#90EE90
```