# Entrenamiento Distribuido de ImageNet con Hivemind

Este proyecto implementa un entrenamiento distribuido y descentralizado de ResNet50 sobre ImageNet utilizando [Hivemind](https://github.com/learning-at-home/hivemind). A continuación se detallan los componentes principales de la arquitectura.

## 1. Modelo
*   **Arquitectura:** ResNet50 (`torchvision.models.resnet50`).
*   **Configuración:**
    *   Se inicializa **sin pre-entrenamiento** (`pretrained=False`).
    *   La última capa lineal (`fc`) se adapta para **1000 clases** (ImageNet).
*   **Estrategia de Entrenamiento:**
    *   **Master (CPU):** Mantiene los pesos sincronizados con la DHT de Hivemind.
    *   **Shadow (Device):** Copia local en GPU/MPS para realizar el cómputo intensivo (forward/backward).

## 2. Optimizador
*   **Base:** SGD (`torch.optim.SGD`).
    *   Learning Rate: `0.1`
    *   Momentum: `0.9`
    *   Weight Decay: `1e-4`
*   **Distribuido:** `hivemind.Optimizer`.
    *   Envuelve al optimizador base para promediar gradientes y parámetros con otros pares (peers) a través de la DHT.
    *   Permite el entrenamiento colaborativo sin un nodo maestro central.
*   **Scheduler:** `MultiStepLR`.
    *   Hitos (Milestones): Épocas 30, 60, 90.
    *   Gamma: 0.1 (reducción del LR).

## 3. Carga de Datos (Data Loading)
El proyecto utiliza **WebDataset** para una carga de datos eficiente y escalable desde la nube.

*   **Formato:** Archivos `.tar` (shards) que se transmiten (stream) directamente.
*   **Fuente:** Google Cloud Storage (Bucket: `caso-estudio-2`).
*   **Estructura:**
    *   Entrenamiento: `imagenet-wds/train/train-{000000..000640}.tar`
    *   Validación: `imagenet-wds/val/train-{000000..000049}.tar`
*   **Pipeline:**
    1.  **Streaming:** Descarga de shards al vuelo usando `curl` (optimizado para evitar timeouts).
    2.  **Decodificación:** Conversión de bytes a imágenes PIL.
    3.  **Transformaciones:**
        *   *Train:* RandomResizedCrop, Flip, Normalización.
        *   *Val:* Resize, CenterCrop, Normalización.
    4.  **Mapeo de Clases:** Generación automática de índices (0-999) basada en la estructura de carpetas del bucket original.
*   **Optimización (Mac/MPS):**
    *   Se implementa `ThreadedWebDataset` para cuando `num_workers=0`.
    *   Utiliza un hilo en segundo plano para pre-cargar batches, evitando cuellos de botella de I/O en sistemas donde el multiprocessing de PyTorch es inestable.

## 4. Características Adicionales
*   **Métricas:** Registro secuencial de logs de sistema y entrenamiento en CSV (`stats/<hostname>/`).
*   **Checkpointing:**
    *   Guarda el último estado (`latest_checkpoint.pt`) en cada validación.
    *   Guarda el mejor modelo (`best_checkpoint.pt`) solo si mejora el accuracy.
*   **Descubrimiento de Pares (Peer Discovery):**
    *   Mecanismo basado en GCS para anunciar y descubrir IPs de pares automáticamente.
    *   Facilita la conexión inicial sin configuración manual de direcciones `multiaddr`.
