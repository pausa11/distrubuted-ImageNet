# Distributed ImageNet Training with Hivemind & GCS

Este proyecto implementa un entrenamiento distribuido de ResNet50 sobre ImageNet utilizando [Hivemind](https://github.com/learning-at-home/hivemind). Una de las características principales es que **no requiere descargar el dataset localmente**; los datos se transmiten (stream) directamente desde un bucket de Google Cloud Storage (GCS) durante el entrenamiento.

## Requisitos Previos

1.  **Python 3.8+** y las dependencias instaladas (`torch`, `hivemind`, `google-cloud-storage`, etc.).
2.  **Google Cloud SDK** instalado para la autenticación.

## Configuración y Autenticación

Para que el script pueda leer las imágenes de tu bucket privado de GCS, necesitas configurar las "Application Default Credentials". Ejecuta el siguiente comando en tu terminal:

```bash
gcloud auth application-default login
```

Esto abrirá una ventana en tu navegador. Inicia sesión con la cuenta de Google que tiene acceso al bucket. Una vez completado, se guardará un archivo JSON con tus credenciales que las librerías de Google usarán automáticamente.

## Cómo Ejecutar

Para iniciar un nodo de entrenamiento (peer), ejecuta el siguiente comando desde la carpeta raíz del proyecto:

```bash
python3 src/dhtPeerImagenet.py \
  --bucket_name caso-estudio-2 \
  --train_prefix ILSVRC2012_img_train \
  --val_prefix ILSVRC2012_img_val \
  --device mps
```

### Explicación de los argumentos:

*   `--bucket_name`: El nombre de tu bucket en Google Cloud (ej: `caso-estudio-2`).
*   `--train_prefix`: La ruta o carpeta dentro del bucket donde se encuentran las imágenes de entrenamiento.
*   `--val_prefix`: La ruta o carpeta dentro del bucket donde se encuentran las imágenes de validación.
*   `--device`: El dispositivo de aceleración a utilizar:
    *   `mps`: Para Macs con Apple Silicon (M1/M2/M3).
    *   `cuda`: Para GPUs NVIDIA.
    *   `cpu`: Para usar solo el procesador (más lento).

## Notas Adicionales

*   **Primer inicio**: Al arrancar, el script listará todos los archivos del bucket para crear un índice. Esto puede tomar unos minutos dependiendo de la cantidad de imágenes, pero solo ocurre al inicio.
*   **Entrenamiento Distribuido**: Hivemind se encarga de coordinar el entrenamiento con otros peers que se conecten al mismo experimento.