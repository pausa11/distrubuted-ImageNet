# Inicio Rápido - Configuración de Datasets

Esta guía te ayudará a configurar rápidamente datasets para el entrenamiento distribuido.

## 🚀 Opción Rápida (Tiny-ImageNet para pruebas)

```bash
# 1. Descargar dataset automáticamente
./scripts/setup_dataset.sh tiny-imagenet local

# 2. Iniciar entrenamiento
cd neural-network-train/src
python trainGossipImagenet.py --data_root ../data
```

## 📖 Guías Detalladas

### Para Experimentos Pequeños (<16 peers)
Ver [DATASET_HOSTING.md - Sección: Experimentos Pequeños](DATASET_HOSTING.md#1-experimentos-pequeños-3-16-peers-1gb)

### Para Experimentos Grandes (>16 peers, ImageNet1k)
Ver [DATASET_HOSTING.md - Sección: Experimentos Grandes](DATASET_HOSTING.md#2-experimentos-grandes-16-peers-1gb)

### Para Archivos Muy Grandes (>5GB)
Ver [DATASET_HOSTING.md - Sección: Manejo de Archivos Grandes](DATASET_HOSTING.md#manejo-de-archivos-grandes)

## 🎯 Casos de Uso Comunes

### Caso 1: Desarrollo Local (1-3 peers)
```bash
# Usar CIFAR-10 (descarga automática)
cd neural-network-train/src
python trainGossipCifar.py --data_root ../data
```

### Caso 2: Pruebas con Múltiples Máquinas (3-10 peers)
```bash
# En cada máquina:
./scripts/setup_dataset.sh tiny-imagenet local

# Luego seguir las instrucciones en README.md para iniciar peers
```

### Caso 3: Producción con ImageNet1k (>10 peers)
```bash
# Opción A: Academic Torrents (recomendado)
./scripts/setup_dataset.sh imagenet1k torrent

# Opción B: S3 (si tienes bucket configurado)
./scripts/setup_dataset.sh imagenet1k s3

# Ver DATASET_HOSTING.md para más opciones
```

## 📦 Estructura de Directorios Resultante

Después de ejecutar el script de setup:

```
distrubuted-ImageNet/
├── data/
│   ├── tiny-imagenet-200/      # Tiny-ImageNet (si lo descargaste)
│   │   ├── train/
│   │   │   ├── n01443537/     # 200 carpetas de clases
│   │   │   │   ├── images/    # Imágenes de entrenamiento
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── val/
│   │       ├── n01443537/     # Carpetas de validación
│   │       └── ...
│   │
│   ├── ILSVRC2012/             # ImageNet1k (opcional, grande)
│   │   ├── train/
│   │   └── val/
│   │
│   └── cifar-10-batches-py/    # CIFAR-10 (descarga automática)
│
└── checkpoints/                # Modelos guardados automáticamente
    ├── imagenet_tiny_resnet50_5011_best.pt
    └── ...
```

## 🔍 Verificación

Para verificar que el dataset está correctamente configurado:

```bash
# Ver estructura del dataset
tree -L 3 data/tiny-imagenet-200/

# Contar clases de entrenamiento
ls -d data/tiny-imagenet-200/train/*/ | wc -l
# Debería mostrar: 200

# Ver espacio utilizado
du -sh data/
```

## ⚠️ Troubleshooting

### Error: "No such file or directory: data/tiny-imagenet-200/train"
**Solución**: Ejecuta el script de setup primero:
```bash
./scripts/setup_dataset.sh tiny-imagenet local
```

### Error: "Connection timeout" durante descarga
**Solución**: El script usa `wget -c` para reanudar. Vuelve a ejecutar:
```bash
./scripts/setup_dataset.sh tiny-imagenet local
```

### Dataset corrupto o incompleto
**Solución**: Elimina y vuelve a descargar:
```bash
rm -rf data/tiny-imagenet-200
./scripts/setup_dataset.sh tiny-imagenet local
```

## 📚 Más Información

- **Guía Completa**: [DATASET_HOSTING.md](DATASET_HOSTING.md)
- **README Principal**: [README.md](README.md)
- **Dataloader con Chunks**: [neural-network-train/src/chunked_dataset.py](neural-network-train/src/chunked_dataset.py)

---

**Consejo**: Empieza con Tiny-ImageNet para familiarizarte con el sistema antes de escalar a ImageNet1k completo.
