# Entrenamiento Distribuido con ImageNet

Sistema de entrenamiento distribuido descentralizado usando Hivemind para ImageNet y otros datasets.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración de Datasets](#configuración-de-datasets)
- [Uso](#uso)
- [Ejemplos](#ejemplos)
- [Arquitectura](#arquitectura)
- [Contribuir](#contribuir)

## ✨ Características

- **Entrenamiento descentralizado** usando [Hivemind](https://github.com/learning-at-home/hivemind)
- **Soporte para múltiples datasets**:
  - CIFAR-10 (pequeño, para pruebas)
  - Tiny-ImageNet-200 (mediano, 200 clases)
  - ImageNet ILSVRC2012 (grande, 1000 clases)
- **Checkpointing automático** del mejor modelo
- **Monitoreo en tiempo real** con frontend web
- **Carga eficiente de datos grandes** mediante chunks

## 🔧 Requisitos

### Sistema
- Python 3.8+
- GPU NVIDIA (recomendado, CUDA compatible)
- 8GB+ RAM (16GB+ recomendado para ImageNet)
- Espacio en disco:
  - CIFAR-10: ~200MB
  - Tiny-ImageNet: ~300MB
  - ImageNet1k: ~150GB

### Software
- PyTorch 2.2+
- Hivemind 1.1.11
- Node.js 18+ (para frontend, opcional)

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/pausa11/distrubuted-ImageNet.git
cd distrubuted-ImageNet
```

### 2. Instalar dependencias de Python

```bash
cd neural-network-train
pip install -r requirements.txt
```

### 3. Instalar frontend (opcional)

```bash
cd front-end-cliente
npm install
```

## 📊 Configuración de Datasets

### Opción 1: Script Automático (Recomendado)

```bash
# Para Tiny-ImageNet (rápido, ideal para pruebas)
./scripts/setup_dataset.sh tiny-imagenet local

# Para ImageNet1k (grande, requiere registro en image-net.org)
./scripts/setup_dataset.sh imagenet1k torrent
```

### Opción 2: Manual

Ver la [**Guía Completa de Hosting de Datasets**](DATASET_HOSTING.md) para:
- Opciones de alojamiento (S3, Google Drive, torrents)
- Configuración para diferentes escalas
- Manejo de archivos grandes
- Academic Torrents (recomendado para producción)

### Estructura Esperada

```
distrubuted-ImageNet/
├── data/
│   ├── cifar-10-batches-py/          # CIFAR-10 (descarga automática)
│   ├── tiny-imagenet-200/            # Tiny-ImageNet
│   │   ├── train/
│   │   │   ├── n01443537/
│   │   │   └── ...
│   │   └── val/
│   └── ILSVRC2012/                   # ImageNet1k (opcional)
│       ├── train/
│       └── val/
└── checkpoints/                      # Guardado automático de modelos
```

## 🚀 Uso

### Entrenamiento con CIFAR-10 (Prueba Rápida)

```bash
cd neural-network-train/src

# Peer 1 (inicial)
python trainGossipCifar.py \
  --data_root ../data \
  --batch 64 \
  --lr 0.01 \
  --epochs 10 \
  --host_maddr /ip4/0.0.0.0/tcp/4011 \
  --run_id cifar10-test

# Peer 2 (conectar al primero)
python trainGossipCifar.py \
  --data_root ../data \
  --batch 64 \
  --lr 0.01 \
  --epochs 10 \
  --host_maddr /ip4/0.0.0.0/tcp/4012 \
  --initial_peer /ip4/127.0.0.1/tcp/4011/p2p/<PEER_ID_DEL_PEER1> \
  --run_id cifar10-test
```

### Entrenamiento con Tiny-ImageNet

```bash
cd neural-network-train/src

# Peer 1
python trainGossipImagenet.py \
  --data_root ../data \
  --batch 32 \
  --lr 0.1 \
  --epochs 20 \
  --target_global_bsz 256 \
  --host_maddr /ip4/0.0.0.0/tcp/5011 \
  --run_id imagenet-tiny-v1 \
  --val_interval 500

# Peer 2
python trainGossipImagenet.py \
  --data_root ../data \
  --batch 32 \
  --lr 0.1 \
  --epochs 20 \
  --target_global_bsz 256 \
  --host_maddr /ip4/0.0.0.0/tcp/5012 \
  --initial_peer /ip4/127.0.0.1/tcp/5011/p2p/<PEER_ID> \
  --run_id imagenet-tiny-v1
```

### Reanudar desde Checkpoint

```bash
python trainGossipImagenet.py \
  --data_root ../data \
  --resume_from checkpoints/imagenet_tiny_resnet50_5011_best.pt \
  --host_maddr /ip4/0.0.0.0/tcp/5011 \
  --run_id imagenet-tiny-v1
```

### Con Monitoreo (Frontend)

```bash
# Terminal 1: Backend de control (opcional)
cd neural-network-train/src
python controller_server.py

# Terminal 2: Frontend
cd front-end-cliente
npm run dev

# Terminal 3+: Peers de entrenamiento
python trainGossipImagenet.py \
  --controller http://localhost:8000 \
  ... (otros argumentos)
```

## 💡 Ejemplos

### Experimento Multi-Nodo

```bash
# Nodo 1 (IP: 192.168.1.10)
python trainGossipImagenet.py \
  --host_maddr /ip4/0.0.0.0/tcp/5011 \
  --run_id imagenet-distributed

# Nodo 2 (IP: 192.168.1.20)
python trainGossipImagenet.py \
  --host_maddr /ip4/0.0.0.0/tcp/5011 \
  --initial_peer /ip4/192.168.1.10/tcp/5011/p2p/<PEER_ID_NODO1> \
  --run_id imagenet-distributed

# Nodo 3 (IP: 192.168.1.30)
python trainGossipImagenet.py \
  --host_maddr /ip4/0.0.0.0/tcp/5011 \
  --initial_peer /ip4/192.168.1.10/tcp/5011/p2p/<PEER_ID_NODO1> \
  --run_id imagenet-distributed
```

### Usando Pesos Pre-entrenados

```bash
python trainGossipImagenet.py \
  --pretrained \
  --lr 0.01 \
  ... (otros argumentos)
```

### Dataset con Chunks (Para archivos >5GB)

Ver [`chunked_dataset.py`](neural-network-train/src/chunked_dataset.py) para implementación completa.

```python
from chunked_dataset import ChunkedImageNetDataset
from torch.utils.data import DataLoader

dataset = ChunkedImageNetDataset(
    chunks_dir='./data/imagenet_chunks/',
    transform=transform,
    cache_size=2  # Mantener 2 chunks en memoria
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                   │
│           Visualización de métricas en tiempo real      │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
                         │
┌────────────────────────▼────────────────────────────────┐
│              Controller (FastAPI, opcional)             │
│           Recopila y sirve métricas de peers            │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐              ┌────────▼───────┐
│   Peer 1       │◄────P2P─────►│   Peer 2       │
│  (Hivemind)    │              │  (Hivemind)    │
│                │              │                │
│ - ResNet50     │              │ - ResNet50     │
│ - Optimizer    │              │ - Optimizer    │
│ - DHT          │              │ - DHT          │
└────────────────┘              └────────────────┘
        │                                 │
        │         ┌───────────────┐       │
        └────────►│      DHT      │◄──────┘
                  │ (Descentralizado)
                  │ - Sincronización
                  │ - All-reduce
                  └───────────────┘
```

## 📚 Documentación Adicional

- [**Guía de Hosting de Datasets**](DATASET_HOSTING.md) - Detallado sobre alojamiento y descarga
- [**Chunked Dataset API**](neural-network-train/src/chunked_dataset.py) - Carga eficiente de datos grandes

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Hivemind](https://github.com/learning-at-home/hivemind) - Framework de entrenamiento distribuido
- [PyTorch](https://pytorch.org/) - Framework de deep learning
- [ImageNet](https://www.image-net.org/) - Dataset
- [Academic Torrents](https://academictorrents.com/) - Distribución de datasets

## 📞 Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa la [Guía de Hosting de Datasets](DATASET_HOSTING.md)
2. Busca en [Issues existentes](https://github.com/pausa11/distrubuted-ImageNet/issues)
3. Abre un [nuevo Issue](https://github.com/pausa11/distrubuted-ImageNet/issues/new)

---

**Nota**: Este proyecto es para fines educativos y de investigación. Asegúrate de cumplir con las licencias y términos de uso de los datasets que utilices.
