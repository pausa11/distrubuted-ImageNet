# Guía de Alojamiento de Datasets para Entrenamiento Distribuido

Esta guía explica cómo alojar y acceder a datasets como ImageNet1k para entrenamiento distribuido con múltiples peers.

## Tabla de Contenidos

1. [Opciones de Alojamiento](#opciones-de-alojamiento)
2. [Configuración para Experimentos Pequeños](#configuración-para-experimentos-pequeños)
3. [Configuración para Experimentos Grandes](#configuración-para-experimentos-grandes)
4. [Manejo de Archivos Grandes](#manejo-de-archivos-grandes)
5. [Uso de Academic Torrents](#uso-de-academic-torrents)
6. [Scripts de Descarga](#scripts-de-descarga)

---

## Opciones de Alojamiento

### 1. Experimentos Pequeños (3-16 peers, <1GB)

Para experimentos pequeños o pruebas iniciales, puedes usar servicios gratuitos de alojamiento de archivos:

#### Servicios Recomendados:
- **Google Drive** - Hasta 15GB gratis
- **Dropbox** - Hasta 2GB gratis
- **GitHub Releases** - Hasta 2GB por archivo (recomendado para datasets pequeños)
- **OneDrive** - Hasta 5GB gratis

#### ⚠️ Advertencias:
- Estos servicios **no están diseñados para alto tráfico**
- Pueden **banear tu cuenta** si generas demasiado tráfico
- Ideales solo para **pruebas y desarrollo**

#### Ejemplo de Descarga desde Google Drive:

```bash
# Usando gdown (instalar con: pip install gdown)
gdown https://drive.google.com/uc?id=FILE_ID -O dataset.zip
unzip dataset.zip -d ./data/
```

#### Ejemplo de Descarga desde GitHub Releases:

```bash
# Usando wget o curl
wget https://github.com/usuario/repo/releases/download/v1.0/dataset.tar.gz
tar -xzf dataset.tar.gz -C ./data/
```

---

### 2. Experimentos Grandes (>16 peers, >1GB)

Para experimentos a escala, necesitas una solución robusta:

#### Opción A: Almacenamiento en la Nube (S3-like)

**Amazon S3:**
```bash
# Instalar AWS CLI
pip install awscli

# Configurar credenciales
aws configure

# Descargar dataset
aws s3 sync s3://tu-bucket/imagenet1k ./data/imagenet1k/
```

**Costos aproximados (AWS S3):**
- Almacenamiento: ~$0.023/GB/mes
- Transferencia: ~$0.09/GB (primeros 10TB/mes)
- ImageNet1k (~150GB): ~$3.45/mes + transferencia

**Google Cloud Storage:**
```bash
# Instalar gsutil
pip install gsutil

# Descargar dataset
gsutil -m cp -r gs://tu-bucket/imagenet1k ./data/
```

**Azure Blob Storage:**
```bash
# Instalar Azure CLI
pip install azure-cli

# Descargar dataset
az storage blob download-batch \
  --account-name tu-cuenta \
  --source tu-contenedor \
  --destination ./data/imagenet1k/
```

#### Opción B: Auto-Alojamiento

Si tienes un servidor propio con buena conectividad:

**Configurar servidor HTTP simple:**
```bash
# En el servidor con el dataset
cd /path/to/datasets
python3 -m http.server 8080
```

**Descargar en los peers:**
```bash
wget http://tu-servidor:8080/imagenet1k.tar.gz
tar -xzf imagenet1k.tar.gz -C ./data/
```

**Configurar servidor con nginx (más robusto):**
```nginx
# /etc/nginx/sites-available/datasets
server {
    listen 80;
    server_name datasets.tu-dominio.com;
    
    location /datasets/ {
        alias /path/to/datasets/;
        autoindex on;
        
        # Limitar velocidad de descarga por conexión
        limit_rate 10m;
    }
}
```

---

## Manejo de Archivos Grandes

Archivos mayores a 5GB pueden tardar mucho en descargarse. La mejor práctica es:

### 1. Dividir en Chunks

```bash
# Dividir dataset en chunks de 2GB
split -b 2G imagenet1k.tar.gz imagenet1k.tar.gz.part_

# Para reconstruir:
cat imagenet1k.tar.gz.part_* > imagenet1k.tar.gz
```

### 2. Implementar DataLoader con Carga Incremental

Crea un dataloader personalizado que cargue chunks bajo demanda:

```python
# ejemplo: dataset_loader.py
import os
import tarfile
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io

class ChunkedImageNetDataset(Dataset):
    """Dataset que carga chunks de ImageNet bajo demanda"""
    
    def __init__(self, chunks_dir, transform=None):
        self.chunks_dir = chunks_dir
        self.transform = transform
        self.chunks = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.tar.gz')])
        self.current_chunk_idx = None
        self.current_data = []
        self._load_chunk(0)
    
    def _load_chunk(self, chunk_idx):
        """Carga un chunk específico en memoria"""
        if self.current_chunk_idx == chunk_idx:
            return
        
        chunk_path = os.path.join(self.chunks_dir, self.chunks[chunk_idx])
        print(f"Cargando chunk {chunk_idx+1}/{len(self.chunks)}: {chunk_path}")
        
        self.current_data = []
        with tarfile.open(chunk_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith(('.jpg', '.jpeg', '.png')):
                    # Extraer imagen y label del nombre del archivo
                    # Asume estructura: class_id/image_name.jpg
                    parts = member.name.split('/')
                    if len(parts) >= 2:
                        class_id = parts[-2]
                        image_data = tar.extractfile(member).read()
                        self.current_data.append((image_data, class_id))
        
        self.current_chunk_idx = chunk_idx
    
    def __len__(self):
        # Esto es una aproximación, ajustar según necesidad
        return len(self.chunks) * 10000  # asume ~10k imágenes por chunk
    
    def __getitem__(self, idx):
        # Determinar qué chunk necesitamos
        chunk_idx = idx // 10000
        local_idx = idx % 10000
        
        # Cargar chunk si es necesario
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        if local_idx >= len(self.current_data):
            # Si nos pasamos, cargar siguiente chunk
            chunk_idx += 1
            if chunk_idx >= len(self.chunks):
                raise IndexError("Index out of range")
            self._load_chunk(chunk_idx)
            local_idx = 0
        
        image_data, class_id = self.current_data[local_idx]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, int(class_id)

# Uso:
# dataset = ChunkedImageNetDataset('./data/imagenet_chunks/')
# loader = DataLoader(dataset, batch_size=32, shuffle=False)
```

### 3. Descargar Chunks en Paralelo

```bash
#!/bin/bash
# download_chunks_parallel.sh

BASE_URL="http://tu-servidor/datasets/imagenet1k"
NUM_CHUNKS=75  # ImageNet1k dividido en chunks de ~2GB

# Descargar chunks en paralelo (máximo 4 a la vez)
for i in $(seq 0 $NUM_CHUNKS); do
    CHUNK=$(printf "imagenet1k.tar.gz.part_%02d" $i)
    (
        wget -q "$BASE_URL/$CHUNK" -O "./data/chunks/$CHUNK" && \
        echo "✓ Descargado: $CHUNK"
    ) &
    
    # Limitar a 4 descargas simultáneas
    if [ $(( i % 4 )) -eq 3 ]; then
        wait
    fi
done
wait

echo "✓ Todos los chunks descargados"
```

---

## Uso de Academic Torrents

**Academic Torrents** es la solución más apropiada para compartir datasets académicos grandes.

### Ventajas:
- ✅ Descentralizado - no dependes de un servidor central
- ✅ Escalable - múltiples peers pueden compartir simultáneamente
- ✅ Verificación de integridad incluida
- ✅ Gratis y legal para datasets académicos
- ✅ Permanente - el dataset no desaparecerá

### Configuración:

#### 1. Instalar Cliente Torrent

```bash
# Transmission (CLI)
sudo apt-get install transmission-cli transmission-daemon

# O usar aria2
sudo apt-get install aria2
```

#### 2. Descargar ImageNet desde Academic Torrents

```bash
# Usando transmission-cli
transmission-cli "magnet:?xt=urn:btih:HASH_DEL_TORRENT" \
  -w ./data/

# Usando aria2
aria2c --seed-time=0 "magnet:?xt=urn:btih:HASH_DEL_TORRENT" \
  -d ./data/
```

#### 3. Crear tu Propio Torrent (Opcional)

Si procesaste/modificaste el dataset:

```bash
# Instalar mktorrent
sudo apt-get install mktorrent

# Crear torrent
mktorrent -a http://tracker.academictorrents.com:6969/announce \
  -o imagenet1k-custom.torrent \
  ./data/imagenet1k/

# Subir el .torrent a Academic Torrents (academictorrents.com)
```

### Enlaces de Academic Torrents:

- **ImageNet Object Localization Challenge (ILSVRC2012)**: 
  - URL: https://academictorrents.com/details/imagenet-2012
  - Tamaño: ~150GB
  - 1000 clases, ~1.2M imágenes de entrenamiento

- **Tiny-ImageNet-200**: 
  - URL: https://academictorrents.com/details/tiny-imagenet-200
  - Tamaño: ~250MB
  - 200 clases, 100k imágenes

---

## Scripts de Descarga

### Script Completo de Setup

Crear archivo `scripts/setup_dataset.sh`:

```bash
#!/bin/bash

# Script para configurar datasets para entrenamiento distribuido
# Uso: ./scripts/setup_dataset.sh [tiny-imagenet|imagenet1k] [method]

DATASET=$1
METHOD=$2  # local, gdrive, s3, torrent

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

case $DATASET in
  "tiny-imagenet")
    echo "📦 Descargando Tiny-ImageNet-200..."
    
    case $METHOD in
      "local"|"")
        # Descarga directa desde Stanford
        wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip -O "$DATA_DIR/tiny-imagenet-200.zip"
        unzip -q "$DATA_DIR/tiny-imagenet-200.zip" -d "$DATA_DIR/"
        rm "$DATA_DIR/tiny-imagenet-200.zip"
        ;;
      
      "torrent")
        # Academic Torrents
        aria2c --seed-time=0 \
          "magnet:?xt=urn:btih:TINY_IMAGENET_HASH" \
          -d "$DATA_DIR/"
        ;;
    esac
    
    echo "✓ Tiny-ImageNet-200 listo en: $DATA_DIR/tiny-imagenet-200/"
    ;;
  
  "imagenet1k")
    echo "📦 Descargando ImageNet ILSVRC2012..."
    
    case $METHOD in
      "s3")
        # Requiere credenciales AWS configuradas
        aws s3 sync s3://tu-bucket/ILSVRC2012 "$DATA_DIR/ILSVRC2012/"
        ;;
      
      "torrent")
        # Academic Torrents - método recomendado
        echo "Descargando desde Academic Torrents..."
        aria2c --seed-time=24 \
          --max-upload-limit=10M \
          "magnet:?xt=urn:btih:IMAGENET_HASH" \
          -d "$DATA_DIR/"
        ;;
      
      "gdrive")
        # Requiere gdown instalado: pip install gdown
        echo "Descargando desde Google Drive..."
        gdown YOUR_GDRIVE_FILE_ID -O "$DATA_DIR/ILSVRC2012.tar"
        tar -xf "$DATA_DIR/ILSVRC2012.tar" -C "$DATA_DIR/"
        rm "$DATA_DIR/ILSVRC2012.tar"
        ;;
      
      *)
        echo "❌ Método no válido para ImageNet1k. Usa: s3, torrent, o gdrive"
        exit 1
        ;;
    esac
    
    echo "✓ ImageNet ILSVRC2012 listo en: $DATA_DIR/ILSVRC2012/"
    ;;
  
  *)
    echo "❌ Dataset no válido. Usa: tiny-imagenet o imagenet1k"
    exit 1
    ;;
esac

echo ""
echo "✅ Dataset configurado correctamente!"
echo "   Estructura en: $DATA_DIR/"
```

### Hacer el Script Ejecutable

```bash
chmod +x scripts/setup_dataset.sh

# Ejemplos de uso:
./scripts/setup_dataset.sh tiny-imagenet local
./scripts/setup_dataset.sh imagenet1k torrent
./scripts/setup_dataset.sh imagenet1k s3
```

---

## Recomendaciones por Caso de Uso

| Caso de Uso | Solución Recomendada | Costo | Complejidad |
|-------------|---------------------|-------|-------------|
| Desarrollo/Pruebas (<5 peers) | Google Drive / GitHub Releases | Gratis | Baja |
| Experimento pequeño (5-15 peers) | Auto-hosting / Dropbox | Gratis - Bajo | Media |
| Producción (>15 peers) | Academic Torrents | Gratis | Media |
| Producción empresarial | AWS S3 / GCP / Azure | $$$ | Alta |
| Dataset > 5GB | Academic Torrents + chunking | Gratis | Alta |
| Dataset personalizado | S3 + chunks o auto-hosting | $ - $$$ | Alta |

---

## Verificación de Integridad

Siempre verifica la integridad del dataset descargado:

```bash
# Generar checksum
sha256sum imagenet1k.tar.gz > imagenet1k.sha256

# Verificar checksum
sha256sum -c imagenet1k.sha256
```

---

## Solución de Problemas

### Error: "No space left on device"
- Verifica espacio disponible: `df -h`
- Considera descargar chunks individuales
- Usa almacenamiento externo o montajes de red

### Error: "Connection timeout" durante descarga
- Usa `wget -c` o `aria2c` para reanudar descargas
- Divide en chunks más pequeños
- Considera usar torrent para mayor resiliencia

### Dataset corrupto después de descarga
- Verifica checksums SHA256
- Vuelve a descargar el chunk problemático
- Usa torrent que verifica integridad automáticamente

---

## Contribuciones

Si encuentras mejores métodos o servicios para alojar datasets, por favor abre un issue o pull request.

## Licencias

Asegúrate de cumplir con las licencias de los datasets:
- **ImageNet**: Requiere registro y aceptación de términos
- **Tiny-ImageNet**: Uso educativo permitido
- Consulta siempre los términos antes de redistribuir
