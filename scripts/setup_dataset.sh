#!/bin/bash

# Script para configurar datasets para entrenamiento distribuido
# Uso: ./scripts/setup_dataset.sh [tiny-imagenet|imagenet1k] [method]

set -e  # Salir si hay algún error

DATASET=$1
METHOD=$2  # local, gdrive, s3, torrent

# Directorio de datos
DATA_DIR="./data"
mkdir -p "$DATA_DIR"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_info() {
    echo -e "${YELLOW}📦 $1${NC}"
}

# Verificar que se pasó el dataset
if [ -z "$DATASET" ]; then
    echo_error "Debes especificar un dataset: tiny-imagenet o imagenet1k"
    echo "Uso: ./scripts/setup_dataset.sh [tiny-imagenet|imagenet1k] [method]"
    exit 1
fi

# Método por defecto
if [ -z "$METHOD" ]; then
    METHOD="local"
fi

case $DATASET in
  "tiny-imagenet")
    echo_info "Configurando Tiny-ImageNet-200..."
    
    case $METHOD in
      "local")
        echo_info "Descargando desde Stanford CS231n..."
        if command -v wget &> /dev/null; then
            wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip -O "$DATA_DIR/tiny-imagenet-200.zip"
        elif command -v curl &> /dev/null; then
            curl -L http://cs231n.stanford.edu/tiny-imagenet-200.zip -o "$DATA_DIR/tiny-imagenet-200.zip"
        else
            echo_error "wget o curl no encontrado. Instala uno de ellos."
            exit 1
        fi
        
        echo_info "Descomprimiendo..."
        unzip -q "$DATA_DIR/tiny-imagenet-200.zip" -d "$DATA_DIR/"
        rm "$DATA_DIR/tiny-imagenet-200.zip"
        echo_success "Tiny-ImageNet-200 descargado y descomprimido"
        ;;
      
      "torrent")
        if ! command -v aria2c &> /dev/null; then
            echo_error "aria2c no encontrado. Instálalo con: sudo apt-get install aria2"
            exit 1
        fi
        
        echo_info "Descargando desde Academic Torrents..."
        echo_warning "Método torrent para Tiny-ImageNet no configurado por defecto"
        echo "Tiny-ImageNet es pequeño (~250MB), se recomienda usar 'local' en su lugar"
        echo ""
        echo "Si deseas usar torrents, busca el dataset en:"
        echo "  https://academictorrents.com/browse.php?search=tiny+imagenet"
        echo ""
        echo "Luego usa: aria2c --seed-time=0 'magnet:?xt=urn:btih:HASH' -d $DATA_DIR/"
        exit 1
        ;;
      
      *)
        echo_error "Método no válido para Tiny-ImageNet. Usa: local o torrent"
        exit 1
        ;;
    esac
    
    # Verificar estructura del dataset
    if [ -d "$DATA_DIR/tiny-imagenet-200/train" ]; then
        TRAIN_CLASSES=$(ls -d "$DATA_DIR/tiny-imagenet-200/train"/*/ 2>/dev/null | wc -l)
        echo_success "Dataset verificado: $TRAIN_CLASSES clases de entrenamiento"
        echo_info "Ruta: $DATA_DIR/tiny-imagenet-200/"
    else
        echo_warning "Estructura de directorio no encontrada. Verifica manualmente."
    fi
    ;;
  
  "imagenet1k")
    echo_info "Configurando ImageNet ILSVRC2012 (1000 clases)..."
    
    case $METHOD in
      "s3")
        if ! command -v aws &> /dev/null; then
            echo_error "AWS CLI no encontrado. Instálalo con: pip install awscli"
            exit 1
        fi
        
        echo_warning "Este comando requiere credenciales AWS configuradas"
        echo_info "Configura con: aws configure"
        read -p "¿Continuar? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        read -p "Bucket S3 (ej: s3://mi-bucket/ILSVRC2012): " S3_PATH
        echo_info "Sincronizando desde $S3_PATH..."
        aws s3 sync "$S3_PATH" "$DATA_DIR/ILSVRC2012/"
        ;;
      
      "torrent")
        if ! command -v aria2c &> /dev/null; then
            echo_error "aria2c no encontrado. Instálalo con: sudo apt-get install aria2"
            exit 1
        fi
        
        echo_info "Descargando desde Academic Torrents (método recomendado)..."
        echo_warning "ImageNet es ~150GB. La descarga puede tardar varias horas."
        echo_info "Visita https://academictorrents.com/details/564a77c1e1119da199ff568478ccb6b6c8d36c0c"
        
        read -p "¿Continuar con la descarga? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        
        read -p "¿Continuar con la descarga? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        # Magnet link conocido para ILSVRC2012 en Academic Torrents
        # Fuente: https://academictorrents.com/details/564a77c1e1119da199ff568478ccb6b6c8d36c0c
        # Nota: Verifica que este hash sea el correcto visitando Academic Torrents
        MAGNET="magnet:?xt=urn:btih:564a77c1e1119da199ff568478ccb6b6c8d36c0c&dn=ILSVRC2012_img_train.tar"
        
        echo_warning "IMPORTANTE: Verifica el hash en https://academictorrents.com/"
        echo_info "Hash actual: 564a77c1e1119da199ff568478ccb6b6c8d36c0c"
        
        aria2c --seed-time=24 \
               --max-upload-limit=10M \
               --file-allocation=none \
               "$MAGNET" \
               -d "$DATA_DIR/"
        ;;
      
      "gdrive")
        if ! command -v gdown &> /dev/null; then
            echo_error "gdown no encontrado. Instálalo con: pip install gdown"
            exit 1
        fi
        
        echo_warning "Google Drive puede limitar descargas de archivos grandes"
        echo_info "Necesitas el File ID de tu archivo en Google Drive"
        
        read -p "Google Drive File ID: " GDRIVE_ID
        if [ -z "$GDRIVE_ID" ]; then
            echo_error "File ID no puede estar vacío"
            exit 1
        fi
        
        echo_info "Descargando desde Google Drive..."
        gdown "$GDRIVE_ID" -O "$DATA_DIR/ILSVRC2012.tar"
        
        echo_info "Extrayendo archivo..."
        tar -xf "$DATA_DIR/ILSVRC2012.tar" -C "$DATA_DIR/"
        rm "$DATA_DIR/ILSVRC2012.tar"
        ;;
      
      "manual")
        echo_warning "Descarga manual seleccionada"
        echo_info "Por favor descarga ImageNet ILSVRC2012 desde:"
        echo "  https://www.image-net.org/challenges/LSVRC/2012/"
        echo ""
        echo "Y coloca los archivos en: $DATA_DIR/ILSVRC2012/"
        echo ""
        echo "Estructura esperada:"
        echo "  $DATA_DIR/ILSVRC2012/"
        echo "    ├── train/"
        echo "    │   ├── n01440764/"
        echo "    │   ├── n01443537/"
        echo "    │   └── ..."
        echo "    └── val/"
        echo "        ├── ILSVRC2012_val_00000001.JPEG"
        echo "        ├── ILSVRC2012_val_00000002.JPEG"
        echo "        └── ..."
        ;;
      
      *)
        echo_error "Método no válido para ImageNet1k"
        echo "Métodos disponibles: s3, torrent, gdrive, manual"
        exit 1
        ;;
    esac
    
    if [ "$METHOD" != "manual" ] && [ -d "$DATA_DIR/ILSVRC2012" ]; then
        echo_success "ImageNet ILSVRC2012 configurado"
        echo_info "Ruta: $DATA_DIR/ILSVRC2012/"
    fi
    ;;
  
  *)
    echo_error "Dataset no válido: $DATASET"
    echo "Datasets disponibles:"
    echo "  - tiny-imagenet: Tiny-ImageNet-200 (250MB, 200 clases)"
    echo "  - imagenet1k: ImageNet ILSVRC2012 (150GB, 1000 clases)"
    exit 1
    ;;
esac

echo ""
echo_success "Setup completado!"
echo_info "Verifica el contenido en: $DATA_DIR/"

# Mostrar espacio en disco
DISK_USAGE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
echo_info "Espacio utilizado: $DISK_USAGE"

# Verificar espacio disponible
AVAILABLE_SPACE=$(df -h "$DATA_DIR" | awk 'NR==2 {print $4}')
echo_info "Espacio disponible: $AVAILABLE_SPACE"
