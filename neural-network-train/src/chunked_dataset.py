"""
DataLoader personalizado para cargar datasets grandes divididos en chunks.

Este módulo proporciona una implementación de dataset que puede cargar
archivos tar.gz divididos en chunks bajo demanda, evitando cargar todo
el dataset en memoria de una vez.

Útil para datasets como ImageNet1k (>150GB) que son demasiado grandes
para descargar de una vez o mantener en memoria.
"""

import os
import tarfile
from torch.utils.data import Dataset
from PIL import Image
import io
from typing import List, Tuple, Optional, Callable


class ChunkedImageNetDataset(Dataset):
    """
    Dataset que carga chunks de ImageNet bajo demanda.
    
    Estructura esperada:
        chunks_dir/
        ├── imagenet_chunk_00.tar.gz
        ├── imagenet_chunk_01.tar.gz
        └── ...
    
    Cada chunk debe contener imágenes con la estructura:
        class_id/image_name.jpg
    
    Args:
        chunks_dir: Directorio conteniendo los chunks (.tar.gz)
        transform: Transformaciones a aplicar a las imágenes
        images_per_chunk: Número aproximado de imágenes por chunk (para cálculos)
        cache_size: Número de chunks a mantener en memoria (1 = solo el actual)
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize(256),
        ...     transforms.CenterCrop(224),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = ChunkedImageNetDataset('./data/chunks/', transform=transform)
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """
    
    def __init__(
        self,
        chunks_dir: str,
        transform: Optional[Callable] = None,
        images_per_chunk: int = 10000,
        cache_size: int = 1
    ):
        self.chunks_dir = chunks_dir
        self.transform = transform
        self.images_per_chunk = images_per_chunk
        self.cache_size = cache_size
        
        # Encontrar todos los chunks
        self.chunks = sorted([
            f for f in os.listdir(chunks_dir)
            if f.endswith('.tar.gz') or f.endswith('.tar')
        ])
        
        if not self.chunks:
            raise ValueError(f"No se encontraron chunks en {chunks_dir}")
        
        print(f"Encontrados {len(self.chunks)} chunks en {chunks_dir}")
        
        # Cache de chunks cargados
        self._cache = {}
        self._cache_order = []
        
        # Cargar índice de imágenes del primer chunk para inferir estructura
        self._build_index()
    
    def _build_index(self):
        """Construye un índice de todas las imágenes en todos los chunks"""
        print("Construyendo índice de imágenes...")
        self.image_index = []
        
        for chunk_idx, chunk_name in enumerate(self.chunks):
            chunk_path = os.path.join(self.chunks_dir, chunk_name)
            
            try:
                with tarfile.open(chunk_path, 'r:*') as tar:
                    for member in tar.getmembers():
                        if self._is_image_file(member.name):
                            # Extraer class_id del path
                            class_id = self._extract_class_id(member.name)
                            if class_id is not None:
                                self.image_index.append({
                                    'chunk_idx': chunk_idx,
                                    'member_name': member.name,
                                    'class_id': class_id
                                })
            except Exception as e:
                print(f"Advertencia: Error al indexar {chunk_name}: {e}")
                continue
        
        print(f"✓ Índice construido: {len(self.image_index)} imágenes")
        
        # Crear mapeo de class_id a índice numérico
        unique_classes = sorted(set(img['class_id'] for img in self.image_index))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        print(f"✓ Encontradas {len(unique_classes)} clases")
    
    def _is_image_file(self, filename: str) -> bool:
        """Verifica si un archivo es una imagen"""
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp'))
    
    def _extract_class_id(self, path: str) -> Optional[str]:
        """
        Extrae el class_id del path del archivo.
        Asume estructura: class_id/image_name.jpg
        """
        parts = path.split('/')
        if len(parts) >= 2:
            # El class_id suele estar en la penúltima posición
            return parts[-2]
        return None
    
    def _load_chunk(self, chunk_idx: int) -> dict:
        """
        Carga un chunk específico en memoria.
        Implementa un cache LRU simple.
        """
        # Si ya está en cache, retornar
        if chunk_idx in self._cache:
            return self._cache[chunk_idx]
        
        # Si el cache está lleno, eliminar el más antiguo
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        # Cargar chunk
        chunk_path = os.path.join(self.chunks_dir, self.chunks[chunk_idx])
        print(f"Cargando chunk {chunk_idx+1}/{len(self.chunks)}: {self.chunks[chunk_idx]}")
        
        chunk_data = {}
        with tarfile.open(chunk_path, 'r:*') as tar:
            for member in tar.getmembers():
                if self._is_image_file(member.name):
                    try:
                        image_data = tar.extractfile(member).read()
                        chunk_data[member.name] = image_data
                    except Exception as e:
                        print(f"Advertencia: Error al cargar {member.name}: {e}")
        
        # Agregar a cache
        self._cache[chunk_idx] = chunk_data
        self._cache_order.append(chunk_idx)
        
        return chunk_data
    
    def __len__(self) -> int:
        """Retorna el número total de imágenes"""
        return len(self.image_index)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Carga y retorna una imagen y su etiqueta.
        
        Args:
            idx: Índice de la imagen
        
        Returns:
            Tupla (imagen, etiqueta)
        """
        if idx < 0 or idx >= len(self.image_index):
            raise IndexError(f"Índice {idx} fuera de rango [0, {len(self.image_index)})")
        
        # Obtener info de la imagen del índice
        img_info = self.image_index[idx]
        chunk_idx = img_info['chunk_idx']
        member_name = img_info['member_name']
        class_id = img_info['class_id']
        
        # Cargar chunk si es necesario
        chunk_data = self._load_chunk(chunk_idx)
        
        # Obtener datos de la imagen
        image_data = chunk_data[member_name]
        
        # Cargar imagen
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Convertir class_id a índice numérico
        label = self.class_to_idx[class_id]
        
        return image, label


class StreamingChunkedDataset(Dataset):
    """
    Versión más eficiente que descarga chunks bajo demanda desde una URL.
    
    Útil cuando los chunks están alojados en S3, HTTP server, etc.
    y no quieres descargar todos de una vez.
    
    Args:
        chunk_urls: Lista de URLs a los chunks
        transform: Transformaciones a aplicar
        cache_dir: Directorio temporal para cachear chunks descargados
    
    Example:
        >>> urls = [
        ...     "https://mi-servidor.com/imagenet_chunk_00.tar.gz",
        ...     "https://mi-servidor.com/imagenet_chunk_01.tar.gz",
        ... ]
        >>> dataset = StreamingChunkedDataset(urls, cache_dir='./tmp/cache/')
    """
    
    def __init__(
        self,
        chunk_urls: List[str],
        transform: Optional[Callable] = None,
        cache_dir: str = './tmp/chunks_cache',
        images_per_chunk: int = 10000
    ):
        self.chunk_urls = chunk_urls
        self.transform = transform
        self.cache_dir = cache_dir
        self.images_per_chunk = images_per_chunk
        
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Dataset streaming con {len(chunk_urls)} chunks")
    
    def _download_chunk(self, chunk_idx: int) -> str:
        """Descarga un chunk si no existe localmente"""
        import urllib.request
        
        url = self.chunk_urls[chunk_idx]
        filename = os.path.basename(url)
        local_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(local_path):
            print(f"Descargando chunk {chunk_idx+1}/{len(self.chunk_urls)}...")
            urllib.request.urlretrieve(url, local_path)
            print(f"✓ Descargado: {filename}")
        
        return local_path
    
    def __len__(self) -> int:
        """Estimación del tamaño total"""
        return len(self.chunk_urls) * self.images_per_chunk
    
    def __getitem__(self, idx: int):
        """Similar a ChunkedImageNetDataset pero descarga chunks bajo demanda"""
        chunk_idx = idx // self.images_per_chunk
        local_idx = idx % self.images_per_chunk
        
        # Descargar chunk si es necesario
        chunk_path = self._download_chunk(chunk_idx)
        
        # Leer del chunk (implementación similar a ChunkedImageNetDataset)
        # ... (simplificado para el ejemplo)
        
        raise NotImplementedError("Implementar lógica de carga de imagen")


if __name__ == "__main__":
    # Ejemplo de uso
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Crear dataset
    dataset = ChunkedImageNetDataset(
        chunks_dir='./data/imagenet_chunks/',
        transform=transform,
        images_per_chunk=10000,
        cache_size=2  # Mantener 2 chunks en memoria
    )
    
    # Crear dataloader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # Shuffle=True puede causar muchos cambios de chunk
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset: {len(dataset)} imágenes")
    print(f"Número de clases: {len(dataset.class_to_idx)}")
    
    # Probar carga de primer batch
    for images, labels in loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels[:5]}")
        break
