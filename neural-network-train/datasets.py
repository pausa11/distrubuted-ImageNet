import os
import json
import hashlib
import io
import random
import concurrent.futures
import threading
import time
from typing import Optional, List, Tuple, Set

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from PIL import Image
from tqdm.auto import tqdm

# Try to import google-cloud-storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

def retry_gcs_op(max_retries=3, delay=1.0):
    """Decorator to retry GCS operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Don't retry on certain errors if needed, but generally network/timeout are worth retrying
                    print(f"⚠️ GCS Op failed (attempt {i+1}/{max_retries}): {e}")
                    time.sleep(delay * (2 ** i)) # Exponential backoff
            raise last_exception
        return wrapper
    return decorator

def discover_gcs_files(bucket_name: str, prefix: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Discovers files in a GCS bucket with caching.
    Returns (samples, classes) where samples is list of (blob_name, class_name).
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required. Run `pip install google-cloud-storage`.")

    prefix = prefix.rstrip('/')
    cache_key = hashlib.md5(f"{bucket_name}_{prefix}".encode()).hexdigest()
    cache_file = f"gcs_cache_{cache_key}.json"
    
    samples = []
    classes = []
    
    if os.path.exists(cache_file):
        print(f"Found GCS cache: {cache_file}. Loading...")
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                classes = data['classes']
                samples = data['samples'] # list of [blob_name, class_name]
                print(f"Loaded {len(samples)} images from cache.")
                return samples, classes
        except Exception as e:
            print(f"Failed to load cache: {e}. Will re-list blobs.")

    # We need a temporary client just for listing blobs during init (main process)
    tmp_client = storage.Client.create_anonymous_client()
    tmp_bucket = tmp_client.bucket(bucket_name)
    
    print(f"Listing blobs in gs://{bucket_name}/{prefix} ... (this may take a while)")
    blobs = list(tmp_bucket.list_blobs(prefix=prefix))
    
    samples = []
    classes_set = set()
    
    # Filter for images and parse classes
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for blob in tqdm(blobs, desc="Indexing GCS files"):
        if blob.name.endswith('/'): continue # Skip directories
        ext = os.path.splitext(blob.name)[1].lower()
        if ext in valid_extensions:
            # Structure: prefix/class_name/filename
            # rel_path: class_name/filename
            rel_path = blob.name[len(prefix)+1:]
            parts = rel_path.split('/')
            if len(parts) >= 2:
                class_name = parts[0]
                classes_set.add(class_name)
                samples.append((blob.name, class_name))
    
    classes = sorted(list(classes_set))
    
    # Save to cache
    print(f"Saving GCS listing to cache: {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump({
            'classes': classes,
            'samples': samples
        }, f)
        
    return samples, classes


class GCSImageFolder(Dataset):
    def __init__(self, bucket_name: str, prefix: str, transform=None, classes: list = None):
        """
        A Dataset that streams images from a GCS bucket.
        Expects structure: prefix/class_name/image.jpg
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        
        self.samples, discovered_classes = discover_gcs_files(bucket_name, prefix)
        
        # Override classes if provided (CRITICAL for validation set consistency)
        if classes is not None:
            self.classes = classes
        else:
            self.classes = discovered_classes
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.samples)} images belonging to {len(self.classes)} classes.")
        if len(self.samples) == 0:
            msg = (f"⚠️  WARNING: No images found in gs://{bucket_name}/{prefix}\n"
                   f"    Check if the prefix is correct. The script expects: prefix/class_name/image.jpg")
            print(msg)
            raise RuntimeError(msg)
        
        # Do NOT initialize client here to avoid fork-safety issues with multiprocessing
        self.client = None
        self.bucket = None

    def __len__(self):
        return len(self.samples)

    @retry_gcs_op(max_retries=3)
    def _download_blob(self, blob_name):
        # Lazy initialization of GCS client (per worker process)
        if self.client is None:
            self.client = storage.Client.create_anonymous_client()
            self.bucket = self.client.bucket(self.bucket_name)
        
        blob = self.bucket.blob(blob_name)
        return blob.download_as_bytes()

    def __getitem__(self, idx):
        blob_name, class_name = self.samples[idx]
        
        # Handle case where sample class is not in our class list (if classes was overridden)
        if class_name not in self.class_to_idx:
             # This might happen if we use train classes for val, and val has a class not in train?
             # ImageNet usually has same classes. 
             # Or if we filtered classes. 
             # For now, let's just error or skip? Error is safer.
             # Actually, standard ImageFolder behavior is to rely on folder structure.
             # If we passed explicit classes, we expect samples to match.
             # Let's assume it's fine or fail hard.
             pass

        label = self.class_to_idx[class_name]
        
        image_bytes = self._download_blob(blob_name)
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class ThreadedGCSDataset(IterableDataset):
    def __init__(self, bucket_name: str, prefix: str, transform=None, 
                 buffer_size: int = 256, num_threads: int = 16, shuffle: bool = False, classes: list = None):
        """
        A Threaded IterableDataset that pre-fetches images from GCS using a thread pool.
        This bypasses the need for multiprocessing workers (which crash on Mac/MPS)
        while still allowing parallel downloads.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.buffer_size = buffer_size
        self.num_threads = num_threads
        self.shuffle = shuffle
        
        self.samples, discovered_classes = discover_gcs_files(bucket_name, prefix)
        
        if classes is not None:
            self.classes = classes
        else:
            self.classes = discovered_classes

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"[Threaded] Initialized with {len(self.samples)} images. Threads={num_threads}")

    def __len__(self):
        return len(self.samples)

    def _download_func(self, sample):
        blob_name, class_name = sample
        label = self.class_to_idx[class_name]
        
        if not hasattr(self.local, 'client'):
            self.local.client = storage.Client.create_anonymous_client()
            self.local.bucket = self.local.client.bucket(self.bucket_name)
            
        try:
            # Simple retry loop for threaded context
            for i in range(3):
                try:
                    blob = self.local.bucket.blob(blob_name)
                    image_bytes = blob.download_as_bytes()
                    break
                except Exception as e:
                    if i == 2: raise e
                    time.sleep(0.5 * (2**i))
            
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error downloading {blob_name}: {e}")
            return None

    def __iter__(self):
        # Create thread-local storage
        self.local = threading.local()
        
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
            
        # We will yield from a generator that pushes tasks to executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit initial batch
            futures = []
            
            # Helper to submit more
            def submit_more(idx_iterator, count):
                added = 0
                for _ in range(count):
                    try:
                        i = next(idx_iterator)
                        futures.append(executor.submit(self._download_func, self.samples[i]))
                        added += 1
                    except StopIteration:
                        break
                return added

            idx_iter = iter(indices)
            
            # Fill buffer
            submit_more(idx_iter, self.buffer_size)
            
            while futures:
                # Wait for the oldest future (FIFO)
                f = futures.pop(0)
                result = f.result() # Block until this one is ready
                
                # Refill one
                submit_more(idx_iter, 1)
                
                if result is not None:
                    yield result

def get_dataloaders_imagenet(
    data_root: str,
    batch: int,
    workers: int,
    device: torch.device,
    bucket_name: Optional[str] = None,
    train_prefix: Optional[str] = None,
    val_prefix: Optional[str] = None
):
    """
    Espera un árbol:
      data_root/
        train/0..999/*.JPEG
        val/0..999/*.JPEG
    
    O si bucket_name está definido:
      gs://bucket_name/train_prefix/...
      gs://bucket_name/val_prefix/...
    """
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # pin_memory solo ayuda CPU->CUDA; en MPS/CPU no aporta
    pin = (device.type == "cuda")

    if bucket_name:
        print(f"Using GCS Bucket: {bucket_name}")
        
        if train_prefix:
            final_train_prefix = train_prefix
        else:
            final_train_prefix = os.path.join(data_root, "train") if data_root else "train"
            
        if val_prefix:
            final_val_prefix = val_prefix
        else:
            final_val_prefix = os.path.join(data_root, "val") if data_root else "val"
            
        print(f"  Train prefix: {final_train_prefix}")
        print(f"  Val prefix:   {final_val_prefix}")
        
        # OPTIMIZATION: Use ThreadedGCSDataset if workers=0 (e.g. on Mac)
        if workers == 0:
            print("🚀 Optimizing for WORKERS=0: Using ThreadedGCSDataset for parallel downloads!")
            train_dataset = ThreadedGCSDataset(bucket_name, final_train_prefix, transform=train_t, 
                                               shuffle=True, num_threads=16)
            
            # Extract classes from train to ensure consistency
            train_classes = train_dataset.classes
            
            # Validation doesn't strictly need shuffling, but threading helps speed
            val_dataset   = ThreadedGCSDataset(bucket_name, final_val_prefix,   transform=val_t, 
                                               shuffle=False, num_threads=8, classes=train_classes)

            
            # DataLoader for IterableDataset must have shuffle=False (dataset handles it)
            # and num_workers=0
            train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=0, pin_memory=pin)
            val_loader   = DataLoader(val_dataset,   batch_size=min(128, batch), num_workers=0, pin_memory=pin)
            
            return train_loader, val_loader
            
        else:
            # Standard behavior for Linux/CUDA with multiple workers
            train_dataset = GCSImageFolder(bucket_name, final_train_prefix, transform=train_t)
            train_classes = train_dataset.classes
            val_dataset   = GCSImageFolder(bucket_name, final_val_prefix,   transform=val_t, classes=train_classes)

        
    else:
        train_dir = os.path.join(data_root, "train")
        val_dir   = os.path.join(data_root, "val")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_t)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=val_t)

    # pin_memory solo ayuda CPU->CUDA; en MPS/CPU no aporta
    pin = (device.type == "cuda")
    # persistent_workers=True can cause deadlocks/hangs on macOS with GCS/multiprocessing
    # We disable it to ensure workers are fresh each epoch.
    persistent = False 
    prefetch_train = 2 if workers > 0 else None

    val_workers = min(4, workers if workers else 0)
    prefetch_val = 2 if val_workers > 0 else None
    persistent_val = False

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch_train,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=min(128, batch),
        num_workers=val_workers,
        pin_memory=pin,
        persistent_workers=persistent_val,
        prefetch_factor=prefetch_val,
        drop_last=False,
    )
    return train_loader, val_loader

# ==========================================
#  WebDataset Implementation
# ==========================================

def get_webdataset_loader(
    bucket_name: str,
    prefix: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    is_train: bool = True,
    total_shards: int = 1000,
    val_shards: int = 50, # Assume validation shards are separate or subset
    train_prefix: str = "train",
    val_prefix: str = "val"
):
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError("Please install webdataset: pip install webdataset")

    # Force IPv4 to avoid potential IPv6 timeout/reset issues on some networks
    os.environ["WDS_CURL_ARGS"] = "--ipv4"

    # Construct shard URLs
    # Format: https://storage.googleapis.com/BUCKET/PREFIX/train-{000000..000999}.tar
    base_url = f"https://storage.googleapis.com/{bucket_name}/{prefix}"
    
    if is_train:
        # e.g. train-000000.tar to train-000999.tar
        # We assume shards are named train-XXXXXX.tar
        # Brace expansion syntax for WebDataset
        shard_spec = f"{base_url}/{train_prefix}-{{000000..{total_shards-1:06d}}}.tar"
        shuffle_size = 5000 # Shuffle buffer size
    else:
        # e.g. val-000000.tar
        shard_spec = f"{base_url}/{val_prefix}-{{000000..{val_shards-1:06d}}}.tar"
        shuffle_size = 0 # No shuffle for val

    print(f"[{'TRAIN' if is_train else 'VAL'}] Loading WebDataset from: {shard_spec}")

    # Transforms
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Identity mapping for classes (assuming they are already indices or we handle strings)
    # The 'cls' field in our WDS is bytes: b"n01440764"
    # We need to map this to an integer label 0..999.
    # Ideally, we should load the class_to_idx mapping.
    # For now, we will create a cached mapping or load it.
    
    # HACK: To avoid downloading the mapping every time, we can hardcode or fetch once.
    # Let's try to fetch classes from a file if it exists, or generate it.
    # Since we don't have the mapping easily available without listing files, 
    # we might need the user to provide it or we fetch it from the 'classes.txt' if we saved one.
    # For this implementation, let's assume we can fetch 'classes.json' from the bucket root or similar.
    
    # If we don't have a mapping, we can't train correctly unless the 'cls' in WDS is already an int.
    # Our conversion script saved 'cls' as the folder name (string ID).
    # We need the ImageNet synsets list.
    
    # Load classes to map string IDs to integers
    # We reuse discover_gcs_files to get the sorted class list from the original bucket location
    # This ensures consistency with the pretrained model and previous runs.
    # We assume the cache exists or we can list quickly.
    # Note: We use the ORIGINAL prefix (not WDS) to find classes if possible, 
    # or we can just list the directories in the original prefix.
    # To avoid re-listing 1.2M files, we hope the cache is there.
    
    # If we can't find cache, we might be in trouble for speed.
    # But let's try to use the cache from the standard dataset.
    # We'll call discover_gcs_files with the ORIGINAL prefix (passed as argument or inferred).
    # Actually, we can just pass the original prefix to this function if needed.
    # For now, let's assume 'train' in the bucket has the classes.
    
    # Optimization: Try to load just classes from a 'classes.json' if we uploaded one.
    # If not, fall back to discover_gcs_files but maybe we can optimize it to not list all files?
    # discover_gcs_files caches results, so it should be fast if already run.
    
    _, classes = discover_gcs_files(bucket_name, "ILSVRC2012_img_train" if is_train else "ILSVRC2012_img_val")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    def make_sample(sample):
        # sample is a dict: {'__key__': ..., 'jpg': bytes, 'cls': bytes}
        image_bytes = sample['jpg']
        class_bytes = sample['cls']
        
        # Decode image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if transform:
            img = transform(img)
            
        # Decode class
        class_str = class_bytes.decode('utf-8')
        label = class_to_idx.get(class_str, -1)
        
        # If label is -1, it means we found a class not in our list. 
        # Should not happen if lists match.
        
        return img, label

    # Create WebDataset pipeline
    dataset = (
        wds.WebDataset(shard_spec, resampled=True, handler=wds.warn_and_continue) # resampled=True for infinite stream (good for training)
        .shuffle(shuffle_size)
        .map(make_sample)
        .to_tuple(0, 1) # (img, label)
    )
    
    # DataLoader
    # We use batch_size=None because WebDataset handles batching if we wanted, 
    # but standard PyTorch DataLoader with batch_size works fine too if we yield single items.
    # However, for WDS, it's often better to batch in the pipeline. 
    # But to keep it compatible with our loop, we'll let DataLoader handle batching.
    # num_workers > 0 is fine with WDS.
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader

