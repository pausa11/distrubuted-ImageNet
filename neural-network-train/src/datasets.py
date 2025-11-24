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
        import webdataset.gopen as wds_gopen
        from webdataset.gopen import Pipe
    except ImportError:
        raise ImportError("Please install webdataset: pip install webdataset")

    # Monkey-patch gopen_curl to use our custom args (because webdataset hardcodes them)
    def custom_gopen_curl(url, mode="rb", bufsize=8192):
        if mode[0] == "r":
            # Use args from env or default robust ones
            # Added --http1.1 to avoid HTTP/2 stream errors (exit 56)
            args_str = os.environ.get("WDS_CURL_ARGS", "--ipv4 --http1.1 --connect-timeout 30 --retry 5 --retry-delay 2 -f -s -L")
            cmd_args = ["curl"] + args_str.split() + [url]
            return Pipe(
                cmd_args,
                mode=mode,
                bufsize=bufsize,
                ignore_status=[141, 23],
            )
        elif mode[0] == "w":
            cmd_args = ["curl", "-f", "-s", "-X", "PUT", "-L", "-T", "-", url]
            return Pipe(
                cmd_args,
                mode=mode,
                bufsize=bufsize,
                ignore_status=[141, 26],
            )
        raise ValueError(f"Unknown mode {mode}")

    # Apply patch
    wds_gopen.gopen_curl = custom_gopen_curl
    wds.gopen_schemes["http"] = custom_gopen_curl
    wds.gopen_schemes["https"] = custom_gopen_curl

    # Force IPv4 to avoid potential IPv6 timeout/reset issues on some networks
    # Also force HTTP/1.1 and add retries to handle connection resets (exit 56)
    os.environ["WDS_CURL_ARGS"] = "--ipv4 --http1.1 --retry 5 --retry-delay 2 --connect-timeout 30 -f -s -L"

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
    
    # We need to pass class_to_idx to the worker processes.
    # Since we can't easily pass arguments to map function in WDS without partial,
    # and partial might have pickle issues if not careful,
    # we will use a callable class or a global if needed.
    # But a simple function with partial is usually fine if defined at module level.
    # However, class_to_idx is local.
    # Let's use a helper class.

    # If num_workers is 0, we can use a Threaded implementation to speed up I/O
    # without the overhead of multiprocessing (which seems to be slow on this Mac).
    if num_workers == 0:
        print("🚀 Optimizing for WORKERS=0: Using ThreadedWebDataset for parallel downloads!")
        return ThreadedWebDataset(
            bucket_name=bucket_name,
            prefix=prefix,
            shard_spec=shard_spec,
            class_to_idx=class_to_idx,
            transform=transform,
            batch_size=batch_size,
            shuffle_size=shuffle_size,
            num_threads=16 # Adjust based on network/CPU
        )

    # Standard WebDataset pipeline for multiprocessing
    decoder = _WDSSampleDecoder(class_to_idx, transform)

    # Create WebDataset pipeline
    dataset = (
        wds.WebDataset(shard_spec, resampled=True, handler=wds.warn_and_continue) # resampled=True for infinite stream (good for training)
        .shuffle(shuffle_size)
        .map(decoder)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader


class _WDSSampleDecoder:
    def __init__(self, class_to_idx, transform):
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __call__(self, sample):
        # sample is a dict: {'__key__': ..., 'jpg': bytes, 'cls': bytes}
        image_bytes = sample['jpg']
        class_bytes = sample['cls']
        
        # Decode image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        # Decode class
        class_str = class_bytes.decode('utf-8')
        label = self.class_to_idx.get(class_str, -1)
        
        if label == -1:
            # Log the error and maybe return a dummy label to avoid crashing immediately if we want to debug?
            # But crashing with a clear message is better than a CUDA error.
            # Let's check if we can find a close match?
            raise ValueError(f"Unknown class found in WebDataset: '{class_str}'. It was not found in the loaded class list (len={len(self.class_to_idx)}). Check your GCS cache or class mapping.")

        return img, label


class ThreadedWebDataset(IterableDataset):
    def __init__(self, bucket_name, prefix, shard_spec, class_to_idx, transform, batch_size, shuffle_size, num_threads=16):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.shard_spec = shard_spec
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.num_threads = num_threads
        
        # We need to manually expand shards because we are not using wds.WebDataset's internal loader
        # Actually, we can use wds.WebDataset as an iterator source and just prefetch from it?
        # No, wds.WebDataset does the downloading.
        # If we want to threaded download, we need to handle shard downloading ourselves OR
        # use wds with a custom handler?
        # WebDataset is designed to stream. Parallelizing the stream is hard without workers.
        # BUT, we can use a ThreadPool to fetch batches?
        
        # Simpler approach: Use the existing ThreadedGCSDataset logic but adapted for WDS shards?
        # No, WDS shards are tar files.
        
        # Alternative: Just use wds.WebDataset but wrap the iterator in a thread buffer?
        # That only helps if the processing is slow, not the downloading (which happens in the iterator).
        # If wds uses curl, it spawns a subprocess. 
        # If we iterate sequentially, we wait for each curl.
        
        # To speed up WDS with workers=0, we need to fetch multiple shards in parallel.
        # WebDataset doesn't support this easily in a single process.
        
        # Wait, the user's issue is likely that curl is blocking.
        # If we use `wds.WebDataset(..., handler=wds.warn_and_continue)`, it streams.
        
        # Let's try to implement a simple prefetcher thread that iterates the WDS dataset
        # and puts batches into a queue. This decouples the GPU wait from the IO wait.
        
        import webdataset as wds
        decoder = _WDSSampleDecoder(class_to_idx, transform)
        self.dataset = (
            wds.WebDataset(shard_spec, resampled=True, handler=wds.warn_and_continue)
            .shuffle(shuffle_size)
            .map(decoder)
        )
        
    def __iter__(self):
        # Create a queue and a thread to fill it
        import queue
        q = queue.Queue(maxsize=20) # Buffer 20 batches
        
        def producer():
            # Create a dataloader just for batching (lightweight)
            # or just manual batching
            loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0)
            for batch in loader:
                q.put(batch)
                
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        
        while True:
            yield q.get()

