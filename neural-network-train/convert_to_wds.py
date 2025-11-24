import os
import tarfile
import io
import time
import random
import argparse
import logging
from google.cloud import storage
import webdataset as wds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gcs_blob_stream(bucket_name, blob_name):
    """Returns a file-like object for streaming a GCS blob."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Open the blob in read mode as a stream
    return blob.open("rb")

def upload_shard(bucket_name, prefix, shard_id, shard_data):
    """Uploads a shard (bytes) to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{prefix}/train-{shard_id:06d}.tar"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(shard_data, content_type="application/x-tar")
    logger.info(f"Uploaded shard {shard_id} to gs://{bucket_name}/{blob_name}")

def process_imagenet(args):
    """
    Main processing loop:
    1. Stream outer tar (contains n01440764.tar, etc.)
    2. Extract inner tars in memory
    3. Extract images from inner tars
    4. Buffer and shuffle images
    5. Write to WebDataset shards and upload
    """
    
    # Parse GCS path
    if not args.input_path.startswith("gs://"):
        raise ValueError("Input path must be gs://bucket/path")
    
    parts = args.input_path[5:].split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    output_parts = args.output_path[5:].split('/', 1)
    out_bucket_name = output_parts[0]
    out_prefix = output_parts[1].rstrip('/')

    logger.info(f"Starting conversion ({args.mode}) from gs://{bucket_name}/{blob_name}")
    
    # Validation Labels Loading
    val_labels_map = {}
    if args.mode == 'val':
        logger.info("Fetching ImageNet validation ground truth...")
        try:
            # This file maps line N -> synset_id (e.g. n01440764)
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
            import urllib.request
            with urllib.request.urlopen(url) as f:
                lines = f.read().decode('utf-8').strip().splitlines()
            # Map filename index (1-based) to synset string
            # ILSVRC2012_val_00000001.JPEG -> lines[0]
            val_labels_map = {i+1: line.strip() for i, line in enumerate(lines)}
            logger.info(f"Loaded {len(val_labels_map)} validation labels.")
        except Exception as e:
            logger.warning(f"Could not fetch validation labels: {e}. Validation accuracy will be 0.")

    # Buffer for shuffling
    sample_buffer = []
    shard_id = args.start_shard
    
    # Open the main stream
    start_time = time.time()
    total_images = 0
    
    try:
        stream = get_gcs_blob_stream(bucket_name, blob_name)
        
        # Open outer tar
        with tarfile.open(fileobj=stream, mode="r|*") as outer_tar:
            for member in outer_tar:
                if not member.isfile():
                    continue

                # MODE: TRAIN (Nested Tars)
                if args.mode == 'train':
                    if not member.name.endswith('.tar'):
                        continue
                    
                    # Extract inner tar to memory
                    class_id = os.path.splitext(os.path.basename(member.name))[0]
                    logger.info(f"Processing class: {class_id}")
                    
                    f_obj = outer_tar.extractfile(member)
                    if f_obj is None: continue
                    
                    inner_tar_bytes = f_obj.read()
                    inner_stream = io.BytesIO(inner_tar_bytes)
                    
                    try:
                        with tarfile.open(fileobj=inner_stream, mode="r") as inner_tar:
                            for inner_member in inner_tar:
                                if not inner_member.isfile(): continue
                                fname = inner_member.name
                                if os.path.splitext(fname)[1].lower() not in ['.jpg', '.jpeg', '.png']: continue
                                    
                                img_obj = inner_tar.extractfile(inner_member)
                                if img_obj:
                                    img_data = img_obj.read()
                                    key = f"{class_id}/{os.path.splitext(fname)[0]}"
                                    sample = {
                                        "__key__": key,
                                        "jpg": img_data,
                                        "cls": class_id.encode("utf-8") 
                                    }
                                    sample_buffer.append(sample)
                                    total_images += 1
                                    
                                    if len(sample_buffer) >= args.buffer_size:
                                        # ... (Flush logic shared below or duplicated? Let's duplicate for safety/clarity in this patch)
                                        # Actually, better to refactor flush, but for patch simplicity I'll keep flow.
                                        # WAIT, I can't duplicate easily in replace_file_content without context.
                                        # I will rely on the shared flush logic if I can structure it.
                                        # But the original code had the flush inside the inner loop.
                                        # I'll implement the flush here for TRAIN.
                                        pass 
                    except Exception as e:
                        logger.error(f"Error processing inner tar {member.name}: {e}")

                # MODE: VAL (Flat Images)
                elif args.mode == 'val':
                    fname = member.name
                    if os.path.splitext(fname)[1].lower() not in ['.jpg', '.jpeg', '.png']:
                        continue
                        
                    # Parse index from filename: ILSVRC2012_val_00000001.JPEG
                    try:
                        # Extract number
                        idx_str = fname.split('_')[-1].split('.')[0]
                        idx = int(idx_str)
                        class_id = val_labels_map.get(idx, "unknown")
                    except:
                        class_id = "unknown"
                        
                    img_obj = outer_tar.extractfile(member)
                    if img_obj:
                        img_data = img_obj.read()
                        key = f"{class_id}/{os.path.splitext(fname)[0]}"
                        sample = {
                            "__key__": key,
                            "jpg": img_data,
                            "cls": class_id.encode("utf-8")
                        }
                        sample_buffer.append(sample)
                        total_images += 1

                # SHARED FLUSH LOGIC (Check buffer)
                if len(sample_buffer) >= args.buffer_size:
                    logger.info(f"Buffer full ({len(sample_buffer)}). Shuffling and writing shards...")
                    random.shuffle(sample_buffer)
                    while len(sample_buffer) >= args.shard_size:
                        chunk = sample_buffer[:args.shard_size]
                        sample_buffer = sample_buffer[args.shard_size:]
                        
                        shard_buffer = io.BytesIO()
                        with wds.TarWriter(shard_buffer) as sink:
                            for s in chunk:
                                sink.write(s)
                        
                        upload_shard(out_bucket_name, out_prefix, shard_id, shard_buffer.getvalue())
                        shard_id += 1

        # Process remaining samples in buffer
        if sample_buffer:
            logger.info(f"Processing remaining {len(sample_buffer)} samples...")
            random.shuffle(sample_buffer)
            while sample_buffer:
                chunk = sample_buffer[:args.shard_size]
                sample_buffer = sample_buffer[args.shard_size:]
                
                shard_buffer = io.BytesIO()
                with wds.TarWriter(shard_buffer) as sink:
                    for s in chunk:
                        sink.write(s)
                
                upload_shard(out_bucket_name, out_prefix, shard_id, shard_buffer.getvalue())
                shard_id += 1

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"Done! Processed {total_images} images in {duration:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ImageNet tar to WebDataset")
    parser.add_argument("--input_path", type=str, required=True, help="gs://bucket/ILSVRC2012_img_train.tar")
    parser.add_argument("--output_path", type=str, required=True, help="gs://bucket/imagenet-wds/train")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Shuffle buffer size (images)")
    parser.add_argument("--shard_size", type=int, default=2000, help="Images per shard")
    parser.add_argument("--start_shard", type=int, default=0, help="Starting shard ID")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="Mode: train (nested tars) or val (flat images)")
    
    args = parser.parse_args()
    process_imagenet(args)
