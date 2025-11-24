from google.cloud import storage
import os

def list_blobs(bucket_name, prefix):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix, max_results=20)
    
    print(f"Listing blobs in gs://{bucket_name}/{prefix}")
    found = False
    for blob in blobs:
        print(blob.name)
        found = True
    
    if not found:
        print("No blobs found.")

if __name__ == "__main__":
    list_blobs("caso-estudio-2", "imagenet-wds/val/")
