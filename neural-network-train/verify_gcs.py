from google.cloud import storage

def list_public_bucket(bucket_name):
    print(f"Attempting to access public bucket: {bucket_name}")
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(max_results=20))
        
        print(f"Successfully accessed bucket. Found {len(blobs)} items (showing first 20):")
        for blob in blobs:
            print(f" - {blob.name} ({blob.size} bytes)")
            
    except Exception as e:
        print(f"Error accessing bucket: {e}")

if __name__ == "__main__":
    list_public_bucket("caso-estudio-2")
