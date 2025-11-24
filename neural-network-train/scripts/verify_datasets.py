
import sys
import os
import torch

# Add the current directory to sys.path to make imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from datasets import get_webdataset_loader
    print("Successfully imported get_webdataset_loader")
    
    # Try to trigger the import inside the function
    # We pass dummy arguments. The function might fail later, but we want to see if it passes the import line.
    try:
        # We need to mock GCS_AVAILABLE or ensure it doesn't crash before the import.
        # The import is at the beginning of the function (after some checks).
        # Actually, looking at the code, it's after "if num_workers == 0" check? 
        # No, it's before.
        
        # Let's just run it and catch the expected error (e.g. missing bucket) 
        # but if it's ImportError, we know it failed.
        
        get_webdataset_loader(
            bucket_name="dummy-bucket",
            prefix="dummy-prefix",
            batch_size=1,
            num_workers=0,
            device=torch.device("cpu"),
            is_train=True
        )
    except ImportError as e:
        print(f"Import failed inside function: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Function called and failed with (likely expected) error: {e}")
        print("This means the import 'from imagenet_classes import IMAGENET_SYNSETS' likely succeeded!")
        
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
