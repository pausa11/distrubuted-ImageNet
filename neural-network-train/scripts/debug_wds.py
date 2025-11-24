import os
import sys

# Force IPv4 and HTTP/1.1 to see if it fixes the issue
# Must be set BEFORE importing webdataset if it reads at module level (though usually it reads at call time)
os.environ["WDS_CURL_ARGS"] = "--ipv4 --http1.1 --retry 5 --retry-delay 2 --connect-timeout 30 -f -s -L" 

import webdataset as wds
import webdataset.gopen as wds_gopen
from webdataset.gopen import Pipe

# Monkey-patch gopen_curl to use our custom args
def custom_gopen_curl(url, mode="rb", bufsize=8192):
    if mode[0] == "r":
        # Use args from env or default robust ones
        args_str = os.environ.get("WDS_CURL_ARGS", "--connect-timeout 30 --retry 30 --retry-delay 2 -f -s -L")
        cmd_args = ["curl"] + args_str.split() + [url]
        print(f"DEBUG: Executing curl: {cmd_args}")
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

wds_gopen.gopen_curl = custom_gopen_curl
wds.gopen_schemes["http"] = custom_gopen_curl
wds.gopen_schemes["https"] = custom_gopen_curl

url = "https://storage.googleapis.com/caso-estudio-2/imagenet-wds/train/train-{000000..000010}.tar"

print(f"Testing WebDataset with URL: {url}")
print(f"WDS_CURL_ARGS: {os.environ.get('WDS_CURL_ARGS', 'Not Set')}")

def make_sample(sample):
    # Mock make_sample
    return sample['__key__'], 0

dataset = (
    wds.WebDataset(url)
    .map(make_sample)
    # .to_tuple(0, 1) # Ensure this is commented out here too
)

count = 0
try:
    for sample in dataset:
        count += 1
        if count % 10 == 0:
            print(f"Loaded {count} samples...", end='\r')
        if count >= 100:
            break
    print(f"\nSuccessfully loaded {count} samples.")
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
