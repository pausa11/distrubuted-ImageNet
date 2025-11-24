#!/bin/bash

# Stop on error
set -e

echo "🚀 Setting up environment for WebDataset conversion..."

# 1. Activate Virtual Environment
if [ -f "../venv/bin/activate" ]; then
    echo "Activating existing venv (../venv)..."
    source ../venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating existing venv (./venv)..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 2. Install dependencies
pip install google-cloud-storage webdataset

# 2. Run the conversion script
# Adjust the bucket name if different!
BUCKET_NAME="caso-estudio-2"
INPUT_PATH="gs://${BUCKET_NAME}/ILSVRC2012_img_train.tar"
OUTPUT_PATH="gs://${BUCKET_NAME}/imagenet-wds/train"

echo "Starting conversion..."
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"

# Run with unbuffered output
python3 -u convert_to_wds.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --buffer_size 50000 \
  --shard_size 2000

echo "✅ Conversion complete!"
