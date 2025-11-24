# Logging and Metrics for Scientific Analysis

I have added a comprehensive logging system to your training script to collect data for your research paper.

## New Files
- **`metrics.py`**: A new module that handles:
    - **System Resource Monitoring**: CPU, Memory, and Network usage.
    - **Training Metrics Logging**: Loss, Accuracy, and Learning Rate.
- **`system_metrics.csv`**: Automatically created when you run the script. Logs system resources every second.
- **`training_metrics.csv`**: Automatically created when you run the script. Logs training progress for every batch.

## Data Collected

### 1. System Metrics (`system_metrics.csv`)
Logged every 1 second (configurable in `metrics.py`).
- `timestamp`: ISO 8601 timestamp.
- `cpu_percent`: CPU usage percentage.
- `memory_percent`: RAM usage percentage.
- `memory_used_gb`: RAM usage in GB.
- `net_sent_mb`: Network data sent (MB) since last sample.
- `net_recv_mb`: Network data received (MB) since last sample.

### 2. Training Metrics (`training_metrics.csv`)
Logged every training step (batch) and validation epoch.
- `timestamp`: ISO 8601 timestamp.
- `epoch`: Current global epoch.
- `batch`: Total local batches processed.
- `loss`: Training loss for the current batch.
- `accuracy`: Validation accuracy (only present when validation occurs, otherwise empty).
- `learning_rate`: Current learning rate.

## How to Use
1. **Run the training script** as usual:
   ```bash
   python3 dhtPeerImagenet.py --initial_peer ...
   ```
2. **Wait for training** to proceed.
3. **Check the CSV files** in the same directory:
   - `system_metrics.csv`
   - `training_metrics.csv`
4. **Plot the data**: You can now import these CSV files into Python (pandas/matplotlib), Excel, or any other tool to generate graphs for your paper.

## Requirements
I have added `psutil` to `requirements.txt` and installed it for you.
