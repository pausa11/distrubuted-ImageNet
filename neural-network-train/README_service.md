# Deploying as a Systemd Service

To keep the initial peer running permanently on your Google Cloud VM, follow these steps.

## 1. Upload Files
Upload the code and the service file to your VM. You can use `scp` or `gcloud compute scp`.

```bash
# Example upload (run from your local machine)
gcloud compute scp --recurse neural-network-train/ YOUR_VM_NAME:~/
```

## 2. Install Dependencies
SSH into your VM and ensure dependencies are installed.

```bash
ssh YOUR_VM_IP
cd neural-network-train
pip3 install -r requirements.txt
```

## 3. Configure the Service
1. **Edit `imagenet-training.service`**:
   - Update `User=` to your VM username (e.g., `daniel`).
   - Update `WorkingDirectory=` to the absolute path of the `neural-network-train` folder on the VM (e.g., `/home/daniel/neural-network-train`).
   - Check `ExecStart=` path to python (run `which python3` to confirm).

2. **Copy to systemd**:
   ```bash
   sudo cp imagenet-training.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

## 4. Start and Enable
Start the service and enable it to run on boot.

```bash
sudo systemctl start imagenet-training
sudo systemctl enable imagenet-training
```

## 5. Check Status
Verify it's running and check logs.

```bash
# Check status
sudo systemctl status imagenet-training

# View logs
sudo journalctl -u imagenet-training -f
```

## 6. Verify GCS Announcement
The service should announce its IP to `gs://caso-estudio-2/initial_peer.txt`. You can verify this locally:

```bash
gsutil cat gs://caso-estudio-2/initial_peer.txt
```
