# Deploy Connect4 Anvil Backend on AWS EC2 or Lightsail

**All files in this folder (`aws-deploy`) are what you upload to the server.** Upload the entire contents of this folder (plus your trained model file) into one folder on EC2/Lightsail (e.g. `connect4app`), then follow the steps below.

This guide covers running the Connect4 Anvil backend in Docker on **AWS EC2** (Ubuntu) or **AWS Lightsail** (Bitnami). The same steps apply to both; only the home directory name differs (`ubuntu` vs `bitnami`). **Requirements use `tensorflow-cpu`** (inference only; models are already trained) for a smaller image.

---

## Prerequisites

- AWS account
- An **EC2** instance (Ubuntu 22.04 or similar) or a **Lightsail** instance (Bitnami)
- SSH access to the instance
- A way to upload files: **FileZilla**, **scp**, or **rsync**

---

## What to upload

Upload the contents of **this folder** (`aws-deploy`) into **one folder** on the server (e.g. `connect4app`):

| File | Description |
|------|--------------|
| `Dockerfile` | Builds the container (includes `server.py` and `connect4_policy_player.py`) |
| `docker-compose.yml` | Runs the Anvil uplink service with volume mount |
| `requirements.txt` | Python dependencies (tensorflow-cpu for inference only) |
| `server.py` | Anvil backend — set Uplink key and model path |
| `connect4_policy_player.py` | Policy logic (win/block + legal masking); used by `server.py` |
| Your trained model | e.g. `connect4_cnn_best.keras` or `connect4_cnn_best.h5` — place in the **same** folder |

---

## Directory layout on the server

Create one folder and put everything in it.

- **EC2 (Ubuntu):** `/home/ubuntu/connect4app/`
- **Lightsail (Bitnami):** `/home/bitnami/connect4app/`

Example contents:

```
/home/ubuntu/connect4app/   (or /home/bitnami/connect4app/)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── server.py
├── connect4_policy_player.py
└── connect4_cnn_best.keras
```

---

## Model path in `server.py`

The container mounts the host’s `/home` at `/FOLDERNAME`. So paths **inside the container** are:

- **EC2:** `/FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras`
- **Lightsail:** `/FOLDERNAME/bitnami/connect4app/connect4_cnn_best.keras`

In `server.py`, set `MODEL_PATH` to the correct one (and match your model filename, e.g. `.keras` or `.h5`):

```python
# EC2 Ubuntu:
MODEL_PATH = "/FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras"

# Lightsail Bitnami:
MODEL_PATH = "/FOLDERNAME/bitnami/connect4app/connect4_cnn_best.keras"
```

---

## Do you need to paste your Anvil code?

The `server.py` in this folder already has `get_ai_move(board)` and `check_winner_server(board, piece)` and connects with your Uplink key. Only set `MODEL_PATH` for your server (EC2 vs Lightsail) and add your trained model file to the server folder.

---

## 1. Install Docker on the instance

SSH into the instance, then run the following (Ubuntu/Debian; works on EC2 Ubuntu and Lightsail Bitnami). Reference: [Docker Engine install – Debian](https://docs.docker.com/engine/install/debian/).

**Add Docker’s repository and install:**

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Check that Docker works:**

```bash
sudo docker run hello-world
```

You should see “Hello from Docker!” and a short message.

---

## 2. Build and run the backend

1. On the server, go to the folder where you uploaded the files (e.g. `connect4app`):

   ```bash
   cd /home/ubuntu/connect4app
   # or on Lightsail:
   cd /home/bitnami/connect4app
   ```

2. Build the image:

   ```bash
   sudo docker compose build
   ```

3. Start the container in the background:

   ```bash
   sudo docker compose up -d
   ```

4. Watch logs to confirm the backend starts and connects to Anvil:

   ```bash
   sudo docker compose logs -f
   ```

   You should see something like “Backend ready.” and then Anvil connection output. Press `Ctrl+C` to stop following logs (the container keeps running).

---

## 3. Updating after changes

1. Stop the container:

   ```bash
   cd /home/ubuntu/connect4app   # or /home/bitnami/connect4app
   sudo docker compose down
   ```

2. Update files on the server (e.g. edit `server.py`, replace the model file).

3. Rebuild and start again:

   ```bash
   sudo docker compose build
   sudo docker compose up -d
   sudo docker compose logs -f
   ```

---

## 4. Optional: clean up old Docker data

To free disk space from old images and build cache:

```bash
sudo docker system prune -a --filter "until=10m"
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create a folder (e.g. `connect4app`) under `/home/ubuntu` (EC2) or `/home/bitnami` (Lightsail). |
| 2 | Upload the contents of this `aws-deploy` folder plus your model file. |
| 3 | Edit `server.py` if needed: set `MODEL_PATH` to the correct container path for your server. |
| 4 | Install Docker (see commands above). |
| 5 | `cd` to the app folder, run `sudo docker compose build`, then `sudo docker compose up -d`. |
| 6 | Check `sudo docker compose logs -f` for “Backend ready.” and Anvil connection. |
| 7 | Open your Anvil app in the browser; the bot will use this backend. |

The same setup works on **EC2** (Ubuntu AMI) and **Lightsail** (Bitnami); only the home path (`ubuntu` vs `bitnami`) and thus `MODEL_PATH` in `server.py` change.

---

## Remaining steps for you

Before or after you deploy, do the following:

1. **Set your Anvil Uplink key**  
   In `server.py`, replace the placeholder with your real key if not already set. Get it from the Anvil app: **App → Settings → Uplink**.

2. **Set the model path**  
   In `server.py`, set `MODEL_PATH` to the correct container path for your server:
   - **EC2 Ubuntu:** `/FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras` (or your model filename).
   - **Lightsail Bitnami:** `/FOLDERNAME/bitnami/connect4app/connect4_cnn_best.keras`.

3. **Put the model file on the server**  
   Upload your trained model (e.g. `connect4_cnn_best.keras` or `.h5`) into the same app folder as `server.py` (e.g. `/home/ubuntu/connect4app/` or `/home/bitnami/connect4app/`).

4. **Anvil Uplink**  
   In your Anvil app, enable the Uplink (App → Settings → Uplink) and ensure this backend is running on EC2/Lightsail. Your client already calls `get_ai_move(board)` and `check_winner_server(board, piece)` — no client code changes are required.

5. **Anvil app clone (optional)**  
   To make the Server Module rely only on the Uplink for `get_ai_move` and `check_winner_server`, clone the app and apply the edited Server Module as described in [ANVIL_CLONE.md](../ANVIL_CLONE.md). Then push the commit to Anvil from the clone directory if you have Git/SSH access.
