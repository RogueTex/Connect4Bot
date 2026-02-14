# AWS EC2 / Lightsail deployment

This folder contains everything needed to run the Connect4 Anvil backend on AWS EC2 or Lightsail.

**Contents:**

- `Dockerfile` – builds the container
- `docker-compose.yml` – runs the Anvil uplink service
- `requirements.txt` – **tensorflow-cpu** only (inference; models are already trained). Smaller image than full TensorFlow.
- `server.py` – Anvil backend (`get_ai_move`, `check_winner_server`). Set `MODEL_PATH` for your server.
- `connect4_policy_player.py` – policy logic used by `server.py`
- `AWS_LIGHTSAIL_DEPLOY.md` – full deployment instructions

**To deploy:** Upload everything in this folder (plus your trained model file, e.g. `connect4_cnn_best.keras`) to a folder on your EC2/Lightsail instance, then follow **AWS_LIGHTSAIL_DEPLOY.md**.

High-level steps: [NEXT_STEPS.md](../NEXT_STEPS.md) in the project root.
