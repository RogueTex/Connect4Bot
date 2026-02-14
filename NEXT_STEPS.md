# Next steps: Connect4 Anvil app + EC2 backend

This document is the master checklist for going live: how the Uplink works, whether to publish the Anvil app, and the exact order of steps. For more detail, see [AWS_LIGHTSAIL_DEPLOY.md](AWS_LIGHTSAIL_DEPLOY.md) and [ANVIL_CLONE.md](ANVIL_CLONE.md).

**Published app URL:** https://MSBA25optim2-30.anvil.app  

**EC2/Lightsail files:** All deployment files (Dockerfile, docker-compose, requirements, server.py, etc.) are in the **[aws-deploy](aws-deploy/)** folder. Upload the contents of that folder to your server.

---

## 1. How the Uplink works

The backend on **EC2 runs server.py**, which opens an **outbound** connection **to** Anvil (to `anvil.works`). So:

- **Uplink is not “Anvil connecting to EC2.”**
- **EC2 connects to Anvil.**
- No inbound port or public URL is required on EC2 for the Uplink; the container only needs outbound internet.

When a user opens your Anvil app (in the browser), the app calls `anvil.server.call('get_ai_move', ...)` and `anvil.server.call('check_winner_server', ...)`. Anvil routes those calls to the **connected Uplink** (your server.py on EC2). So the flow is:

**Browser → Anvil → Uplink (EC2)**

---

## 2. Should you publish the Anvil app?

**Publish** in Anvil = make the app available at a public URL (e.g. `yourapp.anvil.app`). It is **optional**.

- **If you only want to test:** Run the app from the Anvil editor (Run → Run App). No need to publish. As long as the Uplink is connected and server.py is running on EC2, the app will use the EC2 backend.
- **If you want others to use it (or a stable link):** Use **Publish** in Anvil (e.g. Publish → Publish App) to get a public URL. The same Uplink applies: when anyone opens that URL, the app still uses the EC2 backend while the Uplink is connected.

**Summary:** Enable and connect the Uplink first; publishing is a separate, optional step for getting a public link.

---

## 3. Ordered next steps (checklist)

Follow in order. The Anvil clone Server Module change is already committed (push if you haven’t yet).

### On the Anvil side (do first)

1. **Push the Anvil app** (if not already): From `anvil-files/Connect_4`, run `git push` so the updated Server Module (Uplink-only) is on Anvil.
2. **Enable the Uplink:** In the Anvil app, go to **App → Settings → Uplink**. Turn the Uplink on and copy the **Uplink key** (you will paste it into server.py).
3. **(Optional) Publish the app:** Only if you want a public URL; otherwise run the app from the editor for testing.

### On the EC2/Lightsail side

4. **Edit server.py locally (in `aws-deploy/`):** Set `ANVIL_UPLINK_KEY` to the key from step 2. Set `MODEL_PATH` to the correct path for your server (e.g. `/FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras` for EC2 Ubuntu or `/FOLDERNAME/bitnami/connect4app/connect4_cnn_best.keras` for Lightsail).
5. **Upload files to the server:** Upload the entire contents of the **[aws-deploy](aws-deploy/)** folder (Dockerfile, docker-compose.yml, requirements.txt, server.py, connect4_policy_player.py) plus your trained model file (e.g. `connect4_cnn_best.keras`) into one folder on the server (e.g. `/home/ubuntu/connect4app/` or `/home/bitnami/connect4app/`). See [aws-deploy/AWS_LIGHTSAIL_DEPLOY.md](aws-deploy/AWS_LIGHTSAIL_DEPLOY.md) for the full guide.
6. **Install Docker on the instance** (if not already): Use the commands in [aws-deploy/AWS_LIGHTSAIL_DEPLOY.md](aws-deploy/AWS_LIGHTSAIL_DEPLOY.md).
7. **Build and run:** In that folder on the server, run `sudo docker compose build`, then `sudo docker compose up -d`, then `sudo docker compose logs -f` to confirm you see “Backend ready.” and the Uplink connecting.

### Verify

8. **Check the Uplink in Anvil:** In the app, **Settings → Uplink** should show the backend as connected (green or “Connected”).
9. **Use the app:** Open the app at [https://MSBA25optim2-30.anvil.app](https://MSBA25optim2-30.anvil.app) (or Run from editor). Play a game; the AI should use the CNN policy from EC2. If the Uplink disconnects, the app will show an error when calling `get_ai_move` or `check_winner_server`.

---

## 4. Quick reference

- **EC2/Lightsail deployment (all files in one folder):** [aws-deploy/](aws-deploy/) and [aws-deploy/AWS_LIGHTSAIL_DEPLOY.md](aws-deploy/AWS_LIGHTSAIL_DEPLOY.md)
- **Anvil clone and Server Module edit:** [ANVIL_CLONE.md](ANVIL_CLONE.md)
- **Remaining steps for you (keys, paths, Uplink):** See “Remaining steps for you” in [aws-deploy/AWS_LIGHTSAIL_DEPLOY.md](aws-deploy/AWS_LIGHTSAIL_DEPLOY.md)
