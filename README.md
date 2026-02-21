# Connect 4 AI (CNN + Transformer)

Course project for **RM294 — Spring 2026**.

A Connect 4 web app built with Anvil that calls an AWS/Lightsail Uplink backend to run trained models (CNN v2 as the default CNN, plus a Transformer option).

![Connect 4 landing](assetsForReadme/Connect%204%20Image.png)

![Gameplay](assetsForReadme/Gameplay.gif)

## What We Built

- **Anvil UI** with login, gameplay, and a CNN vs Transformer model selector.
- **Uplink backend on AWS/Lightsail** that loads models and serves `get_ai_move` and `check_winner_server`.
- **Training pipeline** using MCTS data and CNN/Transformer notebooks.

## Data + Modeling

- **Game data generation**: Self-play and MCTS rollouts were used to create state-action pairs for training. Logs are stored as JSONL for later analysis and finetuning.
- **Modeling strategy**: CNN v2 is the default policy model, with a Transformer model available as an alternative. Inference uses a policy + rules filter for legal moves.
- **Metrics (summary)**: We tracked win rates in head-to-head evaluation and sanity-checked policy quality with AI-vs-AI games and human playtests.

## How It Works (High Level)

1. User plays in the Anvil app.
2. The Anvil client calls server functions via Uplink.
3. The AWS container loads the model(s) and returns the AI move.
4. Game results and optional logs are persisted as JSONL.

## Deployment Steps (Summary)

1. **Anvil**: Enable Uplink and get the Uplink key.
2. **AWS/Lightsail**: Upload `aws-deploy/` and model files to the instance.
3. **Configure**: Set `ANVIL_UPLINK_KEY` and model paths in `aws-deploy/server.py`.
4. **Run**:
   - `docker-compose build`
   - `docker-compose up -d`
   - `docker-compose logs -f`
5. **Verify**: Uplink shows “Connected” and the app returns AI moves.

If the published Anvil URL is used, ensure the Uplink is routed to the published environment or enable “Send server calls from other environments to this Uplink”.

## Links

- **Anvil app (published)**: https://msba25optim2-group30.anvil.app/
- **Anvil app clone**: https://anvil.works/build#clone:ZYMBBJME4FOBYPVB=MRHPAQMRQHGS5L7UHINHPSFB

## Repository Structure

- `anvil-files/` — Anvil app source (client + server modules, theme assets)
- `aws-deploy/` — Dockerized Uplink backend (server, compose, requirements)
- `training/` — Model training notebooks
- `dataset_generators/` — Data generation utilities
- `assets/` — UI and report assets used in the project
- `assetsForReadme/` — README screenshots and gameplay GIF
- `FilesOnLightsail/` — Snapshot of files staged on the Lightsail instance

## Notes

- CNN v2 is the active CNN model used by the UI (mapped to `cnn2`).
- Transformer model is optional and can be disabled if memory constrained.
