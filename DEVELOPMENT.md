# Development Guide

This guide helps new contributors set up a local development environment and understand the training pipeline.

## Prerequisites

- Python 3.10+
- - [uv](https://docs.astral.sh/uv/) or pip
  - - GPU recommended for training (Colab T4 works well)
    - - Docker (for backend deployment only)
     
      - ## Local Setup
     
      - ```bash
        # Clone the repo
        git clone https://github.com/RogueTex/Connect4Bot.git
        cd Connect4Bot

        # Create virtual environment
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\Scripts\activate

        # Install dependencies
        pip install tensorflow numpy scipy
        ```

        ## Training

        ### Which notebook to use?

        | Notebook | Purpose |
        |----------|----------|
        | `Connect4_CNN_Only_Training.ipynb` | Train the default residual CNN (recommended) |
        | `Connect4_CNN_Transformer_Training.ipynb` | Train CNN + Transformer simultaneously |
        | `Connect4_CNN2_Loss_Finetune_Colab.ipynb` | Fine-tune CNN on game-log correction data |

        ### Running on Google Colab

        1. Upload the notebook to Colab
        2. 2. Upload `datasets/connect4_combined_unique.npz` to your Drive
           3. 3. Update the dataset path in the notebook
              4. 4. Enable GPU: Runtime → Change runtime type → T4 GPU
                 5. 5. Run all cells
                   
                    6. ### Dataset Format
                   
                    7. - Input: `(N, 6, 7, 2)` board states (float32)
                       - - Labels: `(N, 7)` one-hot policy targets
                         - - Source: `datasets/connect4_combined_unique.npz`
                          
                           - ## Backend Development
                          
                           - The inference backend runs as an Anvil Uplink service on AWS Lightsail.
                          
                           - ```bash
                             # Start locally (requires Anvil Uplink key)
                             cd aws-deploy
                             pip install anvil-uplink tensorflow
                             export ANVIL_UPLINK_KEY=your_key_here
                             python server.py
                             ```

                             ## Code Style

                             - Follow PEP 8 for Python files
                             - - Keep notebooks clean: restart kernel and run all before committing
                               - - Use meaningful variable names in training code
