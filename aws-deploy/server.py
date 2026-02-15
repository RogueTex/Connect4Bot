# Connect4 Anvil backend – runs in Docker on EC2/Lightsail.
#
# Exposes:
# - get_ai_move(board, model_type, game_id=None, metadata=None)
# - check_winner_server(board, piece)
# - start_game_log(model_type="cnn", game_id=None, metadata=None)
# - log_game_result(game_id, result, winner=None, final_board=None, metadata=None)
# model_type is "cnn", "cnn2", or "transformer". Set ANVIL_UPLINK_KEY and MODEL_PATH_* below.

import json
import os
import hashlib
import uuid
from datetime import datetime, timezone

import numpy as np
import anvil.server
from tensorflow import keras

from connect4_policy_player import policy_move_with_rules

# ----- Configuration: set these for your deployment -----
# Get your Uplink key from the Anvil app: App → Settings → Uplink
ANVIL_UPLINK_KEY = "server_LHX5YY2FM3VENGWKOFBRDFEF-ZYMBBJME4FOBYPVB"

# Container paths to the model files. The volume maps host /home to container /FOLDERNAME.
# Files in /home/ubuntu/ on host -> /FOLDERNAME/ubuntu/ in container. Use .keras (Keras 3 format; requires TF 2.16+).
MODEL_PATH_CNN = "/FOLDERNAME/ubuntu/connect4_cnn_final.keras"
MODEL_PATH_CNN2 = "/FOLDERNAME/ubuntu/connect4_cnn_v2_final.keras"
MODEL_PATH_TRANSFORMER = "/FOLDERNAME/ubuntu/connect4_transformer_final.keras"

# Optional game-decision logging for model improvement loops.
ENABLE_GAME_LOGGING = os.getenv("ENABLE_GAME_LOGGING", "0") == "1"
GAME_LOG_PATH = os.getenv("GAME_LOG_PATH", "/FOLDERNAME/ubuntu/connect4_game_logs.jsonl")


class SimpleConnect4:
    """Minimal game object for policy_move_with_rules: legal_moves, copy, current_player, make_move, winner, encode."""

    ROWS, COLS = 6, 7

    def __init__(self, grid=None, current_player=1):
        if grid is None:
            self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        else:
            self.grid = np.array(grid, dtype=np.int8)
        self.current_player = int(current_player)
        self.winner = 0
        self._check_winner()

    def legal_moves(self):
        return [c for c in range(self.COLS) if self.grid[0, c] == 0]

    def copy(self):
        other = SimpleConnect4(self.grid.copy(), self.current_player)
        other.winner = self.winner
        return other

    def make_move(self, col):
        if col not in self.legal_moves():
            return False
        for r in range(self.ROWS - 1, -1, -1):
            if self.grid[r, col] == 0:
                self.grid[r, col] = self.current_player
                break
        self._check_winner()
        self.current_player *= -1
        return True

    def _check_winner(self):
        g = self.grid
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if g[r, c] != 0 and g[r, c] == g[r, c + 1] == g[r, c + 2] == g[r, c + 3]:
                    self.winner = g[r, c]
                    return
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if g[r, c] != 0 and g[r, c] == g[r + 1, c] == g[r + 2, c] == g[r + 3, c]:
                    self.winner = g[r, c]
                    return
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if g[r, c] != 0 and g[r, c] == g[r + 1, c + 1] == g[r + 2, c + 2] == g[r + 3, c + 3]:
                    self.winner = g[r, c]
                    return
                if g[r + 3, c] != 0 and g[r + 3, c] == g[r + 2, c + 1] == g[r + 1, c + 2] == g[r, c + 3]:
                    self.winner = g[r + 3, c]
                    return

    def encode(self, perspective=1):
        # Returns (6, 7, 2) float32: ch0 = current player stones, ch1 = opponent stones.
        g = self.grid.astype(np.float32)
        if perspective == 1:
            cur, opp = 1.0, -1.0
        else:
            cur, opp = -1.0, 1.0
        ch0 = (g == cur).astype(np.float32)
        ch1 = (g == opp).astype(np.float32)
        return np.stack([ch0, ch1], axis=-1)


# Custom layers for Transformer (from training notebook)
class BoardPatchEmbedding(keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
    def build(self, input_shape):
        self.proj = keras.layers.Dense(self.embed_dim)
        super().build(input_shape)
    def call(self, x):
        import tensorflow as tf
        x = tf.reshape(x, [-1, 42, 2])
        return self.proj(x)

class SinusoidalPositionalEmbedding(keras.layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.embed_dim = seq_len, embed_dim
    def build(self, input_shape):
        pe = np.zeros((1, self.seq_len, self.embed_dim), dtype=np.float32)
        for pos in range(self.seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[0, pos, i] = np.sin(pos / 10000**(i / self.embed_dim))
                if i + 1 < self.embed_dim:
                    pe[0, pos, i+1] = np.cos(pos / 10000**(i / self.embed_dim))
        self.pos_emb = self.add_weight(shape=(1, self.seq_len, self.embed_dim),
            initializer=keras.initializers.Constant(pe), trainable=False)
        super().build(input_shape)
    def call(self, x):
        return x + self.pos_emb

CUSTOM_OBJECTS = {"BoardPatchEmbedding": BoardPatchEmbedding, "SinusoidalPositionalEmbedding": SinusoidalPositionalEmbedding}

# Load models once at startup (Transformer needs custom_objects for custom layers)
model_cnn = keras.models.load_model(MODEL_PATH_CNN)
try:
    model_cnn2 = keras.models.load_model(MODEL_PATH_CNN2)
except Exception as exc:
    print(f"Warning: could not load CNN2 model at {MODEL_PATH_CNN2}: {exc}")
    print("Falling back to CNN v1 model for model_type='cnn2'.")
    model_cnn2 = model_cnn
model_transformer = keras.models.load_model(MODEL_PATH_TRANSFORMER, custom_objects=CUSTOM_OBJECTS)


def _append_game_log(entry):
    if not ENABLE_GAME_LOGGING:
        return
    try:
        os.makedirs(os.path.dirname(GAME_LOG_PATH), exist_ok=True)
        with open(GAME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as exc:
        print(f"Warning: failed to write game log to {GAME_LOG_PATH}: {exc}")


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _normalize_game_result(result):
    # Normalize many possible client strings to {win, loss, draw}
    if result is None:
        return "unknown"
    val = str(result).strip().lower()
    if val in {"win", "won", "ai_win", "bot_win"}:
        return "win"
    if val in {"loss", "lose", "lost", "ai_loss", "bot_loss", "human_win"}:
        return "loss"
    if val in {"draw", "tie", "stalemate"}:
        return "draw"
    return val


@anvil.server.callable
def check_winner_server(board, piece):
    # Optimized 4-in-a-row detection
    for r in range(6):
        for c in range(4):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    for r in range(3):
        for c in range(7):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    for r in range(3):
        for c in range(4):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    for r in range(3, 6):
        for c in range(4):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False


@anvil.server.callable
def start_game_log(model_type="cnn", game_id=None, metadata=None):
    """
    Starts a game lifecycle for logging and returns a stable game_id.
    Call this once from Anvil when a new game begins.
    """
    gid = str(game_id).strip() if game_id else str(uuid.uuid4())
    _append_game_log(
        {
            "event_type": "game_start",
            "timestamp_utc": _utc_now_iso(),
            "game_id": gid,
            "model_type": (model_type or "cnn").lower(),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    )
    return gid


@anvil.server.callable
def log_game_result(game_id, result, winner=None, final_board=None, metadata=None):
    """
    Logs final game outcome. Call this once when game ends.
    result should usually be one of: win/loss/draw (from AI perspective).
    """
    gid = str(game_id).strip() if game_id else ""
    if not gid:
        return False

    final_board_norm = None
    if isinstance(final_board, list) and len(final_board) == 6:
        final_board_norm = np.array(final_board, dtype=np.int8).tolist()

    _append_game_log(
        {
            "event_type": "game_end",
            "timestamp_utc": _utc_now_iso(),
            "game_id": gid,
            "result": _normalize_game_result(result),
            "winner": winner,
            "final_board": final_board_norm,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    )
    return True


@anvil.server.callable
def get_ai_move(board, model_type="cnn", game_id=None, metadata=None):
    """
    Returns the AI (policy) move for the current board.
    board: 6x7 list of lists. 0 = empty, 1 and 2 (or 1 and -1) for the two players.
    model_type: "cnn", "cnn2", or "transformer" (which bot to play against).
    game_id: optional stable id for joining move logs with final outcome logs.
    metadata: optional dict for extra context (difficulty, client version, etc).
    Current player is inferred from the number of pieces on the board.
    """
    model_key = (model_type or "cnn").lower()
    if model_key == "transformer":
        model = model_transformer
    elif model_key == "cnn2":
        model = model_cnn2
    else:
        model = model_cnn
    # Normalize: if Anvil uses 1 and 2, convert 2 -> -1 for the policy player
    grid = [list(row) for row in board]
    has_two = any(cell == 2 for row in grid for cell in row)
    if has_two:
        for r in range(6):
            for c in range(7):
                if grid[r][c] == 2:
                    grid[r][c] = -1
    grid = np.array(grid, dtype=np.int8)
    # Infer current player: even number of pieces -> player 1, odd -> player -1
    n_pieces = np.count_nonzero(grid)
    turn_index = int(n_pieces)
    current_player = 1 if n_pieces % 2 == 0 else -1
    game = SimpleConnect4(grid=grid, current_player=current_player)
    valid_cols = game.legal_moves()
    if not valid_cols:
        return None
    col, debug = policy_move_with_rules(
        game,
        model,
        perspective=game.current_player,
        return_debug=True,
    )
    _append_game_log(
        {
            "event_type": "ai_move",
            "timestamp_utc": _utc_now_iso(),
            "game_id": str(game_id).strip() if game_id else None,
            "model_type": model_key,
            "turn_index": turn_index,
            "current_player": int(game.current_player),
            "board_hash": hashlib.sha1(grid.tobytes()).hexdigest(),
            "board": grid.tolist(),
            "chosen_col": int(col),
            "decision_source": debug.get("decision_source"),
            "win_col": debug.get("win_col"),
            "block_col": debug.get("block_col"),
            "legal_moves": debug.get("legal_moves"),
            "raw_probs": debug.get("raw_probs"),
            "scores": debug.get("scores"),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    )
    return col


if __name__ == "__main__":
    if ENABLE_GAME_LOGGING:
        print(f"Game logging enabled. Writing JSONL to: {GAME_LOG_PATH}")
    anvil.server.connect(ANVIL_UPLINK_KEY)
    print("Backend ready.")
    anvil.server.wait_forever()
