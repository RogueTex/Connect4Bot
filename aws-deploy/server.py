# Connect4 Anvil backend – runs in Docker on EC2/Lightsail.
#
# Exposes get_ai_move(board) and check_winner_server(board, piece). No Anvil client
# code changes needed. Set ANVIL_UPLINK_KEY and MODEL_PATH below before deployment.

import numpy as np
import anvil.server
from tensorflow import keras

from connect4_policy_player import policy_move_with_rules

# ----- Configuration: set these for your deployment -----
# Get your Uplink key from the Anvil app: App → Settings → Uplink
ANVIL_UPLINK_KEY = "PASTE_YOUR_ANVIL_UPLINK_KEY"  # From Anvil: App → Settings → Uplink (do not commit real key)

# Container path to the model file. The volume maps host /home to container /FOLDERNAME.
# EC2 Ubuntu:   /FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras
# Lightsail:   /FOLDERNAME/bitnami/connect4app/connect4_cnn_best.keras
# Put your .keras or .h5 file in the same folder as server.py on the server.
MODEL_PATH = "/FOLDERNAME/ubuntu/connect4app/connect4_cnn_best.keras"


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


# Load model once at startup
model = keras.models.load_model(MODEL_PATH)


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
def get_ai_move(board):
    """
    Returns the AI (policy) move for the current board. Board: 6x7 list of lists.
    Uses 0 = empty, 1 and -1 (or 1 and 2; 2 is normalized to -1) for the two players.
    Current player is inferred from the number of pieces on the board.
    """
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
    current_player = 1 if n_pieces % 2 == 0 else -1
    game = SimpleConnect4(grid=grid, current_player=current_player)
    valid_cols = game.legal_moves()
    if not valid_cols:
        return None
    col = policy_move_with_rules(game, model, perspective=game.current_player)
    return col


if __name__ == "__main__":
    anvil.server.connect(ANVIL_UPLINK_KEY)
    print("Backend ready.")
    anvil.server.wait_forever()
