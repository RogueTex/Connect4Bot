# Anvil Integration: Policy Player with Win/Block + Legal Masking

To maximize win rate when your Connect 4 bot plays in the Anvil web app, use `policy_move_with_rules` instead of raw model predictions. This ensures the bot never misses obvious wins or blocks and never selects illegal moves.

## Usage

### 1. Copy or import the policy player

Place `connect4_policy_player.py` in the same directory as your Anvil backend Python file (PYTHONFILE.py). Then:

```python
from connect4_policy_player import policy_move_with_rules, find_winning_move
```

If you cannot import (e.g. Anvil server structure), copy the contents of `connect4_policy_player.py` into your backend file.

### 2. Game state requirements

Your game object must implement:

- `legal_moves()` — returns list of valid column indices (0–6)
- `copy()` — returns a deep copy of the game
- `current_player` — 1 for plus, -1 for minus
- `make_move(col)` — applies move, returns success
- `winner` — set after a winning move (or 0 for draw)
- `encode(perspective=1)` — returns `(6, 7, 2)` float32 array (option-b: ch0=current player, ch1=opponent)

### 3. Getting a move from your model

```python
# Instead of:
# col = np.argmax(model.predict(board)[0])

# Use:
col = policy_move_with_rules(game, model, perspective=1)
```

For minus’s turn, encode from minus’s perspective:

```python
col = policy_move_with_rules(game, model, perspective=game.current_player)
```

### 4. Flow

`policy_move_with_rules` does:

1. **Win check** — if the current player can win, returns that column
2. **Block check** — if the opponent can win next move, returns the blocking column
3. **Model + legal masking** — otherwise uses the model, with illegal columns masked to -1e9

## Files

- `connect4_policy_player.py` — defines `find_winning_move` and `policy_move_with_rules`
- Use with both CNN and Transformer (same input/output: 6x7x2 → 7-class softmax)
