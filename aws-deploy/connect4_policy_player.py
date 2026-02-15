"""
Connect 4 Policy Player: Win/block + legal-move masking for CNN and Transformer.

Use policy_move_with_rules() when playing games so the bot never misses
obvious wins/blocks and never selects illegal moves.
"""

import numpy as np


def find_winning_move(game, player):
    """
    Return the column that wins for the given player, or None if no winning move exists.
    game must have: legal_moves(), copy(), current_player, make_move(col), winner
    """
    for col in game.legal_moves():
        test = game.copy()
        old_p = test.current_player
        test.current_player = player
        test.make_move(col)
        if test.winner == player:
            return col
        test.current_player = old_p
    return None


def policy_move_with_rules(game, model, perspective=1, return_debug=False):
    """
    Get the best move for the current player using:
    1. Win check: if we can win, play it
    2. Block check: if opponent can win, block it
    3. Otherwise: use model with legal-move masking

    game: Connect4 game with legal_moves(), copy(), current_player, make_move(), winner, encode()
    model: Keras model expecting (batch, 6, 7, 2) input, returns (batch, 7) softmax
    perspective: 1 for plus, -1 for minus (encoding perspective)

    Returns:
      - int column 0-6 (default)
      - (int, dict) if return_debug=True
    """
    debug = {
        "decision_source": None,
        "win_col": None,
        "block_col": None,
        "legal_moves": game.legal_moves(),
        "raw_probs": None,
        "scores": None,
    }

    # 1. Win check
    win_col = find_winning_move(game, game.current_player)
    if win_col is not None:
        debug["decision_source"] = "win_rule"
        debug["win_col"] = int(win_col)
        return (int(win_col), debug) if return_debug else int(win_col)

    # 2. Block check
    block_col = find_winning_move(game, -game.current_player)
    if block_col is not None:
        debug["decision_source"] = "block_rule"
        debug["block_col"] = int(block_col)
        return (int(block_col), debug) if return_debug else int(block_col)

    # 3. Model with legal-move masking
    x = game.encode(perspective=perspective)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    legal = debug["legal_moves"]
    mask = np.full(7, -1e9, dtype=np.float32)
    for c in legal:
        mask[c] = 0.0
    scores = probs + mask
    col = int(np.argmax(scores))
    debug["decision_source"] = "model"
    debug["raw_probs"] = [float(v) for v in probs.tolist()]
    debug["scores"] = [float(v) for v in scores.tolist()]
    return (col, debug) if return_debug else col
