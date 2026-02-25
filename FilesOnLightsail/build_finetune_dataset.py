"""
Build a fine-tuning dataset from Connect4 JSONL logs.

Expected log events:
- ai_move (written by get_ai_move)
- game_end (written by log_game_result)

Output:
- .npz with X_train (N, 6, 7, 2), y_train (N,), sample_weight (N,)
"""

import argparse
import json
from collections import Counter

import numpy as np


def _encode_from_grid(grid, current_player):
    g = np.array(grid, dtype=np.int8)
    cur = 1 if int(current_player) >= 0 else -1
    opp = -cur
    ch0 = (g == cur).astype(np.float32)
    ch1 = (g == opp).astype(np.float32)
    return np.stack([ch0, ch1], axis=-1)


def _load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from Connect4 logs.")
    parser.add_argument("--log", required=True, help="Path to connect4_game_logs.jsonl")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--win-weight", type=float, default=1.5, help="Sample weight for moves in won games")
    parser.add_argument("--draw-weight", type=float, default=1.0, help="Sample weight for moves in drawn games")
    parser.add_argument("--loss-weight", type=float, default=0.6, help="Sample weight for moves in lost games")
    parser.add_argument(
        "--keep-model",
        default="",
        help="Optional model filter (cnn/transformer). Empty means keep all.",
    )
    args = parser.parse_args()

    events = _load_events(args.log)
    if not events:
        raise SystemExit("No events found in log file.")

    end_info_by_game = {}
    for e in events:
        if e.get("event_type") == "game_end" and e.get("game_id"):
            end_info_by_game[str(e["game_id"])] = {
                "result": str(e.get("result", "unknown")).lower(),
                "winner_model": str(e.get("winner_model", "")).lower() if e.get("winner_model") else "",
            }

    X, y, w = [], [], []
    meta_game_id, meta_model, meta_source, meta_result, meta_turn = [], [], [], [], []

    for e in events:
        if e.get("event_type") != "ai_move":
            continue
        model_type = str(e.get("model_type", "cnn")).lower()
        actor_model = str(e.get("actor_model", model_type)).lower()
        if args.keep_model and actor_model != args.keep_model.lower():
            continue
        board = e.get("board")
        chosen_col = e.get("chosen_col")
        current_player = e.get("current_player")
        if board is None or chosen_col is None or current_player is None:
            continue

        game_id = str(e.get("game_id") or "")
        end_info = end_info_by_game.get(game_id, {})
        result = str(end_info.get("result", "unknown")).lower()
        winner_model = str(end_info.get("winner_model", "")).lower()
        if winner_model:
            if actor_model == winner_model:
                result = "win"
            else:
                result = "loss"
        elif result == "draw":
            result = "draw"
        if result == "win":
            sw = args.win_weight
        elif result == "loss":
            sw = args.loss_weight
        elif result == "draw":
            sw = args.draw_weight
        else:
            sw = 1.0

        X.append(_encode_from_grid(board, current_player))
        y.append(int(chosen_col))
        w.append(float(sw))
        meta_game_id.append(game_id)
        meta_model.append(actor_model)
        meta_source.append(str(e.get("decision_source", "")))
        meta_result.append(result)
        meta_turn.append(int(e.get("turn_index", -1)))

    if not X:
        raise SystemExit("No ai_move samples found after filtering.")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    w_arr = np.array(w, dtype=np.float32)

    np.savez_compressed(
        args.out,
        X_train=X_arr,
        y_train=y_arr,
        sample_weight=w_arr,
        game_id=np.array(meta_game_id, dtype=object),
        model_type=np.array(meta_model, dtype=object),
        decision_source=np.array(meta_source, dtype=object),
        game_result=np.array(meta_result, dtype=object),
        turn_index=np.array(meta_turn, dtype=np.int64),
    )

    result_counts = Counter(meta_result)
    print(f"Saved: {args.out}")
    print(f"Samples: {len(X_arr)}")
    print(f"Result mix: {dict(result_counts)}")
    print(f"Move distribution: {np.bincount(y_arr, minlength=7).tolist()}")


if __name__ == "__main__":
    main()
