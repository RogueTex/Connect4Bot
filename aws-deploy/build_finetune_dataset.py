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
from dataclasses import dataclass

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


@dataclass
class TeacherLabel:
    move: int
    source: str
    score: float = 0.0


def _check_winner(board, player):
    p = int(player)
    g = board
    for r in range(6):
        for c in range(7):
            if g[r][c] != p:
                continue
            if c <= 3 and all(g[r][c + k] == p for k in range(4)):
                return True
            if r <= 2 and all(g[r + k][c] == p for k in range(4)):
                return True
            if r <= 2 and c <= 3 and all(g[r + k][c + k] == p for k in range(4)):
                return True
            if r >= 3 and c <= 3 and all(g[r - k][c + k] == p for k in range(4)):
                return True
    return False


def _legal_moves(board):
    return [c for c in range(7) if board[0][c] == 0]


def _apply_move(board, col, player):
    b = [row[:] for row in board]
    for r in range(5, -1, -1):
        if b[r][col] == 0:
            b[r][col] = int(player)
            return b, r
    return None, None


def _find_winning_move(board, player):
    for c in _legal_moves(board):
        nb, _ = _apply_move(board, c, player)
        if nb is not None and _check_winner(nb, player):
            return c
    return None


def _random_policy_move(board, player, rng):
    win_col = _find_winning_move(board, player)
    if win_col is not None:
        return win_col
    block_col = _find_winning_move(board, -player)
    if block_col is not None:
        return block_col
    legal = _legal_moves(board)
    if not legal:
        return None
    return int(rng.choice(np.array(legal, dtype=np.int8)))


def _rollout_value(board, player_to_move, root_player, rng, max_steps=42):
    b = [row[:] for row in board]
    cur = int(player_to_move)

    for _ in range(max_steps):
        move = _random_policy_move(b, cur, rng)
        if move is None:
            return 0.0
        b, _ = _apply_move(b, move, cur)
        if _check_winner(b, cur):
            return 1.0 if cur == root_player else -1.0
        if not _legal_moves(b):
            return 0.0
        cur = -cur
    return 0.0


def _teacher_move_mcts(board, current_player, rollouts, rng):
    legal = _legal_moves(board)
    if not legal:
        return None

    win_col = _find_winning_move(board, current_player)
    if win_col is not None:
        return TeacherLabel(move=win_col, source="teacher_tactical_win", score=1.0)

    block_col = _find_winning_move(board, -current_player)
    if block_col is not None:
        return TeacherLabel(move=block_col, source="teacher_tactical_block", score=1.0)

    per_move = max(4, int(rollouts / max(1, len(legal))))
    best_col = legal[0]
    best_score = -1e9

    for col in legal:
        nb, _ = _apply_move(board, col, current_player)
        if nb is None:
            continue
        if _check_winner(nb, current_player):
            return TeacherLabel(move=col, source="teacher_tactical_win", score=1.0)

        total = 0.0
        for _ in range(per_move):
            total += _rollout_value(nb, -current_player, current_player, rng)
        avg = total / float(per_move)
        if avg > best_score:
            best_score = avg
            best_col = col

    return TeacherLabel(move=best_col, source="teacher_mcts", score=float(best_score))


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from Connect4 logs.")
    parser.add_argument("--log", required=True, help="Path to connect4_game_logs.jsonl")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--win-weight", type=float, default=1.5, help="Sample weight for moves in won games")
    parser.add_argument("--draw-weight", type=float, default=1.0, help="Sample weight for moves in drawn games")
    parser.add_argument("--loss-weight", type=float, default=0.6, help="Sample weight for moves in lost games")
    parser.add_argument(
        "--focus-losses",
        action="store_true",
        help="Keep only ai_move events from games where result is loss (AI perspective).",
    )
    parser.add_argument(
        "--require-model-source",
        action="store_true",
        help="Keep only ai_move events with decision_source='model'.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["none", "mcts"],
        default="none",
        help="Optional corrective label generation mode.",
    )
    parser.add_argument(
        "--teacher-on",
        choices=["loss", "all"],
        default="loss",
        help="Apply teacher relabeling only on loss samples (default) or all samples.",
    )
    parser.add_argument(
        "--teacher-rollouts",
        type=int,
        default=140,
        help="Total rollout budget per state for MCTS teacher.",
    )
    parser.add_argument(
        "--teacher-max-samples",
        type=int,
        default=0,
        help="Cap number of samples to relabel with teacher (0 means unlimited).",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=1.7,
        help="Override sample weight for teacher-relabelled samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for teacher rollout policy.",
    )
    parser.add_argument(
        "--keep-model",
        default="",
        help="Optional model filter (cnn/transformer). Empty means keep all.",
    )
    args = parser.parse_args()

    events = _load_events(args.log)
    if not events:
        raise SystemExit("No events found in log file.")

    rng = np.random.default_rng(args.seed)

    end_info_by_game = {}
    for e in events:
        if e.get("event_type") == "game_end" and e.get("game_id"):
            end_info_by_game[str(e["game_id"])] = {
                "result": str(e.get("result", "unknown")).lower(),
                "winner_model": str(e.get("winner_model", "")).lower() if e.get("winner_model") else "",
            }

    X, y, w = [], [], []
    meta_game_id, meta_model, meta_source, meta_result, meta_turn = [], [], [], [], []
    meta_original_col, meta_label_source = [], []
    teacher_relabels = 0
    teacher_applied = 0

    for e in events:
        if e.get("event_type") != "ai_move":
            continue
        model_type = str(e.get("model_type", "cnn")).lower()
        actor_model = str(e.get("actor_model", model_type)).lower()
        if args.keep_model and actor_model != args.keep_model.lower():
            continue
        decision_source = str(e.get("decision_source", "")).lower()
        if args.require_model_source and decision_source != "model":
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

        if args.focus_losses and result != "loss":
            continue

        label_col = int(chosen_col)
        label_source = "logged"
        should_teacher = args.teacher_mode != "none"
        if should_teacher and args.teacher_on == "loss" and result != "loss":
            should_teacher = False
        if should_teacher and args.teacher_max_samples > 0 and teacher_applied >= args.teacher_max_samples:
            should_teacher = False

        if should_teacher and args.teacher_mode == "mcts":
            teacher = _teacher_move_mcts(board, int(current_player), args.teacher_rollouts, rng)
            if teacher is not None:
                teacher_applied += 1
                label_col = int(teacher.move)
                label_source = teacher.source
                if label_col != int(chosen_col):
                    teacher_relabels += 1
                sw = max(float(sw), float(args.teacher_weight))

        X.append(_encode_from_grid(board, current_player))
        y.append(label_col)
        w.append(float(sw))
        meta_game_id.append(game_id)
        meta_model.append(actor_model)
        meta_source.append(decision_source)
        meta_result.append(result)
        meta_turn.append(int(e.get("turn_index", -1)))
        meta_original_col.append(int(chosen_col))
        meta_label_source.append(label_source)

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
        original_chosen_col=np.array(meta_original_col, dtype=np.int64),
        label_source=np.array(meta_label_source, dtype=object),
    )

    result_counts = Counter(meta_result)
    print(f"Saved: {args.out}")
    print(f"Samples: {len(X_arr)}")
    print(f"Result mix: {dict(result_counts)}")
    print(f"Move distribution: {np.bincount(y_arr, minlength=7).tolist()}")
    if args.teacher_mode != "none":
        print(f"Teacher applied: {teacher_applied}")
        print(f"Teacher relabeled: {teacher_relabels}")
        print(f"Label sources: {dict(Counter(meta_label_source))}")


if __name__ == "__main__":
    main()
