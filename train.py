"""Command-line training:  python train.py --genre classical --epochs 30"""
import argparse

import music_processor as mp


def main():
    p = argparse.ArgumentParser(description="Train the MelodyAI LSTM on a MIDI dataset.")
    p.add_argument("--genre", help="dataset sub-folder name (e.g. classical, jazz)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--max-files", type=int, default=100)
    a = p.parse_args()

    mp.ensure_directories()
    genres = mp.available_genres()
    if not genres:
        print(mp.DATASET_HELP)
        return
    if not a.genre:
        print("Available genres:", ", ".join(f"{g} ({n} files)" for g, n in genres.items()))
        print("Run again with --genre <name>")
        return

    import model  # imported late: TensorFlow is slow to load
    result = model.train_model(
        a.genre, a.epochs, a.batch_size, a.seq_len, a.learning_rate, a.max_files,
        progress_cb=lambda **kw: print(kw["message"]) if "message" in kw else None)
    print("Training complete:", result)


if __name__ == "__main__":
    main()
