"""
model.py
--------
LSTM model definition, training, saving/loading and note generation.
"""
import json
import os
import threading
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import music_processor as mp

_loaded = {}                     # genre -> (model, meta) cache
_predict_lock = threading.Lock()


def build_model(vocab_size, seq_len, learning_rate, units=256):
    """Embedding -> 2 stacked LSTMs -> Dense -> softmax over the vocabulary."""
    model = keras.Sequential([
        layers.Input(shape=(seq_len,), dtype="int32"),
        layers.Embedding(vocab_size, 64),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(units),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


class ProgressCallback(keras.callbacks.Callback):
    """Sends loss values after each epoch to the web app / console."""

    def __init__(self, total_epochs, progress_cb):
        super().__init__()
        self.total, self.cb = total_epochs, progress_cb
        self.loss, self.val_loss = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(round(float(logs.get("loss", 0)), 4))
        if "val_loss" in logs:
            self.val_loss.append(round(float(logs["val_loss"]), 4))
        if self.cb:
            self.cb(
                epoch=epoch + 1, total_epochs=self.total,
                progress=int(15 + 85 * (epoch + 1) / self.total),
                loss_history=list(self.loss), val_loss_history=list(self.val_loss),
                message=f"Epoch {epoch + 1}/{self.total} - loss {self.loss[-1]}",
            )


def train_model(genre, epochs=30, batch_size=64, seq_len=32,
                learning_rate=0.001, max_files=100, progress_cb=None):
    """Train an LSTM on the genre dataset and save it to models/."""
    def report(**kw):
        if progress_cb:
            progress_cb(**kw)

    mp.ensure_directories()
    mp.validate_genre(genre, must_have_dataset=True)

    report(message="Reading MIDI files with music21 (this can take a while)...", progress=1,
           epoch=0, total_epochs=epochs, loss_history=[], val_loss_history=[])

    def file_progress(i, n, name):
        report(message=f"Reading MIDI {i}/{n}: {name}", progress=int(1 + 12 * i / n))

    songs, used, skipped = mp.load_dataset_tokens(genre, max_files, file_progress)
    vocab, stats = mp.build_vocab(songs)
    X, y = mp.make_sequences(songs, vocab, seq_len)
    stats.head(200).to_csv(mp.MODELS_DIR / f"{genre}_token_stats.csv", index=False)

    report(message=f"Building LSTM: {len(X)} training sequences, {len(vocab)} unique notes", progress=14)
    model = build_model(len(vocab), seq_len, learning_rate)
    cb = ProgressCallback(epochs, progress_cb)
    val_split = 0.1 if len(X) >= 500 else 0.0
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=val_split,
              shuffle=True, verbose=2, callbacks=[cb])

    # Save model + metadata (vocabulary and a pool of seed sequences)
    model.save(mp.model_path(genre))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(X), min(100, len(X)), replace=False)
    meta = {
        "genre": genre, "seq_len": seq_len, "vocab": vocab,
        "seeds": X[pick].tolist(), "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "files_used": used, "files_skipped": len(skipped),
        "num_sequences": int(len(X)), "final_loss": cb.loss[-1] if cb.loss else None,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp.meta_path(genre).write_text(json.dumps(meta), encoding="utf-8")
    _loaded.pop(genre, None)
    return {k: v for k, v in meta.items() if k not in ("vocab", "seeds")}


def _load(genre):
    if genre not in _loaded:
        mp.validate_genre(genre, must_be_trained=True)
        model = keras.models.load_model(mp.model_path(genre))
        meta = json.loads(mp.meta_path(genre).read_text(encoding="utf-8"))
        _loaded[genre] = (model, meta)
    return _loaded[genre]


def _sample(probs, temperature, rng):
    """Pick the next note index; higher temperature = more surprising notes."""
    probs = np.asarray(probs, dtype="float64")
    logits = np.log(np.maximum(probs, 1e-9)) / temperature
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_tokens(genre, length=100, temperature=1.0, seed=None, progress_cb=None):
    """Generate `length` new tokens with the trained LSTM."""
    model, meta = _load(genre)
    vocab, seq_len = meta["vocab"], meta["seq_len"]
    rng = np.random.default_rng(seed)
    pattern = list(meta["seeds"][int(rng.integers(len(meta["seeds"])))])
    out = []
    with _predict_lock:
        for i in range(length):
            x = np.array([pattern], dtype="int32")
            probs = model(x, training=False).numpy()[0]
            idx = _sample(probs, temperature, rng)
            out.append(vocab[idx])
            pattern = pattern[1:] + [idx]
            if progress_cb and (i % 5 == 0 or i == length - 1):
                progress_cb(progress=int(100 * (i + 1) / length),
                            message=f"Generating note {i + 1}/{length}")
    return out
