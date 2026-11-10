"""
app.py - Flask backend for MelodyAI.

Endpoints:
  GET  /                    homepage
  GET  /genres              datasets found + models trained
  POST /train               start training in a background thread
  POST /generate            generate a MIDI file with the trained LSTM
  GET  /status              current training / generation status
  GET  /history             list of generated files
  GET  /download/<file>     download a generated MIDI file
"""
import random
import re
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory

import music_processor as mp

app = Flask(__name__)
mp.ensure_directories()

SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_\-]+\.mid$")   # blocks ../ and odd names

_lock = threading.Lock()
status = {
    "state": "idle",            # idle | training | generating | error
    "message": "Idle", "progress": 0, "epoch": 0, "total_epochs": 0,
    "loss_history": [], "val_loss_history": [], "error": None,
}


def set_status(**kwargs):
    with _lock:
        status.update(kwargs)


def try_begin(state, message):
    """Atomically claim the app for one task. Returns False if it is busy."""
    with _lock:
        if status["state"] in ("training", "generating"):
            return False
        status.update(state=state, message=message, progress=0, epoch=0, total_epochs=0,
                      loss_history=[], val_loss_history=[], error=None)
        return True


def get_lstm():
    """Import TensorFlow code lazily with a friendly error if it is missing."""
    try:
        import model
        return model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: pip install -r requirements.txt "
            f"(details: {exc})")


# ---- Input validation -------------------------------------------------------
def read_number(data, name, lo, hi, default=None, integer=True):
    value = data.get(name, default)
    if value is None or value == "":
        raise ValueError(f"'{name}' is required.")
    try:
        value = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")
    if not lo <= value <= hi:
        raise ValueError(f"'{name}' must be between {lo} and {hi}.")
    return value


def error(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


# ---- Routes -----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/genres")
def genres():
    datasets = mp.available_genres()
    trained = mp.trained_genres()
    return jsonify({
        "datasets": datasets, "trained": trained,
        "dataset_empty": not datasets,
        "instructions": mp.DATASET_HELP if not datasets else "",
    })


@app.route("/status")
def get_status():
    with _lock:
        return jsonify(dict(status))


def _train_worker(genre, cfg):
    try:
        lstm = get_lstm()
        result = lstm.train_model(genre, progress_cb=set_status, **cfg)
        set_status(state="idle", progress=100, error=None,
                   message=f"Training finished for '{genre}'. Final loss: {result['final_loss']}")
    except Exception as exc:
        set_status(state="error", error=str(exc), message="Training failed.")


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    try:
        genre = mp.validate_genre(data.get("genre"), must_have_dataset=True)
        cfg = {
            "epochs": read_number(data, "epochs", 1, 500, 30),
            "batch_size": read_number(data, "batch_size", 8, 512, 64),
            "seq_len": read_number(data, "seq_len", 8, 128, 32),
            "learning_rate": read_number(data, "learning_rate", 1e-5, 0.1, 0.001, integer=False),
        }
    except ValueError as exc:
        return error(str(exc))
    if not try_begin("training", f"Starting training for '{genre}'..."):
        return error("The app is busy. Wait for the current task to finish.", 409)
    threading.Thread(target=_train_worker, args=(genre, cfg), daemon=True).start()
    return jsonify({"ok": True, "message": f"Training started for '{genre}'."}), 202


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    try:
        genre = mp.validate_genre(data.get("genre"), must_be_trained=True)
        length = read_number(data, "length", 16, 500, 100)
        temperature = read_number(data, "temperature", 0.1, 2.0, 1.0, integer=False)
        seed_raw = data.get("seed")
        seed = (read_number({"seed": seed_raw}, "seed", 0, 2**31 - 1)
                if seed_raw not in (None, "") else random.randint(0, 2**31 - 1))
    except ValueError as exc:
        return error(str(exc))
    if not try_begin("generating", "Generating music..."):
        return error("The app is busy. Wait for the current task to finish.", 409)
    try:
        lstm = get_lstm()
        tokens = lstm.generate_tokens(genre, length, temperature, seed, progress_cb=set_status)
        stamp = datetime.now()
        filename = f"{genre}_{stamp:%Y%m%d_%H%M%S}_seed{seed}.mid"
        mp.tokens_to_midi(tokens, mp.OUTPUT_DIR / filename)
        set_status(state="idle", progress=100, message=f"Saved {filename}")
        return jsonify({"ok": True, "filename": filename, "seed": seed,
                        "notes": len(tokens), "created": f"{stamp:%Y-%m-%d %H:%M:%S}"})
    except Exception as exc:
        set_status(state="error", error=str(exc), message="Generation failed.")
        return error(f"Generation failed: {exc}", 500)


@app.route("/history")
def history():
    files = sorted(mp.OUTPUT_DIR.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([{
        "filename": p.name,
        "created": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_kb": round(p.stat().st_size / 1024, 1),
    } for p in files[:50] if SAFE_FILENAME.match(p.name)])


@app.route("/download/<filename>")
def download(filename):
    if not SAFE_FILENAME.match(filename):
        return error("Invalid filename.", 400)
    if not (mp.OUTPUT_DIR / filename).is_file():
        return error("File not found.", 404)
    # send_from_directory also refuses paths that escape OUTPUT_DIR
    return send_from_directory(mp.OUTPUT_DIR, filename, as_attachment=True,
                               mimetype="audio/midi")


if __name__ == "__main__":
    # threaded=True lets the browser poll /status while training runs
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
