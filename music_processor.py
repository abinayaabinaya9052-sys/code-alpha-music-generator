"""
music_processor.py
------------------
Everything related to MIDI: finding datasets, extracting notes with music21,
turning notes into integer sequences for the LSTM, and writing MIDI files.

A "token" is a short string describing one musical event:
    "60|1.0"        -> MIDI pitch 60 (middle C), lasting 1.0 quarter-notes
    "60.64.67|2.0"  -> a chord (C-E-G) lasting 2.0 quarter-notes
    "R|0.5"         -> a rest lasting 0.5 quarter-notes
"""
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Paths (always relative to this file, never hardcoded) -----------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MIDI_EXTENSIONS = {".mid", ".midi"}
ROOT_GENRE = "general"              # MIDI files placed directly in dataset/
GENRE_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

# ---- Encoding settings ------------------------------------------------------
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]   # allowed note lengths
MAX_CHORD_NOTES = 4          # keep only the highest N pitches of a chord
MAX_TOKENS_PER_FILE = 2500   # keeps training time reasonable
MIN_TOKEN_COUNT = 2          # ignore tokens seen fewer times than this
MAX_VOCAB = 800              # cap vocabulary size

DATASET_HELP = (
    "No MIDI files found. Create folders such as dataset/classical and "
    "dataset/jazz, copy .mid files into them, then click 'Refresh'. "
    "See README.md for legal dataset sources."
)


# ---- Directory / genre helpers ---------------------------------------------
def ensure_directories():
    for d in (DATASET_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_midi_files(genre):
    """Return all MIDI files belonging to a genre folder."""
    if genre == ROOT_GENRE:
        folder, files = DATASET_DIR, DATASET_DIR.glob("*")
    else:
        folder = DATASET_DIR / genre
        files = folder.rglob("*") if folder.is_dir() else []
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in MIDI_EXTENSIONS)


def available_genres():
    """{genre_name: number_of_midi_files} for genres that have a dataset."""
    genres = {}
    if DATASET_DIR.is_dir():
        for sub in sorted(DATASET_DIR.iterdir()):
            if sub.is_dir() and GENRE_RE.match(sub.name):
                n = len(list_midi_files(sub.name))
                if n:
                    genres[sub.name] = n
    n = len(list_midi_files(ROOT_GENRE))
    if n:
        genres[ROOT_GENRE] = n
    return genres


def model_path(genre):
    return MODELS_DIR / f"{genre}_model.keras"


def meta_path(genre):
    return MODELS_DIR / f"{genre}_meta.json"


def trained_genres():
    """{genre: metadata} for every genre that has a saved model."""
    result = {}
    if MODELS_DIR.is_dir():
        for mp in sorted(MODELS_DIR.glob("*_meta.json")):
            genre = mp.name[: -len("_meta.json")]
            if GENRE_RE.match(genre) and model_path(genre).exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                    meta.pop("seeds", None)
                    meta.pop("vocab", None)
                    result[genre] = meta
                except (OSError, ValueError):
                    continue
    return result


def validate_genre(genre, must_have_dataset=False, must_be_trained=False):
    """Raise ValueError with a friendly message if the genre is not usable."""
    if not isinstance(genre, str) or not GENRE_RE.match(genre):
        raise ValueError("Invalid genre name.")
    if must_have_dataset and genre not in available_genres():
        raise ValueError(f"No MIDI dataset found for genre '{genre}'. {DATASET_HELP}")
    if must_be_trained and genre not in trained_genres():
        raise ValueError(f"The model for '{genre}' has not been trained yet. Train it first.")
    return genre


# ---- MIDI -> tokens ---------------------------------------------------------
def _quantize(length):
    return min(DURATIONS, key=lambda d: abs(d - length))


def parse_midi_file(path):
    """Extract a list of tokens from one MIDI file (raises on corrupt files)."""
    from music21 import converter  # imported here so the web app starts quickly

    score = converter.parse(str(path))
    # chordify merges all parts into one stream of simultaneous notes (chords)
    flat = score.chordify().flatten().notesAndRests
    tokens = []
    for el in flat:
        length = float(el.duration.quarterLength)
        if length < 0.125:          # skip grace notes / zero-length events
            continue
        dur = _quantize(length)
        if el.isRest:
            tokens.append(f"R|{dur}")
        else:
            pitches = sorted({p.midi for p in el.pitches})[-MAX_CHORD_NOTES:]
            tokens.append(".".join(str(p) for p in pitches) + f"|{dur}")
        if len(tokens) >= MAX_TOKENS_PER_FILE:
            break
    return tokens


def load_dataset_tokens(genre, max_files=100, progress_cb=None):
    """
    Read up to `max_files` MIDI files of a genre. Invalid files are skipped.
    Returns (list_of_token_lists, number_of_files_used, list_of_skipped_names).
    """
    files = list_midi_files(genre)[:max_files]
    if not files:
        raise ValueError(DATASET_HELP)
    songs, skipped = [], []
    for i, path in enumerate(files, 1):
        try:
            tokens = parse_midi_file(path)
            if len(tokens) >= 20:
                songs.append(tokens)
            else:
                skipped.append(path.name)
        except Exception:                      # corrupted / unsupported file
            skipped.append(path.name)
        if progress_cb:
            progress_cb(i, len(files), path.name)
    if not songs:
        raise ValueError("None of the MIDI files could be read. Check that they are valid MIDI files.")
    return songs, len(songs), skipped


# ---- Tokens -> numbers ------------------------------------------------------
def build_vocab(songs):
    """Return vocabulary list (most common tokens first) and a stats DataFrame."""
    counts = Counter(t for s in songs for t in s)
    stats = pd.DataFrame(counts.most_common(), columns=["token", "count"])
    kept = stats[stats["count"] >= MIN_TOKEN_COUNT].head(MAX_VOCAB)
    vocab = kept["token"].tolist()
    if len(vocab) < 5:
        raise ValueError("Dataset is too small or too uniform. Add more MIDI files.")
    return vocab, stats


def make_sequences(songs, vocab, seq_len, max_samples=120000):
    """
    Sliding windows: X = `seq_len` previous notes, y = the next note.
    Windows never cross from one song into the next.
    """
    index = {t: i for i, t in enumerate(vocab)}
    X, y = [], []
    for song in songs:
        ids = [index[t] for t in song if t in index]
        for i in range(len(ids) - seq_len):
            X.append(ids[i:i + seq_len])
            y.append(ids[i + seq_len])
    if len(X) < 50:
        raise ValueError("Not enough note data to train. Add more/longer MIDI files or lower the sequence length.")
    X, y = np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)
    if len(X) > max_samples:                   # subsample very large datasets
        keep = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
        keep.sort()
        X, y = X[keep], y[keep]
    return X, y


# ---- Tokens -> MIDI ---------------------------------------------------------
def tokens_to_midi(tokens, out_path, bpm=100):
    """Write generated tokens to a valid MIDI file using music21."""
    from music21 import chord, instrument, note, stream, tempo

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    s.append(instrument.Piano())
    for tok in tokens:
        pitch_part, dur_part = tok.split("|")
        dur = float(dur_part)
        if pitch_part == "R":
            s.append(note.Rest(quarterLength=dur))
        elif "." in pitch_part:
            s.append(chord.Chord([int(p) for p in pitch_part.split(".")], quarterLength=dur))
        else:
            s.append(note.Note(int(pitch_part), quarterLength=dur))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s.write("midi", fp=str(out_path))
    return out_path
