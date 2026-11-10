# MelodyAI – AI Music Generation Using LSTM Neural Networks

A beginner-friendly web app that **learns from MIDI files** with an LSTM (TensorFlow/Keras)
and **composes new music** note by note. Nothing is pre-recorded or faked: every generated
file comes from the trained network sampling the next note.

## Project structure
```
MelodyAI/
├── app.py               Flask server (routes, validation, status)
├── music_processor.py   MIDI reading (music21), sequences, MIDI writing
├── model.py             LSTM model, training, generation
├── train.py             Optional command-line training script
├── requirements.txt
├── dataset/             <- put your MIDI files here (dataset/classical, dataset/jazz)
├── models/              saved trained models (created by training)
├── output/              generated MIDI files
├── templates/index.html
└── static/css/style.css, static/js/script.js
```
**Create/open files in this order:** `requirements.txt` → `music_processor.py` → `model.py` →
`train.py` → `app.py` → `templates/index.html` → `static/css/style.css` → `static/js/script.js`.
(If you downloaded the zip, everything already exists – just unzip.)

## Windows step-by-step

### 1. Install Python and VS Code
* Python **3.10, 3.11 or 3.12** (64-bit): https://www.python.org/downloads/ – tick **"Add python.exe to PATH"**.
  (TensorFlow does not yet support the very newest Python versions – use 3.12 if unsure.)
* VS Code: https://code.visualstudio.com/ and install its *Python* extension.

### 2. Open the project and create a virtual environment
In VS Code: **File → Open Folder → MelodyAI**, then **Terminal → New Terminal**:
```powershell
python -m venv venv
venv\Scripts\activate
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
(or use Command Prompt and run `venv\Scripts\activate.bat`).

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This downloads TensorFlow (large) – be patient.

### 4. Add MIDI files (legal datasets)
Create genre folders and copy `.mid` files in:
```
dataset\classical\*.mid
dataset\jazz\*.mid
```
Any folder name (letters, digits, `_`, `-`) becomes a genre. **You must verify each licence yourself.**
Options commonly used for learning:
* **Your own MIDI files** or MIDI you composed/exported – always safe.
* **piano-midi.de** (Bernd Krueger) – classical piano; published under a Creative Commons licence (check the site for the current terms and attribution requirements).
* **MAESTRO dataset** (Google Magenta) – piano performances, CC BY-NC-SA 4.0 (non-commercial; large download, use a subset).
* **Public-domain compositions** from MIDI archives such as Mutopia Project / IMSLP-derived files (check each file's licence).
* For **jazz**, look for datasets that state a clear licence (for example, jazz solo transcription datasets from research projects) or use your own jazz MIDI files. Do **not** train on copyrighted commercial MIDI collections you have no rights to.

Tips: 20–100 files of one style work best; keep each genre in its own folder.
The app only offers a genre if it has MIDI files **and** a trained model, so it never claims support for a style it was not trained on.

### 5. Train the LSTM
Option A – in the browser (step 7), or Option B – command line:
```powershell
python train.py                       # lists genres found
python train.py --genre classical --epochs 30 --batch-size 64 --seq-len 32 --learning-rate 0.001
```
Loss is printed after each epoch. The model is saved to `models\classical_model.keras`
and reused automatically – no retraining needed. Quick test: `--epochs 5 --max-files 10`.
Good-sounding results usually need many files and 50+ epochs; a GPU is not required but CPU training is slow.

### 6. Start the Flask app
```powershell
python app.py
```

### 7. Open the website
Go to **http://127.0.0.1:5000**. If the dataset folder is empty, the page shows instructions.

### 8. Generate, listen, download
1. (Optional) Train from the **Train the model** card; watch progress and the loss chart.
2. Pick a trained genre, set **length**, **creativity** (temperature: low = safe/repetitive, high = surprising) and an optional **seed** (same seed + settings → same music).
3. Click **Generate Music**. The track loads in the player – press **Play/Pause**.
4. Click **Download MIDI**. Older files are in the history table (**Load** / **Download**).

Browser playback uses the Tone.js and @tonejs/midi libraries loaded from a CDN (internet needed
for the player only). If they can't load, the page says so and you can still download the MIDI
and open it with Windows Media Player, VLC (with a soundfont), MuseScore, etc.

## How to test each feature
| Feature | How to test |
|---|---|
| Dataset handling | Start with empty `dataset/` → warning appears. Add files → click Refresh. Put a renamed `.txt` file as `bad.mid` → it is skipped, training continues. |
| Training | `python train.py --genre classical --epochs 3 --max-files 5` → loss printed, `models/` gets `.keras` + `_meta.json`. |
| Status | While training from the web page, the progress bar and loss chart update every second. |
| Generation | Generate twice with the same seed → identical notes; different seeds → different music. |
| MIDI export | New file appears in `output/` and opens in any MIDI player. |
| Download security | Visit `http://127.0.0.1:5000/download/..%5Capp.py` → "Invalid filename". |
| Validation | POST bad values (e.g. length 99999) → clear error message toast. |

## API
`GET /` · `GET /genres` · `POST /train` · `POST /generate` · `GET /status` · `GET /history` · `GET /download/<filename>`

Example: `{"genre":"classical","length":120,"temperature":1.0,"seed":42}` to `/generate`.

## Troubleshooting
* **`pip install tensorflow` fails / "no matching distribution"** – you probably have an unsupported Python (e.g. 3.13) or 32-bit Python. Install Python 3.12 64-bit and recreate the venv.
* **`DLL load failed` when importing TensorFlow** – install the *Microsoft Visual C++ Redistributable 2015-2022 (x64)*.
* **`ModuleNotFoundError`** – the venv isn't active. Run `venv\Scripts\activate`, then `pip install -r requirements.txt`.
* **Training is very slow** – lower `--max-files`, `--epochs`, or raise `--batch-size`. Reading MIDI with music21 can take a few seconds per file.
* **"Dataset is too small"** – add more/longer MIDI files or lower sequence length.
* **Generated music sounds random** – train longer, use more files, lower creativity (0.6–0.9).
* **No sound in browser** – check internet (CDN libraries), click Play once (browsers require a click), unmute the tab. Otherwise download the MIDI.
* **Port 5000 already in use** – change `port=5000` in `app.py`.
* **Model/genre missing after copying the project** – copy both `models/<genre>_model.keras` and `models/<genre>_meta.json`.
* **TensorFlow can't load an old `.keras` file after upgrading** – retrain.

## Notes
* Free and offline: no API keys or paid services.
* Output quality depends on dataset size/quality; this is a learning project, not a studio-grade composer.
* Generation is by sampling the LSTM's probabilities with temperature, seeded by a real sequence from your dataset.
