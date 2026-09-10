# SETUP.md

## Quick Start

```bash
git clone <repo-url>
cd brahmanda
python -m venv .venv
```

**Mac/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Known Issues and Fixes (learned the hard way - read before you hit them yourself)

### 1. "Cannot copy out of meta tensor" (Windows, sometimes Mac)

**Symptom:** embedding step fails with `NotImplementedError: Cannot copy out of meta tensor; no data!`

**Cause:** a known compatibility bug between PyTorch 2.2+ and older `sentence-transformers` versions when loading BERT-based models.

**Fix:** already handled by the `sentence-transformers>=6.0.0` pin in `requirements.txt`. If you still hit this on a fresh install, run:
```bash
pip install --upgrade sentence-transformers transformers
```
Downgrading PyTorch instead is *not* a reliable fix on newer Python versions - PyPI may not even offer pre-2.2 wheels for your Python version.

### 2. `[Errno 22] Invalid argument` during embedding (Windows specifically)

**Symptom:** embedding fails with a generic `OSError: [Errno 22] Invalid argument`, no further detail.

**Cause:** Windows path length/character limits, usually triggered when the repo sits inside a deeply nested folder (a synced cloud-storage path like OneDrive is a common culprit - "OneDrive - Org Name\OneDrive\project-name\..." adds up fast).

**Fix:** point the HuggingFace model cache somewhere short, outside the deep path:

**Windows (PowerShell):**
```powershell
$env:HF_HOME = "C:\hf_cache"
$env:SENTENCE_TRANSFORMERS_HOME = "C:\hf_cache"
```

**Mac/Linux** (rarely needed, but same idea if you hit a similar path issue):
```bash
export HF_HOME=~/hf_cache
```

### 3. `uvicorn: command not found`

**Cause:** your virtual environment isn't activated, or `uvicorn`/`fastapi` were never installed in it.

**Fix:** confirm your prompt shows the environment name (e.g. `(.venv)` or your env's name) before running any command. If it's active and still missing:
```bash
pip install uvicorn fastapi
```

### 4. Multi-line commands breaking in the terminal (any platform)

**Symptom:** a pasted `for` loop executes each line separately instead of as one block, or a `#!/bin/bash` shebang line throws `event not found` in zsh.

**Cause:** terminal paste-mode handling multi-line input inconsistently.

**Fix:** prefer single-line commands with `;` separators over multi-line pastes when running loops directly at the prompt. If using a saved script file, always invoke it explicitly (`bash script.sh` or `python script.py`) rather than pasting its contents into the terminal.

### 5. Path errors like `scripts/scripts/file.py` or files "not found" that definitely exist

**Cause:** running a command from inside the wrong directory (e.g. from `scripts/` instead of the repo root), so relative paths double up or fail.

**Fix:** always confirm your current directory before running anything:
```bash
pwd          # Mac/Linux
```
```powershell
Get-Location # Windows
```
All commands in this project assume you're running from the repo root, not from inside `scripts/`.

## Running the Dashboard

```bash
# Mac/Linux
BASE_PATH="." OVX_PATH="crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
  CORPUS_PATH=scorer_corpus.jsonl PYTHONPATH=scripts \
  uvicorn scripts.redflag_api_v5:app --port 8000 --workers 1
```

```powershell
# Windows (PowerShell)
$env:BASE_PATH = "."
$env:OVX_PATH = "crude_oil_data\oil_price_data\CBOE Crude Oil Volatility Historical Data.csv"
$env:CORPUS_PATH = "scorer_corpus.jsonl"
$env:PYTHONPATH = "scripts"
uvicorn scripts.redflag_api_v5:app --port 8000 --workers 1
```

Then open `redflag_dashboard_v6.html` in a browser. The server must be running first; the dashboard is a static file that calls `http://localhost:8000`.

**Known limitation:** the walk-forward backtest panel has an unresolved file-naming bug (`v3` vs `v4` results files) - expect it to error if you try it. Everything else on the dashboard is functional.
