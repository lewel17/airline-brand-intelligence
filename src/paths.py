"""Percorsi condivisi da tutti gli script della pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"              # dataset originale (non versionato)
PROCESSED = ROOT / "data" / "processed"  # file intermedi (non versionati)
RESULTS = ROOT / "data" / "results"      # risultati aggregati (versionati)
MODELS = ROOT / "models"                 # modello finetunato (non versionato)
ASSETS = ROOT / "assets"

for cartella in (PROCESSED, RESULTS, MODELS, ASSETS):
    cartella.mkdir(parents=True, exist_ok=True)
