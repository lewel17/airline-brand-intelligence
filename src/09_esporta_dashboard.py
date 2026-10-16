"""Step 9 - Esporta i dati minimi per la dashboard (senza il testo delle recensioni)."""
import pandas as pd
from paths import PROCESSED, ROOT

df = pd.read_csv(PROCESSED / "recensioni_con_sentiment.csv")
df[['AirlineName', 'OverallScore', 'Recommended', 'sentiment']].to_csv(
    ROOT / "data" / "dashboard_data.csv", index=False)
print("Salvato: data/dashboard_data.csv")
