"""Step 2 - Pulizia: rimuove recensioni vuote, troppo corte e duplicate."""
import pandas as pd
from paths import PROCESSED

df = pd.read_csv(PROCESSED / "recensioni_ryanair_easyjet.csv")
print("Partenza:", len(df), "recensioni")

# 1. Recensioni senza testo
df = df.dropna(subset=['Review'])
print("Dopo aver tolto le recensioni vuote:", len(df))

# 2. Recensioni troppo corte (meno di 20 caratteri)
df = df[df['Review'].astype(str).str.len() >= 20]
print("Dopo aver tolto le troppo corte:", len(df))

# 3. Doppioni identici
df = df.drop_duplicates(subset=['Review'])
print("Dopo aver tolto i doppioni:", len(df))

df.to_csv(PROCESSED / "recensioni_pulite.csv", index=False)
print("Salvato: data/processed/recensioni_pulite.csv")
print(df['AirlineName'].value_counts())
