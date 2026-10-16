"""Step 1 - Estrae le recensioni di Ryanair ed EasyJet dal dataset completo."""
import pandas as pd
from paths import RAW, PROCESSED

df = pd.read_csv(RAW / "AirlineReviews.csv")
print("Recensioni totali nel dataset:", len(df))

# Teniamo solo Ryanair ed EasyJet
maschera = df['AirlineName'].str.contains('ryanair|easyjet', case=False, na=False)
df_due = df[maschera].copy()

# Uniformiamo i nomi (easyJet -> EasyJet)
df_due['AirlineName'] = df_due['AirlineName'].str.replace('easyJet', 'EasyJet', case=False)

print("Recensioni Ryanair + EasyJet:", len(df_due))
print(df_due['AirlineName'].value_counts())

df_due.to_csv(PROCESSED / "recensioni_ryanair_easyjet.csv", index=False)
print("Salvato: data/processed/recensioni_ryanair_easyjet.csv")
