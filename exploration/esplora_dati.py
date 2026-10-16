import pandas as pd

df = pd.read_csv("AirlineReviews.csv")

# Cerchiamo come sono scritti i nomi delle due compagnie
# Prima vediamo tutti i nomi che contengono "ryan" o "easy"
nomi = df['AirlineName'].unique()

print("Nomi che contengono 'ryan':")
for nome in nomi:
    if 'ryan' in str(nome).lower():
        print("  -", nome)

print("\nNomi che contengono 'easy':")
for nome in nomi:
    if 'easy' in str(nome).lower():
        print("  -", nome)

# Contiamo quante recensioni per Ryanair e EasyJet
print("\n--- CONTEGGIO ---")
ryanair = df[df['AirlineName'].str.contains('ryanair', case=False, na=False)]
easyjet = df[df['AirlineName'].str.contains('easyjet', case=False, na=False)]
print("Recensioni Ryanair:", len(ryanair))
print("Recensioni EasyJet:", len(easyjet))