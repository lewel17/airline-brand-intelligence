import pandas as pd

df = pd.read_csv("recensioni_ryanair_easyjet.csv")

# Vediamo una recensione intera di esempio
print("--- ESEMPIO DI RECENSIONE ---")
print("Compagnia:", df.iloc[0]['AirlineName'])
print("Titolo:", df.iloc[0]['Title'])
print("Voto:", df.iloc[0]['OverallScore'])
print("Testo:", df.iloc[0]['Review'][:400])

# Controlliamo i campi vuoti nelle colonne che ci interessano
print("\n--- CAMPI VUOTI (NaN) ---")
colonne_utili = ['Review', 'OverallScore', 'AirlineName', 'Recommended', 'DateFlown']
print(df[colonne_utili].isnull().sum())

# Quanto sono lunghe le recensioni?
df['lunghezza'] = df['Review'].astype(str).str.len()
print("\n--- LUNGHEZZA RECENSIONI (in caratteri) ---")
print("Minima:", df['lunghezza'].min())
print("Massima:", df['lunghezza'].max())
print("Media:", int(df['lunghezza'].mean()))

# Distribuzione dei voti
print("\n--- DISTRIBUZIONE VOTI (OverallScore) ---")
print(df['OverallScore'].value_counts().sort_index())