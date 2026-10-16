"""Step 4 - Crea il dataset etichettato per il finetuning e lo divide 70/15/15.

Etichette dal voto Skytrax: 1-4 = negativo (0), 7-10 = positivo (1).
I voti 5-6 vengono scartati perche' ambigui.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from paths import PROCESSED

df = pd.read_csv(PROCESSED / "recensioni_pulite.csv")
print("Recensioni totali:", len(df))

df = df.dropna(subset=['OverallScore'])
print("Recensioni con voto:", len(df))

def assegna_etichetta(voto):
    if voto <= 4:
        return 0   # negativo
    elif voto >= 7:
        return 1   # positivo
    return None    # 5-6 scartati

df['label'] = df['OverallScore'].apply(assegna_etichetta)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("Recensioni dopo aver scartato i voti 5-6:", len(df))
print(df['label'].value_counts())

df[['Review', 'label']].to_csv(PROCESSED / "dati_finetuning.csv", index=False)

# 70% training, 15% validation, 15% test (stratificati)
train, resto = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
validation, test = train_test_split(resto, test_size=0.50, random_state=42, stratify=resto['label'])

print("Training:", len(train), "| Validation:", len(validation), "| Test:", len(test))

train.to_csv(PROCESSED / "ft_train.csv", index=False)
validation.to_csv(PROCESSED / "ft_validation.csv", index=False)
test.to_csv(PROCESSED / "ft_test.csv", index=False)
print("Salvati: ft_train.csv, ft_validation.csv, ft_test.csv")
