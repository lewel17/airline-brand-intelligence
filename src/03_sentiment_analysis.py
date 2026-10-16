"""Step 3 - Sentiment analysis con distilBERT (modello pre-addestrato SST-2).

Nota: e' il modello usato per classificare l'intero corpus nella tesi.
Il finetuning (step 4-7) e' stato usato per validarne l'affidabilita'
sul dominio delle recensioni aeree.
"""
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from paths import PROCESSED

df = pd.read_csv(PROCESSED / "recensioni_pulite.csv")
print("Recensioni da analizzare:", len(df))

print("Carico il modello...")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analizza(testo):
    # Il modello accetta fino a 512 token: tagliamo i testi troppo lunghi
    testo = str(testo)[:1500]
    risultato = sentiment(testo)[0]
    return pd.Series([risultato['label'], risultato['score']])

tqdm.pandas()
df[['sentiment', 'sentiment_score']] = df['Review'].progress_apply(analizza)

df.to_csv(PROCESSED / "recensioni_con_sentiment.csv", index=False)
print("Salvato: data/processed/recensioni_con_sentiment.csv")

print("\n--- SENTIMENT PER COMPAGNIA ---")
print(df.groupby('AirlineName')['sentiment'].value_counts())
