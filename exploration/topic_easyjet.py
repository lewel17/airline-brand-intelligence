import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

df = pd.read_csv("recensioni_con_sentiment.csv")

# QUI cambia la compagnia: EasyJet invece di Ryanair
easyjet = df[df['AirlineName'] == 'EasyJet'].copy()
testi = easyjet['Review'].astype(str).tolist()
print("Recensioni EasyJet da analizzare:", len(testi))

vectorizer = CountVectorizer(stop_words="english", min_df=2)

print("Analisi dei temi in corso...")

cluster_model = HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)

modello = BERTopic(
    language="english",
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer,
    verbose=True
)
temi, probabilita = modello.fit_transform(testi)

print("\n--- TEMI TROVATI IN EASYJET ---")
info = modello.get_topic_info()
print(info[['Topic', 'Count', 'Name']].to_string(index=False))

print("\n--- PAROLE CHIAVE DEI PRIMI TEMI ---")
for topic_num in range(0, min(6, len(info) - 1)):
    parole = modello.get_topic(topic_num)
    parole_top = [p[0] for p in parole[:8]]
    print(f"Tema {topic_num}: {', '.join(parole_top)}")