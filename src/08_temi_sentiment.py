"""Step 8 - Topic modeling (BERTopic + HDBSCAN) per compagnia, incrociato con il sentiment.

min_cluster_size diverso per compagnia: EasyJet ha meno recensioni (1.406 vs 2.144),
quindi serve una soglia piu' bassa per far emergere temi di dimensione comparabile.

UMAP ha un seed fisso (random_state=42) per rendere i temi riproducibili.
Attenzione: i risultati della tesi vengono da una run senza seed, quindi
rilanciando lo script i numeri dei singoli temi possono differire di qualche punto.
"""
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from paths import PROCESSED, RESULTS

df = pd.read_csv(PROCESSED / "recensioni_con_sentiment.csv")

def analizza(dati, min_cluster):
    testi = dati['Review'].astype(str).tolist()
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric='cosine', random_state=42)
    cluster_model = HDBSCAN(min_cluster_size=min_cluster, min_samples=5, prediction_data=True)
    modello = BERTopic(language="english", umap_model=umap_model, hdbscan_model=cluster_model,
                       vectorizer_model=CountVectorizer(stop_words="english", min_df=2),
                       verbose=False)
    temi, _ = modello.fit_transform(testi)
    dati = dati.copy()
    dati['topic'] = temi
    return dati, modello

righe = []
for compagnia, min_c in [("Ryanair", 15), ("EasyJet", 10)]:
    sub, modello = analizza(df[df['AirlineName'] == compagnia], min_c)
    for _, r in modello.get_topic_info().iterrows():
        tid = r['Topic']
        if tid == -1:          # recensioni non assegnate a nessun tema
            continue
        rec = sub[sub['topic'] == tid]
        if len(rec) < 15:      # temi troppo piccoli
            continue
        neg = (rec['sentiment'] == 'NEGATIVE').sum()
        righe.append({
            "Compagnia": compagnia,
            "Tema_id": tid,
            "Recensioni": len(rec),
            "Perc_negative": round(neg / len(rec) * 100),
            "Parole_chiave": ", ".join(p[0] for p in modello.get_topic(tid)[:5]),
        })

risultati = pd.DataFrame(righe)
risultati.to_csv(RESULTS / "risultati_temi_sentiment.csv", index=False)
print("Salvato: data/results/risultati_temi_sentiment.csv")
print(risultati.to_string(index=False))
