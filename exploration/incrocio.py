import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

df = pd.read_csv("recensioni_con_sentiment.csv")

# Funzione che fa il topic modeling e attacca i temi alle recensioni
def analizza_temi(dati, min_cluster):
    testi = dati['Review'].astype(str).tolist()
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    cluster_model = HDBSCAN(min_cluster_size=min_cluster, min_samples=5, prediction_data=True)
    modello = BERTopic(language="english", hdbscan_model=cluster_model,
                       vectorizer_model=vectorizer, verbose=False)
    temi, _ = modello.fit_transform(testi)
    dati = dati.copy()
    dati['topic'] = temi
    return dati, modello

# Analizziamo le due compagnie
for compagnia, min_c in [("Ryanair", 15), ("EasyJet", 10)]:
    print(f"\n{'='*50}")
    print(f"  {compagnia.upper()}")
    print('='*50)

    sub = df[df['AirlineName'] == compagnia]
    sub, modello = analizza_temi(sub, min_c)

    # Per ogni tema, calcoliamo quante recensioni e la % di negative
    info = modello.get_topic_info()
    for _, riga in info.iterrows():
        topic_id = riga['Topic']
        if topic_id == -1:
            continue  # saltiamo il gruppo "non assegnato"

        recensioni_tema = sub[sub['topic'] == topic_id]
        totale = len(recensioni_tema)
        if totale < 15:
            continue  # saltiamo i temi troppo piccoli

        neg = len(recensioni_tema[recensioni_tema['sentiment'] == 'NEGATIVE'])
        perc_neg = round(neg / totale * 100)

        # Parole chiave del tema
        parole = modello.get_topic(topic_id)
        parole_top = ", ".join([p[0] for p in parole[:5]])

        print(f"\nTema {topic_id} ({totale} recensioni) - {perc_neg}% negative")
        print(f"  Parole: {parole_top}")