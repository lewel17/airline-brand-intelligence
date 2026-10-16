import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("recensioni_con_sentiment.csv")

ryanair = df[df['AirlineName'] == 'Ryanair'].copy()
testi = ryanair['Review'].astype(str).tolist()
print("Recensioni Ryanair da analizzare:", len(testi))

# Questo dice a BERTopic di ignorare le stop words inglesi
# e di togliere le parole troppo rare
vectorizer = CountVectorizer(stop_words="english", min_df=2)

print("Analisi dei temi in corso...")
modello = BERTopic(
    language="english",
    min_topic_size=15,
    nr_topics=10,
    vectorizer_model=vectorizer,
    verbose=True
)
temi, probabilita = modello.fit_transform(testi)

print("\n--- TEMI TROVATI IN RYANAIR ---")
info = modello.get_topic_info()
print(info[['Topic', 'Count', 'Name']].to_string(index=False))

# Per i primi temi, mostriamo le parole chiave principali
print("\n--- PAROLE CHIAVE DEI PRIMI TEMI ---")
for topic_num in range(0, min(6, len(info) - 1)):
    parole = modello.get_topic(topic_num)
    parole_top = [p[0] for p in parole[:8]]
    print(f"Tema {topic_num}: {', '.join(parole_top)}")