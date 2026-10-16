from transformers import pipeline

# Carichiamo il modello di sentiment analysis pre-addestrato
# La prima volta lo scarica da internet, ci mette un minuto
print("Carico il modello... (la prima volta scarica, attendi)")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Proviamo su 5 frasi di esempio, due positive due negative una neutra
frasi = [
    "The flight was on time and the crew was lovely, great experience!",
    "Worst airline ever. Delayed 3 hours and lost my luggage.",
    "Cheap ticket but they charge for everything, hidden costs everywhere.",
    "Comfortable seats and smooth boarding, would fly again.",
    "The flight departed as scheduled."
]

print("\n--- RISULTATI ---")
for frase in frasi:
    risultato = sentiment(frase)[0]
    print(f"{risultato['label']} ({risultato['score']:.2f}) → {frase}")