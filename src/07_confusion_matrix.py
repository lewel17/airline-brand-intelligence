"""Step 7 - Confusion matrix del modello finetunato sul test set."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from paths import ASSETS, MODELS, PROCESSED

test = pd.read_csv(PROCESSED / "ft_test.csv")
testi = test['Review'].astype(str).tolist()
veri = test['label'].tolist()
print("Valutazione sul test set:", len(test), "recensioni\n")

cartella_ft = str(MODELS / "modello_finetuned")
# Il tokenizer non cambia con il finetuning: si usa quello del modello base
TOKENIZER = "distilbert-base-uncased-finetuned-sst-2-english"
ft = pipeline("sentiment-analysis", model=cartella_ft, tokenizer=TOKENIZER,
              truncation=True, max_length=256)
pred = [1 if ft(t)[0]['label'] == 'POSITIVE' else 0 for t in testi]

cm = confusion_matrix(veri, pred)
print("Confusion matrix:\n", cm)
print("\nReport dettagliato:")
print(classification_report(veri, pred, target_names=['Negativo', 'Positivo']))

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap='Blues')
classi = ['Negativo', 'Positivo']
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(classi); ax.set_yticklabels(classi)
ax.set_xlabel('Predetto dal modello')
ax.set_ylabel('Valore reale')
ax.set_title('Confusion Matrix - Modello finetunato')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)

plt.tight_layout()
plt.savefig(ASSETS / 'confusion_matrix.png', dpi=150)
print("Immagine salvata: assets/confusion_matrix.png")
