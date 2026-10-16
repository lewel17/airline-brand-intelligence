"""Step 6 - Confronto sul test set: distilBERT base vs distilBERT finetunato."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline
from paths import MODELS, PROCESSED

test = pd.read_csv(PROCESSED / "ft_test.csv")
testi = test['Review'].astype(str).tolist()
veri = test['label'].tolist()
print("Recensioni di test:", len(test))

def predici(modello):
    return [1 if modello(t)[0]['label'] == 'POSITIVE' else 0 for t in testi]

base = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True, max_length=256)
cartella_ft = str(MODELS / "modello_finetuned")
# Il tokenizer non cambia con il finetuning: si usa quello del modello base
TOKENIZER = "distilbert-base-uncased-finetuned-sst-2-english"
ft = pipeline("sentiment-analysis", model=cartella_ft, tokenizer=TOKENIZER,
              truncation=True, max_length=256)

pred_base, pred_ft = predici(base), predici(ft)
acc_base, f1_base = accuracy_score(veri, pred_base), f1_score(veri, pred_base)
acc_ft, f1_ft = accuracy_score(veri, pred_ft), f1_score(veri, pred_ft)

print("\nCONFRONTO MODELLI")
print("distilBERT BASE       - Accuracy:", round(acc_base, 3), "F1:", round(f1_base, 3))
print("distilBERT FINETUNATO - Accuracy:", round(acc_ft, 3), "F1:", round(f1_ft, 3))
print("Differenza accuracy:", round((acc_ft - acc_base) * 100, 1), "punti percentuali")
