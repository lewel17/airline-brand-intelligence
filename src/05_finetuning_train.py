"""Step 5 - Finetuning di distilBERT sulle recensioni etichettate.

Epoche: 2. Con 3 epoche la validation loss continua a salire
(0.130 -> 0.144 -> 0.155) a fronte di un guadagno di accuracy minimo:
segnale di overfitting, quindi il modello finale usa 2 epoche.
"""
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from paths import MODELS, PROCESSED

MODELLO_BASE = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT = MODELS / "modello_finetuned"

train_df = pd.read_csv(PROCESSED / "ft_train.csv")
val_df = pd.read_csv(PROCESSED / "ft_validation.csv")

train_ds = Dataset.from_pandas(train_df[['Review', 'label']])
val_ds = Dataset.from_pandas(val_df[['Review', 'label']])

tokenizer = AutoTokenizer.from_pretrained(MODELLO_BASE)
model = AutoModelForSequenceClassification.from_pretrained(MODELLO_BASE, num_labels=2)

def tokenizza(esempi):
    return tokenizer(esempi['Review'], truncation=True, padding='max_length', max_length=256)

train_ds = train_ds.map(tokenizza, batched=True)
val_ds = val_ds.map(tokenizza, batched=True)

def calcola_metriche(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1,
            'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir=str(MODELS / "checkpoints"),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,          # tiene solo il checkpoint migliore (ognuno pesa ~800 MB)
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    seed=42,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                  eval_dataset=val_ds, compute_metrics=calcola_metriche)

print("Inizio finetuning...")
trainer.train()

trainer.save_model(str(OUTPUT))
tokenizer.save_pretrained(str(OUTPUT))
print("Modello salvato in models/modello_finetuned")
