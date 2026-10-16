# Exploration

These are early experiments, kept to document how the project developed. They are not part of the final pipeline: they read the CSV files from the current folder and use the original file names.

| Script | What it does |
|---|---|
| `test_scraping.py` | First attempt at scraping Skytrax directly. Replaced by the Kaggle dataset |
| `esplora_dati.py`, `esplora2.py` | First look at the dataset: airline names, missing values, review lengths, score distribution |
| `test_sentiment.py` | distilBERT sanity check on 5 example sentences |
| `topic_ryanair.py`, `topic_easyjet.py` | First BERTopic runs per airline, used to tune parameters |
| `incrocio.py` | First cross-analysis of themes and sentiment. Became `src/08_temi_sentiment.py` |
