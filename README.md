# Nigerian Pidgin Sentiment Analyzer

**3MTT AI/ML Fellowship Project — Oche Abel**

## Problem
Nigerian businesses receive customer feedback in Pidgin/English that generic (English-only) sentiment tools misread, because they aren't trained on Pidgin vocabulary, spelling variation, or code-mixing.

## MVP
A text classifier that labels a piece of feedback as **positive**, **negative**, or **neutral**, returning a prediction plus confidence scores per class.

## Dataset
`pidgin_sentiment_v5_refined2.csv` — 11,491 labeled examples, no missing values, no exact duplicates.

| label | count | % |
|---|---|---|
| negative | 4,859 | 42.3% |
| positive | 4,388 | 38.2% |
| neutral | 2,244 | 19.5% |

**Note on composition:** the dataset blends two styles of text:
- **~39% (4,474 rows)** on-topic, templated business-feedback sentences (e.g. *"Sha, the passport office too slow but e better pass before."*) covering things like ATMs, delivery, network, POS, hospitals, school portals.
- **~61% (7,017 rows)** organic, social-media-style Pidgin text — broader vocabulary and spelling variation, but not specifically about products/services, and occasionally containing profanity/slurs and ambiguous sentiment.

The full dataset was used for training per project decision. This gives the model wider Pidgin vocabulary coverage at the cost of being less narrowly tuned to customer-feedback language specifically — see Limitations below.

## Approach
1. Light cleaning: lowercase, strip URLs/@mentions/stray emoji artifacts, collapse whitespace. Deliberately minimal — Pidgin sentiment signal often lives in spelling and emphasis (e.g. repeated letters), so no aggressive normalization or stopword removal.
2. Stratified 80/20 train/test split (preserves class proportions, important since `neutral` is the minority class).
3. TF-IDF vectorization (word unigrams + bigrams, `min_df=2`, `max_df=0.9`, sublinear TF scaling).
4. Compared two linear classifiers: Logistic Regression and Linear SVM, both with `class_weight="balanced"` to counter the class imbalance.
5. Selected the better model by macro-F1 (fairer than accuracy given class imbalance).

## Results

**Logistic Regression** was the best model: **78.0% accuracy, 0.791 macro-F1**.

| class | precision | recall | f1-score | support |
|---|---|---|---|---|
| negative | 0.748 | 0.809 | 0.777 | 972 |
| neutral | 0.918 | 0.777 | 0.842 | 449 |
| positive | 0.759 | 0.751 | 0.755 | 878 |

Confusion matrix: `results/confusion_matrix.png`

22.0% of test examples were misclassified — mostly cases with mixed or subtle sentiment, or organic/social-media text where sentiment is genuinely ambiguous even to a human reader.

## Files
```
notebook/Pidgin_Sentiment_Analyzer.ipynb   # full pipeline: EDA -> train -> evaluate -> save -> demo
data/pidgin_sentiment_v5_refined2.csv      # dataset
results/confusion_matrix.png               # saved from a notebook run
sentiment_model.joblib                     # trained pipeline (TF-IDF + Logistic Regression)
README.md
```

## How to run
1. Open `notebook/Pidgin_Sentiment_Analyzer.ipynb` in Google Colab.
2. Upload `pidgin_sentiment_v5_refined2.csv` when prompted (or mount Drive and edit `DATA_PATH`).
3. Run all cells top to bottom (~1–2 minutes).
4. Try your own sentences in the inference demo cell at the end.

## Limitations
- Because ~61% of the training data is general Pidgin social-media text rather than customer feedback, the model's real-world performance on actual business feedback may differ from the 78% test accuracy reported here — the test set is drawn from the same mixed distribution as training.
- `neutral` is the smallest class; despite the strong neutral F1 here, watch this class specifically if you expand the dataset later.
- This is a classical ML baseline (TF-IDF + linear model), which is appropriate for the MVP scope. A transformer-based multilingual model (e.g. fine-tuned AfriBERTa or XLM-R) would likely handle the noisier organic text and code-mixing better, but is out of scope here.
- Some source rows contain profanity/slurs. Worth filtering before using this repo/demo in a public-facing context (e.g. the 2–3 min demo video).

## Suggested tools used
Python, pandas, scikit-learn, Google Colab.
