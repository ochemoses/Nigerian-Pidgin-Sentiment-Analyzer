# Naija Pidgin Sentiment Reader — Dashboard

Flask web app for the Nigerian Pidgin Sentiment Analyzer.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python app.py
```
This starts the server at http://127.0.0.1:5000 and opens it in your browser automatically.

## Files
- `app.py` — Flask backend (loads `sentiment_model.joblib`, serves the dashboard, exposes `/api/predict`)
- `templates/index.html` — dashboard page
- `static/style.css`, `static/app.js` — dashboard styling and logic
- `sentiment_model.joblib` — trained model (must be in the same folder as `app.py`)

## Notes
- The gauge needle position is `P(positive) − P(negative)`, mapped to a -90°..90° sweep.
- Recent-checks history is kept in the browser tab only (not saved anywhere) — refreshing the page clears it.
