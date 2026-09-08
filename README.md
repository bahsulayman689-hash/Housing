# Advanced House Price Prediction Engine & Observability Suite

A Streamlit dashboard for house price valuation using an XGBoost regression pipeline, with live telemetry monitoring and a Gemini-powered chat assistant that answers buyer questions grounded in the actual model output.

## Features

- **Engine Inference Console** — enter property details and get an instant fair-market valuation
- **Live System Monitoring** — latency and valuation history charts, raw prediction logs
- **Model Specs & Telemetry** — architecture and preprocessing summary
- **Raw Diagnostics & Data Rules** — schema and feature contract for the model
- **Floating chat assistant** — answers buyer questions using Gemini, grounded in the session's real prediction data (falls back to rule-based answers if no API key is configured)

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Add the model artifact

Place your trained pipeline (StandardScaler + XGBoost, saved as a single scikit-learn `Pipeline`) in the project root as:

```
best_model.pkl
```

Optional assets, also in the project root:
- `logo.png` — app header logo
- `IMG-20260704-WA0629.jpg` — sidebar profile picture

### 3. Configure the Gemini API key

Create `.streamlit/secrets.toml` in the project root (same folder as `app.py`):

```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

**Important:**
- The folder must be named exactly `.streamlit` (dot first) — on Windows, create it from a terminal (`mkdir .streamlit`), not File Explorer, which can mangle dot-folder names.
- Save the file as plain UTF-8 (no BOM).
- Never commit `secrets.toml` — add it to `.gitignore`.

Alternatively, set the key as an environment variable instead:

```bash
set GEMINI_API_KEY=your-actual-key     # Windows
export GEMINI_API_KEY=your-actual-key  # macOS/Linux
```

If no key is found, the chatbot automatically falls back to simple rule-based responses — the app still runs fine without Gemini.

### 4. Run the app

```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

- `requirements.txt` drives the cloud build — make sure it's committed.
- `secrets.toml` is **not** committed (it's gitignored). Instead, paste its contents into the app's **Settings → Secrets** panel in the Streamlit Cloud dashboard.

## Troubleshooting

If the chatbot shows a caption like `💡 LLM not active — ...`, the message after the dash tells you exactly why (package not installed, secrets not found, or invalid API key/model) — fix that specific cause and restart the app.

## Project structure

```
.
├── app.py
├── best_model.pkl
├── requirements.txt
├── README.md
├── logo.png                  (optional)
├── IMG-20260704-WA0629.jpg   (optional)
└── .streamlit/
    └── secrets.toml
```
