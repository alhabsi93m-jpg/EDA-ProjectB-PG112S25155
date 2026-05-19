# EDA Mini Project B — Time-Series Forecasting Starter

Student: Marwa  
Student ID: PG112S25155

This repository contains a starter Streamlit app for Mini Project B using a cleaned time-series dataset slice.

## Files

- `app.py` — one-file Streamlit starter app
- `requirements.txt` — required Python packages
- `data/dataset_sample.csv` — cleaned dataset sample, capped at 250,000 rows

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Choose branch `main`.
6. Set the main file path to `app.py`.
7. Deploy.

## OpenRouter API key

The app does not hardcode any API key. For AI grading, add your key using one of these methods:

- Streamlit Secrets: `OPENROUTER_API_KEY`
- Environment variable: `OPENROUTER_API_KEY`
- Password input field inside the app

## What to submit

Submit the following to your instructor:

- Streamlit app URL
- GitHub repo URL
- Exported `submission.json`
- Exported `project_card.md`
- Required screenshots from the deployed app
