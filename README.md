# Skin + Me — Customer Voice Intelligence Dashboard

A Streamlit dashboard built for the **BAUD A2 Unstructured Data Advisory Pitch**.
Visualises sentiment and thematic analysis of **1,082 Trustpilot reviews**
(real + synthetic) for independent skincare brand Skin + Me.

## What it shows

- **Volume & sentiment:** rating distribution and monthly sentiment trend
- **Theme analysis:** what customers praise vs. what they complain about
- **Customer & product mix:** conditions treated, verified vs. unverified sentiment, top products
- **Recommendations:** three evidenced actions with indicative commercial impact and a 30-day proposal

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select this repo, branch `main`, main file path `app.py`.
4. Deploy.

## Data disclosure

A portion of the 1,082-review dataset was generated via template-based GenAI
methods grounded in real Trustpilot themes. All findings are indicative and
intended to demonstrate analytical capability, not to support inferential claims.
