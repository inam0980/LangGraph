# 📈 AI Investment Research Agent

A server-side-rendered Django app that researches any public company and
returns an **INVEST / PASS** recommendation. The analysis is produced by a
**LangGraph** multi-step workflow powered by **Google Gemini** and **LangChain**
tools, with live financial data from **yfinance**.

```
START → Company Research → Financial Analysis → News Analysis
      → Risk Analysis → Investment Decision → END
```

---

## ✨ Features

- **Multi-step AI workflow** orchestrated with LangGraph.
- **5 LangChain tools**: company info, financial data, news, sentiment, decision.
- **Live market data** via yfinance (no paid data API needed).
- **Scores out of 10**: financial health, risk, sentiment, and an overall score.
- **Server-side rendering** with plain Django templates (no JS framework).
- **Environment-based config** with `.env`.

---

## 🗂️ Project Structure

```
lang/
├── manage.py                 # Django CLI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Copy to .env and fill in
├── README.md
├── config/                   # Django project (settings, urls, wsgi/asgi)
│   ├── settings.py
│   └── urls.py
└── research/                 # The main app
    ├── models.py             # AnalysisRecord (saved runs)
    ├── views.py              # home + analyze views
    ├── urls.py               # app routes
    ├── templates/research/   # base / home / results pages
    └── agent/                # 🧠 the AI layer
        ├── llm.py            # Gemini integration + JSON helper
        ├── state.py          # shared LangGraph state (TypedDict)
        ├── tools.py          # 5 LangChain tools
        ├── nodes.py          # the 5 workflow steps
        └── workflow.py       # builds + runs the LangGraph graph
```

---

## 🚀 Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
# copy the example file
cp .env.example .env      # Windows: copy .env.example .env
```
Open `.env` and set your **Gemini API key** (free from
<https://aistudio.google.com/app/apikey>):
```
GOOGLE_API_KEY=your-real-key
```

### 4. Set up the database
```bash
python manage.py migrate
```

### 5. Run the server
```bash
python manage.py runserver
```
Open <http://127.0.0.1:8000/> and analyze a ticker (e.g. `AAPL`).

---

## 🧠 How the workflow works

Each node receives the shared **state** (`research/agent/state.py`) and returns
a partial update that LangGraph merges back in:

| Node | Reads | Produces |
|------|-------|----------|
| Company Research | ticker | name, industry, sector, market cap, CEO |
| Financial Analysis | ticker | metrics + analysis + `financial_score` (1–10) |
| News Analysis | ticker, company | headlines + summary + sentiment + `sentiment_score` |
| Risk Analysis | all above | regulatory/competition/financial/market + `risk_score` |
| Investment Decision | all above | recommendation, `overall_score`, strengths, risks, reasoning |

The deterministic data steps call yfinance; the reasoning steps call Gemini and
ask for **JSON-only** responses, which `llm.ask_json()` parses safely.

---

## 📋 Example output

```
Recommendation: INVEST
Overall Score: 8/10

Financial Score: 9/10
Risk Score: 4/10
Sentiment Score: 7/10

Strengths:
- Strong revenue growth and high margins
- Healthy free cash flow

Risks:
- Elevated valuation (high P/E)
- Regulatory scrutiny in key markets

Reasoning:
...
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is **not** financial
advice. Always do your own research before investing.
# LangGraph
