# 📘 Interview Prep — AI Investment Research Agent

> Is file me poora project **step by step** samjhaya gaya hai — jaise interviewer
> puchhe *"ye project tumne kaise banaya, kya-kya kiya, kaise kaam karta hai?"*
> to aap confidently jawab de sako. Niche Hinglish me explanation + interview
> Q&A hai.

---

## 1. One-line pitch (interview me sabse pehle ye bolo)

> "Maine ek **AI Investment Research Agent** banaya hai — ek Django web app jisme
> aap kisi public company ka ticker (ya naam) daalte ho, aur ek **LangGraph
> multi-step AI workflow** us company ko research karke **INVEST / PASS**
> recommendation deta hai, saath me financial score, risk score aur sentiment
> score. AI ke liye **Google Gemini** use kiya, live financial data ke liye
> **yfinance**, aur tools **LangChain** se banaye."

---

## 2. Tech Stack (aur kyun choose kiya)

| Layer | Tech | Kyun |
|-------|------|------|
| Web framework | **Django** | Mature, batteries-included, server-side rendering (no JS framework chahiye) |
| Templates | **Django Templates** | Simple SSR, fast to build, koi React/Next overhead nahi |
| AI orchestration | **LangGraph** | Multi-step workflow ko ek graph (nodes + edges) ki tarah model karta hai — clean aur visual |
| LLM tooling | **LangChain** | Standardized `@tool` interface, Gemini integration |
| LLM | **Google Gemini** (`gemini-1.5-flash`) | Fast, sasta, free tier available |
| Market data | **yfinance** | Free, no API key, Yahoo Finance se live data |
| Config | **python-dotenv** | Secrets `.env` me, code me nahi |
| DB | **SQLite** | Demo/student project ke liye perfect, zero setup |

---

## 3. Project ka flow (high level)

```
User browser
   │  ticker daala (e.g. "AAPL" ya "Google")
   ▼
Django View (analyze)
   │  run_analysis(ticker) call
   ▼
LangGraph Workflow
   START
     → Company Research Node      (yfinance se profile)
     → Financial Analysis Node    (yfinance metrics + Gemini score)
     → News Analysis Node         (yfinance news + Gemini sentiment)
     → Risk Analysis Node         (Gemini se 4 risks + score)
     → Investment Decision Node   (Gemini se final verdict)
   END
   ▼
Result dict (company, financials, news, risk, decision)
   ▼
Django Template (results.html) → user ko dikhta hai
```

---

## 4. Folder structure (aur har folder ka kaam)

```
langproject/
├── manage.py                 # Django CLI entry point
├── .env                      # SECRETS (Gemini key) — git me NAHI jaata
├── .env.example              # Template (placeholder values)
├── requirements.txt          # Dependencies
├── langproject/              # Project config
│   ├── settings.py           # dotenv load, apps, Gemini settings
│   └── urls.py               # routes → research app
└── research/                 # MAIN app
    ├── models.py             # AnalysisRecord (har analysis DB me save)
    ├── views.py              # home + analyze views (controllers)
    ├── urls.py               # app ke routes
    ├── templates/research/   # base / home / results HTML
    └── agent/                # 🧠 AI ka dimaag
        ├── llm.py            # Gemini connection + JSON parser
        ├── state.py          # nodes ke beech shared data (TypedDict)
        ├── tools.py          # 5 LangChain tools + ticker resolver
        ├── nodes.py          # 5 workflow steps
        └── workflow.py       # LangGraph graph banata + chalata hai
```

---

## 5. Step-by-step: maine kaise banaya

### Step 1 — Django project + app setup
- `django-admin startproject langproject` se project banaya.
- `research` naam ki app banayi aur `INSTALLED_APPS` me add ki.
- `settings.py` me **python-dotenv** add kiya taaki `GOOGLE_API_KEY` aur
  `GEMINI_MODEL` `.env` file se padhe — secrets kabhi code me nahi.

### Step 2 — Gemini integration (`agent/llm.py`)
- `get_llm()` function banaya jo `ChatGoogleGenerativeAI` (LangChain ka Gemini
  wrapper) return karta hai.
- **Important trick:** `ask_json()` helper banaya. LLM se hum **JSON-only**
  reply maangte hain (scores, lists, labels), phir use safely parse karte hain.
  Agar model `​```json` code fences laga de to wo strip karte hain, aur agar
  JSON galat ho to crash nahi — error dict return karke gracefully degrade.

### Step 3 — Shared state (`agent/state.py`)
- `AgentState` ek **TypedDict** hai (`total=False` = saare keys optional).
- Ye wo object hai jo har node ke beech pass hota hai. Har node apna hissa
  (`company`, `financials`, `news`, `risk`, `decision`) bharta hai.

### Step 4 — LangChain Tools (`agent/tools.py`)
5 tools banaye (requirement ke hisaab se):
1. `company_information_tool` — naam, industry, sector, market cap, CEO
2. `financial_data_tool` — revenue, margins, P/E, debt, cash flow
3. `news_research_tool` — recent headlines
4. `sentiment_analysis_tool` — Gemini se positive/neutral/negative + score
5. `investment_decision_tool` — sab data combine karke final verdict

> **Design choice (interview gold):** Heavy logic plain helper functions me
> rakha (`get_company_information` etc.), aur `@tool` wrappers patle rakhe.
> Isse nodes seedha functions call karte hain (deterministic + fast), aur tools
> bhi available rehte hain agar future me LLM ko khud tools chalwane ho (agentic).

> **Bonus — `resolve_ticker()`:** User "Google" ya "Apple" likhe to bhi chale.
> Pehle direct ticker try karta hai, fail ho to `yf.Search()` se company naam
> se symbol dhoondhta hai (e.g. "Google" → "GOOG").

### Step 5 — Workflow Nodes (`agent/nodes.py`)
Har node ka same contract:
```python
def node(state: AgentState) -> AgentState:
    # kuch compute karo
    return {"some_key": value}   # PARTIAL state update
```
LangGraph ye partial update merge kar deta hai. 5 nodes:
- **company_research_node** — yfinance se profile.
- **financial_analysis_node** — metrics nikaale, phir Gemini se 1–10 health score.
- **news_analysis_node** — news laaya, Gemini se summary + sentiment score.
- **risk_analysis_node** — Gemini se 4 risks (regulatory, competition,
  financial, market) + 1–10 risk score.
- **investment_decision_node** — sab combine karke INVEST/PASS + overall score
  + strengths + risks + reasoning.

> `_clamp_score()` helper har score ko 1–10 ke beech force karta hai (LLM kabhi
> galti se 12 ya 0 de de to bhi safe).

### Step 6 — LangGraph graph (`agent/workflow.py`)
```python
graph = StateGraph(AgentState)
graph.add_node("company_research", company_research_node)
# ... baaki nodes
graph.add_edge(START, "company_research")
graph.add_edge("company_research", "financial_analysis")
# ... sequential edges ... → END
workflow = graph.compile()
```
- Graph **ek baar** import time pe compile hota hai (har request pe nahi —
  efficiency).
- `run_analysis(ticker)` = public entry point jo Django views call karte hain.

### Step 7 — Django Views + URLs (`research/views.py`, `urls.py`)
- `home` view → input form dikhata hai.
- `analyze` view (POST) → `run_analysis()` chalata hai, result `AnalysisRecord`
  me save karta hai, phir `results.html` render karta hai.

### Step 8 — Templates (SSR)
- `base.html` — layout + dark theme CSS + **loading spinner overlay**.
- `home.html` — input form.
- `results.html` — overview, financials, news, risk, scores, recommendation,
  reasoning sab cards me.

### Step 9 — UX: Loading spinner
- Analysis me 10–30 sec lagte hain (multiple Gemini calls), isliye form submit
  pe ek full-screen **spinner overlay** dikhata hai aur button disable kar deta
  hai (double-submit rokne ke liye). Pure CSS + thoda vanilla JS.

---

## 6. Scoring logic (interviewer ye zaroor puchhega)

- **Financial Score (1–10):** Gemini ko metrics deke health judge karwate hain
  (revenue growth, margins, debt, P/E, cash flow).
- **Sentiment Score (1–10):** News headlines ka overall tone — 1 = bahut
  negative, 10 = bahut positive, 5 = neutral.
- **Risk Score (1–10):** 1 = bahut kam risk, 10 = bahut zyada risk.
- **Overall Score + INVEST/PASS:** Decision node sab ko consider karke deta hai.
  INVEST tabhi jab evidence ka balance favorable ho.

---

## 7. Error handling & robustness (ye batana achha impression banata hai)

- **Invalid ticker:** Agar company data na mile to friendly error + home pe wapas.
- **Name vs ticker:** `resolve_ticker()` naam ko symbol me convert karta hai.
- **LLM galat JSON de:** `ask_json()` safe defaults return karta hai, crash nahi.
- **yfinance news format change:** Code purane aur naye dono format handle karta hai.
- **DB save fail:** Best-effort try/except — page kabhi block nahi hota.
- **Score out of range:** `_clamp_score()` 1–10 me force karta hai.

---

## 8. Sample output format

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
- High valuation (elevated P/E)
- Regulatory scrutiny

Reasoning: ...
```

---

## 9. Possible Interview Questions + Answers

**Q: LangGraph kyun, simple functions se kaam ho jaata?**
A: Haan ho jaata, par LangGraph workflow ko **graph (nodes + edges)** ki tarah
model karta hai — readable, extendable, aur future me conditional branching
(e.g. risk zyada ho to extra check) ya loops aasani se add kar sakte hain.

**Q: State management kaise hua?**
A: Ek `AgentState` TypedDict shared state hai. Har node partial update return
karta hai jo LangGraph automatically merge kar deta hai. Mutable global state
nahi, clean data flow.

**Q: LLM se structured output kaise nikaala?**
A: Prompt me bola "JSON ONLY in this exact shape", phir `ask_json()` se parse
kiya — code fences strip + error fallback. (Alternative: PydanticOutputParser
ya structured output mode.)

**Q: Secrets kaise handle kiye?**
A: `.env` file me, `python-dotenv` se load. `.env.example` template commit hota
hai (placeholder), asli `.env` `.gitignore` me hai.

**Q: Scalability / production me kya change karoge?**
A: Long analysis ko **async/Celery background task** me daalunga (abhi request
block karta hai), Postgres use karunga, Gemini calls pe caching + rate limiting,
aur results page ko polling/websocket se update karunga.

**Q: Sabse bada challenge kya tha?**
A: LLM ka unpredictable output — kabhi extra text, kabhi galat JSON. Isliye
robust parsing (`ask_json`) aur safe defaults banane pade taaki app kabhi crash
na ho.

**Q: yfinance reliable hai?**
A: Free hai par unofficial (Yahoo scraping). Production me paid data provider
(Alpha Vantage, Polygon, etc.) better hoga. Maine error handling rakhi hai
taaki missing data pe app gracefully "N/A" dikhaye.

---

## 10. Disclaimer (interviewer ko bata dena)

> Ye project **educational** hai, real financial advice nahi. AI output ko
> blindly trust nahi karna chahiye — ye ek research *assistant* hai, decision
> maker nahi.

---

### 🎯 One-line closing (interview ke end me)

> "Is project se maine seekha kaise **LLM ko ek structured, multi-step pipeline**
> me orchestrate karte hain (LangGraph), live data (yfinance) ke saath combine
> karte hain, aur sab kuch ek clean Django web app me wrap karke robust error
> handling ke saath deliver karte hain."
