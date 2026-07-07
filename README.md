# 📊 CSV Chat Assistant — Chat with Any Dataset in Plain English

**Upload a CSV. Ask a question. Get the answer — *and the pandas code that computed it.***

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B%20via%20Groq-F55036)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)

Most "chat with your data" demos have a dirty secret: the LLM never actually computes anything — it guesses. This app doesn't guess. The LLM **writes one line of pandas code**, that code runs **in a sandbox against your real dataframe**, and the result comes back with the code displayed right under the answer. Every number is reproducible, every claim is auditable.

![App demo — groupby question against 8,807-row Netflix dataset](screenshots/q1_groupby_countries.png)

---

## How it works

The agent is a **hand-built ReAct loop** (Reason → Act → Observe) in ~200 lines — no black-box agent framework:

```mermaid
flowchart LR
    A["🧑 Question"] --> B["🧠 LLM sees schema only<br/>(columns, dtypes, 3 sample rows)"]
    B --> C["✍️ Writes ONE pandas expression"]
    C --> D["🔒 Sandboxed eval<br/>(no builtins, no imports, no file I/O)"]
    D -- "error" --> E["🔁 One retry<br/>with the error message"]
    E --> D
    D -- "result" --> F["💬 LLM phrases a one-sentence answer"]
    F --> G["Answer + code + table<br/>shown in chat"]
    E -- "fails again" --> H["❌ Real error shown —<br/>never a fabricated answer"]
```

Key properties:

- **The LLM never sees your data** — only the schema (column names, dtypes, 3 sample rows). A 10-million-row CSV costs the same tokens as a 10-row one, and your data stays private.
- **pandas does the math, not the LLM.** LLMs are unreliable at arithmetic over raw data; pandas is exact. The LLM is used only for what it's good at: translating English into code, and results into English.
- **Exactly one retry on failure.** The error message is fed back once; if the second attempt also fails, the user sees the *real* error. The app never invents a plausible-looking wrong answer.
- **Honest refusals.** Ask about data that isn't there ("Which titles won an Oscar?") and the agent returns a structured `UNANSWERABLE` response instead of hallucinating.

## Why not `create_pandas_dataframe_agent`?

LangChain's prebuilt pandas agent lives in `langchain_experimental` — a package that is **officially unmaintained**, with LangChain's own docs warning its examples "may be outdated or broken." This project deliberately builds the loop manually instead. That's not a missing feature; it's the point:

- The whole agent is ~200 readable lines I can explain, debug, and extend.
- The sandbox is *my* trust boundary, not an opaque dependency's.
- The prompt rules encode lessons learned from real messy data (see below).

## The sandbox

Executing LLM-generated code is the riskiest part of this design, so it runs behind three layers:

1. **`compile(code, mode="eval")`** — statements (`import os`, assignments, chained commands) are a syntax error before anything runs. Only a single expression is possible.
2. **Empty `__builtins__`** — the code's entire universe is two names: `df` and `pd`. `open()`, `__import__`, `exec` simply don't exist inside.
3. **Token denylist** — blocks dunder escapes (`__class__`-walking), `getattr` smuggling, and — a hole found by attacking my own sandbox — **pandas' own file I/O** (`pd.read_csv` can read arbitrary disk paths, so all `read_*`/`to_*` disk methods are rejected).

## Demo — 5 questions, 5 pandas patterns

Tested live against a real Netflix catalog dataset (8,807 rows × 12 columns).

| # | Pattern | Question |
|---|---------|----------|
| 1 | groupby + aggregation | Which 10 countries have the most titles, and what share are TV Shows? |
| 2 | filter + mean | Average duration of movies released after 2015? |
| 3 | sort + head | The 5 longest movies and their durations? |
| 4 | filter + count | How many titles were added to Netflix in 2021? |
| 5 | crosstab | Is there a relationship between type and rating? |

**1 — groupby + aggregation** — see the hero screenshot above. US leads with 2,818 titles; TV-show share ranges from 8% (India) to 79% (South Korea).

**2 — filter + mean**

![Average movie duration after 2015](screenshots/q2_filter_mean_duration.png)

**3 — sort + head** — *Black Mirror: Bandersnatch* (312 min) tops the list.

![5 longest movies](screenshots/q3_sort_head_longest_movies.png)

**4 — filter + count** — note the generated code: it strips whitespace and parses dates with `format='mixed'`, because this dataset's `date_added` column hides a leading space that breaks naive parsing.

![Titles added in 2021](screenshots/q4_filter_count_added_2021.png)

**5 — crosstab** — movies dominate PG/PG-13/R; TV skews TV-MA/TV-14. (Only **2** R-rated TV shows exist in the entire catalog.)

![Type vs rating crosstab](screenshots/q5_crosstab_type_vs_rating.png)

**Bonus: a genuinely hard question.** *"Build a time series of titles added per year — does the trend differ between Movies and TV Shows?"* requires date parsing, null handling, a two-level groupby, and a pivot — generated and executed as a single expression:

![Time series of titles added per year by type](screenshots/bonus_time_series_growth.png)

## Battle scars: prompt rules learned from real data

The first version handled clean questions fine and fell over on real-world mess. Each failure became a permanent rule in the code-generation prompt:

| Failure | Rule now in the prompt |
|---|---|
| `" August 4, 2017"` (leading space) broke strict date formats | Always `pd.to_datetime(col.str.strip(), format='mixed', errors='coerce')` |
| `.astype(int)` crashed on titles with missing durations | Extract numbers with `pd.to_numeric(..., errors='coerce')` |
| "Build a time series" made the LLM call `.plot()` | Never plot — return the grouped data; the UI renders tables |
| Comma-separated `cast` cells | `.str.split(', ')` + `.explode()` before grouping |
| Vague "tell me about this data" | Fall back to `df.describe(include='all')` |

## Setup & run

```bash
git clone https://github.com/arnavvpareek/csv-chat-assistant.git
cd csv-chat-assistant
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Create a free API key at [console.groq.com](https://console.groq.com), then create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Run:

```bash
streamlit run app.py
```

Upload any CSV and start asking questions.

## Project structure

```
├── agent.py          # Sandboxed executor + ReAct agent loop (~200 lines)
├── app.py            # Streamlit chat UI
├── requirements.txt  # Pinned dependencies
├── screenshots/      # Demo evidence
└── .env              # Your Groq API key (git-ignored)
```

## Limitations (the honest section)

1. **The sandbox is a guardrail, not a jail.** A restricted in-process `eval` defends against an LLM going off-script — which is the actual threat model here — but the Python security community's consensus is that no in-process eval sandbox can fully contain a *determined human attacker* (creative object-graph traversals keep being discovered). A production deployment handling adversarial users would execute generated code in an isolated subprocess or container. I chose the eval approach consciously, knowing where its boundary lies.

2. **One expression per question limits analytical depth.** Multi-stage analyses (fit a regression, then compare residuals across groups) can't fit in a single pandas expression, and the agent has no conversational memory — each question is answered from scratch, so "now break that down by year" won't understand what "that" is. Supporting true multi-step reasoning would need a plan-and-execute loop with intermediate state, which trades the simplicity and auditability that make this design easy to trust.

3. **Answers depend on an LLM reading a schema, not a data dictionary.** Column names like `rating` are ambiguous (content rating? user score?). The model usually infers correctly from sample values, but a misread column can produce a *correct-looking* answer to the wrong question — the displayed code is the user's tool for catching this, and it requires them to read it.

---

*Built by [Arnav Pareek](https://github.com/arnavvpareek) — Python · LangChain · Groq · Streamlit*
