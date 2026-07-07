"""Agent loop: turns a natural-language question into a pandas expression,
executes it in a sandbox, and phrases the result as an answer.

Built manually (ReAct pattern) instead of using langchain_experimental's
create_pandas_dataframe_agent, which is unmaintained.
"""

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class UnsafeCodeError(Exception):
    """Raised when generated code fails the safety checks."""


# Substrings that must never appear in generated code. `__` alone blocks the
# whole family of dunder escapes (__import__, __builtins__, __class__, ...).
BLOCKED_TOKENS = [
    "__",
    "import",
    "open(",
    "exec",
    "eval",
    "compile",
    "input",
    "globals",
    "locals",
    "vars(",
    "getattr",
    "setattr",
    "delattr",
    "os.",
    "sys.",
    "subprocess",
    "breakpoint",
    # pandas' own file I/O — pd is exposed in the namespace, so its readers
    # and writers must be blocked explicitly (pd.read_csv etc. can touch disk).
    "read_",
    "to_csv",
    "to_excel",
    "to_pickle",
    "to_parquet",
    "to_sql",
    "to_hdf",
    "to_feather",
    "to_stata",
    "to_clipboard",
    "to_json",
    "to_html",
    "to_latex",
    "to_xml",
]


def clean_code(code: str) -> str:
    """Strip markdown fences and surrounding noise from LLM output."""
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[len("python"):]
    return code.strip().strip("`").strip()


def run_pandas_expression(df: pd.DataFrame, code: str):
    """Evaluate a single pandas expression against df in a restricted namespace.

    The namespace exposes only `df` and `pd`; __builtins__ is emptied so the
    expression cannot reach open(), import machinery, os/sys, or file I/O.
    A denylist rejects dunder tricks before evaluation. compile(..., "eval")
    guarantees the code is one expression — statements are a syntax error.

    Note: a restricted eval is a guardrail against an LLM going off-script,
    not a hardened jail against a determined human attacker.
    """
    code = clean_code(code)

    lowered = code.lower()
    for token in BLOCKED_TOKENS:
        if token in lowered:
            raise UnsafeCodeError(f"Blocked token in generated code: {token!r}")

    compiled = compile(code, "<llm_generated>", "eval")

    safe_namespace = {"__builtins__": {}, "df": df, "pd": pd}
    return eval(compiled, safe_namespace)


def get_llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def describe_schema(df: pd.DataFrame) -> str:
    """Compact schema summary sent to the LLM instead of the full dataset."""
    lines = [f"The dataframe `df` has {len(df)} rows and {len(df.columns)} columns."]
    lines.append("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {dtype}")
    lines.append("First 3 rows:")
    lines.append(df.head(3).to_string())
    return "\n".join(lines)


CODE_PROMPT = r"""You are a data analyst. A pandas DataFrame named `df` is already loaded.

{schema}

Question: {question}

Write ONE single pandas expression that computes the answer.
Rules:
- Exactly one expression, no assignments, no imports, no print().
- Use only `df` and `pd`.
- Do not use file operations (read_*, to_csv, etc.).
- The expression may be complex: method chaining, .assign(), .pipe(), and
  lambdas are all fine, as long as it is one expression.
- To parse dates from strings, ALWAYS use
  pd.to_datetime(col.str.strip(), format='mixed', errors='coerce') —
  real-world data has inconsistent formats and stray whitespace. Never pass
  an explicit strftime format.
- If a column holds comma-separated lists (e.g. multiple actors in one
  cell), split with .str.split(', ') and use .explode() before grouping.
- To extract numbers from text columns (e.g. "90 min"), use
  pd.to_numeric(col.str.extract(r'(\d+)', expand=False), errors='coerce')
  — never .astype(int), which crashes on missing values.
- To limit a crosstab or groupby to the N most common categories, first
  filter rows with .isin(df['col'].value_counts().head(N).index).
- Drop missing values (.dropna()) in columns you rely on before exploding
  or grouping them.
- NEVER call .plot() or any charting — return the underlying data instead
  (a Series or DataFrame); the UI renders it as a table. For "time series"
  or "trend" questions, return counts/values grouped by period.
- When asked for "top" or "most frequent", return a manageable number of
  rows (e.g. .head(10)), not the entire sorted table.
- If the question is vague or asks for a general overview (e.g. "tell me
  about this data", "summarize this dataset"), do NOT guess a specific
  computation — return df.describe(include='all') so the user gets a
  broad statistical summary.
- If the question cannot be answered from these columns at all, the entire
  expression must be just a string literal: 'UNANSWERABLE: <brief reason>'.
  Do not wrap it in df operations.
- Output ONLY the expression — no explanation, no markdown fences.
"""

RETRY_SUFFIX = """
Your previous attempt was:
{code}
It failed with this error:
{error}
Write ONE corrected pandas expression. Output ONLY the expression.
"""

ANSWER_PROMPT = """A user asked this question about a dataset: {question}

The pandas expression `{code}` was executed and returned this result:
{result}

Answer the user's question in one short, natural sentence using that result.
Do not mention pandas or code. If the result is a table, just summarize what
it shows in one sentence.
"""


def ask_agent(df: pd.DataFrame, question: str) -> dict:
    """Full ReAct loop: question -> pandas code -> sandboxed run -> answer.

    Returns a dict with:
      - "answer": one-sentence natural-language answer
      - "code":   the pandas expression that produced the result
      - "result": the raw result (scalar, Series, or DataFrame)
    Raises RuntimeError if the code fails twice (one retry, as designed).
    """
    llm = get_llm()
    schema = describe_schema(df)

    prompt = CODE_PROMPT.format(schema=schema, question=question)
    code = clean_code(llm.invoke(prompt).content)

    try:
        result = run_pandas_expression(df, code)
    except Exception as first_error:
        retry_prompt = prompt + RETRY_SUFFIX.format(code=code, error=first_error)
        code = clean_code(llm.invoke(retry_prompt).content)
        try:
            result = run_pandas_expression(df, code)
        except Exception as second_error:
            raise RuntimeError(
                f"The generated code failed twice. Last attempt:\n{code}\n"
                f"Error: {second_error}"
            ) from second_error

    if isinstance(result, str) and result.startswith("UNANSWERABLE"):
        reason = result.removeprefix("UNANSWERABLE:").strip()
        return {
            "answer": f"This dataset can't answer that question. {reason}",
            "code": code,
            "result": None,
        }

    # Truncate huge results so the answer prompt stays small; the UI still
    # gets the full result for table display.
    result_text = str(result)
    if len(result_text) > 1500:
        result_text = result_text[:1500] + "\n... (truncated)"

    answer = llm.invoke(
        ANSWER_PROMPT.format(question=question, code=code, result=result_text)
    ).content.strip()

    return {"answer": answer, "code": code, "result": result}
