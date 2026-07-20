# Multi-Agent Analytics Assistant

**Level 2 Project — CrewAI + Ollama + FastMCP + Streamlit**

---

## What This System Does

This system accepts a plain-English analytics question, routes it to specialist AI agents, calls local Python tools and a local MCP server, and returns a structured 10-section answer — all running entirely on your own machine with no external API calls.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install MCP support separately if the auto-prompt fails
#    (see "Known Issues Fixed" section below for why this matters)
pip install mcp
pip install "crewai-tools[mcp]"

# 3. Start Ollama in a separate terminal
ollama serve
ollama pull llama3.2:latest

# 4. Test the MCP server standalone (run as a MODULE, not a script)
python -m mcp_server.server

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The sidebar model dropdown defaults to `ollama/llama3.2:latest`.

---

## Project Structure

```
level_2_multi_agent_mcp_project/
│
├── app.py                          # Streamlit UI + CrewAI crew runner
├── requirements.txt                # All pinned dependencies
│
├── config/
│   ├── agents.yaml                 # Agent roles, goals, backstories
│   └── tasks.yaml                  # Manager task with 10-section output spec
│
├── agents/
│   ├── supervisor_agent.py         # Builds Supervisor Agent — NO tools (see note below)
│   ├── data_analyst_agent.py       # Builds Analyst Agent with 5 + MCP tools
│   └── data_scientist_agent.py     # Builds Scientist Agent with 5 + MCP tools
│
├── function_tools/
│   ├── supervisor_tools.py         # 5 tools — called directly from app.py, not the agent
│   ├── analyst_tools.py            # 5 local tools for Data Analyst Agent
│   └── scientist_tools.py          # 5 local tools for Data Scientist Agent
│
├── mcp_server/
│   ├── server.py                   # FastMCP server — registers 10 MCP tools
│   ├── tools/
│   │   ├── csv_profile_tools.py    # MCP Tools 1, 4 — profiling + quality
│   │   ├── sql_tools.py            # MCP Tools 2, 3 — DuckDB query + SQL validation
│   │   ├── kpi_tools.py            # MCP Tools 5, 9 — KPI catalog + data dictionary
│   │   ├── ml_tools.py             # MCP Tools 6, 7, 8 — ML use cases, features, anomaly
│   │   └── report_tools.py         # MCP Tool 10 — final markdown report generator
│   └── sample_data/
│       ├── events_sample.csv       # 40 rows — platform events with user, type, status
│       ├── transactions_sample.csv # 40 rows — payments with amount, fraud flags
│       └── customers_sample.csv    # 33 rows — customers with churn labels
│
├── tests/
│   ├── test_supervisor_tools.py    # 27 tests for supervisor tools
│   ├── test_analyst_tools.py       # 31 tests for analyst tools
│   ├── test_scientist_tools.py     # 32 tests for scientist tools
│   └── test_mcp_tools.py           # 49 tests for MCP tool functions
│
└── docs/
    ├── README.md                   # This file
    ├── architecture.md             # System architecture explanation
    ├── mcp_tool_catalog.md         # Input/output spec for all 10 MCP tools
    └── demo_script.md              # Step-by-step presentation walkthrough
```

---

## ⚠️ Important Architectural Note — Read Before Editing Agents

<br>

**The Supervisor Agent has ZERO tools attached to it.** This is not an oversight — it's a hard requirement enforced by CrewAI itself.

<br>

| | What You Might Expect | What's Actually True |
|---|---|---|
| Supervisor's 5 tools | Attached directly to the agent, like the other two agents | **Never attached** — CrewAI raises `"Manager agent should not have tools"` if you try |
| Where the 5 tools run instead | — | Called as plain Python functions **directly inside `app.py`**, before and after `crew.kickoff()` |
| Why | CrewAI's `Process.hierarchical` mode requires the `manager_agent` to only delegate — it auto-injects its own internal delegation tools and refuses to coexist with custom ones | This is a framework-level constraint, confirmed directly against CrewAI's own source code |

<br>

If you ever try to re-add `tools=[...]` to `build_supervisor_agent()`, the app will fail immediately on every single run with the exact error above.

---

## Agents

| Agent | Role | Tools | Delegation |
|---|---|---|---|
| **Supervisor Agent** | Classifies requests, builds work plans, delegates, validates final output | **0 tools on the agent itself** — its 5 tools are called directly from `app.py` | `allow_delegation: true` |
| **Data Analyst Agent** | Profiles data, suggests KPIs, validates SQL, designs dashboards | 5 function + 7 MCP tools | `allow_delegation: false` |
| **Data Scientist Agent** | Classifies ML problems, engineers features, detects risks, plans pipelines | 5 function + 4 MCP tools | `allow_delegation: false` |

---

## Function Tools (15 total)

**Important calling convention:** every function below is wrapped with CrewAI's `@tool(...)` decorator, which turns it into a `Tool` object — not a plain function. If you ever call one manually (e.g. for testing), you must use `.run(param_name=value)` with keyword arguments, not `function_name(value)` directly. Calling it the plain way raises `TypeError: 'Tool' object is not callable`.

```python
# Wrong — raises TypeError
classify_user_request("some text")

# Correct
classify_user_request.run(user_message="some text")
```

### Supervisor Tools *(called directly from `app.py`, not by the agent)*
| Tool | Purpose |
|---|---|
| `classify_user_request` | Maps request to: analytics, data_science, sql, dashboard, data_quality, mixed |
| `create_agent_work_plan` | Generates step-by-step delegation plan |
| `summarize_chat_history` | Compresses history to save context window |
| `validate_final_response_structure` | Checks all 10 sections are present |
| `estimate_context_usage` | Estimates token usage (1 token ≈ 4 chars) |

### Analyst Tools
| Tool | Purpose |
|---|---|
| `profile_dataframe` | Row count, columns, types, nulls, duplicates, samples |
| `suggest_kpi_metrics` | Domain + column-aware KPI suggestions |
| `generate_dashboard_layout` | Dashboard sections, charts, filters, drill-downs |
| `validate_sql_safety` | Blocks DELETE/DROP/UPDATE, warns on SELECT * and missing LIMIT |
| `explain_query_result` | Converts metric + trend + % change into business language |

### Scientist Tools
| Tool | Purpose |
|---|---|
| `recommend_ml_problem_type` | Maps goal to: classification, regression, clustering, forecasting, anomaly |
| `suggest_feature_engineering` | Time, user, transaction, behavioural feature ideas per column type |
| `detect_ml_data_risks` | Class imbalance, leakage candidates, outliers, time-split requirement |
| `recommend_evaluation_metrics` | Context-aware metric selection per problem type |
| `create_ml_pipeline_plan` | 9-stage end-to-end ML pipeline with tools, risks, owners |

---

## MCP Tools (10 total)

*(These are plain Python functions, NOT wrapped with `@tool` — call them normally, no `.run()` needed.)*

| # | Tool | Used By |
|---|---|---|
| 1 | `mcp_profile_csv` | Analyst, Scientist |
| 2 | `mcp_run_duckdb_query` | Analyst |
| 3 | `mcp_validate_sql` | Supervisor, Analyst |
| 4 | `mcp_detect_data_quality_issues` | Analyst, Scientist |
| 5 | `mcp_generate_kpi_catalog` | Supervisor, Analyst |
| 6 | `mcp_recommend_ml_use_cases` | Supervisor, Scientist |
| 7 | `mcp_feature_engineering_suggestions` | Scientist |
| 8 | `mcp_anomaly_detection_summary` | Analyst, Scientist |
| 9 | `mcp_create_data_dictionary` | Supervisor, Analyst |
| 10 | `mcp_generate_report_markdown` | Supervisor |

Full input/output spec for each: see `docs/mcp_tool_catalog.md`.

---

## Safety Controls

- **SQL**: All queries validated before execution — DELETE/DROP/UPDATE/ALTER/INSERT blocked at regex + AST level
- **File access**: All file reads resolved to absolute paths and checked against `sample_data/` — prevents `../../etc/passwd` traversal
- **File size**: 50 MB limit enforced in MCP tools
- **File type**: Only `.csv` files accepted in file reading tools
- **Error handling**: Raw tracebacks never shown to users — always caught and shown as clean messages
- **Result capping**: DuckDB queries capped at 500 rows to protect context window
- **No shell execution**: No `subprocess`, `os.system`, or `eval` calls anywhere

---

## Known Issues Fixed During Development

<br>

This project went through real debugging. Documenting these here so the fixes aren't accidentally reversed later.

<br>

| # | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | `TypeError: 'Tool' object is not callable` when testing tools manually | `@tool(...)` wraps functions into `Tool` objects, not plain functions | Use `.run(param=value)` instead of calling directly |
| 2 | `ModuleNotFoundError: No module named 'mcp_server'` when running `python mcp_server/server.py` | Running a script inside a package only adds that script's own folder to `sys.path`, not the project root | Run as a module instead: `python -m mcp_server.server` |
| 3 | `pip install mcp crewai-tools'[mcp]'` auto-install prompt fails with a parse error | CrewAI's auto-installer builds a malformed string with stray quotes, passed to a resolver that isn't a shell | Install both packages manually: `pip install mcp` then `pip install "crewai-tools[mcp]"` |
| 4 | `Exception: Manager agent should not have tools` | CrewAI forbids any `manager_agent` in `Process.hierarchical` from holding custom tools | Removed all tools from `build_supervisor_agent()`; the 5 supervisor tools are now called directly in `app.py` as pre/post-processing steps around `crew.kickoff()` |
| 5 | `st.session_state has no attribute "activity_log"` during crew execution | CrewAI's `step_callback` runs on a background `ThreadPoolExecutor` thread — Streamlit's session state is only accessible from the main thread | Added a thread-safe buffer (`buffer_event`, `buffer_tool_call`, `buffer_delegation`) that background threads write to safely; `flush_event_buffer()` drains it into `st.session_state` on the main thread after `kickoff()` returns |
| 6 | `recommend_ml_problem_type` misclassified some goals (e.g. regression goal returned "classification") | The keyword `"will"` was too generic and matched classification before more specific rules could fire; forecasting kewords were checked after regression, so `"revenue"` in a forecasting goal matched regression first | Removed `"will"` from classification keywords; reordered rules so forecasting is checked before regression |
| 7 | `KeyError: 'group'` in feature engineering tests | Test file used the wrong dict key — the actual function returns `"category"`, not `"group"` | Fixed the 3 affected test assertions to use `g["category"]` |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_supervisor_tools.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=function_tools --cov=mcp_server/tools -v
```

**Current test counts** (verified directly from source, not estimated):

| File | Test Count |
|---|---|
| `test_supervisor_tools.py` | 27 |
| `test_analyst_tools.py` | 31 |
| `test_scientist_tools.py` | 32 |
| `test_mcp_tools.py` | 49 |
| **Total** | **139** |

---

## Demo Prompt

```
Analyze the events_sample.csv file. First profile the dataset, then identify
data quality issues, suggest dashboard KPIs, recommend ML use cases, suggest
feature engineering ideas, and provide a final implementation plan.
```

**Expected flow:**
1. `app.py` calls `classify_user_request`, `create_agent_work_plan`, `estimate_context_usage` directly (Python, not agent-invoked)
2. Supervisor delegates based on the pre-computed classification
3. Analyst → `mcp_profile_csv` → `mcp_detect_data_quality_issues` → `mcp_generate_kpi_catalog`
4. Scientist → `mcp_recommend_ml_use_cases` → `mcp_feature_engineering_suggestions`
5. `app.py` calls `validate_final_response_structure` directly on the result → final 10-section answer shown to user

Full walkthrough with timing: see `docs/demo_script.md`.

---

## Libraries Used

| Purpose | Library |
|---|---|
| Agent framework | `crewai`, `crewai-tools` |
| MCP server | `mcp` (FastMCP) |
| Local LLM | `ollama`, `litellm` |
| UI | `streamlit` |
| Data processing | `pandas`, `numpy` |
| Local SQL | `duckdb` |
| SQL parsing | `sqlglot`, `sqlparse` |
| Data validation | `pandera` |
| ML / Statistics | `scikit-learn`, `scipy` |
| Config | `pyyaml` |