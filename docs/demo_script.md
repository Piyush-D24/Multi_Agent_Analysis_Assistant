# Demo Script

A step-by-step walkthrough for presenting the Multi-Agent Analytics Assistant.
Estimated total time: **8–10 minutes**.

---

## Before You Start (Setup Checklist)

Run these in order, in a terminal, **before** the audience is watching:

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — confirm the model is pulled
ollama pull llama3.2:latest

# Terminal 3 — sanity-check the MCP server runs standalone
python mcp_server/server.py
# (Ctrl+C once you see it start cleanly — this is just a smoke test)

# Terminal 3 — launch the app
streamlit run app.py
```

Confirm the browser opens to `http://localhost:8501` and the sidebar loads
without errors before you begin presenting.

---

## Part 1 — Show the Architecture (1 minute)

**Say:** "This is a multi-agent analytics system. A Supervisor Agent receives
questions and delegates to two specialists — a Data Analyst and a Data
Scientist — who use both local Python tools and a separate MCP server for
their work."

**Do:** Point at the sidebar — expand the **Agents** section and the
**MCP Tools** section so the audience sees the tool inventory before anything
runs.

---

## Part 2 — Select a Dataset (30 seconds)

**Say:** "I'll pick a sample dataset for the agents to work with."

**Do:** In the sidebar, select `events_sample.csv` from the dataset dropdown.
Confirm **MCP Server** toggle is switched **on**.

---

## Part 3 — Run the Main Demo Prompt (3–4 minutes)

**Do:** Type this exact prompt into the chat input:

```
Analyze the events_sample.csv file. First profile the dataset, then identify
data quality issues, suggest dashboard KPIs, recommend ML use cases, suggest
feature engineering ideas, and provide a final implementation plan.
```

**While it runs, narrate the Activity Timeline on the right:**

| What appears in the timeline | What to say |
|---|---|
| `🧠 Supervisor Agent — Thinking...` | "The Supervisor is reasoning about how to break this down." |
| `🔧 Calling tool: classify_user_request` | "First it classifies the intent using a local function tool." |
| `🌐 MCP tool call: mcp_profile_csv` | "Now it's delegated to the Analyst, which is calling the MCP server to profile the actual CSV file." |
| `🌐 MCP tool call: mcp_detect_data_quality_issues` | "Same MCP server, different tool — checking for missing values, duplicates, outliers." |
| `🌐 MCP tool call: mcp_recommend_ml_use_cases` | "Now the Data Scientist agent has taken over, matching the dataset columns against known ML use cases." |
| `✅ Crew completed in Xs` | "And here's the final structured answer." |

**Do:** Once the response appears, scroll through it and point out the 10
sections are all present — this is what `validate_final_response_structure`
enforces before the Supervisor is allowed to return an answer.

---

## Part 4 — Demonstrate Safety Controls (2 minutes)

This is the part that shows the system isn't just "an LLM wrapper" — it has
real guardrails.

**Do:** Type this prompt:

```
Run this SQL query on the events data: DELETE FROM events WHERE status = 'failed'
```

**Say while it processes:** "Watch what happens — the agent has a SQL
validation tool that runs before anything is executed."

**Expected result:** The agent should refuse to run the query and explain
that `DELETE` is a blocked keyword. Point to the `validate_sql_safety` /
`mcp_validate_sql` tool call in the timeline as proof this was caught by code,
not just the LLM being cautious.

**Optional second safety demo:** Show the file access restriction by pointing
to the code in `csv_profile_tools.py` — explain that even if a prompt tried
to reference a path outside `sample_data/`, `_safe_load_csv` would block it
before pandas ever touched the file.

---

## Part 5 — Show a Second Dataset for Variety (1–2 minutes)

**Do:** Switch the dataset dropdown to `customers_sample.csv`. Type:

```
This dataset has a churn column. Recommend the right ML problem type,
the features I should engineer, and the evaluation metrics I should use.
```

**Say while it runs:** "This shows the Data Scientist agent's dedicated
tools — problem type classification, feature engineering suggestions tied
to actual column names in this file, not generic advice, and evaluation
metrics chosen based on the problem type."

**Do:** When the response comes back, point out that `recommend_ml_problem_type`
correctly identifies this as `classification` because of the word "churn" in
both the prompt and the column name — and that the suggested metrics include
Recall/Precision/F1, not RMSE (which would be wrong for a classification task).

---

## Part 6 — Show the Sidebar Metrics (30 seconds)

**Say:** "Every run tracks metrics — total agent runs, total tool calls, and
an estimate of context window usage, which matters because we're running a
local Ollama model with a limited context window, not a cloud API."

**Do:** Point at the **Session Metrics** panel and the **Delegation Trace**
showing which agents fired in what order.

---

## Part 7 — Wrap-Up Talking Points (1 minute)

Close with these points, adjusting based on audience:

- "All 30 files are organized by responsibility — agents, function tools,
  MCP tools, and the Streamlit layer are fully decoupled."
- "The MCP server can be tested completely independently of CrewAI or
  Streamlit — it's a standard MCP server any MCP client could connect to."
- "115 unit tests cover the function tools and MCP tool logic directly,
  without needing Ollama running at all."
- "Safety isn't optional — SQL validation and file path sandboxing happen
  in code, not just via prompting the LLM to behave."

---

## If Something Goes Wrong During the Demo

| Problem | Likely Cause | Quick Fix |
|---|---|---|
| App hangs with no timeline updates | Ollama isn't running | Check Terminal 1 — restart `ollama serve` |
| "Connection refused" error | Wrong Ollama port or model not pulled | Run `ollama pull llama3.1` again |
| MCP tools never fire, only function tools | `MCPServerAdapter` failed silently | Check the Activity Timeline for the `⚠️ MCP server unavailable` message — fall back to explaining function tools instead |
| Response missing sections | LLM ignored task instructions | This is a known LLM reliability limitation — mention that `validate_final_response_structure` exists precisely to catch this, and rerun the prompt |
| Very slow responses (>60s) | Local model on CPU without GPU acceleration | Mention this is expected for local LLM inference — cloud APIs would be faster but this demo intentionally uses zero-cost local infrastructure |

---

## Total Time Budget

| Part | Duration |
|---|---|
| Setup (before audience) | Not counted |
| Part 1 — Architecture | 1 min |
| Part 2 — Dataset selection | 0.5 min |
| Part 3 — Main demo prompt | 3–4 min |
| Part 4 — Safety demo | 2 min |
| Part 5 — Second dataset | 1–2 min |
| Part 6 — Metrics | 0.5 min |
| Part 7 — Wrap-up | 1 min |
| **Total** | **~9–11 min** |