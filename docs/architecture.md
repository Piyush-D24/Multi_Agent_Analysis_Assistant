# System Architecture

## Overview

```
Streamlit UI (app.py)
        │
        ▼
CrewAI Hierarchical Crew
        │
        ▼
Supervisor Agent (manager_agent — 0 tools, delegates only)
        │
        ├──► Data Analyst Agent (5 function tools + 7 MCP tools)
        │
        └──► Data Scientist Agent (5 function tools + 4 MCP tools)
        │
        ▼
MCP Server (analytics_mcp_server — 10 tools, launched as a subprocess)
        │
        ▼
Ollama (local LLM — llama3.2:latest)
```

---

## Why the Supervisor Has No Tools

CrewAI enforces this rule in `Process.hierarchical`:

```python
if manager.tools is not None and len(manager.tools) > 0:
    raise Exception("Manager agent should not have tools")
```

| | Where It Actually Runs |
|---|---|
| `classify_user_request`, `create_agent_work_plan`, `estimate_context_usage` | Called directly in `app.py`, **before** `crew.kickoff()` |
| `validate_final_response_structure` | Called directly in `app.py`, **after** `crew.kickoff()` returns |

The Supervisor Agent itself only delegates — it never calls a tool.

---

## Request Flow

```
1. User sends a message in Streamlit chat
2. app.py calls classify_user_request, create_agent_work_plan,
   estimate_context_usage directly (plain Python, main thread)
3. MCPServerAdapter launches mcp_server/server.py as a subprocess
4. 3 Agents built: Supervisor (no tools), Analyst (12 tools), Scientist (9 tools)
5. One Task built from tasks.yaml, with results from step 2 injected into it
6. Crew.kickoff() runs — Supervisor delegates to Analyst and/or Scientist
7. app.py calls validate_final_response_structure directly on the result
8. Final 10-section markdown shown in chat
```

---

## Threading Model (Important Fix)

CrewAI runs `step_callback` on a background `ThreadPoolExecutor` thread, not Streamlit's main thread. `st.session_state` can only be touched from the main thread.

| | Old (Broken) | Fixed |
|---|---|---|
| `step_callback` writes to | `st.session_state` directly | A thread-safe buffer (`buffer_event`, `buffer_tool_call`, `buffer_delegation`) |
| When Streamlit sees the updates | Immediately → crashes | After `kickoff()` returns, via `flush_event_buffer()` on the main thread |

```
Background thread          Main thread
     │                           │
step_callback()                 │
     │                           │
buffer_event(...)  ← safe        │
     │                           │
     └──── kickoff() returns ───►│
                                 │
                          flush_event_buffer()
                                 │
                          st.session_state updated
```

---

## MCP Connection

```
app.py ──StdioServerParameters──► subprocess: python -m mcp_server.server
       ◄──────tool discovery────── FastMCP lists all @mcp.tool() functions
       ──────tool call───────────► executes function, returns dict
       ◄──────result───────────── injected into agent's LLM context
```

---

## Security Layers

| Layer | Mechanism |
|---|---|
| SQL | Regex block-list + sqlglot AST check → SELECT-only |
| File access | Path resolved + checked against `sample_data/` → blocks `../` traversal |
| File type/size | `.csv` only, 50 MB max (MCP tools) |
| Errors | Caught everywhere, shown as clean messages — no raw tracebacks |