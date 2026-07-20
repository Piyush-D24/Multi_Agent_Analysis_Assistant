import json
import sys
import threading
import time
from pathlib import Path
 
# ── Add project root to sys.path so imports resolve correctly ─
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import yaml
from crewai import Crew, LLM, Process, Task

from agents.data_analyst_agent import build_data_analyst_agent
from agents.data_scientist_agent import build_data_scientist_agent
from agents.supervisor_agent import build_supervisor_agent
from function_tools.supervisor_tools import (
    classify_user_request,
    create_agent_work_plan,
    estimate_context_usage,
    validate_final_response_structure,
)

st.set_page_config(
    page_title="Multi-Agent Analytics Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_DIR   = ROOT / "config"
SAMPLE_DIR   = ROOT / "mcp_server" / "sample_data"
MCP_SERVER   = ROOT / "mcp_server" / "server.py"

AVAILABLE_MODELS = [
    "ollama/llama3.2:latest",
    "ollama/mistral",
    "ollama/mixtral",
    "ollama/gemma2",
    "ollama/deepseek-r1",
    "ollama/phi3",
]

SAMPLE_FILES = [f.name for f in SAMPLE_DIR.glob("*.csv")]

AGENT_TOOLS_MAP = {
    "Supervisor Agent":       ["classify_user_request", "create_agent_work_plan",
                               "summarize_chat_history", "validate_final_response_structure",
                               "estimate_context_usage"],
    "Data Analyst Agent":     ["profile_dataframe", "suggest_kpi_metrics",
                               "generate_dashboard_layout", "validate_sql_safety",
                               "explain_query_result"],
    "Data Scientist Agent":   ["recommend_ml_problem_type", "suggest_feature_engineering",
                               "detect_ml_data_risks", "recommend_evaluation_metrics",
                               "create_ml_pipeline_plan"],
}

MCP_TOOLS_LIST = [
    "mcp_profile_csv",
    "mcp_run_duckdb_query",
    "mcp_validate_sql",
    "mcp_detect_data_quality_issues",
    "mcp_generate_kpi_catalog",
    "mcp_recommend_ml_use_cases",
    "mcp_feature_engineering_suggestions",
    "mcp_anomaly_detection_summary",
    "mcp_create_data_dictionary",
    "mcp_generate_report_markdown",
]

_buffer_lock = threading.Lock()
_event_buffer = {
    "activity_log": [],       # list of {icon, text, time} dicts
    "tool_call_count": 0,     # int — incremented per tool call
    "delegation_trace": [],   # list of agent role strings
}

def reset_event_buffer():
    """Clear the buffer at the START of a crew run (main thread)."""
    with _buffer_lock:
        _event_buffer["activity_log"] = []
        _event_buffer["tool_call_count"] = 0
        _event_buffer["delegation_trace"] = []

def buffer_event(icon: str, text: str):
    """
    Thread-safe append — called from CrewAI's background threads.
    Does NOT touch st.session_state. Safe to call from any thread.
    """
    with _buffer_lock:
        _event_buffer["activity_log"].append({
            "icon": icon,
            "text": text,
            "time": time.strftime("%H:%M:%S"),
        })

def buffer_tool_call():
    """Thread-safe increment of the tool call counter."""
    with _buffer_lock:
        _event_buffer["tool_call_count"] += 1

def buffer_delegation(agent_role: str):
    """Thread-safe append to the delegation trace."""
    with _buffer_lock:
        if agent_role and agent_role not in _event_buffer["delegation_trace"]:
            _event_buffer["delegation_trace"].append(agent_role)

def flush_event_buffer():
    """
    Drain the buffer into st.session_state.
    MUST be called from the MAIN thread only — call this right after
    crew.kickoff() returns (success or failure), never from inside
    step_callback or task_callback.
    """
    with _buffer_lock:
        buffered_log = list(_event_buffer["activity_log"])
        buffered_tool_count = _event_buffer["tool_call_count"]
        buffered_delegation = list(_event_buffer["delegation_trace"])
 
    st.session_state.activity_log.extend(buffered_log)
    st.session_state.total_tool_calls += buffered_tool_count
    for agent_role in buffered_delegation:
        if agent_role not in st.session_state.delegation_trace:
            st.session_state.delegation_trace.append(agent_role)

def init_session_state():
    defaults = {
        "messages":         [],       # chat history [{role, content}]
        "activity_log":     [],       # timeline events [{icon, text, time}]
        "selected_model":   AVAILABLE_MODELS[0],
        "mcp_enabled":      True,
        "selected_file":    SAMPLE_FILES[0] if SAMPLE_FILES else "",
        "crew_metrics":     {},       # last run metrics
        "delegation_trace": [],       # which agent did what
        "context_estimate": 0,
        "total_runs":       0,
        "total_tool_calls": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
 
init_session_state()

@st.cache_data
def load_agents_config() -> dict:
    with open(CONFIG_DIR / "agents.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_tasks_config() -> dict:
    with open(CONFIG_DIR / "tasks.yaml", "r") as f:
        return yaml.safe_load(f)

def log_activity(icon: str, text: str):
    """Append an event to the activity log and rerender sidebar."""
    st.session_state.activity_log.append({
        "icon": icon,
        "text": text,
        "time": time.strftime("%H:%M:%S"),
    })

def step_callback(agent_output):
    """
    Called by CrewAI after every agent step.
 
    IMPORTANT: This function runs on a CrewAI background thread
    (ThreadPoolExecutor), NOT the main Streamlit thread. It must
    NEVER touch st.session_state directly — that will crash with
    "st.session_state has no attribute ...". Instead, it writes to
    the thread-safe buffer defined above, which gets flushed into
    st.session_state later, on the main thread, by flush_event_buffer().
 
    agent_output is a crewai.agents.agent_builder.output_parser.AgentAction
    or AgentFinish object depending on the CrewAI version.
    We guard all attribute access so a version change doesn't crash the app.
    """
    try:
        # -- Identify which agent just ran --
        agent_role = getattr(agent_output, "agent", "Unknown Agent")

        # -- What did it do? --
        tool_used  = getattr(agent_output, "tool", None)
        tool_input = getattr(agent_output, "tool_input", None)
        output     = getattr(agent_output, "output", None)
        thought    = getattr(agent_output, "thought", None)

        if thought:
            buffer_event("🧠", f"**{agent_role}** — Thinking: {str(thought)[:120]}...")

        if tool_used:
            buffer_event("🔧", f"**{agent_role}** — Calling tool: `{tool_used}`")
            if tool_input:
                buffer_event("📥", f"Tool input: `{str(tool_input)[:100]}`")
            buffer_tool_call()

            # Track MCP vs function tool
            tool_lower = str(tool_used).lower()
            if "mcp" in tool_lower:
                buffer_event("🌐", f"MCP tool call: `{tool_used}`")
            else:
                buffer_event("⚡", f"Function tool call: `{tool_used}`")

        if output and not tool_used:
            buffer_event("✅", f"**{agent_role}** — Step complete.")

        # Track delegation (thread-safe, buffered)
        buffer_delegation(str(agent_role))

    except Exception:
        # Never let a callback crash the main flow
        buffer_event("⚠️", "Step callback encountered a minor issue (non-fatal).")

def build_llm(model_name: str) -> LLM:
    """
    Build a CrewAI LLM object pointing to a local Ollama model.
 
    CrewAI uses LiteLLM under the hood. The 'ollama/' prefix tells
    LiteLLM to route to the local Ollama server at localhost:11434.
    """
    return LLM(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,       # Low temp → more deterministic, better for structured output
        timeout=300,           # 5 min timeout for long reasoning chains
        max_tokens=4096,
    )

def build_manager_task(
    supervisor_agent,
    tasks_config: dict,
    user_prompt: str,
    chat_history: str,
    selected_file: str,
    classification_json: str = "",
    work_plan_json: str = "",
) -> Task:
    """
    Build the single manager task with the user's prompt injected.
 
    The task description template uses {chat_history}, {user_prompt},
    and {selected_file} placeholders defined in tasks.yaml.
 
    classification_json and work_plan_json are the PRE-COMPUTED outputs
    of classify_user_request and create_agent_work_plan — called directly
    in run_crew() rather than by the agent, since the Supervisor can no
    longer hold tools once it acts as manager_agent (see
    agents/supervisor_agent.py for why). Injecting them into the task
    description means the LLM still receives this classification/plan
    context, even though it didn't call the tools itself.
    """
    task_cfg = tasks_config["analytics_manager_task"]

    description = task_cfg["description"].format(
        chat_history=chat_history or "No previous messages.",
        user_prompt=user_prompt,
    )

    if selected_file:
        description += (
            f"\n\n=== SELECTED DATASET ===\n"
            f"The user has selected: `{selected_file}`\n"
            f"This file is in mcp_server/sample_data/. "
            f"Pass this filename to any profiling or MCP tools that need it."
        )

    if classification_json:
        description += (
            f"\n\n=== PRE-COMPUTED REQUEST CLASSIFICATION ===\n"
            f"{classification_json}\n"
            f"(This was already computed for you by classify_user_request. "
            f"Use it to decide delegation — do not re-classify.)"
        )

    if work_plan_json:
        description += (
            f"\n\n=== PRE-COMPUTED WORK PLAN ===\n"
            f"{work_plan_json}\n"
            f"(This was already computed for you by create_agent_work_plan. "
            f"Follow these steps when delegating.)"
        )

    return Task(
        description=description,
        expected_output=task_cfg["expected_output"],
        agent=supervisor_agent,
    )

def format_chat_history(messages: list) -> str:
    """
    Convert session messages list to a plain string for the task prompt.
    Only includes the last 6 messages to manage context window.
    """
    if not messages:
        return ""
    recent = messages[-6:]
    lines = []
    for msg in recent:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        content = str(msg["content"])[:300]  # Truncate long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def run_crew(user_prompt: str) -> str:
    """
    Build and run the full CrewAI hierarchical crew.
 
    Flow:
      1. Load configs
      2. Build LLM
      3. Try to connect MCP server (graceful fallback if unavailable)
      4. Build agents with tools
      5. Build task
      6. Build crew with Process.hierarchical
      7. kickoff() and return result
    """
    log_activity("🚀", "Starting crew run...")
    reset_event_buffer()
    st.session_state.delegation_trace = []

    agents_config = load_agents_config()
    tasks_config  = load_tasks_config()
    llm           = build_llm(st.session_state.selected_model)
    chat_history  = format_chat_history(st.session_state.messages[:-1])

    log_activity("🔧", "Calling tool: classify_user_request")
    classification_json = classify_user_request.run(user_message=user_prompt)
    log_activity("📥", f"Classification: {classification_json[:100]}")

    log_activity("🔧", "Calling tool: create_agent_work_plan")
    classification = json.loads(classification_json)
    work_plan_json = create_agent_work_plan.run(
        user_message=user_prompt, intent=classification.get("intent", "mixed")
    )

    log_activity("🔧", "Calling tool: estimate_context_usage")
    total_text = user_prompt + chat_history
    context_json = estimate_context_usage.run(
        text_content=total_text, context_window_size=8192
    )
    context_result = json.loads(context_json)
    st.session_state.context_estimate = context_result["estimated_input_tokens"]

    st.session_state.total_tool_calls += 3

    mcp_tools_for_analyst   = []
    mcp_tools_for_scientist = []

    if st.session_state.mcp_enabled:
        try:
            from mcp import StdioServerParameters
            from crewai_tools import MCPServerAdapter

            server_params = StdioServerParameters(
                command=sys.executable,        # same Python that runs app.py
                args=[str(MCP_SERVER)],
                env=None,
            )

            log_activity("🌐", "Connecting to MCP server...")

            adapter = MCPServerAdapter(server_params)
            all_mcp_tools = adapter.tools  # list of CrewAI Tool objects

            analyst_tool_names = {
                "tool_mcp_profile_csv",
                "tool_mcp_run_duckdb_query",
                "tool_mcp_validate_sql",
                "tool_mcp_detect_data_quality_issues",
                "tool_mcp_generate_kpi_catalog",
                "tool_mcp_create_data_dictionary",
                "tool_mcp_generate_report_markdown",
            }
            scientist_tool_names = {
                "tool_mcp_recommend_ml_use_cases",
                "tool_mcp_feature_engineering_suggestions",
                "tool_mcp_anomaly_detection_summary",
                "tool_mcp_generate_report_markdown",
            }

            for tool in all_mcp_tools:
                tool_name = getattr(tool, "name", "")
                if tool_name in analyst_tool_names:
                    mcp_tools_for_analyst.append(tool)
                if tool_name in scientist_tool_names:
                    mcp_tools_for_scientist.append(tool)
 
            log_activity("✅", f"MCP server connected. {len(all_mcp_tools)} tools loaded.")
 
        except ImportError:
            log_activity("⚠️", "crewai-tools[mcp] not installed. Running without MCP tools.")
            log_activity("💡", "Run: pip install 'crewai-tools[mcp]' to enable MCP.")
            adapter = None
        except Exception as e:
            log_activity("⚠️", f"MCP server unavailable: {str(e)[:80]}. Using function tools only.")
            adapter = None
    else:
        adapter = None
        log_activity("ℹ️", "MCP server disabled by user. Using function tools only.")

    log_activity("🤖", "Building Supervisor Agent...")
    supervisor = build_supervisor_agent(agents_config, llm, step_callback)
 
    log_activity("📊", "Building Data Analyst Agent...")
    analyst = build_data_analyst_agent(
        agents_config, llm, step_callback, mcp_tools_for_analyst
    )

    log_activity("🔬", "Building Data Scientist Agent...")
    scientist = build_data_scientist_agent(
        agents_config, llm, step_callback, mcp_tools_for_scientist
    )

    log_activity("📋", "Building manager task...")
    manager_task = build_manager_task(
        supervisor_agent=supervisor,
        tasks_config=tasks_config,
        user_prompt=user_prompt,
        chat_history=chat_history,
        selected_file=st.session_state.selected_file,
        classification_json=classification_json,
        work_plan_json=work_plan_json,
    )

    log_activity("⚙️", "Assembling hierarchical crew...")
    crew = Crew(
        agents=[analyst, scientist],
        tasks=[manager_task],
        manager_agent=supervisor,
        process=Process.hierarchical,
        verbose=True,
        memory=False,          # Disabled for local Ollama (needs embeddings)
        planning=False,        # Supervisor handles planning via tools
        step_callback=step_callback,
        task_callback=lambda output: buffer_event(
            "📦", f"Task output received ({len(str(output))} chars)"
        ),
    )
    log_activity("▶️", "Crew kickoff — agents are working...")
    start_time = time.time()

    try:
        result = crew.kickoff()
        elapsed = round(time.time() - start_time, 1)

        flush_event_buffer()

        st.session_state.crew_metrics = {
            "elapsed_seconds": elapsed,
            "agents_used": len(st.session_state.delegation_trace),
            "total_tool_calls": st.session_state.total_tool_calls,
        }
        st.session_state.total_runs += 1

        log_activity("🎉", f"Crew completed in {elapsed}s.")
        log_activity("🔧", "Calling tool: validate_final_response_structure")

        validation_json = validate_final_response_structure.run(response_text=str(result))
        validation = json.loads(validation_json)

        st.session_state.total_tool_calls += 1

        if validation["is_valid"]:
            log_activity("✅", "Response structure validated — all 10 sections present.")
        else:
            log_activity(
                "⚠️",
                f"Response missing sections: {', '.join(validation['missing_sections'])}"
            )

        return str(result)

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        # Flush whatever progress was buffered before the failure occurred,
        # so the Activity Timeline still shows what happened up to that point.
        flush_event_buffer()
        log_activity("❌", f"Crew run failed after {elapsed}s: {str(e)[:100]}")
        raise

    finally:
        if adapter:
            try:
                adapter.__exit__(None, None, None)
            except Exception:
                pass

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.subheader("🧠 Ollama Model")
        st.session_state.selected_model = st.selectbox(
            "Select model",
            AVAILABLE_MODELS,
            index=0,
            help="Model must be pulled via 'ollama pull <model>' first.",
        )

        st.subheader("📂 Dataset")
        if SAMPLE_FILES:
            st.session_state.selected_file = st.selectbox(
                "Active dataset",
                SAMPLE_FILES,
                help="This file will be passed to profiling and MCP tools.",
            )
        else:
            st.warning("No CSV files found in mcp_server/sample_data/")

        st.subheader("🌐 MCP Server")
        st.session_state.mcp_enabled = st.toggle(
            "Enable MCP tools",
            value=st.session_state.mcp_enabled,
            help="Disable to use only local function tools (faster but fewer capabilities).",
        )
        if st.session_state.mcp_enabled:
            st.success("MCP server: enabled")
        else:
            st.info("MCP server: disabled")

        st.divider()

        st.subheader("🤖 Agents")
        for agent_name, tools in AGENT_TOOLS_MAP.items():
            with st.expander(agent_name):
                for t in tools:
                    st.markdown(f"- `{t}`")

        st.subheader("🌐 MCP Tools")
        with st.expander("10 available tools"):
            for t in MCP_TOOLS_LIST:
                st.markdown(f"- `{t}`")

        st.divider()

        st.subheader("📈 Session Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Total Runs",       st.session_state.total_runs)
        col2.metric("Tool Calls",       st.session_state.total_tool_calls)

        if st.session_state.context_estimate:
            usage_pct = round(st.session_state.context_estimate / 8192 * 100, 1)
            st.metric("Est. Context Tokens", st.session_state.context_estimate)
            st.progress(min(usage_pct / 100, 1.0), text=f"{usage_pct}% of 8192")

        if st.session_state.crew_metrics:
            m = st.session_state.crew_metrics
            st.metric("Last Run Time", f"{m.get('elapsed_seconds', 0)}s")

        st.divider()

        if st.session_state.delegation_trace:
            st.subheader("🔀 Delegation Trace")
            for i, agent in enumerate(st.session_state.delegation_trace, 1):
                st.markdown(f"`{i}.` {agent}")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages         = []
            st.session_state.activity_log     = []
            st.session_state.delegation_trace = []
            st.session_state.crew_metrics     = {}
            st.session_state.total_tool_calls = 0
            st.rerun()

def render_activity_timeline():
    st.subheader("📡 Activity Timeline")

    if not st.session_state.activity_log:
        st.info("No activity yet. Send a message to start the crew.")
        return

    events = list(reversed(st.session_state.activity_log[-30:]))
    for event in events:
        st.markdown(
            f"`{event['time']}` {event['icon']} {event['text']}",
            unsafe_allow_html=False,
        )

def render_chat():
    st.subheader("💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if not st.session_state.messages:
        st.markdown("**Try one of these prompts:**")
        example_prompts = [
            "Analyze events_sample.csv — profile it, find data quality issues, suggest KPIs, and recommend ML use cases.",
            "What SQL query would show me the top 5 event types by count? Validate the query first.",
            "Suggest feature engineering ideas for customers_sample.csv and build an ML pipeline plan for churn prediction.",
            "Generate a complete KPI catalog for an ecommerce platform using transactions_sample.csv.",
        ]
        cols = st.columns(2)
        for i, prompt in enumerate(example_prompts):
            if cols[i % 2].button(prompt[:60] + "...", key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    if user_input := st.chat_input("Ask the analytics crew anything..."):
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Agents are working — check the Activity Timeline →"):
                log_activity("💬", f"User: {user_input[:80]}...")
                try:
                    result = run_crew(user_input)
                    st.markdown(result)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result}
                    )

                except Exception as e:
                    error_msg = (
                        f"**The crew encountered an error.**\n\n"
                        f"**What to check:**\n"
                        f"- Is Ollama running? Run `ollama serve` in a terminal.\n"
                        f"- Is the model pulled? Run `ollama pull llama3.1`.\n"
                        f"- Check the Activity Timeline for the exact failure point.\n\n"
                    )
                    st.error(error_msg)
                    with st.expander("🔍 Debug details (expand to see error)"):
                        st.code(str(e)[:500])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
        st.rerun()

def main():
    st.title("🤖 Multi-Agent Analytics Assistant")
    st.caption(
        "Powered by CrewAI · Ollama · FastMCP | "
        f"Model: `{st.session_state.selected_model}` | "
        f"Dataset: `{st.session_state.selected_file}`"
    )
    st.divider()

    render_sidebar()

    chat_col, timeline_col = st.columns([2, 1])

    with chat_col:
        render_chat()

    with timeline_col:
        render_activity_timeline()
        if st.session_state.activity_log:
            if st.button("🔄 Refresh Timeline", use_container_width=True):
                st.rerun()

if __name__ == "__main__":
    main()