from typing import Any, Callable, Dict, List, Optional
from crewai import Agent, LLM
from crewai.tools import BaseTool

from function_tools.analyst_tools import (
    explain_query_result,
    generate_dashboard_layout,
    profile_dataframe,
    suggest_kpi_metrics,
    validate_sql_safety,
)

def build_data_analyst_agent(
    agents_config: Dict[str, Any],
    llm: LLM,
    step_callback: Optional[Callable[[Any], None]] = None,
    extra_tools: Optional[List[BaseTool]] = None,
) -> Agent:
    """
    Build and return the Data Analyst Agent with function tools.

    The analyst handles data profiling, KPI suggestion, SQL validation,
    dashboard layout design, and business result explanation.

    Args:
        agents_config:  Loaded agents.yaml dict.
        llm:            Pre-built CrewAI LLM object (Ollama).
        step_callback:  Optional Streamlit activity timeline callback.
        extra_tools:    Additional tools injected at runtime — used to
                        pass MCP server tools from MCPServerAdapter.
                        These are appended AFTER the base function tools.
 
    Returns:
        A fully configured CrewAI Agent with tools attached.
    """
    cfg = agents_config["data_analyst_agent"]

    base_tools = [
        profile_dataframe,
        suggest_kpi_metrics,
        generate_dashboard_layout,
        validate_sql_safety,
        explain_query_result,
    ]

    all_tools = base_tools + (extra_tools or [])

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm,
        tools=all_tools,
        verbose=bool(cfg.get("verbose", True)),
        allow_delegation=bool(cfg.get("allow_delegation", False)),
        max_iter=int(cfg.get("max_iter", 5)),
        max_retry_limit=int(cfg.get("max_retry_limit", 2)),
        respect_context_window=True,
        use_system_prompt=True,
        step_callback=step_callback,
    )