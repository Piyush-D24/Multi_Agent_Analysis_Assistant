
from typing import Any, Callable, Dict, List, Optional

from crewai import Agent, LLM
from crewai.tools import BaseTool

from function_tools.scientist_tools import (
    create_ml_pipeline_plan,
    detect_ml_data_risks,
    recommend_evaluation_metrics,
    recommend_ml_problem_type,
    suggest_feature_engineering,
)

def build_data_scientist_agent(
    agents_config: Dict[str, Any],
    llm: LLM,
    step_callback: Optional[Callable[[Any], None]] = None,
    extra_tools: Optional[List[BaseTool]] = None,
) -> Agent:
    """
    Build and return the Data Scientist Agent with function tools.
 
    The scientist handles ML problem classification, feature engineering,
    data risk detection, evaluation metric selection, and pipeline planning.
 
    Args:
        agents_config:  Loaded agents.yaml dict.
        llm:            Pre-built CrewAI LLM object (Ollama).
        step_callback:  Optional Streamlit activity timeline callback.
        extra_tools:    MCP tools injected from MCPServerAdapter at runtime.
 
    Returns:
        A fully configured CrewAI Agent with tools attached.
    """
    cfg = agents_config["data_scientist_agent"]
 
    base_tools = [
        recommend_ml_problem_type,
        suggest_feature_engineering,
        detect_ml_data_risks,
        recommend_evaluation_metrics,
        create_ml_pipeline_plan,
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