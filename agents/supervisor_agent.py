from typing import Any, Callable, Dict, Optional
from crewai import Agent, LLM

def build_supervisor_agent(
    agents_config: Dict[str, Any],
    llm: LLM,
    step_callback: Optional[Callable[[Any], None]] = None,
) -> Agent:
    """
    Build and return the Supervisor Agent — WITHOUT tools.

    The supervisor is the manager_agent in the CrewAI hierarchical crew.
    Per CrewAI's constraint, it can ONLY delegate — never call tools
    directly. Its 5 function tools are invoked separately, directly in
    app.py, as pre-processing (classify + plan + estimate context) and
    post-processing (validate final response) steps around crew.kickoff().
    Returns:
        A fully configured CrewAI Agent with NO tools attached.
    """
    cfg = agents_config["supervisor_agent"]

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm,
        verbose=bool(cfg.get("verbose", True)),
        allow_delegation=bool(cfg.get("allow_delegation", True)),
        max_iter=int(cfg.get("max_iter", 6)),
        max_retry_limit=int(cfg.get("max_retry_limit", 2)),
        respect_context_window=True,
        use_system_prompt=True,
        step_callback=step_callback,
    )