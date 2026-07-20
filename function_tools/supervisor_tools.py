import json
from typing import Any
from crewai.tools import tool

@tool("classify_user_request")
def classify_user_request(user_message: str) -> str:
    """
    Classify the user's request into an analytics intent category.
    Categories:
      analytics      → general data analysis, EDA, trends
      data_science   → ML, AI, predictions, models, clustering
      sql            → SQL queries, data extraction
      dashboard      → dashboards, charts, visualizations, KPIs
      data_quality   → missing values, validation, data issues
      architecture   → system design, pipeline, infrastructure
      mixed          → request spans multiple categories
    Returns a JSON string with: intent, recommended_agent, reason.
    """
    text = user_message.lower()

    # keyword maps per intent
    keyword_map = {
        "sql": ["sql", "query", "select", "join", "where", "table", "database", "duckdb"],
        "dashboard": ["dashboard", "kpi", "chart", "visualization", "report", "metric", "trend"],
        "data_science": ["ml", "machine learning", "predict", "model", "cluster", "forecast",
                         "anomaly", "feature", "train", "classification", "regression", "neural"],
        "data_quality": ["missing", "null", "duplicate", "quality", "clean", "invalid", "outlier",
                         "validation", "check", "issue"],
        "architecture": ["pipeline", "architecture", "infrastructure", "deploy", "kafka", "airflow",
                         "design", "system", "workflow"],
        "analytics": ["analyze", "analysis", "eda", "explore", "profile", "summarize",
                      "insight", "pattern", "distribution"],
    }

    agent_map = {
        "sql": "Data Analyst Agent",
        "dashboard": "Data Analyst Agent",
        "data_quality": "Data Analyst Agent",
        "analytics": "Data Analyst Agent",
        "data_science": "Data Scientist Agent",
        "architecture": "Supervisor Agent",
        "mixed": "Supervisor Agent (delegate to both)",
    }

    scores = {intent: 0 for intent in keyword_map}
    for intent, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text:
                scores[intent] += 1

    top_intents = [k for k, v in scores.items() if v > 0]

    if len(top_intents) == 0:
        # Default to analytics if nothing matches
        intent = "analytics"
        reason = "No specific keywords detected. Defaulting to general analytics."
    elif len(top_intents) == 1:
        intent = top_intents[0]
        reason = f"Matched keywords for '{intent}' category."
    else:
        # Multiple categories matched → mixed
        intent = "mixed"
        reason = f"Request spans multiple categories: {', '.join(top_intents)}."

    result = {
        "intent": intent,
        "recommended_agent": agent_map.get(intent, "Supervisor Agent"),
        "reason": reason,
        "matched_categories": top_intents,
    }
    return json.dumps(result, indent=2)

@tool("create_agent_work_plan")
def create_agent_work_plan(user_message: str, intent: str = "mixed") -> str:
    """
    Generate a step-by-step work plan for the agent crew.

    Based on the user's message and intent classification, decide which
    agents and tools should be used and in what order.

    Returns a JSON string with: intent, steps (list), agents_needed.
    """
    text = user_message.lower()
 
    # Build step list dynamically based on what the user is asking
    steps = []
    agents_needed = []

    # start with context check
    steps.append("Supervisor: Classify the user request using classify_user_request tool.")
    steps.append("Supervisor: Estimate context window usage using estimate_context_usage tool.")

    needs_analyst = any(kw in text for kw in [
        "profile", "csv", "data", "kpi", "dashboard", "sql", "query",
        "analyze", "quality", "missing", "duplicate", "chart", "report"
    ])

    needs_scientist = any(kw in text for kw in [
        "ml", "model", "predict", "forecast", "cluster", "feature",
        "anomaly", "train", "classification", "regression", "pipeline"
    ])

    # If neither matched but intent is mixed or analytics, use analyst as default agent
    if not needs_analyst and not needs_scientist:
        needs_analyst = True

    if needs_analyst:
        agents_needed.append("Data Analyst Agent")
        steps.append("Data Analyst: Call profile_dataframe or mcp_profile_csv to understand the dataset.")
        steps.append("Data Analyst: Call mcp_detect_data_quality_issues to find data problems.")
        steps.append("Data Analyst: Call suggest_kpi_metrics or mcp_generate_kpi_catalog for KPIs.")
        steps.append("Data Analyst: Call generate_dashboard_layout for a visual report structure.")

        if "sql" in text or "query" in text:
            steps.append("Data Analyst: Call validate_sql_safety or mcp_validate_sql to check queries.")

    if needs_scientist:
        agents_needed.append("Data Scientist Agent")
        steps.append("Data Scientist: Call recommend_ml_problem_type to classify the ML use case.")
        steps.append("Data Scientist: Call detect_ml_data_risks to identify risks before training.")
        steps.append("Data Scientist: Call suggest_feature_engineering or mcp_feature_engineering_suggestions.")
        steps.append("Data Scientist: Call recommend_evaluation_metrics for the identified problem type.")
        steps.append("Data Scientist: Call create_ml_pipeline_plan for an end-to-end plan.")

    steps.append("Supervisor: Collect all specialist outputs.")
    steps.append("Supervisor: Call validate_final_response_structure to ensure 10 sections are present.")
    steps.append("Supervisor: Combine all outputs into a final structured markdown response.")
 
    result = {
        "intent": intent,
        "agents_needed": agents_needed if agents_needed else ["Data Analyst Agent"],
        "total_steps": len(steps),
        "steps": steps,
    }
    return json.dumps(result, indent=2)

@tool("summarize_chat_history")
def summarize_chat_history(chat_history: str) -> str:
    """
    Compress a long chat history into a short summary to preserve
    context window space.
 
    Extracts: what the user is building, what has been discussed,
    what was the last question, and what is pending.
 
    Returns a JSON string with: summary, last_user_intent, message_count.
    """
    if not chat_history or not chat_history.strip():
        return json.dumps({
            "summary": "No chat history available. This is the start of the conversation.",
            "last_user_intent": "unknown",
            "message_count": 0,
        }, indent=2)
 
    lines = [line.strip() for line in chat_history.strip().split("\n") if line.strip()]
    message_count = len(lines)

    # Extract last user message
    last_user_line = ""
    for line in reversed(lines):
        if line.upper().startswith("USER:"):
            last_user_line = line.replace("USER:", "").strip()
            break

    # Build a condensed summary from first + last few lines
    if message_count <= 6:
        context_lines = lines
    else:
        context_lines = lines[:3] + ["[... earlier messages ...]"] + lines[-3:]
 
    condensed = " | ".join(context_lines)

    # Try to infer the topic
    full_text = chat_history.lower()
    topics = []
    if any(kw in full_text for kw in ["crewai", "agent", "supervisor"]):
        topics.append("multi-agent system with CrewAI")
    if any(kw in full_text for kw in ["ollama", "llm", "model"]):
        topics.append("local Ollama LLM setup")
    if any(kw in full_text for kw in ["mcp", "server", "tool"]):
        topics.append("MCP server tools")
    if any(kw in full_text for kw in ["streamlit", "ui", "dashboard"]):
        topics.append("Streamlit UI")
    if any(kw in full_text for kw in ["csv", "data", "dataset", "pandas"]):
        topics.append("data analysis and profiling")
    if any(kw in full_text for kw in ["ml", "machine learning", "model", "predict"]):
        topics.append("machine learning")

    topic_summary = (
        "The conversation covers: " + ", ".join(topics) + "."
        if topics else "General analytics conversation."
    )

    result = {
        "summary": f"{topic_summary} {message_count} messages exchanged. Recent context: {condensed[:300]}",
        "last_user_intent": last_user_line[:200] if last_user_line else "Not identified",
        "message_count": message_count,
        "topics_detected": topics,
    }
    return json.dumps(result, indent=2)

@tool("validate_final_response_structure")
def validate_final_response_structure(response_text: str) -> str:
    """
    Check whether the final response contains all 10 required sections.

    Required sections:
      1. Direct Answer
      2. Dataset Summary
      3. Data Quality Findings
      4. Recommended KPIs
      5. Recommended Dashboard
      6. ML Use Cases
      7. Feature Engineering Ideas
      8. Risks and Limitations
      9. Next Steps
      10. Agent Work Summary

    Returns a JSON string with: is_valid, missing_sections, present_sections, score.
    """
    required_sections = [
        "Direct Answer",
        "Dataset Summary",
        "Data Quality Findings",
        "Recommended KPIs",
        "Recommended Dashboard",
        "ML Use Cases",
        "Feature Engineering Ideas",
        "Risks and Limitations",
        "Next Steps",
        "Agent Work Summary",
    ]

    present = []
    missing = []

    text_lower = response_text.lower()

    for section in required_sections:
        # Check for exact heading or close variation
        section_lower = section.lower()
        if (
            section_lower in text_lower
            or section_lower.replace(" ", "_") in text_lower
            or section_lower.replace(" ", "-") in text_lower
        ):
            present.append(section)
        else:
            missing.append(section)

    score = round((len(present) / len(required_sections)) * 100, 1)
    is_valid = len(missing) == 0

    result = {
        "is_valid": is_valid,
        "score_percent": score,
        "sections_present": len(present),
        "sections_missing": len(missing),
        "present_sections": present,
        "missing_sections": missing,
        "recommendation": (
            "Response is complete. Ready to return to user."
            if is_valid
            else f"Add the following missing sections before returning: {', '.join(missing)}"
        ),
    }
    return json.dumps(result, indent=2)

@tool("estimate_context_usage")
def estimate_context_usage(
    text_content: str,
    context_window_size: int = 8192,
) -> str:
    """
    Estimate how much of the model's context window is being used.
 
    Uses the rule: 1 token ≈ 4 characters.
    Helps the Supervisor decide whether to summarize chat history
    before building the next prompt.
 
    Returns a JSON string with token estimates and usage percentage.
    """
    if not text_content:
        estimated_tokens = 0
    else:
        estimated_tokens = max(1, len(text_content) // 4)
 
    usage_percent = round((estimated_tokens / context_window_size) * 100, 2)

    if usage_percent < 50:
        safety = "safe"
        advice = "Context usage is low. No summarization needed."
    elif usage_percent < 75:
        safety = "moderate"
        advice = "Context usage is moderate. Consider summarizing older chat history."
    elif usage_percent < 90:
        safety = "high"
        advice = "Context usage is high. Summarize chat history before continuing."
    else:
        safety = "critical"
        advice = "Context window is nearly full. Must summarize or truncate history immediately."
 
    result = {
        "estimated_input_tokens": estimated_tokens,
        "context_window": context_window_size,
        "usage_percent": usage_percent,
        "safety_level": safety,
        "advice": advice,
        "characters_counted": len(text_content),
    }
    return json.dumps(result, indent=2)