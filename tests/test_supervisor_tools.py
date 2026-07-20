 
import json
import sys
from pathlib import Path
 
import pytest
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from function_tools.supervisor_tools import (
    classify_user_request,
    create_agent_work_plan,
    estimate_context_usage,
    summarize_chat_history,
    validate_final_response_structure,
)
 
# ============================================================
# Tool 1 — classify_user_request
# ============================================================
 
class TestClassifyUserRequest:
 
    def test_dashboard_intent(self):
        result = json.loads(classify_user_request.run(
            user_message="Create a KPI dashboard for revenue"
        ))
        assert result["intent"] == "dashboard"
        assert result["recommended_agent"] == "Data Analyst Agent"
 
    def test_data_science_intent(self):
        result = json.loads(classify_user_request.run(
            user_message="Train a model to predict churn using ML"
        ))
        assert result["intent"] == "data_science"
        assert result["recommended_agent"] == "Data Scientist Agent"
 
    def test_sql_intent(self):
        result = json.loads(classify_user_request.run(
            user_message="Write a SELECT query to join orders and customers"
        ))
        assert result["intent"] == "sql"
 
    def test_data_quality_intent(self):
        result = json.loads(classify_user_request.run(
            user_message="Find missing values and duplicate rows in my dataset"
        ))
        assert result["intent"] == "data_quality"
 
    def test_mixed_intent(self):
        result = json.loads(classify_user_request.run(
            user_message="Create a dashboard for KPIs and also predict customer churn using ML"
        ))
        assert result["intent"] == "mixed"
 
    def test_empty_input_defaults_to_analytics(self):
        result = json.loads(classify_user_request.run(user_message=""))
        assert result["intent"] == "analytics"
 
    def test_returns_required_keys(self):
        result = json.loads(classify_user_request.run(user_message="Analyze my data"))
        assert "intent" in result
        assert "recommended_agent" in result
        assert "reason" in result
 
 
# ============================================================
# Tool 2 — create_agent_work_plan
# ============================================================
 
class TestCreateAgentWorkPlan:
 
    def test_analyst_steps_included_for_data_request(self):
        result = json.loads(create_agent_work_plan.run(
            user_message="Profile the CSV and suggest KPIs", intent="analytics"
        ))
        steps_text = " ".join(result["steps"]).lower()
        assert "analyst" in steps_text
        assert "profile" in steps_text
 
    def test_scientist_steps_included_for_ml_request(self):
        result = json.loads(create_agent_work_plan.run(
            user_message="Recommend ML problem type and suggest features",
            intent="data_science"
        ))
        steps_text = " ".join(result["steps"]).lower()
        assert "scientist" in steps_text
        assert "ml" in steps_text or "recommend" in steps_text
 
    def test_supervisor_validation_always_at_end(self):
        result = json.loads(create_agent_work_plan.run(
            user_message="Analyze anything", intent="mixed"
        ))
        last_steps = " ".join(result["steps"][-3:]).lower()
        assert "validate" in last_steps or "combine" in last_steps
 
    def test_returns_required_keys(self):
        result = json.loads(create_agent_work_plan.run(
            user_message="Analyze data", intent="analytics"
        ))
        assert "steps" in result
        assert "agents_needed" in result
        assert "total_steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) > 0
 
 
# ============================================================
# Tool 3 — summarize_chat_history
# ============================================================
 
class TestSummarizeChatHistory:
 
    def test_empty_history(self):
        result = json.loads(summarize_chat_history.run(chat_history=""))
        assert result["message_count"] == 0
        assert "no chat history" in result["summary"].lower()
 
    def test_message_count_correct(self):
        history = "USER: Hello\nASSISTANT: Hi\nUSER: What is ML?\nASSISTANT: Machine learning."
        result = json.loads(summarize_chat_history.run(chat_history=history))
        assert result["message_count"] == 4
 
    def test_detects_crewai_topic(self):
        history = "USER: I want to build a CrewAI agent\nASSISTANT: Sure, let's set up CrewAI"
        result = json.loads(summarize_chat_history.run(chat_history=history))
        assert "multi-agent system" in result["summary"].lower() or \
               "crewai" in str(result["topics_detected"]).lower()
 
    def test_extracts_last_user_message(self):
        history = "USER: First question\nASSISTANT: Answer\nUSER: Second question"
        result = json.loads(summarize_chat_history.run(chat_history=history))
        assert "second question" in result["last_user_intent"].lower()
 
    def test_returns_required_keys(self):
        result = json.loads(summarize_chat_history.run(chat_history="USER: Hello"))
        assert "summary" in result
        assert "last_user_intent" in result
        assert "message_count" in result
 
 
# ============================================================
# Tool 4 — validate_final_response_structure
# ============================================================
 
class TestValidateFinalResponseStructure:
 
    COMPLETE_RESPONSE = """
    ## 1. Direct Answer
    Here is the answer.
    ## 2. Dataset Summary
    The dataset has 100 rows.
    ## 3. Data Quality Findings
    No major issues.
    ## 4. Recommended KPIs
    Revenue, Churn Rate.
    ## 5. Recommended Dashboard
    Revenue Dashboard.
    ## 6. ML Use Cases
    Churn prediction.
    ## 7. Feature Engineering Ideas
    Days since last login.
    ## 8. Risks and Limitations
    Small dataset.
    ## 9. Next Steps
    1. Fix quality issues.
    ## 10. Agent Work Summary
    Supervisor delegated to both agents.
    """
 
    def test_complete_response_passes(self):
        result = json.loads(validate_final_response_structure.run(
            response_text=self.COMPLETE_RESPONSE
        ))
        assert result["is_valid"] is True
        assert result["sections_missing"] == 0
 
    def test_missing_sections_detected(self):
        partial = "## 1. Direct Answer\nSome answer.\n## 2. Dataset Summary\nSome data."
        result = json.loads(validate_final_response_structure.run(response_text=partial))
        assert result["is_valid"] is False
        assert result["sections_missing"] > 0
        assert len(result["missing_sections"]) > 0
 
    def test_score_is_100_for_complete_response(self):
        result = json.loads(validate_final_response_structure.run(
            response_text=self.COMPLETE_RESPONSE
        ))
        assert result["score_percent"] == 100.0
 
    def test_empty_response_fails(self):
        result = json.loads(validate_final_response_structure.run(response_text=""))
        assert result["is_valid"] is False
        assert result["sections_present"] == 0
 
    def test_returns_required_keys(self):
        result = json.loads(validate_final_response_structure.run(response_text="some text"))
        assert "is_valid" in result
        assert "missing_sections" in result
        assert "present_sections" in result
        assert "score_percent" in result
 
 
# ============================================================
# Tool 5 — estimate_context_usage
# ============================================================
 
class TestEstimateContextUsage:
 
    def test_empty_text_returns_zero_tokens(self):
        result = json.loads(estimate_context_usage.run(
            text_content="", context_window_size=8192
        ))
        assert result["estimated_input_tokens"] == 0
 
    def test_token_estimate_is_chars_divided_by_4(self):
        text = "a" * 400  # 400 chars → ~100 tokens
        result = json.loads(estimate_context_usage.run(
            text_content=text, context_window_size=8192
        ))
        assert result["estimated_input_tokens"] == 100
 
    def test_safe_level_below_50_percent(self):
        text = "a" * 400
        result = json.loads(estimate_context_usage.run(
            text_content=text, context_window_size=8192
        ))
        assert result["safety_level"] == "safe"
 
    def test_critical_level_above_90_percent(self):
        text = "a" * 30000
        result = json.loads(estimate_context_usage.run(
            text_content=text, context_window_size=8192
        ))
        assert result["safety_level"] == "critical"
 
    def test_custom_context_window(self):
        text = "a" * 4000  # 1000 tokens
        result = json.loads(estimate_context_usage.run(
            text_content=text, context_window_size=2000
        ))
        assert result["context_window"] == 2000
        assert result["safety_level"] in ("moderate", "high")
 
    def test_returns_required_keys(self):
        result = json.loads(estimate_context_usage.run(
            text_content="hello", context_window_size=8192
        ))
        assert "estimated_input_tokens" in result
        assert "usage_percent" in result
        assert "safety_level" in result
        assert "advice" in result