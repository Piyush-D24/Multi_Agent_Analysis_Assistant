
import json
import sys
from pathlib import Path
 
import pytest
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from mcp_server.tools.csv_profile_tools import (
    mcp_detect_data_quality_issues,
    mcp_profile_csv,
)
from mcp_server.tools.kpi_tools import (
    mcp_create_data_dictionary,
    mcp_generate_kpi_catalog,
)
from mcp_server.tools.ml_tools import (
    mcp_anomaly_detection_summary,
    mcp_feature_engineering_suggestions,
    mcp_recommend_ml_use_cases,
)
from mcp_server.tools.report_tools import mcp_generate_report_markdown
from mcp_server.tools.sql_tools import mcp_run_duckdb_query, mcp_validate_sql
 
# ============================================================
# MCP Tool 1 — mcp_profile_csv
# ============================================================
 
class TestMcpProfileCsv:
 
    def test_events_file_profiles_successfully(self):
        result = mcp_profile_csv("events_sample.csv")
        assert result["status"] == "success"
        assert result["rows"] == 40
        assert result["columns"] > 0
 
    def test_customers_file_profiles_successfully(self):
        result = mcp_profile_csv("customers_sample.csv")
        assert result["status"] == "success"
        assert "churn" in result["column_names"]
 
    def test_returns_numeric_stats(self):
        result = mcp_profile_csv("transactions_sample.csv")
        assert "numeric_stats" in result
        assert "amount" in result["numeric_stats"]
 
    def test_path_traversal_blocked(self):
        result = mcp_profile_csv("../../etc/passwd")
        assert result["status"] == "error"
        assert "denied" in result["error"].lower()
 
    def test_nonexistent_file_returns_error(self):
        result = mcp_profile_csv("ghost.csv")
        assert result["status"] == "error"
 
    def test_sample_rows_returned(self):
        result = mcp_profile_csv("events_sample.csv")
        assert "sample_rows" in result
        assert len(result["sample_rows"]) <= 5
 
 
# ============================================================
# MCP Tool 2 — mcp_run_duckdb_query
# ============================================================
 
class TestMcpRunDuckdbQuery:
 
    def test_valid_select_returns_rows(self):
        result = mcp_run_duckdb_query(
            "SELECT event_type, COUNT(*) as cnt FROM data GROUP BY event_type LIMIT 5",
            "events_sample.csv"
        )
        assert result["status"] == "success"
        assert "rows" in result
        assert len(result["rows"]) > 0
 
    def test_delete_is_blocked(self):
        result = mcp_run_duckdb_query(
            "DELETE FROM data WHERE user_id = 'U101'",
            "events_sample.csv"
        )
        assert result["status"] == "blocked"
 
    def test_drop_is_blocked(self):
        result = mcp_run_duckdb_query("DROP TABLE data", "events_sample.csv")
        assert result["status"] == "blocked"
 
    def test_result_columns_returned(self):
        result = mcp_run_duckdb_query(
            "SELECT user_id, status FROM data LIMIT 3",
            "events_sample.csv"
        )
        assert result["status"] == "success"
        assert "user_id" in result["columns"]
        assert "status" in result["columns"]
 
    def test_nonexistent_file_returns_error(self):
        result = mcp_run_duckdb_query("SELECT * FROM data LIMIT 1", "ghost.csv")
        assert result["status"] == "error"
 
    def test_path_traversal_blocked(self):
        result = mcp_run_duckdb_query("SELECT 1", "../../etc/passwd")
        assert result["status"] == "error"
        assert "denied" in result["error"].lower()
 
 
# ============================================================
# MCP Tool 3 — mcp_validate_sql
# ============================================================
 
class TestMcpValidateSql:
 
    def test_safe_select_passes(self):
        result = mcp_validate_sql(
            "SELECT user_id, COUNT(*) FROM events WHERE date > '2024-01-01' GROUP BY user_id LIMIT 100"
        )
        assert result["is_safe"] is True
 
    def test_delete_is_blocked(self):
        result = mcp_validate_sql("DELETE FROM events")
        assert result["is_safe"] is False
        assert "DELETE" in result["blocked_reason"]
 
    def test_select_star_warning(self):
        result = mcp_validate_sql("SELECT * FROM events LIMIT 10")
        assert result["is_safe"] is True
        warnings = " ".join(result["warnings"]).lower()
        assert "select *" in warnings or "*" in warnings
 
    def test_no_limit_warning(self):
        result = mcp_validate_sql("SELECT user_id FROM events")
        warnings = " ".join(result["warnings"]).lower()
        assert "limit" in warnings
 
    def test_no_where_warning(self):
        result = mcp_validate_sql("SELECT user_id FROM events LIMIT 100")
        warnings = " ".join(result["warnings"]).lower()
        assert "where" in warnings
 
    def test_empty_query_fails(self):
        result = mcp_validate_sql("")
        assert result["is_safe"] is False
 
 
# ============================================================
# MCP Tool 4 — mcp_detect_data_quality_issues
# ============================================================
 
class TestMcpDetectDataQualityIssues:
 
    def test_events_has_issues(self):
        result = mcp_detect_data_quality_issues("events_sample.csv")
        assert result["status"] == "success"
        # events has nulls in 'amount' column
        assert result["total_issues"] > 0
 
    def test_issue_has_required_fields(self):
        result = mcp_detect_data_quality_issues("events_sample.csv")
        for issue in result["issues"]:
            assert "check" in issue
            assert "severity" in issue
            assert "fix" in issue
 
    def test_severity_values_are_valid(self):
        result = mcp_detect_data_quality_issues("events_sample.csv")
        valid_severities = {"critical", "high", "medium", "low"}
        for issue in result["issues"]:
            assert issue["severity"] in valid_severities
 
    def test_path_traversal_blocked(self):
        result = mcp_detect_data_quality_issues("../../root/.bashrc")
        assert result["status"] == "error"
 
    def test_returns_overall_assessment(self):
        result = mcp_detect_data_quality_issues("events_sample.csv")
        assert "overall" in result
 
 
# ============================================================
# MCP Tool 5 — mcp_generate_kpi_catalog
# ============================================================
 
class TestMcpGenerateKpiCatalog:
 
    def test_ecommerce_kpis_returned(self):
        result = mcp_generate_kpi_catalog("ecommerce")
        assert result["status"] == "success"
        assert result["total_kpis"] > 0
 
    def test_kpi_has_formula_and_grain(self):
        result = mcp_generate_kpi_catalog("saas")
        for kpi in result["kpis"]:
            assert "formula" in kpi
            assert "grain" in kpi
            assert "business_use" in kpi
 
    def test_with_file_detects_columns(self):
        result = mcp_generate_kpi_catalog("events", "events_sample.csv")
        assert result["status"] == "success"
        assert len(result["detected_columns"]) > 0
 
    def test_unknown_domain_returns_kpis_without_crash(self):
        result = mcp_generate_kpi_catalog("nanotechnology")
        assert result["status"] == "success"
        # Returns empty list or generic — should not error
 
 
# ============================================================
# MCP Tool 6 — mcp_recommend_ml_use_cases
# ============================================================
 
class TestMcpRecommendMlUseCases:
 
    def test_customers_file_matches_churn(self):
        result = mcp_recommend_ml_use_cases(file_name="customers_sample.csv")
        assert result["status"] == "success"
        use_case_names = [u["use_case"].lower() for u in result["ml_use_cases"]]
        assert any("churn" in name for name in use_case_names)
 
    def test_transactions_file_matches_fraud(self):
        result = mcp_recommend_ml_use_cases(file_name="transactions_sample.csv")
        use_case_names = [u["use_case"].lower() for u in result["ml_use_cases"]]
        assert any("fraud" in name or "transaction" in name for name in use_case_names)
 
    def test_columns_param_works_without_file(self):
        result = mcp_recommend_ml_use_cases(columns="customer_id,churn,revenue,date")
        assert result["status"] == "success"
        assert len(result["ml_use_cases"]) > 0
 
    def test_use_case_has_required_fields(self):
        result = mcp_recommend_ml_use_cases(file_name="customers_sample.csv")
        for uc in result["ml_use_cases"]:
            assert "use_case" in uc
            assert "problem_type" in uc
            assert "business_value" in uc
 
    def test_no_input_returns_error(self):
        result = mcp_recommend_ml_use_cases()
        assert result["status"] == "error"
 
 
# ============================================================
# MCP Tool 7 — mcp_feature_engineering_suggestions
# ============================================================
 
class TestMcpFeatureEngineeringSuggestions:
 
    def test_events_file_returns_time_features(self):
        result = mcp_feature_engineering_suggestions(file_name="events_sample.csv")
        assert result["status"] == "success"
        group_names = [g["group"].lower() for g in result["features"]]
        assert any("time" in g for g in group_names)
 
    def test_each_feature_has_name_formula_reason(self):
        result = mcp_feature_engineering_suggestions(file_name="events_sample.csv")
        for group in result["features"]:
            for feature in group["features"]:
                assert "name" in feature
                assert "formula" in feature
                assert "reason" in feature
 
    def test_reminders_include_leakage_warning(self):
        result = mcp_feature_engineering_suggestions(file_name="events_sample.csv")
        reminders = " ".join(result["critical_reminders"]).lower()
        assert "leakage" in reminders
 
 
# ============================================================
# MCP Tool 8 — mcp_anomaly_detection_summary
# ============================================================
 
class TestMcpAnomalyDetectionSummary:
 
    def test_zscore_runs_on_amount_column(self):
        result = mcp_anomaly_detection_summary("transactions_sample.csv", "amount", "zscore")
        assert result["status"] == "success"
        assert "zscore" in result["method_results"]
 
    def test_iqr_runs_on_amount_column(self):
        result = mcp_anomaly_detection_summary("transactions_sample.csv", "amount", "iqr")
        assert result["status"] == "success"
        assert "iqr" in result["method_results"]
 
    def test_both_method_runs_both(self):
        result = mcp_anomaly_detection_summary("transactions_sample.csv", "amount", "both")
        assert "zscore" in result["method_results"]
        assert "iqr" in result["method_results"]
 
    def test_nonexistent_column_returns_error(self):
        result = mcp_anomaly_detection_summary("transactions_sample.csv", "nonexistent_col")
        assert result["status"] == "error"
 
    def test_nonnumeric_column_returns_error(self):
        result = mcp_anomaly_detection_summary("events_sample.csv", "event_type")
        assert result["status"] == "error"
        assert "numeric" in result["error"].lower()
 
    def test_returns_plain_english_summary(self):
        result = mcp_anomaly_detection_summary("transactions_sample.csv", "amount")
        assert "summary" in result
        assert len(result["summary"]) > 0
 
 
# ============================================================
# MCP Tool 9 — mcp_create_data_dictionary
# ============================================================
 
class TestMcpCreateDataDictionary:
 
    def test_events_dictionary_created(self):
        result = mcp_create_data_dictionary("events_sample.csv")
        assert result["status"] == "success"
        assert len(result["columns"]) > 0
 
    def test_each_column_has_required_fields(self):
        result = mcp_create_data_dictionary("events_sample.csv")
        for col in result["columns"]:
            assert "name" in col
            assert "dtype" in col
            assert "possible_meaning" in col
            assert "sample_values" in col
            assert "null_count" in col
 
    def test_id_columns_flagged(self):
        result = mcp_create_data_dictionary("events_sample.csv")
        id_cols = [c for c in result["columns"] if c["is_likely_id"]]
        # event_id, user_id, session_id should be flagged
        assert len(id_cols) > 0
 
    def test_date_columns_flagged(self):
        result = mcp_create_data_dictionary("events_sample.csv")
        date_cols = [c for c in result["columns"] if c["is_likely_date"]]
        assert len(date_cols) > 0  # 'timestamp' should be flagged
 
 
# ============================================================
# MCP Tool 10 — mcp_generate_report_markdown
# ============================================================
 
class TestMcpGenerateReportMarkdown:
 
    def test_all_10_sections_present_when_data_provided(self):
        result = mcp_generate_report_markdown(
            dataset_summary='{"rows": 40}',
            data_quality='{"issues": []}',
            kpis='{"kpis": []}',
            ml_use_cases='{"ml_use_cases": []}',
            feature_ideas='{"features": []}',
            dashboard_layout='{"sections": []}',
            risks="Small dataset.",
            agent_work_summary="Supervisor delegated to both agents.",
            domain="ecommerce",
            file_name="events_sample.csv",
        )
        assert result["status"] == "success"
        assert result["section_count"] == 10
        assert result["all_sections_present"] is True
 
    def test_empty_inputs_still_produce_10_sections(self):
        # All placeholders should kick in
        result = mcp_generate_report_markdown()
        assert result["section_count"] == 10
 
    def test_markdown_report_is_string(self):
        result = mcp_generate_report_markdown(domain="saas")
        assert isinstance(result["markdown_report"], str)
        assert len(result["markdown_report"]) > 100
 
    def test_domain_appears_in_report(self):
        result = mcp_generate_report_markdown(domain="fintech")
        assert "fintech" in result["markdown_report"].lower()