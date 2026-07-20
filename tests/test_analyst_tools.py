 
import json
import sys
from pathlib import Path
 
import pytest
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from function_tools.analyst_tools import (
    explain_query_result,
    generate_dashboard_layout,
    profile_dataframe,
    suggest_kpi_metrics,
    validate_sql_safety,
)
 
# ============================================================
# Tool 1 — profile_dataframe
# ============================================================
 
class TestProfileDataframe:
 
    def test_valid_csv_returns_success(self):
        result = json.loads(profile_dataframe.run(file_name="events_sample.csv"))
        assert "error" not in result
        assert result["rows"] > 0
        assert result["columns"] > 0
 
    def test_returns_column_names(self):
        result = json.loads(profile_dataframe.run(file_name="events_sample.csv"))
        assert "column_names" in result
        assert isinstance(result["column_names"], list)
        assert len(result["column_names"]) > 0
 
    def test_returns_sample_records(self):
        result = json.loads(profile_dataframe.run(file_name="events_sample.csv"))
        assert "sample_records" in result
        assert len(result["sample_records"]) <= 3
 
    def test_detects_missing_values(self):
        result = json.loads(profile_dataframe.run(file_name="events_sample.csv"))
        assert "missing_values" in result
        assert "amount" in result["missing_values"]
 
    def test_path_traversal_blocked(self):
        result = json.loads(profile_dataframe.run(file_name="../../etc/passwd"))
        assert "error" in result
        assert "denied" in result["error"].lower() or "outside" in result["error"].lower()
 
    def test_nonexistent_file_returns_error(self):
        result = json.loads(profile_dataframe.run(file_name="doesnotexist.csv"))
        assert "error" in result
 
 
# ============================================================
# Tool 2 — suggest_kpi_metrics
# ============================================================
 
class TestSuggestKpiMetrics:
 
    def test_ecommerce_domain_returns_kpis(self):
        result = json.loads(suggest_kpi_metrics.run(
            domain="ecommerce",
            columns="order_id,customer_id,revenue,order_date,status"
        ))
        assert result["total_kpis_suggested"] > 0
        assert len(result["recommended_kpis"]) > 0
 
    def test_kpi_has_required_fields(self):
        result = json.loads(suggest_kpi_metrics.run(
            domain="ecommerce", columns="revenue,customer_id"
        ))
        for kpi in result["recommended_kpis"]:
            assert "name" in kpi
            assert "formula" in kpi
            assert "business_use" in kpi
 
    def test_revenue_column_triggers_revenue_kpi(self):
        result = json.loads(suggest_kpi_metrics.run(
            domain="general", columns="revenue,date,user_id"
        ))
        kpi_names = [k["name"].lower() for k in result["recommended_kpis"]]
        assert any("revenue" in name for name in kpi_names)
 
    def test_unknown_domain_falls_back_to_column_based(self):
        result = json.loads(suggest_kpi_metrics.run(
            domain="aerospace", columns="flight_id,altitude,speed"
        ))
        assert "recommended_kpis" in result
 
    def test_deduplication_no_duplicate_kpi_names(self):
        result = json.loads(suggest_kpi_metrics.run(
            domain="ecommerce", columns="revenue,order_id,customer_id"
        ))
        names = [k["name"] for k in result["recommended_kpis"]]
        assert len(names) == len(set(names)), "Duplicate KPI names found"
 
 
# ============================================================
# Tool 3 — generate_dashboard_layout
# ============================================================
 
class TestGenerateDashboardLayout:
 
    def test_returns_dashboard_name(self):
        result = json.loads(generate_dashboard_layout.run(
            domain="ecommerce", kpis="Total Revenue,Churn Rate",
            columns="customer_id,revenue,date"
        ))
        assert "dashboard_name" in result
        assert len(result["dashboard_name"]) > 0
 
    def test_sections_are_present(self):
        result = json.loads(generate_dashboard_layout.run(
            domain="saas", kpis="MRR,Churn Rate", columns="customer_id,revenue,status"
        ))
        assert "sections" in result
        assert len(result["sections"]) > 0
 
    def test_each_section_has_required_fields(self):
        result = json.loads(generate_dashboard_layout.run(
            domain="ecommerce", kpis="Revenue", columns="order_id,revenue"
        ))
        for section in result["sections"]:
            assert "section" in section
            assert "chart_types" in section
            assert "filters" in section
 
    def test_always_includes_overview_section(self):
        result = json.loads(generate_dashboard_layout.run(
            domain="fintech", kpis="Transaction Volume",
            columns="transaction_id,amount"
        ))
        section_names = [s["section"].lower() for s in result["sections"]]
        assert "overview" in section_names
 
    def test_event_column_triggers_event_section(self):
        result = json.loads(generate_dashboard_layout.run(
            domain="events", kpis="Event Volume",
            columns="user_id,event_type,timestamp"
        ))
        section_names = " ".join([s["section"].lower() for s in result["sections"]])
        assert "event" in section_names
 
 
# ============================================================
# Tool 4 — validate_sql_safety
# ============================================================
 
class TestValidateSqlSafety:
 
    def test_valid_select_passes(self):
        result = json.loads(validate_sql_safety.run(
            sql_query="SELECT event_type, COUNT(*) FROM events GROUP BY event_type LIMIT 10"
        ))
        assert result["is_safe"] is True
        assert result["blocked_reason"] is None
 
    def test_delete_is_blocked(self):
        result = json.loads(validate_sql_safety.run(
            sql_query="DELETE FROM users WHERE id = 1"
        ))
        assert result["is_safe"] is False
        assert "DELETE" in result["blocked_reason"]
 
    def test_drop_is_blocked(self):
        result = json.loads(validate_sql_safety.run(sql_query="DROP TABLE customers"))
        assert result["is_safe"] is False
 
    def test_update_is_blocked(self):
        result = json.loads(validate_sql_safety.run(
            sql_query="UPDATE orders SET status = 'cancelled'"
        ))
        assert result["is_safe"] is False
 
    def test_select_star_generates_warning(self):
        result = json.loads(validate_sql_safety.run(sql_query="SELECT * FROM events LIMIT 10"))
        assert result["is_safe"] is True
        warnings = " ".join(result["warnings"]).lower()
        assert "select *" in warnings or "column" in warnings
 
    def test_no_limit_generates_warning(self):
        result = json.loads(validate_sql_safety.run(sql_query="SELECT id FROM events"))
        assert result["is_safe"] is True
        warnings = " ".join(result["warnings"]).lower()
        assert "limit" in warnings
 
    def test_empty_query_is_blocked(self):
        result = json.loads(validate_sql_safety.run(sql_query=""))
        assert result["is_safe"] is False
 
    def test_column_name_with_blocked_keyword_not_false_positive(self):
        result = json.loads(validate_sql_safety.run(
            sql_query="SELECT created_at, user_id FROM events LIMIT 100"
        ))
        assert result["is_safe"] is True
 
 
# ============================================================
# Tool 5 — explain_query_result
# ============================================================
 
class TestExplainQueryResult:
 
    def test_decreasing_revenue_returns_explanation(self):
        result = json.loads(explain_query_result.run(
            metric="monthly_revenue", trend="decreasing", change_percent=-12.5
        ))
        assert "explanation" in result
        assert "12.5" in result["explanation"]
        assert result["trend"] == "decreasing"
 
    def test_increasing_trend_is_positive(self):
        result = json.loads(explain_query_result.run(
            metric="monthly_revenue", trend="increasing", change_percent=15.0
        ))
        assert "📈" in result["emoji"]
        assert "possible_causes" in result
 
    def test_stable_trend_handled(self):
        result = json.loads(explain_query_result.run(
            metric="active_users", trend="stable", change_percent=0.5
        ))
        assert "➡️" in result["emoji"]
        assert result["severity"] == "minor"
 
    def test_severity_significant_above_25_percent(self):
        result = json.loads(explain_query_result.run(
            metric="churn_rate", trend="increasing", change_percent=30.0
        ))
        assert result["severity"] == "significant"
 
    def test_severity_minor_below_10_percent(self):
        result = json.loads(explain_query_result.run(
            metric="churn_rate", trend="decreasing", change_percent=-5.0
        ))
        assert result["severity"] == "minor"
 
    def test_returns_suggested_actions(self):
        result = json.loads(explain_query_result.run(
            metric="monthly_revenue", trend="decreasing", change_percent=-20.0
        ))
        assert "suggested_actions" in result
        assert len(result["suggested_actions"]) > 0
 
    def test_churn_metric_returns_churn_specific_causes(self):
        result = json.loads(explain_query_result.run(
            metric="churn_rate", trend="increasing", change_percent=25.0
        ))
        causes = " ".join(result["possible_causes"]).lower()
        assert any(kw in causes for kw in ["churn", "satisfaction", "product", "onboard", "compet"])