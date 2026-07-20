
import json
import sys
from pathlib import Path
 
import pytest
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from function_tools.scientist_tools import (
    create_ml_pipeline_plan,
    detect_ml_data_risks,
    recommend_evaluation_metrics,
    recommend_ml_problem_type,
    suggest_feature_engineering,
)
 
# ============================================================
# Tool 1 — recommend_ml_problem_type
# ============================================================
 
class TestRecommendMlProblemType:
 
    def test_churn_maps_to_classification(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Predict whether a customer will churn next month",
            columns="customer_id,activity_count,churn"
        ))
        assert result["problem_type"] == "classification"
 
    def test_revenue_prediction_maps_to_regression(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Predict how much revenue a customer will generate",
            columns="customer_id,revenue,orders"
        ))
        assert result["problem_type"] == "regression"
 
    def test_segmentation_maps_to_clustering(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Segment customers into groups based on their behaviour",
            columns="customer_id,spend,frequency"
        ))
        assert result["problem_type"] == "clustering"
 
    def test_forecast_maps_to_forecasting(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Forecast next month revenue using time series data",
            columns="date,revenue,month"
        ))
        assert result["problem_type"] == "forecasting"
 
    def test_anomaly_detection_keyword(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Detect unusual and anomalous transactions in payment data",
            columns="transaction_id,amount,status"
        ))
        assert result["problem_type"] == "anomaly_detection"
 
    def test_returns_algorithms(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Predict churn", columns="customer_id,churn"
        ))
        assert "suitable_algorithms" in result
        assert len(result["suitable_algorithms"]) > 0
 
    def test_returns_timeline(self):
        result = json.loads(recommend_ml_problem_type.run(
            user_goal="Predict churn", columns="customer_id,churn"
        ))
        assert "typical_timeline" in result
 
 
# ============================================================
# Tool 2 — suggest_feature_engineering
# ============================================================
 
class TestSuggestFeatureEngineering:
 
    def test_timestamp_column_triggers_time_features(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="user_id,event_type,timestamp,amount", data_domain="ecommerce"
        ))
        groups = [g["category"].lower() for g in result["feature_engineering_ideas"]]
        assert any("time" in g for g in groups)
 
    def test_user_column_triggers_user_features(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="user_id,event_type,timestamp", data_domain="events"
        ))
        groups = [g["category"].lower() for g in result["feature_engineering_ideas"]]
        assert any("user" in g for g in groups)
 
    def test_amount_column_triggers_transaction_features(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="customer_id,amount,order_date", data_domain="ecommerce"
        ))
        groups = [g["category"].lower() for g in result["feature_engineering_ideas"]]
        assert any("transaction" in g for g in groups)
 
    def test_each_group_has_ideas(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="user_id,amount,timestamp", data_domain="general"
        ))
        for group in result["feature_engineering_ideas"]:
            assert "ideas" in group
            assert len(group["ideas"]) > 0
 
    def test_returns_general_tips(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="user_id,amount", data_domain="general"
        ))
        assert "general_tips" in result
        tips = " ".join(result["general_tips"]).lower()
        assert "leakage" in tips or "split" in tips
 
    def test_no_columns_returns_generic_group(self):
        result = json.loads(suggest_feature_engineering.run(
            columns="col_a,col_b", data_domain="general"
        ))
        assert len(result["feature_engineering_ideas"]) > 0
 
 
# ============================================================
# Tool 3 — detect_ml_data_risks
# ============================================================
 
class TestDetectMlDataRisks:
 
    def test_valid_file_runs_without_error(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="customers_sample.csv", target_column="churn"
        ))
        assert "error" not in result
        assert "risks" in result
 
    def test_detects_missing_target_when_target_not_in_file(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="events_sample.csv", target_column="nonexistent_column"
        ))
        risk_types = [r["risk"] for r in result["risks"]]
        assert "Missing Target Column" in risk_types
 
    def test_detects_time_split_requirement_for_date_columns(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="events_sample.csv", target_column="status"
        ))
        risk_types = [r["risk"] for r in result["risks"]]
        assert "Time-Based Split Required" in risk_types
 
    def test_path_traversal_blocked(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="../../etc/passwd", target_column=""
        ))
        assert "error" in result
 
    def test_returns_severity_summary(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="customers_sample.csv", target_column="churn"
        ))
        assert "severity_summary" in result
        assert "overall_assessment" in result
 
    def test_returns_total_risks_count(self):
        result = json.loads(detect_ml_data_risks.run(
            file_name="events_sample.csv", target_column="status"
        ))
        assert "total_risks_detected" in result
        assert isinstance(result["total_risks_detected"], int)
 
 
# ============================================================
# Tool 4 — recommend_evaluation_metrics
# ============================================================
 
class TestRecommendEvaluationMetrics:
 
    def test_classification_returns_f1_and_auc(self):
        result = json.loads(recommend_evaluation_metrics.run(
            problem_type="classification"
        ))
        primary = " ".join(result["primary_metrics"]).lower()
        assert "f1" in primary or "auc" in primary
 
    def test_regression_returns_rmse(self):
        result = json.loads(recommend_evaluation_metrics.run(problem_type="regression"))
        primary = " ".join(result["primary_metrics"]).lower()
        assert "rmse" in primary or "mae" in primary
 
    def test_clustering_avoids_accuracy(self):
        result = json.loads(recommend_evaluation_metrics.run(problem_type="clustering"))
        avoid = " ".join(result["avoid_metrics"]).lower()
        assert "accuracy" in avoid
 
    def test_forecasting_returns_mape(self):
        result = json.loads(recommend_evaluation_metrics.run(problem_type="forecasting"))
        all_metrics = result["primary_metrics"] + result["secondary_metrics"]
        metric_str = " ".join(all_metrics).lower()
        assert "mape" in metric_str or "rmse" in metric_str
 
    def test_context_with_costly_misses_recommends_recall(self):
        result = json.loads(recommend_evaluation_metrics.run(
            problem_type="classification",
            business_context="missing fraud cases is very costly"
        ))
        advice = " ".join(result["context_specific_advice"]).lower()
        assert "recall" in advice
 
    def test_unknown_problem_type_returns_default(self):
        result = json.loads(recommend_evaluation_metrics.run(problem_type="unknown_type"))
        assert "primary_metrics" in result
        assert len(result["primary_metrics"]) > 0
 
    def test_returns_business_notes(self):
        result = json.loads(recommend_evaluation_metrics.run(problem_type="classification"))
        assert "business_notes" in result
        assert len(result["business_notes"]) > 0
 
 
# ============================================================
# Tool 5 — create_ml_pipeline_plan
# ============================================================
 
class TestCreateMlPipelinePlan:
 
    def test_returns_9_stages(self):
        result = json.loads(create_ml_pipeline_plan.run(problem_type="classification"))
        assert result["total_stages"] == 9
 
    def test_all_stage_names_present(self):
        result = json.loads(create_ml_pipeline_plan.run(problem_type="regression"))
        stage_names = [s["name"] for s in result["pipeline_stages"]]
        expected = [
            "Data Ingestion", "Data Validation", "Feature Engineering",
            "Train-Test Split", "Model Training", "Model Evaluation",
            "Model Registry", "Batch or Real-Time Inference",
            "Monitoring and Retraining"
        ]
        for expected_name in expected:
            assert expected_name in stage_names, f"Stage '{expected_name}' missing"
 
    def test_each_stage_has_tools_and_risks(self):
        result = json.loads(create_ml_pipeline_plan.run(problem_type="classification"))
        for stage in result["pipeline_stages"]:
            assert "tools" in stage
            assert "risks" in stage
            assert "outputs" in stage
 
    def test_critical_leakage_warning_present(self):
        result = json.loads(create_ml_pipeline_plan.run(problem_type="classification"))
        warning = result["critical_warning"].lower()
        assert "leakage" in warning or "split" in warning
 
    def test_team_size_note_changes(self):
        solo_result = json.loads(create_ml_pipeline_plan.run(
            problem_type="classification", team_size="solo"
        ))
        large_result = json.loads(create_ml_pipeline_plan.run(
            problem_type="classification", team_size="large"
        ))
        assert solo_result["team_recommendation"] != large_result["team_recommendation"]
 
    def test_data_source_appears_in_stage_1(self):
        result = json.loads(create_ml_pipeline_plan.run(
            problem_type="classification", data_source="BigQuery"
        ))
        stage_1_desc = result["pipeline_stages"][0]["description"].lower()
        assert "bigquery" in stage_1_desc
