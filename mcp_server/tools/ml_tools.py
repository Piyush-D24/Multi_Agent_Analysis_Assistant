from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats

SAFE_DATA_DIR = (Path(__file__).parent.parent / "sample_data").resolve()

def _safe_load_csv(file_name: str) -> pd.DataFrame:
    resolved = (SAFE_DATA_DIR / file_name).resolve()
    if not str(resolved).startswith(str(SAFE_DATA_DIR)):
        raise ValueError(f"Access denied: '{file_name}' is outside allowed directory.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    return pd.read_csv(resolved)

def mcp_recommend_ml_use_cases(
    file_name: str = "",
    columns: str = "",
    domain: str = "general",
) -> Dict[str, Any]:
    """
    Recommend ML use cases based on dataset columns and domain.

    Used by: Data Scientist Agent, Supervisor Agent.

    Either provide a file_name (auto-reads columns from CSV)
    OR provide a comma-separated columns string.

    Each use case includes:
      use_case, problem_type, required_columns,
      business_value, complexity, estimated_timeline.
    Returns:
        Dict with domain, detected_columns, ml_use_cases list.
    """
    # ── Detect columns ─────────────────────
    detected_columns = []
 
    if file_name:
        try:
            df = _safe_load_csv(file_name)
            detected_columns = [c.lower() for c in df.columns]
        except Exception as e:
            return {"status": "error", "error": str(e)}
    elif columns:
        detected_columns = [c.strip().lower() for c in columns.split(",")]
    else:
        return {
            "status": "error",
            "error": "Provide either file_name or columns parameter.",
        }
 
    col_str = " ".join(detected_columns)
 
    # ── Use case library with column triggers ────────────────
    use_case_library = [
        {
            "use_case": "Customer Churn Prediction",
            "problem_type": "classification",
            "trigger_keywords": ["churn", "status", "customer", "subscription", "cancel", "active"],
            "required_columns": ["customer_id", "status or churn label"],
            "business_value": "Identify at-risk customers before they leave. Reduces revenue loss.",
            "complexity": "medium",
            "estimated_timeline": "4–6 weeks",
        },
        {
            "use_case": "Fraud Detection",
            "problem_type": "anomaly_detection / classification",
            "trigger_keywords": ["transaction", "payment", "amount", "fraud", "fail", "flag"],
            "required_columns": ["transaction_id", "amount", "status"],
            "business_value": "Catch fraudulent transactions before they complete.",
            "complexity": "high",
            "estimated_timeline": "6–8 weeks",
        },
        {
            "use_case": "Revenue Forecasting",
            "problem_type": "forecasting",
            "trigger_keywords": ["revenue", "amount", "date", "order", "sales", "price"],
            "required_columns": ["date or timestamp", "revenue or amount"],
            "business_value": "Predict next month/quarter revenue for planning.",
            "complexity": "medium",
            "estimated_timeline": "4–5 weeks",
        },
        {
            "use_case": "Customer Segmentation",
            "problem_type": "clustering",
            "trigger_keywords": ["customer", "user", "segment", "behaviour", "rfm", "spend"],
            "required_columns": ["customer_id", "any behavioural columns"],
            "business_value": "Group customers by behaviour for targeted campaigns.",
            "complexity": "low",
            "estimated_timeline": "2–3 weeks",
        },
        {
            "use_case": "Event Anomaly Detection",
            "problem_type": "anomaly_detection",
            "trigger_keywords": ["event", "event_type", "timestamp", "log", "error", "fail"],
            "required_columns": ["event_type", "timestamp"],
            "business_value": "Detect unusual platform activity or system errors automatically.",
            "complexity": "medium",
            "estimated_timeline": "3–4 weeks",
        },
        {
            "use_case": "Product Recommendation",
            "problem_type": "recommendation",
            "trigger_keywords": ["product", "item", "purchase", "order", "user_id", "click"],
            "required_columns": ["user_id", "product_id or item_id", "interaction signal"],
            "business_value": "Increase basket size and repeat purchases via personalisation.",
            "complexity": "high",
            "estimated_timeline": "6–10 weeks",
        },
        {
            "use_case": "Demand Forecasting",
            "problem_type": "forecasting",
            "trigger_keywords": ["quantity", "demand", "inventory", "stock", "sku", "supply"],
            "required_columns": ["date", "quantity or demand"],
            "business_value": "Reduce overstock and stockout by predicting demand accurately.",
            "complexity": "medium",
            "estimated_timeline": "5–6 weeks",
        },
        {
            "use_case": "Lifetime Value Prediction",
            "problem_type": "regression",
            "trigger_keywords": ["revenue", "customer", "lifetime", "clv", "ltv", "spend"],
            "required_columns": ["customer_id", "revenue or spend", "date"],
            "business_value": "Prioritise high-LTV customers for retention and upsell.",
            "complexity": "medium",
            "estimated_timeline": "4–5 weeks",
        },
    ]
 
    # Match use cases to detected columns 
    matched = []
    for uc in use_case_library:
        score = sum(1 for kw in uc["trigger_keywords"] if kw in col_str)
        if score > 0:
            uc_copy = dict(uc)
            uc_copy.pop("trigger_keywords")
            uc_copy["relevance_score"] = score
            uc_copy["relevance_note"] = (
                f"{score} matching column signals found."
            )
            matched.append(uc_copy)
 
    # Sort by relevance descending
    matched.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "status": "success",
        "file": file_name or "(columns provided directly)",
        "domain": domain,
        "detected_columns": detected_columns,
        "total_use_cases_matched": len(matched),
        "ml_use_cases": matched if matched else [
            {
                "use_case": "General Analytics Exploration",
                "problem_type": "clustering or classification",
                "required_columns": detected_columns,
                "business_value": "No strong ML signal detected. Start with EDA and clustering.",
                "complexity": "low",
                "estimated_timeline": "2–3 weeks",
                "relevance_score": 0,
                "relevance_note": "No specific column signals matched known use cases.",
            }
        ],
    }

def mcp_feature_engineering_suggestions(
    file_name: str = "",
    columns: str = "",
) -> Dict[str, Any]:
    """
    Suggest feature engineering ideas for the given dataset.
 
    Used by: Data Scientist Agent.
 
    Reads actual column names (from file or direct input) and produces
    specific feature ideas based on what columns are present —
    not generic advice.
 
    Args:
        file_name: CSV in sample_data/ (auto-detects columns).
        columns:   Comma-separated column names if no file provided.
 
    Returns:
        Dict with feature groups, each tied to detected column types.
    """
    detected_columns = []
 
    if file_name:
        try:
            df = _safe_load_csv(file_name)
            detected_columns = [c.lower() for c in df.columns]
        except Exception as e:
            return {"status": "error", "error": str(e)}
    elif columns:
        detected_columns = [c.strip().lower() for c in columns.split(",")]
    else:
        return {"status": "error", "error": "Provide file_name or columns."}
 
    feature_groups = []

    time_cols = [c for c in detected_columns if any(
        kw in c for kw in ["date", "time", "timestamp", "created", "updated"]
    )]
    if time_cols:
        feature_groups.append({
            "group": "Time-Based Features",
            "source_columns": time_cols,
            "features": [
                {"name": "hour_of_day",             "formula": "EXTRACT(HOUR FROM timestamp)",          "reason": "Captures intraday patterns (peak hours)"},
                {"name": "day_of_week",              "formula": "EXTRACT(DOW FROM timestamp)",           "reason": "Weekly seasonality signal"},
                {"name": "is_weekend",               "formula": "day_of_week IN (6, 7)",                 "reason": "Binary — weekend vs weekday behaviour differs"},
                {"name": "days_since_last_event",    "formula": "current_date - MAX(event_date) per user","reason": "Recency signal — key for churn and retention"},
                {"name": "rolling_7_day_event_count","formula": "COUNT(*) OVER (PARTITION BY user ORDER BY date ROWS 7 PRECEDING)", "reason": "Short-term activity burst signal"},
                {"name": "rolling_30_day_event_count","formula": "COUNT(*) OVER (PARTITION BY user ORDER BY date ROWS 30 PRECEDING)","reason": "Medium-term engagement"},
                {"name": "month_of_year",            "formula": "EXTRACT(MONTH FROM timestamp)",         "reason": "Annual seasonality"},
            ],
        })

    user_cols = [c for c in detected_columns if any(
        kw in c for kw in ["user", "customer", "account", "member", "client"]
    )]
    if user_cols:
        feature_groups.append({
            "group": "User Aggregation Features",
            "source_columns": user_cols,
            "features": [
                {"name": "total_events_per_user",     "formula": "COUNT(*) GROUP BY user_id",            "reason": "Overall engagement depth"},
                {"name": "distinct_event_types_used", "formula": "COUNT(DISTINCT event_type) per user",   "reason": "Breadth of platform usage"},
                {"name": "session_count_last_30_days","formula": "COUNT(DISTINCT session_id) per user in 30d","reason": "Frequency of return visits"},
                {"name": "days_since_registration",   "formula": "current_date - account_created_at",     "reason": "Account age — younger accounts churn faster"},
                {"name": "support_ticket_count",      "formula": "COUNT(tickets) GROUP BY user_id",       "reason": "Proxy for dissatisfaction"},
            ],
        })

    txn_cols = [c for c in detected_columns if any(
        kw in c for kw in ["amount", "revenue", "price", "transaction", "payment", "order", "cost"]
    )]
    if txn_cols:
        feature_groups.append({
            "group": "Transaction Features",
            "source_columns": txn_cols,
            "features": [
                {"name": "avg_transaction_amount",       "formula": "AVG(amount) GROUP BY user_id",          "reason": "Spending level indicator"},
                {"name": "max_transaction_amount",       "formula": "MAX(amount) GROUP BY user_id",          "reason": "High-value event signal"},
                {"name": "std_transaction_amount",       "formula": "STDDEV(amount) GROUP BY user_id",       "reason": "Spending consistency / volatility"},
                {"name": "total_spend_last_30_days",     "formula": "SUM(amount) per user in last 30 days",  "reason": "Recent monetary value"},
                {"name": "days_since_last_purchase",     "formula": "current_date - MAX(order_date)",        "reason": "Recency — core RFM feature"},
                {"name": "purchase_frequency",           "formula": "COUNT(orders) / account_age_in_months", "reason": "How often this customer buys"},
                {"name": "refund_rate",                  "formula": "COUNT(refunds) / COUNT(orders)",        "reason": "Satisfaction / potential fraud signal"},
            ],
        })

    event_cols = [c for c in detected_columns if any(
        kw in c for kw in ["event", "action", "type", "status", "click", "error", "fail"]
    )]
    if event_cols:
        feature_groups.append({
            "group": "Event & Behavioural Features",
            "source_columns": event_cols,
            "features": [
                {"name": "failed_event_ratio_24h",    "formula": "COUNT(status='failed') / COUNT(*) per user in 24h","reason": "Platform health / fraud signal"},
                {"name": "error_count_7_days",        "formula": "COUNT(event_type='error') per user in 7d",         "reason": "Dissatisfaction / churn signal"},
                {"name": "login_attempts_last_hour",  "formula": "COUNT(event_type='login') per user in 1h",         "reason": "Brute-force / account takeover signal"},
                {"name": "click_through_rate",        "formula": "clicks / impressions per user",                    "reason": "Content relevance signal"},
                {"name": "session_duration_avg",      "formula": "AVG(session_end - session_start) per user",        "reason": "Quality of engagement"},
            ],
        })

    if not feature_groups:
        feature_groups.append({
            "group": "Generic Features (no specific column types detected)",
            "source_columns": detected_columns,
            "features": [
                {"name": "row_count_per_id",    "formula": "COUNT(*) GROUP BY any_id_column",    "reason": "Frequency of occurrence"},
                {"name": "null_indicator_flags","formula": "CASE WHEN col IS NULL THEN 1 ELSE 0 END","reason": "Missingness as a feature — may be informative"},
                {"name": "target_encoding",     "formula": "AVG(target) GROUP BY categorical_col","reason": "Encode categoricals by target mean"},
            ],
        })

    return {
        "status": "success",
        "file": file_name or "(columns provided directly)",
        "detected_columns": detected_columns,
        "total_feature_groups": len(feature_groups),
        "features": feature_groups,
        "critical_reminders": [
            "Split train/test BEFORE engineering features — prevents data leakage.",
            "Drop raw ID columns (user_id, order_id) before model training.",
            "Apply StandardScaler to numeric features for distance-based models.",
            "For rolling features, use TimeSeriesSplit, not random split.",
        ],
    }

def mcp_anomaly_detection_summary(
    file_name: str,
    numeric_column: str,
    method: str = "zscore",
) -> Dict[str, Any]:
    """
    Detect anomalies in a numeric column of a CSV file.

    Used by: Data Scientist Agent, Data Analyst Agent.

    Methods supported:
      zscore  — flags values where |Z| > 3 (assumes normal distribution)
      iqr     — flags values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
      both    — runs both methods and returns union of anomalies

    Args:
        file_name:      CSV in sample_data/.
        numeric_column: Column to run anomaly detection on.
        method:         'zscore', 'iqr', or 'both'.

    Returns:
        Dict with method used, thresholds, anomaly count, anomaly rows,
        and a plain-English summary.
    """
    try:
        df = _safe_load_csv(file_name)
 
        if numeric_column not in df.columns:
            return {
                "status": "error",
                "error": f"Column '{numeric_column}' not found. Available: {list(df.columns)}",
            }
 
        series = df[numeric_column].dropna()
 
        if not pd.api.types.is_numeric_dtype(series):
            return {
                "status": "error",
                "error": f"Column '{numeric_column}' is not numeric (dtype: {series.dtype}).",
            }
 
        results = {}
        anomaly_indices = set()

        if method in ("zscore", "both"):
            z_scores = np.abs(stats.zscore(series))
            z_anomaly_idx = series.index[z_scores > 3].tolist()
            anomaly_indices.update(z_anomaly_idx)
            results["zscore"] = {
                "method": "Z-Score (threshold: |Z| > 3)",
                "assumes": "Approximately normal distribution",
                "anomaly_count": len(z_anomaly_idx),
                "threshold_upper": float(series.mean() + 3 * series.std()),
                "threshold_lower": float(series.mean() - 3 * series.std()),
                "anomaly_row_indices": z_anomaly_idx[:20],  # cap for context window
            }

        if method in ("iqr", "both"):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomaly_idx = series.index[
                (series < lower_bound) | (series > upper_bound)
            ].tolist()
            anomaly_indices.update(iqr_anomaly_idx)
            results["iqr"] = {
                "method": "IQR (Interquartile Range)",
                "assumes": "Works without normality assumption — robust to skewed data",
                "anomaly_count": len(iqr_anomaly_idx),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "threshold_lower": float(lower_bound),
                "threshold_upper": float(upper_bound),
                "anomaly_row_indices": iqr_anomaly_idx[:20],
            }

        all_anomaly_indices = sorted(list(anomaly_indices))
        anomaly_rows = df.loc[all_anomaly_indices].head(20).to_dict(orient="records")

        col_stats = {
            "mean":   round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std":    round(float(series.std()), 4),
            "min":    float(series.min()),
            "max":    float(series.max()),
            "count":  len(series),
        }

        total_anomalies = len(all_anomaly_indices)
        anomaly_pct = round(total_anomalies / len(series) * 100, 2)

        if total_anomalies == 0:
            summary = f"No anomalies detected in '{numeric_column}' using {method} method."
        elif anomaly_pct < 1:
            summary = (
                f"{total_anomalies} anomalies detected ({anomaly_pct}% of data) in "
                f"'{numeric_column}'. Low rate — likely genuine outliers worth investigating."
            )
        elif anomaly_pct < 5:
            summary = (
                f"{total_anomalies} anomalies detected ({anomaly_pct}% of data) in "
                f"'{numeric_column}'. Moderate rate — check for data entry errors or "
                f"legitimate extreme events."
            )
        else:
            summary = (
                f"{total_anomalies} anomalies detected ({anomaly_pct}% of data) in "
                f"'{numeric_column}'! High rate — check for systematic data quality issues, "
                f"wrong units, or upstream pipeline problems."
            )

        return {
            "status": "success",
            "file": file_name,
            "column": numeric_column,
            "method": method,
            "column_stats": col_stats,
            "total_anomalies": total_anomalies,
            "anomaly_percent": anomaly_pct,
            "summary": summary,
            "method_results": results,
            "anomaly_sample_rows": anomaly_rows,
        }

    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Anomaly detection failed: {str(e)}"}