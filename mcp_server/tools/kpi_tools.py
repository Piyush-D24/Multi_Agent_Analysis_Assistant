from pathlib import Path
from typing import Any, Dict
import pandas as pd

SAFE_DATA_DIR = (Path(__file__).parent.parent / "sample_data").resolve()

def _safe_load_csv(file_name: str) -> pd.DataFrame:
    resolved = (SAFE_DATA_DIR / file_name).resolve()
    if not str(resolved).startswith(str(SAFE_DATA_DIR)):
        raise ValueError(f"Access denied: '{file_name}' is outside allowed directory.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    return pd.read_csv(resolved)

def mcp_generate_kpi_catalog(
    domain: str,
    file_name: str = "",
) -> Dict[str, Any]:
    """
    Generate a KPI catalog from dataset columns and business domain.
 
    Used by: Data Analyst Agent, Supervisor Agent.
 
    Each KPI entry includes:
      name, formula, grain (daily/weekly/monthly), business_use,
      required_columns, and alert_threshold_hint.
    Returns:
        domain, detected_columns, kpis (list of full KPI dicts).
    """
    detected_columns = []
    if file_name:
        try:
            df = _safe_load_csv(file_name)
            detected_columns = [c.lower() for c in df.columns]
        except Exception:
            detected_columns = []
 
    domain_lower = domain.lower()
    all_kpis = {
        "ecommerce": [
            {
                "name": "Total Revenue",
                "formula": "SUM(revenue) or SUM(amount)",
                "grain": "daily / monthly",
                "business_use": "Core financial health metric.",
                "required_columns": ["revenue", "amount", "price"],
                "alert_threshold_hint": "Alert if daily revenue drops > 15% vs 7-day average.",
            },
            {
                "name": "Average Order Value (AOV)",
                "formula": "SUM(revenue) / COUNT(DISTINCT order_id)",
                "grain": "monthly",
                "business_use": "Higher AOV = larger baskets = better unit economics.",
                "required_columns": ["revenue", "order_id"],
                "alert_threshold_hint": "Alert if AOV drops > 10% month-over-month.",
            },
            {
                "name": "Repeat Purchase Rate",
                "formula": "COUNT(customers with ≥2 orders) / COUNT(DISTINCT customer_id)",
                "grain": "monthly",
                "business_use": "Measures loyalty. Target > 30% for healthy ecommerce.",
                "required_columns": ["customer_id", "order_id"],
                "alert_threshold_hint": "Alert if rate drops below 20%.",
            },
            {
                "name": "Order Cancellation Rate",
                "formula": "COUNT(status='cancelled') / COUNT(order_id)",
                "grain": "weekly",
                "business_use": "Tracks fulfilment problems and customer dissatisfaction.",
                "required_columns": ["status", "order_id"],
                "alert_threshold_hint": "Alert if rate exceeds 5%.",
            },
            {
                "name": "Monthly Active Customers",
                "formula": "COUNT(DISTINCT customer_id) per calendar month",
                "grain": "monthly",
                "business_use": "Tracks active customer base growth.",
                "required_columns": ["customer_id"],
                "alert_threshold_hint": "Alert if growth is < 0% month-over-month.",
            },
        ],
        "saas": [
            {
                "name": "Monthly Recurring Revenue (MRR)",
                "formula": "SUM(monthly_subscription_value)",
                "grain": "monthly",
                "business_use": "The single most important SaaS health metric.",
                "required_columns": ["subscription_value", "revenue"],
                "alert_threshold_hint": "Alert if MRR growth < 5% month-over-month.",
            },
            {
                "name": "Churn Rate",
                "formula": "churned_customers / customers_at_period_start",
                "grain": "monthly",
                "business_use": "< 2% monthly churn is healthy for SaaS.",
                "required_columns": ["customer_id", "status", "churn"],
                "alert_threshold_hint": "Alert if monthly churn > 3%.",
            },
            {
                "name": "Net Revenue Retention (NRR)",
                "formula": "(MRR_end + expansion - churn - contraction) / MRR_start",
                "grain": "monthly",
                "business_use": "> 100% NRR means existing customers are growing spend.",
                "required_columns": ["revenue", "customer_id"],
                "alert_threshold_hint": "Alert if NRR drops below 90%.",
            },
            {
                "name": "Customer Acquisition Cost (CAC)",
                "formula": "total_marketing_spend / new_customers_acquired",
                "grain": "monthly",
                "business_use": "Efficiency of growth investment. Compare to LTV.",
                "required_columns": ["customer_id"],
                "alert_threshold_hint": "Alert if CAC > 1/3 of LTV.",
            },
        ],
        "fintech": [
            {
                "name": "Transaction Volume",
                "formula": "COUNT(transaction_id) per day",
                "grain": "daily",
                "business_use": "Platform throughput — primary growth indicator.",
                "required_columns": ["transaction_id"],
                "alert_threshold_hint": "Alert if daily volume drops > 20% vs prior week.",
            },
            {
                "name": "Failed Transaction Rate",
                "formula": "COUNT(status='failed') / COUNT(transaction_id)",
                "grain": "hourly / daily",
                "business_use": "Indicates fraud, network issues, or UX problems.",
                "required_columns": ["status", "transaction_id"],
                "alert_threshold_hint": "Alert if failure rate exceeds 2%.",
            },
            {
                "name": "Average Transaction Amount",
                "formula": "SUM(amount) / COUNT(transaction_id)",
                "grain": "monthly",
                "business_use": "Tracks spending behavior trends.",
                "required_columns": ["amount"],
                "alert_threshold_hint": "Alert if drops > 10% month-over-month.",
            },
        ],
        "events": [
            {
                "name": "Event Volume by Type",
                "formula": "COUNT(*) GROUP BY event_type",
                "grain": "hourly / daily",
                "business_use": "Understand which events dominate platform usage.",
                "required_columns": ["event_type"],
                "alert_threshold_hint": "Alert if any event type volume drops > 30%.",
            },
            {
                "name": "Failed Event Rate",
                "formula": "COUNT(status='failed') / COUNT(*)",
                "grain": "hourly",
                "business_use": "Platform reliability and error rate monitoring.",
                "required_columns": ["status"],
                "alert_threshold_hint": "Alert if failed rate > 5% in any hour.",
            },
            {
                "name": "Active Users per Day",
                "formula": "COUNT(DISTINCT user_id) per day",
                "grain": "daily",
                "business_use": "DAU is the core engagement metric for event-based platforms.",
                "required_columns": ["user_id"],
                "alert_threshold_hint": "Alert if DAU drops > 15% vs prior week.",
            },
            {
                "name": "Events per Active User",
                "formula": "COUNT(events) / COUNT(DISTINCT user_id)",
                "grain": "daily",
                "business_use": "Depth of engagement — more events per user = higher retention.",
                "required_columns": ["user_id"],
                "alert_threshold_hint": "Alert if drops below 3 events/user/day.",
            },
        ],
    }

    # Match domain
    matched = None
    for key in all_kpis:
        if key in domain_lower or domain_lower in key:
            matched = key
            break

    kpis = all_kpis.get(matched, [])

    # Filter by detected columns if available
    if detected_columns:
        filtered = []
        for kpi in kpis:
            req = [c.lower() for c in kpi["required_columns"]]
            if any(any(r in col for col in detected_columns) for r in req):
                kpi["column_match"] = "Matched — required columns found in dataset."
                filtered.append(kpi)
            else:
                kpi["column_match"] = "Partial — required columns may be missing."
                filtered.append(kpi)  # Still include but flag
        kpis = filtered

    return {
        "status": "success",
        "domain": domain,
        "matched_library": matched or "general",
        "detected_columns": detected_columns,
        "total_kpis": len(kpis),
        "kpis": kpis,
    }

def mcp_create_data_dictionary(file_name: str) -> Dict[str, Any]:
    """
    Generate a data dictionary from a CSV file.

    Used by: Data Analyst Agent, Supervisor Agent.

    For each column returns:
      name, dtype, possible_meaning (inferred from column name),
      sample_values, null_count, unique_count, is_likely_id,
      is_likely_date, is_likely_categorical.
    Returns:
        Dict with file metadata and a 'columns' list of column profiles.
    """
    try:
        df = _safe_load_csv(file_name)
        columns = []
 
        for col in df.columns:
            col_lower = col.lower()
            series = df[col]

            # Infer meaning from column name
            if any(kw in col_lower for kw in ["id", "key", "uuid", "guid"]):
                meaning = f"Unique identifier for {col_lower.replace('_id', '').replace('_key', '')} entity."
                is_id = True
            elif any(kw in col_lower for kw in ["date", "time", "timestamp", "created", "updated"]):
                meaning = f"Date or timestamp — when the {col_lower.split('_')[0]} occurred."
                is_id = False
            elif any(kw in col_lower for kw in ["revenue", "amount", "price", "cost", "fee"]):
                meaning = f"Monetary value — likely in the company's reporting currency."
                is_id = False
            elif any(kw in col_lower for kw in ["status", "state", "type", "category", "label"]):
                meaning = f"Categorical label or status indicator."
                is_id = False
            elif any(kw in col_lower for kw in ["count", "qty", "quantity", "total", "num"]):
                meaning = f"Numeric count or quantity of something."
                is_id = False
            elif any(kw in col_lower for kw in ["name", "title", "description", "text", "comment"]):
                meaning = f"Free-text or descriptive field."
                is_id = False
            else:
                meaning = f"Purpose unclear — review with data owner."
                is_id = False

            samples = series.dropna().unique()[:5].tolist()
            samples = [str(s) for s in samples]

            is_date = any(kw in col_lower for kw in ["date", "time", "timestamp"])
            is_categorical = (
                series.dtype == object
                or (series.nunique() < 20 and series.dtype != "float64")
            )

            columns.append({
                "name": col,
                "dtype": str(series.dtype),
                "possible_meaning": meaning,
                "sample_values": samples,
                "null_count": int(series.isnull().sum()),
                "unique_count": int(series.nunique()),
                "is_likely_id": is_id,
                "is_likely_date": is_date,
                "is_likely_categorical": is_categorical and not is_id,
            })

        return {
            "status": "success",
            "file": file_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns,
        }

    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Dictionary generation failed: {str(e)}"}