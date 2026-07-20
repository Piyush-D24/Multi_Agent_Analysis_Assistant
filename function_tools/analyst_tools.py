import json
import re
import pandas as pd
import sqlglot
from pathlib import Path
from typing import Any
from crewai.tools import tool

SAFE_DATA_DIR = Path(__file__).parent.parent / "mcp_server" / "sample_data"

def _safe_resolve(file_path: str) -> Path:
    """
    Resolve a file path and confirm it stays inside sample_data/.
    Raises ValueError if the path would escape the allowed directory.
    This prevents directory traversal attacks like ../../etc/passwd.
    """
    resolved = (SAFE_DATA_DIR / file_path).resolve()
    if not str(resolved).startswith(str(SAFE_DATA_DIR.resolve())):
        raise ValueError(
            f"Access denied: '{file_path}' is outside the allowed data directory."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved

@tool("profile_dataframe")
def profile_dataframe(file_name: str) -> str:
    """
    Profile a CSV file from the sample_data directory.
 
    Returns row count, column count, column names, data types,
    missing value counts, duplicate row count, and 3 sample records.
 
    Only files inside mcp_server/sample_data/ are accessible.
    Pass just the filename, e.g. 'events_sample.csv'.
    """
    try:
        path = _safe_resolve(file_name)
        df = pd.read_csv(path)

        missing_values = df.isnull().sum()
        missing_dict = {
            col: int(count)
            for col, count in missing_values.items()
            if count > 0
        }

        result = {
            "file": file_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing_dict,
            "total_missing_cells": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "sample_records": df.head(3).to_dict(orient="records"),
        }
        return json.dumps(result, indent=2, default=str)

    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to profile file: {str(e)}"}, indent=2)

@tool("suggest_kpi_metrics")
def suggest_kpi_metrics(domain: str, columns: str) -> str:
    """
    Suggest relevant business KPIs based on the dataset domain and columns.
    Returns a JSON string with recommended_kpis (list of dicts with
    name, formula, grain, and business_use).
    """
    # Parse column list
    col_list = [c.strip().lower() for c in columns.split(",")]
    domain_lower = domain.lower()

    kpi_library = {
        "ecommerce": [
            {"name": "Total Revenue", "formula": "SUM(revenue)", "grain": "daily/monthly",
             "business_use": "Core revenue health metric."},
            {"name": "Average Order Value (AOV)", "formula": "SUM(revenue) / COUNT(order_id)",
             "grain": "monthly", "business_use": "Higher AOV means bigger baskets."},
            {"name": "Repeat Purchase Rate", "formula": "customers_with_2+_orders / total_customers",
             "grain": "monthly", "business_use": "Measures customer loyalty."},
            {"name": "Order Cancellation Rate", "formula": "cancelled_orders / total_orders",
             "grain": "weekly", "business_use": "Tracks fulfilment and satisfaction issues."},
            {"name": "Monthly Active Customers", "formula": "COUNT(DISTINCT customer_id)",
             "grain": "monthly", "business_use": "Tracks active customer base growth."},
            {"name": "Customer Lifetime Value (CLV)", "formula": "AOV × purchase_frequency × avg_lifespan",
             "grain": "annual", "business_use": "Prioritize high-value customer segments."},
        ],
        "saas": [
            {"name": "Monthly Recurring Revenue (MRR)", "formula": "SUM(monthly_subscription_value)",
             "grain": "monthly", "business_use": "Core SaaS health metric."},
            {"name": "Churn Rate", "formula": "churned_customers / customers_at_period_start",
             "grain": "monthly", "business_use": "Revenue and retention health."},
            {"name": "Customer Acquisition Cost (CAC)", "formula": "marketing_spend / new_customers",
             "grain": "monthly", "business_use": "Efficiency of growth spend."},
            {"name": "Net Revenue Retention (NRR)", "formula": "(MRR_end - churn + expansion) / MRR_start",
             "grain": "monthly", "business_use": ">100% means existing customers are expanding."},
            {"name": "Daily Active Users (DAU)", "formula": "COUNT(DISTINCT user_id) per day",
             "grain": "daily", "business_use": "Product engagement indicator."},
        ],
        "fintech": [
            {"name": "Transaction Volume", "formula": "COUNT(transaction_id)", "grain": "daily",
             "business_use": "Platform throughput and growth."},
            {"name": "Average Transaction Amount", "formula": "SUM(amount) / COUNT(transaction_id)",
             "grain": "monthly", "business_use": "Tracks transaction size trends."},
            {"name": "Failed Transaction Rate", "formula": "failed_txns / total_txns",
             "grain": "daily", "business_use": "Indicates fraud, network, or UX issues."},
            {"name": "Customer Onboarding Rate", "formula": "new_verified_users / applications",
             "grain": "monthly", "business_use": "KYC funnel effectiveness."},
        ],
        "healthcare": [
            {"name": "Patient Readmission Rate", "formula": "readmissions / total_discharges",
             "grain": "monthly", "business_use": "Key quality of care indicator."},
            {"name": "Average Length of Stay", "formula": "SUM(stay_days) / COUNT(patients)",
             "grain": "monthly", "business_use": "Efficiency and bed utilization."},
            {"name": "Appointment No-Show Rate", "formula": "no_shows / scheduled_appointments",
             "grain": "weekly", "business_use": "Capacity planning and patient engagement."},
        ],
    }

    # Find matching domain (fuzzy)
    matched_domain = None
    for key in kpi_library:
        if key in domain_lower or domain_lower in key:
            matched_domain = key
            break

    # Column-based extras (works for any domain)
    column_kpis = []
    if any(c in col_list for c in ["revenue", "amount", "price", "total"]):
        column_kpis.append({
            "name": "Total Revenue", "formula": "SUM(revenue or amount)",
            "grain": "monthly", "business_use": "Core financial health."
        })
    if any(c in col_list for c in ["status", "state", "result"]):
        column_kpis.append({
            "name": "Success/Failure Rate by Status",
            "formula": "COUNT(*) GROUP BY status",
            "grain": "daily", "business_use": "Identify failure patterns."
        })
    if any(c in col_list for c in ["user_id", "customer_id", "account_id"]):
        column_kpis.append({
            "name": "Active Unique Users/Customers",
            "formula": "COUNT(DISTINCT user_id or customer_id)",
            "grain": "daily/monthly", "business_use": "Engagement and retention tracking."
        })
    if any(c in col_list for c in ["event_type", "event", "action"]):
        column_kpis.append({
            "name": "Event Frequency by Type",
            "formula": "COUNT(*) GROUP BY event_type",
            "grain": "hourly/daily", "business_use": "Understand platform usage patterns."
        })
    if any(c in col_list for c in ["date", "timestamp", "created_at", "event_date"]):
        column_kpis.append({
            "name": "Daily/Weekly Trend",
            "formula": "COUNT(*) or SUM(value) GROUP BY date",
            "grain": "daily", "business_use": "Spot seasonality and growth trends."
        })

    # Combine domain KPIs + column KPIs
    domain_kpis = kpi_library.get(matched_domain, []) if matched_domain else []
    all_kpis = domain_kpis + column_kpis

    # Deduplicate by name
    seen = set()
    unique_kpis = []
    for kpi in all_kpis:
        if kpi["name"] not in seen:
            seen.add(kpi["name"])
            unique_kpis.append(kpi)

    result = {
        "domain": domain,
        "matched_domain_library": matched_domain or "generic (column-based)",
        "columns_analyzed": col_list,
        "total_kpis_suggested": len(unique_kpis),
        "recommended_kpis": unique_kpis,
    }
    return json.dumps(result, indent=2)

@tool("generate_dashboard_layout")
def generate_dashboard_layout(domain: str, kpis: str, columns: str) -> str:
    """
    Suggest a dashboard structure based on domain, KPIs, and dataset columns.

    Args:
        domain:  Business domain, e.g. 'ecommerce'.
        kpis:    Comma-separated KPI names, e.g. 'Total Revenue,Churn Rate'.
        columns: Comma-separated column names from the dataset.

    Returns a JSON string with dashboard_name, sections (each with charts,
    kpi_cards, filters), and drill-down views.
    """
    kpi_list = [k.strip() for k in kpis.split(",")]
    col_list = [c.strip().lower() for c in columns.split(",")]

    # Build sections dynamically
    sections = []

    # Overview
    overview_kpis = kpi_list[:3] if kpi_list else ["Primary KPI"]
    sections.append({
        "section": "Overview",
        "chart_types": ["KPI cards", "Line chart (trend over time)"],
        "kpi_cards": overview_kpis,
        "filters": ["Date range picker", "Business unit / segment"],
        "drill_down": "Click KPI card → see daily breakdown",
    })
    if any(kw in domain.lower() for kw in ["ecommerce", "retail", "shop"]):
        sections.append({
            "section": "Revenue & Orders",
            "chart_types": ["Bar chart (revenue by category)", "Funnel chart (order stages)"],
            "kpi_cards": ["Total Revenue", "AOV", "Order Cancellation Rate"],
            "filters": ["Product category", "Region", "Order status"],
            "drill_down": "Click category → see product-level breakdown",
        })
    elif any(kw in domain.lower() for kw in ["saas", "subscription", "software"]):
        sections.append({
            "section": "Subscription Health",
            "chart_types": ["Area chart (MRR over time)", "Cohort table (retention)"],
            "kpi_cards": ["MRR", "Churn Rate", "NRR"],
            "filters": ["Plan tier", "Cohort month", "Region"],
            "drill_down": "Click cohort → see individual plan breakdown",
        })
    else:
        sections.append({
            "section": "Performance Summary",
            "chart_types": ["Bar chart (top metrics)", "Heatmap (activity by day)"],
            "kpi_cards": kpi_list[1:4] if len(kpi_list) > 1 else ["Key Metric"],
            "filters": ["Date range", "Category"],
            "drill_down": "Click bar → record-level detail table",
        })
    if any(c in col_list for c in ["user_id", "customer_id", "account_id"]):
        sections.append({
            "section": "User / Customer Analysis",
            "chart_types": ["Scatter plot (engagement vs value)", "Table (top users)"],
            "kpi_cards": ["Active Users", "Repeat Rate"],
            "filters": ["Segment", "Cohort", "Activity level"],
            "drill_down": "Click user → see individual transaction history",
        })
 
    if any(c in col_list for c in ["event_type", "event", "action"]):
        sections.append({
            "section": "Event Stream Analysis",
            "chart_types": ["Stacked bar (events by type)", "Line (event volume over time)"],
            "kpi_cards": ["Events per User", "Failed Event Rate"],
            "filters": ["Event type", "Platform", "Time window"],
            "drill_down": "Click event type → see session-level detail",
        })

    sections.append({
        "section": "Data Quality Monitor",
        "chart_types": ["Table (missing value counts)", "Bar (duplicate rate by table)"],
        "kpi_cards": ["Missing Value %", "Duplicate Rate"],
        "filters": ["Column name", "Date"],
        "drill_down": "Click column → see null value distribution",
    })

    result = {
        "dashboard_name": f"{domain.title()} Analytics Dashboard",
        "total_sections": len(sections),
        "global_filters": ["Date range", "Region", "Segment", "Data source"],
        "sections": sections,
        "export_options": ["PDF", "CSV", "Slack alert"],
        "refresh_schedule": "Recommended: daily batch or real-time streaming",
    }
    return json.dumps(result, indent=2)

@tool("validate_sql_safety")
def validate_sql_safety(sql_query: str) -> str:
    """
    Check whether a SQL query is safe to execute.

    Rules enforced:
      - Only SELECT statements are allowed.
      - DELETE, UPDATE, DROP, ALTER, INSERT, CREATE, TRUNCATE are blocked.
      - Warns if SELECT * is used (no column projection).
      - Warns if no LIMIT clause is present (could return huge results).
      - Warns if no date/time filter is detected on event data queries.
      - Uses sqlglot for AST-level parsing (more reliable than regex).

    Returns a JSON string with: is_safe, blocked_reason, warnings, cleaned_query.
    """
    if not sql_query or not sql_query.strip():
        return json.dumps({
            "is_safe": False,
            "blocked_reason": "Empty SQL query provided.",
            "warnings": [],
            "original_query": sql_query,
        }, indent=2)

    warnings = []
    blocked_reason = None
    query_upper = sql_query.strip().upper()
 
    # Block list (regex pre-check before AST)
    blocked_keywords = ["DELETE", "UPDATE", "DROP", "ALTER", "INSERT",
                        "CREATE", "MERGE", "TRUNCATE", "EXEC", "EXECUTE"]
    for kw in blocked_keywords:
        if re.search(rf"\b{kw}\b", query_upper):
            blocked_reason = f"Blocked keyword detected: '{kw}'. Only SELECT queries are allowed."
            break
    if not blocked_reason:
        try:
            parsed = sqlglot.parse(sql_query)
            for statement in parsed:
                # Confirm it's a SELECT statement at AST level
                if not isinstance(statement, sqlglot.expressions.Select):
                    blocked_reason = (
                        f"Non-SELECT statement detected at AST level: "
                        f"{type(statement).__name__}. Blocked for safety."
                    )
                    break
        except Exception as parse_err:
            warnings.append(f"SQL parsing warning (sqlglot): {str(parse_err)}")

    if not blocked_reason:
        if re.search(r"\bSELECT\s+\*", query_upper):
            warnings.append(
                "SELECT * used — consider specifying columns explicitly "
                "for better performance and to avoid exposing sensitive fields."
            )

        if not re.search(r"\bLIMIT\b", query_upper):
            warnings.append(
                "No LIMIT clause detected — query could return very large results. "
                "Add LIMIT 1000 or similar."
            )

        date_keywords = ["date", "timestamp", "created_at", "updated_at",
                         "event_date", "order_date", "day", "month", "year"]
        has_date_filter = any(kw in sql_query.lower() for kw in date_keywords)
        if not has_date_filter:
            warnings.append(
                "No date/time filter detected. For event or transaction tables, "
                "always filter by a date range to avoid full table scans."
            )

    result = {
        "is_safe": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "warnings": warnings,
        "original_query": sql_query,
        "recommendation": (
            "Query is safe to execute (review warnings above)."
            if not blocked_reason
            else "Do NOT execute this query. Fix the blocked issue first."
        ),
    }
    return json.dumps(result, indent=2)

@tool("explain_query_result")
def explain_query_result(metric: str, trend: str, change_percent: float) -> str:
    """
    Convert a numeric query result into a plain English business explanation.
    Returns a JSON string with: explanation, possible_causes, suggested_actions.
    """
    metric_display = metric.replace("_", " ").title()
    direction = trend.lower()
    change_abs = abs(change_percent)

    if change_abs >= 25:
        severity = "significant"
    elif change_abs >= 10:
        severity = "moderate"
    else:
        severity = "minor"

    if direction == "increasing":
        explanation = (
            f"{metric_display} increased by {change_abs:.1f}% ({severity} growth). "
            f"This is a positive signal for the business."
        )
        emoji = "📈"
    elif direction == "decreasing":
        explanation = (
            f"{metric_display} decreased by {change_abs:.1f}% ({severity} decline). "
            f"This warrants investigation to identify root causes."
        )
        emoji = "📉"
    else:
        explanation = (
            f"{metric_display} remained stable (change: {change_percent:.1f}%). "
            f"No major shifts detected in this period."
        )
        emoji = "➡️"

    metric_lower = metric.lower()
    possible_causes = []
    suggested_actions = []

    if any(kw in metric_lower for kw in ["revenue", "sales", "amount"]):
        if direction == "decreasing":
            possible_causes = [
                "Lower customer acquisition this period",
                "Reduced repeat purchase rate",
                "Seasonal demand drop",
                "Increased competition or pricing pressure",
                "Product or checkout issues affecting conversion",
            ]
            suggested_actions = [
                "Check new customer acquisition numbers vs. prior period",
                "Review cart abandonment rate",
                "Analyze discount/promotion effectiveness",
                "Compare cohort retention for recent vs. older customers",
            ]
        else:
            possible_causes = ["Successful marketing campaign", "Seasonal uplift",
                                "New product launch", "Pricing increase"]
            suggested_actions = ["Identify top-performing campaigns to replicate",
                                  "Check if growth is from new vs. returning customers"]

    elif any(kw in metric_lower for kw in ["churn", "cancellation"]):
        if direction == "increasing":
            possible_causes = [
                "Product dissatisfaction or missing features",
                "Competitive alternatives becoming available",
                "Pricing changes or economic pressure on customers",
                "Onboarding failures for recent cohorts",
            ]
            suggested_actions = [
                "Run exit surveys with churned customers",
                "Segment churn by plan tier and cohort",
                "Review recent product changes for negative impact",
                "Increase customer success outreach for at-risk accounts",
            ]
        else:
            possible_causes = ["Improved onboarding", "Customer success interventions",
                                "Product improvement", "Pricing adjustments"]
            suggested_actions = ["Identify which retention initiative had the most impact"]

    elif any(kw in metric_lower for kw in ["user", "customer", "active"]):
        if direction == "decreasing":
            possible_causes = ["Marketing spend reduction", "Seasonal effect",
                                "Technical issues reducing access", "Competitor launches"]
            suggested_actions = ["Review acquisition channel performance",
                                  "Check for technical incidents in the period",
                                  "Compare against same period last year (seasonality)"]
        else:
            possible_causes = ["Successful campaign", "Word-of-mouth growth", "Product improvements"]
            suggested_actions = ["Double down on the top acquisition channel"]

    else:
        # Generic fallback
        possible_causes = [
            "External market changes",
            "Internal product or operational changes",
            "Data collection issues (check pipeline)",
            "Seasonal patterns",
        ]
        suggested_actions = [
            "Compare with the same period last year",
            "Segment the metric by key dimensions to isolate the driver",
            "Check for data pipeline issues if the change seems extreme",
        ]

    result = {
        "metric": metric_display,
        "trend": direction,
        "change_percent": change_percent,
        "severity": severity,
        "emoji": emoji,
        "explanation": explanation,
        "possible_causes": possible_causes,
        "suggested_actions": suggested_actions,
    }
    return json.dumps(result, indent=2)