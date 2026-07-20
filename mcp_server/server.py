from mcp.server.fastmcp import FastMCP
from tools.csv_profile_tools import (
    mcp_detect_data_quality_issues,
    mcp_profile_csv,
)
from tools.kpi_tools import (
    mcp_create_data_dictionary,
    mcp_generate_kpi_catalog,
)
from tools.ml_tools import (
    mcp_anomaly_detection_summary,
    mcp_feature_engineering_suggestions,
    mcp_recommend_ml_use_cases,
)
from tools.report_tools import mcp_generate_report_markdown
from tools.sql_tools import mcp_run_duckdb_query, mcp_validate_sql

mcp = FastMCP(
    name="analytics_mcp_server",
    instructions=(
        "This MCP server exposes 10 analytics tools for data profiling, "
        "SQL validation, KPI generation, ML use case recommendation, "
        "feature engineering, anomaly detection, and report generation. "
        "All file access is restricted to the sample_data/ directory."
    ),
)

@mcp.tool()
def tool_mcp_profile_csv(file_name: str) -> dict:
    """
    Profile a CSV file from sample_data/ and return full statistics.

    Returns row count, column count, data types, missing values,
    duplicate rows, numeric stats, and 5 sample rows.

    Used by: Data Analyst Agent, Data Scientist Agent.
    """
    return mcp_profile_csv(file_name)

@mcp.tool()
def tool_mcp_run_duckdb_query(sql_query: str, file_name: str) -> dict:
    """
    Run a read-only SQL SELECT query on a local CSV using DuckDB.
 
    Reference the CSV as 'data' in your SQL.
    Example: SELECT event_type, COUNT(*) FROM data GROUP BY event_type LIMIT 20
 
    Only SELECT is allowed. DELETE, DROP, UPDATE etc. are blocked.
    Results are capped at 500 rows.
 
    Used by: Data Analyst Agent.
    """
    return mcp_run_duckdb_query(sql_query, file_name)

@mcp.tool()
def tool_mcp_validate_sql(sql_query: str) -> dict:
    """
    Validate a SQL query for safety and best practices.

    Checks: SELECT-only, no blocked keywords, LIMIT present,
    WHERE clause present, no SELECT *, no date filter missing.

    Used by: Supervisor Agent, Data Analyst Agent.
    """
    return mcp_validate_sql(sql_query)

@mcp.tool()
def tool_mcp_detect_data_quality_issues(file_name: str) -> dict:
    """
    Detect data quality problems in a CSV file.

    Checks: missing values, duplicate rows, constant columns,
    high-cardinality strings, negative values in positive columns,
    and statistical outliers (Z-score > 3).

    Used by: Data Analyst Agent, Data Scientist Agent.
    """
    return mcp_detect_data_quality_issues(file_name)

@mcp.tool()
def tool_mcp_generate_kpi_catalog(domain: str, file_name: str = "") -> dict:
    """
    Generate a KPI catalog for the given business domain.
 
    Optionally reads column names from a CSV to filter KPIs
    that match available data.
 
    Domains: ecommerce, saas, fintech, healthcare, events, general.
 
    Used by: Data Analyst Agent, Supervisor Agent.
    """
    return mcp_generate_kpi_catalog(domain, file_name)

@mcp.tool()
def tool_mcp_recommend_ml_use_cases(
    file_name: str = "",
    columns: str = "",
    domain: str = "general",
) -> dict:
    """
    Recommend ML use cases based on dataset columns.
 
    Provide either file_name (auto-reads CSV) or columns
    as a comma-separated string.
 
    Each use case includes: problem_type, required_columns,
    business_value, complexity, timeline.
 
    Used by: Data Scientist Agent, Supervisor Agent.
    """
    return mcp_recommend_ml_use_cases(file_name, columns, domain)

@mcp.tool()
def tool_mcp_feature_engineering_suggestions(
    file_name: str = "",
    columns: str = "",
) -> dict:
    """
    Suggest feature engineering ideas based on dataset columns.
 
    Groups features by type: time-based, user aggregation,
    transaction, and behavioural. Each idea includes a formula
    and the reason it is useful for ML.
 
    Used by: Data Scientist Agent.
    """
    return mcp_feature_engineering_suggestions(file_name, columns)

@mcp.tool()
def tool_mcp_anomaly_detection_summary(
    file_name: str,
    numeric_column: str,
    method: str = "zscore",
) -> dict:
    """
    Detect anomalies in a numeric column using statistical methods.
 
    Methods:
      zscore — flags |Z-score| > 3 (assumes normal distribution)
      iqr    — flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
      both   — runs both and returns union of anomalies
 
    Used by: Data Scientist Agent, Data Analyst Agent.
    """
    return mcp_anomaly_detection_summary(file_name, numeric_column, method)

@mcp.tool()
def tool_mcp_create_data_dictionary(file_name: str) -> dict:
    """
    Generate a data dictionary from a CSV file.
 
    For each column returns: name, dtype, inferred meaning,
    sample values, null count, unique count, and flags for
    ID columns, date columns, and categorical columns.
 
    Used by: Data Analyst Agent, Supervisor Agent.
    """
    return mcp_create_data_dictionary(file_name)

@mcp.tool()
def tool_mcp_generate_report_markdown(
    dataset_summary: str = "",
    data_quality: str = "",
    kpis: str = "",
    ml_use_cases: str = "",
    feature_ideas: str = "",
    dashboard_layout: str = "",
    risks: str = "",
    agent_work_summary: str = "",
    domain: str = "General Analytics",
    file_name: str = "",
) -> dict:
    """
    Combine all tool outputs into a final structured markdown report.
 
    Always produces all 10 required sections. Empty inputs get
    placeholder text so the response structure validator always passes.
 
    Used by: Supervisor Agent as the final assembly step.
 
    Arguments:
        dataset_summary:    JSON string from mcp_profile_csv.
        data_quality:       JSON string from mcp_detect_data_quality_issues.
        kpis:               JSON string from mcp_generate_kpi_catalog.
        ml_use_cases:       JSON string from mcp_recommend_ml_use_cases.
        feature_ideas:      JSON string from mcp_feature_engineering_suggestions.
        dashboard_layout:   JSON string from generate_dashboard_layout.
        risks:              Free-text risk summary.
        agent_work_summary: Agent delegation log.
        domain:             Business domain label.
        file_name:          Dataset filename for report header.
    """
    return mcp_generate_report_markdown(
        dataset_summary=dataset_summary,
        data_quality=data_quality,
        kpis=kpis,
        ml_use_cases=ml_use_cases,
        feature_ideas=feature_ideas,
        dashboard_layout=dashboard_layout,
        risks=risks,
        agent_work_summary=agent_work_summary,
        domain=domain,
        file_name=file_name,
    )
    
if __name__ == "__main__":
    mcp.run()