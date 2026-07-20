import re
from pathlib import Path
from typing import Any, Dict
import duckdb
import sqlglot

SAFE_DATA_DIR = (Path(__file__).parent.parent / "sample_data").resolve()

BLOCKED_KEYWORDS = [
    "DELETE", "UPDATE", "DROP", "ALTER", "INSERT",
    "CREATE", "MERGE", "TRUNCATE", "EXEC", "EXECUTE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "COPY",
]

def _block_check(sql: str) -> str | None:
    """
    Return the first blocked keyword found, or None if query is safe.
    Uses word-boundary regex to avoid false positives.
    e.g. 'created_at' should NOT trigger 'CREATE'.
    """
    sql_upper = sql.upper()
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", sql_upper):
            return kw
    return None

def _safe_csv_path(file_name: str) -> str:
    """
    Resolve a CSV path and confirm it stays inside sample_data/.
    Returns the absolute path string (used in DuckDB SQL).
    """
    resolved = (SAFE_DATA_DIR / file_name).resolve()
    if not str(resolved).startswith(str(SAFE_DATA_DIR)):
        raise ValueError(f"Access denied: '{file_name}' is outside the allowed data directory.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    return str(resolved)

def mcp_run_duckdb_query(sql_query: str, file_name: str) -> Dict[str, Any]:
    """
    Run a read-only SQL query on a local CSV file using DuckDB.
 
    DuckDB can query CSV files directly as if they were tables.
    The CSV is referenced in the SQL as read_csv_auto('path/to/file.csv').
 
    Used by: Data Analyst Agent.
 
    Safety:
      - Only SELECT queries are allowed.
      - All write keywords (DELETE, DROP, etc.) are blocked.
      - File access is restricted to sample_data/ only.
      - Results are capped at 500 rows to protect context window.
 
    Args:
        sql_query:  A SELECT SQL query. Use 'data' as the table alias.
                    The tool wraps the CSV path automatically.
                    Example: "SELECT * FROM data LIMIT 10"
        file_name:  CSV filename inside sample_data/, e.g. 'events_sample.csv'.
 
    Returns:
        Dict with columns, rows (list of dicts), row_count, and truncated flag.
    """
    try:
        blocked = _block_check(sql_query)
        if blocked:
            return {
                "status": "blocked",
                "error": f"Keyword '{blocked}' is not allowed. Only SELECT queries are permitted.",
            }
 
        csv_path = _safe_csv_path(file_name)
        actual_sql = re.sub(
            r"\bFROM\s+data\b",
            f"FROM read_csv_auto('{csv_path}')",
            sql_query,
            flags=re.IGNORECASE,
        )

        con = duckdb.connect(database=":memory:", read_only=False)
 
        # Add row cap safety net at execution level
        result = con.execute(actual_sql).fetchdf()
        con.close()

        truncated = False
        if len(result) > 500:
            result = result.head(500)
            truncated = True

        return {
            "status": "success",
            "file": file_name,
            "original_query": sql_query,
            "executed_query": actual_sql,
            "columns": list(result.columns),
            "row_count": len(result),
            "truncated": truncated,
            "truncation_note": "Results capped at 500 rows." if truncated else None,
            "rows": result.to_dict(orient="records"),
        }

    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "error": str(e)}
    except duckdb.Error as e:
        return {"status": "error", "error": f"DuckDB execution error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Query failed: {str(e)}"}

def mcp_validate_sql(sql_query: str) -> Dict[str, Any]:
    """
    Validate a SQL query for safety and best practices before execution.
 
    Used by: Supervisor Agent, Data Analyst Agent.
 
    This is a deeper validation than the function tool validate_sql_safety.
    It uses sqlglot AST parsing for statement-type detection,
    plus regex for keyword blocking, and heuristics for warnings.
    Returns:
        is_safe (bool), blocked_reason, warnings (list), ast_statement_type.
    """
    if not sql_query or not sql_query.strip():
        return {
            "is_safe": False,
            "blocked_reason": "Empty query.",
            "warnings": [],
            "ast_statement_type": None,
        }

    warnings = []
    blocked_reason = None
    ast_type = "unknown"

    blocked_kw = _block_check(sql_query)
    if blocked_kw:
        blocked_reason = f"Blocked keyword: '{blocked_kw}'. Only SELECT is allowed."

    if not blocked_reason:
        try:
            parsed_statements = sqlglot.parse(sql_query)
            for stmt in parsed_statements:
                ast_type = type(stmt).__name__
                if not isinstance(stmt, sqlglot.expressions.Select):
                    blocked_reason = (
                        f"Non-SELECT statement at AST level: '{ast_type}'. Blocked."
                    )
                    break
        except Exception as parse_err:
            warnings.append(f"sqlglot parse warning: {str(parse_err)}")

    if not blocked_reason:
        sql_upper = sql_query.upper()

        if re.search(r"\bSELECT\s+\*", sql_upper):
            warnings.append(
                "SELECT * used — specify column names explicitly to avoid "
                "returning sensitive data and to improve performance."
            )

        if not re.search(r"\bLIMIT\b", sql_upper):
            warnings.append(
                "No LIMIT clause — add LIMIT 1000 to prevent large result sets."
            )

        if not re.search(r"\bWHERE\b", sql_upper):
            warnings.append(
                "No WHERE clause — query will scan the entire table. "
                "Add filters to improve performance."
            )

        date_terms = ["date", "timestamp", "created_at", "updated_at",
                      "day", "month", "year", "event_date", "order_date"]
        if not any(term in sql_query.lower() for term in date_terms):
            warnings.append(
                "No date/time filter detected — for event or transaction tables, "
                "always filter by a date range."
            )

        subquery_depth = sql_query.upper().count("SELECT")
        if subquery_depth > 3:
            warnings.append(
                f"High query complexity: {subquery_depth} SELECT keywords detected. "
                "Consider breaking into CTEs (WITH clauses) for readability."
            )
    return {
        "is_safe": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "warnings": warnings,
        "ast_statement_type": ast_type,
        "warning_count": len(warnings),
        "original_query": sql_query,
        "recommendation": (
            "Safe to execute — address warnings for best results."
            if not blocked_reason
            else "DO NOT execute — fix the blocked issue first."
        ),
    }