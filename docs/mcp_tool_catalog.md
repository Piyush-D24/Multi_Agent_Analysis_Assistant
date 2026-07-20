# MCP Tool Catalog

**Server name:** `analytics_mcp_server`
**Total tools:** 10
**Transport:** stdio (launched as a subprocess by `MCPServerAdapter` in `app.py`)

This catalog documents every tool exposed by the MCP server, exactly as
registered in `mcp_server/server.py`.

---

## Tool 1 — `mcp_profile_csv`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/csv_profile_tools.py` |
| **Used by** | Data Analyst Agent, Data Scientist Agent |
| **Purpose** | Full structural profile of a CSV — shape, types, nulls, duplicates, numeric stats, samples |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | Yes | Filename only, e.g. `'events_sample.csv'`. Must be inside `mcp_server/sample_data/` |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"success"` or `"error"` |
| `rows` | int | Total row count |
| `columns` | int | Total column count |
| `column_names` | list[str] | All column names |
| `data_types` | dict | Column → dtype mapping |
| `missing_values` | dict | Column → null count (only nonzero) |
| `total_missing_cells` | int | Sum of all nulls across the dataset |
| `missing_percent` | float | % of all cells that are null |
| `duplicate_rows` | int | Count of fully duplicate rows |
| `numeric_stats` | dict | Per numeric column: min, max, mean, std, nulls |
| `sample_rows` | list[dict] | First 5 rows |

**Safety:** Path traversal blocked, `.csv` extension enforced, 50 MB size limit.

---

## Tool 2 — `mcp_run_duckdb_query`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/sql_tools.py` |
| **Used by** | Data Analyst Agent |
| **Purpose** | Execute a read-only SQL SELECT query against a local CSV using DuckDB |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `sql_query` | string | Yes | A SELECT query. Use `data` as the table alias, e.g. `SELECT * FROM data LIMIT 10` |
| `file_name` | string | Yes | CSV filename in `sample_data/` |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"success"`, `"error"`, or `"blocked"` |
| `columns` | list[str] | Result column names |
| `row_count` | int | Number of rows returned |
| `truncated` | bool | `true` if results were capped at 500 rows |
| `rows` | list[dict] | The actual query results |

**Safety:** Blocks `DELETE`, `UPDATE`, `DROP`, `ALTER`, `INSERT`, `CREATE`, `MERGE`, `TRUNCATE`, `EXEC`, `GRANT`, `REVOKE`, `ATTACH`, `DETACH`, `COPY`. Results hard-capped at 500 rows.

---

## Tool 3 — `mcp_validate_sql`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/sql_tools.py` |
| **Used by** | Supervisor Agent, Data Analyst Agent |
| **Purpose** | Deep SQL safety validation using both regex and sqlglot AST parsing, before a query is ever executed |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `sql_query` | string | Yes | The SQL query to validate |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `is_safe` | bool | `true` if no blocking issue found |
| `blocked_reason` | string or null | Why the query was blocked, if applicable |
| `warnings` | list[str] | Non-blocking issues: `SELECT *`, missing `LIMIT`, missing `WHERE`, missing date filter, high complexity |
| `ast_statement_type` | string | The sqlglot-parsed statement class name |
| `recommendation` | string | Plain-English verdict |

**Safety:** Same blocked keyword list as Tool 2, plus AST-level confirmation the statement is a `Select` type.

---

## Tool 4 — `mcp_detect_data_quality_issues`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/csv_profile_tools.py` |
| **Used by** | Data Analyst Agent, Data Scientist Agent |
| **Purpose** | Runs 6 independent quality checks and returns a severity-tagged issue list |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | Yes | CSV filename in `sample_data/` |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"success"` or `"error"` |
| `total_issues` | int | Count of all issues found |
| `severity_summary` | dict | Count of issues per severity level |
| `overall` | string | One-line verdict based on worst severity present |
| `issues` | list[dict] | Each with `check`, `column`, `severity`, `detail`, `fix` |

**Checks performed:** Missing values, duplicate rows, constant columns, high-cardinality strings (>50 unique), negative values in positive-only columns, statistical outliers (Z-score > 3).

---

## Tool 5 — `mcp_generate_kpi_catalog`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/kpi_tools.py` |
| **Used by** | Data Analyst Agent, Supervisor Agent |
| **Purpose** | Generate a domain-specific KPI catalog, optionally filtered by dataset columns |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `domain` | string | Yes | One of: `ecommerce`, `saas`, `fintech`, `events`, or any other (falls back to general) |
| `file_name` | string | No | Optional CSV to auto-detect columns and filter relevant KPIs |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `matched_library` | string | Which domain library was used |
| `detected_columns` | list[str] | Columns read from file, if provided |
| `total_kpis` | int | Count of KPIs returned |
| `kpis` | list[dict] | Each with `name`, `formula`, `grain`, `business_use`, `required_columns`, `alert_threshold_hint` |

---

## Tool 6 — `mcp_recommend_ml_use_cases`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/ml_tools.py` |
| **Used by** | Data Scientist Agent, Supervisor Agent |
| **Purpose** | Match dataset columns against a library of 8 known ML use cases and rank by relevance |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | No* | CSV to auto-detect columns |
| `columns` | string | No* | Comma-separated columns, used if `file_name` not given |
| `domain` | string | No | Domain hint, default `"general"` |

*One of `file_name` or `columns` must be provided.

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `total_use_cases_matched` | int | Number of use cases that scored > 0 |
| `ml_use_cases` | list[dict] | Each with `use_case`, `problem_type`, `required_columns`, `business_value`, `complexity`, `estimated_timeline`, `relevance_score` |

---

## Tool 7 — `mcp_feature_engineering_suggestions`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/ml_tools.py` |
| **Used by** | Data Scientist Agent |
| **Purpose** | Generate concrete, formula-backed feature ideas grouped by detected column type |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | No* | CSV to auto-detect columns |
| `columns` | string | No* | Comma-separated columns |

*One of the two required.

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `total_feature_groups` | int | Number of feature groups generated |
| `features` | list[dict] | Each group has `group`, `source_columns`, `features` (list of `name`, `formula`, `reason`) |
| `critical_reminders` | list[str] | Always includes the data-leakage warning |

---

## Tool 8 — `mcp_anomaly_detection_summary`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/ml_tools.py` |
| **Used by** | Data Scientist Agent, Data Analyst Agent |
| **Purpose** | Statistical anomaly detection on a single numeric column |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | Yes | CSV filename in `sample_data/` |
| `numeric_column` | string | Yes | The column to analyse — must be numeric |
| `method` | string | No | `"zscore"` (default), `"iqr"`, or `"both"` |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `total_anomalies` | int | Count of flagged rows |
| `anomaly_percent` | float | % of data flagged |
| `summary` | string | Plain-English verdict |
| `method_results` | dict | Per-method thresholds and anomaly indices |
| `anomaly_sample_rows` | list[dict] | Up to 20 sample anomalous rows |

---

## Tool 9 — `mcp_create_data_dictionary`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/kpi_tools.py` |
| **Used by** | Data Analyst Agent, Supervisor Agent |
| **Purpose** | Infer the business meaning of each column and produce a data dictionary |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `file_name` | string | Yes | CSV filename in `sample_data/` |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `columns` | list[dict] | Each with `name`, `dtype`, `possible_meaning`, `sample_values`, `null_count`, `unique_count`, `is_likely_id`, `is_likely_date`, `is_likely_categorical` |

---

## Tool 10 — `mcp_generate_report_markdown`

| Field | Value |
|---|---|
| **Source file** | `mcp_server/tools/report_tools.py` |
| **Used by** | Supervisor Agent (final assembly step) |
| **Purpose** | Combine outputs from all other tools into one markdown report with all 10 required sections |

**Input Parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `dataset_summary` | string | No | Output of `mcp_profile_csv` |
| `data_quality` | string | No | Output of `mcp_detect_data_quality_issues` |
| `kpis` | string | No | Output of `mcp_generate_kpi_catalog` |
| `ml_use_cases` | string | No | Output of `mcp_recommend_ml_use_cases` |
| `feature_ideas` | string | No | Output of `mcp_feature_engineering_suggestions` |
| `dashboard_layout` | string | No | Output of `generate_dashboard_layout` (function tool) |
| `risks` | string | No | Free-text risk summary |
| `agent_work_summary` | string | No | Which agents/tools ran |
| `domain` | string | No | Report header label, default `"General Analytics"` |
| `file_name` | string | No | Dataset filename for report header |

**Output Fields**

| Field | Type | Description |
|---|---|---|
| `section_count` | int | How many of the 10 sections are present (always 10 — placeholders fill gaps) |
| `all_sections_present` | bool | Always `true` by design |
| `markdown_report` | string | The full formatted report |

---

## Quick Reference — Which Agent Uses Which Tool

| Tool | Supervisor | Analyst | Scientist |
|---|:---:|:---:|:---:|
| `mcp_profile_csv` | | ✅ | ✅ |
| `mcp_run_duckdb_query` | | ✅ | |
| `mcp_validate_sql` | ✅ | ✅ | |
| `mcp_detect_data_quality_issues` | | ✅ | ✅ |
| `mcp_generate_kpi_catalog` | ✅ | ✅ | |
| `mcp_recommend_ml_use_cases` | ✅ | | ✅ |
| `mcp_feature_engineering_suggestions` | | | ✅ |
| `mcp_anomaly_detection_summary` | | ✅ | ✅ |
| `mcp_create_data_dictionary` | ✅ | ✅ | |
| `mcp_generate_report_markdown` | ✅ | | |