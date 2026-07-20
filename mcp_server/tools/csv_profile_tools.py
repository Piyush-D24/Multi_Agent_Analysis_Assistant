from pathlib import Path
from typing import Any, Dict

import pandas as pd

SAFE_DATA_DIR = (Path(__file__).parent.parent / "sample_data").resolve()

def _safe_load_csv(file_name: str) -> pd.DataFrame:
    """
    Load a CSV from sample_data/ only.
    Raises ValueError on path traversal attempts.
    Raises FileNotFoundError if the file doesn't exist.
    """
    resolved = (SAFE_DATA_DIR / file_name).resolve()

    if not str(resolved).startswith(str(SAFE_DATA_DIR)):
        raise ValueError(
            f"Access denied: '{file_name}' is outside the allowed data directory."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"File not found in sample_data/: {file_name}")
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"Only .csv files are supported. Got: {resolved.suffix}")
    if resolved.stat().st_size > 50 * 1024 * 1024:  # 50 MB limit
        raise ValueError(f"File '{file_name}' exceeds the 50 MB size limit.")

    return pd.read_csv(resolved)

def mcp_profile_csv(file_name: str) -> Dict[str, Any]:
    """
    Read a CSV file from sample_data/ and return a full profile.
    Used by: Data Analyst Agent, Data Scientist Agent.
    Returns:
        rows, columns, column_names, data_types, missing_values,
        duplicates, numeric_stats, sample_rows.
    """
    try:
        df = _safe_load_csv(file_name)

        # Missing values per column (only report non-zero)
        missing = {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        }

        # Numeric summary stats
        numeric_stats = {}
        for col in df.select_dtypes(include="number").columns:
            numeric_stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "nulls": int(df[col].isnull().sum()),
            }

        return {
            "status": "success",
            "file": file_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing,
            "total_missing_cells": int(df.isnull().sum().sum()),
            "missing_percent": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_stats": numeric_stats,
            "sample_rows": df.head(5).to_dict(orient="records"),
        }

    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Profiling failed: {str(e)}"}

def mcp_detect_data_quality_issues(file_name: str) -> Dict[str, Any]:
    """
    Detect common data quality problems in a CSV file.
    Used by: Data Analyst Agent, Data Scientist Agent.
    Checks:
      - Missing values (per column and total)
      - Duplicate rows
      - Constant columns (zero variance — useless for ML)
      - High-cardinality string columns (> 50 unique values)
      - Negative values in columns that should be positive
        (detected by column name keywords: amount, price, count, age, qty)
      - Outliers in numeric columns (Z-score > 3)
    Returns a structured dict with all detected issues and severity levels.
    """
    try:
        df = _safe_load_csv(file_name)
        issues = []

        # Missing values
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                null_pct = round(null_count / len(df) * 100, 2)
                severity = "critical" if null_pct > 30 else "high" if null_pct > 10 else "medium"
                issues.append({
                    "check": "Missing Values",
                    "column": col,
                    "severity": severity,
                    "detail": f"{null_count} nulls ({null_pct}% of rows)",
                    "fix": f"Impute with median/mode or drop rows where '{col}' is null.",
                })

        # Duplicate rows
        dupe_count = int(df.duplicated().sum())
        if dupe_count > 0:
            issues.append({
                "check": "Duplicate Rows",
                "column": "ALL",
                "severity": "medium",
                "detail": f"{dupe_count} fully duplicate rows detected.",
                "fix": "Call df.drop_duplicates() before analysis or training.",
            })

        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append({
                    "check": "Constant Column",
                    "column": col,
                    "severity": "low",
                    "detail": f"Column '{col}' has only 1 unique value: '{df[col].iloc[0] if len(df) > 0 else 'N/A'}'",
                    "fix": f"Drop '{col}' — it carries zero information.",
                })

        # High-cardinality string columns
        for col in df.select_dtypes(include="object").columns:
            n_unique = df[col].nunique()
            if n_unique > 50:
                issues.append({
                    "check": "High Cardinality",
                    "column": col,
                    "severity": "medium",
                    "detail": f"'{col}' has {n_unique} unique string values.",
                    "fix": "Use target encoding or hash encoding instead of one-hot encoding.",
                })

        # Negative values in positive-only columns
        positive_keywords = ["amount", "price", "cost", "revenue", "count",
                             "qty", "quantity", "age", "duration", "size"]
        for col in df.select_dtypes(include="number").columns:
            if any(kw in col.lower() for kw in positive_keywords):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    issues.append({
                        "check": "Negative Values in Positive Column",
                        "column": col,
                        "severity": "high",
                        "detail": f"'{col}' has {neg_count} negative values. Expected all positive.",
                        "fix": "Investigate source — may indicate data entry errors or refunds.",
                    })

        # Outliers (Z-score > 3)
        for col in df.select_dtypes(include="number").columns:
            col_std = df[col].std()
            if col_std == 0:
                continue
            z_scores = ((df[col] - df[col].mean()) / col_std).abs()
            outlier_count = int((z_scores > 3).sum())
            if outlier_count > 0:
                issues.append({
                    "check": "Statistical Outliers",
                    "column": col,
                    "severity": "low",
                    "detail": f"'{col}' has {outlier_count} values with |Z-score| > 3.",
                    "fix": f"Cap with IQR clipping or apply log transform to '{col}'.",
                })

        # Summary
        severity_counts = {}
        for issue in issues:
            s = issue["severity"]
            severity_counts[s] = severity_counts.get(s, 0) + 1

        return {
            "status": "success",
            "file": file_name,
            "rows_checked": len(df),
            "columns_checked": len(df.columns),
            "total_issues": len(issues),
            "severity_summary": severity_counts,
            "overall": (
                "Critical issues — fix before any analysis or training."
                if severity_counts.get("critical", 0) > 0
                else "High-severity issues detected — review before proceeding."
                if severity_counts.get("high", 0) > 0
                else "Low/medium issues only — safe to proceed with care."
                if issues
                else "No data quality issues detected."
            ),
            "issues": issues,
        }

    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Quality check failed: {str(e)}"}