import json
from pathlib import Path
import pandas as pd
from crewai.tools import tool

SAFE_DATA_DIR = Path(__file__).parent.parent / "mcp_server" / "sample_data"

def _safe_resolve(file_path: str) -> Path:
    """Resolve a path and ensure it stays inside sample_data/."""
    resolved = (SAFE_DATA_DIR / file_path).resolve()
    if not str(resolved).startswith(str(SAFE_DATA_DIR.resolve())):
        raise ValueError(f"Access denied: '{file_path}' is outside the allowed data directory.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved

@tool("recommend_ml_problem_type")
def recommend_ml_problem_type(user_goal: str, columns: str) -> str:
    """
    Classify the machine learning use case based on the user's goal and
    the dataset columns available.
    Possible outputs:
      classification, regression, clustering, forecasting,
      anomaly_detection, recommendation, ranking
    Returns a JSON string with: problem_type, target_variable, reason,
    suitable_algorithms, typical_timeline.
    """
    goal_lower = user_goal.lower()
    col_list = [c.strip().lower() for c in columns.split(",")]

    rules = [
        (
            ["churn", "fraud", "default", "click", "convert",
             "yes/no", "binary", "whether", "classify", "detect fraud"],
            "classification",
            "binary target column (e.g. churn, is_fraud, converted)",
            "The goal involves predicting a yes/no or categorical outcome.",
        ),
        (
            ["forecast", "next month", "next week", "future", "time series",
             "trend", "seasonal", "demand", "predict sales"],
            "forecasting",
            "time-series target (e.g. sales, demand, visits)",
            "The goal involves predicting future values using historical time-series data.",
        ),
        (
            ["price", "revenue", "cost", "predict amount", "forecast value",
             "estimate", "how much", "score", "rating", "numeric"],
            "regression",
            "continuous numeric target (e.g. price, revenue, score)",
            "The goal involves predicting a continuous numeric value.",
        ),
        (
            ["segment", "group", "cluster", "similar", "profile",
             "identify types", "no label", "unsupervised"],
            "clustering",
            "no target — groups are discovered from data",
            "The goal is to find natural groups in data without predefined labels.",
        ),
        (
            ["anomaly", "outlier", "unusual", "abnormal", "rare event",
             "unexpected", "intrusion", "spike"],
            "anomaly_detection",
            "no supervised target — anomalies are flagged",
            "The goal is to identify rare or abnormal patterns in data.",
        ),
        (
            ["recommend", "suggestion", "personaliz", "collaborative",
             "what to show", "next product", "next item"],
            "recommendation",
            "user-item interaction data (e.g. user_id, item_id, rating)",
            "The goal is to recommend relevant items to users.",
        ),
        (
            ["rank", "prioritize", "score and sort", "learning to rank", "order by relevance"],
            "ranking",
            "relevance score or click data",
            "The goal is to rank items or results by predicted relevance.",
        ),
    ]

    matched_type = None
    matched_reason = None
    matched_target = None

    for keywords, prob_type, target_hint, reason in rules:
        if any(kw in goal_lower for kw in keywords):
            matched_type = prob_type
            matched_target = target_hint
            matched_reason = reason
            break

    if not matched_type:
        matched_type = "classification"
        matched_reason = "No clear pattern matched. Defaulting to classification as the most common ML use case."
        matched_target = "binary or multi-class target column"

    algo_map = {
        "classification": [
            "Logistic Regression (baseline)", "Random Forest",
            "XGBoost / LightGBM", "Neural Network (for high-dimensional data)",
        ],
        "regression": [
            "Linear Regression (baseline)", "Ridge / Lasso (regularized)",
            "Random Forest Regressor", "XGBoost Regressor", "Neural Network",
        ],
        "clustering": [
            "K-Means (if k is known)", "DBSCAN (for density-based clusters)",
            "Hierarchical Clustering", "Gaussian Mixture Models",
        ],
        "forecasting": [
            "ARIMA / SARIMA (statistical baseline)",
            "Prophet (good for seasonal data)",
            "LightGBM with lag features",
            "LSTM / Temporal Fusion Transformer (deep learning)",
        ],
        "anomaly_detection": [
            "Isolation Forest (tree-based, robust)",
            "Z-score / IQR (statistical, fast)",
            "DBSCAN (density-based)",
            "Autoencoder (deep learning for complex patterns)",
        ],
        "recommendation": [
            "Collaborative Filtering (user-item matrix factorization)",
            "Matrix Factorization (SVD, ALS)",
            "LightFM (hybrid: content + collaborative)",
            "Two-tower Neural Networks",
        ],
        "ranking": [
            "LambdaMART (gradient boosted ranking)",
            "RankNet (pairwise neural)",
            "XGBoost with ranking objective",
        ],
    }

    timeline_map = {
        "classification":    "4–8 weeks (data prep, baseline, tuning, evaluation)",
        "regression":        "3–6 weeks",
        "clustering":        "2–4 weeks (exploratory, no ground truth needed)",
        "forecasting":       "5–8 weeks (time-series alignment is complex)",
        "anomaly_detection": "3–5 weeks",
        "recommendation":    "6–10 weeks (needs interaction data collection)",
        "ranking":           "5–8 weeks",
    }

    result = {
        "problem_type": matched_type,
        "target_variable": matched_target,
        "reason": matched_reason,
        "suitable_algorithms": algo_map.get(matched_type, []),
        "columns_analyzed": col_list,
        "typical_timeline": timeline_map.get(matched_type, "4–6 weeks"),
    }
    return json.dumps(result, indent=2)

@tool("suggest_feature_engineering")
def suggest_feature_engineering(columns: str, data_domain: str = "general") -> str:
    """
    Suggest feature engineering ideas based on available columns and domain.
    Returns a JSON string with a list of feature ideas grouped by type.
    """
    col_list = [c.strip().lower() for c in columns.split(",")]
    domain_lower = data_domain.lower()
 
    features = []

    time_cols = [c for c in col_list if any(kw in c for kw in
                 ["date", "time", "timestamp", "created", "updated", "day", "hour"])]
    if time_cols:
        features.append({
            "category": "Time-Based Features",
            "source_columns": time_cols,
            "ideas": [
                "hour_of_day — extract hour from timestamp (captures daily patterns)",
                "day_of_week — 0=Monday … 6=Sunday (weekly seasonality)",
                "is_weekend — binary flag for Saturday/Sunday",
                "days_since_last_event — recency signal",
                "rolling_7_day_count — activity volume in last 7 days",
                "rolling_30_day_count — activity volume in last 30 days",
                "month_of_year — seasonality signal",
                "is_business_hours — binary: 9am–6pm on weekdays",
            ],
        })

    user_cols = [c for c in col_list if any(kw in c for kw in
                 ["user", "customer", "account", "client", "member"])]
    if user_cols:
        features.append({
            "category": "User / Customer Aggregation Features",
            "source_columns": user_cols,
            "ideas": [
                "total_events_per_user — overall activity count",
                "events_per_user_last_1_hour — short-term burst signal",
                "events_per_user_last_24_hours — daily activity",
                "days_since_registration — account age",
                "days_since_last_login — recency / churn risk",
                "session_count_last_30_days — engagement depth",
                "avg_session_duration — quality of engagement",
                "number_of_support_tickets — dissatisfaction proxy",
            ],
        })

    txn_cols = [c for c in col_list if any(kw in c for kw in
                ["revenue", "amount", "price", "transaction", "order", "payment", "cost"])]
    if txn_cols:
        features.append({
            "category": "Transaction / Revenue Features",
            "source_columns": txn_cols,
            "ideas": [
                "average_transaction_amount — spending level",
                "max_transaction_amount — high-value signal",
                "std_transaction_amount — spending consistency",
                "total_spend_last_30_days — recent revenue",
                "days_since_last_purchase — recency",
                "purchase_frequency — how often customer buys",
                "lifetime_revenue — total customer value",
                "refund_rate — satisfaction / fraud indicator",
            ],
        })

    event_cols = [c for c in col_list if any(kw in c for kw in
                  ["event", "action", "type", "status", "click", "view", "error", "fail"])]
    if event_cols:
        features.append({
            "category": "Behavioral / Event Features",
            "source_columns": event_cols,
            "ideas": [
                "failed_event_ratio_last_24h — error rate (churn/fraud signal)",
                "distinct_event_types_per_session — breadth of usage",
                "login_attempts_last_hour — brute-force / fraud signal",
                "page_views_before_purchase — engagement depth",
                "error_events_in_last_7_days — product quality signal",
                "click_through_rate — content relevance",
                "bounce_rate_per_session — exit intent signal",
            ],
        })

    if "ecommerce" in domain_lower:
        features.append({
            "category": "E-Commerce Specific Features",
            "source_columns": col_list,
            "ideas": [
                "cart_abandonment_rate — purchase intent signal",
                "avg_items_per_order — basket size",
                "return_rate — satisfaction and product-fit signal",
                "days_between_orders — purchase frequency",
                "preferred_device — mobile vs desktop behavior",
                "discount_sensitivity — did user buy after discount only?",
            ],
        })
    elif "fintech" in domain_lower:
        features.append({
            "category": "Fintech Specific Features",
            "source_columns": col_list,
            "ideas": [
                "failed_payment_ratio — financial stress signal",
                "transaction_velocity — transactions per hour",
                "unusual_merchant_category — anomaly signal",
                "cross_border_transaction_flag — high-risk indicator",
                "account_balance_trend — financial health",
            ],
        })

    if not features:
        features.append({
            "category": "General Features (no domain-specific columns detected)",
            "source_columns": col_list,
            "ideas": [
                "id_count — frequency of unique IDs",
                "null_flag_per_column — missingness indicator as a feature",
                "row_recency_rank — rank records by date",
                "target_encoding — encode categorical columns by target mean",
            ],
        })

    result = {
        "domain": data_domain,
        "columns_analyzed": col_list,
        "total_feature_groups": len(features),
        "feature_engineering_ideas": features,
        "general_tips": [
            "Always split train/test BEFORE engineering features to prevent data leakage.",
            "Drop ID columns before training — they have no predictive signal.",
            "Scale numeric features (StandardScaler) for distance-based models.",
            "One-hot encode or target-encode high-cardinality categoricals carefully.",
        ],
    }
    return json.dumps(result, indent=2)

@tool("detect_ml_data_risks")
def detect_ml_data_risks(file_name: str, target_column: str = "") -> str:
    """
    Identify risks in a dataset before model training.
    Checks:
      - Missing target column
      - Class imbalance (for classification)
      - Duplicate records
      - High-cardinality columns (risk of overfitting)
      - Constant columns (zero variance, useless features)
      - Data leakage candidates (columns that wouldn't exist at prediction time)
      - Outliers in numeric columns (Z-score > 3)
      - Time-based split requirement (if date columns exist)
    Returns a JSON string with a list of detected risks and recommendations.
    """
    try:
        path = _safe_resolve(file_name)
        df = pd.read_csv(path)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Could not read file: {str(e)}"}, indent=2)
 
    risks = []

    if target_column:
        if target_column not in df.columns:
            risks.append({
                "risk": "Missing Target Column",
                "severity": "critical",
                "detail": f"Column '{target_column}' not found in dataset.",
                "recommendation": f"Add the target column '{target_column}' or choose a different target.",
            })
        else:
            missing_target = df[target_column].isnull().sum()
            if missing_target > 0:
                risks.append({
                    "risk": "Missing Values in Target Column",
                    "severity": "critical",
                    "detail": f"Target '{target_column}' has {missing_target} missing values.",
                    "recommendation": "Drop rows with missing target or investigate cause.",
                })

            if df[target_column].dtype == object or df[target_column].nunique() <= 20:
                counts = df[target_column].value_counts(normalize=True)
                min_ratio = counts.min()
                if min_ratio < 0.1:
                    risks.append({
                        "risk": "Class Imbalance",
                        "severity": "high",
                        "detail": f"Minority class is only {min_ratio*100:.1f}% of data. Distribution: {counts.to_dict()}",
                        "recommendation": "Use SMOTE, class_weight='balanced', or collect more minority samples.",
                    })

    dupes = df.duplicated().sum()
    if dupes > 0:
        risks.append({
            "risk": "Duplicate Records",
            "severity": "medium",
            "detail": f"{dupes} duplicate rows detected ({dupes/len(df)*100:.1f}% of data).",
            "recommendation": "Drop duplicates with df.drop_duplicates() before splitting.",
        })

    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique()
        if n_unique > 50:
            risks.append({
                "risk": "High-Cardinality Column",
                "severity": "medium",
                "detail": f"Column '{col}' has {n_unique} unique values.",
                "recommendation": f"Use target encoding or hash encoding for '{col}'. Avoid one-hot encoding.",
            })

    for col in df.columns:
        if df[col].nunique() == 1:
            risks.append({
                "risk": "Constant Column",
                "severity": "low",
                "detail": f"Column '{col}' has only one unique value: '{df[col].iloc[0]}'.",
                "recommendation": f"Drop '{col}' — it has zero predictive value.",
            })

    leakage_hints = ["future_", "final_", "_result", "_outcome", "_after",
                     "post_", "approved", "closed", "completed"]
    for col in df.columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in leakage_hints):
            if col.lower() != target_column.lower():
                risks.append({
                    "risk": "Data Leakage Candidate",
                    "severity": "high",
                    "detail": f"Column '{col}' may contain information only available after the prediction event.",
                    "recommendation": f"Verify that '{col}' is knowable at prediction time. If not, remove it.",
                })

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            continue
        z_scores = ((df[col] - df[col].mean()) / df[col].std()).abs()
        outlier_count = (z_scores > 3).sum()
        if outlier_count > 0:
            risks.append({
                "risk": "Outliers Detected",
                "severity": "low",
                "detail": f"Column '{col}' has {int(outlier_count)} values with |Z-score| > 3.",
                "recommendation": f"Cap outliers with IQR clipping or log-transform '{col}'.",
            })

    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                 ["date", "time", "timestamp", "created", "day", "month"])]
    if date_cols:
        risks.append({
            "risk": "Time-Based Split Required",
            "severity": "medium",
            "detail": f"Date-like columns detected: {date_cols}.",
            "recommendation": (
                "Do NOT use random train/test split. "
                "Use a temporal split: train on older data, test on recent data. "
                "Random split causes data leakage in time-series models."
            ),
        })

    severity_counts = {}
    for r in risks:
        s = r["severity"]
        severity_counts[s] = severity_counts.get(s, 0) + 1

    result = {
        "file": file_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_risks_detected": len(risks),
        "severity_summary": severity_counts,
        "risks": risks,
        "overall_assessment": (
            "Critical issues found — resolve before training." if severity_counts.get("critical", 0) > 0
            else "High-risk issues found — review before training." if severity_counts.get("high", 0) > 0
            else "Low to medium risks only — safe to proceed with care."
        ),
    }
    return json.dumps(result, indent=2, default=str)

@tool("recommend_evaluation_metrics")
def recommend_evaluation_metrics(problem_type: str, business_context: str = "") -> str:
    """
    Suggest evaluation metrics appropriate for the ML problem type.
    Returns a JSON string with primary_metrics, secondary_metrics,
    avoid_metrics, and business_notes.
    """
    pt = problem_type.lower().strip()
    ctx = business_context.lower()

    metrics_library = {
        "classification": {
            "primary_metrics": ["ROC-AUC", "PR-AUC (Precision-Recall AUC)", "F1-Score"],
            "secondary_metrics": ["Precision", "Recall", "Accuracy (only if balanced classes)"],
            "avoid_metrics": ["Accuracy alone when classes are imbalanced"],
            "business_notes": (
                "Use Recall when missing a positive case is costly (e.g. fraud, disease, churn). "
                "Use Precision when false alarms are expensive (e.g. compliance flags). "
                "ROC-AUC is threshold-independent — useful for model comparison. "
                "PR-AUC is better than ROC-AUC when the positive class is rare."
            ),
        },
        "regression": {
            "primary_metrics": ["RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "R² (Coefficient of Determination)"],
            "secondary_metrics": ["MAPE (Mean Absolute Percentage Error)", "Median Absolute Error"],
            "avoid_metrics": ["R² alone — it does not reflect prediction scale"],
            "business_notes": (
                "RMSE penalizes large errors more heavily — use when big errors are costly. "
                "MAE treats all errors equally — more interpretable for business stakeholders. "
                "MAPE expresses error as a percentage — useful for business reporting but unreliable near zero."
            ),
        },
        "clustering": {
            "primary_metrics": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
            "secondary_metrics": ["Inertia (for K-Means elbow method)"],
            "avoid_metrics": ["Accuracy (no ground truth in unsupervised learning)"],
            "business_notes": (
                "Silhouette Score ranges from -1 to +1; higher is better. "
                "Run multiple K values and plot the elbow curve for K-Means. "
                "Complement quantitative metrics with business validation: do the clusters make sense?"
            ),
        },
        "forecasting": {
            "primary_metrics": ["RMSE", "MAE", "MAPE", "SMAPE (Symmetric MAPE)"],
            "secondary_metrics": ["Forecast Bias", "Coverage (prediction interval accuracy)"],
            "avoid_metrics": ["R² for time-series — it's misleading with autocorrelation"],
            "business_notes": (
                "MAPE is popular for forecasting but fails near zero values. Use SMAPE instead. "
                "Always compare against a naive baseline (e.g. 'last week's value'). "
                "Evaluate separately on peak vs. normal periods."
            ),
        },
        "anomaly_detection": {
            "primary_metrics": ["Precision@K", "Recall@K", "F1-Score (if labels available)", "AUC-ROC"],
            "secondary_metrics": ["False Positive Rate", "Mean Time to Detect (MTTD)"],
            "avoid_metrics": ["Accuracy — anomalies are rare so accuracy is misleadingly high"],
            "business_notes": (
                "If labeled anomalies exist, treat as imbalanced classification and use PR-AUC. "
                "If unsupervised, evaluate with domain expert review of top-K flagged records. "
                "Business metric: How many real anomalies were caught vs. false alerts triggered?"
            ),
        },
        "recommendation": {
            "primary_metrics": ["Precision@K", "Recall@K", "NDCG (Normalized Discounted Cumulative Gain)"],
            "secondary_metrics": ["Hit Rate", "Coverage", "Diversity Score"],
            "avoid_metrics": ["RMSE on ratings alone — it ignores ranking quality"],
            "business_notes": (
                "Offline metrics (NDCG, Precision@K) may not match online A/B test results. "
                "Always validate with an online experiment. "
                "Also track business metrics: CTR, conversion rate, revenue per recommendation."
            ),
        },
        "ranking": {
            "primary_metrics": ["NDCG@K", "MRR (Mean Reciprocal Rank)", "MAP (Mean Average Precision)"],
            "secondary_metrics": ["Precision@K", "ERR (Expected Reciprocal Rank)"],
            "avoid_metrics": ["Accuracy — ranking quality is positional, not binary"],
            "business_notes": (
                "NDCG rewards putting highly relevant items near the top. "
                "MRR focuses on the rank of the first relevant result — good for search. "
                "Always define K (e.g. NDCG@10 means top-10 results)."
            ),
        },
    }

    metrics = metrics_library.get(pt, {
        "primary_metrics": ["F1-Score", "ROC-AUC"],
        "secondary_metrics": ["Precision", "Recall"],
        "avoid_metrics": ["Accuracy alone"],
        "business_notes": "Problem type not recognized. Defaulting to classification metrics.",
    })

    context_notes = []
    if any(kw in ctx for kw in ["miss", "costly", "expensive", "critical", "fraud", "churn", "disease"]):
        context_notes.append(
            "High cost of false negatives detected → prioritize RECALL over Precision. "
            "Tune classification threshold to increase recall even at the cost of more false alarms."
        )
    if any(kw in ctx for kw in ["false alarm", "alert fatigue", "compliance", "precision"]):
        context_notes.append(
            "False alarms are costly → prioritize PRECISION. "
            "Accept lower recall to ensure flagged cases are highly likely to be true positives."
        )

    result = {
        "problem_type": problem_type,
        "primary_metrics": metrics["primary_metrics"],
        "secondary_metrics": metrics["secondary_metrics"],
        "avoid_metrics": metrics["avoid_metrics"],
        "business_notes": metrics["business_notes"],
        "context_specific_advice": context_notes if context_notes else [
            "No specific business context provided. Use primary metrics as default."
        ],
    }
    return json.dumps(result, indent=2)

@tool("create_ml_pipeline_plan")
def create_ml_pipeline_plan(
    problem_type: str,
    data_source: str = "CSV file",
    team_size: str = "small",
) -> str:
    """
    Create an end-to-end ML pipeline plan.
    Returns a JSON string with all 9 pipeline stages, tools, owners, and risks.
    """
    pipeline_stages = [
        {
            "stage": 1,
            "name": "Data Ingestion",
            "description": f"Collect raw data from {data_source} and load into a staging area.",
            "tools": ["pandas (CSV)", "SQLAlchemy (database)", "Kafka (streaming)"],
            "outputs": ["Raw DataFrame or staging table"],
            "risks": ["Missing files", "Schema drift", "Encoding issues"],
            "owner": "Data Engineer",
        },
        {
            "stage": 2,
            "name": "Data Validation",
            "description": "Check schema, data types, missing values, duplicates, and outliers.",
            "tools": ["pandera", "great_expectations", "detect_ml_data_risks tool"],
            "outputs": ["Validation report", "Data quality score"],
            "risks": ["Failing validation silently", "Drifted schema from upstream"],
            "owner": "Data Engineer / Data Scientist",
        },
        {
            "stage": 3,
            "name": "Feature Engineering",
            "description": "Create derived features, encode categoricals, and handle missing values.",
            "tools": ["pandas", "scikit-learn (Pipeline, ColumnTransformer)", "Feature-engine"],
            "outputs": ["Feature matrix X", "Target vector y"],
            "risks": ["Data leakage if done before split", "High-cardinality encoding explosion"],
            "owner": "Data Scientist",
        },
        {
            "stage": 4,
            "name": "Train-Test Split",
            "description": (
                "Temporal split if time-based data, stratified split for imbalanced classification, "
                "random split otherwise."
            ),
            "tools": ["scikit-learn train_test_split", "TimeSeriesSplit for forecasting"],
            "outputs": ["X_train, X_test, y_train, y_test"],
            "risks": ["Random split on time-series causes leakage", "Small test set from imbalanced data"],
            "owner": "Data Scientist",
        },
        {
            "stage": 5,
            "name": "Model Training",
            "description": (
                f"Train a {problem_type} model. Start with a simple baseline, "
                "then tune with cross-validation."
            ),
            "tools": ["scikit-learn", "XGBoost", "LightGBM", "statsmodels (forecasting)"],
            "outputs": ["Trained model artifact (.pkl or .joblib)"],
            "risks": ["Overfitting on small datasets", "Slow training on large data without GPU"],
            "owner": "Data Scientist",
        },
        {
            "stage": 6,
            "name": "Model Evaluation",
            "description": "Evaluate on hold-out test set using primary and secondary metrics.",
            "tools": ["scikit-learn metrics", "SHAP (explainability)", "yellowbrick (visual)"],
            "outputs": ["Evaluation report", "Confusion matrix / residual plots", "Feature importance"],
            "risks": ["Only checking one metric", "Not testing on edge cases or rare classes"],
            "owner": "Data Scientist / Business Stakeholder",
        },
        {
            "stage": 7,
            "name": "Model Registry",
            "description": "Save and version the trained model with metadata.",
            "tools": ["MLflow", "DVC", "joblib.dump()", "pickle (simple)"],
            "outputs": ["Versioned model file", "Model card (metadata YAML)"],
            "risks": ["No versioning — can't roll back", "Missing metadata for future debugging"],
            "owner": "MLOps / Data Scientist",
        },
        {
            "stage": 8,
            "name": "Batch or Real-Time Inference",
            "description": (
                "Deploy model for predictions. Batch: scheduled scoring job. "
                "Real-time: REST API with FastAPI or Flask."
            ),
            "tools": ["FastAPI", "Flask", "BentoML", "Airflow (batch)", "Kafka (stream)"],
            "outputs": ["Prediction scores", "Prediction file or API endpoint"],
            "risks": ["Schema mismatch at inference time", "Latency too high for real-time",
                      "Model not loaded efficiently"],
            "owner": "ML Engineer / Backend Engineer",
        },
        {
            "stage": 9,
            "name": "Monitoring and Retraining",
            "description": (
                "Track model performance over time. Alert on data drift or accuracy degradation. "
                "Retrain on new data when drift is detected."
            ),
            "tools": ["Evidently AI (drift detection)", "MLflow (tracking)", "Grafana / Datadog"],
            "outputs": ["Drift alerts", "Model performance dashboard", "Automated retraining trigger"],
            "risks": ["Silent model degradation", "Concept drift without data drift",
                      "Retraining without re-validation"],
            "owner": "MLOps / Data Scientist",
        },
    ]

    team_notes = {
        "solo": (
            "For a solo team: skip MLflow, use joblib/pickle, deploy with Streamlit or Flask locally. "
            "Automate retraining with a simple cron job."
        ),
        "small": (
            "For a small team: use MLflow for tracking, FastAPI for serving, "
            "GitHub Actions for CI/CD, and Evidently AI for drift monitoring."
        ),
        "large": (
            "For a large team: use Kubernetes + KServe or SageMaker for serving, "
            "Airflow for orchestration, Feast for feature store, "
            "and enterprise MLflow or Vertex AI for model registry."
        ),
    }

    result = {
        "problem_type": problem_type,
        "data_source": data_source,
        "team_size": team_size,
        "team_recommendation": team_notes.get(team_size.lower(), team_notes["small"]),
        "total_stages": len(pipeline_stages),
        "pipeline_stages": pipeline_stages,
        "estimated_timeline": "8–12 weeks for a production-grade pipeline from scratch.",
        "critical_warning": (
            "Always do train-test split BEFORE feature engineering. "
            "Feature engineering on the full dataset before splitting causes data leakage "
            "and produces falsely optimistic evaluation scores."
        ),
    }
    return json.dumps(result, indent=2)