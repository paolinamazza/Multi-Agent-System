import os
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import math
from agents import function_tool


# Load datasets from uploaded files
df_accesso = pd.read_csv("EntryAccessoAmministrati_202501.csv")
df_pendolarismo = pd.read_csv("EntryPendolarismo_202501.csv")
df_amministrati = pd.read_csv("EntryAmministratiPerFasciaDiReddito_202501.csv")
df_stipendi = pd.read_csv("EntryAccreditoStipendi_202501.csv")

df_accesso.rename(columns={
    'regione_residenza_domicilio': 'region_of_residence',
    'amministrazione_appartenenza': 'administration',
    'sesso': 'gender',
    'eta_max': 'age_max',
    'eta_min': 'age_min',
    'modalita_autenticazione': 'access_method',
    'numero_occorrenze': 'number_of_users'
}, inplace=True)

df_pendolarismo.rename(columns={
    'provincia_della_sede': 'workplace_province',
    'comune_della_sede': 'municipality',
    'stesso_comune': 'same_municipality',
    'ente': 'administration',
    'numero_amministrati': 'number_of_users',
    'distance_min_KM': 'distance_min_km',
    'distance_max_KM': 'distance_max_km'
}, inplace=True)

df_amministrati.rename(columns={
    'comparto': 'organizational_unit',
    'regione_residenza': 'region_of_residence',
    'sesso': 'gender',
    'eta_min': 'age_min',
    'eta_max': 'age_max',
    'aliquota_max': 'max_tax_rate',
    'fascia_reddito_min': 'income_bracket_min',
    'fascia_reddito_max': 'income_bracket_max',
    'numerosita': 'number_of_users'
}, inplace=True)

df_stipendi.rename(columns={
    'comune_della_sede': 'municipality',
    'amministrazione': 'administration',
    'eta_min': 'age_min',
    'eta_max': 'age_max',
    'sesso': 'gender',
    'modalita_pagamento': 'payment_method',
    'numero': 'number_of_payments'
}, inplace=True)


df_stipendi["age_max"] = pd.to_numeric(df_stipendi["age_max"].astype(str).str.strip(), errors="coerce").fillna(0).astype(int)
df_amministrati["age_max"] = pd.to_numeric(df_amministrati["age_max"].astype(str).str.strip(), errors="coerce").fillna(0).astype(int)
df_amministrati["income_bracket_min"] = (
    df_amministrati["income_bracket_min"]
    .astype(str).str.lower().str.replace("oltre i", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.strip()
    .replace("", "0") 
)
df_amministrati["income_bracket_min"] = pd.to_numeric(df_amministrati["income_bracket_min"], errors="coerce").fillna(0).astype(int)
df_amministrati["income_bracket_max"] = (
    df_amministrati["income_bracket_max"]
    .astype(str).str.lower().str.replace("fino a", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.strip()
    .replace("", "0")
)
df_amministrati["income_bracket_max"] = pd.to_numeric(df_amministrati["income_bracket_max"], errors="coerce").fillna(0).astype(int)
df_pendolarismo["distance_min_km"] = pd.to_numeric(
    df_pendolarismo["distance_min_km"].astype(str).str.strip(), errors="coerce"
).fillna(0).astype(int)

df_pendolarismo["distance_max_km"] = pd.to_numeric(
    df_pendolarismo["distance_max_km"].astype(str).str.strip(), errors="coerce"
).fillna(0).astype(int)
df_accesso["age_max"] = pd.to_numeric(df_accesso["age_max"].astype(str).str.strip(), errors="coerce").fillna(0).astype(int)

dataframes = {
    "df_stipendi": df_stipendi,
    "df_amministrati": df_amministrati,
    "df_pendolarismo": df_pendolarismo,
    "df_accesso": df_accesso
}

openai_api_key = "..."
os.environ['OPENAI_API_KEY'] = openai_api_key


# --- TOOL: Semantic Matching Engine for Datasets and Columns ---

"""
This section defines the `match_data` tool, which performs intelligent semantic matching
between a user's natural-language query and the available datasets.

Purpose:
- Identify which dataset(s) are most relevant to the query.
- Select the most appropriate **numeric column** for analysis or visualization.
- Optionally detect a **group-by** column if the query implies comparison (e.g., "by gender", "grouped by region").

How it works:
1. Embeds the user query using OpenAI's embedding model.
2. Compares it to:
   - each dataset's textual description
   - each column name and its semantic description
3. Returns a structured dictionary including:
   - matched datasets
   - best numeric column
   - optional group-by column
   - reasoning summary

Output:
This tool is used internally by all downstream tools and agents
to ensure that the analysis is grounded in the most appropriate data subset.

"""

# Load embedding model
embedding_model = OpenAIEmbedding()

# --- Column Descriptions (Auto-Inferred) ---
column_description_map = {
    "accesso": {
        "region_of_residence": "region where the employee resides",
        "administration": "type of administration",
        "gender": "employee gender",
        "age_max": "maximum age of the employee group",
        "age_min": "minimum age of the employee group",
        "access_method": "login authentication method used",
        "number_of_users": "number of employees in the group"
    },
    "pendolarismo": {
        "workplace_province": "province where the office is located",
        "municipality": "municipality of the office",
        "same_municipality": "whether residence and office are the same",
        "administration": "administration name",
        "number_of_users": "number of commuting employees",
        "distance_min_km": "minimum commuting distance in kilometers",
        "distance_max_km": "maximum commuting distance in kilometers",
    },
    "amministrati": {
        "organizational_unit": "organizational unit of the employee",
        "region_of_residence": "region where the employee resides",
        "gender": "employee gender",
        "age_min": "minimum employee age",
        "age_max": "maximum employee age",
        "max_tax_rate": "maximum tax rate applied",
        "income_bracket_min": "minimum income for the bracket",
        "income_bracket_max": "maximum income for the bracket",
        "number_of_users": "number of employees per income bracket"
    },
    "stipendi": {
        "municipality": "municipality of the office",
        "administration": "administration type",
        "age_min": "minimum employee age",
        "age_max": "maximum employee age",
        "gender": "employee gender",
        "payment_method": "salary payment method",
        "number_of_payments": "number of payments processed"
    }
}


# --- Parameters ---
DATASET_SIMILARITY_THRESHOLD = 0.4
COLUMN_SIMILARITY_THRESHOLD = 0.3

EMPLOYER_ALIASES = {
    "sector": ["sector", "administration", "organization"],
    "administration": ["administration", "sector", "organization"],
    "organization": ["organization", "sector", "administration"],
}


def preprocess_column(col_name: str, dataset_name: str) -> str:
    """Return enriched column text for semantic matching."""
    description = column_description_map.get(dataset_name, {}).get(col_name, "")
    if description:
        return f"{col_name.replace('_', ' ').lower()} - {description.lower()}"
    else:
        return col_name.replace("_", " ").lower()

@function_tool
def match_data(query: str) -> Dict:
    """
    ## ROLE
    You are a **semantic dataset matching tool**. Your job is to identify the dataset(s) and column(s) most relevant to a user's natural language query.
    This is a **mandatory first step** for all types of analysis or visualization.

    ## PERSISTENCE
    You are a tool, not a chatbot. Do not wait for confirmation. Always execute immediately and return structured results.
    Your output may be used downstream by tools that generate plots or perform statistical summaries.

    ## TOOL-CALLING RULES
    - Always run on the full query.
    - Use semantic similarity to compare query against dataset descriptions and column names.
    - Never return null or skip your turn.
    - Do not ask the user anything. Make the best decision possible based on the input.
    - ALWAYS return a list of at least one matched dataset, with at least one numeric column.

    ## PLANNING STRATEGY
    1. Embed the full user query.
    2. Compare the query embedding with each dataset description.
    3. For each matching dataset (based on semantic score):
       a. Select the most relevant numeric column.
       b. Optionally select a group-by column if grouping is inferred from the query.
    4. If no numeric columns are found in a dataset, exclude it from the result.
    5. Include the full list of columns and a clear explanation per dataset.

    ## OUTPUT FORMAT
    Return a dictionary with:
    - `matched_datasets`: List of dicts, each with:
        - `dataset_name`: name of the dataset
        - `column`: selected numeric column
        - `group_by`: optional group-by column
        - `columns`: full list of available columns
        - `description`: explanation of the match and selected columns
    - `description`: Summary of matched dataset(s), always in the format:
         Matched {N} dataset(s): {name1}, {name2}, ...

    - If no numeric column is found for any dataset, return:
        {
          "matched_datasets": [],
          "description": " No numeric columns found in any dataset. Please refine your query."
        }

    ## EXAMPLES

    Input: "Compare salaries by gender across regions"
    Output:
    {
        "matched_datasets": [
            {
                "dataset_name": "stipendi",
                "column": "salary_count",
                "group_by": "gender",
                "columns": [...],
                "description": "Matched to 'stipendi' with similarity score 0.8234.\nSelected numeric column: 'salary_count'.\nGroup by: gender"
            }
        ],
        "description": " Matched 1 dataset(s): stipendi"
    }

    Input: "What is the distribution of login methods?"
    Output:
    {
        "matched_datasets": [
            {
                "dataset_name": "accesso",
                "column": "total_employees",
                "group_by": "authentication_method",
                "columns": [...],
                "description": "Matched to 'accesso' with similarity score 0.8492.\nSelected numeric column: 'total_employees'.\nGroup by: authentication_method"
            }
        ],
        "description": " Matched 1 dataset(s): accesso"
    }
    """
    datasets = [df_accesso,df_pendolarismo, df_amministrati, df_stipendi]

    try:
        query_embedding = embedding_model.get_query_embedding(query)
        matched_datasets = []

        for df in datasets:
            if df.empty:
                continue

            dataset_embedding = embedding_model.get_text_embedding(info["description"])
            dataset_score = float(embedding_model.similarity(query_embedding, dataset_embedding))

            if dataset_score < DATASET_SIMILARITY_THRESHOLD:
                continue

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                continue

            for alias in EMPLOYER_ALIASES["sector"]:   
                if alias in df.columns and alias not in numeric_cols:
                    numeric_cols.append(alias)

            numeric_similarities = {
                col: float(
                    embedding_model.similarity(
                        query_embedding,
                        embedding_model.get_text_embedding(preprocess_column(col, name))
                    )
                )
                for col in numeric_cols
            }
            best_numeric_col, best_numeric_score = max(
                numeric_similarities.items(), key=lambda item: item[1]
            )
            if best_numeric_col not in df.columns and best_numeric_col in EMPLOYER_ALIASES:
              for alt in EMPLOYER_ALIASES[best_numeric_col]:
                  if alt in df.columns:
                      best_numeric_col = alt
                      break
            if best_numeric_col not in df.columns:
                continue

            if best_numeric_score < COLUMN_SIMILARITY_THRESHOLD:
                continue

            # Optional group-by
            best_groupby = None
            if " by " in query.lower() or "group" in query.lower():
                categorical_cols = df.select_dtypes(include="object").columns.tolist()
                if categorical_cols:
                    cat_similarities = {
                        col: float(embedding_model.similarity(query_embedding, embedding_model.get_text_embedding(preprocess_column(col, name))))
                        for col in categorical_cols
                    }
                    best_groupby, best_groupby_score = max(cat_similarities.items(), key=lambda item: item[1])
                    if best_groupby_score < COLUMN_SIMILARITY_THRESHOLD:
                        best_groupby = None

            matched_datasets.append({
                "dataset_name": name,
                "score": dataset_score,  
                "column": best_numeric_col,
                "group_by": best_groupby,
                "columns": list(df.columns),
                "description": (
                    f"Matched to '{name}' with dataset score {round(dataset_score, 4)}.\n"
                    f"Selected numeric column: '{best_numeric_col}' (score {round(best_numeric_score, 4)}).\n"
                    f"{'Group by: ' + best_groupby if best_groupby else 'No group-by column identified.'}"
                )
            })

        if not matched_datasets:
            return {
                "matched_datasets": [],
                "description": " No suitable numeric columns or datasets matched your query. Please refine your request."
            }

        # --- Prioritize best matches ---
        matched_datasets.sort(key=lambda x: x["score"], reverse=True)

        description = f" Matched {len(matched_datasets)} dataset(s): " + ", ".join([m['dataset_name'] for m in matched_datasets])

        return {
            "matched_datasets": matched_datasets,
            "description": description
        }

    except Exception as e:
        return {
            "matched_datasets": [],
            "description": f" An unexpected error occurred: {str(e)}"
        }




# --- TOOL: Detailed Single-Column Analysis Engine ---

"""
This section defines the `single_column` tool, which performs a **complete and adaptive statistical analysis**
on a single column (numeric or categorical) from one of the four real-world datasets.

Purpose:
- Analyze a single column in-depth, returning relevant descriptive statistics, outliers, distributions, and quality metrics.
- Automatically adjusts the analysis depending on whether the column is **numeric** or **categorical**.
- Used when the user asks things like:
  - “What is the mean and standard deviation of max commuting distance?”
  - “Are there outliers in income?”
  - “How many values are missing or zero?”
  - “What are the most common values in gender?”

Key Capabilities:
- For **numeric columns**:
  - Computes central tendency (mean, median, mode), spread (std dev, IQR), shape (skewness, kurtosis)
  - Detects and quantifies outliers (IQR method)
  - Performs a **normality test** (Shapiro-Wilk)
  - Returns highest/lowest values and binned frequency distributions
  - Evaluates data quality (missing values, zero counts, variance)
  - Adds contextual breakdown (e.g. by gender or region) if relevant
- For **categorical columns**:
  - Computes value counts and top categories
  - Calculates **entropy** as a measure of diversity
  - Identifies most and least frequent values
  - Reports missing values and completeness
  - Adds breakdown context (e.g. sector x gender)

Output:
- A dictionary with relevant fields depending on the column type
- Includes:
  - `summary_statistics`
  - `outliers`
  - `normality_test` (if applicable)
  - `data_quality`
  - `value_frequencies`
  - `top/lowest values`
  - `additional_context` (if meaningful)

Robustness:
- Automatically detects column type and reroutes to `analyze_categorical_data()` if not numeric.
- Handles edge cases (missing columns, non-numeric data, short series) gracefully.
"""

@function_tool
def single_column(dataset_name: str, column: str) -> Dict[str, Any]:
    """
    Tool Name: single_column
    Purpose: Perform a complete statistical summary (central tendency, dispersion, shape, outliers, quality)
               on a single numeric or categorical column from one of the four real datasets.

    # Tool-Calling Guidance:
    - Use this tool when the user asks about statistics like mean, median, outliers, missing values, or the shape of a single column.
    - Works for both numeric and categorical columns.
    - DO NOT guess: if the column is not found, return an error.
    - If the column is categorical (object or category), it will automatically delegate to `analyze_categorical_data`.

    # Planning Strategy:
    1. Validate dataset and column existence.
    2. Determine if the column is numeric or categorical.
    3. For numeric: compute descriptive stats, outliers, normality, and context.
    4. For categorical: compute frequencies, entropy, and context.
    5. Add contextual breakdown if available (region, gender, sector, etc.)

    # Output Format:
    - Return a minimal answer and in natural language
    - ONLY return what the user asked, not more and not less.
    - Errors must be minimized as much as possible. If you have an error, reparaphrase the problem and try to solve it until you find the solution.
    """
    # Mapping name → DataFrame
    dataset_map = {
        "stipendi": df_stipendi,
        "accesso": df_accesso,
        "amministrati": df_amministrati,
        "pendolarismo": df_pendolarismo
    }

    if dataset_name not in dataset_map:
        return {"error": f" Dataset '{dataset_name}' not found."}

    df = dataset_map[dataset_name]
    if column not in df.columns:
        return {"error": f" Column '{column}' not present in dataset '{dataset_name}'."}

    # Check if column is categorical or numeric
    original_series = df[column]
    if pd.api.types.is_object_dtype(original_series):
        return analyze_categorical_data(dataset_name, df, column)

    # Clean and select column for numeric analysis
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        return {"error": f" Column '{column}' does not contain valid numeric values."}

    # Statistical calculations
    summary = {
        "count": int(series.count()),
        "mean": round(series.mean(), 2),
        "median": round(series.median(), 2),
        "std_dev": round(series.std(), 2),
        "min": round(series.min(), 2),
        "max": round(series.max(), 2),
        "quantiles": {
            "Q1 (25%)": round(series.quantile(0.25), 2),
            "Q2 (50%)": round(series.quantile(0.50), 2),
            "Q3 (75%)": round(series.quantile(0.75), 2),
            "90th percentile": round(series.quantile(0.90), 2),
            "95th percentile": round(series.quantile(0.95), 2),
            "99th percentile": round(series.quantile(0.99), 2),
        },
        "skewness": round(series.skew(), 2),
        "kurtosis": round(series.kurt(), 2),
        "range": round(series.max() - series.min(), 2),
        "interquartile_range (IQR)": round(series.quantile(0.75) - series.quantile(0.25), 2),
    }

    # Outlier detection using IQR method
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_info = {
        "num_outliers": int(outliers.count()),
        "outlier_percentage": round((outliers.count() / series.count()) * 100, 2),
        "outlier_bounds": {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
        }
    }

    # Frequencies of unique values (if there are few)
    unique_values = series.value_counts().sort_index()
    if len(unique_values) <= 20:
        value_frequencies = unique_values.to_dict()
    else:
        # Bin the values into ranges
        bin_edges = np.linspace(series.min(), series.max(), 10)
        bin_labels = [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(bin_edges)-1)]
        binned_values = pd.cut(series, bins=bin_edges, labels=bin_labels)
        value_frequencies = binned_values.value_counts().sort_index().to_dict()

    # Normality test 
    # Only perform if we have enough data but not too much
    normality_test = {}
    if 3 <= len(series) <= 5000:  # Shapiro-Wilk works best on this range
        try:
            stat, p_value = stats.shapiro(series.sample(min(5000, len(series))))
            normality_test = {
                "test_name": "Shapiro-Wilk",
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "is_normal": p_value > 0.05,
                "interpretation": "Data appears to be normally distributed" if p_value > 0.05 else "Data does not appear to be normally distributed"
            }
        except Exception:
            # If test fails, skip it
            pass

    # Add coefficient of variation
    if summary["mean"] != 0:
        summary["coefficient_of_variation"] = round((summary["std_dev"] / summary["mean"]) * 100, 2)

    # Add mode
    mode_values = series.mode()
    if not mode_values.empty:
        summary["mode"] = round(float(mode_values.iloc[0]), 2)
        summary["mode_count"] = int(series.value_counts().iloc[0])

    # Add variance
    summary["variance"] = round(series.var(), 2)

    # Add basic shape description
    summary["distribution_shape"] = get_distribution_shape(summary["skewness"], summary["kurtosis"])

    # Check for potential zero or missing values issues
    data_quality = {
        "missing_values": int(df[column].isna().sum()),
        "missing_percentage": round((df[column].isna().sum() / len(df)) * 100, 2),
        "zeros": int((df[column] == 0).sum()),
        "zeros_percentage": round(((df[column] == 0).sum() / len(df)) * 100, 2)
    }

    # Add summary of top highest and lowest values
    top_values = series.nlargest(5).to_dict()
    top_values = {f"Rank {i+1}": {"index": idx, "value": round(val, 2)} for i, (idx, val) in enumerate(top_values.items())}

    bottom_values = series.nsmallest(5).to_dict()
    bottom_values = {f"Rank {i+1}": {"index": idx, "value": round(val, 2)} for i, (idx, val) in enumerate(bottom_values.items())}

    # Get additional context/dimensions if appropriate
    context = get_additional_context(df, column, dataset_name)

    # Final output
    result = {
        "dataset": dataset_name,
        "column": column,
        "data_type": "numeric",
        "summary_statistics": summary,
        "outliers": outlier_info,
        "highest_values": top_values,
        "lowest_values": bottom_values,
        "value_frequencies": value_frequencies,
        "data_quality": data_quality
    }

    # Add normality test if available
    if normality_test:
        result["normality_test"] = normality_test

    # Add context if available
    if context:
        result["additional_context"] = context

    return result

def analyze_categorical_data(dataset_name: str, df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze categorical data column.
    """
    series = df[column]

    # Value counts
    value_counts = series.value_counts()
    top_n = min(10, len(value_counts))

    # Calculate frequencies
    freq_counts = value_counts.iloc[:top_n].to_dict()
    freq_percent = (value_counts.iloc[:top_n] / value_counts.sum() * 100).round(2).to_dict()

    # Calculate entropy (diversity measure)
    probabilities = value_counts / value_counts.sum()
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    # Check for missing values
    missing_count = series.isna().sum()

    # Most frequent vs least frequent
    most_frequent = value_counts.index[0] if not value_counts.empty else None
    least_frequent = value_counts.index[-1] if not value_counts.empty else None

    # Get additional context if appropriate
    context = get_additional_context(df, column, dataset_name)

    return {
        "dataset": dataset_name,
        "column": column,
        "data_type": "categorical",
        "unique_values": len(value_counts),
        "top_values": freq_counts,
        "top_percentages": freq_percent,
        "entropy": round(entropy, 2),
        "most_frequent": {
            "value": most_frequent,
            "count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            "percentage": round(100 * value_counts.iloc[0] / len(series), 2) if not value_counts.empty else 0
        },
        "least_frequent": {
            "value": least_frequent,
            "count": int(value_counts.iloc[-1]) if not value_counts.empty else 0,
            "percentage": round(100 * value_counts.iloc[-1] / len(series), 2) if not value_counts.empty else 0
        },
        "data_quality": {
            "missing_values": int(missing_count),
            "missing_percentage": round(100 * missing_count / len(series), 2),
            "completeness": round(100 * (1 - missing_count / len(series)), 2)
        },
        "additional_context": context
    }

def get_distribution_shape(skewness: float, kurtosis: float) -> str:
    """
    Get a simple description of the distribution shape based on skewness and kurtosis.
    """
    shape = []

    # Skewness interpretation
    if skewness < -1:
        shape.append("highly negatively skewed")
    elif -1 <= skewness < -0.5:
        shape.append("moderately negatively skewed")
    elif -0.5 <= skewness < -0.1:
        shape.append("slightly negatively skewed")
    elif -0.1 <= skewness <= 0.1:
        shape.append("approximately symmetric")
    elif 0.1 < skewness <= 0.5:
        shape.append("slightly positively skewed")
    elif 0.5 < skewness <= 1:
        shape.append("moderately positively skewed")
    else:  # skewness > 1
        shape.append("highly positively skewed")

    # Kurtosis interpretation (using excess kurtosis: normal = 0)
    # Note: pandas.kurt() returns Fisher's definition (normal = 0)
    if kurtosis < -1:
        shape.append("very platykurtic (much flatter than normal)")
    elif -1 <= kurtosis < -0.5:
        shape.append("platykurtic (flatter than normal)")
    elif -0.5 <= kurtosis <= 0.5:
        shape.append("mesokurtic (similar to normal)")
    elif 0.5 < kurtosis <= 1:
        shape.append("leptokurtic (more peaked than normal)")
    else:  # kurtosis > 1
        shape.append("very leptokurtic (much more peaked than normal)")

    return ", ".join(shape)

def get_additional_context(df: pd.DataFrame, column: str, dataset_name: str) -> Dict[str, Any]:
    """
    Get additional context for the column based on the dataset.
    """
    context = {}
    if dataset_name == "accesso":
        if column == "employee_count":
            gender_breakdown = df.groupby("gender")["employee_count"].sum().to_dict()
            context["gender_breakdown"] = gender_breakdown

            if "region_of_residence" in df.columns:
                region_breakdown = df.groupby("region_of_residence")["employee_count"].sum().nlargest(5).to_dict()
                context["top_regions"] = region_breakdown

    elif dataset_name == "pendolarismo":
        if column == "distance_min_km" or column == "distance_max_km":
            if "province_of_office" in df.columns:
                province_avg = df.groupby("province_of_office")[column].mean().nlargest(5).round(2).to_dict()
                context["top_provinces_avg"] = province_avg

        elif column == "number_of_employees" or column == "employee_count":
            if "organization" in df.columns:
                org_breakdown = df.groupby("organization")[column].sum().nlargest(5).to_dict()
                context["top_organizations"] = org_breakdown

    elif dataset_name == "stipendi":
        if column == "employee_count" or column == "number":
            if "sector" in df.columns and "gender" in df.columns:
                sector_gender = df.groupby(["sector", "gender"])[column].sum().nlargest(5).to_dict()
                context["top_sector_gender_combinations"] = sector_gender

        elif column == "max_tax_rate":
            # Get average by income bracket
            if "income_bracket_max" in df.columns:
                bracket_avg = df.groupby("income_bracket_max")[column].mean().sort_index().round(2).to_dict()
                context["tax_rate_by_income_bracket"] = bracket_avg

    elif dataset_name == "stipendi":
        if column == "salary_count":
            if "payment_method" in df.columns:
                method_breakdown = df.groupby("payment_method")[column].sum().to_dict()
                context["payment_method_breakdown"] = method_breakdown

            if "age_max" in df.columns and "gender" in df.columns:
                # Create age groups
                df["age_group"] = pd.cut(df["age_max"], bins=[0, 34, 44, 54, 64, 100],
                                        labels=["18-34", "35-44", "45-54", "55-64", "65+"])
                age_gender = df.groupby(["age_group", "gender"])[column].sum().to_dict()
                context["age_gender_breakdown"] = age_gender

    return context




# --- TOOL: Aggregated Group-by Analysis Engine ---

"""
This section defines the `multi_column` tool, which performs a **grouped aggregation analysis**
on any numeric column by a categorical grouping column from one of the four real datasets.

Purpose:
- Answer user questions like:
  - “What is the average salary by region?”
  - “Which gender has the highest number of payments?”
  - “Show me the total number of employees per administration.”
- It supports various aggregations (mean, median, max, min, sum, std, count) and handles synonym resolution.

Key Capabilities:
1. Matches dataset and column names, even when synonyms or partial terms are used (e.g., "gender", "sex", "income", "stipendio").
2. Normalizes aggregation functions using natural synonyms (e.g., "avg" → "mean").
3. Handles categorical → numeric groupings with automatic cleaning and conversion.
4. Filters and cleans data before analysis to ensure robustness.
5. Produces enriched results with:
   - top group,
   - bottom group,
   - overall stats,
   - full grouped output (dictionary),
   - optional percentage distributions (when aggregation is `sum` or `count`).

Output:
- A dictionary containing:
  - aggregation info (group_by, target_column, function used)
  - top and bottom groups (with value, count, and % if applicable)
  - overall stats (records, total sum, mean)
  - full group-by result (as a dict)
- Handles errors gracefully (e.g., missing columns, unsupported aggregations, empty data).

Example use cases:
- Visual summaries like "Bar chart of salary counts by gender"
- Textual insights such as "Top 3 regions by number of entries"
- Supporting statistical comparisons between categories

Robustness:
- Synonym resolution for both `group_by` and `target_column`
- Aggregation fallback handling
- NaN and object-type cleaning with conversion to numeric
- Explicit error reporting if nothing matches or data is invalid

Integrated in:
- InsightBuilder → to generate statistical comparisons
- Visualization tools → to group and plot aggregated values
- Conversational agent → to explain distributions and category-level stats
"""

@function_tool
def multi_column(
    dataset_name: str,
    group_by: str,
    target_column: str,
    agg_function: str
) -> Dict[str, Any]:
    """
    # Persistence
    You are a persistent tool-calling agent. Continue until the user's request is fully resolved.

    # Tool-calling
    Use this tool to group a dataset by a categorical column and compute an aggregation over a numeric column. Do not guess columns—check for exact matches or known synonyms. If no match is found, return an explicit error message.

    # Planning
    1. Check dataset and column validity.
    2. Map synonyms for robustness.
    3. Clean the data.
    4. Group by the requested column and apply the aggregation.
    5. Sort and return top and bottom group, plus overall stats and full results.

    # Output Formatting
    - Return a minimal answer and in natural language
    - ONLY return what the user asked, not more and not less.
    - Errors must be minimized as much as possible. If you have an error, reparaphrase the problem and try to solve it until you find the solution.
    """
    # Dataset mapping
    dataset_map = {
        "stipendi": df_stipendi,
        "accesso": df_accesso,
        "amminsistrati": df_accesso,
        "pendolarismo": df_pendolarismo
    }

    if dataset_name not in dataset_map:
        return {"error": f" Dataset '{dataset_name}' not found."}

    df = dataset_map[dataset_name]

    # Handle common synonyms and variations in column names
    column_synonyms = {
        # Region-related
        "region": ["region", "regions", "area", "location", "geographical_area", "territory"],
        "north": ["north", "northern", "nord"],
        "south": ["south", "southern", "sud"],

        # Gender-related
        "gender": ["gender", "sex", "m/f"],
        "male": ["male", "m", "men", "man"],
        "female": ["female", "f", "women", "woman"],

        # Age-related
        "age": ["age", "age_range", "age_group", "ages", "years"],

        # Authentication-related
        "access_method": ["auth_method", "authentication", "authentication_method", "login_method", "access_method"],
        "spid": ["spid", "digital_identity"],
        "cie": ["cie", "electronic_id", "id_card"],

        # Administration-related
        "administration": ["administration", "admin", "public_administration", "institution", "organization"],

        # Salary-related
        "salary": ["salary", "wage", "income", "payment", "stipendio"],

        # Access-related
        "access": ["access", "accesses", "entry", "entries", "login"],
    }

    # Try to find the right column based on synonyms if the exact match fails
    if group_by not in df.columns:
        found = False
        for actual_col, synonyms in column_synonyms.items():
            if group_by.lower() in synonyms and actual_col in df.columns:
                group_by = actual_col
                found = True
                break

        if not found:
            # Try partial matches if no exact synonym found
            possible_columns = []
            for col in df.columns:
                if group_by.lower() in col.lower():
                    possible_columns.append(col)

            if possible_columns:
                group_by = possible_columns[0]  # Choose the first match
            else:
                return {"error": f" Grouping column '{group_by}' not found in dataset."}

    # Similar synonym handling for target column
    if target_column not in df.columns:
        found = False
        for actual_col, synonyms in column_synonyms.items():
            if target_column.lower() in synonyms and actual_col in df.columns:
                target_column = actual_col
                found = True
                break

        if not found:
            # Try partial matches if no exact synonym found
            possible_columns = []
            for col in df.columns:
                if target_column.lower() in col.lower():
                    possible_columns.append(col)

            if possible_columns:
                target_column = possible_columns[0]  # Choose the first match
            else:
                return {"error": f" Target column '{target_column}' not found in dataset."}

    # Support additional aggregation synonyms
    agg_synonyms = {
        "mean": ["mean", "average", "avg"],
        "median": ["median", "middle", "med"],
        "max": ["max", "maximum", "highest", "largest"],
        "min": ["min", "minimum", "lowest", "smallest"],
        "sum": ["sum", "total"],
        "std": ["std", "standard deviation", "deviation", "variance"],
        "count": ["count", "number", "quantity", "frequency"]
    }

    # Normalize aggregation function
    agg_function_normalized = None
    for agg, synonyms in agg_synonyms.items():
        if agg_function.lower() in synonyms:
            agg_function_normalized = agg
            break

    if not agg_function_normalized:
        return {"error": f" Aggregation '{agg_function}' not supported. Choose from: mean, median, max, min, sum, std, count."}

    agg_function = agg_function_normalized

    # Handle special cases: filter by specific region, gender, etc.
    filter_condition = None
    if group_by.lower() == "region" and any(region in ["north", "south", "center", "islands"] for region in df["region"].unique()):
        north_regions = ["north", "northern", "nord"]
        south_regions = ["south", "southern", "sud"]
        center_regions = ["center", "central", "centro"]
        islands_regions = ["islands", "isole"]

        if any(region.lower() in north_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("North", case=False, na=False)
        elif any(region.lower() in south_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("South", case=False, na=False)
        elif any(region.lower() in center_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("Center", case=False, na=False)
        elif any(region.lower() in islands_regions for region in [target_column]):
            filter_condition = df["region"].str.contains("Islands", case=False, na=False)

    # Apply filter if needed
    if filter_condition is not None:
        df = df[filter_condition]
        if df.empty:
            return {"error": " No data matching the filter criteria."}

    # Data cleaning
    df_clean = df[[group_by, target_column]].dropna()

    # Handle the case where target column might need special processing
    if df_clean[target_column].dtype == 'object':
        # Try to convert strings to numeric if possible
        df_clean[target_column] = pd.to_numeric(df_clean[target_column].str.replace('[^0-9.]', '', regex=True), errors='coerce')
    else:
        df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')

    df_clean = df_clean.dropna()

    if df_clean.empty:
        return {"error": " No valid data after cleaning."}

    agg_map = {
        "mean": df_clean.groupby(group_by)[target_column].mean(),
        "median": df_clean.groupby(group_by)[target_column].median(),
        "max": df_clean.groupby(group_by)[target_column].max(),
        "min": df_clean.groupby(group_by)[target_column].min(),
        "sum": df_clean.groupby(group_by)[target_column].sum(),
        "std": df_clean.groupby(group_by)[target_column].std(),
        "count": df_clean.groupby(group_by)[target_column].count()
    }

    result_series = agg_map[agg_function].dropna().sort_values(ascending=False)

    if result_series.empty:
        return {"error": f" No data available for {agg_function} of {target_column} grouped by {group_by}."}

    top_group = result_series.idxmax()
    bottom_group = result_series.idxmin()

    grouped_stats = df_clean.groupby(group_by)[target_column].agg(['count', 'mean', 'sum', 'min', 'max'])
    total_records = df_clean[target_column].count()
    total_sum = df_clean[target_column].sum()

    percentages = {}
    if agg_function == "count":
        for group, count in result_series.items():
            percentages[str(group)] = round((count / total_records) * 100, 2)
    elif agg_function == "sum":
        for group, sum_val in result_series.items():
            percentages[str(group)] = round((sum_val / total_sum) * 100, 2)

    return {
        "dataset": dataset_name,
        "aggregation": {
            "group_by": group_by,
            "target_column": target_column,
            "agg_function": agg_function
        },
        "top_group": {
            "group": str(top_group),
            "value": round(float(result_series[top_group]), 2),
            "percentage": percentages.get(str(top_group), None),
            "count": int(grouped_stats.loc[top_group, 'count'])
        },
        "bottom_group": {
            "group": str(bottom_group),
            "value": round(float(result_series[bottom_group]), 2),
            "percentage": percentages.get(str(bottom_group), None),
            "count": int(grouped_stats.loc[bottom_group, 'count'])
        },
        "overall_stats": {
            "total_groups": len(result_series),
            "total_records": int(total_records),
            "total_sum": round(float(total_sum), 2),
            "overall_mean": round(float(df_clean[target_column].mean()), 2)
        },
        "full_grouped_result": {str(k): round(float(v), 2) for k, v in result_series.to_dict().items()}
    }






# --- TOOL: Cross-Dataset Keyword Matching Engine ---

"""
This section defines the `multi_dataset` tool, which performs a **cross-dataset analysis**
based on keyword detection from a user query. It is used when the query mentions multiple
topics that span across different datasets (e.g., "access", "income", "commuting", "Abruzzo").

Purpose:
- Automatically detect **which datasets are relevant** to a query using keyword matching.
- Apply simple summaries (e.g., unique value counts and samples) for each relevant column.
- Filter datasets (e.g., by region) when specified in the keyword mapping.

Output:
- JSON-formatted dictionary with:
  - Original query
  - List of involved datasets
  - Per-dataset result, including:
    - Column analyzed
    - Number of unique values
    - A sample of values
  - Handles partial failure (e.g., missing columns) gracefully.

Use Cases:
- Used for broad, high-level queries where the user asks for "summary by region" or
  references more than one domain (e.g., salaries + commuting + access).
- Helpful as a **triage** tool to identify what parts of the data are worth exploring.

Key Capabilities:
- Keyword-based matching: checks for the presence of dataset-specific terms.
- Per-dataset config: controls which columns to analyze, and how to filter by region if relevant.
- Flexible fallback: if a column is not found, skips gracefully.
- Lightweight summary: avoids full statistical computation — just counts and samples.

Planning Strategy:
1. Parse the query and extract keywords.
2. Match those keywords to datasets and predefined target columns.
3. For each dataset:
   - Apply regional filters (e.g. "ABRUZZO") if defined.
   - Count unique values in target columns.
   - Return first 5 sample values (if any).
4. Aggregate results in a unified response structure.

Limitations:
- Currently uses fixed keyword mappings.
- Assumes column names are accurate or mapped manually.
- Does not perform deep statistical analysis.

Integration:
- This tool is useful when the query might trigger **multiple datasets** at once and is
  routed by the conversation agent or InsightBuilder as a “wide match” request.
"""

@function_tool
def multi_dataset(query: str) -> Dict[str, Any]:
    """
    Analyzes queries that span multiple datasets by identifying keywords
    and extracting meaningful summary insights from relevant columns.

    Persistence: Continue until all relevant datasets are analyzed.
    Tool-calling: Use keyword mapping to determine datasets and columns.
    Planning: Parse query, filter datasets, summarize columns.
    Output Formatting: JSON with query, datasets, and structured results.

    Parameters:
    - query: Natural language query involving multiple datasets

    Returns:
    - Return a minimal answer and in natural language
    - ONLY return what the user asked, not more and not less.
    - Errors must be minimized as much as possible. If you have an error, reparaphrase the problem and try to solve it until you find the solution.
    """
    # Keyword mapping with more explicit dataset-specific keywords
    keywords_for_df = {
        "accesso": {
            "keywords": ["accesso", "access", "administration", "region",],
            "columns": ["administration", "region_of_residence"],
            "region_column": "region_of_residence",
            "filter_value": "ABRUZZO"
        },
        "stipendi": {
            "keywords": ["municipality", "payment", "payment method", "income", "salary", "administration"],
            "columns": ["administration"],
            "region_column": None,
            "filter_value": None
        },
        "amministrati": {
            "keywords": ["organizational unit","unit", "income","income bracket", "sector", "number of users"],
            "columns": ["sector", "region_of_residence"],
            "region_column": "region_of_residence",
            "filter_value": "ABRUZZO"
        },
        "pendolarismo": {
            "keywords": ["pendularism", "commute", "travel", "km", "administration"],
            "columns": ["organization"],
            "region_column": None, 
            "filter_value": None
        }
    }

    # Identify datasets involved in the query
    involved_datasets = []
    for dataset, info in keywords_for_df.items():
        if any(kw.lower() in query.lower() for kw in info['keywords']):
            involved_datasets.append(dataset)

    # If no datasets found, return an error
    if not involved_datasets:
        return {
            "error": " No specific datasets could be identified from the query."
        }

    # Initialize results container
    multi_dataset_results = {
        "query": query,
        "datasets_analyzed": involved_datasets,
        "results": {}
    }

    # Analyze each dataset
    for dataset in involved_datasets:
        try:
            # Get the dataframe dynamically
            df = globals()[f"df_{dataset}"]

            # Get dataset-specific information
            dataset_info = keywords_for_df[dataset]

            # Check each column for counting or filtering
            for column in dataset_info['columns']:
                # Check if column exists in dataframe
                if column not in df.columns:
                    continue

                # If there's a region filter
                if dataset_info['region_column'] and dataset_info['filter_value']:
                    filtered_df = df[df[dataset_info['region_column']] == dataset_info['filter_value']]
                else:
                    filtered_df = df

                # Count unique values in the column
                if column in filtered_df.columns:
                    unique_values = filtered_df[column].unique()

                    # Store results
                    multi_dataset_results["results"][dataset] = {
                        "column": column,
                        "unique_values_count": len(unique_values),
                        "sample_values": list(unique_values[:5])  # First 5 unique values
                    }

                    # Break after first successful analysis
                    break

        except Exception as e:
            multi_dataset_results["results"][dataset] = {
                "error": f"Error analyzing {dataset}: {str(e)}"
            }

    return multi_dataset_results
