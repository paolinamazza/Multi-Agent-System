import pandas as pd
import io
import sys
import asyncio
import traceback
import builtins
import numpy as np
import os
import json
import hashlib
import re
import time
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage # Import BaseMessage to format history


# Configure the LLM
llm_model = "mistral-nemo"
DATA_DIR = "./datasets"  # Update this to your data directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Cache Management using File Metadata (Modification Time & Size) ---
def get_cache_filepath(filename: str) -> str:
    """Generates the file path for the cache file based on the original filename."""
    # Hash the filename to create a unique cache
    filename_hash = hashlib.md5(filename.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{filename_hash}_metadata.json")


def get_file_state_key(filepath: str) -> str | None:
    """Generates a key representing the file's current state (mtime, size)."""
    try:
        stat_info = os.stat(filepath)
        return f"{stat_info.st_mtime}-{stat_info.st_size}"
    except FileNotFoundError:
        print(f"Warning: File not found for state key generation: {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Error getting file state for {filepath}: {e}")
        return None


def load_cached_metadata(
    filename: str, current_file_state_key: str | None
) -> dict | None:
    """Loads metadata and optional head sample from cache if it exists and matches the current file state key."""
    cache_file = get_cache_filepath(filename)
    if not current_file_state_key:  # Cannot validate cache if current state is unknown
        return None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache_content = json.load(f)

            # Check if the cached state matches the current file state
            if cache_content.get("file_state_key") == current_file_state_key:
                print(f"Cache hit: Using cached data for {filename}")
                # Return the full cached content including metadata and head sample JSON
                return cache_content

            print(f"Cache invalid: File state changed for {filename}")
        except json.JSONDecodeError:
            print(f"Warning: Cache file for {filename} is corrupted. Re-analyzing.")
        except Exception as e:
            print(f"Warning: Error loading cache for {filename}: {e}. Re-analyzing.")
    else:
        print(f"Cache miss: No cache file found for {filename}")
    return None  # Cache miss or invalid


def save_cached_metadata(
    filename: str, metadata: dict, file_state_key: str | None, head_sample_json: str | None = None
):
    """Saves metadata, file state key, and optional head sample (as JSON string) to the local cache."""
    cache_file = get_cache_filepath(filename)
    if not file_state_key:
        print(
            f"Warning: Cannot save cache for {filename} without a valid file state key."
        )
        return
    cache_content = {
        "file_state_key": file_state_key,
        "metadata": metadata,
        "cached_at": time.time(),  # Optional: timestamp for cache creation
    }
    if head_sample_json is not None:
         cache_content["head_sample"] = head_sample_json # Store head as JSON string

    try:
        with open(cache_file, "w") as f:
            json.dump(cache_content, f, indent=2)
        print(f"Metadata and head sample cached for {filename}")
    except Exception as e:
        print(f"Error saving cache for {filename}: {e}")


# --- Data Analysis and Loading ---
def analyze_data(df: pd.DataFrame) -> dict:
    """Analyzes a DataFrame and returns a concise metadata dictionary, including robust missing value detection and data type classification."""
    column_metadata = {}
    for col in df.columns:
        dtype = str(df[col].dtype)

        # Start with standard NaN check
        missing_mask = df[col].isna()
        missing_percent = missing_mask.mean() * 100

        is_numeric_string = False
        non_numeric_examples = []
        # Calculate unique count before handling missing strings for object type
        unique_count = df[col].nunique()

        # Determine high-level data type
        high_level_type = "string" # Default
        if pd.api.types.is_numeric_dtype(df[col]):
            high_level_type = "numerical"
        elif dtype == "object":
            # Additional check for empty strings or strings with only whitespace in object columns
            empty_string_mask = df[col].astype(str).str.strip() == ''
            combined_missing_mask = missing_mask | empty_string_mask
            missing_percent = combined_missing_mask.mean() * 100

            # Re-calculate unique count excluding empty/whitespace strings for better representation
            unique_count = df.loc[~combined_missing_mask, col].nunique()

            # Use regex that handles potential surrounding whitespace
            non_missing_series = df.loc[~combined_missing_mask, col].astype(str).str.strip()
            if not non_missing_series.empty:
                 potential_numeric = non_missing_series.str.fullmatch(r"^-?\d+(\.\d+)?$")
                 numeric_match_rate = potential_numeric.mean()

                 if numeric_match_rate > 0.9:
                     non_numeric_examples = (
                     non_missing_series.loc[~potential_numeric]
                         .dropna()
                         .unique()
                         .tolist()
                     )
                     if not non_numeric_examples or len(non_numeric_examples) < 5:
                         is_numeric_string = True

            # Refine high-level type for object columns
            if is_numeric_string:
                high_level_type = "numerical"
            # Heuristic for categorical: low unique count relative to total rows OR low absolute unique count
            elif unique_count / len(df) < 0.1 and unique_count < 50:
                 high_level_type = "categorical"
            else:
                 high_level_type = "string" # Default remains for other object types


        column_metadata[col] = {
            "type": dtype,
            "high_level_type": high_level_type, # Added the new classification
            "missing_percent": float(missing_percent),  # Ensure JSON serializable
            "unique_count": int(unique_count),  # Ensure JSON serializable
            "is_numeric_string": is_numeric_string,
            "non_numeric_examples": non_numeric_examples[:5],  # Limit examples
        }
    return column_metadata


# Modified to return file_paths and only load DataFrame on cache miss/re-analysis
def load_data_and_extract_metadata(data_dir: str) -> tuple[dict, dict, dict]:
    """
    Loads dataframes (only if cache is missed) and extracts/caches metadata.
    Uses file state (mtime, size). Also caches a small head sample.
    Returns:
        raw_dfs_on_miss: dict of {filename: DataFrame} (only for cache misses)
        metadata_cache: dict of {filename: {"metadata": metadata_dict, "head_sample": head_json_string}}
        file_paths: dict of {filename: filepath} for all processed files
    """
    raw_dfs_on_miss = {} # Will only store DataFrames for cache misses/re-analysis
    metadata_cache = {}
    file_paths = {} # Store file paths for all files, regardless of cache hit/miss

    for filename in os.listdir(data_dir):
        if filename.endswith((".csv", ".xlsx", ".parquet")):
            file_path = os.path.join(data_dir, filename)
            file_paths[filename] = file_path # Store file path

            current_file_state_key = get_file_state_key(file_path)
            cached_content = load_cached_metadata(filename, current_file_state_key) # Load full cached content

            if cached_content is not None:
                # Cache hit: Use cached metadata and head sample, DO NOT load DataFrame yet
                print(f"Cache hit: Using cached data for {filename}")
                # metadata_cache now stores the structure {"metadata": ..., "head_sample": ...}
                metadata_cache[filename] = cached_content
                # DataFrame is NOT loaded and NOT added to raw_dfs_on_miss here
            else:
                # Cache miss or invalid: Load DataFrame, analyze, cache
                print(f"Cache miss or invalid: Analyzing and caching metadata for {filename}")
                try:
                    # Load DataFrame for analysis
                    if filename.endswith(".csv"):
                        df = pd.read_csv(file_path)
                    elif filename.endswith(".xlsx"):
                        df = pd.read_excel(file_path)
                    elif filename.endswith(".parquet"):
                        df = pd.read_parquet(file_path)
                    else:
                        print(f"Skipping unsupported file type: {filename}")
                        continue # Skip this file

                    # Analyze data
                    metadata = analyze_data(df)
                    head_sample_json = df.head(5).to_json(orient='split') # Get and format head sample

                    # metadata_cache stores the structure {"metadata": ..., "head_sample": ...}
                    metadata_cache[filename] = {"metadata": metadata, "head_sample": head_sample_json}
                    raw_dfs_on_miss[filename] = df # Store the DataFrame for this cache miss ONLY

                    # Save to cache with current file state key, INCLUDING head
                    save_cached_metadata(filename, metadata, current_file_state_key, head_sample_json)

                except Exception as e:
                    print(f"Error loading or processing {filename}: {e}")

    return raw_dfs_on_miss, metadata_cache, file_paths # Return file_paths


# --- TOOL 1: Query Normalizer ---
class NormalizeQueryTool:
    def __init__(self, llm):
        self.llm = llm

    # Modified forward method to accept relevant_metadata (includes high_level_type)
    def forward(self, user_input: str, relevant_metadata: dict = None) -> str:
        user_input = user_input.replace("datset", "dataset")  # Simple typo correction

        # Prepare metadata context for the prompt, NOW INCLUDING high_level_type
        metadata_context = "No metadata available."
        if relevant_metadata:
            metadata_lines = []
            for df_sanitized_filename, meta_content in relevant_metadata.items():
                 # Extract the actual metadata dict from the cached content structure
                 metadata = meta_content.get("metadata", {})
                 metadata_lines.append(f"Dataset: {df_sanitized_filename}")
                 for col, col_meta in metadata.items():
                     # Include high_level_type in the prompt
                     metadata_lines.append(f"  - Column '{col}': Type={col_meta.get('type')}, HighLevelType={col_meta.get('high_level_type', 'unknown')}, Missing={col_meta.get('missing_percent', 0):.2f}%, Unique={col_meta.get('unique_count', 0)}")
                      # Add other relevant metadata details as needed
            metadata_context = "\n".join(metadata_lines)


        prompt = f"""
        You are a data analyst AI.
        Convert the following natural language question into a clear, concise **technical instruction** that describes a specific operation on a pandas DataFrame.
        You have access to metadata about the relevant dataset(s). Use this metadata, especially the column names and types, to ensure the instruction is accurate and uses the correct column names from the dataset.
        Correct any potential typos in column names based on the provided metadata.
        Requirements:
        - Do NOT repeat the question or include any commentary.
        - Do NOT wrap the instruction in quotes or say things like "The instruction is:".
        - Use simple imperative language (e.g. "Calculate", "Filter", "Show", etc.).
        - The instruction should be direct and executable, e.g., "Filter rows...", "Calculate the mean...".
        - Focus on the core operation, not data loading or setup.
        - **CRITICAL: Use the exact column names found in the provided metadata.** If the user's query contains a column name that is a likely typo of a column name in the metadata, use the correct column name from the metadata in the instruction.
        Relevant Datasets Metadata:
        {metadata_context}

        User question:
        {user_input}

        Structured instruction (using correct column names from metadata):
        """
        response = self.llm.invoke(prompt).strip()
        # Basic cleanup of common LLM artifacts
        response = re.sub(r'^["\']?(.*?)["\']?$', r"\1", response)  # Remove potential surrounding quotes
        response = response.replace("Instruction:", "").strip()
        return response

# --- TOOL 2: Data Operation Code Generator ---
class DataOperationCodeGenerator:
    def __init__(self, llm):
        self.llm = llm # Kept for potential future use

    # Modified to accept relevant_metadata (includes high_level_type)
    def forward(self, instruction: str, relevant_metadata_content: dict = None, data_insights: str = "") -> str:
        # relevant_metadata_content keys are sanitized filenames, values are {"metadata": ..., "head_sample": ...}
        metadata_context = "No metadata available."
        numeric_string_columns = []
        if relevant_metadata_content:
             # Extract only the metadata dicts for JSON dump
             metadata_only = {k: v.get("metadata", {}) for k, v in relevant_metadata_content.items()}
             metadata_context = json.dumps(metadata_only, indent=2)

             # Iterate through metadata for each dataframe (keys here are sanitized names)
             for df_sanitized_filename, meta_content in relevant_metadata_content.items():
                 metadata = meta_content.get("metadata", {})
                 for col, col_meta in metadata.items():
                     if col_meta.get('is_numeric_string', False):
                         # Include sanitized filename for clarity in prompt
                         numeric_string_columns.append(f"{df_sanitized_filename}.{col}")


        # List the available DataFrame variables for the LLM (based on sanitized names)
        available_df_vars = list(relevant_metadata_content.keys()) if relevant_metadata_content else []
        available_df_vars_str = ", ".join(available_df_vars) if available_df_vars else "No dataframes available."


        prompt = f"""
        Generate Python code to perform the following operation.
        The relevant pandas DataFrames will be provided as variables in the execution environment.
        Each variable is named after the sanitized filename of the dataset (e.g., `df_sales_data_csv`).
        Instruction: {instruction}

        Datasets Metadata (relevant parts, includes high_level_type):
        {metadata_context}

        Data Insights and Quality Issues (Pay close attention to these!):
        {data_insights if data_insights else "None provided."}

        Potential Numeric String Columns (MUST BE CONVERTED FIRST if needed for numeric operations):
        {", ".join(numeric_string_columns) if numeric_string_columns else "None identified"}

        Available DataFrame Variables: {available_df_vars_str}

        Code Requirements:
        - Generate ONLY executable Python code for the task.
        ABSOLUTELY NO other text.
        - Assume pandas is imported as pd and numpy as np.
        - Access dataframes directly using their variable names (e.g., `df_yourfilename_csv`). DO NOT assume a 'dfs' dictionary exists.
        - Include necessary type conversions, especially for numeric operations on columns identified as 'is_numeric_string'.
        Be explicit about which dataframe/column you are converting (e.g., `df_sales['col'] = pd.to_numeric(df_sales['col'], errors='coerce')`).
        - Handle potential missing values (NaN) appropriately based on the instruction (e.g., using .dropna(), .fillna(), or letting pandas handle them if acceptable).
        - If the operation involves multiple dataframes (e.g., merging, comparing), write code that uses the variable names of the relevant dataframes.
        - DO NOT include any import statements (import pandas as pd, import numpy as np).
        - DO NOT include markdown formatting like ```python ... ``` or ```.
        - DO NOT add ANY explanatory text, comments, or introductions before or after the code block.
        - Prioritize correctness and robustness. If the instruction is ambiguous regarding missing values or types, make a reasonable assumption (e.g., dropna for calculations).
        Python Code (Only the code, nothing else):
        **CRITICAL INSTRUCTION: The final result of the operation MUST be assigned to a variable named `result`.**
        **For operations that display data (like showing head, tail, describe, info), assign the resulting DataFrame or Series to the 'result' variable.**
        - **For other operations, store the final calculated or transformed result in a variable named 'result'.**
        - **The last line of the code must be of the form: result = ... (unless the code naturally ends with an assignment to 'result').**
        - Do NOT use the 'return' statement.
        The code should be valid in global (non-function) scope.
        - **Ensure the code explicitly assigns the final output to the 'result' variable.**
        - **Example:** If the instruction is "Calculate the mean of column 'Amount'", the code should be `result = df_sales['Amount'].mean()`.
        If the instruction is "Show the head of the dataframe", the code should be `result = df_sales.head()`.
        If the instruction is "Get the most frequent value in column 'Category'", the code should be `result = df_data['Category'].value_counts().index[0]`.
        Python Code (Only the code, nothing else):
        """

        raw_output = self.llm.invoke(prompt).strip()
        print(f"Debug: Raw LLM Output:\n---\n{raw_output}\n---")

        cleaned_code = self._cleanup_code(raw_output)

        if not cleaned_code:
            print("Error: Code generation failed or produced empty output.")
            return "# Error: Code generation failed."

        print(f"Debug: Cleaned Code:\n---\n{cleaned_code}\n---")
        return cleaned_code

    def _cleanup_code(self, code: str) -> str:
        """
        Cleans the generated code to extract the executable Python.
        This method is more flexible and aims to handle various LLM output formats.
        """

        # 1. Remove markdown-style code blocks
        code = re.sub(r"```(?:python)?\s*\n(.*?)\n```", r"\1", code, flags=re.DOTALL | re.IGNORECASE).strip()

        # 2. Remove any leading or trailing non-code text
        lines = code.split('\n')
        code_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines or lines that are clearly not code (keep some comments like # Store result)
            if not line or line.lower().startswith(('explanation:', 'reasoning:', 'thought:')):
                continue
            code_lines.append(line)

        code = "\n".join(code_lines).strip()

        # 3. Remove common intro/outro phrases
        code = re.sub(r"^(here is the .*?:|i hope this helps!|```|python code:)", "", code, flags=re.IGNORECASE).strip()
        code = re.sub(r"(```)$", "", code).strip()

        return code

# --- TOOL 2b: Visualization Code Generator ---
class VisualizationCodeGenerator:
    def __init__(self, llm):
        self.llm = llm

    # Modified to accept relevant_metadata (includes high_level_type) AND chat_history_text AND precomputed_result_obj
    def forward(self, instruction: str, relevant_metadata_content: dict = None, data_insights: str = "", chat_history_text: str = "", precomputed_result_obj: any = None) -> str:
        # relevant_metadata_content keys are sanitized filenames, values are {"metadata": ..., "head_sample": ...}
        metadata_context = "No metadata available."
        if relevant_metadata_content:
             # Extract only the metadata dicts for JSON dump
             metadata_only = {k: v.get("metadata", {}) for k, v in relevant_metadata_content.items()}
             metadata_context = json.dumps(metadata_only, indent=2)

        # List the available DataFrame variables for the LLM (based on sanitized names)
        available_df_vars = list(relevant_metadata_content.keys()) if relevant_metadata_content else []
        available_df_vars_str = ", ".join(available_df_vars) if available_df_vars else "No dataframes available."

        # Add precomputed result context
        precomputed_result_context = "No precomputed result available."
        if precomputed_result_obj is not None:
            # Convert structured result (like Series or DataFrame) to a string representation
            try:
                if isinstance(precomputed_result_obj, (pd.Series, pd.DataFrame)):
                    # Use to_string() for Series and to_markdown() for DataFrames for better formatting
                    precomputed_result_context = f"Precomputed Result (type: {type(precomputed_result_obj).__name__}):\n{precomputed_result_obj.to_string() if isinstance(precomputed_result_obj, pd.Series) else precomputed_result_obj.to_markdown()}"
                else:
                    # Fallback for other types
                    precomputed_result_context = f"Precomputed Result (type: {type(precomputed_result_obj).__name__}): {str(precomputed_result_obj)}"
            except Exception as e:
                precomputed_result_context = f"Error formatting precomputed result: {e}"


        prompt = """
        Generate Python code using matplotlib, seaborn, or plotly to visualize data according to the instruction.
        The relevant pandas DataFrames will be provided as variables in the execution environment, named after their sanitized filenames (e.g., `df_sales_data_csv`).
        Instruction: "{instruction}"

        Available DataFrame Variables: {available_df_vars_str}
        Datasets Metadata (relevant parts):
        {metadata_context}

        Data Insights / Quality Issues (Consider these when generating code, e.g., for handling missing data):
        {data_insights}

        **Chat History (for context, includes previous tool outputs):**
        {chat_history_text}

        **Precomputed Result from Previous Step (Use this for plotting if the instruction refers to a calculated value):**
        {precomputed_result_context}

        Code Requirements:
        - **CRITICAL:** If the instruction asks to visualize a *computation result* (like averages, sums, counts, unique values, etc.) that was likely produced by a previous DataProcessingAgent step:
            - **CAREFULLY READ the 'Precomputed Result' section above.**
            - **Identify the specific values and their corresponding labels/keys** within that 'Precomputed Result'.
            - **DO NOT recalculate these values from the original DataFrame.**
            - **Instead, create a small pandas Series or DataFrame** *at the beginning of your visualization code* containing these *specific, pre-computed values parsed from the 'Precomputed Result' context*.
            - The structure should reflect the data you extracted (e.g., if it's a Series output like "Label1: Value1, Label2: Value2", create `data_to_plot = pd.Series({{'Label1': Value1, 'Label2': Value2}})`).
            - Generate the plotting code (e.g., a bar chart using matplotlib or seaborn) to visualize *this `data_to_plot` Series or small DataFrame*, not the original columns from the large dataset DataFrame.
            - For a bar chart comparing different series values, `data_to_plot.plot(kind='bar')` using matplotlib is a simple approach.
        - If the instruction is to visualize data *directly* from the DataFrame (e.g., "plot the distribution of column X", "create a scatter plot of Y vs Z"), then access and use the original DataFrame columns as needed.
        - Use one of: matplotlib.pyplot (as plt), seaborn (as sns), or plotly.express (as px).
        - Include necessary imports ONLY for the plotting libraries used (e.g., `import matplotlib.pyplot as plt`).
        - Assume pandas (pd) and numpy (np) are already imported.
        - Handle potential missing values appropriately *only if* you are accessing data directly from the original DataFrame.
        - Set clear titles and axis labels for the plot.
        - Convert data types if necessary for plotting (e.g., using `pd.to_numeric` for numeric axes, handling potential errors).
        - Generate ONLY executable Python code. NO explanations, comments (unless essential), or markdown formatting like ```python.
        - **IMPORTANT:** Unlike data processing, visualization code usually doesn't need to assign the final plot object to a 'result' variable.
        - Focus on generating the code that displays or saves the plot (e.g., using `plt.show()` or `fig.show()`).
        - The execution environment will capture standard output, but not the plot image directly.
        Python Code (Only the code, nothing else):
        """.format(
            instruction=instruction,
            available_df_vars_str=available_df_vars_str,
            metadata_context=metadata_context,
            data_insights=data_insights if data_insights else "None provided.",
            chat_history_text=chat_history_text,
            precomputed_result_context=precomputed_result_context
        )

        raw_output = self.llm.invoke(prompt).strip()
        print(f"Debug: Raw LLM Output:\n---\n{raw_output}\n---")

        cleaned_code = self._cleanup_code(raw_output)

        if not cleaned_code:
            print("Error: Code generation failed or produced empty output.")
            return "# Error: Code generation failed."

        print(f"Debug: Cleaned Code:\n---\n{cleaned_code}\n---")
        return cleaned_code

    def _cleanup_code(self, code: str) -> str:
        """
        Cleans the generated code to extract the executable Python.
        This method is more flexible and aims to handle various LLM output formats.
        """

        # 1. Remove markdown-style code blocks
        code = re.sub(r"```(?:python)?\s*\n(.*?)\n```", r"\1", code, flags=re.DOTALL | re.IGNORECASE).strip()

        # 2. Remove any leading or trailing non-code text
        lines = code.split('\n')
        code_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines or lines that are clearly not code (keep some comments like # Store result)
            if not line or line.lower().startswith(('explanation:', 'reasoning:', 'thought:')):
                continue
            code_lines.append(line)

        code = "\n".join(code_lines).strip()

        # 3. Remove common intro/outro phrases
        code = re.sub(r"^(here is the .*?:|i hope this helps!|```|python code:)", "", code, flags=re.IGNORECASE).strip()
        code = re.sub(r"(```)$", "", code).strip()

        return code


# --- TOOL 3: Code Executor ---
class ExecuteCodeTool:
    # Modified __init__ to accept file_paths
    def __init__(self, llm, file_paths: dict):
        self.llm = llm # Kept for potential future use
        self.file_paths = file_paths # Store file paths
        self.local_vars = {
            "pd": pd,
            "np": np,
        }  # Base globals; dataframes added dynamically

    # Modified forward method to accept sanitized_to_original_map
    # Removed dfs parameter as DFs are loaded inside this method
    def forward(self, code: str, sanitized_to_original_map: dict) -> tuple[str, any]:
        """
        Executes the given Python code within a safe environment.
        Returns both the printed output and a structured result object (if available).
        """

        loaded_dfs = {}
        result_to_pass = None
        output = ""

        try:
            # Load necessary DataFrames
            for sanitized_name, original_filename in sanitized_to_original_map.items():
                if original_filename in self.file_paths:
                    file_path = self.file_paths[original_filename]
                    try:
                        print(f"Debug: Loading DataFrame for execution: {original_filename}")
                        if original_filename.endswith(".csv"):
                            df = pd.read_csv(file_path)
                        elif original_filename.endswith(".xlsx"):
                            df = pd.read_excel(file_path)
                        elif original_filename.endswith(".parquet"):
                            df = pd.read_parquet(file_path)
                        else:
                            print(f"Warning: Unsupported file type: {original_filename}")
                            continue
                        loaded_dfs[sanitized_name] = df
                    except Exception as e:
                        return f"Error loading dataset {original_filename}: {e}", None
                else:
                    return f"Error: File path not found for dataset {original_filename}.", None

            # Setup local variables
            self.local_vars.update({
                "pd": pd,
                "np": np,
                **loaded_dfs
            })

            print(f"Debug: ExecuteCodeTool local_vars keys before exec: {list(self.local_vars.keys())}")

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                # Safety: Block dangerous file operations
                if re.search(r"open\(|os\.|shutil\.", code):
                    raise Exception("Error: File I/O operations are forbidden.")

                # Execute code
                exec(code, self.local_vars)

                # Capture output
                output = sys.stdout.getvalue() or sys.stderr.getvalue()

                # Extract 'result' if exists
                result_to_pass = self.local_vars.get("result", None)

            except TimeoutError:
                output = "Error: Code execution timed out."
            except Exception as e:
                output = f"Error: Code execution failed.\n{e}\n{traceback.format_exc()}"
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                # Clean up only loaded DFs
                for var in loaded_dfs.keys():
                    if var in self.local_vars:
                        del self.local_vars[var]

        except Exception as e:
            return f"Error during execution setup: {e}", None

        return output, result_to_pass



# --- TOOL 5: Data Insights Agent ---
class DataInsightsAgent:
    # Modified __init__ to accept metadata_cache and file_paths
    def __init__(self, llm, cache_dir: str, metadata_cache: dict, file_paths: dict):
        self.llm = llm
        self.cache_dir = cache_dir
        self.insight_cache = {}  # In-memory cache for quick access
        self.metadata_cache = metadata_cache # Store the metadata cache (now contains metadata + head_sample)
        self.file_paths = file_paths # Store file paths

    # Modified _generate_insights to accept optional df_head (instead of full df)
    def _generate_insights(self, df_head: pd.DataFrame | None, query: str, metadata: dict) -> str:
        """Generates data insights using the LLM, focusing on columns relevant to the query and using metadata."""
        metadata_context = json.dumps(metadata, indent=2) if metadata else "No metadata available."
        df_head_context = ""
        if df_head is not None:
             df_head_context = f"""
        Here is the head of the DataFrame for context (do NOT refer to specific values from the head):
        {df_head.to_markdown(index=False, numalign="left", stralign="left")}
             """

        prompt = f"""
        You are an AI assistant that provides a **single, very brief sentence summary** of data quality issues relevant to a user's query, based *only* on provided metadata.
        The user's query is: "{query}"
        Here is the pre-computed metadata for the relevant dataset:
        {metadata_context}
        {df_head_context} # Include head context if available

        Based *strictly* on the user query and the provided metadata, summarize the **most critical data quality findings** (like significant missing data percentages, numeric strings needing conversion, high_level_type, or other notable issues from the metadata) that are directly relevant to the query.
        Your output should be **only** the summary sentence(s), following this format:
        "The metadata highlights potential issues with [issue 1, e.g., X% missing data], [issue 2, e.g., numeric strings needing conversion], etc. in the [relevant column(s)]. These findings should be considered when [user's requested operation]."
        If there are no notable issues relevant to the query based on metadata, state "No significant data quality issues found relevant to the query."
        Concise Summary:
        """
        insights = self.llm.invoke(prompt).strip()
        return insights

    def forward(self, query: str, filename: str) -> str:
        """Retrieves or generates data insights, using caching."""
        # filename here is the ORIGINAL filename

        # Create a cache key that includes the query to make insights query-specific
        cache_key = f"{filename}:{hashlib.md5(query.encode()).hexdigest()}"

        if cache_key in self.insight_cache:
            print(f"Insights Cache Hit (Original filename + Query Hash): {filename}")
            return self.insight_cache[cache_key]

        # --- Cache Miss or Invalid for Insight ---
        print(f"Generating insights for {filename}")

        try:
            # Retrieve cached content (metadata + head sample) for the current file
            # metadata_cache stores {"metadata": ..., "head_sample": ...}
            cached_content = self.metadata_cache.get(filename, {})
            metadata = cached_content.get("metadata", {})
            cached_head_json = cached_content.get("head_sample")
            df_head_for_insights = None

            if cached_head_json:
                try:
                    # Ensure df_head_for_insights is a DataFrame
                    df_head_for_insights = pd.read_json(io.StringIO(cached_head_json), orient='split')
                    print(f"Debug: Using cached head for insights for {filename}")
                except Exception as e:
                    print(f"Warning: Failed to load cached head for {filename}: {e}")
                    df_head_for_insights = None # Reset if loading fails

            # If cached head is not available, load only the head from the file path
            if df_head_for_insights is None:
                file_path = self.file_paths.get(filename) # Use stored file_paths
                if file_path:
                    try:
                        print(f"Debug: Loading head for insights for {filename}")
                        # Use nrows for efficiency
                        if filename.endswith(".csv"):
                            df_head_for_insights = pd.read_csv(file_path, nrows=5)
                        elif filename.endswith(".xlsx"):
                            # Reading only head from xlsx can be tricky, may need to load more or cache
                            # For simplicity, load full and take head, or rely on caching
                            print(f"Warning: Loading full file {filename} to get head for insights (XLSX/Parquet). Consider caching head.")
                            if filename.endswith(".xlsx"):
                                df_full = pd.read_excel(file_path)
                                df_head_for_insights = df_full.head(5)
                            elif filename.endswith(".parquet"):
                                df_full = pd.read_parquet(file_path)
                                df_head_for_insights = df_full.head(5)
                        else:
                            print(f"Warning: Unsupported file type for loading head: {filename}")


                    except Exception as e:
                        print(f"Error loading head for insights for {filename}: {e}")
                        # Continue without head if loading fails

            # Pass df_head_for_insights (could be None), query, and metadata to the insight generation helper
            insights = self._generate_insights(df_head_for_insights, query, metadata) # Pass df_head_for_insights

            # Cache the insights using the query-specific cache key
            self.insight_cache[cache_key] = insights
            print(f"Generated and Cached Insights (Original filename + Query Hash): {filename}")
            return insights

        except Exception as e:
            print(f"Error generating insights for {filename}: {e}")
            return f"Error: Could not generate data insights for {filename}. {e}"


class DataProcessingAgent:
    # Modified __init__ to only accept metadata_cache and tools
    def __init__(self, llm, metadata_cache, tools, smart_analyst): # Removed dfs parameter
        self.llm = llm
        # No longer store raw_dfs_all here, rely on ExecuteCodeTool to load
        self.metadata_cache = metadata_cache # metadata_cache now stores {"metadata": ..., "head_sample": ...}
        self.code_executor = tools["execute_code"] # This tool now loads DFs
        self.code_generator = DataOperationCodeGenerator(llm)
        self.tools = tools # Access to get_insights and normalize_query
        self.smart_analyst = smart_analyst # Store reference to SmartDataAnalyst

    def forward(self, input: str, context: dict = {}) -> dict:
        """
        Executes data processing logic.
        Relies on metadata for planning and ExecuteCodeTool for DataFrame access.
        """
        print(f"Debug: DataProcessingAgent received query: {input}")

        # Find relevant original filenames using SmartDataAnalyst's method (now LLM-based, uses file_paths and metadata)
        relevant_original_filenames = self.smart_analyst.find_matching_dfs(input)
        print(f"Debug: Relevant Original Filenames for Data Processing: {relevant_original_filenames}")

        if not relevant_original_filenames:
            return "Error: Could not find a matching dataset for your query."

        # --- Get metadata and create the sanitized name map (no DataFrame loading here) ---
        sanitized_to_original_map = {}
        combined_metadata_content = {} # Use the metadata_cache content structure

        for original_filename in relevant_original_filenames:
             sanitized_name = self.smart_analyst.get_sanitized_df_name(original_filename)
             sanitized_to_original_map[sanitized_name] = original_filename

             # Get the metadata_cache content (metadata + head sample) for the original filename
             metadata_content = self.metadata_cache.get(original_filename, {})
             combined_metadata_content[sanitized_name] = metadata_content


        if not sanitized_to_original_map:
            return "Error: Could not retrieve relevant datasets for processing."


        # --- Pass metadata (content structure) to NormalizeQueryTool ---
        instruction = self.tools["normalize_query"].forward(input, combined_metadata_content)
        print(f"Debug: Normalized Instruction (after using metadata): {instruction}")

        # Get insights for the first targeted dataframe (simplification)
        # Use the original filename from the map
        first_sanitized_filename = list(sanitized_to_original_map.keys())[0]
        first_original_filename = sanitized_to_original_map.get(first_sanitized_filename)

        data_insights = "None provided."
        if first_original_filename:
            # Call the DataInsightsAgent's forward method with the query and original filename
            # The insights agent now uses the metadata which includes 'high_level_type' and a head sample
            data_insights = self.tools["get_insights"].forward(
                input, first_original_filename
            )
        else:
             print(f"Warning: Could not get insights for sanitized filename '{first_sanitized_filename}' as original filename was not found.")

        print(f"Debug: Data Insights: {data_insights}")


        # Generate code, passing the instruction, combined_metadata_content, and insights
        code = self.code_generator.forward(instruction, combined_metadata_content, data_insights)
        print(f"Debug: Generated Code:\n{code}")


        # Execute code, passing the sanitized_to_original_map. ExecuteCodeTool loads DFs.
        execution_result, result_obj = self.code_executor.forward(code, sanitized_to_original_map)
        print(f"Debug: Execution Result: {execution_result}")
        # Save the result into SmartDataAnalyst
        if result_obj is not None:
            self.smart_analyst.latest_structured_result = result_obj

        return {
            "output_text": execution_result,
            "structured_result": result_obj
        }


from langchain_core.messages import BaseMessage # Ensure this import is at the top of your file

class VisualizationAgent:
    # Initialize with metadata_cache, tools, and smart_analyst reference
    def __init__(self, llm, metadata_cache, tools, smart_analyst):
        self.llm = llm
        self.metadata_cache = metadata_cache # Stores {"metadata": ..., "head_sample": ...}
        # Ensure the specific code generator is correctly instantiated if needed elsewhere,
        # otherwise using a potentially shared LLM is fine.
        self.code_generator = VisualizationCodeGenerator(llm)
        self.code_executor = tools["execute_code"] # Tool to execute code and load DFs
        self.tools = tools # Access to other tools like get_insights, normalize_query
        self.smart_analyst = smart_analyst # Reference to the main analyst class

    def _format_chat_history(self, chat_memory) -> str:
        """Formats LangChain memory messages into a simple text string."""
        messages = chat_memory.chat_memory.messages
        if not messages:
            return "No history available."
        history_lines = []
        for msg in messages:
            # Ensure msg is a BaseMessage instance before accessing type/content
            if isinstance(msg, BaseMessage):
                history_lines.append(f"{msg.type.upper()}: {msg.content}")
            else:
                # Handle unexpected message format if necessary
                history_lines.append(str(msg)) # Fallback to string representation
        return "\n".join(history_lines)

    # Modified generate_visualization to access and pass chat history and precomputed result
    def forward(self, input: str, context: dict = {}) -> dict:
        print(f"Debug: VisualizationAgent received query: {input}")

        filenames = self.smart_analyst.find_matching_dfs(input)
        if not filenames:
            return {"output_text": "Error: No matching dataset found.", "structured_result": None}

        sanitized_to_original_map = {self.smart_analyst.get_sanitized_df_name(f): f for f in filenames}
        combined_metadata_content = {sanitized: self.metadata_cache.get(orig, {}) for sanitized, orig in sanitized_to_original_map.items()}

        # Get insights for the first targeted dataframe (if available)
        data_insights = "None provided."
        if filenames:
             data_insights = self.tools["get_insights"].forward(input, filenames[0])
        else:
             print("Warning: No relevant filename found to fetch insights for visualization.")


        # Get precomputed result from context or fallback to latest structured result
        precomputed = context.get("intermediate_results", {}).get("last_structured_result")
        if precomputed is None:
            precomputed = self.smart_analyst.latest_structured_result  # fallback

        # Get and format chat history from the main analyst's memory
        chat_history_text = self._format_chat_history(self.smart_analyst.memory)
        print(f"Debug: Passing Chat History to Viz Code Gen:\n---\n{chat_history_text}\n---") # Debug history
        print(f"Debug: Passing Precomputed Result to Viz Code Gen: {precomputed}") # Debug precomputed result


        # Call code generator, passing both chat_history_text and precomputed_result_obj
        vis_code = self.code_generator.forward(
            instruction=input,
            relevant_metadata_content=combined_metadata_content,
            data_insights=data_insights,
            chat_history_text=chat_history_text, # Pass formatted chat history
            precomputed_result_obj=precomputed # Pass the precomputed result object
        )

        print(f"Debug: Generated Visualization Code:\n{vis_code}")

        # Execute the code
        output, result_obj = self.code_executor.forward(vis_code, sanitized_to_original_map)
        print(f"Debug: Visualization Execution Output:\n{output}")

        # Storing the result_obj might not be relevant for visualizations,
        # but we keep it for consistency if needed later.
        if result_obj is not None: # Check if result_obj is not None before assigning
            self.smart_analyst.latest_structured_result = result_obj  #might be irrelevant 

        # Check if the execution output contains plot information or an error
        final_output_text = output
        if "Error" in output:
             print(f"Error during visualization execution: {output}")
        elif not output.strip() and "plt.show()" not in vis_code and "fig.show()" not in vis_code:
             # If execution produced no stdout and didn't explicitly show plot, inform user
             final_output_text = "Visualization code executed. If it saved a file, it should be available. If it was meant to display, the environment might not support direct display."
        elif not output.strip():
             final_output_text = "Visualization code executed, plot should be displayed if environment supports it."


        return {
            "output_text": final_output_text,
            "structured_result": result_obj # Pass along result_obj if it exists
        }

class SmartDataAnalyst:
    """Main orchestrator for local, memory-powered, multi-step LLM-based data analysis."""

    def __init__(self, llm, data_dir, cache_dir):
        self.llm = llm
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.latest_structured_result = None

        # Load metadata (with caching) and capture file paths.
        # raw_dfs_on_miss will only contain DFs if cache was missed during _load_data.
        # metadata_cache will contain metadata + head_sample for all files.
        # file_paths will contain paths for all files.
        self.raw_dfs_on_miss, self.metadata_cache, self.file_paths = self._load_data()

        # Check if any file paths were found.
        if not self.file_paths:
             raise RuntimeError("No datasets found or processed. Please add files to the datasets directory.")


        # Initialize base tools
        self.tools = {
            # Pass metadata_cache and file_paths to DataInsightsAgent
            "get_insights": DataInsightsAgent(llm, cache_dir, self.metadata_cache, self.file_paths),
            # Pass file_paths to ExecuteCodeTool
            "execute_code": ExecuteCodeTool(llm, self.file_paths), # Pass file_paths here
            # Initialize NormalizeQueryTool here, it will receive metadata in its forward call
            "normalize_query": NormalizeQueryTool(llm),
        }

        # Agents - Pass the metadata cache and tools dictionary
        self.data_processing_agent = DataProcessingAgent(
            llm,
            self.metadata_cache, self.tools, self # Removed self.raw_dfs
        )
        self.visualization_agent = VisualizationAgent(
             llm, self.metadata_cache, self.tools, self # Removed self.raw_dfs
        )

        # Create dynamic tool lookup (this will map tool names to their forward methods)
        self.tool_lookup = {
            "DataProcessingAgent": self.data_processing_agent.forward,
            "VisualizationAgent": self.visualization_agent.forward}
        # Wrap them into LangChain Tools (chainable + callable)
        self.langchain_tools = [
            Tool(
                name="DataProcessingAgent",
                func=self.data_processing_agent.forward,
                description="Use ONLY for calculations, filtering, transformations. Input MUST be a concise English description of the required calculation (e.g., 'calculate the mean of column X'). DO NOT provide code." # Refined description
            ),
            Tool(
                name="VisualizationAgent",
                func=self.visualization_agent.forward,
                description="Use ONLY to create plots/charts of existing data or results PREVIOUSLY calculated by DataProcessingAgent. Input MUST be a concise English description of the plot (e.g., 'plot X vs Y'). Cannot perform calculations." # Refined description
            )
        ]


        # Enable memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # LangChain Agent Executor

        system_prompt = """
        You are a Smart Data Analysis Agent. Your goal is to answer user queries about datasets accurately, without guessing insights or analysis. Your aim is provide factually accurate results and let the user decide on their own.

        Available Tools:
        - DataProcessingAgent: Use ONLY for data calculations, filtering, aggregation, or transformations. Input MUST be a concise English description of the calculation needed (e.g., 'calculate the mean of column X'). DO NOT provide code.
        - VisualizationAgent: Use ONLY for generating plots or charts. It plots existing data or results ALREADY calculated by DataProcessingAgent. Input MUST be a concise English description of the plot needed (e.g., 'plot X vs Y', 'create a bar chart of the averages'). DO NOT provide code.
        
        DO NOT:
        - Guess or hallucinate insights about the data
        - Interpret the result or chart
        - Provide statistical analysis, assumptions, or summaries
        - Include markdown, links, or images
        
        **CRITICAL WORKFLOW for Calculation + Plotting:**
        1. **Identify Need:** If the query asks for BOTH calculation (e.g., average, sum, count) AND plotting:
        2. **MUST Use DataProcessingAgent FIRST:** Your first action MUST be to call DataProcessingAgent. The Action Input must describe ONLY the calculation needed (e.g., "Calculate the average of minimum distance and maximum distance").
        3. **Wait for Result:** You will receive the calculated result from the DataProcessingAgent.
        4. **MUST Use VisualizationAgent SECOND:** Your second action MUST be to call VisualizationAgent. The Action Input must describe how to plot the result you just received (e.g., "Plot the calculated averages in a bar chart").
        5. **DO NOT skip the calculation step.** VisualizationAgent CANNOT perform calculations.

        **General Instructions:**
        - Decide if a tool is needed. If yes, strictly follow the workflow above.
        - **Format:** Always use the Thought/Action/Action Input format.
        - **Action Input:** MUST be a concise English description. **NO PYTHON CODE ALLOWED in Action Input.**
        
        Thought: [Your reasoning about the query and which tool/step is next]
        Action: [DataProcessingAgent or VisualizationAgent]
        Action Input: [Concise English description of the task for THAT specific tool (NO CODE!)]


        """

        self.agent_executor = initialize_agent(
        tools=self.langchain_tools,
        llm=self.llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=self.memory,
        verbose=True,
        agent_kwargs={
            "system_message": system_prompt,
        },
        handle_parsing_errors=True
    )



        print("\n✅ SmartDataAnalyst initialized with local LLM, memory, and tool orchestration.")


    def _load_data(self):
        """Loads metadata (with caching) and DataFrame only if cache is missed. Returns file paths."""
        print("📥 Loading datasets and extracting metadata...")
        # This will return DFs ONLY for cache misses/re-analysis, plus metadata_cache and file_paths
        dfs_on_miss, meta_cache_content, paths = load_data_and_extract_metadata(self.data_dir)
        print(f"✅ Processed {len(paths)} dataset(s). Loaded {len(dfs_on_miss)} DFs for initial analysis.")
        # Note: meta_cache_content contains {"metadata": ..., "head_sample": ...}
        return dfs_on_miss, meta_cache_content, paths # Return the file_paths dictionary

    def get_sanitized_df_name(self, original_filename: str) -> str:
        """Generates a sanitized name for an original filename."""
        # Sanitize filename to be a valid Python variable name
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_]', '_', original_filename)
        # Prepend 'df_' to avoid potential conflicts with Python keywords or builtins
        if not sanitized_filename.startswith('df_'):
             sanitized_filename = 'df_' + sanitized_filename
        return sanitized_filename


    def find_matching_dfs(self, query: str) -> list[str]:
        """
        Uses the LLM to identify the relevant DataFrames based on the user query.
        Returns a list of original filenames for the identified datasets.
        Operates using metadata and file paths, not full DataFrames.
        """
        # find_matching_dfs should now work based on available file_paths and metadata
        if not self.file_paths: # Check file_paths instead of raw_dfs
            print("Warning: No data file paths found to search.")
            return []

        # Pass metadata to LLM for finding relevant datasets
        available_datasets_info = "\n".join([
            # Use metadata for context, not necessarily the full DataFrame
            f"- {filename} (Columns: {', '.join(self.metadata_cache.get(filename, {}).get('metadata', {}).keys())})"
            for filename in self.file_paths.keys() if filename in self.metadata_cache and self.metadata_cache.get(filename, {}).get('metadata') is not None # Ensure metadata exists
        ])


        prompt = f"""
        Based on the following user query, identify which of the available datasets are relevant.
        Consider the intent of the query and the columns available in each dataset based on the provided metadata.
        User Query: {query}

        Available Datasets (based on metadata):
        {available_datasets_info}

        Relevant Datasets:
        List the **original filenames ONLY**, one filename per line.
        **ABSOLUTELY CRITICAL: EACH RELEVANT FILENAME MUST APPEAR EXACTLY ONE TIME IN YOUR LIST.**
        **DO NOT list the same filename more than once, even if it is highly relevant.**
        **DO NOT include any other text, explanations, bullet points, numbering, or formatting whatsoever.**
        If multiple DIFFERENT datasets are relevant, list each unique filename on a new line.
        REMOVE ALL DUPLICATES IN THE LIST
        If none are relevant, just output the single word "None".
        Relevant Filenames:
        """
        llm_response = self.llm.invoke(prompt).strip()
        print(f"Debug: LLM response for dataset selection:\n---\n{llm_response}\n---")

        # Process the LLM response - expecting only filenames or "None"
        relevant_original_filenames = [line.strip() for line in llm_response.split('\n') if line.strip() and line.strip().lower() != 'none']

        # Filter to ensure only actual found filenames are returned (from file_paths)
        relevant_original_filenames = [f for f in relevant_original_filenames if f in self.file_paths]

        if not relevant_original_filenames:
             print("Debug: No relevant datasets selected by LLM or found in file paths.")

        return relevant_original_filenames

    def _format_intermediate_results(self, results: dict) -> str:
        if not results:
            return "None"
        return "\n".join([f"{k}: {v[0]}" for k, v in results.items()])

    def _describe_result(self, result_obj: any) -> str:
        if isinstance(result_obj, pd.Series):
            return f"Series with {len(result_obj)} values"
        elif isinstance(result_obj, pd.DataFrame):
            return f"DataFrame with {result_obj.shape[0]} rows and {result_obj.shape[1]} columns"
        return f"Object of type {type(result_obj).__name__}"

    def _pick_best_result_for_plotting(self, results: dict) -> any:
        """Simple heuristic: prefer latest Series or small DataFrame."""
        for _, (_, obj) in reversed(results.items()):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj
        return None

    def run(self, user_query: str):
        if not self.file_paths:
            return "⚠️ No data loaded. Please add datasets."

        try:
            # Normal invoke call
            response = self.agent_executor.invoke({
                "input": user_query,
                "chat_history": self.memory.chat_memory.messages
            })

            return response["output"]

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error during agent execution: {e}"






# --- Main Execution ---
async def main():
    try:
        # Initialize the LLM and Agent
        # Ensure OllamaLLM is imported correctly
        llm = OllamaLLM(model=llm_model, temperature=0.0)  # Adjust temperature as needed
        agent = SmartDataAnalyst(llm, DATA_DIR, CACHE_DIR)

        print("\n--- Data Analysis Agent Initialized Successfully! ---")

        # Interactive loop
        print("\nHello. how can I help you today?\n\nType 'q' to exit the chat.")

        while True:
            try:
                user_query = input("\n>>> You: ")
                if user_query.lower() in ["q", "quit", "exit"]:
                    print("Exiting.")
                    break
                if not user_query.strip():
                    continue

                print("\nProcessing your query...")
                # Run the agent logic
                response = agent.run(user_query)

                print("\n<<< Agent:")
                # LangChain agent.run returns a dictionary, the output is in the 'output' key
                if isinstance(response, dict) and 'output' in response:
                     print(response['output'])
                else:
                     print(response) # Fallback just in case


            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())  # Print the full traceback

    except Exception as e:
        print(f"Failed to initialize the agent: {e}")
        print(traceback.format_exc())  # Print the full traceback for init errors too


if __name__ == "__main__":
    asyncio.run(main())