from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner
from agents import ModelSettings, function_tool
from pydantic import BaseModel
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import io
import base64
from pandas.api.types import CategoricalDtype
from typing import Dict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
import contextlib
import traceback
import tiktoken  

# ----- OPENAI API KEY -----
openai_api_key = "...."
os.environ['OPENAI_API_KEY'] = openai_api_key

# Verify that the key is set
print(f"OpenAI API key set: {bool(openai_api_key)}")



# ----- LOAD AND PREPARE DATA -----
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


# --- Fix df_stipendi ---
df_stipendi["age_max"] = pd.to_numeric(df_stipendi["age_max"], errors="coerce").astype("Int64")

# --- Fix df_amministrati ---
df_amministrati["age_max"] = pd.to_numeric(df_amministrati["age_max"], errors="coerce").astype("Int64")

df_amministrati["income_bracket_min"] = (
    df_amministrati["income_bracket_min"]
    .astype(str).str.lower().str.replace("oltre i", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.strip()
)
df_amministrati["income_bracket_min"] = pd.to_numeric(df_amministrati["income_bracket_min"], errors="coerce").astype("Int64")

df_amministrati["income_bracket_max"] = (
    df_amministrati["income_bracket_max"]
    .astype(str).str.lower().str.replace("fino a", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.strip()
    .replace("", "0")  # opzionale: puoi rimuoverlo se vuoi che resti NaN
)
df_amministrati["income_bracket_max"] = pd.to_numeric(df_amministrati["income_bracket_max"], errors="coerce").astype("Int64")

# --- Fix df_pendolarismo ---
df_pendolarismo["distance_min_km"] = pd.to_numeric(
    df_pendolarismo["distance_min_km"], errors="coerce"
).astype("Int64")

df_pendolarismo["distance_max_km"] = pd.to_numeric(
    df_pendolarismo["distance_max_km"], errors="coerce"
).astype("Int64")

# --- Fix df_accesso ---
df_accesso["age_max"] = pd.to_numeric(df_accesso["age_max"], errors="coerce").astype("Int64")














# ----- TOOLS -----
#InsightBuilder
@function_tool
def InsightBuilder(prompt: str) -> str:
    """
    ## ROLE:
    You are a tool that 
    - understands a natural-language query, 
		- carefully analyzes the user's request to determine exactly which columns and datasets 
	    are required, and what type of analysis must be performed
	  - generates executable Python code from a natural-language query.
    The output should be a complete, idiomatic, and modular Python script that follows a precise analysis workflow.

    ## STANDARD WORKFLOW: IT IS ALWAYS MANDATORY
    1. Understand ALL the user's requests and intent clearly, **even if there is more than one request you have to make sure to answer all of them.**
    - Identify whether the user is asking for data analysis or visualization.
    - Understand the precise relationship being queried (e.g., correlation, comparison, grouping).
    - If multiple intents are present (e.g. "group and compare") capture ALL of them.

    2. Parse the Natural-Language Prompt
    - Extract the names of ALL AND ONLY the variables that are necessary to respond to the request.
    - Identify which dataset (among df_accesso, df_pendolarismo, df_amministrati, df_stipendi) contains each variable.
    - Determine the nature of the feature (categorical or numeric) and the type of analysis required.
    
    3. Dataset Dictionary (they are available in memory)
    - **df_accesso**: Access records by region, administration, age, gender, and login method  
      Columns:
        • region_of_residence - region where the employee resides  
        • administration - type of administration  
        • gender - employee gender  
        • age_max - maximum age of the employee group  
        • age_min - minimum age of the employee group  
        • access_method - login authentication/access method used  
        • number_of_users - number of users/employees in the group  

    - **df_pendolarismo**: Commuting data by administration and office location  
      Columns:
        • workplace_province - province where the office is located  
        • municipality - municipality of the office  
        • same_municipality - whether residence and office are the same  
        • administration - administration name  
        • number_of_users - number of commuting employees/users
        • distance_min_km - minimum commuting distance in km  
        • distance_max_km - maximum commuting distance in km  

    - **df_amministrati**: Demographic and tax bracket data by region and sector  
      Columns:
        • organizational_unit - organizational unit or sector where the employee works  
        • region_of_residence - region where the employee resides  
        • gender - employee gender  
        • age_min - minimum employee age  
        • age_max - maximum employee age  
        • max_tax_rate - maximum tax rate applied  
        • income_bracket_min - minimum income for the bracket  
        • income_bracket_max - maximum income for the bracket  
        • number_of_users - number of employees per income bracket  

    - **df_stipendi**: Salary records by age, gender, administration and payment method  
      Columns:
        • municipality - municipality of the office  
        • administration - administration type  
        • age_min - minimum employee age  
        • age_max - maximum employee age  
        • gender - employee gender  
        • payment_method - salary payment method  
        • number_of_payments - number of payments processed  

    4. Solve Ambiguity and Column Mismatch
    - If a referenced column doesn't exist in the dataset, match it to the most semantically similar column using string similarity (e.g., "administration" vs. "organization").
    - Warn if multiple columns could match the same label; select the best one based on context or overlap in values.

    5. Code Planning and Structure
    - The Python code you generate must follow a modular, reproducible, and fully executable pipeline. 
    - This structure is mandatory and must be followed in every case, adapting to the feature/target types and datasets involved.
		- Required libraries: pandas, numpy, matplotlib.pyplot, seaborn.
    
    - Structure:
        a. Load and inspect the DataFrames.
		        - Use the four datasets (df_accesso, df_pendolarismo, df_amministrati, df_stipendi) already loaded in memory.
		        - Print the column names of the releant DataFrames to confirm variable presence.
		        - Validate that selected columns exist OR fallback to best-matching ones.
		        - Show a .head() preview for debugging.
        b. Identify the variables.
		        - Clearly mark the feature variable(s) (explanatory, independent, or grouping variable) and the target(s) (outcome, dependent, measured variable), if necessary.
		        - Use .dtypes and value previews to classify each variable.
        c. IF more than one plot is needed to answer to the user request, use ALWAYS subplots. THIS IS MANDATORY.
            - Create the figure with `fig, axs = plt.subplots(...)` and plot all charts into `axs[i]`
           - DO NOT create multiple separate `plt.figure()` instances.
           - Ensure the figure is returned as a single object.

        d. Aggregate the grouping column, if there is one and if it is necessary.
        e. Aggregate the target column (e.g. weighted average by employee count), if necessary.
        g. Compute requested statistics (e.g. Pearson correlation or t-test), if necessary.
        h. Visualize with plots, if necessary.

    6. IF the analysis involves merging two datasets:
	    1. Load both datasets and print their columns.
	    2. Detect join key candidates (matching or semantically similar), and select the one with most shared non-null values. 
        - IMPORTANT: Ignore NaN values, null values, or values that are not present in both datasets.
        - What matters is the correct execution of the code!
	    3. Generate the code.
        JOINS:
        - df_accesso and df_amministrati: join on "region_of_residence"
        - df_accesso and df_stipendi: join on "administration"
        - df_accesso and df_pendolarismo: join on "administration" 
        - df_pendolarismo and df_stipendi: join on "municipality" if the user asks, for example, "by administration" or "for each administration"
            OR on "municipality" if the user asks, for example, "by municipality" or "for each administration"

    
        EXAMPLE: Determines if there is a correlation between the method of portal access and the average commuting distance for each administration.
        1. Calculate the average commuting distance from df_pendolarismo, grouped by administration.
        2. Compute the dominant portal access method per administration from df_accesso.
        3. Merge the two datasets on administration.
        4. Convert the access_method to categorical codes to perform a correlation.
        5. Compute the correlation (e.g. Pearson).


        EXAMPLE OF CODE: Plot the average commuting distance per age group
        ```python
        import matplotlib.pyplot as plt
        df_pendolarismo["avg_km"] = (df_pendolarismo["distance_min_km"] + df_pendolarismo["distance_max_km"]) / 2
        merged = df_pendolarismo.merge(df_stipendi, on=["municipality"], how="inner")
        merged["age_group"] = merged["age_min"].astype(str) + "-" + merged["age_max"].astype(str)
        merged.groupby("age_group")["avg_km"].mean().plot(kind="bar", title="Avg Commute by Age Group", figsize=(10,5))
        plt.ylabel("Average Distance (km)")
        plt.tight_layout()
        plt.show()
        ```
    
    7. Code Quality
    - All column accesses should be guarded with `if col in df.columns`.
    - Use assertions or print statements to confirm key steps.
    - Comments should clearly mark each section of the process.
    - Code must run without external file dependencies (datasets are preloaded).

    8. Validation
    - Validate intermediate steps: print head of merged table, number of groups, etc.

    9. Reiteration on Failure
    - If any required column or computation fails, retry a second time.
    - If you get still an error, raise a clear error and recommend possible column alternatives or clarify user intent.

    10. Final Output
    - Deliver the complete Python code as a string, with no markdown formatting or explanation, that will be executed by CodeRunner(code)..
    - Append a short human-readable summary (e.g. “Correlation between X and Y is 0.12, suggesting a weak linear relationship.")”
    """



#CodeRunner
@function_tool
def CodeRunner(code: str) -> dict:
    """
    Executes Python code and returns:
      - `output`: everything printed during execution
    """
    buf_out = io.StringIO()
    try:
        plt.close('all')  # reset plot state
        ns = {
            '__builtins__': __builtins__,
            'pd': pd,
            'plt': plt,
            'os': os,
            'np': np,
            'df_accesso': df_accesso,
            'df_pendolarismo': df_pendolarismo,
            'df_amministrati': df_amministrati,
            'df_stipendi': df_stipendi,
        }

        with contextlib.redirect_stdout(buf_out):
            exec(code, ns)

        return {
            'output': buf_out.getvalue().strip(),
        }

    except Exception as e:
        import traceback
        return {
            'output': f"Error: {e}\n{traceback.format_exc()}",
        }



#ResultExplainer
@function_tool
def ResultExplainer(code: str, output: str, prompt: str) -> str:
    """
    Uses GPT-4.1 to turn the code output into a fluent, natural-language explanation.
    All parameters must be provided explicitly.
    Automatically includes the raw output table inside a markdown code block if applicable.
    """
    client = OpenAI()

    # Wrap the output in a code block if it looks like a table (i.e., contains multiple lines and columns)
    def format_output_for_display(raw_output):
        lines = raw_output.strip().split("\n")
        if len(lines) >= 2 and any("|" in line or "\t" in line or line.count(" ") > 3 for line in lines):
            return f"""\n{raw_output.strip()}\n"""
        return raw_output.strip()
    formatted_output = format_output_for_display(output)
    explanation_prompt = f"""
You are a helpful data analyst. A user ran this Python code and received the following result.
Your task is to explain this result in fluent, natural English.

--- USER PROMPT ---
{prompt}

--- PYTHON CODE ---
{code}

--- RAW OUTPUT ---
{formatted_output}

INSTRUCTIONS:
- If the output contains a table, PRINT THE TABLE IN A CLEAN FORMAT using aligned text or markdown (e.g. using triple backticks).
- If it's a single number or set of numbers, clearly describe what they represent.
- ALWAYS show the actual printed output.
- NEVER NEVER NEVER say the chart “couldn't be rendered”
- ONLY COMMENT THE RESULTS, NOT THE CODE. Don't say what worked and what didn't during the execution, just talk about the output!
- Do NOT invent or assume missing visuals — only explain printed outputs.
- Then provide a kind, informative, and concise explanation of what it shows.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are a data analyst who explains outputs in natural, user-friendly English. Always show raw output tables if relevant, returning the WHOLE table."
            },
            {
                "role": "user",
                "content": explanation_prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ----- MEMORY -----
encoding = tiktoken.get_encoding("cl100k_base")  

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

async def agent_with_memory(user_query: str, agent, memory_buffer: list, max_token_length=3000):
    """
    Token-aware memory: truncate the memory buffer based on the total number of tokens.
    """
    memory_buffer.append(user_query)

    while True:
        full_memory_text = "\n".join(memory_buffer)
        if count_tokens(full_memory_text) <= max_token_length:
            break
        memory_buffer.pop(0)

    past_context = "\n".join(memory_buffer[:-1]) 
    full_query = (
        f"Old context (summary of previous questions):\n{past_context}\n\n"
        f"New question:\n{user_query}"
    ) if past_context else user_query

    result = await Runner.run(agent, full_query,max_turns=20)

    if isinstance(result.final_output, str):
        return {
            "output": result.final_output,
            "memory_buffer": memory_buffer
        }

    if isinstance(result.final_output, dict):
        result["memory_buffer"] = memory_buffer
        return result

    return {
        "output": f" Unexpected result type: {type(result.final_output)}",
        "memory_buffer": memory_buffer
    }




# ----- AGENTS -----
# ----- VISUALIZATION AGENT -----
visualization_agent = Agent(
    name="VisualizationAgent",
    instructions=(
        """
        ## ROLE
        You are a specialized agent for generating visualizations based on user requests.
        Your task is to generate a visualization (matplotlib chart) based on the user's request.

        ## STANDARD WORKFLOW: IT IS ALWAYS MANDATORY
        1. Use `InsightBuilder(prompt=query)` to 
           - deeply understand the user request, 
           - identify the relevant datasets and the relevant columns, 
           - generate the FULL Python code for plotting
        2. Execute the code using `CodeRunner(code)` to return the chart.
        3. If useful or requested use ResultExplainer(code=..., output=..., prompt=query) to provide a natural-language explanation of the chart.
        3. If the chart is empty or invalid, explain that clearly and politely.
        
        ## RULES
        - MAKE SURE YOU ALWAYS ANSWER TO ALL REQUESTS, NOT JUST ONE.
        - NEVER skip code execution.
        - ALWAYS return a visual output.
        - If not only a simple plot request, but also insight is requested, **ensure you return both the chart and a meaningful explanation of it.**
        - Always choose the visualization type, group-by, or filter that provides the clearest insight given the user request.
        - If multiple chart types are valid (e.g., grouped bar vs. line), select the one that best shows category differences or trends.
        - If multiple datasets are needed, make sure to merge or combine them correctly.
        - If the user has not specified a filter (e.g. age group), default to “include all” unless doing so would clutter the visual excessively.
        - You are the expert — act autonomously. NEVER ask the user to disambiguate.

        ## UNCERTAINTY HANDLING
        - If the user query is ambiguous, incomplete, or could support multiple visual styles or groupings:
            → DO NOT ask the user to clarify.
            → DO NOT delay execution or return fallback text.
            → INSTEAD, use your best judgment based on:
                - common-sense defaults,
                - available columns,
                - the structure of the dataset,
                - and the most informative visual for the type of data.

        ## OUTPUT FORMAT
        - A properly rendered matplotlib chart is MANDATORY.
        - Insight (textual explanation) can be included if helpful or requested, but it is not strictly mandatory.
        """
        ),
        model="gpt-4.1",
        tools=[
            InsightBuilder, CodeRunner, ResultExplainer]
)

# ----- ANALYSIS AGENT -----
data_processing_agent = Agent(
    name="DataProcessingAgent",
    instructions=(
        """
        ## ROLE
        You are a specialized agent for performing data analysis tasks (NOT visualization).
        Your task is to generate, execute, and explain the result of a Python-based analysis, based on the user's request
        
        ## STANDARD WORKFLOW: IT IS ALWAYS MANDATORY
        1. Use `InsightBuilder(prompt=query)` to 
           - deeply understand the user request, 
           - identify the relevant datasets and the relevant columns, 
           - generate the FULL Python code for the analysis
        2. Execute the code using `CodeRunner(code)` to get the result and provide a natural-language explanation.
        3. Use ResultExplainer(code=..., output=..., prompt=query) to provide a natural-language explanation of the output.

        
        ## RULES
        - MAKE SURE YOU ALWAYS ANSWER TO ALL REQUESTS, NOT JUST ONE.

        - NEVER skip code execution. DO NOT skip steps.
        - ALWAYS return a textual output.
        - DO NOT output the Python code.
        - DO NOT describe the code or how it works.
        - DO NOT comment on assumptions, uncertainty, or the pipeline itself.
        - IMPORTANT: if the result is a table, you MUST include the table(s) WITH the explanation.

        ## UNCERTAINTY HANDLING
        - If the user query is ambiguous, incomplete, or could support multiple visual styles or groupings:
            → DO NOT ask the user to clarify.
            → DO NOT delay execution or return fallback text.
            → INSTEAD, use your best judgment based on:
                - common-sense defaults,
                - available columns,
                - the structure of the dataset,
                - and the most informative visual for the type of data.


        ## OUTPUT
        - Your final output must be a clean, natural-language explanation of the result, returned by `ResultExplainer`.
        - It must be fluent, concise, and easy to understand — suitable for a non-technical user.
        - If the result is a table, return also the table with the explanation, describing what the table shows and any important trends or values.
        - If the result is a number, explain what it represents.
        - If the result is empty, explain that clearly and politely.
        """
    ),
    model="gpt-4.1",
    tools=[InsightBuilder, CodeRunner, ResultExplainer]
)

# ----- CONVERSATION AGENT (ORCHESTRATOR) -----
conversation_agent = Agent(
    name="ConversationAgent",
    instructions=(
        """ 
        ## ROLE
        You are a specialized agent for understanding user requests and routing them to the appropriate sub-agent (DataProcessingAgent or VisualizationAgent or both).
        
        ## MEMORY-AWARE INTELLIGENCE
        - You ALWAYS have access to the list of previous user queries (memory buffer).
        - If the current query is ambiguous, short, or refers to something mentioned earlier (e.g. "and now group it", "do the same for gender", "plot that"), you MUST use the memory buffer to reconstruct the user's intent.
        - You must understand the semantic relationship between the current and previous questions.
        - Treat the memory buffer as contextual history: leverage it to disambiguate vague references, maintain continuity, and follow up accurately.
        - DO NOT ask the user to rephrase. YOU must infer what they mean.

        ## INSTRUCTIONS:
        - Analyze user prompts to determine intent.
        - Your job is to detect whether the prompt requires:
            A. Insights → route to DataProcessingAgent
            B. Visuals → route to VisualizationAgent
            C. BOTH → You MUST:
                1. Call VisualizationAgent with the full prompt to generate the chart
                2. Wait for its result and print it first
                3. Then, call DataProcessingAgent with the SAME prompt
                4. RETURN BOTH RESULTS
        ROUTING CRITERIA:
        - DataProcessor: Prompts requesting statistics, comparisons, or textual summaries.
        - Visualizer: Prompts requesting plots, charts, graphs, or any form of data visualization.
        - BOTH: Prompts that require both a chart and a deep analysis.
        EXAMPLES OF SIGNAL WORDS
        - Visual-only → plot, bar chart, draw, graph, visualize, show, histogram, scatter
        - Insight-only → explain, difference, analyze, distribution, interpret, summarize, correlation, compare
        - BOTH → any mix of the above

        ## UNCERTAINTY HANDLING
        - If the user query is ambiguous, incomplete, or could support multiple visual styles or groupings:
            → DO NOT ask the user to clarify.
            → DO NOT delay execution or return fallback text.
            → INSTEAD, use your best judgment based on:
                - common-sense defaults,
                - available columns,
                - the structure of the dataset,
                - and the most informative visual for the type of data.

        ## OUTPUT
        - Forward only the final result from sub-agents.
        - Forward the combined output (explanation + chart) if both were needed. Otherwise, just return the result of the single relevant agent.
        - Ensure outputs are concise, relevant, and devoid of extraneous information.
        """
    ),
    model="gpt-4.1",
    handoffs=[data_processing_agent, visualization_agent]
)

