# Introduction

This project examines the design and creation of an intelligent, modular multi-agent system to natural language-based data analysis and visualization. The system was built to provide an accessible, flexible, and auditable interface that allows users, especially non-technical ones, to query complex public sector data using easy conversation-based queries.

The overarching issue addressed in this project is how to bridge the gap between unstructured user input and structured data analysis, without invoking rigid pre-programmed logic. To do this, we designed a multi-agent system powered by OpenAI's newly announced Agent SDK, which allows function-aware agents to interpret instructions, call tools, and coordinate tasks in a modular and composable way.

Throughout the project, we used a two-path development methodology: one on local open-source frameworks to give offline independence and confidentiality and the other on cloud-native frameworks to have maximum reasoning capacity and deployment ease. We chose the latter in the end through thorough testing because it had improved performance, scalability, and developer experience.

The official system is made up of three collaborating agents: an orchestrator that interprets the user's intent and assigns tasks, a data analysis agent that can create, run, and interpret Python code, and a Visualization agent that creates charts when queried by the user. All agents have shared memory, reason as a group, and employ reusable tools for parsing, running code, and explaining results.

The interface, designed by Streamlit, is a simple, interactive chat-based interface. It is facilitated with persistent memory, reproducible results, and asynchronous processing, which makes the system applicable to public administration practitioners as well as broader civic participation.

The rest of this report describes the methods, tools, and design decisions that guided the evolution of this system from early prototypes through the integrated platform.

# Environment Setup

Our GitHub repository is structured into three main components:

- **main.ipynb**: This Jupyter notebook contains:
  - the initial Exploratory Data Analysis,
  - the complete definition of all tools, including InsightBuilder, CodeRunner, and ResultExplainer,
  - the token-aware memory manager (`agent_with_memory`),
  - the three OpenAI agents (`ConversationAgent`, `DataProcessingAgent`, and `VisualizationAgent`),
  - a one-liner command to launch the Streamlit interface locally.

- **main.py**: This Python script replicates the same logic contained in `main.ipynb`, specifically all agent and tool definitions, the memory logic, and the loaded datasets. It acts as the backend for the Streamlit interface, ensuring that Streamlit can access all functions and agent logic. This separation is required because Streamlit apps must import core logic from external `.py` modules rather than notebooks.

- **app.py**: This is the Streamlit front-end script. It defines the interface layout, manages the chat history and memory buffer, and allows users to:
  - enter queries,
  - clear the conversation,
  - re-run the last command,
  - view generated visualizations and results inline.

- **requirements.txt**: The file that includes all the necessary Python libraries and version specifications used in the development of the project (e.g., `openai`, `streamlit`, `matplotlib`, `pandas`, etc.). This setup ensures that any user with Python installed can fully reproduce the functionality and interface of the project, whether to extend it, test it, or deploy it.
  - To recreate the full environment, simply run the following command from the root of the repository:

    ```bash
    pip install -r requirements.txt
    ```

- **Evaluation_Questions.csv**: This file contains a detailed log of the system evaluation process, including:
  - natural-language queries,
  - the expected output (computed via manual Python code),
  - the actual output produced by the agents,
  - a binary field indicating whether the output was correct,
  - a final accuracy metric.

- **Additionals/**: A folder containing the code that was not included in our official version but mentioned in the README.

## How to Run

To directly try our agents, you can run Streamlit by:

- **A)** Going to the end of the `main.ipynb` notebook in the Streamlit section and running the cell:

  ```bash
  !streamlit run app.py
  ```
- **B)** by running from the root of the repository
  ```bash
  streamlit run app.py
  ```

# Data Analysis & Preprocessing

We started our project by conducting an extensive EDA to assess data quality, structure, and completeness across all four datasets used in the system. Each dataset captures a different facet of public sector personnel, allowing for a multi-dimensional exploration of digital service access, salary distribution, mobility, and socio-demographic patterns.

- **EntryAccessoAmministrati_202501.csv** contains portal access records, grouped by region of residence, administration type, age range, gender, and authentication method. This dataset reflects how users interact with digital public services.
- **EntryAccreditoStipendi_202501.csv** reports the number of salary payments disbursed, segmented by administration, municipality, gender, age group, and payment method (e.g., bank transfer, prepaid card).
- **EntryPendolarismo_202501.csv** captures commuting patterns: including workplace province and municipality, whether employees work in their home municipality, commuting distance ranges (min/max), and employee counts per route.
- **EntryAmministratiPerFasciaDiReddito_202501.csv** provides a demographic breakdown of personnel by region, sector, income bracket, age group, and gender, giving insight about the distribution of public employees across income levels.

Several columns are shared across datasets, such as gender, administration, municipality, age_min/age_max, and region_of_residence, that will enable meaningful joins and cross-analysis.

The structure of the datasets reveals key behavioral and demographic dimensions of the public sector workforce:

- **Regional and Sectoral Engagement Gaps**: We observed strong disparities in how regions engage with public digital services. For example, Campania and Lazio stand out, each logging over 218000 portal accesses. Similarly, the Ministry of Education dominates both platform usage and salary counts, likely due to its scale and workforce size.
- **Gender Discrepancies**: Although women account for over 61% of salary recipients, they represent only 46% of portal users. This suggests that female employees might be underrepresented in digital platform interactions or face access limitations.
- **Authentication Preferences**: SPID (the Italian Public Digital Identity System) is now the default and most used login method, while CIE (Electronic ID) and CNS (National Services Card) remain marginal, possibly due to usability or availability issues.
- **Mobility Patterns**: Commuting data reveals that 3 out of 4 public employees travel outside their home municipality. Major commuting hubs include Rome, Naples, and Milan, with most travel occurring over short distances, but long-range commuting is still non-negligible in some areas.
- **Territorial Gaps**: Interestingly, several regions appear in salary or employment records but do not appear in access data, pointing to territorial gaps in the uptake of digital services.

A major issue identified early was inconsistent column types. Several fields that should be numerical (e.g., age_max, income_bracket_min, distance_max_km) were incorrectly stored as strings or floats, often due to formatting (e.g., "oltre i 28000"). To resolve this all numeric fields were converted, and textual noise was stripped. This normalization step was crucial for enabling the agents to automatically match columns to user queries, infer correct data types when generating Python code (e.g., numeric vs. categorical), and avoid runtime errors when aggregating, plotting, or filtering data.

Each dataset originally came with heterogeneous Italian column names, but to enable easier semantic interpretation and ensure cross-dataset consistency we translated all columns in English, making sure that columns with the same meaning in different datasets had the same name.

The EDA revealed both strong foundations for analysis and critical preprocessing requirements. Thanks to a standardization effort, all datasets were successfully aligned in terms of column names, data types, and structure. These steps were fundamental to enabling robust querying, seamless code generation, and agent-based reasoning in subsequent phases of the project.

# Methods

At the beginning of this project, we explored two alternative approaches to building our multi-agent system: a local agent architecture using lightweight open-source models (specifically mistral-nemo), and a cloud-based architecture leveraging the newly released OpenAI Agent SDK. This dual-track strategy was initially motivated by both technical and strategic considerations.

On one hand, the local setup offered greater privacy and full offline control, giving us autonomy from third-party providers and allowing deployment in restricted environments. On the other hand, OpenAI's Agent SDK, despite being newly introduced during the development window, offered a compelling promise: having access to the OpenAI model to create modular, function-aware agents capable of interpreting complex instructions, calling structured tools, and reasoning cooperatively through orchestration and memory.

However, given that these were new tools, the implementation difficulty and the timeline for achieving a functional system were uncertain. This led us to adopt a dual-track development strategy: simultaneously pursuing this higher-risk, higher-reward OpenAI path alongside a more established local AI agent approach, each with its own distinct set of pros and cons.

After testing both paths, we ultimately chose the OpenAI Agent SDK for our final implementation. While the local agent proved viable and well-designed, it required extensive prompt engineering and the inclusion of robust code-level cleanups to mitigate issues such as model hallucination due to its limited reasoning capabilities and was far more demanding in terms of hardware. The OpenAI approach, in contrast, was more powerful, flexible, and, despite initial concerns, surprisingly easy to implement thanks to its built-in support for agent composition and function routing. Accuracy was prioritized over privacy as misinformation could lead to irreparable damage. 

## Local Agent Implementation (Initial Prototype)

Our first prototype followed a local-first approach, prioritizing user control and data protection. The system was designed around the mistral-nemo model, chosen for its balance between performance and compute requirements.

To support querying of arbitrary datasets without prior preprocessing, we implemented a column caching system: each dataset was scanned for key metrics (e.g., type, missing values, value range) and cached locally. This metadata was later used to adapt prompts and automatically clean or transform the data before analysis.

The core agent, SmartDataAnalyst, coordinated several tools:

- **NormalizeQueryTool**: parsed the user request into structured, interpretable actions.
- **DataInsights**: analyzed the cached metadata and generated natural-language hints about data quality.
- **DataOperationCodeGenerator** and **VisualizationCodeGenerator**: created executable Python code based on instructions, insights, metadata, and chat history.
- **ExecuteCodeTool**: ran the generated code in a controlled environment.

While this system achieved high modularity and interpretability, it required complex prompt engineering to overcome model limitations, extensive error handling, and robust post-generation cleanup to ensure functional outputs.

The greatest trade-off in our opinion was performance: while privacy-preserving and locally executable, the system was slower, more fragile, and almost not responsive at all to ambiguous queries.

## OpenAI SDK Implementation (Final Architecture)

Our second and definitive approach focuses on increasing the agent's reasoning power: it adopts a modular multi-agent architecture powered by the newly released OpenAI Agent SDK, a tool introduced during the same weeks as this project’s development. The Agent SDK represents a significant innovation in the field of intelligent systems, not merely due to its modularity—which is also achievable in other frameworks—but primarily because it provides direct access to OpenAI’s most powerful models, such as GPT-4. This enhances the reasoning depth and reliability of each agent in the system.

The SDK enables function-aware agents to communicate, orchestrate decisions, and execute structured workflows autonomously. Unlike traditional local LLM pipelines, which may be constrained by limited model capacity or hardware, this cloud-based approach supports:

- Agent roles with persistent instructions
- Composable tools (via `@function_tool`) that simulate external APIs or operations
- Structured handoffs and routing logic between agents

All while leveraging the advanced cognitive capabilities of state-of-the-art OpenAI models.

### Agent Architecture

We implemented three coordinated agents:

1. **ConversationAgent**: The orchestrator agent. It interprets user queries, leverages token-aware memory (`agent_with_memory()`), and routes tasks to the appropriate sub-agent(s):
   - To **VisualizationAgent** for chart generation
   - To **DataProcessingAgent** for analysis and interpretation
   - Or to both, if the query requires analysis + plot

2. **VisualizationAgent**: This agent is specialized in:
   - Understanding the request and generating full matplotlib code via `InsightBuilder(prompt)`
   - Executing it via `CodeRunner(code)`
   - Optionally calling `ResultExplainer` to comment on the chart

3. **DataProcessingAgent**: This agent handles textual analysis and statistical reasoning. It also:
   - Uses `InsightBuilder` to generate Python analysis code
   - Runs the code via `CodeRunner`
   - Passes the printed output to `ResultExplainer` for natural-language summary

All tools are implemented with `@function_tool`, exposing functionality to the agents:

- **InsightBuilder**: Understands, parses the query and produces a complete, modular Python script (data loading, filtering, aggregation, plotting or computation).
- **CodeRunner**: Executes the generated code in a controlled namespace, capturing stdout output.
- **ResultExplainer**: Explains the results based on the output and user intent.

These tools are reusable and inspectable, and follow a strict workflow including feature identification, dataset validation, aggregation and filtering logic, error handling and plotting conventions (e.g. using subplots if needed).

## Memory Management

The `agent_with_memory()` function implements token-aware conversation memory. It dynamically manages the chat history passed to the agent by:

- Counting tokens using `tiktoken` with the `cl100k_base` encoding (compatible with GPT-4.1)
- Truncating older queries if the total context length exceeds a token budget
- Reconstructing a coherent query by prepending relevant conversational context
- Passing the final prompt to the selected agent for execution

This ensures the agent remains aware of prior exchanges, enabling follow-up questions, chained instructions, and context-dependent analysis, all while respecting token constraints imposed by the model.

## Reproducibility and Execution

All outputs are generated from the actual datasets loaded into memory, using Python code that is both generated and executed without hallucination or fake data. Charts and insights are created live, and the result includes both code output and explanation (for visualization only when necessary or useful), making the system fully auditable.

## Project Evolution: From Hardcoded Tools to Prompt-Based Reasoning

As mentioned earlier, as we progressed through the project, we realized that adopting the OpenAI Agent SDK significantly simplified the architecture compared to our initial implementation.

While the **Visualization Agent** was already modular and robust from the beginning, the **Data Processing Agent** had originally been built using a much more hardcoded and rigid toolset.

In the initial version, the `DataProcessingAgent` relied on four tools (you can find them in the folder `Additionals/Hardcoded_DataProcessAgent_Tools.py`):

- `match_data`: a semantic matching engine using LlamaIndex embeddings to identify relevant datasets and columns based on the user’s query. Based on its output, the `DataProcessingAgent` would call one of the other three tools.
- `single_column`: a statistical analysis tool for a single column, covering mean, variance, outliers, normality tests, and data quality.
- `multi_column`: a group-by aggregation tool handling various aggregation functions like mean, sum, or count, including edge-case handling and synonym resolution.
- `multi_dataset`: a broad cross-dataset inspection tool for queries involving multiple domains.

### Drawbacks of the Old Architecture

- **High maintenance complexity**: Each tool had hundreds of lines of logic and custom condition handling.
- **Limited flexibility**: New types of analysis or unseen query formulations required manual updates.
- **Slower performance**: Semantic matching introduced latency.
- **Hard separation of concerns**: Chaining operations (e.g., filter + group + stats) was hard to manage across tools.

### Transition to Prompt-Based Reasoning

We began experimenting with a prompt-only solution for the generation of Python code, as was already the case for the `VisualizationAgent`. This led us to consolidate the tools into a single, smarter module: `InsightBuilder`.

In the intermediate phase, we used a two-step pipeline:

1. `match_data`: parsed the user query, identified relevant dataset(s), and selected appropriate numeric and grouping columns.
2. `InsightBuilder`: received the parsed metadata and built executable Python code.

Eventually, we discovered that `InsightBuilder` alone was capable of:

- Understanding the query
- Parsing the correct dataset and columns
- Generating the appropriate Python code—all in one step

So we **completely removed** `match_data` from the pipeline.

### Benefits of the New Prompt-Based Design

- Faster execution (no embedding computation)
- More flexible with free-form queries
- Reduced token usage and latency
- Robust to vague/ambiguous input
- Chainable operations in one call

The only limitation was rare minor hallucination, but results remained executable, reproducible, and easier to debug.

### Final Note: Agent Fusion Attempt

We also experimented with merging `DataProcessingAgent` and `VisualizationAgent` into a single agent. However, this approach performed worse: the system had difficulties distinguishing whether the query required only insights, only plots, or both.

### Final Architecture

So, to recap, our final version includes:

- A **ConversationAgent** for query interpretation and routing
- A **DataProcessingAgent** and **VisualizationAgent** that:
  - Use `InsightBuilder` to generate code
  - Use `CodeRunner` to execute it
  - Use `ResultExplainer` to explain it


  

