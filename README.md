# Multi Agent System for NoiPA
**Team Members**:
- Paolina Beatrice Mazza (p.mazza@studenti.luiss.it)
- Carlotta Menasci (carlotta.menasci@studenti.luiss.it)
- Alessandro Pausilli (alessandro.pausilli@studenti.luiss.it)

# Introduction

This project examines the design and creation of an intelligent, modular multi-agent system that takes natural language-based user query as input and produces accurate data analysis and visualization.

The overarching issue addressed in this project is how to bridge the gap between unstructured user input and structured data analysis, without relying on rigid pre-programmed logic. To do this, we designed a multi-agent system powered by OpenAI's newly announced Agent SDK, which allows function-aware agents to interpret instructions, call tools, and coordinate tasks in a modular and composable way.

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

- **Evaluation_Questions.csv**: This file contains a detailed log of the system evaluation process, including:
  - natural-language queries,
  - the expected output (computed via manual Python code),
  - the actual output produced by the agents,
  - a binary field indicating whether the output was correct,
  - a final accuracy metric.

- **EntryAccessoAmministrati_202501.csv**, **EntryAccreditoStipendi_202501.csv**, **EntryAmministratiPerFasciaDiReddito_202501.csv**, and **EntryPendolarismo_202501.csv**: The four structured datasets, containing public sector administrative information, the project relied on.

- **Additionals**: A folder that contains the files with the code that was not included in our official version but mentioned in the README, specifically **hardcoded_DataProcessAgent_Tools.py** and **local_agent_project.py**.

- **Images**: A folder containing all the images present in this README.

- **noipalogo.png**: The logo, created by us on Canva, for our Streamlit frontend interface.

- **MULTI-AGENT SYSTEM.pptx**: Our powerpoint presentation of the project.

- **requirements.txt**: The file that includes all the necessary Python libraries and version specifications used in the development of the project (e.g., `openai`, `streamlit`, `matplotlib`, `pandas`, etc.). This setup ensures that any user with Python installed can fully reproduce the functionality and interface of the project, whether to extend it, test it, or deploy it.

## How to Run
To run this project locally and ensure everything works as expected, follow these steps:

### 1. Download or clone the repository
```bash
git clone https://github.com/paolinamazza/Multi-Agent-System791391.git
cd Multi-Agent-System
```

### 2. Create and activate a virtual environment
It is highly recommended to use a virtual environment to avoid conflicts with other Python packages.
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install all dependencies
You can recreate the whole Python environment by just running:
```bash
pip install -r requirements.txt
```
This will install all required libraries (such as openai, streamlit, pandas, matplotlib, etc.) listed in the requirements.txt file.

### 4. Set the OpenAI API Key
To use the OpenAI-powered agents, you need to provide a valid API key. This is done directly in the `main.py` file, right after the library imports.
You will find the following section in main.py:
```bash
# ----- OPENAI API KEY -----
openai_api_key = "...."
os.environ['OPENAI_API_KEY'] = openai_api_key
```

Replace the "...." with your personal OpenAI API key.


### 5. Run the application
To directly try our agents, you can run Streamlit in one of the following ways:
- **A)** Going to the end of the `main.ipynb` notebook in the Streamlit section and running the cell: ```bash !streamlit run app.py ``` 
- **B)** by running from the root of the repository ```bash streamlit run app.py ```

After these steps, the Streamlit interface will open in your browser, allowing you to interact with the agents through a chat-based interface.


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

Our first prototype followed a local-first approach, prioritizing user control and data protection. The system was designed around the mistral-nemo model, chosen for its balance between performance and compute requirements. Although it is not part of our official project, we decided to include in the repository its code, that you can find in the `local_agent_project.py` file in the `Additionals/` folder.

To support querying of arbitrary datasets without prior preprocessing, we implemented a column caching system: each dataset was scanned for key metrics (e.g., type, missing values, value range) and cached locally. This metadata was later used to adapt prompts and automatically clean or transform the data before analysis.

The core agent, SmartDataAnalyst, coordinated several tools:

- **NormalizeQueryTool**: parsed the user request into structured, interpretable actions.
- **DataInsights**: analyzed the cached metadata and generated natural-language hints about data quality.
- **DataOperationCodeGenerator** and **VisualizationCodeGenerator**: created executable Python code based on instructions, insights, metadata, and chat history.
- **ExecuteCodeTool**: ran the generated code in a controlled environment.

While this system achieved high modularity and interpretability, it required complex prompt engineering to overcome model limitations, extensive error handling, and robust post-generation cleanup to ensure functional outputs.

The greatest trade-off in our opinion was performance: while privacy-preserving and locally executable, the system was slower, more fragile, and almost not responsive at all to ambiguous queries.

## OpenAI SDK Implementation (Final Architecture)
Our second and definitive approach focuses on increasing the agent's reasoning power: it adopts a modular multi-agent architecture powered by the newly released OpenAI Agent SDK, a tool introduced during the same weeks as this project’s development. The Agent SDK represents a significant innovation in the field of intelligent systems, not merely due to its modularity—which is also achievable in other frameworks—but primarily because it provides direct access to OpenAI’s most powerful models, such as GPT-4. This enhances the reasoning depth and reliability of each agent in the system. The SDK enables function-aware agents to communicate, orchestrate decisions, and execute structured workflows autonomously. Unlike traditional local LLM pipelines, which may be constrained by limited model capacity or hardware, this cloud-based approach supports agent roles with persistent instructions, composable tools (via @function_tool) that simulate external APIs or operations, and structured handoffs and routing logic between agents—all while leveraging the advanced cognitive capabilities of state-of-the-art OpenAI models.

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

### InsightBuilder: 
Transforms a natural language query into a modular, executable Python script that performs data analysis or visualization based on the user’s request. It is a prompt-only tool, no code needed here.


**How it works**:
1. **Intent Understanding**  
   Parses the user’s natural-language request and identifies:
   - What kind of analysis is needed (e.g., correlation, grouping, plotting).
   - All sub-tasks or multiple intents (e.g., filtering + comparison).
   - Which columns and datasets are involved.

2. **Dataset Resolution**  
   Recognizes which variables exist in which of the four available datasets (`df_accesso`, `df_pendolarismo`, `df_amministrati`, `df_stipendi`), and classifies them as categorical or numeric.

3. **Join Handling**  
   If a query requires merging datasets:
   - It automatically determines the correct key.
   - Uses string similarity to resolve ambiguous column names.

4. **Modular Code Generation**  
   Generates a full Python script with the following mandatory structure:
   - Load and inspect datasets.
   - Validate columns.
   - Identify grouping and target variables.
   - Perform computation (aggregation, correlation, etc.).
   - Generate subplots if more than one plot is needed.
   - Print debug outputs for validation.

5. **Final Output**  
   Returns a clean Python script as a string, ready to be executed by the `CodeRunner`.


### CodeRunner:
Executes the Python script generated by `InsightBuilder` inside a controlled namespace, and returns only the printed output.


**Execution details**:
- All four datasets (`df_accesso`, `df_pendolarismo`, `df_amministrati`, `df_stipendi`) are preloaded in memory.
- Uses `contextlib.redirect_stdout` to capture any printed output.
- Automatically resets plots via `plt.close('all')` before execution.
- Handles exceptions and returns the full traceback in case of error.


### ResultExplainer:
Takes the Python code, its printed output, and the original query, and produces a clear, fluent natural-language explanation of the result.

**Process**
- Uses OpenAI’s GPT-4.1 model to interpret:
  - What was computed in the code.
  - What the printed output shows (e.g., numbers, tables, summaries).
- If a table is present, it is re-rendered in clean, readable Markdown.
- If a single value or numeric output is returned, it is interpreted and contextualized.

**Final Output**  
A well-formed, user-friendly explanation that clearly describes the insight, conclusion, or pattern identified by the executed analysis — even for non-technical users.

![FinalVersion](Images/FinalVersion.png)


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

In the initial version, the `DataProcessingAgent` relied on four tools (although they are not part of our official project, we decided to include in the repository their code, that you can find in the `hardcoded_DataProcessAgent_Tools.py` file in the `Additionals/` folder):

- `match_data`: a semantic matching engine using LlamaIndex embeddings to identify relevant datasets and columns based on the user’s query. Based on its output, the `DataProcessingAgent` would call one of the other three tools.
- `single_column`: a statistical analysis tool for a single column, covering mean, variance, outliers, normality tests, and data quality.
- `multi_column`: a group-by aggregation tool handling various aggregation functions like mean, sum, or count, including edge-case handling and synonym resolution.
- `multi_dataset`: a broad cross-dataset inspection tool for queries involving multiple domains.

![FirstVersion](Images/FirstVersion.png)


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

![SecondVersion](Images/SecondVersion.png)

### Benefits of the New Prompt-Based Design

- Faster execution (no embedding computation)
- More flexible with free-form queries
- Reduced token usage and latency
- Robust to vague/ambiguous input
- Chainable operations in one call

The only limitation was rare minor hallucination, but results remained executable, reproducible, and easier to debug.

### Final Note: Agent Fusion Attempt

We also experimented with merging `DataProcessingAgent` and `VisualizationAgent` into a single agent. However, this approach performed worse: the system had difficulties distinguishing whether the query required only insights, only plots, or both.
![SingleAgent](Images/SingleAgent.png)


### Final Architecture

So, to recap, our final version includes:

- A **ConversationAgent** for query interpretation and routing
- A **DataProcessingAgent** and **VisualizationAgent** that:
  - Use `InsightBuilder` to generate code
  - Use `CodeRunner` to execute it
  - Use `ResultExplainer` to explain it

![FinalVersion](Images/FinalVersion.png)


# Evaluation & Results

To evaluate the performance of our agents, we conducted a structured comparison against a set of manually written baseline scripts that we consider as ground truth for each query, that you can find in `Evaluation_Questions.csv`. The purpose was to verify the correctness and robustness of our agents when responding.

Each experiment in our evaluation followed this structure:

- **Prompt**: A natural-language question posed to the agent.
- **Expected Output + Python Code**: The correct answer, computed using a manually written Python script.
- **Agent Output**: The answer generated by our multi-agent system.
- **Correctness Evaluation**: A binary label (YES/NO) indicating whether the agent's response matched the expected answer.

### Quantitative Results

To measure the system's performance, we computed the **Accuracy** metric, defined as the percentage of correctly answered queries. The system achieved:

- **96% accuracy**, correctly answering **22 out of 23** queries.
- Coverage across **visual, statistical, and multi-dataset** queries.
- Correct handling of **ambiguous queries**, vague phrasing, and cross-dataset reasoning in most cases.

### Qualitative Observations

Throughout development, we tested the agents continuously to iteratively improve reliability. Some observations from repeated experiments:

- **Consistency**: The agents reliably regenerate the same result for deterministic queries, even across sessions.
- **Reasoning Depth**: They can parse vague or layered requests like and produce correct logic chaining across datasets.
- **Dynamic Plotting**: Visualizations are always grounded in data and generated live from code, the agents never return static or hallucinated figures (see examples in `Evaluation_Questions.csv`)
- **Multiple Languages**: They are also able to understand and respond to queries formulated both in English and Italian, making the system accessible to a broader range of users in public administration contexts.

### Key Strengths

- **Tool Composability**: By modularizing code generation (`InsightBuilder`), execution (`CodeRunner`), and explanation (`ResultExplainer`), agents remain interpretable and extensible.
- **Multi-dataset Integration**: The system can combine datasets (e.g., Access + Stipendi + Amministrati) in real time, dynamically joining based on column alignment and semantic hints.
- **Reproducibility**: All results are generated with real datasets using traceable Python code—no hardcoding or hidden logic.

### Known Limitations

Despite strong performance, the system presents a few limitations:

- **Occasional Hallucinations**: In a small number of cases, especially with vague or underspecified prompts, the agent may hallucinate column names or logic. For example, it once generated "Biometric" as an authentication method that does not exist in the data.
- **Join Complexity**: Merging datasets with partial overlaps (e.g., inconsistent admin names across files) was the most challenging task. This required injecting explicit logic in `InsightBuilder` to force safe joins with validation steps.
- **Latency**: On larger queries with multiple tools involved, the system may take several seconds to respond due to live code execution.

## Evaluation with Custom Analytical Queries
After verifying the agent's technical accuracy and reproducibility, we tested it using a curated set of five complex analytical queries that were specifically designed to challenge the system.

The queries were formulated to take full advantage of the four NoiPA datasets and aimed to assess the agent’s analytical capabilties. 

### Question 1
Calculate the percentage distribution of access methods to the NoiPA portal among users aged 18-30 compared to those over 50, broken down by region of residence.
![ReplyQ1](Images/ReplyQ1.png)


### Question 2
Identifiy the most used payment method for each age group and generate a graph showing whether there are correlations between gender and payment method preference.
![ReplyQ2](Images/ReplyQ2.png)


### Question 3
Analyze commuting data to identify which administrations have the highest percentage of employees who travel more than 20 miles to work.
![ReplyQ3](Images/ReplyQ3.png)


### Question 4
Compare the gender distribution of staff among the five municipalities with the largest number of employees, highlighting any significant differences in representation by age group.
![ReplyQ4](Images/ReplyQ4.1.png)
![ReplyQ4](Images/ReplyQ4.2.png)


### Question 5
Determine if there is a correlation between the method of portal access (from EntryAccessAdministration) and the average commuting distance (from EntryPendularity) for each administration.
![ReplyQ5](Images/ReplyQ5.png)



Despite the high level of difficulty and the variety of analysis types required, the agent produced correct and complete answers for all 5 queries. Each response included accurate computations, proper use of the right datasets, and relevant insights or visual outputs when appropriate.

This confirms that the agent is both technically robust and semantically reliable, capable of handling multi-step analytical queries with high precision, making it a powerful assistant for real-world public sector data exploration.











## Comparing the Two Agent Architectures in Practice

After having proved the high accuracy of our multi-agent system, we were interested in comparing it to our abandoned local version. The screenshots below show how the two systems—our OpenAI-based agent (light blue background) and the local agent created with Mistral-Nemo (black background)—responded to the identical three questions on topics like gender inequality, incomes, and digital authentication methods.
  
### Question 1
Calculate the average of the column minimum distance in KM and maximum distance in KM and plot the result in a bar chart.
![LocalQ11](Images/LocalQ11.jpeg)
![LocalQ12](Images/LocalQ12.jpeg)
![LocalQ13](Images/LocalQ13.jpeg)
![OAQ1](Images/OAQ1.jpeg)
![OAQ12](Images/OAQ12.jpeg)

### Question 2
Are there sectors where women both (a) access the portal less often (Accesso) and (b) earn significantly less than men (Stipendi + Amministrati)?
![LocalQ21](Images/LocalQ21.jpeg)
![LocalQ22](Images/LocalQ22.jpeg)
![OAQ2](Images/OAQ2.jpeg)

### Question 3
For each economic sector, what is the average minimum and maximum income bracket?
![LocalQ3](Images/LocalQ3.jpeg)
![OAQ3](Images/OAQ3.jpeg)

### Question 3
What is the most used authentication method to access the portal in each region?
![LocalQ4](Images/LocalQ4.jpeg)
![OAQ4](Images/OAQ4.jpeg)


The difference is evident right away. The OpenAI agent was also more consistent and more informative. It provided clean, well-formatted answers with readable tables, interpreted ambiguous queries appropriately, and even handled missing data gracefully (e.g., by suggesting alternative approaches when the gender column wasn't present). Above all, its answers always agreed with the available data—it didn't make anything up. 

The local agent, on the other hand, faced a series of issues: it could not properly understand the user query, resulting in code execution errors due to type incompatibilities or missing columns, and when it did the answer was wrong. When prompted for authentication methods by region, for example, it hallucinated authentication types ("Biometric" and "Password") that weren't even in the dataset, as well as wrongly interpret the results (by saying that only Puglia had "Password" when Basilicata did too). 

This last consideration was especially enlightening. It ensured that the local model, while accurate with straight forward queries, for more nuanced ones hallucinated with confidence, resulting in the wrong decisions taken by the user from such erroneous information, negatively affecting countless people. 

Ultimately, these comparisons made it evident that while the local agent provided us with greater control and could be run offline, the OpenAI Agent SDK was the better fit for our project. It was easier to use, smarter in handling real-world data, and much more dependable when it came to producing accurate, grounded results.

## What We Learned

Developing this project provided us with deep insight  agent-based systems built on large language models. One of the most important lessons we learned is that **prompting is everything**: small changes in how a task is phrased—sometimes just one additional sentence—can drastically alter the agent’s behavior, accuracy, or output format. Designing robust, reusable prompts required extensive trial-and-error, as even well-defined tasks could lead to unexpected results without the right structure or examples.

We also discovered that despite their impressive capabilities, **agents are not always fully interpretable**. While we could inspect the generated code and responses, understanding why the model chose one approach over another wasn’t always straightforward. This “black box” dimension makes debugging more about steering behavior through iteration than tracing logic deterministically.

Another key takeaway was the importance of **tool design** and **prompt modularity**. Wrapping logic inside structured tools like `InsightBuilder`, `CodeRunner`, and `ResultExplainer` helped reduce hallucinations and gave us more control over execution. It also forced us to think clearly about the boundaries between user intent, agent reasoning, and final output.

This project didn’t just teach us how to build agents; it taught us how to work with them: to guide them, debug them, collaborate with them—and ultimately, to design systems that are both powerful and accountable.



# Conclusion

We can conclude that our project demonstrates the successful design and implementation of a modular, intelligent multi-agent system for natural language-based data analysis and visualization, tailored for complex public sector datasets. By leveraging the OpenAI Agent SDK, we created a robust and user-friendly interface where users, regardless of technical background, can explore structured data through simple, conversational queries. Our final system features three coordinated agents (`ConversationAgent`, `DataProcessingAgent`, `VisualizationAgent`) working with shared tools and memory to process queries, generate executable code, visualize data, and explain results.

The tool successfully bridges the gap between free-form user input and structured data analysis, offering a powerful, reproducible, and auditable framework that can empower public administration.

Despite these achievements, some challenges and limitations remain. Most notably, the system occasionally struggles with highly ambiguous or complex multi-dataset queries, especially when data requires non-trivial joins. While `InsightBuilder` mitigated many of these cases through structured prompting, occasional hallucinations or execution failures still occur.

Furthermore, our evaluation focused primarily on fixed datasets. Future work should expand the evaluation to broader datasets and include real-world users to assess usability, trust, and performance.

Another promising direction could be integrating a more robust semantic understanding layer to improve dataset matching.



