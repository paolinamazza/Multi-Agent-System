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

  

