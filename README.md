# Introduction
This project examines the design and creation of an intelligent, modular multi-agent system to natural language-based data analysis and visualization. The system was built to provide an accessible, flexible, and auditable interface that allows users, especially non-technical ones, to query complex public sector data using easy conversation-based queries.

The overarching issue addressed in this project is how to bridge the gap between unstructured user input and structured data analysis, without invoking rigid pre-programmed logic. To do this, we designed a multi-agent system powered by OpenAI's newly announced Agent SDK, which allows function-aware agents to interpret instructions, call tools, and coordinate tasks in a modular and composable way.

Throughout the project, we used a two-path development methodology: one on local open-source frameworks to give offline independence and confidentiality and the other on cloud-native frameworks to have maximum reasoning capacity and deployment ease. We chose the latter in the end through thorough testing because it had improved performance, scalability, and developer experience.

The official system is made up of three collaborating agents: an orchestrator that interprets the user's intent and assigns tasks, a data analysis agent that can create, run, and interpret Python code, and a Visualization agent that creates charts when queried by the user. All agents have shared memory, reason as a group, and employ reusable tools for parsing, running code, and explaining results.

The interface, designed by Streamlit, is a simple, interactive chat-based interface. It is facilitated with persistent memory, reproducible results, and asynchronous processing, which makes the system applicable to public administration practitioners as well as broader civic participation.

The rest of this report describes the methods, tools, and design decisions that guided the evolution of this system from early prototypes through the integrated platform.

# Environment Setup

Our GitHub repository is structured into three main components:
•	main.ipynb: This Jupyter notebook contains:
o	the initial Exploratory Data Analysis,
o	the complete definition of all tools, including InsightBuilder, CodeRunner, and ResultExplainer,
o	the token-aware memory manager (agent_with_memory),
o	the three OpenAI agents (ConversationAgent, DataProcessingAgent, and VisualizationAgent),
o	a one-liner command to launch the Streamlit interface locally.

•	main.py: This Python script replicates the same logic contained in main.ipynb, specifically all agent and tool definitions, the memory logic, the loaded datasets. It acts as the backend for the Streamlit interface, ensuring that Streamlit can access all functions and agent logic. This separation is required because Streamlit apps must import core logic from external .py modules rather than notebooks.

•	app.py: This is the Streamlit front-end script. It defines the interface layout, manages the chat history and memory buffer, and allows users to:
o	enter queries,
o	clear the conversation,
o	re-run the last command,
o	view generated visualizations and results inline.

•	requirements.txt: The file that includes all the necessary Python libraries and version specifications used in the development of the project (e.g., openai, streamlit, matplotlib, pandas, etc.). This setup ensures that any user with Python installed can fully reproduce the functionality and interface of the project, whether to extend it, test it, or deploy it.
o	To recreate the full environment, simply run the following command from the root of the repository: pip install -r requirements.txt

•	Evaluation_Questions.csv: This file contains a detailed log of the system evaluation process, including:
o	natural-language queries,
o	the expected output (computed via manual Python code),
o	the actual output produced by the agents,
o	a binary field indicating whether the output was correct,
o	a final accuracy metric.

•	Additionals: a folder containing the code that was not included in our official version but mentioned in the ReadMe

To directly try our agents, you can run Streamlit by 
-	A) going at the end of the main.ipynb notebook in the Streamlit section and running the cell !streamlit run app.py
-	B) by running streamlit run app.py from the root of the repository.

![image](https://github.com/user-attachments/assets/d4cd6315-de61-414a-b0f3-fc898a07e539)
