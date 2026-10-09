# 💊 PharmaSense AI — Agentic AI for Pharmaceutical Research

## 📌 Project Overview

**PharmaSense AI** is an AI-powered pharmaceutical research assistant designed to support drug discovery and clinical research workflows.

The project aims to use Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and AI agents to analyze pharmaceutical research documents, retrieve relevant information, summarize findings, and assist researchers in making informed decisions.

The project uses a synthetic pharmaceutical dataset for development and experimentation.

## 🎯 Project Objectives

* Analyze pharmaceutical compound information.
* Explore clinical trial data and research outcomes.
* Retrieve relevant information from research documents using RAG.
* Use an LLM to generate context-aware summaries and answers.
* Develop AI agent workflows to support research tasks.
* Organize research information for easier access and analysis.

## 🗂️ Dataset Description

The project dataset contains seven tables representing different aspects of pharmaceutical research.

| Table            | Description                                       |
| ---------------- | ------------------------------------------------- |
| `compounds`      | Pharmaceutical compound details                   |
| `trials`         | Clinical trial information and timelines          |
| `sites`          | Clinical trial site information                   |
| `lab_results`    | Laboratory results and measurements               |
| `adverse_events` | Recorded adverse events                           |
| `research_doc`   | Research document metadata and content references |
| `agent_logs`     | AI agent activity and execution logs              |

**Note:** The dataset is synthetic and is intended for project development and learning, not real-world clinical decision-making.

## 🛠️ Technologies Used

* **Programming:** Python
* **Data Analysis:** Pandas, NumPy
* **Development Environment:** Visual Studio Code, Jupyter Notebook
* **LLM:** Anthropic Claude API (planned integration)
* **AI Techniques:** Retrieval-Augmented Generation (RAG), prompt engineering, AI agents
* **Data Storage and Processing:** Excel, CSV, and Python data-processing tools
* **Version Control:** Git and GitHub

*The technology list will be updated as each component is implemented.*

## 🧠 Key Concepts

### 1. Large Language Models (LLMs)

LLMs process natural-language questions and generate responses based on the input and the context provided to them.

### 2. Retrieval-Augmented Generation (RAG)

RAG retrieves relevant information from a project's research documents and supplies that information to the LLM to help generate grounded answers.

### 3. AI Agents

AI agents can coordinate tasks such as searching research information, retrieving documents, summarizing findings, and recording task execution.

### 4. Data Analysis

Data profiling and cleaning help identify missing values, understand relationships between datasets, and prepare data for downstream AI workflows.

## ⚙️ Project Workflow

1. **Data Understanding** — Inspect the seven datasets and understand their structure.
2. **Data Cleaning** — Validate data types, missing values, duplicates, and identifiers.
3. **LLM Setup** — Configure the language model API securely.
4. **Document Processing** — Prepare research documents for retrieval.
5. **RAG Implementation** — Retrieve relevant document passages for user questions.
6. **Agent Development** — Build workflows to coordinate research tasks.
7. **Testing and Evaluation** — Test responses for relevance, accuracy, and appropriate source grounding.
8. **Documentation** — Record the architecture, implementation, and results.

## 📁 Project Structure

```text
pharmasense_AI/
│
├── data/
│   ├── compounds
│   ├── trials
│   ├── sites
│   ├── lab_results
│   ├── adverse_events
│   ├── research_doc
│   └── agent_logs
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── rag_pipeline.py
│   └── agent_workflow.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*This is a suggested structure. Update it to match the files and folders actually present in your repository.*

## 🔐 API Key Security

API keys and credentials are stored in a local `.env` file and must never be committed to GitHub.

The `.env` file is excluded through `.gitignore`. The repository may include `.env.example` with placeholder values only.

## 🚀 Expected Outcomes

* A structured pharmaceutical research dataset.
* An LLM-powered question-answering workflow.
* A RAG pipeline for retrieving relevant research information.
* AI agent workflows for selected research tasks.
* A documented and reproducible AI project suitable for a technical portfolio.

## ⚠️ Disclaimer

This project is intended for educational and research purposes. It is not a substitute for professional medical advice, clinical validation, regulatory review, or decisions made by qualified healthcare professionals.

## 👩‍💻 Author

**Thenmozhi G**

GitHub: [honey-boot](https://github.com/honey-boot)

---

⭐ If you find this project interesting, feel free to explore the repository and follow its development.
