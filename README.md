# Tool-Calling AI Agent with RAG

A local AI agent built with Python and Ollama that can execute tools, perform multi-step calculations, retrieve information from a document knowledge base, and produce answers using tool results.

This project explores how LLMs can interact with Python functions, how tool arguments can be validated with Pydantic, and how retrieval-augmented generation (RAG) can be integrated into an agent workflow.

## Features

- **Tool calling:** The LLM can request Python functions based on the user's question.
- **Multi-step execution:** The agent can perform sequential operations, such as calculating a percentage and adding the result to another number.
- **Tool validation:** Pydantic validates tool arguments before execution.
- **Centralized tool configuration:** Tool functions and their argument schemas are registered in one configuration.
- **RAG integration:** The agent can retrieve relevant passages from a local document collection.
- **Local embeddings:** Sentence Transformers generates embeddings using `all-MiniLM-L6-v2`.
- **Semantic search:** NumPy cosine similarity ranks document chunks by relevance.
- **Agent tracing:** The application records tool calls, arguments, and results.
- **Automated evaluation:** Unit tests and end-to-end scenarios check calculation correctness, input validation, retrieval, and agent behavior.

## Architecture

```text
                    User question
                          |
                          v
                   Local LLM (Ollama)
                          |
                    Tool selection
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
          Calculator   Date/Time   Knowledge Search
              |           |           |
              |           |       RAG retrieval
              |           |           |
              +-----------+-----------+
                          |
                    Tool results
                          |
                          v
                    Agent loop
                          |
                          v
                    Final answer
```

### RAG retrieval pipeline

```text
Text documents
      |
      v
Document chunking with overlap
      |
      v
Sentence Transformers embeddings
      |
      v
Cosine similarity search
      |
      v
Relevant chunks
      |
      v
Agent receives retrieved context
```

## Technology stack

- Python
- Ollama
- Qwen3 1.7B local model
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model
- NumPy
- Pydantic
- Python `unittest`

## Project structure

```text
project-03-tool-calling/
├── main.py
├── tools.py
├── tool_models.py
├── tool_config.py
├── rag_tool.py
├── embeddings.py
├── similarity.py
├── chunking.py
├── evaluate.py
├── evaluate_agent.py
├── requirements.txt
├── .gitignore
├── documents/
│   ├── python.txt
│   ├── automation.txt
│   ├── databases.txt
│   ├── networking.txt
│   ├── javascript.txt
│   └── ai.txt
└── tests/
    ├── test_tools.py
    ├── test_tool_models.py
    ├── test_agent_logic.py
    └── test_rag.py
    └── test_rag_errors.py
```

*The tree above describes the intended project layout; adjust filenames if your repository differs.*

## Requirements

- Python installed on your computer
- Ollama installed and running
- Sufficient memory to run the selected local model
- Internet access for initial package and embedding-model downloads

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Derobm-code/PROJECT_3_TOOL_CALLING.git
cd project-03-tool-calling
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the local LLM

```bash
ollama pull qwen3:1.7b
```

Make sure Ollama is running before launching the application.

## Run the application

```bash
python main.py
```

Example questions:

```text
What is 25% of 840?

Calculate 25% of 840 and then add 100.

What time is it?

According to my knowledge base, how is Python used for automation?

According to my knowledge base, if a task takes 840 seconds
and its duration is reduced by 25%, how much time is saved
and how long remains?
```

The agent can request relevant tools, receive their results, and continue the conversation until it produces a final response or reaches its iteration limit.

## Run the evaluation suite

Run all deterministic tests and end-to-end agent scenarios:

```bash
python evaluate.py
```

Run only the live agent scenarios:

```bash
python evaluate_agent.py
```

The evaluation checks:

- Calculation correctness
- Pydantic argument validation
- Tool dispatch
- Knowledge-base request detection
- Retrieval of expected documents
- Tool selection and ordering across selected scenarios
- Expected results in multi-step workflows

Evaluation: 23 unit tests and 3 end-to-end agent scenarios passed in the development environment.

**Note:** Passing these tests verifies the implemented functions and selected scenarios. It does not guarantee that a local LLM will always choose the correct tool or produce a fully accurate answer.

## Example workflow

User request:

> According to my knowledge base, Python is used for automation. If a task takes 840 seconds and its duration is reduced by 25%, how much time is saved and how long remains?

Expected tool sequence:

```text
1. search_knowledge_base
2. calculate_remaining_after_reduction
```

Expected calculation:

```text
Original duration: 840 seconds
Time saved:        210 seconds
Remaining duration: 630 seconds
```

## Engineering concepts demonstrated

This project demonstrates practical experience with:

- LLM tool calling and function dispatch
- Multi-step agent execution
- Pydantic validation of model-generated inputs
- Local embedding models
- Semantic retrieval and cosine similarity
- Text chunking and overlapping context
- RAG integration
- Agent execution tracing
- Unit testing and end-to-end evaluation
- Separation of application logic, tools, schemas, and retrieval components

## Current limitations

- The agent uses a small local model, so tool selection and response quality can vary.
- The knowledge base contains a limited collection of local text documents.
- Retrieval uses a simple in-memory index and cosine similarity rather than a dedicated vector database.
- The application currently uses a command-line interface.
- Evaluation covers selected scenarios and does not establish universal reliability.

## Future improvements

- Add a web interface.
- Improve retrieval evaluation and grounding checks.
- Add structured logging and more robust error handling.
- Introduce a persistent vector database.
- Add API-based access through FastAPI.
- Containerize the application with Docker.
- Expand automated evaluation and regression tests.

## Author

Built as a hands-on project to develop practical skills in LLM application engineering, tool calling, RAG, validation, and AI evaluation.