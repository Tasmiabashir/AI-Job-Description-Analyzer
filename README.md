# AI Job Description Analyzer

A mini GenAI project that takes a raw, unstructured job description and turns it into clean, structured JSON — using prompt engineering techniques instead of a custom ML model. Built as a Week 2 mini project to practice the core building blocks behind real-world LLM tools (HR tech, ATS systems, recruiting bots).

## What it does

Feed it a messy job posting like this:

```
Python Backend Developer
Requirements:
- Python, Django, FastAPI, REST API, PostgreSQL, Git
Experience: 2 Years
Salary: 150000 PKR
```

...and it returns:

```json
{
  "role": "Python Backend Developer",
  "skills": ["Python", "Django", "FastAPI", "REST API", "PostgreSQL", "Git"],
  "experience": "2 Years",
  "salary": "150000 PKR",
  "summary": "Backend Python Developer"
}
```

It then simulates a **function call** (`get_salary(role)`) to demonstrate how an LLM can trigger real backend logic based on extracted data — a core pattern behind tool-using AI agents.

## Concepts Covered

- Prompt Engineering
- Role Prompting (`"You are a Senior HR Manager"`)
- Few-Shot Prompting (one input/output example before the real task)
- Prompt Templates (f-string based)
- Structured JSON Output from an LLM
- Parsing LLM output with `json.loads()`
- Function Calling (simulated)

## Project Structure

```
ai-job-description-analyzer/
│
├── job_description_analyzer.py   # Main script (prompting, model call, JSON parsing, function calling demo)
└── README.md
```

This is a single-script project — no backend/frontend split needed at this stage.

## Setup

```bash
pip install transformers sentencepiece accelerate
python job_description_analyzer.py
```

Or run directly in Google Colab / Jupyter.

> **Note:** `transformers` v5.x removed the `text2text-generation` pipeline shortcut. This project loads the model directly via `AutoTokenizer` + `AutoModelForSeq2SeqLM` instead of using `pipeline()`, so it works on any `transformers` version.

## Tech Stack

- **Python**
- **Hugging Face Transformers** (`AutoTokenizer`, `AutoModelForSeq2SeqLM`)
- **google/flan-t5-large** — free, open-weight instruction-following model
- **SentencePiece** — tokenizer backend for T5
- **Accelerate** — efficient model loading
- **json** (Python standard library) — parsing structured LLM output

## Notes

- `flan-t5-large` is a small open model, so JSON output isn't always 100% valid — this is a known limitation of smaller models compared to API-based models like GPT-4o or Claude. The `try/except` around `json.loads()` handles this gracefully.
- The function-calling step is **simulated** (a hardcoded dictionary lookup) rather than a real LLM-triggered tool call — this mirrors the simplified version of function calling before plugging into a real framework like OpenAI's function calling or LangChain tools.
- Possible next step: add a retry-on-failure loop that re-prompts the model if `json.loads()` fails, to make JSON output more reliable.
