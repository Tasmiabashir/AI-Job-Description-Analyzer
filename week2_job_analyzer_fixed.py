# ============================================
# Week 2 Mini Project
# AI Job Description Analyzer
# Concepts Covered:
# 1. Prompt Engineering
# 2. Role Prompting
# 3. Few-Shot Prompting
# 4. Prompt Template
# 5. Structured JSON Output
# 6. json.loads()
# 7. Function Calling (Simulation)
# ============================================
# Install Libraries
!pip install -q transformers sentencepiece accelerate
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("Libraries Loaded Successfully!")

# ============================================
# Load Free Hugging Face Model
# ------------------------------------------------
# FIX: transformers v5 removed the "text2text-generation"
# pipeline() shortcut. So instead of pipeline(...), we load
# the tokenizer + model directly and call .generate() ourselves.
# This works on ANY transformers version, old or new.
# ============================================
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model Loaded!")

# ============================================
# Job Description
# ============================================
job_description = """
Python Backend Developer
Requirements:
- Python
- Django
- FastAPI
- REST API
- PostgreSQL
- Git
Experience:
2 Years
Salary:
150000 PKR
"""

# ============================================
# Prompt Engineering
# ============================================
prompt = f"""
You are a Senior HR Manager.
Analyze the following Job Description.
Example:
Input:
Python Developer with Django and FastAPI
Output:
{{
"role":"Python Developer",
"skills":["Python","Django","FastAPI"],
"experience":"2 Years",
"salary":"100000 PKR",
"summary":"Backend Python Developer"
}}
Now Analyze:
{job_description}
Return ONLY JSON.
"""
print("\nPrompt Ready!\n")

# ============================================
# Ask AI
# ------------------------------------------------
# This replaces generator(prompt, max_new_tokens=200)
# Same job, done manually: tokenize -> generate -> decode
# ============================================
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
output_ids = model.generate(**inputs, max_new_tokens=200)
ai_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("="*60)
print("AI RESPONSE")
print("="*60)
print(ai_output)

# ============================================
# JSON Parsing
# ============================================
print("\nAttempting JSON Parsing...\n")
try:
    data = json.loads(ai_output)
    print("Role :", data["role"])
    print("Skills :", data["skills"])
    print("Experience :", data["experience"])
    print("Salary :", data["salary"])
    print("Summary :", data["summary"])
except Exception as e:
    print("Model did not return valid JSON.")
    print("Reason :", e)

# ============================================
# Function Calling Concept
# ============================================
print("\n")
print("="*60)
print("FUNCTION CALLING DEMO")
print("="*60)

def get_salary(role):
    database = {
        "Python Developer":"150000 PKR",
        "AI Engineer":"250000 PKR",
        "Data Scientist":"220000 PKR",
        "Backend Developer":"180000 PKR"
    }
    return database.get(role,"Salary Not Found")

role = "Python Developer"
print("LLM decides to call get_salary()")
salary = get_salary(role)
print("Function Returned :", salary)

# ============================================
# Project Completed
# ============================================
print("\n")
print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)
