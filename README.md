# OmniSupport AI

OmniSupport AI is an intelligent customer support platform that uses a fine-tuned open-source Large Language Model (LLM), Retrieval-Augmented Generation (RAG), and a web frontend to deliver concise, structured, and helpful support replies.

The system is designed to show a clear improvement between:

- the **base model** output, and
- the **fine-tuned model + RAG** output

This makes it especially useful for a PBL demonstration, viva, and GitHub portfolio project.

---

## What this project does

OmniSupport AI helps a user ask a support-related question and receive a response that is:

- concise
- structured
- practical
- customer-support oriented
- easier to read on the frontend

The model is trained to behave like a support assistant, not a generic chatbot.

---

## Project workflow at a glance

```text
User Query
   ↓
Frontend (Wix / Web UI)
   ↓
Backend API (FastAPI)
   ↓
RAG Retrieval (FAISS + embeddings)
   ↓
Base Model and Fine-tuned Model
   ↓
Structured response for frontend
Features
Fine-tuning an open-source LLM using customer support data
RAG-based context injection for more grounded answers
Side-by-side comparison of base vs trained model
Support-friendly output formatting
FastAPI backend for chat inference
Hugging Face model upload support
Hugging Face Space deployment support
Google Colab training workflow
Google Drive checkpoint saving
Easy integration with a frontend such as Wix
What the model is

This project uses an open-source LLM as the base model and fine-tunes it for customer support behavior.

The model is trained to:

answer support queries in a calm and professional tone
keep replies short and clear
provide step-by-step troubleshooting
avoid long, vague paragraphs
work better when given retrieved context from RAG

The model is not trained from scratch. Instead, the project uses:

a pretrained base model
LoRA fine-tuning
optional RAG context
structured output formatting in the backend
Repository structure

A recommended structure is:

OmniSupport-AI/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
│
├── training/
│   ├── OmniSupport_Llama3_1_RAG_SideBySide.ipynb
│   ├── customer_support_tickets.csv
│   └── cleaned_dataset.jsonl
│
├── backend/
│   └── app.py
│
├── docs/
│   ├── Project_Workflow.md
│   ├── Architecture.md
│   └── API.md
│
├── assets/
│   ├── architecture.png
│   └── demo.png
│
└── scripts/
    ├── prepare_dataset.py
    └── build_rag_index.py
Files you must upload to GitHub

These are the important project files that should be in the repository:

1. Colab notebook

training/OmniSupport_Llama3_1_RAG_SideBySide.ipynb

This notebook contains:

model loading
token login
dataset loading
LoRA fine-tuning
RAG setup
base vs trained comparison
2. Backend API

backend/app.py

This is the FastAPI application used to:

load the model
build prompts
generate responses
return structured output for the frontend
3. Requirements

requirements.txt

This file is needed so anyone can install the dependencies.

4. Dataset

training/customer_support_tickets.csv

This is the raw training dataset.

If you already cleaned it, also include:

training/cleaned_dataset.jsonl

5. Documentation files
docs/Project_Workflow.md
docs/Architecture.md
docs/API.md
6. Environment example

.env.example

This shows users what environment variables they need without exposing secrets.

7. Git ignore file

.gitignore

This prevents secrets, caches, checkpoints, and large files from being committed.

Files you should NOT upload

Do not upload:

your real Hugging Face token
Google Drive checkpoints
.cache folders
__pycache__
temporary Colab runtime files
large base model files
any private credentials
Prerequisites

To run the project, you need:

Python 3.10 or newer
Google Colab account
Hugging Face account
Hugging Face token
Access to the gated Llama model, if you are using Meta Llama
A GPU runtime in Colab
Git installed locally if you want to clone the repo
Step 1 — Clone the repository
git clone https://github.com/<your-username>/OmniSupport-AI.git
cd OmniSupport-AI
Step 2 — Create a Hugging Face account

You need a Hugging Face account to:

download the model
authenticate in Colab
upload your trained adapter
create a Hugging Face Space

Go to the Hugging Face website and sign in.

Step 3 — Create your Hugging Face token

Create a new token in your account settings.

Recommended permissions:

for downloading a gated model: read
for uploading a model or Space files: write

If you are only using the token to load a gated model in Colab, read is usually enough.

If you plan to push to Hugging Face Hub, use write.

Step 4 — Open the Colab notebook

Open:

training/OmniSupport_Llama3_1_RAG_SideBySide.ipynb

This notebook should be run in Google Colab with GPU enabled.

Step 5 — Enable GPU in Colab

In Colab, go to:

Runtime → Change runtime type → GPU

Use a GPU runtime, not TPU, because this notebook is built for standard PyTorch + BitsAndBytes + LoRA training.

Step 6 — Install dependencies

Inside the notebook, the first setup cell should install dependencies such as:

transformers
datasets
accelerate
bitsandbytes
peft
trl
sentence-transformers
faiss-cpu
huggingface_hub

If you want to install them locally, use:

pip install -r requirements.txt
Step 7 — Log in to Hugging Face inside Colab

The notebook should include a cell like:

from huggingface_hub import login
login()

When prompted, paste your Hugging Face token.

This token is used only inside the notebook session.

Step 8 — Accept access to the gated model

If you are using Meta Llama, you must request and accept access on the Hugging Face model page.

Once access is approved:

the notebook can download the base model
fine-tuning can begin

If access is pending, the model download will fail.

Step 9 — Run the notebook from top to bottom

The notebook should do the following in order:

install dependencies
log in to Hugging Face
load the tokenizer and base model
load the dataset
clean and format the dataset
prepare LoRA fine-tuning
train the model
build the RAG index
compare base vs trained output
save checkpoints to Google Drive

Run the notebook cell by cell, or use “Run all” if it is already prepared correctly.

Step 10 — Save model checkpoints to Google Drive

Because Colab sessions are temporary, you must save checkpoints to Google Drive.

Typical outputs to save:

LoRA adapter weights
tokenizer files
training checkpoints
any cleaned dataset artifacts

If your notebook contains something like:

from google.colab import drive
drive.mount('/content/drive')

then mount your Drive and save the output folder there.

Example:

model.save_pretrained("/content/drive/MyDrive/omnisupport-model")
tokenizer.save_pretrained("/content/drive/MyDrive/omnisupport-model")
Step 11 — Upload the trained adapter to Hugging Face

Once training is done, you can upload the adapter instead of the whole model.

This is the recommended way if you fine-tuned with LoRA.

Typical files to upload to a Hugging Face model repo:

adapter_model.safetensors
adapter_config.json
tokenizer files
README.md for the model card

Do not upload the full base model unless you have a very specific reason to do so.

Step 12 — Create a Hugging Face model repo

You can create a model repository on Hugging Face and upload the adapter there.

The repository will then store:

your fine-tuned adapter
model card
metadata
usage instructions

This makes it easy for others to download your adapter and use it with the base model.

Step 13 — Create a Hugging Face Space

If you want a live demo, create a Hugging Face Space.

Recommended choices:

Gradio Space if you want a simple frontend demo
Docker Space if you want to host the FastAPI backend

Hugging Face Spaces are Git-based repositories and support:

static apps
Gradio
Streamlit
Docker-based deployments
Step 14 — Add secrets to Hugging Face Space

If your Space needs private values, such as:

Hugging Face token
model repo ID
adapter repo ID
backend API URL

then add them as Space secrets instead of hardcoding them.

This is the recommended secure deployment method.

Step 15 — Run the FastAPI backend

The backend file is:

backend/app.py

It should expose endpoints like:

GET /
GET /health
POST /chat

The /chat endpoint should:

receive the user question
apply the system prompt
optionally use history/context
call the model
return a structured response
Step 16 — Connect the frontend

The frontend should call the backend /chat endpoint.

The backend response should provide:

acknowledgement
steps
follow-up question

This makes it easy to render a nice UI on Wix or another frontend.

Recommended frontend display:

a header with the acknowledgement
a bullet list for troubleshooting steps
a small follow-up line if needed
Step 17 — Use the structured response object

The backend should not rely only on plain text.

It should return a structured response such as:

{
  "answer": "...",
  "presentation": {
    "acknowledgement": "...",
    "steps": ["...", "...", "..."],
    "follow_up": "..."
  },
  "model_id": "...",
  "adapter_id": "..."
}

This is what lets the frontend look professional and readable.

Step 18 — Compare base vs trained output

The notebook includes a comparison mode that shows:

base model response
fine-tuned model response
optional RAG context

This is one of the strongest parts of the project because it visibly demonstrates the value of training.

How to run the project locally
Option A — Notebook only

Use the Colab notebook to train and test everything.

Option B — Backend only

Run the FastAPI backend after the model or adapter is available.

Option C — Full project

Use:

Colab notebook for training
Hugging Face for model hosting
FastAPI backend for inference
Wix or another frontend for the UI
Environment variables

Create a .env file locally using .env.example as a guide.

Example variables:

MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
ADAPTER_ID=your-username/omnisupport-adapter
HF_TOKEN=your_token_here
MAX_NEW_TOKENS=120
TEMPERATURE=0.4
TOP_P=0.9

Never commit your real token.

Dataset format

The notebook should convert support tickets into instruction-following examples.

A good training sample looks like:

<|begin_of_text|>
<|user|>
My laptop touchpad is not working.
<|assistant|>
Try these steps:
1. Check touchpad settings.
2. Restart the laptop.
3. Update the touchpad driver.
<|end_of_text|>

This format teaches the model to answer in a support style.

RAG workflow

RAG helps the model retrieve relevant context before answering.

Typical RAG pipeline:

embed support texts
store them in FAISS
retrieve the closest matches
insert the retrieved context into the prompt
generate the final answer

This improves accuracy and relevance.

Deliverables

If everything is set up correctly, you should have these deliverables:

training notebook
backend API
cleaned dataset
trained adapter or model repo
RAG index generation workflow
GitHub repository
optional Hugging Face Space demo
Common issues
Access denied to the model

You need to:

request access
wait for approval
use the correct Hugging Face token
Colab session resets

Save outputs to Google Drive.

Output is too long

Reduce:

max_new_tokens
temperature
prompt verbosity
Frontend still shows long paragraphs

Make the backend return structured output and have the frontend render each field separately.

Recommended deployment order
train in Colab
save adapter to Google Drive
upload adapter to Hugging Face
deploy backend on Hugging Face Space or another server
connect frontend to backend
test with sample support queries
What this project demonstrates

OmniSupport AI demonstrates:

open-source model fine-tuning
efficient parameter training
retrieval augmentation
backend API design
frontend presentation logic
deployment workflow using modern AI tooling
Future improvements

Possible upgrades include:

voice input/output
multilingual support
better RAG corpus
automatic escalation to human support
logging and analytics
response evaluation dashboard
better model comparison interface
License

Choose a license such as MIT or Apache 2.0.

Acknowledgements

This project uses tools from the Hugging Face ecosystem, including:

transformers
datasets
peft
trl
huggingface_hub
Spaces
Final note

To run the full project, the most important files are:

training/OmniSupport_Llama3_1_RAG_SideBySide.ipynb
backend/app.py
requirements.txt
training/customer_support_tickets.csv
.env.example
.gitignore
README.md

Everything else is supporting documentation or optional deployment material.


If you want, I can now turn this into a **GitHub-ready README file** with badges, a table of contents, and a nicer a
