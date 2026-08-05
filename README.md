🎧 OmniSupport AI
OmniSupport AI is an intelligent customer support platform that leverages a fine-tuned open-source Large Language Model (LLM), Retrieval-Augmented Generation (RAG), and a sleek web frontend to deliver concise, structured, and highly effective support replies.

Designed specifically to demonstrate the qualitative leap between a base model and a fine-tuned model + RAG, this project is ideal for PBL demonstrations, academic vivas, and strengthening a professional GitHub portfolio.

🎯 What This Project Does
OmniSupport AI allows users to ask support-related questions and receive responses that are:

Concise & Practical: Focuses on actionable troubleshooting steps.

Structured: Formatted specifically for easy frontend rendering.

Customer-Support Oriented: Maintains a calm, professional, and helpful tone.

Context-Aware: Avoids vague paragraphs by grounding answers in retrieved RAG context.

🚀 Features
Custom Fine-Tuning: Uses LoRA to fine-tune an open-source LLM specifically on customer support data.

RAG Context Injection: Embeds support texts using FAISS to ground the model's answers in reality.

A/B Comparison: Side-by-side evaluation of the base model versus the trained model.

API-Driven Architecture: A robust FastAPI backend handles chat inference and structured output.

Cloud & Local Deployment: Full support for Google Colab training, Google Drive checkpointing, and Hugging Face Space deployment.

Frontend Ready: Outputs clean JSON structures perfect for seamless integration with Wix or custom Web UIs.

🏗️ Architecture & Workflow
Plaintext
User Query ➔ Frontend (Wix / Web UI) ➔ Backend API (FastAPI) ➔ RAG Retrieval (FAISS) ➔ Fine-tuned LLM ➔ Structured Response
Repository Structure
Plaintext
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
🛠️ Getting Started
Prerequisites
Python 3.10+

Google Colab Account (with a GPU runtime enabled)

Hugging Face Account & Token (with read access for gated models, write for uploading adapters)

Access to the gated Llama model (if using Meta Llama)

Git installed locally

1. Installation
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/NullVoid45/OmniSupport-AI.git
cd OmniSupport-AI
pip install -r requirements.txt
2. Model Training (Google Colab)
Open training/OmniSupport_Llama3_1_RAG_SideBySide.ipynb in Google Colab.

Ensure your runtime is set to GPU (Runtime → Change runtime type → GPU).

Log in to Hugging Face when prompted by the notebook using your HF Token.

Run the notebook from top to bottom. The workflow will:

Load the tokenizer and base model.

Clean and format the dataset.

Execute LoRA fine-tuning.

Build the RAG index.

Run a base vs. trained output comparison.

Save your progress: Mount your Google Drive and save the LoRA adapter weights, tokenizer files, and checkpoints to prevent data loss when the Colab session ends.

3. Deployment & Hosting
Once your adapter is trained, upload the adapter_model.safetensors, adapter_config.json, and tokenizer files to a new Hugging Face model repository.

To run the backend live:

Create a Hugging Face Space (Docker-based is recommended for the FastAPI backend).

Add your private values (HF token, model repo ID, etc.) as Space Secrets rather than hardcoding them.

Start the FastAPI server (backend/app.py), which will expose the /chat, /health, and / endpoints.

⚙️ Configuration (.env)
Create a local .env file using .env.example as a guide. Never commit your real tokens to version control.

Code snippet
MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
ADAPTER_ID=NullVoid45/omnisupport-adapter
HF_TOKEN=your_huggingface_token_here
MAX_NEW_TOKENS=120
TEMPERATURE=0.4
TOP_P=0.9
📝 Data Formats
Structured API Response
The backend is designed to return a structured JSON response instead of a plain text block, allowing the frontend to render an aesthetic UI (e.g., a header, a bulleted list of steps, and a follow-up line).

JSON
{
  "answer": "Full text response...",
  "presentation": {
    "acknowledgement": "I understand your touchpad is not working.",
    "steps": [
      "Check touchpad settings.",
      "Restart the laptop.",
      "Update the touchpad driver."
    ],
    "follow_up": "Did this resolve your issue?"
  },
  "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
  "adapter_id": "NullVoid45/omnisupport-adapter"
}
Training Dataset Example
The notebook converts raw support tickets into instruction-following examples to teach the model its support-style behavior:

Plaintext
<|begin_of_text|>
<|user|>
My laptop touchpad is not working.
<|assistant|>
Try these steps:
1. Check touchpad settings.
2. Restart the laptop.
3. Update the touchpad driver.
<|end_of_text|>
⚠️ Important Notes & Best Practices
Repository Hygiene: Do NOT upload your real HF token, Google Drive checkpoints, .cache folders, __pycache__, or the massive base model files to GitHub. Ensure your .gitignore is properly configured.

Gated Models: If using Meta Llama, ensure you have requested and been granted access on Hugging Face before starting your Colab training run, or the download will fail.

Output Length: If the generated text is too long, reduce max_new_tokens, lower the temperature, or refine the system prompt verbosity.

🔮 Future Improvements
Integration of voice input/output capabilities.

Multilingual support for global customer service.

Expanded and dynamic RAG corpus.

Automated escalation logic for human support routing.

A dedicated response evaluation dashboard with logging and analytics.

Built with tools from the Hugging Face ecosystem (transformers, datasets, peft, trl, huggingface_hub).
