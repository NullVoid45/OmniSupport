import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# Configuration
# -----------------------------
MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_ID = os.getenv("ADAPTER_ID", "Time-space-paradox/omnisupport-llama")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
You are OmniSupport AI, an expert technical support assistant.
Provide precise, structured, and highly concise troubleshooting steps.

Rules:
- Acknowledge the user's issue briefly.
- Provide exactly 2-3 clear, actionable, technical troubleshooting steps.
- Number each step clearly.
- Keep each step concise and to the point. Avoid generic advice.
- Ask one short, specific follow-up question to narrow down the problem if needed.
- Do not use filler words, long paragraphs, or greetings.
- Do not mention you are an AI.
""".strip(),
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "4"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))

# -----------------------------
# Globals
# -----------------------------
model = None
tokenizer = None
model_lock = threading.Lock()


# -----------------------------
# Schemas
# -----------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="One of: system, user, assistant")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat history in chronological order",
    )
    customer_name: Optional[str] = Field(default=None, description="Optional customer name")
    context: Optional[str] = Field(default=None, description="Optional retrieved support context")


class SupportPresentation(BaseModel):
    acknowledgement: str
    steps: List[str]
    follow_up: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    presentation: SupportPresentation
    model_id: str
    adapter_id: Optional[str] = None


# -----------------------------
# Model loading
# -----------------------------
def _device_map() -> Optional[str]:
    return "auto" if torch.cuda.is_available() else None


def _torch_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _is_placeholder_name(name: Optional[str]) -> bool:
    if not name:
        return True
    cleaned = name.strip().lower()
    return cleaned in {
        "customer name",
        "customer",
        "name",
        "user",
        "guest",
        "unknown",
        "none",
        "null",
    }


def load_model() -> None:
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return

    with model_lock:
        if model is not None and tokenizer is not None:
            return

        tokenizer_local = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

        if tokenizer_local.pad_token is None:
            tokenizer_local.pad_token = tokenizer_local.eos_token
        tokenizer_local.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=_torch_dtype(),
            device_map=_device_map(),
            low_cpu_mem_usage=True,
        )

        if ADAPTER_ID:
            try:
                base_model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
            except Exception as exc:
                print(f"[WARN] Adapter load failed, continuing with base model: {exc}")

        base_model.eval()
        model = base_model
        tokenizer = tokenizer_local


# -----------------------------
# Prompt building
# -----------------------------
def build_messages(payload: ChatRequest) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if payload.customer_name and not _is_placeholder_name(payload.customer_name):
        messages.append(
            {
                "role": "system",
                "content": f"Customer name: {payload.customer_name}. Use it naturally if helpful.",
            }
        )

    if payload.context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Retrieved support context. Use it only if relevant and do not copy it verbatim:\n"
                    f"{payload.context}"
                ),
            }
        )

    if payload.history:
        trimmed_history = payload.history[-HISTORY_LIMIT:]
        for msg in trimmed_history:
            if msg.role not in {"system", "user", "assistant"}:
                continue
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": payload.message})
    return messages


def build_prompt(messages: List[Dict[str, str]]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


# -----------------------------
# Text cleanup / formatting
# -----------------------------
def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_prompt_echo(text: str) -> str:
    text = text.strip()

    # Common wrapper labels the model may echo.
    text = re.sub(r"^(assistant|response|answer)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^here (is|are).{0,40}:\s*", "", text, flags=re.I)

    # Remove common verbose greetings at the start.
    text = re.sub(r"^dear customer name[,:]?\s*", "", text, flags=re.I)
    text = re.sub(r"^dear customer[,:]?\s*", "", text, flags=re.I)
    text = re.sub(r"^hello[!,:]?\s*", "", text, flags=re.I)
    text = re.sub(r"^hi there[!,:]?\s*", "", text, flags=re.I)
    text = re.sub(r"^thank you for reaching out.*?(?:\.|!|\n)\s*", "", text, flags=re.I | re.DOTALL)

    return text.strip()


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_presentation(answer: str) -> SupportPresentation:
    raw = _strip_prompt_echo(answer)
    
    # Extract acknowledgement: Everything before the first number
    ack_match = re.search(r"^(.*?)(?:(?:^|\s)\d+[\.\)])", raw, flags=re.DOTALL)
    acknowledgement = ""
    if ack_match:
        acknowledgement = ack_match.group(1).strip()
        
    # Extract steps using regex to find "1. text 2. text"
    step_matches = list(re.finditer(r"(?:^|\s)\d+[\.\)]\s*(.*?)(?=(?:\s\d+[\.\)]|$))", raw, flags=re.DOTALL))
    
    steps = []
    for m in step_matches:
        step_text = _normalize_text(m.group(1).strip())
        if step_text:
            steps.append(_sentence_case(step_text))
            
    follow_up = None
    
    if not steps:
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        sentences = [s.strip() for s in sentences if s.strip()]
        for s in sentences:
            s = re.sub(r"^(?:[-•*]|\d+[\.\)])\s*", "", s).strip()
            if not s:
                continue
            if "?" in s and not follow_up:
                follow_up = _sentence_case(s)
            elif s.lower() not in {st.lower() for st in steps}:
                steps.append(_sentence_case(s))
            if len(steps) >= 3 and follow_up:
                break
    else:
        last_step = steps[-1]
        if "?" in last_step:
            parts = re.split(r"(?<=[.!?])\s+", last_step)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1 and "?" in parts[-1]:
                follow_up = parts.pop()
                steps[-1] = " ".join(parts).strip()
            else:
                follow_up = steps.pop()
                
    if not acknowledgement or len(acknowledgement.split()) > 25:
        acknowledgement = "I understand the issue."
        
    if not steps:
        steps = ["Please share more details about the issue."]

    return SupportPresentation(
        acknowledgement=acknowledgement,
        steps=steps[:3],
        follow_up=follow_up
    )


def format_for_frontend(presentation: SupportPresentation) -> str:
    """
    Structured string for frontends that display only one text field.
    Using double newlines to ensure proper paragraph breaks in Markdown and most text renderers.
    """
    lines = []
    ack = presentation.acknowledgement.strip()
    if ack and ack.lower() != "i understand the issue.":
        lines.append(ack)

    for i, step in enumerate(presentation.steps, 1):
        lines.append(f"{i}. {step.strip()}")

    if presentation.follow_up:
        lines.append(f"Follow-up: {presentation.follow_up.strip()}")

    return "\n\n".join(lines).strip()


# -----------------------------
# Generation
# -----------------------------
def generate_answer(payload: ChatRequest) -> ChatResponse:
    if model is None or tokenizer is None:
        load_model()

    assert model is not None and tokenizer is not None

    messages = build_messages(payload)
    prompt = build_prompt(messages)

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        try:
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if not raw_answer:
        presentation = SupportPresentation(
            acknowledgement="I understand the issue.",
            steps=["Please share a little more detail so I can help you better."],
            follow_up=None,
        )
    else:
        presentation = build_presentation(raw_answer)

    answer_text = format_for_frontend(presentation)

    return ChatResponse(
        answer=answer_text,
        presentation=presentation,
        model_id=MODEL_ID,
        adapter_id=ADAPTER_ID or None,
    )


# -----------------------------
# API
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="OmniSupport API", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "OmniSupport API is running.",
        "model_id": MODEL_ID,
        "adapter_id": ADAPTER_ID or "not set",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return generate_answer(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)