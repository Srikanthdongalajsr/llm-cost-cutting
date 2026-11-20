import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# python 01_prompt_caching.py          -> OpenAI  (caches automatically)
# python 01_prompt_caching.py groq     -> Groq    (same API as OpenAI; its cache is best-effort)
# python 01_prompt_caching.py claude   -> Claude  (caches only what you mark with cache_control)
provider = sys.argv[1] if len(sys.argv) > 1 else "openai"

if provider == "groq":
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    model = "openai/gpt-oss-20b"
elif provider == "claude":
    client = anthropic.Anthropic()
    model = "claude-sonnet-5-5"
else:
    client = OpenAI()
    model = "gpt-6-luna"

system_prompt = (Path(__file__).parent / "support_prompt.md").read_text(encoding="utf-8")

questions = [
    "Where is my order 4519?",
    "Do you accept cash on delivery?",
    "How long does a refund take?",
    "Can I cancel a shipped order?",
    "Do you do exchanges?",
]


def ask_openai(system, user):
    options = {}
    if provider == "openai":
        # optional: sends requests with the same prompt to the same cache (Groq rejects this field)
        options["prompt_cache_key"] = "blipkart-support"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **options,
    )
    usage = response.usage
    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
    return usage.prompt_tokens, cached


def ask_claude(system, user):
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},  # cache everything up to here
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    usage = response.usage
    total = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    return total, usage.cache_read_input_tokens


def ask(question, request_info_on_top):
    request_info = f"Request {uuid.uuid4().hex[:8]} at {datetime.now():%H:%M:%S}"

    if request_info_on_top:
        system = request_info + "\n\n" + system_prompt
        user = question
    else:
        system = system_prompt
        user = question + "\n\n" + request_info

    if provider == "claude":
        return ask_claude(system, user)
    return ask_openai(system, user)


for on_top in [True, False]:
    print("\nRequest info at the TOP" if on_top else "\nRequest info at the END")
    for question in questions:
        total, cached = ask(question, on_top)
        print(f"  {cached:>5} of {total} tokens cached   {question}")
