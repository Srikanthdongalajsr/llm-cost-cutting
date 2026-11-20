import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
client = OpenAI()

# The same job two ways: tag 30 support tickets with an intent and an urgency.
#
#   python 05_batch_api.py live     30 normal calls, answers in seconds
#   python 05_batch_api.py submit   the same 30 requests as one batch: 50% off, done within 24 hours
#   python 05_batch_api.py check    is the batch done? if yes, print the tags and the cost

HERE = Path(__file__).parent
MODEL = "gpt-6-luna"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per 1M tokens, live; the batch API charges half
RESULTS = HERE.parent / "results" / "batch_vs_live.json"
BATCH_ID = HERE / "batch_id.txt"

lines = (HERE.parent / "data" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
tickets = [json.loads(line)["message"] for line in lines]
system_prompt = 'Tag a customer support message. Reply with JSON only: {"intent": "status|refund|clarify|other", "urgency": "low|medium|high"}'


def request(ticket):
    return {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket},
        ],
    }


def cost_usd(input_tokens, output_tokens, discount=1.0):
    return round((input_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1_000_000 * discount, 6)


def save(mode, result):
    saved = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    saved[mode] = result
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


def live():
    start = time.perf_counter()
    input_tokens = output_tokens = 0
    for ticket in tickets:
        response = client.chat.completions.create(**request(ticket))
        input_tokens += response.usage.prompt_tokens
        output_tokens += response.usage.completion_tokens
        print(response.choices[0].message.content, "|", ticket)
    save("live", {
        "tickets": len(tickets),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": round(time.perf_counter() - start, 1),
        "cost_usd": cost_usd(input_tokens, output_tokens),
    })


def submit():
    # 1. one line per request, each with an id so we can match the answers later
    batch_file = HERE / "batch_input.jsonl"
    with open(batch_file, "w", encoding="utf-8") as f:
        for i, ticket in enumerate(tickets):
            line = {"custom_id": f"ticket-{i}", "method": "POST", "url": "/v1/chat/completions", "body": request(ticket)}
            f.write(json.dumps(line) + "\n")

    # 2. upload the file, 3. start the batch: this is where we say "nobody is waiting"
    uploaded = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")

    BATCH_ID.write_text(batch.id)
    print(f"submitted {batch.id} ({len(tickets)} requests), status: {batch.status}")
    print("run 'python 05_batch_api.py check' in a few minutes")


def check():
    batch = client.batches.retrieve(BATCH_ID.read_text().strip())
    print(f"{batch.id}: {batch.status}, {batch.request_counts.completed}/{batch.request_counts.total} done")
    if batch.status != "completed":
        return

    # 4. download the answers; they can come back in any order, custom_id says which ticket
    input_tokens = output_tokens = 0
    for line in client.files.content(batch.output_file_id).text.splitlines():
        row = json.loads(line)
        body = row["response"]["body"]
        input_tokens += body["usage"]["prompt_tokens"]
        output_tokens += body["usage"]["completion_tokens"]
        ticket = tickets[int(row["custom_id"].split("-")[1])]
        print(body["choices"][0]["message"]["content"], "|", ticket)

    minutes = (batch.completed_at - batch.created_at) / 60
    save("batch", {
        "tickets": len(tickets),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "minutes": round(minutes, 1),
        "cost_usd": cost_usd(input_tokens, output_tokens, discount=0.5),
    })


mode = sys.argv[1] if len(sys.argv) > 1 else "live"
if mode == "submit":
    submit()
elif mode == "check":
    check()
else:
    live()
