# LLM cost lab

[Contributing Guidelines](CONTRIBUTING.md)

Small, runnable examples of cutting an LLM bill without breaking the answers.
Everything uses one made-up store, Blipkart, and its customer-support agent.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add the keys for the files you want to run
```

Most files only need a free [Groq](https://console.groq.com) key. Each file below says what it needs.

## The examples (`samples/`)

| File | What it shows | Keys |
|---|---|---|
| `01_prompt_caching.py` | The same request with a changing line at the top vs at the end of the prompt. Only the second one gets cached. | OpenAI (default), Groq or Anthropic |
| `02_semantic_cache.py` | Reusing answers for questions that mean the same, with guards: never cache personal or money questions, exact match first, same prompt version, numbers must match. Uses the Jev decision model when keyword rules can't tell. | OpenRouter |
| `03_gateway_router.py` | One gateway (LiteLLM) for every call. A decision model sends each message to: refuse, plain code, the small model or the big model, and checks the cheap answers. Every call is traced in Langfuse with its cost. | Groq, OpenRouter, Langfuse |
| `04_bounded_agent.py` | An agent with limits: max model calls, max tool calls, old turns summarised, old tool results cleared. State is saved after every step, so a chat can resume. | Groq |
| `05_batch_api.py` | The same job (tag 30 tickets) as 30 live calls, and as one batch: 50% off, done within 24 hours, for work nobody is waiting on. | OpenAI |

```bash
cd samples
python 01_prompt_caching.py          # OpenAI
python 01_prompt_caching.py groq
python 01_prompt_caching.py claude
python 02_semantic_cache.py
python 03_gateway_router.py
python 04_bounded_agent.py           # run it twice: the second run picks up the saved chat
python 05_batch_api.py live          # 30 normal calls
python 05_batch_api.py submit        # the same 30 as one batch; saves the batch id to batch_id.txt
python 05_batch_api.py check         # any time later, even after a restart: prints the tags and the cost
```

`support_prompt.md` is the system prompt the samples share (about 1,250 tokens, enough for providers to cache it).

## The lab (`lab.py`)

The same support agent run five ways on the 12 test cases in `data/cases.jsonl`, to measure what each change does to the bill **and** to the pass rate. Needs Groq.

| Run | What changes |
|---|---|
| `v0` | gpt-oss-120b, request ID and time at the **top** of the system prompt |
| `stable` | that line moved to the **end** of the user message, so the start of the prompt never changes |
| `routed` | stable, and a keyword router sends easy messages to gpt-oss-20b |
| `trimmed` | routed, with low reasoning effort and fewer order fields sent to the model |
| `small` | stable, gpt-oss-20b for everything |

```bash
python lab.py v0              # also: stable, routed, trimmed, small
python lab.py v0 2            # a second repeat; the report averages repeats
python lab.py report          # one table from everything in results/
```
Cheapest per chat isn't cheapest per success: `trimmed` cut thinking tokens by 80% and broke refunds. 12 cases is a lab, not a verdict, so re-run it on your own agent.

Prices used: gpt-oss-120b $0.15 / $0.60 and gpt-oss-20b $0.075 / $0.30 per million input / output tokens on Groq, cached input 50% off, ₹94.09 per $ (13 Nov 2026).

## One more

```bash
python semantic_cache.py      # a cache with no guards: hits the wrong order, misses real paraphrases (no key)
```

Runs write their numbers to `results/`.
