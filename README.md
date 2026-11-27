# LLM Cost Lab

## Overview

**LLM Cost Lab** provides a collection of small, runnable examples that demonstrate how to reduce the cost of Large Language Model (LLM) usage without sacrificing answer quality. All examples revolve around a fictional e‑commerce store, *Blipkart*, and a customer‑support agent built on top of various LLM providers.

The repository contains:
- Sample scripts showcasing techniques such as prompt caching, semantic caching, gateway routing, bounded agents, and batch API calls.
- A `lab.py` script that runs the same support agent in multiple configurations on a set of test cases, reporting both cost and success metrics.
- Utility data files and a shared system prompt.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llm-cost-lab.git
cd llm-cost-lab

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables (copy the example and add your keys)
cp .env.example .env
# Edit .env and add the API keys you need (Groq, OpenAI, OpenRouter, etc.)
```

> **Note:** Most examples work with a free Groq key. Specific scripts list any additional keys required.

## Usage Examples

All sample scripts live in the `samples/` directory. Below are common entry points:

```bash
cd samples

# 1️⃣ Prompt caching demo (OpenAI by default)
python 01_prompt_caching.py
# Use a different provider
python 01_prompt_caching.py groq
python 01_prompt_caching.py claude

# 2️⃣ Semantic cache demo (requires OpenRouter key)
python 02_semantic_cache.py

# 3️⃣ Gateway router demo (Groq, OpenRouter, Langfuse optional)
python 03_gateway_router.py

# 4️⃣ Bounded agent demo (state is persisted across runs)
python 04_bounded_agent.py   # first run
python 04_bounded_agent.py   # second run resumes the chat

# 5️⃣ Batch API demo (run live calls or submit as a batch)
python 05_batch_api.py live          # 30 normal calls
python 05_batch_api.py submit        # same 30 calls as a single batch
python 05_batch_api.py check         # later: show tags and total cost
```

### Lab Execution

The `lab.py` script runs the support agent in five different configurations over the 12 test cases in `data/cases.jsonl` and produces a cost / success report.

```bash
python lab.py v0       # baseline configuration
python lab.py stable   # prompt line moved to end of user message
python lab.py routed   # keyword router sends easy messages to a smaller model
python lab.py trimmed  # trimmed payload & reduced reasoning effort
python lab.py small    # everything on the smaller model

# Run a repeat for averaging
python lab.py v0 2

# Generate a consolidated report from all results
python lab.py report
```

## Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the repository.
2. **Create a branch** for your change, e.g. `git checkout -b feature/my-feature`.
3. **Make your changes** and ensure they pass any existing tests (if applicable).
4. **Run the samples** to verify that nothing broke.
5. **Commit** with a clear message and **push** to your fork.
6. Open a **Pull Request** against the `master` branch of this repository.

Please adhere to the existing code style (PEP 8) and add or update documentation as needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

*Happy hacking and may your LLM bills stay low!*
