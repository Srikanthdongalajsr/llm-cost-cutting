import json
import sqlite3
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lab import ORDERS, POLICY, TODAY, eligibility

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(HERE.parent / ".env")


lookups = {"count": 0}


@tool
def get_order(order_id: str) -> str:
    """Look up an order by its ID."""
    lookups["count"] += 1
    o = ORDERS.get(order_id.strip().lstrip("#"))
    return json.dumps({k: o[k] for k in ("item", "amount", "status", "delivered_on", "refunded")}) if o else "No such order."


@tool
def check_policy(order_id: str, amount: float) -> str:
    """Check whether a refund of this amount is allowed."""
    o = ORDERS.get(order_id.strip().lstrip("#"))
    return "%s: %s" % eligibility(o, amount) if o else "No such order."


small = ChatGroq(model="openai/gpt-oss-20b", reasoning_effort="low")
agent = create_agent(
    ChatGroq(model="openai/gpt-oss-120b"), [get_order, check_policy],
    system_prompt=f"You are Blipkart support. Today is {TODAY}. Policy: {POLICY} Always call check_policy before answering a refund. You can check orders and the policy but cannot issue refunds: "
                  "say whether it is eligible and that a colleague will process it. Reply in 1-3 sentences.",
    middleware=[
        ModelCallLimitMiddleware(run_limit=6, thread_limit=40, exit_behavior="end"),   # hard cap on model calls
        ToolCallLimitMiddleware(run_limit=4),                                          # all tools, per message
        ToolCallLimitMiddleware(tool_name="get_order", run_limit=2),                  # the expensive API, per message
        SummarizationMiddleware(model=small, trigger=("messages", 12), keep=("messages", 6)),   # old turns -> summary
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=3000, keep=2)]),     # stale tool results -> "[cleared]"
    ],
    checkpointer=SqliteSaver(sqlite3.connect(HERE / "threads.sqlite", check_same_thread=False)),   # Postgres/Redis savers in prod
)

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "customer-C100-chat-1"}}   # same id = same saved conversation
    for turn in ["Hi, I bought running shoes, order 4519.",
                 "They don't fit. Can I get a refund of ₹2,499?",
                 "And what about order 4555, can I refund that too?"]:
        out = agent.invoke({"messages": [{"role": "user", "content": turn}]}, config=thread)
        print(f"\nYOU: {turn}\nBOT: {out['messages'][-1].content}\n     messages in state: {len(out['messages'])}  get_order calls so far: {lookups['count']}")
    # Later, in a new process (or after a crash): pass only the new message; the checkpointer brings back the history,
    # so nothing is re-looked-up or re-generated. List the saved steps:
    steps = list(agent.get_state_history(thread))
    print(f"\n{len(steps)} checkpoints saved for this thread; resume from any of them with its checkpoint_id.")
