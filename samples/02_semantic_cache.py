import os
import re
import time

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # OPENROUTER_API_KEY for the decision model
model = SentenceTransformer("all-MiniLM-L6-v2")

NEVER_CACHE = {"order_status", "refund", "cancel"}  # the answer depends on the customer or moves money
THRESHOLD = {"faq": 0.70}                          # tuned on real FAQ pairs; any other intent needs 0.90
TTL = 24 * 60 * 60                                 # answers expire after a day, so policy changes show up


# The decision model (Jev on OpenRouter): yes/no questions in, a probability for each out
intent_questions = {
    "personal": {
        "type": "noul",
        "instructions": "`message` is about this customer's own order, parcel or account. A general policy question is not this.",
    },
    "money": {
        "type": "noul",
        "instructions": "`message` asks support to refund, cancel or pay back money for this customer. A general question about how refunds work is not this.",
    },
}


def ask_jev(question):
    response = requests.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": "typesafe/jev-1.13", "state": {"message": question}, "questions": intent_questions},
        timeout=30,
    )
    answers = response.json()["answers"]
    return {name: answer["noul"] for name, answer in answers.items()}


def get_intent(question):
    # cheap rules first: they catch the obvious cases for free
    q = question.lower()
    if "refund" in q and re.search(r"\d", q):
        return "refund"
    if re.search(r"\border\b.*\d|where is", q):
        return "order_status"
    if q.startswith("cancel"):
        return "cancel"

    # everything else: ask the decision model
    scores = ask_jev(question)
    if scores["money"] > 0.5:
        return "refund"
    if scores["personal"] > 0.5:
        return "order_status"
    return "faq"


def normalise(question):
    question = re.sub(r"[^\w\s₹]", "", question.lower())
    return " ".join(question.split())


def numbers_in(question):
    return set(re.findall(r"\d+(?:\.\d+)?", question))


class SemanticCache:
    def __init__(self):
        self.exact = {}
        self.entries = []

    def put(self, question, answer, scope):
        if get_intent(question) in NEVER_CACHE:
            return
        self.exact[(scope, normalise(question))] = (answer, time.time())
        self.entries.append({
            "scope": scope,
            "vector": model.encode(question, normalize_embeddings=True),
            "numbers": numbers_in(question),
            "answer": answer,
            "time": time.time(),
        })

    def get(self, question, scope):
        # 1. never cache personal or money questions
        intent = get_intent(question)
        if intent in NEVER_CACHE:
            return None, f"skip: {intent} is never cached"

        # 2. exact match first, it's free
        hit = self.exact.get((scope, normalise(question)))
        if hit and time.time() - hit[1] < TTL:
            return hit[0], "HIT exact"

        # 3. closest question in the same scope that hasn't expired
        vector = model.encode(question, normalize_embeddings=True)
        best, best_score = None, 0.0
        for entry in self.entries:
            if entry["scope"] != scope or time.time() - entry["time"] > TTL:
                continue
            score = float(vector @ entry["vector"])
            if score > best_score:
                best, best_score = entry, score

        # 4. similar enough for this intent?
        if best is None or best_score < THRESHOLD.get(intent, 0.90):
            return None, f"miss ({best_score:.2f})"

        # 5. "5 days" and "2 days" look almost the same to an embedding: numbers must match exactly
        if numbers_in(question) != best["numbers"]:
            return None, f"blocked ({best_score:.2f}): numbers differ"

        return best["answer"], f"HIT semantic ({best_score:.2f})"


cache = SemanticCache()
scope = ("blipkart", "prompt-v7", "en-IN")  # store, prompt version, language

cache.put("How do I return shoes I bought?", "Request a refund within 30 days of delivery.", scope)
cache.put("How many days do refunds take?", "Refunds take 5-7 working days to show.", scope)
cache.put("Can I pay cash when the order arrives?", "Yes, cash on delivery is accepted.", scope)
cache.put("Do you deliver in 2 days?", "Express delivery takes 1-2 days.", scope)

tests = [
    "what is your return policy for shoes",      # same question, other words -> hit
    "How long does a refund take to show up?",   # same question, other words -> hit
    "Do you accept cash on delivery?",           # same question, other words -> hit
    "Can I pay cash when the order arrives?",    # word for word -> exact hit
    "Do you deliver in 5 days?",                 # different number -> blocked
    "Where is my order 4555?",                   # personal -> never cached
    "Refund order 4519, the shoes don't fit.",   # money -> never cached
    "My parcel still hasn't come, can you check?",   # no order number, rules miss it -> Jev: personal
    "I want my money back for the headphones",       # no "refund" word -> Jev: money
]
for question in tests:
    answer, why = cache.get(question, scope)
    print(f"{why:35} | {question}")

# a new prompt version is a new scope: old answers are not reused
answer, why = cache.get("Do you accept cash on delivery?", ("blipkart", "prompt-v8", "en-IN"))
print(f"{why:35} | Do you accept cash on delivery?  (prompt-v8)")
