# A plain semantic cache, with no guards: reuse the old answer when two questions look alike.
# It prints how similar each pair is, and whether reusing the answer would be right.
# samples/02_semantic_cache.py is the same idea with the guards that fix it.

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.85  # a common default

# (new question, question already in the cache, is reusing its answer right?)
pairs = [
    ("What is your return policy for shoes?", "How do I return shoes I bought?", True),
    ("How long does a refund take to show up?", "When will I get my refund money back?", True),
    ("Do you accept cash on delivery?", "Can I pay cash when the order arrives?", True),
    ("Where is my order 4555?", "Where is my order 4556?", False),
    ("Cancel my order", "Cancel my refund", False),
    ("Refund order 4519, the shoes don't fit.", "Refund order 4519, the shoes are great.", False),
]

for new, cached, should_reuse in pairs:
    a, b = model.encode([new, cached], normalize_embeddings=True)
    score = float(a @ b)
    hit = score >= THRESHOLD
    if hit == should_reuse:
        verdict = "ok"
    elif hit:
        verdict = "WRONG ANSWER"
    else:
        verdict = "missed"
    print(f"{score:.2f}  {'HIT ' if hit else 'miss'}  {verdict:12}  {new}  |  {cached}")
