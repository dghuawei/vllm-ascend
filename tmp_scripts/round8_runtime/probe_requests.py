import json, sys, urllib.request

# usage: probe_requests.py <out.json>  (server at localhost:1995)
OUT = sys.argv[1]
PROMPTS = [
    "The capital of France is",
    "Explain the Pythagorean theorem in one sentence:",
    "def fib(n):",
    "Water boils at",
    "The three laws of robotics were written by",
    "128 * 7 =",
    "Translate to French: good morning",
    "List three prime numbers:",
    "The mitochondria is the",
    "Once upon a time",
]

def req(model, prompt):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
    }
    r = urllib.request.Request(
        "http://localhost:1995/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=300) as resp:
        j = json.loads(resp.read())
    return j["choices"][0]["text"]

res = {}
for m in ["lora-adapter1", "ds"]:
    for i, p in enumerate(PROMPTS):
        try:
            res[f"{m}|{i}"] = req(m, p)
        except Exception as e:  # noqa: BLE001
            res[f"{m}|{i}"] = f"ERROR: {e}"
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT, "entries", len(res))
