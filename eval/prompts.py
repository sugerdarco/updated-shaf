"""Shared prompt formatting + per-type generation budgets.

Kept identical between the single-model baseline and the ensemble so the two are
compared on the same inputs. Returns (prompt_body, max_new) where max_new is a
token budget for single models and a byte budget for the ensemble (bytes ~ tokens
for these short answers; number answers get the largest budget for CoT).
"""

# max_new by answer type (tokens for single model, bytes for ensemble)
BUDGET = {"mc": 24, "number": 320, "openqa": 64}


def build_prompt(item):
    t = item["type"]
    if t == "mc":
        opts = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(item["choices"]))
        body = (
            f"{item['question']}\n{opts}\n"
            "Respond with only the letter of the correct option, e.g. 'Answer: B'."
        )
    elif t == "number":
        body = (
            f"{item['question']}\n"
            "Solve step by step, then give the final numeric answer on the last "
            "line as '#### <number>'."
        )
    else:  # openqa
        body = f"Answer the question in a few words.\nQuestion: {item['question']}\nAnswer:"
    return body, BUDGET[t]
