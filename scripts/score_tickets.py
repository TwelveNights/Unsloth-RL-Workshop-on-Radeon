#!/usr/bin/env python3
"""Score base vs fine-tuned Gemma 4 on the held-out support tickets.

Same logic as the notebook's "Score it" section, runnable on its own:
    python3 scripts/score_tickets.py <studio-adapter-dir> [--limit N]
"""

import argparse
import json
import re
from pathlib import Path

FIELDS = ["order_id", "customer", "phone", "product", "category", "purchase_date", "amount"]

# What a careful prompt engineer would have to write without fine-tuning.
SCHEMA_PROMPT = (
    "Convert this support message to a ticket JSON with exactly these keys: "
    "order_id, customer, phone, product, category, purchase_date, amount. "
    "category is one of: damaged, defective, late_delivery, missing_part, "
    "not_delivered, return_request, wrong_item. purchase_date is YYYY-MM-DD, "
    "amount is a number, and anything not mentioned is null. Respond with the JSON only."
)


def parse(text):
    """(strict, ticket): strict is True only for a bare JSON object; ticket is
    the first {...} found, leniently, so a fenced answer still gets field credit."""
    try:
        return True, json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return False, json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return False, None


def score(rows, answers):
    strict = fields = exact = 0
    for row, answer in zip(rows, answers):
        truth = json.loads(row["messages"][1]["content"])
        ok, ticket = parse(answer)
        strict += ok
        if isinstance(ticket, dict):
            hits = sum(ticket.get(f) == truth[f] for f in FIELDS)
            fields += hits
            exact += hits == len(FIELDS) and set(ticket) == set(FIELDS)
    n = len(rows)
    return {"bare JSON": strict / n, "fields correct": fields / (n * len(FIELDS)), "whole ticket exact": exact / n}


def generate(model, tokenizer, prompts, max_new_tokens=160, batch_size=8):
    """Greedy answers, batched, with thinking off to match how the training
    examples were formatted."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # multimodal models load a processor
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for p in prompts[i : i + batch_size]
        ]
        inputs = tokenizer(text=texts, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        out += tok.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--eval", default=str(Path(__file__).resolve().parent.parent / "data/support_tickets_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    from unsloth import FastModel

    rows = [json.loads(l) for l in open(args.eval)][: args.limit]
    short = [r["messages"][0]["content"] for r in rows]
    schema = [SCHEMA_PROMPT + "\n\n" + p.split("\n\n", 1)[1] for p in short]

    model, tokenizer = FastModel.from_pretrained(args.adapter, load_in_4bit=False)
    FastModel.for_inference(model)

    def report(name, answers):
        print(f"{name:26s}", {k: f"{v:.0%}" for k, v in score(rows, answers).items()}, flush=True)
        print("   e.g.", answers[3][:300].replace("\n", " "), flush=True)

    with model.disable_adapter():
        report("base, short prompt", generate(model, tokenizer, short))
        report("base, schema prompt", generate(model, tokenizer, schema))
    report("fine-tuned, short prompt", generate(model, tokenizer, short))
