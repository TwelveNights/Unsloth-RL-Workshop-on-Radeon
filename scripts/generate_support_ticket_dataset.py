#!/usr/bin/env python3
"""Generate the support-ticket SFT dataset used by the Unsloth Studio workshop.

Each row turns a messy customer support message into a fixed-schema ticket:

    {"order_id": "ORD-24592", "customer": "Jordan", "phone": "(428) 342-2679",
     "product": "desk lamp", "category": "missing_part",
     "purchase_date": "2026-11-24", "amount": 104.75}

The instruction is deliberately short ("Convert this support message to a
ticket JSON."): the schema, the category list, the ISO date format and the
null-for-missing rule are never spelled out, so the model can only learn them
from the examples. That is what supervised fine-tuning is for.

Chat format (a `messages` column of user/assistant turns), which Studio
detects on upload. Not Alpaca: Studio trains Alpaca rows with a generic
"### Instruction / ### Response" prompt, while Studio's Chat page -- and
anyone using the model afterwards -- talks to it through Gemma's own chat
template. Chat rows are trained through that same template.

Usage: python3 scripts/generate_support_ticket_dataset.py
Writes data/support_tickets_train.jsonl and data/support_tickets_eval.jsonl.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

INSTRUCTION = "Convert this support message to a ticket JSON."

NAMES = [
    "Jordan", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Noah", "Lucia",
    "Daniel", "Fatima", "Chris", "Yuki", "Omar", "Grace", "Liam", "Ana",
    "Ben", "Sofia", "Raj", "Maya", "Kofi", "Ingrid", "Mateo", "Hana",
]

PRODUCTS = [
    "wireless headphones", "espresso machine", "laptop stand", "running shoes",
    "desk lamp", "blender", "backpack", "office chair", "tablet case",
    "air purifier", "smartwatch", "bluetooth speaker", "rice cooker",
    "monitor arm", "electric kettle", "yoga mat",
]

# Several phrasings per category, so the model has to learn the category,
# not memorise one sentence.
ISSUES = {
    "damaged": [
        "arrived damaged", "had a cracked casing on arrival",
        "showed up with a big dent", "came with the box crushed and the item broken",
    ],
    "defective": [
        "stopped working after a week", "won't turn on at all",
        "keeps shutting off on its own", "makes a grinding noise and then dies",
    ],
    "late_delivery": [
        "arrived three weeks late", "is still in transit way past the estimate",
        "hasn't shipped yet even though it was promised last week",
    ],
    "missing_part": [
        "is missing a part", "came without the power cable",
        "arrived with no screws in the box", "was missing the instruction manual and a bracket",
    ],
    "not_delivered": [
        "was never delivered", "shows as delivered but never reached me",
        "seems to have been lost by the courier",
    ],
    "return_request": [
        "needs to be returned", "isn't what I expected and I'd like a refund",
        "turned out to be a duplicate order, so I want to send it back",
    ],
    "wrong_item": [
        "was the wrong size", "came in the wrong color",
        "is not the model I ordered", "turned out to be a completely different product",
    ],
}

TEMPLATES = [
    "Hi, my name is {name}. I'm writing about order {order_id}: the {product} {issue}. "
    "{date_clause}{amount_clause}{phone_clause}",
    "{name} here. Order {order_id} is a problem, the {product} {issue}. "
    "{date_clause}{amount_clause}{phone_clause}",
    "Hello support, the {product} from order {order_id} {issue}. "
    "{date_clause}{amount_clause}{phone_clause}Thanks, {name}",
    "Case #{case}. Customer {name} reports the {product} {issue} (ref {order_id}). "
    "{date_clause}{amount_clause}{phone_clause}",
    "hey, {product} {issue}!! order number {order_id}. "
    "{date_clause}{amount_clause}{phone_clause}- {name}",
    "Good afternoon. Tracking {tracking} says one thing, but the {product} {issue}. "
    "Order: {order_id}. {date_clause}{amount_clause}{phone_clause}Regards, {name}.",
]

DATE_FORMATS = [
    lambda d: d.strftime("%m/%d/%Y"),
    lambda d: d.strftime("%b %d, %Y").replace(" 0", " "),
    lambda d: f"{d.day} {d.strftime('%B %Y')}",
    lambda d: d.isoformat(),
    lambda d: f"{d.strftime('%B')} {d.day}, {d.year}",
]

AMOUNT_FORMATS = [
    lambda a: f"${a:,.2f}",
    lambda a: f"{a:.2f} USD",
    lambda a: f"USD {a:,.2f}",
]


def phone(rng):
    return f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"


def make_row(rng):
    name = rng.choice(NAMES)
    product = rng.choice(PRODUCTS)
    category = rng.choice(sorted(ISSUES))
    issue = rng.choice(ISSUES[category])
    order_id = f"ORD-{rng.randint(10000, 99999)}"

    # Each optional field is left out of some messages, so the model also
    # learns to emit null rather than invent a value.
    purchase = date(2026, 1, 1) + timedelta(days=rng.randint(0, 270))
    has_date = rng.random() < 0.8
    amount = round(rng.uniform(9, 1400), 2)
    has_amount = rng.random() < 0.75

    date_clause = rng.choice([
        "I bought it on {}. ", "Purchased {}. ", "Order placed {}. ",
    ]).format(rng.choice(DATE_FORMATS)(purchase)) if has_date else ""
    amount_clause = rng.choice([
        "I paid {}. ", "It cost {}. ", "Total was {}. ",
    ]).format(rng.choice(AMOUNT_FORMATS)(amount)) if has_amount else ""
    # The phone number is a ticket field: an earlier version treated it as a
    # distractor, and 60 steps could not talk Gemma out of its instinct that a
    # customer's phone number belongs in the ticket. Case and tracking numbers
    # remain distractors that look like IDs but never belong in it.
    phone_number = phone(rng) if rng.random() < 0.6 else None
    phone_clause = f"Call me on {phone_number}. " if phone_number else ""

    message = rng.choice(TEMPLATES).format(
        name=name, order_id=order_id, product=product, issue=issue,
        date_clause=date_clause, amount_clause=amount_clause, phone_clause=phone_clause,
        case=rng.randint(10000, 99999), tracking=f"1Z{rng.randint(10**9, 10**10 - 1)}",
    )
    ticket = {
        "order_id": order_id,
        "customer": name,
        "phone": phone_number,
        "product": product,
        "category": category,
        "purchase_date": purchase.isoformat() if has_date else None,
        "amount": amount if has_amount else None,
    }
    return {"messages": [
        {"role": "user", "content": f"{INSTRUCTION}\n\n{' '.join(message.split())}"},
        {"role": "assistant", "content": json.dumps(ticket)},
    ]}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data"
    # Different seeds, so the eval rows are unseen combinations.
    write(data / "support_tickets_train.jsonl", [make_row(random.Random(f"train-{i}")) for i in range(1000)])
    write(data / "support_tickets_eval.jsonl", [make_row(random.Random(f"eval-{i}")) for i in range(100)])
