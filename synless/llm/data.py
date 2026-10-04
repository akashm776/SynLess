"""Auditable numeric distractors, disjoint partitions, and boundary-safe masks."""

from fractions import Fraction
import hashlib
import json
import random
import re
import unicodedata

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def numeric(text):
    value = text.strip()
    pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?"
    if not re.fullmatch(pattern, value):
        raise ValueError(f"Not a strict numeric answer: {value!r}")
    return Fraction(value.replace(",", ""))


def canonical(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    # Exact terminating decimals; fall back to exact fractions otherwise.
    d, twos, fives = value.denominator, 0, 0
    while d % 2 == 0:
        twos, d = twos + 1, d // 2
    while d % 5 == 0:
        fives, d = fives + 1, d // 5
    if d != 1:
        return f"{value.numerator}/{value.denominator}"
    digits = max(twos, fives)
    scaled = abs(value.numerator) * (10 ** digits // value.denominator)
    text = str(scaled).zfill(digits + 1)
    return ("-" if value < 0 else "") + (text[:-digits] + "." + text[-digits:]).rstrip("0").rstrip(".")


def distractors(gold, record_id, seed=1701):
    target = numeric(gold)
    values = []
    offsets = [-2, -1, 1, 2] + [v for i in range(3, 9) for v in (i, -i)]
    for offset in offsets:
        candidate = target + offset
        if target >= 0 and candidate < 0:
            continue
        if candidate != target and candidate not in values:
            values.append(candidate)
        if len(values) == 4:
            break
    random.Random(digest([seed, record_id])).shuffle(values)
    return [canonical(v) for v in values]


def prompt(question):
    return f"Question: {question}\nGive only the final numeric answer.\nAnswer:"


def normalize_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


class ByteTokenizer:
    """ASCII-only offline smoke tokenizer, not a real-model tokenizer."""
    eos_token_id, pad_token_id = 1, 0

    def __call__(self, text, **kwargs):
        if not text.isascii():
            raise ValueError("Smoke tokenizer requires ASCII")
        return {"input_ids": [ord(c)+3 for c in text],
                "offset_mapping": [(i, i+1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i-3) for i in ids if 3 <= i < 259)


def encode(tokenizer, question, answer, max_length):
    prefix = prompt(question)
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    enc = tokenizer(prefix + " " + answer, add_special_tokens=True, return_offsets_mapping=True)
    ids = enc["input_ids"]
    # Joint encoding must not change the prompt prefix or straddle its boundary.
    if ids[:len(prefix_ids)] != prefix_ids or any(s < len(prefix) < e for s, e in enc["offset_mapping"]):
        raise ValueError("prompt_token_boundary")
    start = len(prefix_ids)
    if start == 0 or start >= len(ids) or tokenizer.eos_token_id is None:
        raise ValueError("empty_answer_or_missing_eos")
    numeric_mask = [False] * len(ids)
    # Include numeric content (not a standalone leading-space token) in pooling/NLL.
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        numeric_mask[i] = i >= start and e > len(prefix)+1 and s < len(prefix)+1+len(answer)
    if not any(numeric_mask):
        raise ValueError("empty_numeric_mask")
    ids = ids + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError("overlength")
    return {"input_ids": ids, "labels": [-100]*start + ids[start:],
            "numeric_mask": numeric_mask + [False], "prompt_end": start-1,
            "prompt_ids": prefix_ids}


def build_records(train, test, tokenizer, max_length, split_seed):
    audits = {"duplicates": [], "excluded": [], "original_counts": {"train": len(train), "test": len(test)}}
    train_groups = {normalize_question(r["question"]) for r in train}
    seen = set()
    eligible = {"train": [], "test": []}
    for split, rows in [("train", train), ("test", test)]:
        for index, raw in enumerate(rows):
            key = normalize_question(raw["question"])
            ident = f"{split}:{index}:{digest(key)[:16]}"
            if key in seen or (split == "test" and key in train_groups):
                audits["duplicates"].append({"id": ident, "question_hash": digest(key)})
                continue
            seen.add(key)
            try:
                answer = canonical(numeric(raw["answer"].rsplit("####", 1)[-1]))
                wrong = distractors(answer, digest(key), split_seed)
                correct = encode(tokenizer, raw["question"], answer, max_length)
                # Evaluation eligibility uses correct completion only. No wrong-answer
                # support is needed for M/D/test; A/B eligibility is checked below.
            except (ValueError, ZeroDivisionError) as exc:
                audits["excluded"].append({"id": ident, "reason": str(exc)})
                continue
            eligible[split].append({"id": ident, "question_hash": digest(key),
                "question": raw["question"], "answer": answer, "wrong": wrong, "correct": correct})
    shuffled = eligible["train"].copy()
    random.Random(split_seed).shuffle(shuffled)
    n = len(shuffled)
    a, m, d = int(.4*n), int(.15*n), int(.1*n)
    partitions = {"A": shuffled[:a], "M": shuffled[a:a+m], "D": shuffled[a+m:a+m+d],
                  "B": shuffled[a+m+d:], "test": eligible["test"]}
    for name in ("A", "B"):
        kept = []
        for row in partitions[name]:
            try:
                row["negatives"] = [encode(tokenizer, row["question"], w, max_length) for w in row["wrong"]]
                assert all(x["prompt_ids"] == row["correct"]["prompt_ids"] for x in row["negatives"])
            except ValueError as exc:
                audits["excluded"].append({"id": row["id"], "partition": name, "reason": str(exc)})
                continue
            kept.append(row)
        partitions[name] = kept
    if any(not rows for rows in partitions.values()):
        raise ValueError("An eligible partition is empty")
    audits["eligible_counts"] = {name: len(rows) for name, rows in partitions.items()}
    audits["answer_token_lengths"] = {name: [sum(r["correct"]["numeric_mask"]) for r in rows]
                                      for name, rows in partitions.items()}
    audits["negative_token_lengths"] = {name: [[sum(x["numeric_mask"]) for x in r["negatives"]]
                                               for r in partitions[name]] for name in ("A", "B")}
    return partitions, audits


def collate(encodings, pad_id, device):
    length = max(len(r["input_ids"]) for r in encodings)
    def pad(key, fill):
        return [r[key] + [fill]*(length-len(r[key])) for r in encodings]
    return {"input_ids": torch.tensor(pad("input_ids", pad_id), device=device),
            "labels": torch.tensor(pad("labels", -100), device=device),
            "numeric_mask": torch.tensor(pad("numeric_mask", False), device=device),
            "attention_mask": torch.tensor([[1]*len(r["input_ids"]) + [0]*(length-len(r["input_ids"]))
                                              for r in encodings], device=device),
            "prompt_end": torch.tensor([r["prompt_end"] for r in encodings], device=device)}


def scheduled_batch(rows, step, batch_size, seed):
    """Stateless no-replacement epoch shuffles: exact resume from step alone."""
    selected = []
    for position in range(step*batch_size, (step+1)*batch_size):
        epoch, offset = divmod(position, len(rows))
        order = list(range(len(rows)))
        random.Random(digest([seed, epoch])).shuffle(order)
        selected.append(rows[order[offset]])
    return selected
