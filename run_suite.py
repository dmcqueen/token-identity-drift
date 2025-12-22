# run_suite.py

import os
import csv
import yaml
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model


MODEL_NAMES = ["gpt2", "distilgpt2", "gpt2-medium"]
HALF_LIFE_THRESHOLD = 0.1
CDL_NOT_REACHED = -1


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


def compute_half_life(sims):
    for i, s in enumerate(sims):
        if s < HALF_LIFE_THRESHOLD:
            return i
    return CDL_NOT_REACHED


def compute_cdl(identity_sims, context_sims):
    for i, (i_s, c_s) in enumerate(zip(identity_sims, context_sims)):
        if c_s < i_s:
            return i
    return CDL_NOT_REACHED


def run_single_experiment(model_name, label, prompt, token):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    inputs = tok(prompt, return_tensors="pt")
    ids = inputs["input_ids"][0]
    tokens = tok.convert_ids_to_tokens(ids)

    if token not in tokens:
        raise ValueError(f"Token '{token}' not found in {tokens}")

    idx = tokens.index(token)

    with torch.no_grad():
        out = model(**inputs)

    emb = model.wte.weight[ids[idx]]
    identity = []
    vecs = []

    for h in out.hidden_states:
        v = h[0, idx]
        vecs.append(v)
        identity.append(cosine(v, emb))

    return {
        "label": label,
        "token": token,
        "identity": identity,
        "vecs": vecs,
        "half_life": compute_half_life(identity),
        "final": identity[-1],
    }


def main():
    with open("experiments.yaml") as f:
        experiments = yaml.safe_load(f)["experiments"]

    os.makedirs("results", exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"\n▶ Running suite for {model_name}")
        model_dir = os.path.join("results", model_name)
        os.makedirs(model_dir, exist_ok=True)

        results = []
        by_token = defaultdict(list)

        # ---- run experiments ----
        for exp in experiments:
            r = run_single_experiment(
                model_name,
                exp["label"],
                exp["prompt"],
                exp["token"],
            )
            results.append(r)
            by_token[r["token"]].append(r)

            with open(os.path.join(model_dir, f"{r['label']}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "identity_similarity"])
                for i, s in enumerate(r["identity"]):
                    w.writerow([i, s])

            print(f"✓ [{model_name}] {r['label']}")

        cdl_by_label = {}

        # ---- context + CDL ----
        for token, group in by_token.items():
            if len(group) < 2:
                continue

            a, b = group[:2]
            context = [cosine(v1, v2) for v1, v2 in zip(a["vecs"], b["vecs"])]

            with open(os.path.join(model_dir, f"{token}_context.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "context_similarity"])
                for i, s in enumerate(context):
                    w.writerow([i, s])

            cdl = compute_cdl(a["identity"], context)
            cdl_by_label[a["label"]] = cdl
            cdl_by_label[b["label"]] = cdl

        # ---- summary ----
        with open(os.path.join(model_dir, "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "model",
                "label",
                "token",
                "half_life_layer",
                "final_similarity",
                "context_dominance_layer",
            ])
            for r in results:
                w.writerow([
                    model_name,
                    r["label"],
                    r["token"],
                    r["half_life"],
                    r["final"],
                    cdl_by_label.get(r["label"], CDL_NOT_REACHED),
                ])

        print(f"✓ [{model_name}] summary.csv written")


if __name__ == "__main__":
    main()
