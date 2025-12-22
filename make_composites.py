# make_composites.py

import os
import csv
import glob
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def load_series(path, key):
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[key]))
    return values


def main():
    print("▶ Building composite identity vs context plots...")

    for model_dir in sorted(glob.glob(f"{RESULTS_DIR}/*")):
        if not os.path.isdir(model_dir):
            continue

        model = os.path.basename(model_dir)
        summary_path = os.path.join(model_dir, "summary.csv")

        if not os.path.exists(summary_path):
            continue

        # ---- group labels by token ----
        by_token = {}
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                by_token.setdefault(row["token"], []).append(row["label"])

        # ---- build plots ----
        for token, labels in by_token.items():
            if len(labels) < 2:
                continue  # need at least two contexts

            context_csv = os.path.join(model_dir, f"{token}_context.csv")
            if not os.path.exists(context_csv):
                continue

            context = load_series(context_csv, "context_similarity")
            layers = range(len(context))

            plt.figure(figsize=(8, 5))

            # ---- plot ALL identity curves ----
            for label in labels:
                id_csv = os.path.join(model_dir, f"{label}.csv")
                if not os.path.exists(id_csv):
                    continue

                identity = load_series(id_csv, "identity_similarity")
                plt.plot(
                    layers,
                    identity,
                    marker="o",
                    label=f"Identity: {label}",
                )

            # ---- plot context curve ----
            plt.plot(
                layers,
                context,
                linestyle="--",
                linewidth=2,
                marker="o",
                label="Context similarity",
            )

            plt.xlabel("Layer")
            plt.ylabel("Cosine similarity")
            plt.title(f"{model} — Identity vs Context\nToken: {token}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            out = os.path.join(model_dir, f"{token}_identity_vs_context.png")
            plt.savefig(out, dpi=150)
            plt.close()

            print(f"✓ [{model}] {token}_identity_vs_context.png")

    print("✅ Composite plots complete.")


if __name__ == "__main__":
    main()
