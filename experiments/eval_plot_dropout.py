#!/usr/bin/env python3
"""
plot_dropout_jb_test_only_jittered_horizontal.py

Usage:
    python plot_dropout_jb_test_only_jittered_horizontal.py /path/to/json_dir

Description:
- Reads all JSON files matching proprietary_dropout_results_k_<k>.json
  (or the misspelled proprietary_dropout_restuls_k_<k>.json)
- Extracts test JB rates (prefers final_test_jb_rate)
- Plots Test JB rate vs k for each model
- Adds small horizontal jitter so overlapping points are visible
- Keeps x-axis ticks exactly at observed k values
- Replaces model display names:
      claude-instant-1 → claude-3.5-haiku
      claude-2 → claude-sonnet-4.5
"""

import os
import re
import json
import glob
import argparse
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Accept both correct and misspelled filename patterns
FILENAME_REGEXES = [
    re.compile(r"proprietary_dropout_results_k_([-+]?\d*\.?\d+)\.json$"),
    re.compile(r"proprietary_dropout_restuls_k_([-+]?\d*\.?\d+)\.json$"),
]


def find_k_in_filename(fname):
    """Extract numeric k value from filename."""
    base = os.path.basename(fname)
    for rx in FILENAME_REGEXES:
        m = rx.search(base)
        if m:
            s = m.group(1)
            return float(s) if ("." in s or "e" in s or "E" in s) else int(s)
    return None


def get_test_rate(model_obj):
    """Extract a single representative test JB rate."""
    if "final_test_jb_rate" in model_obj:
        return model_obj["final_test_jb_rate"]

    if "test_jb_rates" in model_obj and isinstance(model_obj["test_jb_rates"], list):
        vals = []
        for v in model_obj["test_jb_rates"]:
            if isinstance(v, (int, float)):
                vals.append(v)
            elif isinstance(v, list):
                nums = [x for x in v if isinstance(x, (int, float))]
                if nums:
                    vals.append(sum(nums) / len(nums))
        if vals:
            return sum(vals) / len(vals)

    if "test_jb" in model_obj and isinstance(model_obj["test_jb"], list):
        total = 0
        count = 0
        for arr in model_obj["test_jb"]:
            if isinstance(arr, list):
                for b in arr:
                    if isinstance(b, bool):
                        count += 1
                        if b:
                            total += 1
        if count:
            return total / count

    return None


def load_data(directory):
    """Load all JSONs and collect test JB rates by model."""
    files = glob.glob(os.path.join(directory, "*.json"))
    selected = []
    for f in files:
        k = find_k_in_filename(f)
        if k is not None:
            selected.append((float(k), f))
    selected.sort(key=lambda x: x[0])

    if not selected:
        raise FileNotFoundError("No matching JSON files found in directory.")

    data = defaultdict(lambda: {"k": [], "test": []})
    for k, path in selected:
        with open(path, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        for model_name, model_obj in j.items():
            test_rate = get_test_rate(model_obj)
            if test_rate is None:
                continue
            data[model_name]["k"].append(k)
            data[model_name]["test"].append(test_rate)
    return data


def plot(data, outpath=None, title="Test JB rate vs token-dropout k"):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_k_values = sorted(set(k for m in data.values() for k in m["k"]))
    jitter_scale = 0.008  # subtle horizontal jitter
    random.seed(0)

    # Rename models for legend display
    name_map = {
        "claude-instant-1": "claude-3.5-haiku",
        "claude-2": "claude-sonnet-4.5"
    }

    for model_name, series in data.items():
        display_name = name_map.get(model_name, model_name)
        pairs = sorted(zip(series["k"], series["test"]), key=lambda x: x[0])
        xs = [k + random.uniform(-jitter_scale, jitter_scale) for (k, _) in pairs]
        ys = [v for (_, v) in pairs]
        ax.plot(xs, ys, marker="o", linestyle="-", label=display_name, alpha=0.9)

    # Axis formatting
    ax.set_xlabel("k (token dropout rate in adversarial suffix)")
    ax.set_ylabel("Test JB rate")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize="small")

    # X ticks only at actual k values
    ax.set_xticks(all_k_values)

    # Ensure k=0 sits on left when only one point
    if len(all_k_values) == 1:
        k0 = all_k_values[0]
        left = k0 - 0.05
        right = k0 + 0.5
        ax.set_xlim(left, right)
    else:
        k_min, k_max = min(all_k_values), max(all_k_values)
        pad = (k_max - k_min) * 0.05
        ax.set_xlim(k_min - pad, k_max + pad)

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Saved plot to {outpath}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot test JB rates vs k with horizontal jitter and renamed models."
    )
    parser.add_argument("directory", nargs="?", default=".", help="Directory with JSON files")
    parser.add_argument("--out", "-o", default="jb_test_vs_k.png", help="Output image filename")
    args = parser.parse_args()

    data = load_data(args.directory)
    if not data:
        print("No data found to plot.")
        return

    plot(data, outpath=args.out)


if __name__ == "__main__":
    main()
