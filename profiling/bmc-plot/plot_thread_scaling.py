#!/usr/bin/env python3
"""Plot thread scaling results from GENIE profiling.

Reads timing CSV files with thread counts encoded in filenames and produces
two-panel plots: stacked bar (absolute time) and speedup vs thread count,
with operations aggregated into readable categories.

Usage:
    python plot_thread_scaling.py -i aou_j2 -o aou_j2_scaling.png
    python plot_thread_scaling.py -i aou_j100 -o aou_j100_scaling.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
import re

# Stack order (insertion order = stack order, bottom to top)
CATEGORIES = [
    "File I/O",
    "Format conversion",
    "BLAS L1",
    "BLAS L2",
    "Memory management",
    "Other",
]

COLORS = dict(zip(CATEGORIES, sns.color_palette("colorblind", len(CATEGORIES))))

# Explicit mapping from every known operation name to its category.
# Unknown names fall back to "Other" with a printed warning.
NAME_TO_CATEGORY = {
    # --- I/O ---
    "root:file_io":                                          "File I/O",
    "root:file_io:covariate":                                "File I/O",
    "root:file_io:environment":                              "File I/O",
    "root:file_io:phenotype":                                "File I/O",
    "root:file_io:annotation":                               "File I/O",
    "root:jackknife:file_io":                                "File I/O",
    "root:jackknife:file_io:genotype":                       "File I/O",
    # --- Format conversion: BED → internal format ---
    "root:jackknife:file_io:genotype:transform":             "Format conversion",
    "root:jackknife:file_io:genotype:compute_maf":           "Format conversion",
    # --- BLAS L1: covariate-adjusted cross products ---
    "root:jackknife:bins:compute_gxe:UXXz":                  "BLAS L1",
    "root:jackknife:bins:compute_gxe:XXUz":                  "BLAS L1",
    "root:jackknife:bins:compute:UXXz":                      "BLAS L1",
    "root:jackknife:bins:compute:XXUz":                      "BLAS L1",
    "root:substract":                                        "BLAS L1",
    "root:trace_estimation":                                 "BLAS L1",
    # --- BLAS L2: core G*v and G^T*v products ---
    "root:jackknife:XXz:Xt_v":                               "BLAS L2",
    "root:jackknife:XXz:Xv":                                 "BLAS L2",
    "root:jackknife:yXXy:Xv":                                "BLAS L2",
    "root:jackknife:bins:compute_gxe:UXXz:XXz:Xt_v":         "BLAS L2",
    "root:jackknife:bins:compute_gxe:UXXz:XXz:Xv":           "BLAS L2",
    "root:jackknife:bins:compute_gxe:XXUz:XXz:Xt_v":         "BLAS L2",
    "root:jackknife:bins:compute_gxe:XXUz:XXz:Xv":           "BLAS L2",
    "root:jackknife:bins:compute_gxe:yXXy:yXXy:Xv":          "BLAS L2",
    "root:jackknife:bins:compute:XXUz:XXz:Xt_v":             "BLAS L2",
    "root:jackknife:bins:compute:XXz:XXz:Xt_v":              "BLAS L2",
    "root:jackknife:bins:compute:XXUz:XXz:Xv":               "BLAS L2",
    "root:jackknife:bins:compute:XXz:XXz:Xv":                "BLAS L2",
    "root:jackknife:bins:compute:yXXy:yXXy:Xv":              "BLAS L2",
    # --- Memory management: allocation/deallocation ---
    "root:jackknife:matmult_setup":                          "Memory management",
    "root:jackknife:bins:matrix_setup":                      "Memory management",
    "root:jackknife:bins:matrix_setup:matmult_setup":        "Memory management",
    "root:jackknife:matrix_setup":                           "Memory management",
    "root:jackknife:bins:matrix_cleanup":                    "Memory management",
    # --- Other: container/loop nodes with small self-time ---
    "root":                                                  "Other",
    "root:initialize":                                       "Other",
    "root:initialize:covariate_regression":                  "Other",
    "root:jackknife":                                        "Other",
    "root:jackknife:XXz":                                    "Other",
    "root:jackknife:yXXy":                                   "Other",
    "root:jackknife:bins":                                   "Other",
    "root:jackknife:bins:compute":                           "Other",
    "root:jackknife:bins:compute_gxe":                       "Other",
    "root:jackknife:bins:compute_gxe:UXXz:XXz":              "Other",
    "root:jackknife:bins:compute_gxe:XXUz:XXz":              "Other",
    "root:jackknife:bins:compute:XXUz:XXz":                  "Other",
    "root:jackknife:bins:compute:XXz:XXz":                   "Other",
    "root:jackknife:bins:compute_gxe:yXXy":                  "Other",
    "root:jackknife:bins:compute_gxe:yXXy:yXXy":             "Other",
    "root:jackknife:bins:compute:yXXy":                      "Other",
    "root:jackknife:bins:compute:yXXy:yXXy":                 "Other",
    "root:jackknife:bins:compute:XXz":                       "Other",
    "root:trace_estimation:linear_solve":                    "Other",
}


def categorize(name: str) -> str:
    """Map a hierarchical operation name to a readable category."""
    cat = NAME_TO_CATEGORY.get(name)
    if cat is None:
        print(f"Warning: unknown operation '{name}', assigning to 'Other'")
        return "Other"
    return cat


def load_thread_data(results_dir: Path) -> pd.DataFrame:
    """Load timing CSV files and extract thread counts from filenames."""
    records = []
    for csv_file in sorted(results_dir.glob("*_timing.csv")):
        match = re.search(r"_t(\d+)_timing\.csv$", csv_file.name)
        if not match:
            continue
        threads = int(match.group(1))

        df = pd.read_csv(csv_file)
        df["threads"] = threads
        dataset = re.sub(r"_t\d+_timing$", "", csv_file.stem)
        df["dataset"] = dataset
        records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize operations and aggregate self_seconds by category."""
    df = df[df["name"].notna() & (df["name"] != "")].copy()
    df["category"] = df["name"].apply(categorize)
    agg = (
        df.groupby(["dataset", "threads", "category"], as_index=False)["self_seconds"]
        .sum()
    )
    return agg


def plot_thread_scaling(
    df: pd.DataFrame,
    output_path: Path = None,
    min_speedup_time: float = 1.0,
):
    """Create stacked bar + speedup plots per dataset."""
    agg = aggregate_by_category(df)
    datasets = sorted(agg["dataset"].unique())

    for ds in datasets:
        ds_agg = agg[agg["dataset"] == ds]
        threads = sorted(ds_agg["threads"].unique())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        # fig.suptitle("GENIE Runtime Analysis", fontsize=13)

        # --- Left panel: stacked bar chart ---
        x_pos = np.arange(len(threads))
        bottoms = np.zeros(len(threads))

        for cat in COLORS:
            heights = []
            for t in threads:
                row = ds_agg[(ds_agg["threads"] == t) & (ds_agg["category"] == cat)]
                heights.append(row["self_seconds"].values[0] if len(row) else 0.0)
            heights = np.array(heights)
            ax1.bar(
                x_pos, heights, bottom=bottoms,
                label=cat, color=COLORS[cat], width=0.7,
            )
            bottoms += heights

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(t) for t in threads])
        ax1.set_xlabel("Threads")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("(a) Absolute Time")
        ax1.grid(axis="y", alpha=0.3)

        # --- Right panel: speedup lines ---
        # Get t1 time per category for baseline
        t1 = threads[0]
        cat_t1 = {}
        for cat in COLORS:
            row = ds_agg[(ds_agg["threads"] == t1) & (ds_agg["category"] == cat)]
            cat_t1[cat] = row["self_seconds"].values[0] if len(row) else 0.0

        for cat in COLORS:
            if cat_t1.get(cat, 0) < min_speedup_time:
                continue
            times = []
            for t in threads:
                row = ds_agg[(ds_agg["threads"] == t) & (ds_agg["category"] == cat)]
                times.append(row["self_seconds"].values[0] if len(row) else np.nan)
            speedup = [cat_t1[cat] / v if v and v > 0 else np.nan for v in times]
            ax2.plot(threads, speedup, "o-", label=cat, color=COLORS[cat])

        # Ideal scaling line
        ax2.plot(threads, threads, "k--", label="ideal", alpha=0.5)

        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log", base=2)
        ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.set_xlabel("Threads")
        ax2.set_ylabel("Speedup (T1 / Tn)")
        ax2.set_title("(b) Speedup")
        ax2.grid(True, alpha=0.3)

        # --- Shared legend below both panels ---
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Merge: use bar handles for categories, add ideal line from ax2
        by_label = dict(zip(labels1, handles1))
        for h, l in zip(handles2, labels2):
            if l not in by_label:
                by_label[l] = h
        # Reorder to match COLORS + ideal
        ordered_labels = [c for c in COLORS if c in by_label] + (
            ["ideal"] if "ideal" in by_label else []
        )
        ordered_handles = [by_label[l] for l in ordered_labels]

        fig.legend(
            ordered_handles, ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.22)

        if output_path:
            if len(datasets) > 1:
                stem = output_path.stem
                suffix = output_path.suffix
                out = output_path.parent / f"{stem}_{ds}{suffix}"
            else:
                out = output_path
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {out}")
        else:
            plt.show()

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GENIE thread scaling results")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).parent / "aou_j2",
            Path(__file__).parent / "aou_j100",
        ],
        help="Directories containing timing CSV files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (e.g., plot.png). If not specified, displays plot.",
    )
    parser.add_argument(
        "--min-speedup-time",
        type=float,
        default=1.0,
        help="Minimum t1 time (seconds) for a category to appear in speedup panel",
    )
    args = parser.parse_args()

    frames = []
    for d in args.input:
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping")
            continue
        part = load_thread_data(d)
        if not part.empty:
            frames.append(part)

    if not frames:
        print("Error: No timing data found")
        return 1

    df = pd.concat(frames, ignore_index=True)
    plot_thread_scaling(df, args.output, args.min_speedup_time)
    return 0


if __name__ == "__main__":
    exit(main())
