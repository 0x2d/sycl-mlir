#!/usr/bin/env python3
import csv
from pathlib import Path

LLVM_PATH = Path(r"/home/oyyc/sycl-bench/build/sycl-bench.csv")
MLIR_PATH = Path(r"/home/oyyc/sycl-bench/build-mlir/sycl-bench-mlir.csv")
OUT_PATH = Path(r"/home/oyyc/sycl-bench/build-mlir/sycl-bench-summary.csv")

# Header line (without leading "# ") taken from the benchmark output:
HEADER = ("Benchmark name,Verification,device-name,kernel-time-mean,kernel-time-median,"
          "kernel-time-min,kernel-time-samples,kernel-time-stddev,kernel-time-throughput,"
          "local-size,problem-size,run-time-mean,run-time-median,run-time-min,"
          "run-time-samples,run-time-stddev,run-time-throughput,sycl-implementation,"
          "throughput-metric").split(",")

KEY_COLS = ["Benchmark name", "local-size", "problem-size"]
# Columns kept once (taken from llvm side, falling back to mlir):
SHARED_COLS = ["device-name"]
# Per-implementation columns:
PER_IMPL_COLS = [
    "Verification",
    "run-time-median",
]


def load(path):
    """Return a dict keyed by (Benchmark name, local-size, problem-size) -> row dict."""
    rows = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw or raw[0].startswith("#"):
                continue
            row = dict(zip(HEADER, raw))
            key = (row["Benchmark name"], row["local-size"], row["problem-size"])
            # Preserve insertion order of first appearance; later duplicates overwrite.
            rows[key] = row
    return rows


def order(keys_llvm, keys_mlir):
    """Preserve llvm order, then append mlir-only keys in their original order."""
    seen = set()
    out = []
    for k in keys_llvm:
        if k not in seen:
            out.append(k)
            seen.add(k)
    for k in keys_mlir:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def main():
    llvm = load(LLVM_PATH)
    mlir = load(MLIR_PATH)

    out_header = (
        KEY_COLS
        + SHARED_COLS
        + [f"{c}_llvm" for c in PER_IMPL_COLS]
        + [f"{c}_mlir" for c in PER_IMPL_COLS]
    )

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for key in order(list(llvm.keys()), list(mlir.keys())):
            l = llvm.get(key)
            m = mlir.get(key)
            ref = l or m
            row = [ref[c] for c in KEY_COLS]
            row += [(l or m).get("device-name", "")]
            row += [(l[c] if l else "") for c in PER_IMPL_COLS]
            row += [(m[c] if m else "") for c in PER_IMPL_COLS]
            writer.writerow(row)

    print(f"Wrote {OUT_PATH}")
    print(f"  llvm rows: {len(llvm)}")
    print(f"  mlir rows: {len(mlir)}")
    common = set(llvm.keys()) & set(mlir.keys())
    print(f"  common keys: {len(common)}")
    print(f"  llvm-only:   {len(set(llvm.keys()) - common)}")
    print(f"  mlir-only:   {len(set(mlir.keys()) - common)}")


if __name__ == "__main__":
    main()
