#!/usr/bin/env python3
"""
AID323 – Satellite Image Analysis
Performance Plotting Script
===========================
Reads sequential_time.txt and parallel_time.txt produced by
benchmark.sh and generates four publication-quality figures:

  1. Execution time vs. MPI process count (per OMP thread count)
  2. Speedup        vs. MPI process count
  3. Parallel efficiency vs. MPI process count
  4. Communication overhead as % of total time
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 12,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "figure.dpi"     : 150,
    "axes.grid"      : True,
    "grid.alpha"     : 0.35,
})

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

os.makedirs("results", exist_ok=True)

# ── Load sequential time ─────────────────────────────────────
try:
    with open("sequential_time.txt") as f:
        T_seq = float(f.read().strip())
except FileNotFoundError:
    print("ERROR: sequential_time.txt not found. Run sequential first.")
    sys.exit(1)

print(f"Sequential baseline: {T_seq:.4f} s")

# ── Load parallel results ────────────────────────────────────
# Columns: nprocs  nthreads  total  comm  ndvi_time  conv_time
try:
    data = np.loadtxt("parallel_time.txt")
except Exception as e:
    print(f"ERROR reading parallel_time.txt: {e}")
    sys.exit(1)

if data.ndim == 1:
    data = data[np.newaxis, :]

nprocs_col  = data[:, 0].astype(int)
nthreads_col= data[:, 1].astype(int)
total_col   = data[:, 2]
comm_col    = data[:, 3]

thread_counts = sorted(set(nthreads_col))
proc_counts   = sorted(set(nprocs_col))

def get_series(nt):
    mask = nthreads_col == nt
    procs = nprocs_col[mask]
    order = np.argsort(procs)
    return procs[order], total_col[mask][order], comm_col[mask][order]

# ── Figure 1: Execution time ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, nt in enumerate(thread_counts):
    procs, times, _ = get_series(nt)
    ax.plot(procs, times, marker=MARKERS[i], color=COLORS[i],
            label=f"{nt} OMP thread{'s' if nt>1 else ''}")

ax.axhline(T_seq, ls="--", color="black", lw=1.5, label=f"Sequential ({T_seq:.2f} s)")
ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Execution Time vs. MPI Process Count")
ax.set_xticks(proc_counts)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("results/fig1_exec_time.png")
print("  Saved results/fig1_exec_time.png")
plt.close(fig)

# ── Figure 2: Speedup ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
# Ideal speedup line
max_proc = max(proc_counts)
ideal_x  = np.array([1, max_proc])
ax.plot(ideal_x, ideal_x, ls=":", color="grey", lw=1.5, label="Ideal speedup")

for i, nt in enumerate(thread_counts):
    procs, times, _ = get_series(nt)
    speedup = T_seq / times
    ax.plot(procs, speedup, marker=MARKERS[i], color=COLORS[i],
            label=f"{nt} OMP thread{'s' if nt>1 else ''}")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Speedup  (T_seq / T_par)")
ax.set_title("Speedup vs. MPI Process Count")
ax.set_xticks(proc_counts)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("results/fig2_speedup.png")
print("  Saved results/fig2_speedup.png")
plt.close(fig)

# ── Figure 3: Parallel Efficiency ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(1.0, ls=":", color="grey", lw=1.5, label="Ideal (100%)")

for i, nt in enumerate(thread_counts):
    procs, times, _ = get_series(nt)
    total_cores = procs * nt
    efficiency  = (T_seq / times) / total_cores
    ax.plot(procs, efficiency * 100, marker=MARKERS[i], color=COLORS[i],
            label=f"{nt} OMP thread{'s' if nt>1 else ''}")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Parallel Efficiency (%)")
ax.set_title("Parallel Efficiency vs. MPI Process Count")
ax.set_xticks(proc_counts)
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("results/fig3_efficiency.png")
print("  Saved results/fig3_efficiency.png")
plt.close(fig)

# ── Figure 4: Communication overhead ────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for i, nt in enumerate(thread_counts):
    procs, times, comms = get_series(nt)
    overhead_pct = (comms / times) * 100.0
    ax.plot(procs, overhead_pct, marker=MARKERS[i], color=COLORS[i],
            label=f"{nt} OMP thread{'s' if nt>1 else ''}")

ax.set_xlabel("Number of MPI Processes")
ax.set_ylabel("Communication Overhead (% of total time)")
ax.set_title("Halo Exchange Overhead vs. MPI Process Count")
ax.set_xticks(proc_counts)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("results/fig4_comm_overhead.png")
print("  Saved results/fig4_comm_overhead.png")
plt.close(fig)

# ── Summary table ────────────────────────────────────────────
print("\n=== Performance Summary ===")
print(f"{'MPI':>5} {'OMP':>5} {'Time(s)':>10} {'Speedup':>10} {'Efficiency':>12} {'Comm%':>8}")
print("-" * 55)
for row in data:
    np_, nt, tot, comm = int(row[0]), int(row[1]), row[2], row[3]
    sp = T_seq / tot
    eff = sp / (np_ * nt) * 100
    print(f"{np_:>5} {nt:>5} {tot:>10.3f} {sp:>10.2f} {eff:>11.1f}% {comm/tot*100:>7.1f}%")

print("\nDone. All plots saved in results/")
