#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("results/_best15_matrices_all_formats.csv")

# Drop rows with NA speedup
df = df.dropna(subset=['Speedup'])
df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
df = df.dropna(subset=['Speedup'])

# Normalize format names to uppercase
df['Format'] = df['Format'].str.upper()

# -----------------------------
# Matrix and format info
# -----------------------------
matrices = df['MatrixFile'].unique()
# Fixed format order (from left to right in bar stack)
formats_order = ['HLL', 'ELL', 'CSRV', 'CSRS', 'COO']  # COO will be last/right
bar_height = 0.15  # height of individual bars

plt.figure(figsize=(12, 8))

# Assign one color per format
colors_list = plt.get_cmap('tab10').colors
format_colors = {fmt: colors_list[i % len(colors_list)] for i, fmt in enumerate(formats_order)}

# Position of bars on y-axis
y = np.arange(len(matrices))

# -----------------------------
# Plot bars in fixed format order
# -----------------------------
for i, fmt in enumerate(formats_order):
    fmt_data = []
    for m in matrices:
        val = df[(df['MatrixFile'] == m) & (df['Format'] == fmt)]['Speedup']
        fmt_data.append(val.values[0] if not val.empty else 0)
    plt.barh(y + i*bar_height, fmt_data, height=bar_height, color=format_colors[fmt], label=fmt)

# -----------------------------
# Baseline
# -----------------------------
plt.axvline(1.0, color='r', linestyle='--', label='Baseline (1x)')

# -----------------------------
# Y-axis labels
# -----------------------------
plt.yticks(y + bar_height*(len(formats_order)-1)/2, matrices, fontsize=12)

# -----------------------------
# Labels and title
# -----------------------------
plt.xlabel("Speedup (BWC / Standard)", fontsize=14)
plt.ylabel("Matrix", fontsize=14)
plt.title("Speedup of BWC vs Standard Formats", fontsize=16)

# -----------------------------
# Legend (formats + baseline)
# -----------------------------
plt.legend(fontsize=12)
plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
plt.savefig("speedup_histogram.png", dpi=300)
print("Histogram saved to speedup_histogram.png")
