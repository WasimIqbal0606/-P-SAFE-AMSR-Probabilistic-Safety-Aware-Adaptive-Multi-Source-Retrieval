"""
Generate publication-quality figures for the B-P-SAFE-AMSR manuscript.
Sources: multi_dataset_summary.csv, latency_breakdown.json
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Global style ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = "results_top_tier_psafe"

# ═══════════════════════════════════════════════════════════
#  FIGURE 1 — Latency Breakdown (Log-Scale Horizontal Bar)
# ═══════════════════════════════════════════════════════════

stages = [
    ("Dense ANN\n(FAISS)", 0.366, "#2196F3"),
    ("Router\nDecision", 0.100, "#4CAF50"),
    ("Graph\nExpansion", 0.110, "#8BC34A"),
    ("Score\nFusion", 0.500, "#CDDC39"),
    ("Feature\nExtraction", 2.655, "#FF9800"),
    ("BM25\nSearch", 13.943, "#FF5722"),
    ("Cross-Encoder\nRe-ranking", 736.686, "#D32F2F"),
]

labels = [s[0] for s in stages]
values = [s[1] for s in stages]
colors = [s[2] for s in stages]

fig1, ax1 = plt.subplots(figsize=(8, 4.5))
y_pos = np.arange(len(stages))
bars = ax1.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, height=0.65)

ax1.set_xscale('log')
ax1.set_xlabel('Latency (ms) — Log Scale')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels)
ax1.invert_yaxis()
ax1.set_xlim(0.05, 1200)

# Annotate each bar with exact ms value
for bar, val in zip(bars, values):
    x_pos = val * 1.15 if val < 500 else val * 0.4
    ha = 'left' if val < 500 else 'right'
    color = 'black' if val < 500 else 'white'
    ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
             f'{val:.3f} ms', va='center', ha=ha,
             fontsize=9.5, fontweight='bold', color=color)

# Add vertical annotation lines for thresholds
ax1.axvline(x=3.121, color='#1565C0', linestyle='--', alpha=0.7, linewidth=1.2)
ax1.text(3.121, -0.6, 'Router overhead\n≈ 3.1 ms', ha='center',
         fontsize=8.5, color='#1565C0', fontweight='bold')

ax1.axvline(x=505.026, color='#B71C1C', linestyle='-.', alpha=0.5, linewidth=1.0)
ax1.text(505.026, 7.1, 'P-SAFE Total\n505 ms', ha='center',
         fontsize=8.5, color='#B71C1C', fontweight='bold')

ax1.set_title('Per-Stage Inference Latency Decomposition (SciFact, High-Recall)',
              fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.2)

fig1.tight_layout()
fig1.savefig(f'{OUTDIR}/latency_breakdown.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/latency_breakdown.png")
plt.close(fig1)


# ═══════════════════════════════════════════════════════════
#  FIGURE 2 — Multi-Dataset Pareto Frontier (QR vs LS)
# ═══════════════════════════════════════════════════════════

# Data from multi_dataset_summary.csv (balanced & high_recall only)
data = {
    'SciFact': {
        'balanced':    {'QR': 0.644, 'LS': 0.379, 'HAR': 0.61},
        'high_recall': {'QR': 0.737, 'LS': 0.314, 'HAR': 0.68},
    },
    'FiQA': {
        'balanced':    {'QR': 0.613, 'LS': 0.348, 'HAR': 0.66},
        'high_recall': {'QR': 0.659, 'LS': 0.423, 'HAR': 0.58},
    },
    'NFCorpus': {
        'balanced':    {'QR': 0.881, 'LS': 0.265, 'HAR': 0.75},
        'high_recall': {'QR': 0.834, 'LS': 0.315, 'HAR': 0.69},
    },
    'ArguAna': {
        'balanced':    {'QR': 0.000, 'LS': 0.610, 'HAR': 0.40},
        'high_recall': {'QR': 0.000, 'LS': 0.551, 'HAR': 0.46},
    },
}

dataset_colors = {
    'SciFact':  '#1976D2',
    'FiQA':     '#388E3C',
    'NFCorpus': '#7B1FA2',
    'ArguAna':  '#D32F2F',
}

fig2, ax2 = plt.subplots(figsize=(8, 5.5))

# Plot each dataset with circle=balanced, square=high_recall
for ds_name, modes in data.items():
    c = dataset_colors[ds_name]
    bal = modes['balanced']
    hr = modes['high_recall']
    
    ax2.scatter(bal['LS'], bal['QR'], s=160, c=c, marker='o',
                edgecolors='white', linewidth=1.2, zorder=5)
    ax2.scatter(hr['LS'], hr['QR'], s=160, c=c, marker='s',
                edgecolors='white', linewidth=1.2, zorder=5)
    
    # Connect balanced → high_recall with a line
    ax2.plot([bal['LS'], hr['LS']], [bal['QR'], hr['QR']],
             color=c, linewidth=1.5, alpha=0.5, zorder=3)
    
    # Label the dataset near the balanced point
    offset_x, offset_y = 0.012, 0.025
    if ds_name == 'ArguAna':
        offset_y = -0.045
    ax2.annotate(ds_name, (bal['LS'], bal['QR']),
                 xytext=(bal['LS'] + offset_x, bal['QR'] + offset_y),
                 fontsize=10, fontweight='bold', color=c,
                 arrowprops=dict(arrowstyle='-', color=c, alpha=0.4))

# Shade the "Self-Protection Zone" (bottom-right)
rect = mpatches.FancyBboxPatch((0.50, -0.05), 0.18, 0.10,
                                boxstyle="round,pad=0.02",
                                facecolor='#FFCDD2', alpha=0.35,
                                edgecolor='#D32F2F', linewidth=1.5,
                                linestyle='--', zorder=1)
ax2.add_patch(rect)
ax2.text(0.59, 0.015, 'Algorithmic\nSelf-Protection\nZone', ha='center',
         fontsize=8.5, fontstyle='italic', color='#B71C1C', fontweight='bold')

# Shade the "Optimal Pareto Region" (top-left to top-right)
rect2 = mpatches.FancyBboxPatch((0.24, 0.58), 0.21, 0.33,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#C8E6C9', alpha=0.25,
                                 edgecolor='#388E3C', linewidth=1.2,
                                 linestyle=':', zorder=1)
ax2.add_patch(rect2)
ax2.text(0.345, 0.92, 'Optimal Pareto\nRegion', ha='center',
         fontsize=8.5, fontstyle='italic', color='#2E7D32', fontweight='bold')

# Reference lines
ax2.axhline(y=0.80, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
ax2.text(0.68, 0.81, 'QR = 80%', fontsize=8, color='gray', alpha=0.6)
ax2.axvline(x=0.30, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
ax2.text(0.305, 0.95, 'LS = 30%', fontsize=8, color='gray', alpha=0.6, rotation=90)

ax2.set_xlabel('Latency Saving (LS) vs. Deep Hybrid ↑ better', fontweight='bold')
ax2.set_ylabel('Quality Retention (QR) vs. Deep Hybrid ↑ better', fontweight='bold')
ax2.set_title('Compute–Quality Pareto Frontier Across BEIR Domains',
              fontweight='bold', pad=15)

ax2.set_xlim(0.22, 0.70)
ax2.set_ylim(-0.07, 1.0)
ax2.grid(True, alpha=0.15)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
           markersize=9, linestyle='None', label='Balanced Mode'),
    Line2D([0], [0], marker='s', color='gray', markerfacecolor='gray',
           markersize=9, linestyle='None', label='High-Recall Mode'),
    mpatches.Patch(facecolor='#FFCDD2', edgecolor='#D32F2F',
                   linestyle='--', label='Self-Protection Zone'),
    mpatches.Patch(facecolor='#C8E6C9', edgecolor='#388E3C',
                   linestyle=':', label='Optimal Pareto Region'),
]
ax2.legend(handles=legend_elements, loc='center left',
           framealpha=0.9, edgecolor='lightgray')

fig2.tight_layout()
fig2.savefig(f'{OUTDIR}/performance_tradeoffs.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/performance_tradeoffs.png")
plt.close(fig2)

print("\n[OK] All publication figures generated successfully.")
