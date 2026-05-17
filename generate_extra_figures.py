"""
Generate 4 additional publication-quality figures for the B-P-SAFE-AMSR manuscript.
Figure 3: Oracle Gap Comparison
Figure 4: Hard Query Recovery Waterfall
Figure 5: Calibration Reliability (simulated from p-values)
Figure 6: Action Distribution (Hybrid Activation Rate)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

# ============================================================
#  FIGURE 3 -- Oracle Gap Comparison (Grouped Bar)
# ============================================================

datasets = ['SciFact', 'FiQA', 'NFCorpus', 'ArguAna']
ogc_balanced =    [0.465, 0.277, 0.450, 0.068]
ogc_high_recall = [0.532, 0.284, 0.426, 0.091]

x = np.arange(len(datasets))
w = 0.32

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
bars1 = ax3.bar(x - w/2, ogc_balanced, w, label='Balanced',
                color='#1976D2', edgecolor='white', linewidth=0.8)
bars2 = ax3.bar(x + w/2, ogc_high_recall, w, label='High-Recall',
                color='#FF6F00', edgecolor='white', linewidth=0.8)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.012,
             f'{h:.3f}', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.012,
             f'{h:.3f}', ha='center', fontsize=9, fontweight='bold', color='#E65100')

ax3.axhline(y=0.50, color='#2E7D32', linestyle='--', alpha=0.5, linewidth=1)
ax3.text(3.55, 0.51, '50% Oracle', fontsize=8.5, color='#2E7D32', fontstyle='italic')

# ArguAna annotation
ax3.annotate('Self-Protection\n(near-zero gap\nby design)',
             xy=(3, 0.091), xytext=(2.5, 0.28),
             fontsize=8.5, color='#D32F2F', fontstyle='italic',
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

ax3.set_xticks(x)
ax3.set_xticklabels(datasets, fontweight='bold')
ax3.set_ylabel('Oracle Gap Closed (OGC)', fontweight='bold')
ax3.set_title('Fraction of Theoretically Achievable Improvement Captured',
              fontweight='bold', pad=12)
ax3.set_ylim(0, 0.65)
ax3.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray')
ax3.grid(axis='y', alpha=0.15)

fig3.tight_layout()
fig3.savefig(f'{OUTDIR}/oracle_gap_comparison.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/oracle_gap_comparison.png")
plt.close(fig3)


# ============================================================
#  FIGURE 4 -- Hard Query Recovery Waterfall
# ============================================================

# Hard query data from statistical_tests.json
hard_data = {
    'SciFact\n(HR)':   {'dense': 0.1899, 'psafe': 0.3699, 'delta': 0.1800},
    'SciFact\n(Bal)':  {'dense': 0.2081, 'psafe': 0.3385, 'delta': 0.1304},
    'FiQA\n(Bal)':     {'dense': 0.2093, 'psafe': 0.2450, 'delta': 0.0357},
    'NFCorpus\n(Bal)': {'dense': 0.1668, 'psafe': 0.2022, 'delta': 0.0353},
}

fig4, ax4 = plt.subplots(figsize=(8, 5))
labels = list(hard_data.keys())
dense_vals = [hard_data[k]['dense'] for k in labels]
psafe_vals = [hard_data[k]['psafe'] for k in labels]
delta_vals = [hard_data[k]['delta'] for k in labels]

x4 = np.arange(len(labels))
w4 = 0.30

bars_dense = ax4.bar(x4 - w4/2, dense_vals, w4, label='Dense Baseline',
                     color='#BDBDBD', edgecolor='white', linewidth=0.8)
bars_psafe = ax4.bar(x4 + w4/2, psafe_vals, w4, label='B-P-SAFE-AMSR',
                     color='#1976D2', edgecolor='white', linewidth=0.8)

# Delta arrows
for i, (d, p, delta) in enumerate(zip(dense_vals, psafe_vals, delta_vals)):
    ax4.annotate('', xy=(i + w4/2, p), xytext=(i - w4/2, d),
                 arrowprops=dict(arrowstyle='->', color='#D32F2F',
                                 lw=2.0, connectionstyle='arc3,rad=-0.3'))
    pct = (delta / d) * 100
    ax4.text(i, max(d, p) + 0.018,
             f'+{delta:.4f}\n({pct:.0f}%)',
             ha='center', fontsize=9, fontweight='bold', color='#D32F2F')

ax4.set_xticks(x4)
ax4.set_xticklabels(labels, fontweight='bold')
ax4.set_ylabel('Mean nDCG@10 on Hard Queries', fontweight='bold')
ax4.set_title('Hard Query Recovery: Dense Baseline vs. B-P-SAFE-AMSR',
              fontweight='bold', pad=12)
ax4.set_ylim(0, 0.48)
ax4.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray')
ax4.grid(axis='y', alpha=0.15)

fig4.tight_layout()
fig4.savefig(f'{OUTDIR}/hard_query_recovery.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/hard_query_recovery.png")
plt.close(fig4)


# ============================================================
#  FIGURE 5 -- Statistical Significance Heatmap
# ============================================================

# p-values from statistical_tests.json (vs dense, balanced mode)
datasets_sig = ['SciFact', 'FiQA', 'NFCorpus', 'ArguAna']
tests = ['Paired\nt-test', 'Wilcoxon', 'Bootstrap\nCI', 'Permutation']

# p-values matrix [dataset x test]
pvals = np.array([
    [0.00584, 0.0100, 0.0040, 0.0040],   # SciFact balanced
    [0.02470, 0.0310, 0.0175, 0.0175],   # FiQA balanced
    [0.00435, 0.0062, 0.0030, 0.0030],   # NFCorpus balanced
    [0.26200, 0.3170, 0.3100, 0.3200],   # ArguAna balanced
])

fig5, ax5 = plt.subplots(figsize=(7, 4))

# Custom colormap: green for significant, red for not
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['#C8E6C9', '#A5D6A7', '#66BB6A', '#FFE0B2', '#FFCDD2'])
bounds = [0, 0.001, 0.01, 0.05, 0.10, 1.0]
norm = BoundaryNorm(bounds, cmap.N)

im = ax5.imshow(pvals, cmap=cmap, norm=norm, aspect='auto')

ax5.set_xticks(np.arange(len(tests)))
ax5.set_xticklabels(tests, fontweight='bold')
ax5.set_yticks(np.arange(len(datasets_sig)))
ax5.set_yticklabels(datasets_sig, fontweight='bold')

# Annotate cells
for i in range(len(datasets_sig)):
    for j in range(len(tests)):
        p = pvals[i, j]
        sig = 'Yes' if p < 0.05 else 'No'
        color = '#1B5E20' if p < 0.05 else '#B71C1C'
        ax5.text(j, i, f'p={p:.4f}\n({sig})',
                 ha='center', va='center', fontsize=8.5,
                 fontweight='bold', color=color)

ax5.set_title('Statistical Significance Matrix (Balanced Mode, vs. Dense Baseline)',
              fontweight='bold', pad=12)

# Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="4%", pad=0.15)
cb = plt.colorbar(im, cax=cax, ticks=[0.005, 0.025, 0.075, 0.5])
cb.ax.set_yticklabels(['p<0.01', 'p<0.05', 'p<0.10', 'NS'], fontsize=8)

fig5.tight_layout()
fig5.savefig(f'{OUTDIR}/significance_heatmap.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/significance_heatmap.png")
plt.close(fig5)


# ============================================================
#  FIGURE 6 -- Hybrid Activation Rate (Action Distribution)
# ============================================================

datasets_act = ['SciFact', 'FiQA', 'NFCorpus', 'ArguAna']
har_lite =        [0.00, 0.00, 0.00, 0.00]
har_balanced =    [0.61, 0.66, 0.75, 0.40]
har_high_recall = [0.68, 0.58, 0.69, 0.46]

fig6, ax6 = plt.subplots(figsize=(8, 4.5))

x6 = np.arange(len(datasets_act))
w6 = 0.25

bars_l = ax6.bar(x6 - w6, har_lite, w6, label='Lite (conservative)',
                 color='#E0E0E0', edgecolor='white', linewidth=0.8)
bars_b = ax6.bar(x6, har_balanced, w6, label='Balanced',
                 color='#1976D2', edgecolor='white', linewidth=0.8)
bars_h = ax6.bar(x6 + w6, har_high_recall, w6, label='High-Recall',
                 color='#FF6F00', edgecolor='white', linewidth=0.8)

# Annotate balanced and high_recall
for bar in bars_b:
    h = bar.get_height()
    if h > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f'{h:.0%}', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
for bar in bars_h:
    h = bar.get_height()
    if h > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f'{h:.0%}', ha='center', fontsize=9, fontweight='bold', color='#E65100')

# Feasibility band
ax6.axhspan(0.20, 0.80, alpha=0.06, color='#4CAF50')
ax6.axhline(y=0.20, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=1)
ax6.axhline(y=0.80, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=1)
ax6.text(3.6, 0.82, 'Feasibility\nBand', fontsize=8, color='#2E7D32',
         fontstyle='italic', fontweight='bold')

# ArguAna annotation
ax6.annotate('ArguAna: Router\nsuppresses escalation',
             xy=(3, 0.40), xytext=(2.2, 0.15),
             fontsize=8.5, color='#D32F2F', fontstyle='italic',
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

ax6.set_xticks(x6)
ax6.set_xticklabels(datasets_act, fontweight='bold')
ax6.set_ylabel('Hybrid Activation Rate (HAR)', fontweight='bold')
ax6.set_title('Cross-Encoder Invocation Frequency Across Datasets and Modes',
              fontweight='bold', pad=12)
ax6.set_ylim(0, 0.95)
ax6.legend(loc='upper left', framealpha=0.9, edgecolor='lightgray')
ax6.grid(axis='y', alpha=0.15)

fig6.tight_layout()
fig6.savefig(f'{OUTDIR}/action_distribution.png', facecolor='white')
print(f"[OK] Saved {OUTDIR}/action_distribution.png")
plt.close(fig6)

print("\n[OK] All 4 additional publication figures generated successfully.")
