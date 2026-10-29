"""Plot the verified 9,000-clip results without rerunning model evaluation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
with (ROOT / 'Performance_Test_Unseen_9000_20261029/metrics.csv').open() as f:
    rows = {r['model']: r for r in csv.DictReader(f) if r['dataset'] == 'all'}
models = ['SVM65', 'Frozen Whisper (512)', 'PCA32+65', '512+65']
labels = ['Acoustic SVM', 'Whisper SVM', 'PCA Fusion SVM', 'Fusion SVM']
colours = ['#8796A4', '#254B66', '#C47D21', '#16818A']
ba = [float(rows[n]['balanced_accuracy']) * 100 for n in models]
errors = [int(rows[n]['errors']) for n in models]
low = [v - float(rows[n]['balanced_accuracy_ci95_low']) * 100 for n, v in zip(models, ba)]
high = [float(rows[n]['balanced_accuracy_ci95_high']) * 100 - v for n, v in zip(models, ba)]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 3.5), sharey=True, gridspec_kw={'width_ratios': [1.45, 1]})
y = range(4)
left.barh(y, ba, height=.52, color=colours, xerr=[low, high], error_kw={'ecolor': '#263746', 'elinewidth': 1.25, 'capsize': 3})
left.set(yticks=list(y), yticklabels=labels, xlim=(0, 117), xticks=[0, 25, 50, 75, 100], xlabel='Balanced accuracy (%)', title='A. Balanced accuracy — higher is better')
for i, v in enumerate(ba):
    left.text(102, i, f'{v:.3f}%', va='center', fontsize=11, fontweight='bold' if i == 3 else 'normal')
right.barh(y, errors, height=.52, color=colours)
right.set(xlim=(0, 300), xticks=[0, 100, 200, 300], xlabel='Errors out of 9,000 clips', title='B. Errors — lower is better')
for i, v in enumerate(errors):
    right.text(v + 7, i, str(v), va='center', fontsize=11, fontweight='bold' if i == 3 else 'normal')
left.invert_yaxis()
for ax in (left, right):
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5E9EC', linewidth=.7)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#BCC5CD')
fig.subplots_adjust(left=.17, right=.98, bottom=.19, top=.85, wspace=.24)
fig.savefig(HERE / 'fig_performance_bars.png', dpi=200, bbox_inches='tight', pad_inches=.12)
plt.close(fig)
