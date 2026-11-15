"""Presentation defaults only; no effect on model evaluation or caches."""
import matplotlib.pyplot as plt

def apply_style():
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.titlesize': 13, 'lines.linewidth': 2.1, 'lines.markersize': 4,
        'axes.axisbelow': True, 'grid.alpha': .15, 'grid.linewidth': .6,
        'legend.framealpha': .95, 'legend.handlelength': 2.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

PAIR_SIZE = (8., 2.15)
SINGLE_SIZE = (6.4, 2.2)
PANEL_SIZE = (3.9, 2.1)
BAND_ALPHA = .25

def context_ticks(ax, ns):
    values = list(ns)
    # Sparse labels keep half-page panels readable; all data points are retained.
    shown = values if len(values) <= 5 else values[::2]
    if values[-1] not in shown: shown.append(values[-1])
    ax.set_xticks(shown, [str(n) for n in shown])

def estimator_label(method):
    return (method.replace(' (train n=128)', '').replace(' train n=128', '')
            .replace('Parameter transformer + solution', 'Parameter + solution')
            .replace('Prediction transformer', 'Direct prediction')
            .replace('Joint parameter + solution', 'Joint parameter + solution')
            .replace('Joint parameter', 'Joint parameter'))
