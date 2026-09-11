"""Shared helpers for the iLQR example visualizations.

CSV loading, target/segment loading, a shared color palette, argparse helpers, and a save/show
finalizer. Generated images go into OUTPUT_DIR — the same directory the C++ examples write their CSVs
to — so everything for a run stays together.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Directory holding generated artifacts (CSVs from the examples, GIFs/PNGs from these scripts).
OUTPUT_DIR = "output"

# Color palette shared across the animations and plots.
BODY_BLUE = "#3b7dd8"      # cart / arm link 1 / quadrotor body
ACCENT_RED = "#d84b3b"     # pole / arm link 2 / end-effector / traces
TARGET_GREEN = "#2ca02c"   # active target marker
ROD_GRAY = "#b0b0b0"       # quadrotor arm
ROTOR_BLACK = "#141414"    # quadrotor rotor pods
PROP_GRAY = "#555555"      # quadrotor propellers
THRUST_ORANGE = "#e67e22"  # quadrotor thrust arrows
BODY_EDGE = "#2c3e50"      # outlines
TRACK_GRAY = "0.7"         # cart track / ground
DIM_GRAY = "0.75"          # inactive target markers
TRACE_ALPHA = 0.25         # end-effector / flight-path trace


def in_output(filename):
    """Path to a file inside OUTPUT_DIR (the default location the examples read/write)."""
    return os.path.join(OUTPUT_DIR, filename)


def read_columns(path, *names):
    """Read the named CSV columns as float np.arrays, in the requested order."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return tuple(np.array([float(row[name]) for row in rows]) for name in names)


def fill_nan_prev(a):
    """Forward-fill NaNs with the previous finite value (control CSVs have a NaN terminal row)."""
    out = a.copy()
    last = 0.0
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def load_targets(path):
    """Load a targets CSV (idx,target_x,target_y) as a list of (x, y) tuples."""
    xs, ys = read_columns(path, "target_x", "target_y")
    return list(zip(xs, ys))


def load_segments(prefix, n_segments, cols):
    """Concatenate per-target trajectory CSVs (`<prefix><i>.csv`) into single arrays.

    Returns (columns, seg): `columns` is a {name: np.array} dict over `cols`, and `seg[k]` is the
    target index that frame k belongs to.
    """
    parts = {c: [] for c in cols}
    seg_parts = []
    for i in range(n_segments):
        arrays = read_columns(f"{prefix}{i}.csv", *cols)
        for c, a in zip(cols, arrays):
            parts[c].append(a)
        seg_parts.append(np.full(arrays[0].size, i, dtype=int))
    columns = {c: np.concatenate(parts[c]) for c in cols}
    return columns, np.concatenate(seg_parts)


def add_save_arg(parser):
    parser.add_argument("--save", default=None,
                        help=f"output file; a bare name is written into {OUTPUT_DIR}/")


def add_animation_args(parser, default_fps=25):
    add_save_arg(parser)
    parser.add_argument("--fps", type=int, default=default_fps,
                        help="frames per second when saving")


def match_axes_width(reference_ax, target_ax):
    """Make target_ax span exactly the drawn width of reference_ax.

    An axes with set_aspect("equal") shrinks its own box to honour the aspect, so it ends up
    narrower than the slot the layout gave it. Call after tight_layout() and a draw.
    """
    reference_ax.figure.canvas.draw()
    reference = reference_ax.get_position(original=False)
    target = target_ax.get_position(original=False)
    target_ax.set_position([reference.x0, target.y0, reference.width, target.height])


def fit_stacked_layout(fig, aspect_ax, plot_ax, plot_height_in):
    """Grow the figure so an aspect-equal panel fills its width with no wasted space.

    `aspect_ax` keeps equal x/y scaling, so its drawn box is pinned to the data's own aspect. Give
    the time series a height in inches and the panel's height follows, rather than guessing a ratio
    that has to be retuned whenever the trajectory bounds move.
    """
    # First pass: measure the margins the labels and titles actually take, in inches.
    fig.tight_layout()
    fig.canvas.draw()
    figure_width, figure_height = fig.get_size_inches()
    panel_slot = aspect_ax.get_position(original=True)
    plot_slot = plot_ax.get_position(original=True)
    side_margins = figure_width * (1.0 - panel_slot.width)
    vertical_overhead = figure_height * (1.0 - panel_slot.height - plot_slot.height)

    (x_low, x_high), (y_low, y_high) = aspect_ax.get_xlim(), aspect_ax.get_ylim()
    data_aspect = (x_high - x_low) / (y_high - y_low)
    panel_height_in = (figure_width - side_margins) / data_aspect

    fig.axes[0].get_gridspec().set_height_ratios([panel_height_in, plot_height_in])
    fig.set_size_inches(figure_width, panel_height_in + plot_height_in + vertical_overhead)
    fig.tight_layout()
    match_axes_width(aspect_ax, plot_ax)


def finalize(obj, args):
    """Save (into OUTPUT_DIR) or interactively show a Figure or FuncAnimation."""
    if not args.save:
        plt.show()
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = args.save if os.path.dirname(args.save) else in_output(args.save)
    if isinstance(obj, FuncAnimation):
        obj.save(out, writer="pillow", fps=getattr(args, "fps", 25))
    else:
        obj.savefig(out, dpi=130)
    print(f"saved {out}")
