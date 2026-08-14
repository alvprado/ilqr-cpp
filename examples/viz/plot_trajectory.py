#!/usr/bin/env python3
"""Plot any iLQR trajectory CSV: every state and control channel versus time.

Reads a generic trajectory CSV written by `ilqr::write_trajectory_csv` (columns k,t,x0..x{n-1},
u0..u{m-1}). It auto-detects the state (x*) and control (u*) columns, so it works for any system.
States are drawn in the top block, controls in the bottom block. With --animate, a vertical time
cursor sweeps across all subplots together.

Usage:
    python examples/viz/plot_trajectory.py                              # static subplots
    python examples/viz/plot_trajectory.py --animate                    # sweeping time cursor
    python examples/viz/plot_trajectory.py --traj output/foo.csv --save fig.png
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import _common


def load_trajectory(path):
    # Auto-detect channels by column name: state channels start with 'x', controls with 'u'.
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    def column(name):
        return np.array([float(r[name]) for r in rows])

    t = column("t")
    states = {c: column(c) for c in fieldnames if c.startswith("x")}
    controls = {c: column(c) for c in fieldnames if c.startswith("u")}
    return t, states, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default=_common.in_output("cartpole_trajectory.csv"),
                        help="trajectory CSV to plot")
    parser.add_argument("--animate", action="store_true",
                        help="sweep a vertical time cursor across the subplots")
    _common.add_animation_args(parser, default_fps=50)
    args = parser.parse_args()

    t, states, controls = load_trajectory(args.traj)
    channels = list(states.items()) + list(controls.items())
    n = len(channels)

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8, 1.6 * n + 1))
    if n == 1:
        axes = [axes]

    cursors = []
    for ax, (name, values) in zip(axes, channels):
        color = _common.BODY_BLUE if name.startswith("x") else _common.ACCENT_RED
        ax.plot(t, values, "-", color=color)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if args.animate:
            cursors.append(ax.axvline(t[0], color="0.4", lw=1))
    axes[0].set_title("iLQR trajectory")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()

    if args.animate:
        def update(i):
            for cursor in cursors:
                cursor.set_xdata([t[i], t[i]])
            return cursors

        interval_ms = 1000.0 * (t[1] - t[0]) if len(t) > 1 else 20.0
        anim = FuncAnimation(fig, update, frames=len(t), interval=interval_ms, blit=True)
        _common.finalize(anim, args)
    else:
        _common.finalize(fig, args)


if __name__ == "__main__":
    main()
