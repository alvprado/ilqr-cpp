#!/usr/bin/env python3
"""Animate the control channels of an iLQR trajectory: each signal draws itself as time advances,
with a marker riding the leading edge.

Made for showing control limits: pass --lower/--upper and the forbidden regions are shaded behind
the traces, with the marker taking the bound's colour on the steps where a channel sits on one.
Reads one or more trajectory CSVs written by `ilqr::write_trajectory_csv` (columns k,t,x*,u*);
several files are concatenated onto one clock, which is what the multi-segment examples write.

`build_axes` is reused by examples/quadrotor/viz/animate_quadrotor_with_controls.py.

Usage:
    python examples/viz/animate_controls.py --traj output/cartpole_trajectory.csv
    python examples/viz/animate_controls.py --traj output/quadrotor_trajectory_{0,1,2,3}.csv \
        --lower 0 --upper 50 --names '$F_1$,$F_2$' --ylabel "rotor thrust [N]" --save thrust.gif
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import _common

# The traces, in channel order. Reserves ACCENT_RED for the bounds and for saturation.
TRACE_COLORS = [_common.BODY_BLUE, _common.THRUST_ORANGE,
                _common.TARGET_GREEN, _common.ROTOR_BLACK]
BOUND_COLOR = _common.ACCENT_RED
# A control is "on its bound" within this fraction of the plotted range. Absolute tolerances are
# no good here: a component the QP left free can settle a whisker inside the bound.
SATURATION_TOL_FRAC = 1e-3
LABEL_GAP_FRAC = 0.045


def load_controls(paths):
    """Concatenate the control channels of several trajectory CSVs onto one continuous clock."""
    times, channels, names, time_offset = [], None, None, 0.0

    for path in paths:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if names is None:
            names = [c for c in rows[0] if c.startswith("u")]
            channels = {name: [] for name in names}

        segment_times = np.array([float(row["t"]) for row in rows])
        times.append(segment_times + time_offset)
        # A trajectory has N controls and N+1 states, so the terminal row carries no control.
        for name in names:
            column = np.array([float(row[name]) for row in rows])
            channels[name].append(_common.fill_nan_prev(column))
        time_offset += segment_times[-1] + (segment_times[1] - segment_times[0])

    return (np.concatenate(times),
            [np.concatenate(channels[name]) for name in names],
            names)


def build_axes(ax, t, signals, labels, lower=None, upper=None, ylabel="control", title=None,
               show_clock=True):
    """Draw the static scenery on `ax`; return (animated artists, update(frame_index))."""
    # Fixed axes: the whole horizon is visible from the first frame, so the viewer sees the limit
    # the signal is heading for rather than discovering it.
    span = list(signals) + [np.array([b]) for b in (lower, upper) if b is not None]
    low, high = min(s.min() for s in span), max(s.max() for s in span)
    margin = 0.12 * (high - low or 1.0)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(low - margin, high + margin)

    # Shade what the solver may not enter, so the clear band is the feasible one.
    if upper is not None:
        ax.axhspan(upper, high + margin, color=BOUND_COLOR, alpha=0.07, zorder=0)
    if lower is not None:
        ax.axhspan(low - margin, lower, color=BOUND_COLOR, alpha=0.07, zorder=0)
    for bound, side in ((lower, "top"), (upper, "bottom")):
        if bound is None:
            continue
        ax.axhline(bound, color=BOUND_COLOR, ls="--", lw=1.2, zorder=1)
        ax.annotate(f"limit {bound:g}", xy=(t[-1], bound),
                    xytext=(-4, 3 if side == "bottom" else -3), textcoords="offset points",
                    ha="right", va=side, color=BOUND_COLOR, fontsize=9)

    traces, markers, tags = [], [], []
    for index, label in enumerate(labels):
        color = TRACE_COLORS[index % len(TRACE_COLORS)]
        (trace,) = ax.plot([], [], "-", color=color, lw=2.0, zorder=3)
        (marker,) = ax.plot([], [], "o", color=color, markersize=7, markeredgecolor="white",
                            markeredgewidth=1.2, zorder=5)
        traces.append(trace)
        markers.append(marker)
        # Backed by a faint box: a label riding a saturated trace sits on the dashed limit line.
        tags.append(ax.text(0, 0, f"  {label}", color=color, fontsize=10, va="center", zorder=5,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.8)))

    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    # Above the axes, so it never lands on a trace or a label.
    clock = (ax.text(0.0, 1.02, "", transform=ax.transAxes, va="bottom", fontsize=10, color="0.35")
             if show_clock else None)

    axis_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_gap = LABEL_GAP_FRAC * axis_span
    saturation_tol = SATURATION_TOL_FRAC * axis_span

    def saturated(value):
        return ((upper is not None and value >= upper - saturation_tol) or
                (lower is not None and value <= lower + saturation_tol))

    def update(i):
        for index, signal in enumerate(signals):
            traces[index].set_data(t[: i + 1], signal[: i + 1])
            markers[index].set_data([t[i]], [signal[i]])
            # A channel on its bound is the whole point of the plot, so it gets the bound's colour.
            markers[index].set_color(BOUND_COLOR if saturated(signal[i])
                                     else TRACE_COLORS[index % len(TRACE_COLORS)])

        # Nudge labels apart when traces overlap, so neither is unreadable.
        previous = None
        for j in sorted(range(len(signals)), key=lambda j: signals[j][i]):
            y = signals[j][i]
            if previous is not None and y - previous < label_gap:
                y = previous + label_gap
            tags[j].set_position((t[i], y))
            previous = y

        if clock is not None:
            clock.set_text(f"t = {t[i]:.2f} s")

    artists = traces + markers + tags + ([clock] if clock is not None else [])
    return artists, update


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", nargs="+", default=[_common.in_output("cartpole_trajectory.csv")],
                        help="one or more trajectory CSVs, concatenated in the order given")
    parser.add_argument("--lower", type=float, default=None, help="lower control limit to draw")
    parser.add_argument("--upper", type=float, default=None, help="upper control limit to draw")
    parser.add_argument("--names", default=None,
                        help="comma-separated trace labels, defaulting to the CSV column names")
    parser.add_argument("--ylabel", default="control", help="y-axis label, units included")
    parser.add_argument("--title", default=None, help="plot title")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth sample; trims frame count (and GIF size)")
    _common.add_animation_args(parser, default_fps=30)
    args = parser.parse_args()

    t, signals, column_names = load_controls(args.traj)
    if args.stride > 1:
        t, signals = t[:: args.stride], [s[:: args.stride] for s in signals]
    labels = args.names.split(",") if args.names else column_names
    if len(labels) != len(signals):
        raise SystemExit(f"got {len(labels)} names for {len(signals)} control channels")

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=110)
    artists, update = build_axes(ax, t, signals, labels, args.lower, args.upper, args.ylabel,
                                 args.title)
    fig.tight_layout()

    def animate(i):
        update(i)
        return artists

    anim = FuncAnimation(fig, animate, frames=len(t), interval=1000.0 * (t[1] - t[0]), blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
