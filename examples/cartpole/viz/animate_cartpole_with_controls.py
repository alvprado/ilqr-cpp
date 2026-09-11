#!/usr/bin/env python3
"""Animate the cart-pole swing-up and its cart force side by side in one GIF.

Left: the cart and pole swinging up (from animate_cartpole.build_axes).
Right: the cart force drawing itself against the limits (from animate_controls.build_axes), with
the marker turning red on the steps where the actuator is saturated.

Both panels advance on the same frame index, so the flat stretches in the force trace line up with
the pumping motion on the left.

Usage:
    python examples/cartpole/viz/animate_cartpole_with_controls.py --lower -3 --upper 3
    python examples/cartpole/viz/animate_cartpole_with_controls.py --lower -3 --upper 3 \
        --fps 50 --save cartpole_with_force.gif
"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# examples/viz, then this example's own viz directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "viz"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _common  # noqa: E402
import animate_cartpole  # noqa: E402
import animate_controls  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default=_common.in_output("cartpole_trajectory.csv"),
                        help="trajectory CSV written by the cartpole example")
    parser.add_argument("--length", type=float, default=1.0,
                        help="full rod length 2l; rod is uniform, COG at length/2")
    parser.add_argument("--lower", type=float, default=None, help="lower force limit to draw")
    parser.add_argument("--upper", type=float, default=None, help="upper force limit to draw")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth sample; trims frame count (and GIF size)")
    parser.add_argument("--dpi", type=int, default=70, help="figure dpi; drives the GIF size")
    _common.add_animation_args(parser, default_fps=50)
    args = parser.parse_args()

    t, cart_x, theta = _common.read_columns(args.traj, "t", "x0", "x1")
    # A trajectory has N controls and N+1 states, so the terminal row carries no control.
    (force,) = (_common.fill_nan_prev(c) for c in _common.read_columns(args.traj, "u0"))

    if args.stride > 1:
        keep = slice(None, None, args.stride)
        t, cart_x, theta, force = t[keep], cart_x[keep], theta[keep], force[keep]

    fig, (swing_ax, force_ax) = plt.subplots(
        1, 2, figsize=(13, 5.0), dpi=args.dpi, gridspec_kw={"width_ratios": [1.0, 1.1]})

    swing_artists, swing_update = animate_cartpole.build_axes(
        swing_ax, t, cart_x, theta, args.length, title="Swing-up")
    force_artists, force_update = animate_controls.build_axes(
        force_ax, t, [force], [r"$F$"], args.lower, args.upper, ylabel="cart force [N]",
        title="Cart force", show_clock=False)

    fig.tight_layout()
    artists = swing_artists + force_artists

    def animate(i):
        swing_update(i)
        force_update(i)
        return artists

    interval_ms = 1000.0 * (t[1] - t[0]) if len(t) > 1 else 20.0
    anim = FuncAnimation(fig, animate, frames=len(t), interval=interval_ms, blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
