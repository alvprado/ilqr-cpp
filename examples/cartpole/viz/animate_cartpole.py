#!/usr/bin/env python3
"""Animate the iLQR cart-pole swing-up from a trajectory CSV.

Reads the generic trajectory CSV written by the `cartpole` example (columns k,t,x0,x1,x2,x3,u0),
interpreting x0 = cart position and x1 = pole angle, and draws the cart + pole over time.

theta (x1) is measured from the upright: theta = 0 is balanced (pole up), theta = pi is hanging down.
Pole tip = (cart_x - L*sin(theta), L*cos(theta)).

Usage:
    python examples/cartpole/viz/animate_cartpole.py                    # interactive window
    python examples/cartpole/viz/animate_cartpole.py --save cartpole.gif  # write into output/
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "viz"))  # examples/viz
import _common  # noqa: E402

# Cart geometry (drawing only).
CART_WIDTH = 0.3
CART_HEIGHT = 0.2
VIEW_PAD = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", default=_common.in_output("cartpole_trajectory.csv"),
                        help="trajectory CSV written by the cartpole example")
    parser.add_argument("--length", type=float, default=1.0,
                        help="full rod length 2l (default: 1.0); rod is uniform, COG at length/2")
    _common.add_animation_args(parser, default_fps=50)
    args = parser.parse_args()

    t, cart_x, theta = _common.read_columns(args.traj, "t", "x0", "x1")
    length = args.length

    # Rod spans pivot -> tip (length 2l); theta measured from upright.
    tip_x = cart_x - length * np.sin(theta)
    tip_y = length * np.cos(theta)

    xmin = min(cart_x.min(), tip_x.min()) - VIEW_PAD
    xmax = max(cart_x.max(), tip_x.max()) + VIEW_PAD

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-length - 0.4, length + 0.4)
    ax.set_aspect("equal")
    ax.axhline(0.0, color=_common.TRACK_GRAY, lw=1)  # the track
    ax.set_title("iLQR cart-pole swing-up")
    ax.set_xlabel("x [m]")

    cart = Rectangle((0, 0), CART_WIDTH, CART_HEIGHT, fc=_common.BODY_BLUE, ec="black", zorder=3)
    ax.add_patch(cart)
    (rod,) = ax.plot([], [], "-", color=_common.ACCENT_RED, lw=6, solid_capstyle="round", zorder=2)
    (pivot,) = ax.plot([], [], "o", color="black", ms=4, zorder=4)
    (trace,) = ax.plot([], [], "-", color=_common.ACCENT_RED, alpha=_common.TRACE_ALPHA, lw=1,
                       zorder=1)
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)

    def init():
        cart.set_xy((cart_x[0] - CART_WIDTH / 2, -CART_HEIGHT / 2))
        rod.set_data([], [])
        pivot.set_data([], [])
        trace.set_data([], [])
        time_text.set_text("")
        return cart, rod, pivot, trace, time_text

    def update(i):
        cart.set_xy((cart_x[i] - CART_WIDTH / 2, -CART_HEIGHT / 2))
        rod.set_data([cart_x[i], tip_x[i]], [0.0, tip_y[i]])
        pivot.set_data([cart_x[i]], [0.0])
        trace.set_data(tip_x[: i + 1], tip_y[: i + 1])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return cart, rod, pivot, trace, time_text

    interval_ms = 1000.0 * (t[1] - t[0]) if len(t) > 1 else 20.0
    anim = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=interval_ms, blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
