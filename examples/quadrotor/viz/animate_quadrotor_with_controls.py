#!/usr/bin/env python3
"""Animate the quadrotor flight and its rotor thrusts side by side in one GIF.

Left: the quadrotor flying its waypoint sequence (from animate_quadrotor.build_axes).
Right: each rotor thrust drawing itself against the limits (from animate_controls.build_axes),
with the marker turning red on the steps where a rotor is saturated.

Both panels advance on the same frame index, so the flat tops in the thrust trace line up with
what the vehicle is doing on the left.

Usage:
    python examples/quadrotor/viz/animate_quadrotor_with_controls.py --upper 50 --lower 0
    python examples/quadrotor/viz/animate_quadrotor_with_controls.py --upper 50 --lower 0 \
        --stride 3 --save quadrotor_with_thrusts.gif
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# examples/viz, then this example's own viz directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "viz"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _common  # noqa: E402
import animate_controls  # noqa: E402
import animate_quadrotor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default=_common.in_output("quadrotor_targets.csv"),
                        help="targets CSV written by the quadrotor example")
    parser.add_argument("--traj-prefix", default=_common.in_output("quadrotor_trajectory_"),
                        help="per-target trajectory CSV prefix; files are <prefix><i>.csv")
    parser.add_argument("--l", type=float, default=1.0, help="rotor arm half-length")
    parser.add_argument("--body-radius", type=float, default=0.25, help="drawn body-circle radius")
    parser.add_argument("--dt", type=float, default=0.02, help="timestep, for the clock")
    parser.add_argument("--lower", type=float, default=None, help="lower thrust limit to draw")
    parser.add_argument("--upper", type=float, default=None, help="upper thrust limit to draw")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth sample; trims frame count (and GIF size)")
    parser.add_argument("--dpi", type=int, default=70, help="figure dpi; drives the GIF size")
    _common.add_animation_args(parser, default_fps=25)
    args = parser.parse_args()

    targets = _common.load_targets(args.targets)
    columns, seg = _common.load_segments(args.traj_prefix, len(targets),
                                         ["x0", "x1", "x2", "u0", "u1"])

    # One clock for both panels: the flight frames are already concatenated, so time is just the
    # frame index scaled by dt.
    n_frames = columns["x0"].size
    t = np.arange(n_frames) * args.dt
    thrusts = [_common.fill_nan_prev(columns["u0"]), _common.fill_nan_prev(columns["u1"])]

    if args.stride > 1:
        keep = slice(None, None, args.stride)
        columns = {name: values[keep] for name, values in columns.items()}
        seg, t = seg[keep], t[keep]
        thrusts = [u[keep] for u in thrusts]
        n_frames = t.size

    fig, (flight_ax, thrust_ax) = plt.subplots(
        1, 2, figsize=(13, 5.6), dpi=args.dpi, gridspec_kw={"width_ratios": [1.0, 1.15]})

    flight_artists, flight_update = animate_quadrotor.build_axes(
        flight_ax, columns, seg, targets, args.l, args.body_radius, args.dt * args.stride,
        title="Flight")
    thrust_artists, thrust_update = animate_controls.build_axes(
        thrust_ax, t, thrusts, [r"$F_1$", r"$F_2$"], args.lower, args.upper,
        ylabel="rotor thrust [N]", title="Rotor thrusts", show_clock=False)

    fig.tight_layout()
    artists = flight_artists + thrust_artists

    def animate(i):
        flight_update(i)
        thrust_update(i)
        return artists

    anim = FuncAnimation(fig, animate, frames=n_frames,
                         interval=1000.0 * args.dt * args.stride, blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
