#!/usr/bin/env python3
"""Animate the iLQR planar quadrotor flying through a sequence of targets in one GIF.

The `quadrotor` example solves one flight per target (chaining each solve from the previous) and
writes `quadrotor_targets.csv` plus one `quadrotor_trajectory_{i}.csv` per target. The scripts
concatenates the segments; the active target is highlighted (green) while the others stay dim.

State: x0 = x, x1 = y, x2 = theta, x3..x5 = velocities. Controls u0 = F1, u1 = F2 (rotor thrusts).
y-up, theta-clockwise-positive convention: body "up" (thrust) axis = (sin th, cos th), arm axis
(toward the right rotor at th=0) = (cos th, -sin th). Thrust arrows are scaled by each rotor's force.

Usage:
    python examples/quadrotor/viz/animate_quadrotor.py                     # interactive window
    python examples/quadrotor/viz/animate_quadrotor.py --save quad.gif     # write into output/
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# examples/viz
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "viz"))
import _common  # noqa: E402

ROTOR_RADIUS_FRAC = 0.55  # rotor pod radius, as a fraction of the body radius
# half-length of the edge-on propeller, as a fraction of the body radius
PROP_HALF_FRAC = 0.9
THRUST_ARROW_FRAC = 0.9   # largest thrust arrow, as a fraction of the arm length


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default=_common.in_output("quadrotor_targets.csv"),
                        help="targets CSV written by the quadrotor example")
    parser.add_argument("--traj-prefix", default=_common.in_output("quadrotor_trajectory_"),
                        help="per-target trajectory CSV prefix; files are <prefix><i>.csv")
    parser.add_argument("--l", type=float, default=1.0,
                        help="rotor arm half-length (default: 1.0)")
    parser.add_argument("--body-radius", type=float,
                        default=0.25, help="drawn body-circle radius")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="timestep, for the on-screen clock")
    _common.add_animation_args(parser)
    args = parser.parse_args()

    targets = _common.load_targets(args.targets)
    l = args.l
    body_r = args.body_radius
    rotor_r = ROTOR_RADIUS_FRAC * body_r
    prop_half = PROP_HALF_FRAC * body_r

    columns, seg = _common.load_segments(args.traj_prefix, len(targets),
                                         ["x0", "x1", "x2", "u0", "u1"])
    x, y, th = columns["x0"], columns["x1"], columns["x2"]
    u0 = _common.fill_nan_prev(columns["u0"])  # right rotor (F1)
    u1 = _common.fill_nan_prev(columns["u1"])  # left rotor (F2)
    n_frames = x.size

    # Body axes: thrust/up = (sin th, cos th), arm = (cos th, -sin th).
    up_x, up_y = np.sin(th), np.cos(th)
    arm_x, arm_y = np.cos(th), -np.sin(th)
    right_x, right_y = x + l * arm_x, y + l * \
        arm_y  # right rotor (force u0 = F1)
    left_x, left_y = x - l * arm_x, y - l * \
        arm_y    # left rotor  (force u1 = F2)

    u_max = float(np.nanmax([np.abs(u0).max(), np.abs(u1).max()])) or 1.0
    tscale = THRUST_ARROW_FRAC * l / u_max

    tx = [t[0] for t in targets]
    ty = [t[1] for t in targets]
    pad = l + THRUST_ARROW_FRAC * l + 0.5
    all_x = np.concatenate([x, np.array(tx)])
    all_y = np.concatenate([y, np.array(ty)])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("iLQR planar quadrotor reaching a sequence of waypoints")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # All targets dim (numbered from 1); the active one is drawn brighter and moves with the sequence.
    ax.plot(tx, ty, "*", color=_common.DIM_GRAY, ms=13, zorder=1)
    for i, (px, py) in enumerate(targets):
        ax.annotate(str(i + 1), (px, py), textcoords="offset points", xytext=(8, 6),
                    color="0.5", fontsize=9)
    (active_target,) = ax.plot([], [], "*",
                               color=_common.TARGET_GREEN, ms=20, zorder=2)

    (trace,) = ax.plot([], [], "-", color=_common.BODY_BLUE, alpha=0.3, lw=1, zorder=3)
    (arm,) = ax.plot([], [], "-", color=_common.ROD_GRAY,
                     lw=4, solid_capstyle="round", zorder=4)
    (prop_l,) = ax.plot([], [], "-", color=_common.PROP_GRAY,
                        lw=3, solid_capstyle="round", zorder=5)
    (prop_r,) = ax.plot([], [], "-", color=_common.PROP_GRAY,
                        lw=3, solid_capstyle="round", zorder=5)
    body = Circle((0, 0), body_r, facecolor=_common.BODY_BLUE, edgecolor=_common.BODY_EDGE, lw=1.5,
                  zorder=6)
    rotor_l = Circle((0, 0), rotor_r, facecolor=_common.ROTOR_BLACK, zorder=6)
    rotor_r_ = Circle((0, 0), rotor_r, facecolor=_common.ROTOR_BLACK, zorder=6)
    for patch in (body, rotor_l, rotor_r_):
        ax.add_patch(patch)

    thrust = ax.quiver([0, 0], [0, 0], [0, 0], [0, 0], color=_common.THRUST_ORANGE, angles="xy",
                       scale_units="xy", scale=1.0, width=0.008, zorder=7)

    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    animated = [trace, arm, prop_l, prop_r, body,
                rotor_l, rotor_r_, thrust, active_target, label]

    def init():
        trace.set_data([], [])
        for line in (arm, prop_l, prop_r):
            line.set_data([], [])
        label.set_text("")
        return animated

    def update(i):
        cx, cy = x[i], y[i]
        rx, ry = right_x[i], right_y[i]
        lx, ly = left_x[i], left_y[i]

        arm.set_data([lx, rx], [ly, ry])
        prop_r.set_data([rx - prop_half * arm_x[i], rx + prop_half * arm_x[i]],
                        [ry - prop_half * arm_y[i], ry + prop_half * arm_y[i]])
        prop_l.set_data([lx - prop_half * arm_x[i], lx + prop_half * arm_x[i]],
                        [ly - prop_half * arm_y[i], ly + prop_half * arm_y[i]])
        body.center = (cx, cy)
        rotor_r_.center = (rx, ry)
        rotor_l.center = (lx, ly)

        thrust.set_offsets(np.array([[rx, ry], [lx, ly]]))
        thrust.set_UVC(np.array([u0[i] * tscale * up_x[i], u1[i] * tscale * up_x[i]]),
                       np.array([u0[i] * tscale * up_y[i], u1[i] * tscale * up_y[i]]))

        trace.set_data(x[: i + 1], y[: i + 1])
        active_target.set_data([tx[seg[i]]], [ty[seg[i]]])
        label.set_text(f"target {seg[i] + 1}  ({tx[seg[i]]:.2f}, {ty[seg[i]]:.2f})   "
                       f"t = {i * args.dt:.2f} s")
        return animated

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=1000.0 * args.dt,
                         blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
