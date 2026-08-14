#!/usr/bin/env python3
"""Animate the iLQR 2-link planar arm reaching a sequence of targets in one GIF.

The `robotic_arm` example solves one reach per target (chaining each solve from the previous) and
writes `robotic_arm_targets.csv` plus one `robotic_arm_trajectory_{i}.csv` per target. This script
concatenates the segments into one continuous animation, highlighting the active target (green) while
the others stay dim.

State: x0 = q1 (shoulder angle), x1 = q2 (elbow angle, relative), measured from the horizontal:
    shoulder = (0, 0); elbow = (l1 cos q1, l1 sin q1); ee = elbow + (l2 cos(q1+q2), l2 sin(q1+q2)).

Usage:
    python examples/robotic_arm/viz/animate_robotic_arm.py                     # interactive window
    python examples/robotic_arm/viz/animate_robotic_arm.py --save arm.gif      # write into output/
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# examples/viz
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "viz"))
import _common  # noqa: E402

VIEW_PAD = 0.3
DIM_TARGET_SIZE = 13
ACTIVE_TARGET_SIZE = 20
LINK_WIDTH = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default=_common.in_output("robotic_arm_targets.csv"),
                        help="targets CSV written by the robotic_arm example")
    parser.add_argument("--traj-prefix", default=_common.in_output("robotic_arm_trajectory_"),
                        help="per-target trajectory CSV prefix; files are <prefix><i>.csv")
    parser.add_argument("--l1", type=float, default=1.0,
                        help="length of link 1 (default: 1.0)")
    parser.add_argument("--l2", type=float, default=0.7,
                        help="length of link 2 (default: 0.7)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="timestep, for the on-screen clock")
    _common.add_animation_args(parser)
    args = parser.parse_args()

    targets = _common.load_targets(args.targets)
    l1, l2 = args.l1, args.l2

    columns, seg = _common.load_segments(
        args.traj_prefix, len(targets), ["x0", "x1"])
    q1, q2 = columns["x0"], columns["x1"]
    n_frames = q1.size

    # Forward kinematics over the whole concatenated trajectory.
    elbow_x = l1 * np.cos(q1)
    elbow_y = l1 * np.sin(q1)
    ee_x = elbow_x + l2 * np.cos(q1 + q2)
    ee_y = elbow_y + l2 * np.sin(q1 + q2)

    reach = l1 + l2
    tx = [t[0] for t in targets]
    ty = [t[1] for t in targets]
    lim = max([reach] + [abs(v) for v in tx + ty]) + VIEW_PAD

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("iLQR 2-link robotic arm reaching a sequence of targets")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # All targets dim (numbered from 1); the active one is drawn brighter and moves with the sequence.
    ax.plot(tx, ty, "*", color=_common.DIM_GRAY, ms=DIM_TARGET_SIZE, zorder=1)
    for i, (x, y) in enumerate(targets):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(8, 6),
                    color="0.5", fontsize=9)
    (active_target,) = ax.plot([], [], "*", color=_common.TARGET_GREEN, ms=ACTIVE_TARGET_SIZE,
                               zorder=3)

    (link1,) = ax.plot([], [], "-", color=_common.BODY_BLUE, lw=LINK_WIDTH, solid_capstyle="round",
                       zorder=4)
    (link2,) = ax.plot([], [], "-", color=_common.ACCENT_RED, lw=LINK_WIDTH, solid_capstyle="round",
                       zorder=4)
    (joints,) = ax.plot([], [], "o", color="black",
                        ms=6, zorder=5)  # shoulder + elbow
    (ee,) = ax.plot([], [], "o", color=_common.ACCENT_RED,
                    ms=8, zorder=6)  # end-effector
    (trace,) = ax.plot([], [], "-", color=_common.ACCENT_RED, alpha=_common.TRACE_ALPHA, lw=1,
                       zorder=2)
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes)

    def init():
        for artist in (link1, link2, joints, ee, trace, active_target):
            artist.set_data([], [])
        label.set_text("")
        return link1, link2, joints, ee, trace, active_target, label

    def update(i):
        link1.set_data([0.0, elbow_x[i]], [0.0, elbow_y[i]])
        link2.set_data([elbow_x[i], ee_x[i]], [elbow_y[i], ee_y[i]])
        joints.set_data([0.0, elbow_x[i]], [0.0, elbow_y[i]])
        ee.set_data([ee_x[i]], [ee_y[i]])
        trace.set_data(ee_x[: i + 1], ee_y[: i + 1])
        active_target.set_data([tx[seg[i]]], [ty[seg[i]]])
        label.set_text(
            f"target {seg[i] + 1}  ({tx[seg[i]]:.2f}, {ty[seg[i]]:.2f})   t = {i * args.dt:.2f} s")
        return link1, link2, joints, ee, trace, active_target, label

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=1000.0 * args.dt,
                         blit=True)
    _common.finalize(anim, args)


if __name__ == "__main__":
    main()
