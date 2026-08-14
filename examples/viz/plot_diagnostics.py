#!/usr/bin/env python3
"""Plot iLQR solver diagnostics from a CSV: total cost and gradient norm per iteration.

Reads a diagnostics CSV (columns: iter,cost,reg,grad_norm,step_size,accepted) written by an example.
Both quantities are shown on a log y-axis; iterations whose forward pass was rejected (no accepted
step) are marked in red.

Usage:
    python examples/viz/plot_diagnostics.py                          # interactive window
    python examples/viz/plot_diagnostics.py --save diagnostics.png   # write into output/
"""
import argparse

import matplotlib.pyplot as plt

import _common


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diag", default=_common.in_output("cartpole_diagnostics.csv"),
                        help="diagnostics CSV written by an example")
    _common.add_save_arg(parser)
    args = parser.parse_args()

    it, cost, grad_norm, accepted = _common.read_columns(args.diag, "iter", "cost", "grad_norm",
                                                         "accepted")
    rejected = accepted < 0.5

    fig, (ax_cost, ax_grad) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_cost.semilogy(it, cost, "-", color=_common.BODY_BLUE)
    if rejected.any():
        ax_cost.scatter(it[rejected], cost[rejected], color="red", s=20, zorder=3,
                        label="rejected step")
        ax_cost.legend()
    ax_cost.set_ylabel("total cost")
    ax_cost.set_title("iLQR convergence")
    ax_cost.grid(True, which="both", alpha=0.3)

    ax_grad.semilogy(it, grad_norm, "-", color=_common.ACCENT_RED)
    ax_grad.set_ylabel("gradient norm")
    ax_grad.set_xlabel("iteration")
    ax_grad.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _common.finalize(fig, args)


if __name__ == "__main__":
    main()
