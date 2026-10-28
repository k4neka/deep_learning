"""
Picture of every map type (one row per difficulty level you choose), after some fire steps.

Usage:
    python maps_gallery.py --difficulty medium --steps 120 --out gallery.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from world import World, MAP_TYPES, TERRAIN_NAMES

TERRAIN_COLORS = ["#5b9bd5", "#9e9e9e", "#b9ad94", "#d6e8a8", "#3f7d3b", "#7b2cbf", "#e9c46a"]
FIRE_CMAP = LinearSegmentedColormap.from_list("fire", ["#ffe066", "#ff9f1c", "#e63946", "#7a0010"])


def hillshade(elev, cell_m=5.0):
    gy, gx = np.gradient(elev, cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def draw_world(ax, w, title):
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in TERRAIN_COLORS])[w.terrain]
    rgb = rgb * (0.65 + 0.35 * hillshade(w.elev, w.cell_m))[..., None]
    ax.imshow(rgb, interpolation="nearest")
    if np.ptp(w.elev) > 5:
        step = 5 if np.ptp(w.elev) < 40 else 10
        ax.contour(w.elev, levels=np.arange(step, w.elev.max(), step), colors="#5c4033",
                   linewidths=0.4, alpha=0.5)
    ax.imshow(np.ma.masked_where(~w.burnt, w.burnt), cmap=matplotlib.colors.ListedColormap(["#2f2f2f"]), alpha=0.85)
    f = w.fire_map()
    ax.imshow(np.ma.masked_where(f <= 0, np.maximum(f, 0.3)), cmap=FIRE_CMAP, vmin=0.3, vmax=1)
    if w.nofly.any():
        ax.contourf(w.nofly.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(w.nofly.astype(float), levels=[0.5], colors="#d00000", linewidths=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="medium", help="one level or a comma list, e.g. easy,extreme")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="gallery.png")
    args = ap.parse_args()
    levels = args.difficulty.split(",")
    fig, axes = plt.subplots(len(levels), len(MAP_TYPES), figsize=(3.1 * len(MAP_TYPES), 3.4 * len(levels) + 0.8),
                             squeeze=False)
    for r, lev in enumerate(levels):
        for c, mt in enumerate(MAP_TYPES):
            w = World(rng=np.random.default_rng(args.seed + c), map_type=mt, difficulty=lev)
            w.reset()
            for _ in range(args.steps):
                w.step()
            draw_world(axes[r, c], w, f"{mt.replace('_', ' ')} ({lev})\nrelief {np.ptp(w.elev):.0f} m")
    handles = [Patch(fc=TERRAIN_COLORS[k], ec="grey", label=TERRAIN_NAMES[k]) for k in range(7)]
    handles += [Patch(fc="#2f2f2f", label="burnt"), Patch(fc="#e63946", label="fire"),
                Patch(fc="white", ec="#d00000", hatch="////", label="no-fly zone")]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=9)
    fig.suptitle(f"Map types after {args.steps} steps (contour lines = elevation)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(args.out, dpi=90)
    print("saved", args.out)


if __name__ == "__main__":
    main()
