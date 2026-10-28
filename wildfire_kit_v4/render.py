"""
Animated output for one episode (v2).

Left:   the real world: terrain, fire, burnt ground, wind arrows, drones and what each one sees
Middle: the firefighters' map: only what the swarm has observed, fading as it gets outdated
Right:  live numbers and curves

Usage:
    python render.py --policy greedy --drones 3 --seed 1000 --out greedy3.mp4
    python render.py --policy ppo --model runs/ppo/model.zip --drones 3 --out ppo3.mp4
    python render.py --policy greedy --drones 3 --out greedy3.gif     # GIF if no ffmpeg
    python render.py --policy greedy --drones 3 --png snapshot.png    # single image (last frame)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no screen needed (works on the cluster)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from env import SwarmCore, VIEW_RADIUS
from world import TERRAIN_NAMES, MAP_TYPES, DIFFICULTY
from maps_gallery import hillshade
from baselines import Greedy, Lawnmower, RandomPolicy
from fire_models import ALL_FIRE_MODELS

TERRAIN_COLORS = ["#5b9bd5", "#9e9e9e", "#b9ad94", "#d6e8a8", "#3f7d3b", "#7b2cbf", "#e9c46a"]
FIRE_CMAP = LinearSegmentedColormap.from_list("fire", ["#ffe066", "#ff9f1c", "#e63946", "#7a0010"])
BURNT_CMAP = ListedColormap(["#2f2f2f"])
DRONE_COLORS = ["#00b4d8", "#ff006e", "#ffbe0b", "#8338ec", "#3a86ff", "#fb5607"]
ALT_NAME = ["low", "medium", "high"]
ALT_MARK = ["v", "o", "^"]
ALT_STYLE = [":", "--", "-"]


def get_policy(name, model_path):
    if name == "ppo":
        from stable_baselines3 import PPO
        from train_ppo import PPOPolicyWrapper, SmallCNN  # noqa: F401  (needed to load the model)
        return PPOPolicyWrapper(PPO.load(model_path, device="cpu"))
    return {"greedy": Greedy(), "lawnmower": Lawnmower(), "random": RandomPolicy()}[name]


def run_episode(args):
    core = SwarmCore(size=args.size, n_drones=args.drones, max_steps=args.steps, record=True,
                     fire_models=args.fire_model)
    core.reset(seed=args.seed, map_type=args.map, difficulty=args.difficulty, fire_model=args.fire_model)
    pol = get_policy(args.policy, args.model)
    pol.reset(core)
    frames, done = [], False
    while True:
        w = core.world
        if core.t % args.every == 0 or done:
            frames.append(dict(
                t=core.t, fire=w.fire_map().copy(), burnt=w.burnt.copy(),
                belief=core.belief.copy(), fresh=core.fresh.copy(),
                pos=core.pos.copy(), alt=core.alt.copy(), alive=core.alive.copy(),
                wx=w.wx.copy(), wy=w.wy.copy(), wind=w.wind_text(), smoke=core.smoke().copy(),
                battery=core.battery.copy(), returning=core.returning.copy(), lost=list(core.lost_reason),
                trails=[list(tr[-30:]) for tr in core.trails], stats=dict(core.history[-1])))
        if done:
            break
        _, done, _ = core.step([pol.act(core, i) for i in range(args.drones)])
    return core, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="greedy", choices=["greedy", "lawnmower", "random", "ppo"])
    ap.add_argument("--model", default="runs/ppo/model.zip")
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--map", default="random", choices=["random"] + MAP_TYPES)
    ap.add_argument("--difficulty", default="medium", choices=list(DIFFICULTY))
    ap.add_argument("--fire-model", default="native", choices=ALL_FIRE_MODELS,
                    help="fire-spread engine (see fire_models/__init__.py)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--every", type=int, default=2, help="draw every N-th step")
    ap.add_argument("--out", default="episode.mp4")
    ap.add_argument("--png", default=None, help="save only the last frame as a PNG")
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    core, frames = run_episode(args)
    w, n = core.world, core.size
    hist = core.history

    # ---------------- static layout
    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.66], height_ratios=[1, 1],
                          left=0.02, right=0.96, top=0.9, bottom=0.17, wspace=0.16, hspace=0.3)
    axW = fig.add_subplot(gs[:, 0])
    axM = fig.add_subplot(gs[:, 1])
    axT = fig.add_subplot(gs[0, 2])
    axP = fig.add_subplot(gs[1, 2])
    for ax in (axW, axM):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    axW.set_title("Real situation (ground truth)", fontsize=12, fontweight="bold")
    axM.set_title("Firefighters' map (what the swarm knows)", fontsize=12, fontweight="bold")
    axT.axis("off")

    rgb = np.array([matplotlib.colors.to_rgb(c) for c in TERRAIN_COLORS])[w.terrain]
    rgb = rgb * (0.65 + 0.35 * hillshade(w.elev, w.cell_m))[..., None]      # relief shading
    faded = 0.45 * rgb + 0.55
    axW.imshow(rgb, interpolation="nearest")
    axM.imshow(faded, interpolation="nearest")
    if np.ptp(w.elev) > 5:
        step = 5 if np.ptp(w.elev) < 40 else 10
        axW.contour(w.elev, levels=np.arange(step, w.elev.max(), step), colors="#5c4033", linewidths=0.4, alpha=0.5)
    for ax in (axW, axM):
        if w.nofly.any():
            ax.contourf(w.nofly.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(w.nofly.astype(float), levels=[0.5], colors="#d00000", linewidths=1.5)
    f0 = frames[0]
    im_burnt = axW.imshow(np.ma.masked_all((n, n)), cmap=BURNT_CMAP, alpha=0.88, interpolation="nearest")
    im_fire = axW.imshow(np.ma.masked_all((n, n)), cmap=FIRE_CMAP, vmin=0.3, vmax=1, interpolation="nearest")
    im_smoke = axW.imshow(np.zeros((n, n)), cmap="Greys", vmin=0, vmax=1, alpha=0.0, interpolation="bilinear")
    im_fog = axM.imshow(np.ones((n, n)), cmap="Greys_r", vmin=0, vmax=1, alpha=0.0, interpolation="nearest")
    im_known = axM.imshow(np.ma.masked_all((n, n)), cmap=FIRE_CMAP, vmin=0.3, vmax=1, interpolation="nearest")

    # wind arrows (environment wind field)
    g = max(8, n // 8)
    yy, xx = np.mgrid[g // 2:n:g, g // 2:n:g]
    quiv = axW.quiver(xx, yy, f0["wx"][g // 2::g, g // 2::g], f0["wy"][g // 2::g, g // 2::g],
                      angles="xy", scale_units="xy", scale=1 / (g * 0.7), color="#0b2545",
                      alpha=0.75, width=0.004, headwidth=4)
    # homes at risk (rings), base, scale bar
    ring_W, = axW.plot([], [], "o", ms=9, mfc="none", mec="#ffd60a", mew=2)
    ring_M, = axM.plot([], [], "o", ms=9, mfc="none", mec="#ffd60a", mew=2)
    for ax in (axW, axM):
        ax.plot(*core.base, marker="*", ms=16, mfc="white", mec="black", zorder=6)
        bar = 100 / w.cell_m
        ax.add_patch(Rectangle((4, n - 7), bar, 1.6, color="black", zorder=6))
        ax.text(4 + bar / 2, n - 8.5, "100 m", ha="center", va="bottom", fontsize=9, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    # drones: marker (shape = altitude), view footprint, trail
    drone_art = []
    for i in range(core.n_drones):
        c = DRONE_COLORS[i % len(DRONE_COLORS)]
        arts = {}
        for key, ax in (("W", axW), ("M", axM)):
            trail, = ax.plot([], [], "-", color=c, lw=1.6, alpha=0.8, zorder=7)
            mark, = ax.plot([], [], "o", ms=11, mfc=c, mec="black", mew=1.5, zorder=9)
            rect = Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=2, zorder=8)
            ax.add_patch(rect)
            arts[key] = (trail, mark, rect)
        drone_art.append(arts)

    info = axT.text(0.0, 1.0, "", va="top", ha="left", fontsize=11, family="monospace",
                    transform=axT.transAxes)

    ts = np.array([h["t"] for h in hist])
    known = np.array([h["known_critical"] for h in hist]) * 100
    burning = np.array([h["burning_ha"] for h in hist])
    axP.plot(ts, known, color="#2a9d8f", lw=2, label="critical fire freshly observed (%)")
    axP.set_ylim(0, 100); axP.set_xlim(0, ts.max()); axP.set_xlabel("step")
    axP.set_ylabel("critical fire known (%)", color="#2a9d8f", fontsize=9)
    axP2 = axP.twinx()
    axP2.plot(ts, burning, color="#e63946", lw=2, label="area burning (ha)")
    axP2.set_ylabel("area burning (ha)", color="#e63946", fontsize=9)
    axP2.set_ylim(0, max(1e-3, burning.max()) * 1.15)
    vline = axP.axvline(0, color="black", lw=1)
    axP.set_title("Over the mission", fontsize=11)

    # legend
    handles = [Patch(fc=TERRAIN_COLORS[k], ec="grey", label=TERRAIN_NAMES[k]) for k in range(7)]
    handles += [Patch(fc="#2f2f2f", label="burnt"),
                Patch(fc="#ffe066", label="fire (weak)"), Patch(fc="#7a0010", label="fire (intense)"),
                Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec="#ffd60a", mew=2, label="homes at risk"),
                Line2D([], [], marker="*", ls="", ms=13, mfc="white", mec="black", label="drone base"),
                Line2D([], [], color="#0b2545", marker=">", lw=1.5, label="wind (from environment)"),
                Patch(fc="#bdbdbd", label="smoke"), Patch(fc="white", ec="#d00000", hatch="////", label="no-fly zone")]
    handles += [Line2D([], [], marker=ALT_MARK[a], ls=ALT_STYLE[a], color="grey", mfc="lightgrey",
                       mec="black", ms=10, lw=2, label=f"drone {ALT_NAME[a]} (view {2 * VIEW_RADIUS[a] + 1} cells)")
                for a in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    title = fig.suptitle("", fontsize=14, fontweight="bold")

    def draw(k):
        fr = frames[k]
        im_burnt.set_data(np.ma.masked_where(~fr["burnt"], fr["burnt"].astype(float)))
        im_fire.set_data(np.ma.masked_where(fr["fire"] <= 0, np.maximum(fr["fire"], 0.3)))
        im_smoke.set_data(np.full((n, n), 0.55))
        im_smoke.set_alpha(np.clip(fr["smoke"] * 0.55, 0, 0.55))
        stale = 1 - fr["fresh"]
        im_fog.set_data(np.ones((n, n)))
        im_fog.set_alpha(np.clip(stale * 0.65, 0, 0.65))
        im_known.set_data(np.ma.masked_where(fr["belief"] <= 0, np.maximum(fr["belief"], 0.3)))
        im_known.set_alpha(np.clip(0.35 + 0.65 * fr["fresh"], 0, 1))
        quiv.set_UVC(fr["wx"][g // 2::g, g // 2::g], fr["wy"][g // 2::g, g // 2::g])
        # homes at risk = within ~20 m of fire (real) / of known fire (map)
        def near(m):
            out = m.copy()
            for _ in range(4):
                out = out | np.roll(out, 1, 0) | np.roll(out, -1, 0) | np.roll(out, 1, 1) | np.roll(out, -1, 1)
            return out
        rw = w.homes & near(fr["fire"] > 0)
        rm = w.homes & near(fr["belief"] > 0)
        ys, xs = np.nonzero(rw[::2, ::2]); ring_W.set_data(xs * 2 + 0.5, ys * 2 + 0.5)
        ys, xs = np.nonzero(rm[::2, ::2]); ring_M.set_data(xs * 2 + 0.5, ys * 2 + 0.5)
        for i, arts in enumerate(drone_art):
            x, y = fr["pos"][i]
            a = int(fr["alt"][i])
            r = VIEW_RADIUS[a]
            tr = np.array(fr["trails"][i])
            for key in ("W", "M"):
                trail, mark, rect = arts[key]
                trail.set_data(tr[:, 0], tr[:, 1])
                mark.set_data([x], [y])
                mark.set_marker(ALT_MARK[a] if fr["alive"][i] else "X")
                rect.set_bounds(x - r - 0.5, y - r - 0.5, 2 * r + 1, 2 * r + 1)
                rect.set_linestyle(ALT_STYLE[a])
                rect.set_visible(bool(fr["alive"][i]))
        s = fr["stats"]
        towards, spd = fr["wind"]
        strength = "light" if spd < 0.4 else "moderate" if spd < 0.7 else "strong"
        lines = [f"Step        {fr['t']:>4d} / {args.steps}",
                 f"Wind        towards {towards}, {strength}",
                 f"Burning     {s['burning_ha']:6.2f} ha",
                 f"Burnt total {s['burnt_ha']:6.2f} ha",
                 f"Homes at risk  {s['homes_threatened']:3d}",
                 f"Critical fire known  {100 * s['known_critical']:4.0f} %",
                 "", "Drones"]
        for i in range(core.n_drones):
            if not fr["alive"][i]:
                st = f"LOST ({fr['lost'][i]})"
            else:
                st = f"{ALT_NAME[int(fr['alt'][i])]:6s}"
                if core.battery_max:
                    st += f"  battery {100 * fr['battery'][i] / core.battery_max:3.0f}%"
                    if fr["returning"][i]:
                        st += "  -> base"
            lines.append(f"  #{i + 1}  {st}")
        info.set_text("\n".join(lines))
        vline.set_xdata([fr["t"], fr["t"]])
        title.set_text(f"Wildfire swarm — {w.map_type.replace('_', ' ')}, {w.difficulty}, fire: {w.fire_model} — policy: {args.policy}, "
                       f"{core.n_drones} drone(s), area {n * w.cell_m:.0f} m × {n * w.cell_m:.0f} m")
        return []

    if args.png:
        draw(len(frames) - 1)
        fig.savefig(args.png, dpi=args.dpi + 20)
        print("saved", args.png)
        return

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=100, blit=False)
    if args.out.endswith(".mp4"):
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        if animation.writers.is_available("ffmpeg"):
            anim.save(args.out, writer=animation.FFMpegWriter(fps=10, bitrate=2500), dpi=args.dpi)
        else:
            args.out = args.out[:-4] + ".gif"
            print("ffmpeg not found, saving a GIF instead")
            anim.save(args.out, writer=animation.PillowWriter(fps=10), dpi=args.dpi)
    else:
        anim.save(args.out, writer=animation.PillowWriter(fps=10), dpi=args.dpi)
    print("saved", args.out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
