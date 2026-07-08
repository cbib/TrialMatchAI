"""First-level retrieval recall@k: bge-m3 (cosine) vs MedCPT (dot), TREC-21 | TREC-22, vw=0.6.
Reads the benchmark JSONs directly (no hardcoded numbers). Two small-multiple panels, two
categorical series (blue=bge-m3, orange=MedCPT), distinct markers as secondary encoding.
Light + dark PNGs. Grade-2 (eligible) recall; cutoffs 10..2000 at equal index spacing."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BENCH = Path("benchmarks/embedders")  # relative to the repo root (run from there)
CUTOFFS = [10, 100, 300, 500, 700, 1000, 2000]
GRADE = "recall_grade2"  # eligible trials
# (label, json file, light color, dark color, marker) -- fixed categorical slots + distinct
# markers as secondary encoding (identity survives colour-blindness / grayscale).
SERIES = [
    ("bge-m3 · general",      BENCH / "bge-m3-vw0.6.json",           "#2a78d6", "#3987e5", "o"),
    ("Qwen3-Embed · general", BENCH / "qwen3-0.6b-vw0.6.json",       "#008300", "#008300", "^"),
    ("MedCPT · clinical",     BENCH / "medcpt-vw0.6.json",           "#eb6834", "#d95926", "s"),
    ("PubMedBERT · clinical", BENCH / "pubmedbert-neuml-vw0.6.json", "#e34948", "#e66767", "v"),
]
TRACKS = [("21", "TREC 2021"), ("22", "TREC 2022")]
SANS = next((f for f in ["DejaVu Sans", "Arial", "Helvetica"]
             if any(f in n for n in fm.fontManager.get_font_names())), "sans-serif")


def load(path: Path, track: str) -> list[float | None]:
    if not path.exists():
        return [None] * len(CUTOFFS)
    d = json.loads(path.read_text())
    rec = d.get("tracks", {}).get(track, {}).get(GRADE, {})
    return [rec.get(f"recall@{k}") for k in CUTOFFS]


def render(mode: str, outfile: str) -> None:
    if mode == "light":
        surface, ink, sub, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#c9c8c4"
    else:
        surface, ink, sub, grid, axis = "#1a1a19", "#ffffff", "#c3c2b7", "#2e2e2b", "#3d3d39"
    plt.rcParams.update({"font.family": SANS})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), dpi=210, sharey=True)
    fig.patch.set_facecolor(surface)
    x = list(range(len(CUTOFFS)))

    for ax, (tkey, ttitle) in zip(axes, TRACKS):
        ax.set_facecolor(surface)
        for y in (0.25, 0.5, 0.75, 1.0):
            ax.axhline(y, color=grid, lw=1, zorder=0)
        for label, path, cl, cd, marker in SERIES:
            color = cl if mode == "light" else cd
            y = load(path, tkey)
            xi = [xx for xx, yy in zip(x, y) if yy is not None]
            yi = [yy for yy in y if yy is not None]
            if not yi:
                continue  # this embedder's run hasn't produced this track yet
            ax.plot(xi, yi, "-", color=color, lw=2.2, zorder=3)
            ax.plot(xi, yi, marker=marker, color=color, ms=6.0, mec=surface, mew=1.4, zorder=4, linestyle="none")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.3, len(CUTOFFS) - 0.7)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=9.5, color=sub)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in CUTOFFS], fontsize=9, color=ink)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(axis)
        ax.set_title(ttitle, fontsize=12.5, fontweight="bold", color=ink, pad=8)
        ax.set_xlabel("first-level candidates retrieved  (k)", fontsize=10, color=sub, labelpad=5)

    handles = [plt.Line2D([0], [0], color=(s[2] if mode == "light" else s[3]), lw=2.4,
                          marker=s[4], ms=6.5, mec=surface, mew=1.3) for s in SERIES]
    fig.legend(handles, [s[0] for s in SERIES], loc="lower center", ncol=2, frameon=False,
               fontsize=10, labelcolor=ink, handlelength=1.5, columnspacing=2.4,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.062, 0.965, "First-level retrieval recall@k — eligible trials found",
             fontsize=15, fontweight="bold", color=ink, ha="left", va="top")
    fig.text(0.062, 0.905, "General vs clinical embedders · hybrid retrieval, vector weight 0.6 · "
             "eligible = qrels grade 2 · TREC Clinical Trials", fontsize=9.5, color=sub, ha="left", va="top")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.26, wspace=0.08)
    fig.savefig(outfile, facecolor=surface, dpi=210, bbox_inches="tight")
    print("wrote", outfile)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    render("light", str(out_dir / "recall_light.png"))
    render("dark", str(out_dir / "recall_dark.png"))
