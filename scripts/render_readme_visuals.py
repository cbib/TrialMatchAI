"""Render README performance charts from saved evaluation data.

Requires matplotlib. Run from any directory; SVGs are written to docs/assets.
An optional --preview-dir writes PNGs for visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
RANKING = (
    ("qwen36_medcpt", "Qwen3.6-35B-A3B", "MedCPT · 2-bit"),
    ("h100_baichuan", "Baichuan-M2-32B", "MedCPT"),
    ("h100_rrf", "Phi-4", "BGE-M3"),
    ("h100_medcpt", "MedGemma", "MedCPT"),
    ("iimedical_medcpt", "II-Medical-8B", "MedCPT"),
)
EMBEDDERS = (
    ("MedCPT", "medcpt-vw0.6.json", "o", "-"),
    ("PubMedBERT", "pubmedbert-neuml-vw0.6.json", "s", "--"),
    ("BGE-M3", "bge-m3-vw0.6.json", "D", "-"),
    ("Qwen3-Embedding", "qwen3-0.6b-vw0.6.json", "^", ":"),
)
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#263541", "muted": "#697986",
              "grid": "#e7edf1", "accent": "#268e87", "secondary": "#788ca4",
              "series": ("#268e87", "#637f9f", "#8e9da9", "#b1a18e")},
    "dark": {"bg": "#0d1117", "ink": "#d8e1e9", "muted": "#95a6b5",
             "grid": "#26333f", "accent": "#67bfb5", "secondary": "#9baeca",
             "series": ("#67bfb5", "#9baeca", "#8b9eac", "#c5b49e")},
}


def load_ranking() -> dict:
    sources = {}
    tracks = {}
    for year in ("21", "22"):
        path = ROOT / f"benchmarks/trec/unjudged-policy-trec{year}.json"
        raw = path.read_bytes()
        tracks[year] = {run["name"]: run for run in json.loads(raw)["runs"]}
        sources[year] = {"path": path.relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256(raw).hexdigest()}
    rows = []
    for name, label, retriever in RANKING:
        values = {}
        for year in tracks:
            run = tracks[year][name]
            result = run["policies"]["exclude"]
            values[year] = {
                "topics": result["num_queries_ranked"],
                "ndcg@10": result["mean"]["ndcg@10"],
                "graded_P@10": result["mean"]["graded_P@10"],
                "evaluation_input_sha256": run["evaluation_input_sha256"],
            }
        count = sum(v["topics"] for v in values.values())
        assert count == 125
        rows.append({"run": name, "model": label, "retriever_label": retriever,
                     "topics": count, "tracks": values,
                     **{key: sum(v[key] * v["topics"] for v in values.values()) / count
                        for key in ("ndcg@10", "graded_P@10")}})
    result = {"schema": 1, "aggregation": "Topic-weighted mean across 75 TREC 2021 and 50 TREC 2022 topics",
              "unjudged_policy": "exclude", "ndcg": "Tie-aware, linear grades; ideal over returned judged trials",
              "graded_precision": "Sum of top-10 grades divided by 20", "sources": sources, "runs": rows}
    path = ROOT / "benchmarks/readme/ranking-comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def figure(theme: dict, title: str, subtitle: str, height: float):
    fig = plt.figure(figsize=(12, height), facecolor=theme["bg"])
    fig.text(.045, .94, title, color=theme["ink"], fontsize=18, weight="medium", va="top")
    fig.text(.045, .85, subtitle, color=theme["muted"], fontsize=10, va="top")
    return fig


def style_axis(ax, theme: dict):
    ax.set_facecolor(theme["bg"])
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=theme["grid"], linewidth=.8)


def save(fig, name: str, mode: str, preview: Path | None):
    path = ASSETS / f"{name}_{mode}.svg"
    fig.savefig(path, facecolor=fig.get_facecolor(),
                metadata={"Date": None, "Title": name.replace("_", " ")})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
                    encoding="utf-8")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}_{mode}.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def ranking_chart(data: dict, theme: dict):
    fig = figure(theme, "Trial ranking", "TREC 2021 + 2022 · 125 topics · mean scores", 4.9)
    gs = fig.add_gridspec(1, 3, left=.045, right=.96, bottom=.13, top=.69,
                          width_ratios=[1.45, 1, 1], wspace=.25)
    names = fig.add_subplot(gs[0])
    names.set_xlim(0, 1)
    names.set_ylim(4.65, -.65)
    names.axis("off")
    for i, row in enumerate(data["runs"]):
        names.text(0, i-.09, row["model"], color=theme["ink"], fontsize=11, va="center")
        names.text(0, i+.20, row["retriever_label"], color=theme["muted"], fontsize=9, va="center")
    for col, key, title, color in ((1, "ndcg@10", "nDCG@10", theme["accent"]),
                                  (2, "graded_P@10", "Graded P@10", theme["secondary"])):
        ax = fig.add_subplot(gs[col])
        style_axis(ax, theme)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(4.65, -.65)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=17)
        for i, row in enumerate(data["runs"]):
            value = row[key]
            ax.barh(i, value, height=.13, color=color)
            ax.plot(value, i, "o", color=color, markersize=4)
            ax.text(value+.025, i, f"{value:.3f}", fontsize=10, color=theme["ink"], va="center")
    return fig


def recall_chart(theme: dict):
    fig = figure(theme, "First-stage retrieval", "Eligible-trial recall · hybrid retrieval · vector weight 0.6", 5.1)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=.07, right=.96, bottom=.25, top=.69, wspace=.14)
    handles = []
    for ax, year in zip(axes, ("21", "22")):
        style_axis(ax, theme)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=theme["grid"], linewidth=.8)
        ax.set_xlim(0, 2075)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 500, 1000, 2000], ["0", "500", "1,000", "2,000"])
        ax.set_yticks([0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(f"TREC 20{year}", loc="left", fontsize=11, color=theme["ink"], pad=14)
        ax.set_xlabel("Candidates retrieved", color=theme["muted"], fontsize=9, labelpad=12)
        ax.axvline(1000, color=theme["grid"], linestyle="--", linewidth=.8, zorder=0)
        for color, (label, filename, marker, linestyle) in zip(theme["series"], EMBEDDERS):
            data = json.loads((ROOT / "benchmarks/embedders" / filename).read_text())
            values = data["tracks"][year]["recall_grade2"]
            cutoffs = sorted(int(key.split("@")[1]) for key in values)
            artist, = ax.plot(cutoffs, [values[f"recall@{k}"] for k in cutoffs],
                              color=color, marker=marker, markersize=3.3, linewidth=1.9,
                              linestyle=linestyle, label=label)
            if year == "21":
                handles.append(artist)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=4, frameon=False, fontsize=9, labelcolor=theme["muted"],
               columnspacing=2.8, handlelength=2.4)
    return fig


def trec23_chart(theme: dict):
    data = json.loads((ROOT / "benchmarks/trec/bge-m3-phi4-trec23.json").read_text())
    metrics = data["runs"][0]["policies"]["exclude"]["mean"]
    fig = figure(theme, "TREC 2023", "BGE-M3 + Phi-4 · 37 judged topics · mean scores", 4.4)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.115, right=.96, bottom=.16, top=.66, wspace=.85)
    panels = (
        ("Ranking", [("nDCG@5", "ndcg@5"), ("nDCG@10", "ndcg@10"), ("nDCG@20", "ndcg@20")]),
        ("Precision and retrieval", [("Graded P@10", "graded_P@10"),
                                    ("P@10 · grade ≥1", "P@10(rel>=1)"),
                                    ("P@10 · eligible", "P@10(eligible)"),
                                    ("Recall@1000 · grade ≥1", "recall@1000")]),
    )
    for ax, (title, rows) in zip(axes, panels):
        style_axis(ax, theme)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(len(rows)-.45, -.65)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        ax.set_yticks(range(len(rows)), [name for name, _ in rows])
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=15)
        for i, (_, key) in enumerate(rows):
            value = metrics[key]
            color = theme["secondary"] if key == "recall@1000" else theme["accent"]
            ax.barh(i, value, height=.16, color=color)
            ax.plot(value, i, "o", color=color, markersize=4)
            ax.text(value+.025, i, f"{value:.4f}", color=theme["ink"], fontsize=10, va="center")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Also save PNG previews")
    args = parser.parse_args()
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "trialmatchai-readme", "axes.unicode_minus": False})
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = load_ranking()
    for mode, theme in THEMES.items():
        save(ranking_chart(data, theme), "performance", mode, args.preview_dir)
        save(recall_chart(theme), "recall", mode, args.preview_dir)
        save(trec23_chart(theme), "trec2023", mode, args.preview_dir)
        print(f"Rendered {mode} ranking, retrieval, and TREC 2023 charts.")


if __name__ == "__main__":
    main()
