"""Shared figure styling: the reference categorical palette in light and dark, and axis setup."""

from __future__ import annotations

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "slots": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
             "slots": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def setup(theme: str):
    """Import matplotlib headless and apply the theme's text and tick colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10, "text.color": c["text"], "axes.labelcolor": c["muted"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.edgecolor": c["grid"],
    })
    return plt, c


def style_axes(ax, c, title: str | None = None) -> None:
    ax.set_facecolor(c["surface"])
    ax.grid(True, which="major", color=c["grid"], lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=c["text"], fontsize=10)
