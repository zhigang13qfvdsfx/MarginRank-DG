from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


PALETTE = {
    "navy": "#315B7D",
    "blue": "#6F9FC4",
    "orange": "#DFA064",
    "teal": "#64AFA3",
    "coral": "#D77B7E",
    "purple": "#9185B8",
    "grey": "#9AA3AA",
    "dark": "#202428",
    "mid": "#5F6870",
    "light": "#D9DEE2",
    "pale_blue": "#E8F0F6",
    "pale_orange": "#F7EDE2",
    "pale_teal": "#E5F2EF",
    "pale_coral": "#F6E7E7",
    "pale_purple": "#EEEAF5",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.1,
            "legend.title_fontsize": 8.3,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
            "axes.edgecolor": PALETTE["dark"],
            "axes.labelcolor": PALETTE["dark"],
            "text.color": PALETTE["dark"],
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.linewidth": 1.4,
            "lines.markersize": 5.0,
        }
    )


def panel_title(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    letter_x: float = -0.13,
    letter_y: float = 1.09,
) -> None:
    ax.set_title(title, loc="center", color=PALETTE["dark"], pad=8)
    ax.text(
        letter_x,
        letter_y,
        label,
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        color=PALETTE["dark"],
        va="top",
        ha="left",
        clip_on=False,
    )


def clean_axes(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=PALETTE["light"], linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)


def off_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 8.2,
    weight: str = "normal",
) -> None:
    from matplotlib.patches import FancyBboxPatch

    edge = edgecolor or facecolor
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edge,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=PALETTE["dark"],
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str | None = None,
    style: str = "-|>",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color or PALETTE["mid"],
            "linewidth": 1.1,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def stable_jitter(n: int, width: float = 0.12) -> list[float]:
    if n <= 1:
        return [0.0] * n
    positions = [((i * 0.61803398875) % 1.0 - 0.5) * 2 * width for i in range(n)]
    return positions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    title: str,
    panel_titles: Iterable[str],
    plot_types: Iterable[str],
    dpi: int = 450,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    tiff_path = output_dir / f"{stem}.tiff"
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Title": title, "Subject": "TransSurv-PDAC main figure"},
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Title": title, "Description": "; ".join(panel_titles)},
    )
    plt.close(fig)

    poppler = os.environ.get("PDFTOPPM") or shutil.which("pdftoppm")
    if not poppler:
        raise FileNotFoundError(
            "pdftoppm was not found. Install Poppler and add it to PATH, "
            "or set the PDFTOPPM environment variable."
        )
    prefix = output_dir / f"_{stem}_raster"
    subprocess.run(
        [
            poppler,
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated_png = Path(str(prefix) + ".png")
    generated_png.replace(png_path)
    with Image.open(png_path) as image:
        image.save(tiff_path, dpi=(dpi, dpi), compression="tiff_lzw")
        pixel_size = list(image.size)
    return {
        "stem": stem,
        "title": title,
        "panels": list(panel_titles),
        "plot_types": list(plot_types),
        "files": {
            "pdf": str(pdf_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "tiff": str(tiff_path),
        },
        "hashes": {
            "pdf": sha256(pdf_path),
            "svg": sha256(svg_path),
            "png": sha256(png_path),
            "tiff": sha256(tiff_path),
        },
        "raster_dpi": dpi,
        "raster_pixels": pixel_size,
        "panel_count": len(list(panel_titles)),
        "plot_type_count": len(set(plot_types)),
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
