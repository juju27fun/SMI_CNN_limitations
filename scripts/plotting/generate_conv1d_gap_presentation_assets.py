#!/usr/bin/env python3
"""Generate slide-ready PNG blocks for the Conv1D vs Conv1DGAP explanation."""

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "conv1d_vs_conv1dgap"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

COLORS = {
    "input": {"fill": "#f7f7f7", "stroke": "#666666"},
    "backbone": {"fill": "#dbeafe", "stroke": "#2563eb"},
    "latent": {"fill": "#eef2ff", "stroke": "#4f46e5"},
    "flatten": {"fill": "#ffedd5", "stroke": "#ea580c"},
    "heavy": {"fill": "#fed7aa", "stroke": "#c2410c"},
    "gap": {"fill": "#dcfce7", "stroke": "#16a34a"},
    "small": {"fill": "#bbf7d0", "stroke": "#15803d"},
    "note": {"fill": "#f7f7f7", "stroke": "#666666"},
}


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def draw_centered_lines(draw, lines, box, title_color="#111111"):
    x0, y0, x1, y1 = box
    title_font = font(46, bold=True)
    body_font = font(32)
    small_font = font(28)

    rendered = []
    for i, line in enumerate(lines):
        fnt = title_font if i == 0 else body_font
        if len(line) > 36 and i != 0:
            parts = wrap(line, width=36)
        else:
            parts = [line]
        for part in parts:
            rendered.append((part, fnt if i < 2 else small_font, i == 0))

    heights = []
    for text, fnt, _ in rendered:
        bbox = draw.textbbox((0, 0), text, font=fnt)
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + 16 * (len(rendered) - 1)
    y = y0 + ((y1 - y0) - total_h) / 2

    for (text, fnt, is_title), h in zip(rendered, heights):
        bbox = draw.textbbox((0, 0), text, font=fnt)
        w = bbox[2] - bbox[0]
        draw.text(((x0 + x1 - w) / 2, y), text, font=fnt, fill=title_color)
        y += h + 16


def make_block(filename, lines, style, size=(1080, 360)):
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    margin = 22
    rect = (margin, margin, size[0] - margin, size[1] - margin)
    palette = COLORS[style]
    draw.rounded_rectangle(
        rect,
        radius=28,
        fill=palette["fill"],
        outline=palette["stroke"],
        width=8,
    )
    draw_centered_lines(draw, lines, rect)
    img.save(OUT / filename)


def make_table_piece(filename, rows, include_header=True):
    widths = [360, 470, 390, 300]
    header_h = 96 if include_header else 0
    row_h = 126
    pad = 22
    size = (sum(widths) + 2 * pad, header_h + row_h * len(rows) + 2 * pad)

    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    x0, y0 = pad, pad
    x1, y1 = size[0] - pad, size[1] - pad
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=22,
        fill="#ffffff",
        outline="#555555",
        width=4,
    )

    header_font = font(31, bold=True)
    cell_font = font(30)
    small_font = font(26)
    headers = ["Model", "Test accuracy", "Best val accuracy", "Params"]

    def cell_text(text, x, y, w, h, fnt, fill="#111111"):
        lines = text.split("\n")
        line_boxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
        total_h = sum(b[3] - b[1] for b in line_boxes) + 8 * (len(lines) - 1)
        cy = y + (h - total_h) / 2
        for line, bbox in zip(lines, line_boxes):
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x + (w - tw) / 2, cy), line, font=fnt, fill=fill)
            cy += th + 8

    y = y0
    if include_header:
        draw.rectangle((x0, y0, x1, y0 + header_h), fill="#f7f7f7")
        cx = x0
        for header, w in zip(headers, widths):
            cell_text(header, cx, y, w, header_h, header_font)
            cx += w
        y += header_h
        draw.line((x0, y, x1, y), fill="#555555", width=3)

    for row_i, row in enumerate(rows):
        model, test, val, params, style = row
        fill = COLORS[style]["fill"]
        stroke = COLORS[style]["stroke"]
        draw.rectangle((x0, y, x1, y + row_h), fill=fill)
        draw.line((x0, y, x1, y), fill=stroke, width=3)
        cx = x0
        values = [model, test, val, params]
        for col_i, (value, w) in enumerate(zip(values, widths)):
            fnt = cell_font if col_i != 1 else small_font
            cell_text(value, cx, y, w, row_h, fnt)
            cx += w
            if col_i < len(widths) - 1:
                draw.line((cx, y, cx, y + row_h), fill="#777777", width=2)
        y += row_h

    img.save(OUT / filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    blocks = [
        ("architecture_01_input.png", ["Input trace", "1 x 625"], "input"),
        (
            "architecture_02_shared_backbone.png",
            [
                "Shared conv backbone",
                "k=5, filters 64 -> 128 -> 256",
                "3x Conv + BN + ReLU + MaxPool /2",
                "206,464 params",
            ],
            "backbone",
        ),
        ("architecture_03_latent_map.png", ["Latent map", "256 x 78"], "latent"),
        ("architecture_04_conv1d_flatten.png", ["Conv1D", "Flatten", "19,968 features"], "flatten"),
        (
            "architecture_05_conv1d_dense_head.png",
            ["Dense head", "19,968 -> 256 -> 3", "5,112,835 params"],
            "heavy",
        ),
        ("architecture_06_conv1d_total.png", ["Conv1D total", "5,319,299 params"], "heavy"),
        (
            "architecture_07_conv1dgap_pool.png",
            ["Conv1DGAP", "Global average pooling", "256 features"],
            "gap",
        ),
        (
            "architecture_08_conv1dgap_dense_head.png",
            ["Dense head", "256 -> 256 -> 3", "66,563 params"],
            "small",
        ),
        ("architecture_09_conv1dgap_total.png", ["Conv1DGAP total", "273,027 params"], "small"),
        ("parameter_01_conv1d.png", ["Conv1D", "5.32M params", "100%"], "heavy"),
        ("parameter_02_conv1dgap.png", ["Conv1DGAP", "273K params", "5.1%"], "small"),
        ("parameter_03_takeaway.png", ["Same task", "about 19.5x smaller"], "note"),
        ("accuracy_01_conv1d.png", ["Conv1D", "96.68% test accuracy", "98.75% best val"], "heavy"),
        ("accuracy_02_conv1dgap.png", ["Conv1DGAP", "97.12% test accuracy", "98.46% best val"], "small"),
        ("accuracy_03_takeaway.png", ["Accuracy stays close", "model size changes a lot"], "note"),
    ]

    for filename, lines, style in blocks:
        make_block(filename, lines, style)

    rows = [
        ("Conv1D", "96.68 ± 0.33%", "98.75%", "5.32M", "heavy"),
        ("Conv1DGAP", "97.12 ± 1.02%", "98.46%", "273K", "small"),
    ]
    make_table_piece("table_00_full_accuracy_params.png", rows, include_header=True)
    make_table_piece("table_01_header.png", [], include_header=True)
    make_table_piece("table_02_conv1d_row.png", [rows[0]], include_header=False)
    make_table_piece("table_03_conv1dgap_row.png", [rows[1]], include_header=False)
    make_table_piece("table_04_conv1d_with_header.png", [rows[0]], include_header=True)
    make_table_piece("table_05_conv1dgap_with_header.png", [rows[1]], include_header=True)

    manifest = OUT / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "Conv1D vs Conv1DGAP presentation assets",
                "",
                "Architecture blocks: architecture_01_input.png through architecture_09_conv1dgap_total.png",
                "Parameter blocks: parameter_01_conv1d.png through parameter_03_takeaway.png",
                "Accuracy blocks: accuracy_01_conv1d.png through accuracy_03_takeaway.png",
                "Tables:",
                "  table_00_full_accuracy_params.png: full table",
                "  table_01_header.png: header only",
                "  table_02_conv1d_row.png: Conv1D row only",
                "  table_03_conv1dgap_row.png: Conv1DGAP row only",
                "  table_04_conv1d_with_header.png: Conv1D half-table",
                "  table_05_conv1dgap_with_header.png: Conv1DGAP half-table",
                "",
                "All PNGs use transparent backgrounds except the table body.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote assets to {OUT}")


if __name__ == "__main__":
    main()
