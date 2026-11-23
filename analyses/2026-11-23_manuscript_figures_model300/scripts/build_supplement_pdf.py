#!/usr/bin/env python3
"""Put the regenerated figures into the manuscript's figure PDF.

Pages 1-12 and 21-28 of the source are copied unchanged. On pages 13-20 (Figure 5 and Figures 5-S1 to 5-S7) the
existing artwork is covered and the new file is placed in its stead; the page's own "Figure 5-Sx" label is left
untouched. Two pages are appended after Figure 5-S7 for the pathway screens of acute preterm lung injury and of
BPD-PH, which the source has no page for; their labels are drawn in Helvetica, the source uses Calibri.
usage: python3 build_supplement_pdf.py <figures_numbered dir> <source pdf> <output pdf>
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

LABEL_SIZE = 18
MARGIN_X = 10
GAP_BELOW_LABEL = 14
BOTTOM_MARGIN = 24

REPLACE = {  # 1-based page of the source -> figure file stem
    13: "Figure_5_main",
    14: "Figure_5-S1_energy",
    15: "Figure_5-S2_deltas_acute_injury",
    16: "Figure_5-S3_deltas_bpd",
    17: "Figure_5-S4_deltas_bpd_ph",
    18: "Figure_5-S5_deltas_rat_hyperoxia",
    19: "Figure_5-S6_deltas_rat_azithromycin",
    20: "Figure_5-S7_venn",
}
APPEND_AFTER = 20
APPEND = [("Figure 5-S8a", "Figure_5-S8a_screen_acute_injury"),
          ("Figure 5-S8b", "Figure_5-S8b_screen_bpd_ph")]


def source_size(figdir: Path, stem: str) -> tuple[float, float, Path, bool]:
    pdf = figdir / f"{stem}.pdf"
    if pdf.exists():
        with fitz.open(pdf) as d:
            r = d[0].rect
        return r.width, r.height, pdf, True
    png = figdir / f"{stem}.png"
    with fitz.open(png) as d:
        r = d[0].rect
    return r.width, r.height, png, False


def place(page: fitz.Page, box: fitz.Rect, src: Path, vector: bool, w: float, h: float) -> fitz.Rect:
    scale = min(box.width / w, box.height / h)
    tw, th = w * scale, h * scale
    x0 = box.x0 + (box.width - tw) / 2
    y0 = box.y0 + (box.height - th) / 2
    target = fitz.Rect(x0, y0, x0 + tw, y0 + th)
    if vector:
        with fitz.open(src) as d:
            page.show_pdf_page(target, d, 0)
    else:
        page.insert_image(target, filename=str(src))
    return target


def main() -> None:
    figdir, src_pdf, out_pdf = (Path(a) for a in sys.argv[1:4])
    src = fitz.open(src_pdf)
    out = fitz.open()
    for i in range(src.page_count):
        page_no = i + 1
        out.insert_pdf(src, from_page=i, to_page=i)
        page = out[-1]
        if page_no in REPLACE:
            stem = REPLACE[page_no]
            label_bottom = max((b[3] for b in page.get_text("blocks") if "Figure" in b[4]), default=40.0)
            for im in page.get_images(full=True):  # the old artwork, removed from the content, not covered
                page.add_redact_annot(page.get_image_bbox(im) + (-1, -1, 1, 1))
            for b in page.get_text("blocks"):  # panel letters of the old layout, keeping the "Figure 5-Sx" label
                if "Figure" not in b[4]:
                    page.add_redact_annot(fitz.Rect(b[:4]) + (-1, -1, 1, 1))
            page.apply_redactions()
            box = fitz.Rect(MARGIN_X, label_bottom + GAP_BELOW_LABEL,
                            page.rect.width - MARGIN_X, page.rect.height - BOTTOM_MARGIN)
            w, h, f, vec = source_size(figdir, stem)
            t = place(page, box, f, vec, w, h)
            print(f"page {page_no}: {f.name} -> {[round(v) for v in t]}{' (vector)' if vec else ''}")
        if page_no == APPEND_AFTER:
            pw, ph = page.rect.width, page.rect.height  # page objects go stale once a page is added
            for label, stem in APPEND:
                new = out.new_page(width=pw, height=ph)
                tw = fitz.get_text_length(label, fontname="helv", fontsize=LABEL_SIZE)
                new.insert_text((498 - tw, 195), label, fontname="helv", fontsize=LABEL_SIZE, color=(0, 0, 0))
                box = fitz.Rect(MARGIN_X, 195 + GAP_BELOW_LABEL, pw - MARGIN_X, ph - BOTTOM_MARGIN)
                w, h, f, vec = source_size(figdir, stem)
                t = place(new, box, f, vec, w, h)
                print(f"appended {label}: {f.name} -> {[round(v) for v in t]}{' (vector)' if vec else ''}")
    out.save(out_pdf, garbage=4, deflate=True)
    print(f"{src.page_count} source pages -> {out.page_count} pages, {out_pdf}")


if __name__ == "__main__":
    main()
