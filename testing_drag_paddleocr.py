#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA parser using PaddleOCR for text extraction.

Usage and CLI shape mirror the PaddleOCR-based script: point the script at one or
more PDFs and it will emit chunked QA text. Recent improvements add optional
preprocessing and Korean clean-up toggles that default to enabled but can be
switched off or tuned via new flags (for example, ``--paddleocr-preprocess-
threshold`` or ``--paddleocr-korean-lexicon``). You can continue invoking the
script exactly as before if you do not need to change those defaults.

To reuse previously detected chunk bounding boxes (instead of recalculating
them), pass ``--reuse-chunks-from <prior_output.json>``. The flag accepts a
saved QA JSON from an earlier run or a template file that contains a
``chunks``/``pieces`` list. Each piece should carry a page index, column tag,
and box coordinates; the script will re-extract text inside those boxes while
preserving the original geometry. Quick examples::

    # Re-OCR a new PDF but reuse boxes from an earlier run
    python paddleocr_QA_parsing_Final.py new_input.pdf \
      --reuse-chunks-from prior_run.json --output-json new_run.json

    # Reuse boxes while also writing chunk previews for auditing
    python paddleocr_QA_parsing_Final.py input.pdf \
      --reuse-chunks-from prior_run.json \
      --chunk-preview-dir previews/reused_boxes

    # Apply reuse plus Korean cleanup options explicitly
    python paddleocr_QA_parsing_Final.py input.pdf \
      --reuse-chunks-from prior_run.json \
      --paddleocr-korean-lexicon common_terms.txt \
      --paddleocr-preprocess-threshold 140

In each case ``prior_run.json`` is the JSON produced by a previous invocation
of the script (or another template file with the same structure). The script
reuses the boxes from that file and writes fresh OCR/text results to the
specified output JSON. Add optional preview or preprocessing flags as desired
while the geometry stays anchored to the prior run.

This script mirrors the flow-based chunking approach of
``pdfplumber_QA_parsing_Final.py`` but replaces PDF text extraction with
PaddleOCR. Subject detection is removed – every detected chunk inside the
column bounding boxes is parsed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pdfplumber
import paddle
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


logger = logging.getLogger(__name__)

# Default column window fractions (relative to page width/height)
DEFAULT_TOP_FRAC = 0.10
DEFAULT_BOTTOM_FRAC = 0.90
DEFAULT_GUTTER_FRAC = 0.005

DEFAULT_SEARCHABLE_FONT_CANDIDATES: Tuple[str, ...] = (
    "HYGoThic-Medium",
    "HYGothic-Medium",
    "HYSMyeongJo-Medium",
    "HeiseiKakuGo-W5",
    "HeiseiMin-W3",
    "STSong-Light",
)

DEFAULT_KOREAN_LEXICON: Tuple[str, ...] = (
    "치명률",
    "사망률",
    "발생률",
    "유병률",
    "생존율",
    "검사율",
    "감염률",
)


@dataclass
class ManualQuestionRegion:
    """Container for a manually marked question and its option regions."""

    id: int
    page_index: int
    rect: Tuple[float, float, float, float]
    options_rect: Optional[Tuple[float, float, float, float]] = None
    option_rects: Dict[int, Tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    def normalized_rect(self) -> Tuple[float, float, float, float]:
        return _normalize_rect(self.rect)

    def normalized_options_rect(self) -> Optional[Tuple[float, float, float, float]]:
        if self.options_rect is None:
            return None
        return _normalize_rect(self.options_rect)

    def iter_option_rects(self):
        for idx, box in sorted(self.option_rects.items()):
            yield idx, _normalize_rect(box)

# =========================
# Helpers
# =========================

def list_pdfs(folder: str) -> List[str]:
    try:
        items = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder, f))
            ]
        )
    except Exception:
        items = []
    return items


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_visible_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[\u00A0\u2000-\u200B]", " ", s)
    s = (
        s.replace("ㆍ", "·")
        .replace("∙", "·")
        .replace("・", "·")
        .replace("•", "·")
    )
    return s


def _enhance_page_image(img: Image.Image, settings: OCRSettings) -> Image.Image:
    if not settings.preprocess_images:
        return img

    working = img.convert("L")

    if settings.preprocess_unsharp:
        working = working.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=3)
        )

    if settings.preprocess_contrast and settings.preprocess_contrast > 0:
        working = ImageEnhance.Contrast(working).enhance(settings.preprocess_contrast)

    if settings.preprocess_threshold is not None:
        thresh = int(settings.preprocess_threshold)
        thresh = max(0, min(255, thresh))
        working = working.point(lambda p: 255 if p >= thresh else 0)

    return working.convert("RGB")


def _korean_postprocess_text(text: str, settings: OCRSettings) -> str:
    if not settings.korean_postprocess:
        return text

    lexicon = set(settings.korean_lexicon)
    if not lexicon:
        return text

    confusion_pairs = (("룰", "률"), ("율", "률"), ("률", "율"))
    tokens = re.split(r"(\W+)", text)
    corrected: List[str] = []

    for tok in tokens:
        if not tok or not tok.strip() or tok in lexicon:
            corrected.append(tok)
            continue

        replacement = tok
        for wrong, right in confusion_pairs:
            candidate = tok.replace(wrong, right)
            if candidate in lexicon:
                replacement = candidate
                break
        corrected.append(replacement)

    return "".join(corrected)


def _coerce_bbox(box: Dict[str, object]) -> Optional[Tuple[float, float, float, float]]:
    try:
        x0 = _safe_float(box.get("x0"), None)
        x1 = _safe_float(box.get("x1"), None)
        top = _safe_float(box.get("top"), None)
        bottom = _safe_float(box.get("bottom"), None)
    except Exception:
        return None
    if None in (x0, x1, top, bottom):
        return None
    return float(x0), float(top), float(x1), float(bottom)


# =========================
# Logging helpers
# =========================


def _parse_log_level(value: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        level = getattr(logging, value.upper())
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}") from exc
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}")
    return level


class HeartbeatLogger:
    """Periodically emit log messages while work is ongoing."""

    def __init__(self, interval: float = 30.0, message: str = "Working..."):
        self.interval = max(1.0, float(interval))
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None:
            return
        logger.debug("Starting heartbeat logger every %.1f seconds", self.interval)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.wait(self.interval):
            logger.info(self.message)

    def stop(self):
        if self._thread is None:
            return
        logger.debug("Stopping heartbeat logger")
        self._stop_event.set()
        self._thread.join()
        self._thread = None


# =========================
# Option / question regex helpers
# =========================
OPTION_RANGES = [
    (0x2460, 0x2473),  # ①-⑳
    (0x2474, 0x2487),  # ⑴-⒇
    (0x2488, 0x249B),  # ⒈-⒛
    (0x24F5, 0x24FE),  # ⓵-⓾
]
OPTION_EXTRA = {0x24EA, 0x24FF, 0x24DB}  # ⓪, ⓿, ⓛ
OPTION_SET = {
    chr(cp)
    for start, end in OPTION_RANGES
    for cp in range(start, end + 1)
}
OPTION_SET.update(chr(cp) for cp in OPTION_EXTRA)
OPTION_CLASS = "".join(sorted(OPTION_SET))
QUESTION_CIRCLED_RANGE = f"{OPTION_CLASS}{chr(0x3250)}-{chr(0x32FF)}"

ASCII_OPTION_RE = re.compile(r"(?<!\d)(?:\(|\[)?(1[0-9]|20|[1-9])\s*[).:]")
ASCII_OPTION_PREFIX_RE = re.compile(
    r"(?m)(\n)\s*(?:\(|\[)?(1[0-9]|20|[1-9])\s+(?=\S)"
)
DIGIT_TO_CIRCLED = {
    str(i): chr(0x2460 + i - 1) if 1 <= i <= 20 else str(i)
    for i in range(1, 21)
}

OPT_SPLIT_RE = re.compile(rf"(?=([{OPTION_CLASS}]))")
CIRCLED_STRIP_RE = re.compile(rf"^[{OPTION_CLASS}]\s*")
QUESTION_START_LINE_RE = re.compile(
    rf"^\s*(?:[{QUESTION_CIRCLED_RANGE}]|[0-9]{{1,3}}[.)]|제\s*[0-9]{{1,3}}\s*문)",
    re.MULTILINE,
)
QUESTION_NUM_RE = re.compile(
    r"^\s*(?:\(\s*(\d{1,3})\s*\)|(\d{1,3})\s*번|(\d{1,3}))\s*[.)]?\s*"
)

DISPUTE_RE = re.compile(
    r"\(?\s*다툼이\s*(?:있는\s*경우|있으면)\s*(?P<site>[^)\n]*?)\s*(?:판례|결정)\s*에\s*의함\)?",
    re.IGNORECASE,
)

LEADING_HEADER_STRIP = re.compile(
    r"^\s*(?:[【\[]\s*[^】\]]+\s*[】\]])\s*(?:\([^)]*\))?\s*"
)


def _strip_header_garbage(text: str) -> str:
    return norm_space(LEADING_HEADER_STRIP.sub("", text or ""))


def _normalize_option_markers(text: str) -> str:
    def repl(match: re.Match) -> str:
        if match.start() == 0:
            return match.group(0)
        prior = text[: match.start()]
        if prior.strip() == "":
            return match.group(0)
        num = match.group(1)
        circled = DIGIT_TO_CIRCLED.get(num)
        if not circled or circled == num:
            return match.group(0)
        trailing = " " if not match.group(0).endswith(" ") else ""
        return f"{circled}{trailing}"

    text = ASCII_OPTION_RE.sub(repl, text)

    def prefix_repl(match: re.Match) -> str:
        lead, num = match.group(1), match.group(2)
        prior = text[: match.start(2)]
        if prior.strip() == "":
            return match.group(0)
        circled = DIGIT_TO_CIRCLED.get(num)
        if not circled or circled == num:
            return match.group(0)
        return f"{lead}{circled} "

    return ASCII_OPTION_PREFIX_RE.sub(prefix_repl, text)


def infer_year_from_filename(path: str) -> Optional[int]:
    fname = os.path.basename(path)
    m = re.search(r"(\d{2})년", fname)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"(20\d{2}|19\d{2})", fname)
    if m:
        return int(m.group(1))
    return None


# =========================
# PaddleOCR extraction
# =========================
@dataclass
class OCRSettings:
    dpi: int = 1000
    languages: Sequence[str] = ("korean",)
    gpu: bool = False
    column_pad_x: float = 2.0
    column_pad_top: float = 2.0
    column_pad_bottom: float = 60.0
    column_filter_top: float = 4.0
    column_filter_bottom: float = 60.0
    preprocess_images: bool = True
    preprocess_contrast: float = 1.25
    preprocess_unsharp: bool = True
    preprocess_threshold: Optional[int] = None
    korean_postprocess: bool = True
    korean_lexicon: Sequence[str] = field(default_factory=lambda: DEFAULT_KOREAN_LEXICON)


class PaddleOCRTextExtractor:
    """Render PDF pages to images and run PaddleOCR within bboxes."""

    def __init__(self, pdf: pdfplumber.pdf.PDF, settings: OCRSettings):
        self.pdf = pdf
        self.settings = settings
        reader_kwargs = {
            "lang": settings.languages[0] if settings.languages else "korean",
        }

        # Accommodate PaddleOCR versions that renamed ``use_gpu`` to ``device``
        # and ``use_angle_cls`` to ``use_textline_orientation``.
        reader_sig = inspect.signature(PaddleOCR.__init__)
        params = reader_sig.parameters

        if "use_gpu" in params:
            reader_kwargs["use_gpu"] = settings.gpu
        elif "device" in params:
            reader_kwargs["device"] = "gpu:0" if settings.gpu else "cpu"

        if "use_textline_orientation" in params:
            reader_kwargs["use_textline_orientation"] = True
        elif "use_angle_cls" in params:
            reader_kwargs["use_angle_cls"] = True

        self.reader = PaddleOCR(**reader_kwargs)
        self._image_cache: Dict[int, Image.Image] = {}
        self._scale = settings.dpi / 72.0
        self.page_word_boxes: Dict[int, List[Dict[str, object]]] = {}
        self.crop_reports: Dict[int, List[Dict[str, Any]]] = {}
        logger.debug(
            "Initialized PaddleOCRTextExtractor: dpi=%s languages=%s gpu=%s pads(x=%.1f top=%.1f bottom=%.1f) filters(top=%.1f bottom=%.1f)",
            settings.dpi,
            ",".join(settings.languages),
            settings.gpu,
            settings.column_pad_x,
            settings.column_pad_top,
            settings.column_pad_bottom,
            settings.column_filter_top,
            settings.column_filter_bottom,
        )

    def _page_image(self, page_index: int) -> Image.Image:
        if page_index not in self._image_cache:
            logger.debug("Rendering page %d at %d DPI", page_index + 1, self.settings.dpi)
            pil = (
                self.pdf.pages[page_index]
                .to_image(resolution=int(self.settings.dpi))
                .original.convert("RGB")
            )
            pil = _enhance_page_image(pil, self.settings)
            self._image_cache[page_index] = pil
        return self._image_cache[page_index]

    def _record_crop_sample(
        self,
        page_index: int,
        column_tag: Optional[str],
        orig_bbox: Tuple[float, float, float, float],
        padded_bbox: Tuple[float, float, float, float],
        filter_top: float,
        filter_bottom: float,
        pad_x: float,
        pad_top: float,
        pad_bottom: float,
        y_cut: Optional[float],
        drop_zone: Optional[Tuple[float, float, float, float]],
        raw_word_count: int,
        kept_word_count: int,
        line_count: int,
    ) -> None:
        drop_tuple: Optional[Tuple[float, float, float, float]] = None
        if drop_zone is not None:
            drop_tuple = tuple(float(v) for v in drop_zone)
        entry: Dict[str, Any] = {
            "column": column_tag or "?",
            "orig_bbox": tuple(float(v) for v in orig_bbox),
            "padded_bbox": tuple(float(v) for v in padded_bbox),
            "filter_band": (float(filter_top), float(filter_bottom)),
            "pad": {
                "x": float(pad_x),
                "top": float(pad_top),
                "bottom": float(pad_bottom),
            },
            "y_cut": float(y_cut) if y_cut is not None else None,
            "drop_zone": drop_tuple,
            "raw_word_count": int(raw_word_count),
            "kept_word_count": int(kept_word_count),
            "line_count": int(line_count),
        }
        bucket = self.crop_reports.setdefault(page_index, [])
        bucket.append(entry)

    def extract_lines(
        self,
        page_index: int,
        bbox: Tuple[float, float, float, float],
        y_tol: float = 3.0,
        y_cut: Optional[float] = None,
        drop_zone: Optional[Tuple[float, float, float, float]] = None,
        column_tag: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        page = self.pdf.pages[page_index]
        page_width = float(page.width)
        page_height = float(page.height)
        x0, y0, x1, y1 = bbox
        if y_cut is not None:
            y0 = max(y0, y_cut)
        if x1 <= x0 or y1 <= y0:
            return []

        pad_x = max(0.0, float(getattr(self.settings, "column_pad_x", 0.0)))
        pad_top = max(0.0, float(getattr(self.settings, "column_pad_top", 0.0)))
        pad_bottom = max(0.0, float(getattr(self.settings, "column_pad_bottom", 0.0)))
        orig_bbox = (x0, y0, x1, y1)
        padded_bbox = (x0, y0, x1, y1)
        if pad_x or pad_top or pad_bottom:
            padded_bbox = (
                max(0.0, x0 - pad_x),
                max(0.0, y0 - pad_top),
                min(page_width, x1 + pad_x),
                min(page_height, y1 + pad_bottom),
            )
            x0, y0, x1, y1 = padded_bbox
        filter_top = max(
            0.0,
            orig_bbox[1]
            - max(float(getattr(self.settings, "column_filter_top", 0.0)), pad_top),
        )
        filter_bottom = min(
            page_height,
            orig_bbox[3]
            + max(float(getattr(self.settings, "column_filter_bottom", 0.0)), pad_bottom),
        )

        scale = self._scale
        im = self._page_image(page_index)
        logger.debug(
            "Extracting lines from page %d bbox=(%.1f, %.1f, %.1f, %.1f) scale=%.3f (padded to %.1f, %.1f, %.1f, %.1f)",
            page_index + 1,
            orig_bbox[0],
            orig_bbox[1],
            orig_bbox[2],
            orig_bbox[3],
            scale,
            x0,
            y0,
            x1,
            y1,
        )
        crop_box = (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        )
        if crop_box[2] - crop_box[0] <= 0 or crop_box[3] - crop_box[1] <= 0:
            self._record_crop_sample(
                page_index,
                column_tag,
                orig_bbox,
                padded_bbox,
                filter_top,
                filter_bottom,
                pad_x,
                pad_top,
                pad_bottom,
                y_cut,
                drop_zone,
                raw_word_count=0,
                kept_word_count=0,
                line_count=0,
            )
            return []

        crop = im.crop(crop_box)
        np_img = np.array(crop)
        results = self.reader.ocr(np_img)

        if results and isinstance(results[0], list) and results and isinstance(
            results[0][0], list
        ):
            parsed_results = results[0]
        else:
            parsed_results = results

        entries = []
        word_entries: List[Dict[str, object]] = []
        for item in parsed_results:
            if not item or len(item) < 2:
                continue
            pts = item[0]
            text_conf = item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                text, conf = text_conf[0], text_conf[1]
            else:
                text, conf = str(text_conf), None
            if not text:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            px0, px1 = min(xs), max(xs)
            py0, py1 = min(ys), max(ys)
            abs_x0 = x0 + px0 / scale
            abs_x1 = x0 + px1 / scale
            abs_y0 = y0 + py0 / scale
            abs_y1 = y0 + py1 / scale
            mid_y = 0.5 * (abs_y0 + abs_y1)
            if mid_y < filter_top or mid_y > filter_bottom:
                logger.debug(
                    "Skipping OCR word outside filter band: page=%d mid_y=%.1f filter=(%.1f, %.1f)",
                    page_index + 1,
                    mid_y,
                    filter_top,
                    filter_bottom,
                )
                continue
            if drop_zone and _rects_intersect(
                (abs_x0, abs_y0, abs_x1, abs_y1), drop_zone
            ):
                continue
            normalized_text = _normalize_visible_text(text)
            normalized_text = _korean_postprocess_text(normalized_text, self.settings)
            entries.append(
                {
                    "x0": float(abs_x0),
                    "x1": float(abs_x1),
                    "top": float(abs_y0),
                    "bottom": float(abs_y1),
                    "text": normalized_text,
                }
            )
            poly = [
                {
                    "x": float(x0 + px / scale),
                    "y": float(y0 + py / scale),
                }
                for px, py in pts
            ]
            word_entries.append(
                {
                    "x0": float(abs_x0),
                    "x1": float(abs_x1),
                    "top": float(abs_y0),
                    "bottom": float(abs_y1),
                    "text": normalized_text,
                    "confidence": float(conf) if conf is not None else None,
                    "column": column_tag,
                    "poly": poly,
                }
            )

        if word_entries:
            bucket = self.page_word_boxes.setdefault(page_index, [])
            bucket.extend(word_entries)

        entries.sort(key=lambda r: (r["top"], r["x0"]))
        grouped: List[List[Dict[str, float]]] = []
        cur: List[Dict[str, float]] = []
        cur_top: Optional[float] = None
        for item in entries:
            top = item["top"]
            if cur_top is None or abs(top - cur_top) <= y_tol:
                cur.append(item)
                cur_top = top if cur_top is None else cur_top
            else:
                grouped.append(cur)
                cur = [item]
                cur_top = top
        if cur:
            grouped.append(cur)

        lines: List[Dict[str, float]] = []
        for group in grouped:
            gx0 = min(it["x0"] for it in group)
            gx1 = max(it["x1"] for it in group)
            gy0 = min(it["top"] for it in group)
            gy1 = max(it["bottom"] for it in group)
            gtext = " ".join(it["text"] for it in group)
            lines.append(
                {
                    "x0": gx0,
                    "x1": gx1,
                    "top": gy0,
                    "bottom": gy1,
                    "y": 0.5 * (gy0 + gy1),
                    "text": gtext,
                }
            )

        lines.sort(key=lambda ln: (ln["top"], ln["x0"]))
        self._record_crop_sample(
            page_index,
            column_tag,
            orig_bbox,
            padded_bbox,
            filter_top,
            filter_bottom,
            pad_x,
            pad_top,
            pad_bottom,
            y_cut,
            drop_zone,
            raw_word_count=len(results),
            kept_word_count=len(word_entries),
            line_count=len(lines),
        )
        logger.debug(
            "Page %d OCR produced %d text lines in column window",
            page_index + 1,
            len(lines),
        )
        return lines


# =========================
# Layout helpers
# =========================


def _normalize_rect(rect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _rect_center(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = _normalize_rect(rect)
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _point_in_rect(point: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    x, y = point
    x0, y0, x1, y1 = _normalize_rect(rect)
    return x0 <= x <= x1 and y0 <= y <= y1


def two_col_bboxes(
    page,
    top_frac: float = DEFAULT_TOP_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
    gutter_frac: float = DEFAULT_GUTTER_FRAC,
):
    w, h = float(page.width), float(page.height)
    top = h * top_frac
    bottom = h * bottom_frac
    gutter = w * gutter_frac
    mid = w * 0.5
    return (0.0, top, mid - gutter, bottom), (mid + gutter, top, w, bottom)


def _rects_intersect(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


# =========================
# Question detection
# =========================

def _extract_qnum_from_text(text: str) -> Optional[int]:
    m = QUESTION_NUM_RE.match(text.strip())
    if not m:
        return None
    raw = next((g for g in m.groups() if g), None)
    if raw is None:
        return None
    try:
        num = int(raw)
    except Exception:
        return None
    if num >= 1000:
        return None
    return num


def detect_question_starts(
    lines: Sequence[Dict[str, float]],
    margin_abs: Optional[float],
    col_left: float,
    tol: float = 1.0,
    last_qnum: Optional[int] = None,
) -> Tuple[List[int], Optional[int]]:
    starts: List[int] = []
    target_rel = None if margin_abs is None else (margin_abs - col_left)
    current_last = last_qnum
    for i, ln in enumerate(lines):
        raw_text = ln.get("text") or ""
        stripped = raw_text.lstrip()
        if not stripped:
            continue
        if stripped[0] in OPTION_SET:
            continue
        text = stripped.rstrip()
        rel = ln["x0"] - col_left
        left_ok = True if target_rel is None else abs(rel - target_rel) <= tol
        text_ok = bool(QUESTION_START_LINE_RE.match(text))
        if not text_ok:
            continue
        qnum = _extract_qnum_from_text(text)
        seq_ok = qnum is not None and (
            current_last is None or qnum == current_last + 1
        )
        if left_ok or seq_ok:
            starts.append(i)
            if qnum is not None:
                current_last = qnum
    return starts, current_last


# =========================
# Chunk building
# =========================

def build_flow_segments(
    pdf: pdfplumber.pdf.PDF,
    extractor: PaddleOCRTextExtractor,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    y_tol: float,
    clip_mode: str,
    ycut_map: Dict[int, Optional[float]],
    band_map: Dict[int, Optional[Tuple[float, float, float, float]]],
):
    column_order = {"L": 0, "R": 1}
    segs = []
    for i, page in enumerate(pdf.pages):
        logger.debug("Building flow segments for page %d", i + 1)
        L, R = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        ycut = ycut_map.get(i + 1) if clip_mode == "ycut" else None
        band = band_map.get(i + 1) if clip_mode == "band" else None
        L_lines = extractor.extract_lines(
            i, L, y_tol=y_tol, y_cut=ycut, drop_zone=band, column_tag="L"
        )
        R_lines = extractor.extract_lines(
            i, R, y_tol=y_tol, y_cut=ycut, drop_zone=band, column_tag="R"
        )
        segs.append((i, "L", L, L_lines))
        segs.append((i, "R", R, R_lines))

    segs.sort(key=lambda entry: (entry[0], column_order.get(entry[1], 99)))
    return segs


def flow_chunk_all_pages(
    pdf: pdfplumber.pdf.PDF,
    extractor: PaddleOCRTextExtractor,
    L_rel_offset: Optional[float],
    R_rel_offset: Optional[float],
    y_tol: float,
    tol: float,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    clip_mode: str,
    ycut_map: Dict[int, Optional[float]],
    band_map: Dict[int, Optional[Tuple[float, float, float, float]]],
    *,
    skip_chunk_detection: bool = False,
):
    segs = build_flow_segments(
        pdf,
        extractor,
        top_frac,
        bottom_frac,
        gutter_frac,
        y_tol,
        clip_mode,
        ycut_map,
        band_map,
    )
    logger.debug("Built %d column segments", len(segs))

    seg_meta = []
    page_text_map: Dict[int, List[Dict[str, object]]] = {
        i: [] for i in range(len(pdf.pages))
    }
    for (pi, col, bbox, lines) in segs:
        L, R = two_col_bboxes(pdf.pages[pi], top_frac, bottom_frac, gutter_frac)
        if col == "L":
            margin_abs = None if L_rel_offset is None else (L[0] + L_rel_offset)
            col_left = L[0]
        else:
            margin_abs = None if R_rel_offset is None else (R[0] + R_rel_offset)
            col_left = R[0]
        if logger.isEnabledFor(logging.DEBUG):
            rel_offset = L_rel_offset if col == "L" else R_rel_offset
            logger.debug(
                "Page %d column %s bbox=%s margin_abs=%s rel_offset=%s",
                pi + 1,
                col,
                tuple(round(v, 2) for v in bbox),
                None if margin_abs is None else round(margin_abs, 2),
                None if rel_offset is None else round(rel_offset, 2),
            )
        seg_meta.append((pi, col, bbox, lines, margin_abs, col_left))
        if lines:
            col_lines = page_text_map.setdefault(pi, [])
            for line in lines:
                entry = dict(line)
                entry["column"] = col
                col_lines.append(entry)

    if skip_chunk_detection:
        logger.debug(
            "Skipping automatic chunk detection; returning OCR lines for %d segments.",
            len(seg_meta),
        )
        per_page_boxes = {i: [] for i in range(len(pdf.pages))}
        return [], per_page_boxes, page_text_map

    seg_starts = []
    last_detected_qnum = None
    for (pi, col, bbox, lines, m_abs, col_left) in seg_meta:
        starts, last_detected_qnum = detect_question_starts(
            lines, m_abs, col_left, tol=tol, last_qnum=last_detected_qnum
        )
        seg_starts.append(starts)
        logger.debug(
            "Detected %d candidate question starts on page %d column %s",
            len(starts),
            pi + 1,
            col,
        )

    chunks = []
    current = None
    for seg_idx, (pi, col, bbox, lines, m_abs, col_left) in enumerate(seg_meta):
        starts = set(seg_starts[seg_idx])
        i = 0
        while i < len(lines):
            if i in starts:
                if current is not None:
                    chunks.append(current)
                current = {
                    "pieces": [],
                    "start": {"page": pi, "col": col, "line_idx": i},
                }
            if current is not None:
                next_mark = min((j for j in starts if j > i), default=None)
                end_idx = (next_mark - 1) if next_mark is not None else (len(lines) - 1)
                block = lines[i : end_idx + 1]
                if block:
                    x0 = min(l["x0"] for l in block)
                    x1 = max(l["x1"] for l in block)
                    top = min(l["top"] for l in block) - 2.0
                    bot = max(l["bottom"] for l in block) + 2.0
                    text = "\n".join(l["text"] for l in block)
                    piece_lines = []
                    for l in block:
                        copy = dict(l)
                        copy.setdefault("column", col)
                        piece_lines.append(copy)
                    current["pieces"].append(
                        {
                            "page": pi,
                            "col": col,
                            "box": {"x0": x0, "x1": x1, "top": top, "bottom": bot},
                            "start_line": i,
                            "end_line": end_idx,
                            "text": text,
                            "lines": piece_lines,
                        }
                    )
                i = end_idx + 1
            else:
                i += 1
    if current is not None:
        chunks.append(current)
    logger.debug("Assembled %d candidate chunks", len(chunks))

    per_page_boxes = {i: [] for i in range(len(pdf.pages))}
    for ch_id, ch in enumerate(chunks, start=1):
        for p in ch.get("pieces", []):
            b = p["box"].copy()
            b["chunk_id"] = ch_id
            b["col"] = p["col"]
            per_page_boxes[p["page"]].append(b)

    return chunks, per_page_boxes, page_text_map


def _normalize_chunk_template(template: Dict[str, object]) -> Optional[Dict[str, object]]:
    pieces_in = template.get("pieces") if isinstance(template, dict) else None
    if not isinstance(pieces_in, list):
        return None

    qnum = template.get("question_number") if isinstance(template, dict) else None
    pieces: List[Dict[str, object]] = []
    for piece in pieces_in:
        if not isinstance(piece, dict):
            continue
        page = piece.get("page")
        try:
            page_index = int(page)
        except Exception:
            continue
        raw_box = piece.get("box") if isinstance(piece.get("box"), dict) else piece
        bbox = _coerce_bbox(raw_box if isinstance(raw_box, dict) else {})
        if not bbox:
            continue
        col = piece.get("col") or raw_box.get("col") if isinstance(raw_box, dict) else None
        normalized_piece: Dict[str, object] = {
            "page": page_index,
            "col": col or "?",
            "box": {
                "x0": bbox[0],
                "top": bbox[1],
                "x1": bbox[2],
                "bottom": bbox[3],
            },
        }

        if isinstance(piece.get("text"), str):
            normalized_piece["text"] = piece["text"]
        if isinstance(piece.get("lines"), list):
            normalized_piece["lines"] = [
                {**ln, "column": ln.get("column") or col}
                for ln in piece["lines"]
                if isinstance(ln, dict)
            ]
        if isinstance(piece.get("start_line"), int):
            normalized_piece["start_line"] = piece["start_line"]
        if isinstance(piece.get("end_line"), int):
            normalized_piece["end_line"] = piece["end_line"]

        pieces.append(normalized_piece)

    if not pieces:
        return None

    normalized = {"pieces": pieces}
    if qnum is not None:
        normalized["question_number"] = qnum
    return normalized


def load_chunk_templates_from_json(path: str) -> List[Dict[str, object]]:
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)

    with open(abs_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    templates: List[Dict[str, object]] = []

    def _maybe_add(entry: Optional[Dict[str, object]]):
        if entry:
            templates.append(entry)

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content") if isinstance(item.get("content"), dict) else None
            source = content.get("source") if content else None
            pieces = source.get("pieces") if isinstance(source, dict) else None
            template = {
                "pieces": pieces or item.get("pieces") or [],
                "question_number": (content or {}).get("question_number"),
            }
            _maybe_add(_normalize_chunk_template(template))
    elif isinstance(data, dict):
        raw_chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else None
        if raw_chunks is None:
            raw_chunks = [data] if data.get("pieces") else []
        for item in raw_chunks:
            _maybe_add(_normalize_chunk_template(item if isinstance(item, dict) else {}))

    return templates


def build_chunks_from_templates(
    templates: Sequence[Dict[str, object]],
    extractor: PaddleOCRTextExtractor,
    y_tol: float,
) -> List[Dict[str, object]]:
    built: List[Dict[str, object]] = []
    for tpl_idx, tpl in enumerate(templates, start=1):
        pieces: List[Dict[str, object]] = []
        raw_pieces = tpl.get("pieces") if isinstance(tpl, dict) else None
        if not isinstance(raw_pieces, list):
            continue
        for raw_piece in raw_pieces:
            if not isinstance(raw_piece, dict):
                continue
            bbox_dict = raw_piece.get("box") if isinstance(raw_piece.get("box"), dict) else raw_piece
            bbox = _coerce_bbox(bbox_dict if isinstance(bbox_dict, dict) else {})
            if not bbox:
                continue
            try:
                page_index = int(raw_piece.get("page"))
            except Exception:
                continue
            col_tag = raw_piece.get("col") or bbox_dict.get("col") if isinstance(bbox_dict, dict) else None
            extracted_lines = extractor.extract_lines(page_index, bbox, y_tol=y_tol, column_tag=col_tag)
            fallback_lines = raw_piece.get("lines") if isinstance(raw_piece.get("lines"), list) else []
            lines_with_fallback = extracted_lines or [
                {**ln, "page": page_index, "column": ln.get("column") or col_tag}
                for ln in fallback_lines
                if isinstance(ln, dict)
            ]
            lines_sorted = sorted(lines_with_fallback, key=lambda ln: (ln.get("top", 0.0), ln.get("x0", 0.0)))

            text_parts = [
                str(ln.get("text") or "")
                for ln in lines_sorted
                if str(ln.get("text") or "").strip()
            ]
            fallback_text = raw_piece.get("text") if isinstance(raw_piece.get("text"), str) else ""
            text = "\n".join(text_parts) if text_parts else fallback_text

            if lines_sorted:
                x0 = min(_safe_float(ln.get("x0"), 0.0) for ln in lines_sorted)
                x1 = max(_safe_float(ln.get("x1"), 0.0) for ln in lines_sorted)
                top = min(_safe_float(ln.get("top"), bbox[1]) for ln in lines_sorted)
                bottom = max(_safe_float(ln.get("bottom"), bbox[3]) for ln in lines_sorted)
            else:
                x0, top, x1, bottom = bbox
            pieces.append(
                {
                    "page": page_index,
                    "col": col_tag or "?",
                    "box": {"x0": x0, "top": top, "x1": x1, "bottom": bottom},
                    "start_line": 0,
                    "end_line": max(len(lines_sorted) - 1, 0),
                    "text": text or raw_piece.get("text") or "",
                    "lines": lines_sorted,
                }
            )

        if not pieces:
            continue

        chunk: Dict[str, object] = {"pieces": pieces}
        qnum = tpl.get("question_number") if isinstance(tpl, dict) else None
        if qnum is not None:
            chunk["manual"] = {"question_id": qnum}
        built.append(chunk)

    return built


def build_manual_chunks_from_regions(
    manual_regions: Sequence[ManualQuestionRegion],
    page_text_map: Dict[int, List[Dict[str, object]]],
    pdf: pdfplumber.pdf.PDF,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
) -> List[Dict[str, object]]:
    if not manual_regions:
        return []

    ordered = sorted(
        manual_regions,
        key=lambda region: (
            region.page_index,
            region.normalized_rect()[1],
            region.normalized_rect()[0],
            region.id,
        ),
    )
    chunks: List[Dict[str, object]] = []

    for region in ordered:
        page_index = region.page_index
        rect = region.normalized_rect()
        options_rect = region.normalized_options_rect()
        option_rects = {idx: box for idx, box in region.iter_option_rects()}
        captured: List[Dict[str, object]] = []

        for line in page_text_map.get(page_index, []):
            x0 = _safe_float(line.get("x0"), 0.0)
            x1 = _safe_float(line.get("x1"), 0.0)
            top = _safe_float(line.get("top"), 0.0)
            bottom = _safe_float(line.get("bottom"), 0.0)
            midpoint = ((x0 + x1) * 0.5, (top + bottom) * 0.5)

            area_tag: Optional[str] = None
            for opt_idx, opt_rect in option_rects.items():
                if _point_in_rect(midpoint, opt_rect):
                    area_tag = f"option:{opt_idx}"
                    break

            if area_tag is None and options_rect and _point_in_rect(midpoint, options_rect):
                area_tag = "options_block"

            if area_tag is None and _point_in_rect(midpoint, rect):
                area_tag = "stem"

            if area_tag is None:
                continue

            copy_line = dict(line)
            copy_line.setdefault("column", line.get("column"))
            copy_line["manual_area"] = area_tag
            captured.append(copy_line)

        if not captured:
            logger.warning(
                "Manual region %s on page %d captured no OCR lines.",
                region.id,
                page_index + 1,
            )
            continue

        captured.sort(key=lambda ln: (_safe_float(ln.get("top"), 0.0), _safe_float(ln.get("x0"), 0.0)))

        text_parts = [str(ln.get("text") or "").strip() for ln in captured]
        text = "\n".join(part for part in text_parts if part)
        if not text.strip():
            logger.warning(
                "Manual region %s on page %d produced empty text and will be skipped.",
                region.id,
                page_index + 1,
            )
            continue

        page = pdf.pages[page_index]
        L_bbox, R_bbox = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        center = _rect_center(rect)
        if _point_in_rect(center, L_bbox):
            col = "L"
        elif _point_in_rect(center, R_bbox):
            col = "R"
        else:
            col = "L" if center[0] <= float(page.width) * 0.5 else "R"

        piece_lines: List[Dict[str, object]] = []
        for line in captured:
            clone = dict(line)
            clone.setdefault("column", col)
            piece_lines.append(clone)

        chunk: Dict[str, object] = {
            "pieces": [
                {
                    "page": page_index,
                    "col": col,
                    "box": {
                        "x0": rect[0],
                        "x1": rect[2],
                        "top": rect[1],
                        "bottom": rect[3],
                    },
                    "start_line": 0,
                    "end_line": len(piece_lines) - 1,
                    "text": text,
                    "lines": piece_lines,
                    "manual": True,
                }
            ],
            "manual": {
                "question_rect": rect,
                "options_rect": options_rect,
                "option_rects": option_rects,
                "page_index": page_index,
                "question_id": region.id,
            },
            "start": {"page": page_index, "col": col, "line_idx": 0},
        }
        chunks.append(chunk)

    return chunks


# =========================
# QA extraction helpers
# =========================


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isinf(result) or math.isnan(result):
        return default
    return result


def _median_or_default(values: Sequence[float], default: float) -> float:
    filtered = [v for v in values if not math.isinf(v) and not math.isnan(v)]
    if not filtered:
        return default
    try:
        return float(statistics.median(filtered))
    except statistics.StatisticsError:
        return default


def _infer_options_from_lines(
    text: str, lines: Optional[Sequence[Dict[str, object]]]
) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    if not lines:
        return None

    cleaned: List[Dict[str, object]] = []
    for entry in lines:
        raw_text = entry.get("text") or ""
        normalized = norm_space(_normalize_visible_text(str(raw_text)))
        if not normalized:
            continue
        cleaned.append(
            {
                "text": normalized,
                "x0": _safe_float(entry.get("x0"), 0.0),
                "x1": _safe_float(entry.get("x1"), 0.0),
                "top": _safe_float(entry.get("top"), 0.0),
                "bottom": _safe_float(entry.get("bottom"), 0.0),
            }
        )

    if len(cleaned) < 4:
        return None

    q_start_idx = next(
        (i for i, item in enumerate(cleaned) if QUESTION_START_LINE_RE.match(item["text"])),
        0,
    )
    relevant = cleaned[q_start_idx:]
    if len(relevant) < 4:
        return None

    heights = [max(it["bottom"] - it["top"], 0.5) for it in relevant]
    line_height = _median_or_default(heights, 12.0)
    gap_threshold = max(line_height * 1.35, 6.0)

    option_start = None
    for idx in range(1, len(relevant)):
        prev = relevant[idx - 1]
        cur = relevant[idx]
        gap = cur["top"] - prev["bottom"]
        if gap > gap_threshold:
            option_start = idx
            break

    if option_start is None:
        total = len(relevant)
        for start in range(total - 3, 0, -1):
            tail = relevant[start:]
            if len(tail) < 3 or len(tail) > 7:
                continue
            short_ratio = sum(1 for item in tail if len(item["text"]) <= 36) / len(tail)
            if short_ratio >= 0.6:
                option_start = start
                break

    if option_start is None:
        return None

    stem_lines = relevant[:option_start]
    option_lines = relevant[option_start:]
    if len(option_lines) < 3:
        return None

    base_indent = _median_or_default([ln["x0"] for ln in option_lines], option_lines[0]["x0"])
    indent_threshold = max(2.5, line_height * 0.2)

    grouped: List[List[str]] = []
    current: List[str] = []
    for line in option_lines:
        txt = line["text"]
        indent = line["x0"]
        new_option = False
        if not current:
            new_option = True
        elif indent <= base_indent + indent_threshold:
            new_option = True
        else:
            new_option = False

        if new_option and current:
            grouped.append(current)
            current = []
        current.append(txt)

    if current:
        grouped.append(current)

    grouped = [grp for grp in grouped if grp]
    if len(grouped) < 3:
        return None

    stem_text = "\n".join(line["text"] for line in stem_lines).strip()
    if not stem_text:
        text_lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(text_lines) > len(grouped):
            stem_text = "\n".join(text_lines[: len(text_lines) - len(grouped)]).strip()

    options: List[Dict[str, str]] = []
    for idx, lines_group in enumerate(grouped, start=1):
        opt_text = norm_space(" ".join(lines_group))
        if not opt_text:
            continue
        circled = DIGIT_TO_CIRCLED.get(str(idx), str(idx))
        options.append({"index": circled, "text": opt_text})

    if len(options) < 3:
        return None

    return stem_text, options


def parse_dispute(stem: str, keep_text: bool = True):
    if not stem:
        return False, None, stem
    m = DISPUTE_RE.search(stem)
    if not m:
        return False, None, norm_space(stem)
    site = norm_space(m.group("site") or "")
    if keep_text:
        return True, (site or None), norm_space(stem)
    new_stem = norm_space(DISPUTE_RE.sub("", stem))
    return True, (site or None), new_stem


def extract_leading_qnum_and_clean(stem: str) -> Tuple[Optional[int], str]:
    if not stem:
        return None, stem
    m = QUESTION_NUM_RE.match(stem)
    if not m:
        return None, stem
    digits = next((g for g in m.groups() if g), None)
    try:
        qnum = int(digits) if digits is not None else None
    except Exception:
        qnum = None
    return qnum, stem[m.end() :].lstrip()


def _trim_to_first_question(text: str) -> Tuple[str, Optional[int]]:
    if not text:
        return text, None
    m = QUESTION_START_LINE_RE.search(text)
    if not m:
        return text, None
    trimmed = text[m.start() :]
    digits = None
    dm = QUESTION_NUM_RE.match(trimmed)
    if dm:
        raw = next((g for g in dm.groups() if g), None)
        try:
            digits = int(raw) if raw is not None else None
        except Exception:
            digits = None
    return trimmed, digits


def sanitize_chunk_text(text: str, expected_next_qnum: Optional[int]) -> str:
    if not text:
        return text

    text = _normalize_visible_text(text)
    trimmed, current_qnum = _trim_to_first_question(text)
    text = trimmed

    if current_qnum is not None:
        target_next = current_qnum + 1
    else:
        target_next = expected_next_qnum

    if target_next is None:
        return text

    for match in QUESTION_START_LINE_RE.finditer(text):
        if match.start() == 0:
            continue
        candidate_slice = text[match.start() :]
        dm = QUESTION_NUM_RE.match(candidate_slice)
        if not dm:
            continue
        raw = next((g for g in dm.groups() if g), None)
        if raw is None:
            continue
        try:
            num = int(raw)
        except Exception:
            continue
        if num >= 1000:
            continue
        if num == target_next:
            return text[: match.start()].rstrip()

    return text


def _extract_from_manual_annotations(
    text: str,
    lines: Optional[Sequence[Dict[str, object]]],
    manual: Dict[str, object],
):
    if not manual or not lines:
        return None

    line_list = list(lines)
    if not line_list:
        return None

    assignments: Dict[int, str] = {}
    for idx, line in enumerate(line_list):
        area = str(line.get("manual_area")) if line.get("manual_area") else None
        if area:
            assignments[idx] = area

    option_rects_raw = manual.get("option_rects") or {}
    option_rects: Dict[int, Tuple[float, float, float, float]] = {}
    for key, rect in option_rects_raw.items():
        try:
            idx = int(key)
        except Exception:
            continue
        option_rects[idx] = _normalize_rect(tuple(rect))

    options_rect = manual.get("options_rect")
    question_rect = manual.get("question_rect")
    options_rect_tuple = (
        _normalize_rect(tuple(options_rect)) if options_rect is not None else None
    )
    question_rect_tuple = (
        _normalize_rect(tuple(question_rect)) if question_rect is not None else None
    )

    for idx, line in enumerate(line_list):
        if idx in assignments:
            continue
        x0 = _safe_float(line.get("x0"), 0.0)
        x1 = _safe_float(line.get("x1"), 0.0)
        top = _safe_float(line.get("top"), 0.0)
        bottom = _safe_float(line.get("bottom"), 0.0)
        midpoint = ((x0 + x1) * 0.5, (top + bottom) * 0.5)

        area = None
        for opt_idx, rect in option_rects.items():
            if _point_in_rect(midpoint, rect):
                area = f"option:{opt_idx}"
                break
        if area is None and options_rect_tuple and _point_in_rect(midpoint, options_rect_tuple):
            area = "options_block"
        if area is None and question_rect_tuple and _point_in_rect(midpoint, question_rect_tuple):
            area = "stem"
        if area is not None:
            assignments[idx] = area

    stem_lines: List[Dict[str, object]] = []
    block_lines: List[Dict[str, object]] = []
    per_option_lines: Dict[int, List[Dict[str, object]]] = {}

    for idx, line in enumerate(line_list):
        tag = assignments.get(idx)
        if tag is None or tag == "stem":
            stem_lines.append(line)
        elif isinstance(tag, str) and tag.startswith("option:"):
            try:
                opt_idx = int(tag.split(":", 1)[1])
            except Exception:
                continue
            per_option_lines.setdefault(opt_idx, []).append(line)
        elif tag == "options_block":
            block_lines.append(line)
        else:
            stem_lines.append(line)

    stem_text = "\n".join(
        str(ln.get("text") or "").strip() for ln in stem_lines if str(ln.get("text") or "").strip()
    )
    if not stem_text.strip():
        stem_text = norm_space(_normalize_visible_text(text or ""))

    dispute, dispute_site, stem_text = parse_dispute(stem_text, keep_text=True)
    stem_text = norm_space(stem_text)
    detected_qnum, stem_text = extract_leading_qnum_and_clean(stem_text)
    stem_text = norm_space(stem_text)

    options: List[Dict[str, str]] = []
    for opt_idx in sorted(per_option_lines):
        opt_text = " ".join(
            str(ln.get("text") or "").strip()
            for ln in per_option_lines[opt_idx]
            if str(ln.get("text") or "").strip()
        )
        opt_text = norm_space(opt_text)
        if not opt_text:
            continue
        marker = DIGIT_TO_CIRCLED.get(str(opt_idx), str(opt_idx))
        options.append({"index": marker, "text": opt_text})

    if not options and block_lines:
        block_text = "\n".join(
            str(ln.get("text") or "").strip()
            for ln in block_lines
            if str(ln.get("text") or "").strip()
        )
        block_text = _normalize_option_markers(block_text)
        parts = [p for p in OPT_SPLIT_RE.split(block_text) if p]
        i = 0
        while i < len(parts):
            sym = parts[i].strip()
            if sym and sym[0] in OPTION_SET:
                raw_txt = parts[i + 1] if (i + 1) < len(parts) else ""
                clean_txt = norm_space(CIRCLED_STRIP_RE.sub("", raw_txt))
                options.append({"index": sym[0], "text": clean_txt})
                i += 2
            else:
                i += 1
        options = [opt for opt in options if opt["text"]]

    if not options:
        return None

    return stem_text, options, dispute, dispute_site, detected_qnum


def extract_qa_from_chunk_text(
    text: str,
    lines: Optional[Sequence[Dict[str, object]]] = None,
    manual: Optional[Dict[str, object]] = None,
):
    if not text:
        return None, None, False, None, None

    if manual:
        manual_result = _extract_from_manual_annotations(text, lines, manual)
        if manual_result is not None:
            return manual_result

    text = _normalize_option_markers(text)
    text = _strip_header_garbage(text)

    first_match = re.search(rf"[{OPTION_CLASS}]", text)
    if first_match:
        first = first_match.start()
        stem, opts_blob = text[:first], text[first:]

        dispute, dispute_site, stem = parse_dispute(stem, keep_text=True)
        stem = norm_space(stem)

        detected_qnum, stem = extract_leading_qnum_and_clean(stem)
        stem = norm_space(stem)

        parts = [p for p in OPT_SPLIT_RE.split(opts_blob) if p]
        options = []
        i = 0
        while i < len(parts):
            sym = parts[i].strip()
            if sym and sym[0] in OPTION_SET:
                raw_txt = parts[i + 1] if (i + 1) < len(parts) else ""
                clean_txt = norm_space(CIRCLED_STRIP_RE.sub("", raw_txt))
                options.append({"index": sym[0], "text": clean_txt})
                i += 2
            else:
                i += 1
        options = [o for o in options if o["index"] in OPTION_SET]
        if not options:
            return None, None, dispute, dispute_site, detected_qnum

        return stem, options, dispute, dispute_site, detected_qnum

    fallback = _infer_options_from_lines(text, lines)
    if not fallback:
        return None, None, False, None, None

    stem_text, fallback_options = fallback
    dispute, dispute_site, stem_text = parse_dispute(stem_text, keep_text=True)
    stem_text = norm_space(stem_text)
    detected_qnum, stem_text = extract_leading_qnum_and_clean(stem_text)
    stem_text = norm_space(stem_text)

    logger.debug(
        "Fallback option reconstruction applied (options=%d)", len(fallback_options)
    )

    return stem_text, fallback_options, dispute, dispute_site, detected_qnum


def format_qa_items_as_text(qa_items: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = []
    for item in qa_items or []:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue

        qnum = content.get("question_number")
        stem = content.get("question_text") or ""
        dispute = bool(content.get("dispute_bool"))
        dispute_site = content.get("dispute_site") or ""
        options = content.get("options") or []

        stem_lines = (stem.splitlines() if stem else [])
        if qnum is not None:
            if stem_lines:
                lines.append(f"{qnum}. {stem_lines[0]}".rstrip())
                lines.extend(stem_lines[1:])
            else:
                lines.append(f"{qnum}.")
        else:
            lines.extend(stem_lines)

        if dispute:
            tag = "[Dispute]" if not dispute_site else f"[Dispute] {dispute_site}"
            lines.append(tag.strip())

        for opt in options:
            if not isinstance(opt, dict):
                continue
            idx = (opt.get("index") or "").strip()
            text = opt.get("text") or ""
            opt_lines = text.splitlines() if text else [""]
            if opt_lines:
                first_line = opt_lines[0]
                prefix = f"{idx} " if idx else ""
                lines.append(f"{prefix}{first_line}".rstrip())
                lines.extend(opt_lines[1:])

        lines.append("")

    if lines and lines[-1] != "":
        lines.append("")

    return "\n".join(lines)


def sort_qa_items_by_question_number(
    qa_items: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return QA records ordered by their numeric question number."""

    def _coerce_question_number(value: object) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.isdigit():
                try:
                    return int(candidate)
                except ValueError:
                    return None
            match = re.match(r"(\d+)", candidate)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    enumerated: List[Tuple[int, Optional[int], Dict[str, Any]]] = []
    for idx, item in enumerate(qa_items or []):
        qnum: Optional[int] = None
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, dict):
                qnum = _coerce_question_number(content.get("question_number"))
        enumerated.append((idx, qnum, item))

    enumerated.sort(key=lambda entry: (0, entry[1]) if entry[1] is not None else (1, entry[0]))
    return [entry[2] for entry in enumerated]


@dataclass
class QAValidationResult:
    total_items: int
    numbered_items: int
    first_number: Optional[int]
    last_number: Optional[int]
    missing_numbers: List[int]
    duplicate_numbers: List[int]
    out_of_order_pairs: List[Tuple[int, int]]
    expected_count: Optional[int] = None
    start_number: Optional[int] = None

    def is_complete(self) -> bool:
        baseline_ok = self.total_items > 0 or (self.expected_count == 0)
        return (
            baseline_ok
            and not self.missing_numbers
            and not self.duplicate_numbers
            and not self.out_of_order_pairs
            and not self.count_mismatch()
        )

    def count_mismatch(self) -> bool:
        return self.expected_count is not None and self.total_items != self.expected_count


def validate_qa_sequence(
    qa_items: Sequence[Dict[str, object]],
    expected_count: Optional[int] = None,
    start_number: Optional[int] = None,
) -> QAValidationResult:
    items = list(qa_items or [])
    numbers: List[int] = []
    counts: Dict[int, int] = {}
    out_of_order: List[Tuple[int, int]] = []
    last_seen: Optional[int] = None

    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue
        qnum = content.get("question_number")
        if not isinstance(qnum, int):
            continue
        numbers.append(qnum)
        counts[qnum] = counts.get(qnum, 0) + 1
        if last_seen is not None and qnum < last_seen:
            out_of_order.append((last_seen, qnum))
        last_seen = qnum

    duplicates = sorted({num for num, count in counts.items() if count > 1})

    missing: List[int] = []
    if numbers:
        numbers_sorted = sorted(set(numbers))
        first = numbers_sorted[0]
        last = numbers_sorted[-1]
        missing.extend(n for n in range(first, last + 1) if counts.get(n, 0) == 0)
    else:
        first = start_number
        last = None

    if expected_count is not None and start_number is not None:
        expected_last = start_number + max(expected_count - 1, 0)
        missing.extend(
            n
            for n in range(start_number, expected_last + 1)
            if counts.get(n, 0) == 0 and n not in missing
        )
    missing.sort()

    return QAValidationResult(
        total_items=len(items),
        numbered_items=len(numbers),
        first_number=first,
        last_number=last,
        missing_numbers=missing,
        duplicate_numbers=duplicates,
        out_of_order_pairs=out_of_order,
        expected_count=expected_count,
        start_number=start_number,
    )


def log_validation_summary(result: QAValidationResult, label: str) -> None:
    base_msg = (
        f"Validation summary for {label}: {result.total_items} total items, "
        f"{result.numbered_items} numbered"
    )
    range_msg = ""
    if result.first_number is not None and result.last_number is not None:
        range_msg = f", range {result.first_number}-{result.last_number}"
    logger.info("%s%s", base_msg, range_msg)

    if result.total_items == 0:
        logger.warning("No QA items detected for %s", label)

    if result.count_mismatch():
        logger.warning(
            "Expected %d items but produced %d", result.expected_count, result.total_items
        )

    if result.missing_numbers:
        logger.warning("Missing question numbers: %s", ", ".join(map(str, result.missing_numbers)))

    if result.duplicate_numbers:
        logger.warning("Duplicate question numbers detected: %s", ", ".join(map(str, result.duplicate_numbers)))

    if result.out_of_order_pairs:
        formatted = ", ".join(f"{a}->{b}" for a, b in result.out_of_order_pairs)
        logger.warning("Out-of-order question numbers: %s", formatted)


# =========================
# Chunk preview images
# =========================

def save_chunk_preview(
    page,
    bbox,
    preview_dir,
    page_index,
    column_tag,
    chunk_idx_in_column,
    global_idx,
    dpi=220,
    pad=2.0,
):
    if not bbox or not preview_dir:
        return None
    abs_dir = os.path.abspath(os.path.expanduser(preview_dir))
    os.makedirs(abs_dir, exist_ok=True)
    width, height = float(page.width), float(page.height)
    x0, top, x1, bottom = map(float, bbox)
    pad = max(0.0, float(pad))
    padded = (
        max(0.0, x0 - pad),
        max(0.0, top - pad),
        min(width, x1 + pad),
        min(height, bottom + pad),
    )
    cropped = page.within_bbox(padded)
    img = cropped.to_image(resolution=int(dpi))
    pil = img.original.convert("RGB")
    del img
    fn = f"p{page_index:03d}_{column_tag}{chunk_idx_in_column:02d}_{global_idx:04d}.jpg"
    out_path = os.path.join(abs_dir, fn)
    pil.save(out_path, format="JPEG", quality=90)
    pil.close()
    return os.path.abspath(out_path)


def save_rasterized_pdf(pdf_path: str, out_path: str, dpi: int) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))
    images: List[Image.Image] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                logger.debug(
                    "Rasterizing page %d/%d at %d DPI for PDF export",
                    page_index,
                    len(pdf.pages),
                    dpi,
                )
                pil = page.to_image(resolution=int(dpi)).original.convert("RGB")
                images.append(pil)
    except Exception:
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        raise

    if not images:
        logger.warning("No pages rendered for raster PDF export of %s", pdf_path)
        return

    first, rest = images[0], images[1:]
    try:
        first.save(
            abs_out,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=int(dpi),
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def _resolve_searchable_font(
    font_spec: Optional[str],
    pdfmetrics,
):
    candidates: List[str] = []
    if font_spec:
        candidates.append(font_spec)
    candidates.extend(DEFAULT_SEARCHABLE_FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            try:
                from reportlab.pdfbase.ttfonts import TTFont  # type: ignore

                alias = os.path.splitext(os.path.basename(expanded))[0]
                pdfmetrics.registerFont(TTFont(alias, expanded))
                logger.info(
                    "Registered TrueType font %s for searchable PDF output", alias
                )
                return alias
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to register TrueType font %s: %s", expanded, exc
                )
                continue
        try:
            pdfmetrics.getFont(candidate)
            return candidate
        except KeyError:
            try:
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore

                pdfmetrics.registerFont(UnicodeCIDFont(candidate))
                logger.debug(
                    "Registered CID font %s for searchable PDF output", candidate
                )
                return candidate
            except KeyError:
                logger.debug(
                    "CID font %s not available in this ReportLab build", candidate
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to register CID font %s: %s", candidate, exc
                )
    try:
        pdfmetrics.getFont("Helvetica")
        logger.warning(
            "Falling back to Helvetica for searchable PDF text layer; CJK glyphs may be missing"
        )
        return "Helvetica"
    except KeyError:  # pragma: no cover - extremely unlikely
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        logger.warning(
            "Falling back to STSong-Light for searchable PDF text layer"
        )
        return "STSong-Light"


def save_searchable_pdf(
    pdf_path: str,
    out_path: str,
    dpi: int,
    page_text_map: Dict[int, List[Dict[str, object]]],
    font_spec: Optional[str] = None,
) -> None:
    try:
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Saving searchable PDFs requires the reportlab package"
        ) from exc

    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    font_name = _resolve_searchable_font(font_spec, pdfmetrics)

    with pdfplumber.open(pdf_path) as pdf:
        canv = canvas.Canvas(abs_out)
        for page_index, page in enumerate(pdf.pages):
            width, height = float(page.width), float(page.height)
            canv.setPageSize((width, height))
            pil = page.to_image(resolution=int(dpi)).original.convert("RGB")
            try:
                canv.drawImage(
                    ImageReader(pil),
                    0,
                    0,
                    width=width,
                    height=height,
                    mask=None,
                )
            finally:
                pil.close()

            lines = page_text_map.get(page_index) or []
            lines_sorted = sorted(lines, key=lambda ln: (ln.get("top", 0.0), ln.get("x0", 0.0)))
            for ln in lines_sorted:
                raw_text = ln.get("text")
                if not raw_text:
                    continue
                text = norm_space(str(raw_text))
                if not text:
                    continue
                try:
                    x0 = float(ln.get("x0", 0.0))
                    top = float(ln.get("top", 0.0))
                    bottom = float(ln.get("bottom", top))
                except (TypeError, ValueError):
                    continue
                line_height = max(bottom - top, 6.0)
                font_size = min(max(line_height * 0.95, 6.0), 36.0)
                baseline = height - top - (line_height * 0.2)
                text_obj = canv.beginText()
                text_obj.setTextRenderMode(3)  # invisible but searchable text layer
                text_obj.setFont(font_name, font_size)
                text_obj.setTextOrigin(x0, baseline)
                text_obj.textLine(text)
                canv.drawText(text_obj)

        canv.showPage()

    canv.save()


def save_bbox_overlay_pdf(
    pdf_path: str,
    out_path: str,
    dpi: int,
    word_map: Dict[int, List[Dict[str, object]]],
    line_map: Optional[Dict[int, List[Dict[str, object]]]] = None,
    word_color: Tuple[int, int, int] = (255, 0, 0),
    line_color: Tuple[int, int, int] = (0, 128, 255),
    question_box_map: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    question_box_color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    def measure_text(draw_obj: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        if hasattr(draw_obj, "textbbox"):
            bbox = draw_obj.textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        width, height = draw_obj.textsize(text, font=font)
        return int(width), int(height)

    images: List[Image.Image] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            scale = float(dpi) / 72.0
            page_total = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages):
                logger.debug(
                    "Rendering OCR overlay for page %d/%d at %d DPI",
                    page_index + 1,
                    page_total,
                    dpi,
                )
                base_img = page.to_image(resolution=int(dpi)).original.convert("RGB")
                draw = ImageDraw.Draw(base_img)
                stroke_words = max(1, int(round(dpi / 300)))
                stroke_lines = max(1, int(round(dpi / 360)))
                stroke_questions = max(1, int(round(dpi / 320)))

                words = word_map.get(page_index) or []
                for word in words:
                    try:
                        x0 = float(word.get("x0")) * scale
                        x1 = float(word.get("x1")) * scale
                        top = float(word.get("top")) * scale
                        bottom = float(word.get("bottom")) * scale
                    except (TypeError, ValueError):
                        continue
                    draw.rectangle((x0, top, x1, bottom), outline=word_color, width=stroke_words)

                if line_map:
                    lines = line_map.get(page_index) or []
                    for line in lines:
                        try:
                            x0 = float(line.get("x0")) * scale
                            x1 = float(line.get("x1")) * scale
                            top = float(line.get("top")) * scale
                            bottom = float(line.get("bottom", top)) * scale
                        except (TypeError, ValueError):
                            continue
                        draw.rectangle((x0, top, x1, bottom), outline=line_color, width=stroke_lines)

                question_entries = []
                if question_box_map:
                    question_entries = question_box_map.get(page_index) or []
                if question_entries:
                    font_size = max(12, int(round(dpi / 18)))
                    label_font: ImageFont.ImageFont
                    try:
                        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                    except Exception:
                        try:
                            label_font = ImageFont.truetype("Arial.ttf", font_size)
                        except Exception:
                            label_font = ImageFont.load_default()
                    label_pad = max(2, int(round(dpi / 220)))
                    def entry_sort_key(entry: Dict[str, Any]):
                        qn = entry.get("question_number")
                        try:
                            primary = int(qn)
                            flag = 0
                        except (TypeError, ValueError):
                            primary = str(qn or "")
                            flag = 1
                        box = entry.get("box") or (0.0, 0.0, 0.0, 0.0)
                        return (flag, primary, box[1])
                    for entry in sorted(
                        question_entries,
                        key=entry_sort_key,
                    ):
                        box = entry.get("box")
                        if not box or len(box) != 4:
                            continue
                        try:
                            x0 = float(box[0]) * scale
                            top = float(box[1]) * scale
                            x1 = float(box[2]) * scale
                            bottom = float(box[3]) * scale
                        except (TypeError, ValueError):
                            continue
                        draw.rectangle(
                            (x0, top, x1, bottom),
                            outline=question_box_color,
                            width=stroke_questions,
                        )
                        label = str(entry.get("question_number") or "?")
                        text_w, text_h = measure_text(draw, label, label_font)
                        tx = x0 + label_pad
                        ty = top + label_pad
                        bg_rect = (
                            tx - label_pad,
                            ty - label_pad,
                            tx + text_w + label_pad,
                            ty + text_h + label_pad,
                        )
                        draw.rectangle(bg_rect, fill=(0, 0, 0))
                        draw.text((tx, ty), label, fill=(255, 255, 255), font=label_font)

                del draw
                images.append(base_img)
    except Exception:
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        raise

    if not images:
        logger.warning("No pages rendered for OCR overlay PDF export of %s", pdf_path)
        return

    first, rest = images[0], images[1:]
    try:
        first.save(
            abs_out,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=int(dpi),
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def save_crop_report(
    pdf_path: str,
    out_path: str,
    crop_report: Dict[int, List[Dict[str, Any]]],
) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    def fmt_bbox(bbox: Sequence[float]) -> str:
        return "(%.2f, %.2f, %.2f, %.2f)" % tuple(float(v) for v in bbox)

    def fmt_band(band: Sequence[float]) -> str:
        return "(%.2f – %.2f)" % (float(band[0]), float(band[1]))

    lines: List[str] = []
    lines.append(f"Crop report for {os.path.basename(pdf_path)}")
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for page_index in sorted(crop_report.keys()):
        entries = crop_report.get(page_index) or []
        lines.append(f"Page {page_index + 1}")
        if not entries:
            lines.append("  (no OCR segments)")
            lines.append("")
            continue
        ordered = sorted(
            entries,
            key=lambda e: (
                0 if str(e.get("column", "")).upper() == "L" else 1,
                (e.get("orig_bbox") or (0.0, 0.0, 0.0, 0.0))[1],
            ),
        )
        for entry in ordered:
            col = str(entry.get("column") or "?").upper()
            orig = entry.get("orig_bbox") or (0.0, 0.0, 0.0, 0.0)
            padded = entry.get("padded_bbox") or orig
            band = entry.get("filter_band") or (0.0, 0.0)
            pad = entry.get("pad") or {}
            drop = entry.get("drop_zone")
            lines.append(f"  Column {col}:")
            lines.append(f"    original bbox : {fmt_bbox(orig)}")
            lines.append(f"    padded bbox   : {fmt_bbox(padded)}")
            lines.append(f"    filter band   : {fmt_band(band)}")
            lines.append(
                "    padding       : x=%s top=%s bottom=%s"
                % (
                    f"{float(pad.get('x', 0.0)):.2f}",
                    f"{float(pad.get('top', 0.0)):.2f}",
                    f"{float(pad.get('bottom', 0.0)):.2f}",
                )
            )
            if entry.get("y_cut") is not None:
                lines.append(f"    y-cut         : {float(entry['y_cut']):.2f}")
            if drop:
                lines.append(f"    drop zone     : {fmt_bbox(drop)}")
            lines.append(
                "    OCR words     : kept %d / raw %d"
                % (int(entry.get("kept_word_count", 0)), int(entry.get("raw_word_count", 0)))
            )
            lines.append(f"    line count    : {int(entry.get('line_count', 0))}")
        lines.append("")

    with open(abs_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


# =========================
# Top-level parse
# =========================

def pdf_to_qa_flow_chunks(
    pdf_path: str,
    year: int,
    start_num: int,
    L_rel: Optional[float],
    R_rel: Optional[float],
    tol: float,
    top_frac: float = DEFAULT_TOP_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
    gutter_frac: float = DEFAULT_GUTTER_FRAC,
    y_tol: float = 3.0,
    clip_mode: str = "none",
    chunk_preview_dir: Optional[str] = None,
    chunk_preview_dpi: int = 220,
    chunk_preview_pad: float = 2.0,
    ocr_settings: Optional[OCRSettings] = None,
    chunk_debug_dir: Optional[str] = None,
    failed_chunk_log_chars: int = 240,
    manual_questions: Optional[Sequence[ManualQuestionRegion]] = None,
    manual_only: bool = False,
    reuse_chunk_templates: Optional[Sequence[Dict[str, object]]] = None,
    reuse_chunk_source: Optional[str] = None,
):
    if ocr_settings is None:
        ocr_settings = OCRSettings()

    out: List[Dict[str, object]] = []
    last_assigned_qno = start_num - 1
    global_idx = 0
    preview_dir = (
        os.path.abspath(os.path.expanduser(chunk_preview_dir))
        if chunk_preview_dir
        else None
    )
    debug_dir = (
        os.path.abspath(os.path.expanduser(chunk_debug_dir))
        if chunk_debug_dir
        else None
    )
    if debug_dir:
        ensure_dir(debug_dir)

    page_text_map: Dict[int, List[Dict[str, object]]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(
            "Opened %s with %d pages (year=%s, start=%s)",
            os.path.basename(pdf_path),
            len(pdf.pages),
            year,
            start_num,
        )
        extractor = PaddleOCRTextExtractor(pdf, ocr_settings)
        ycut_map: Dict[int, Optional[float]] = {}
        band_map: Dict[int, Optional[Tuple[float, float, float, float]]] = {}

        page_text_map = {i: [] for i in range(len(pdf.pages))}

        if reuse_chunk_templates:
            if manual_questions:
                logger.info(
                    "Ignoring %d manual regions because saved chunk templates are being reused.",
                    len(manual_questions),
                )
            if manual_only:
                logger.info(
                    "Manual-only flag is overridden when reusing chunk templates.",
                )
            manual_only = False
            manual_questions = None
            chunks = build_chunks_from_templates(reuse_chunk_templates, extractor, y_tol)
            for ch in chunks:
                for piece in ch.get("pieces") or []:
                    page_index = int(piece.get("page", 0))
                    for ln in piece.get("lines") or []:
                        copy = dict(ln)
                        copy.setdefault("column", piece.get("col"))
                        page_text_map.setdefault(page_index, []).append(copy)
            for lines in page_text_map.values():
                lines.sort(key=lambda ln: (ln.get("top", 0.0), ln.get("x0", 0.0)))
            auto_chunk_count = len(chunks)
            logger.info(
                "Reused %d chunk boxes from %s", auto_chunk_count, reuse_chunk_source or "saved templates"
            )
        else:
            chunks, _, page_text_map = flow_chunk_all_pages(
                pdf,
                extractor,
                L_rel,
                R_rel,
                y_tol,
                tol,
                top_frac,
                bottom_frac,
                gutter_frac,
                clip_mode=clip_mode,
                ycut_map=ycut_map,
                band_map=band_map,
                skip_chunk_detection=manual_only,
            )
            auto_chunk_count = len(chunks)
            manual_region_count = len(manual_questions or [])
            if manual_only:
                if manual_region_count:
                    logger.info(
                        "Manual-only mode enabled: discarding %d auto-detected chunks in favor of %d manual regions.",
                        auto_chunk_count,
                        manual_region_count,
                    )
                elif auto_chunk_count:
                    logger.info(
                        "Manual-only mode enabled: discarding %d auto-detected chunks (no manual regions supplied).",
                        auto_chunk_count,
                    )
                else:
                    logger.info(
                        "Manual-only mode enabled: no manual regions supplied, no automatic chunks available.",
                    )
                chunks = []
            if manual_questions:
                if not manual_only:
                    logger.info(
                        "Manual chunk selection provided %d regions; replacing %d automatic chunks.",
                        manual_region_count,
                        auto_chunk_count,
                    )
                else:
                    logger.debug(
                        "Manual-only mode building %d chunks from manual regions.",
                        manual_region_count,
                    )
                chunks = build_manual_chunks_from_regions(
                    manual_questions,
                    page_text_map,
                    pdf,
                    top_frac,
                    bottom_frac,
                    gutter_frac,
                )
        logger.info("Detected %d raw chunks before QA filtering", len(chunks))

        for idx, ch in enumerate(chunks, start=1):
            pieces = ch.get("pieces") or []
            if not pieces:
                continue
            p1 = pieces[0]["page"] + 1

            expected_next = (
                last_assigned_qno + 1 if last_assigned_qno is not None else None
            )
            raw_text = "\n".join(p["text"] for p in pieces if p.get("text"))
            text = sanitize_chunk_text(raw_text, expected_next)
            debug_basename = f"p{p1:03d}_{pieces[0]['col']}_{idx:04d}"
            if debug_dir:
                raw_path = os.path.join(debug_dir, f"{debug_basename}_raw.txt")
                clean_path = os.path.join(debug_dir, f"{debug_basename}_clean.txt")
                with open(raw_path, "w", encoding="utf-8") as fh:
                    fh.write(raw_text)
                with open(clean_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            chunk_lines: List[Dict[str, object]] = []
            for piece in pieces:
                for ln in piece.get("lines") or []:
                    copy = dict(ln)
                    copy.setdefault("page", piece.get("page"))
                    chunk_lines.append(copy)
            if chunk_lines:
                chunk_lines.sort(key=lambda ln: (ln.get("top", 0.0), ln.get("x0", 0.0)))

            manual_info = ch.get("manual") if isinstance(ch, dict) else None
            manual_qno: Optional[int] = None
            if isinstance(manual_info, dict):
                raw_manual_q = manual_info.get("question_id")
                try:
                    manual_qno = int(raw_manual_q) if raw_manual_q is not None else None
                except Exception:
                    manual_qno = None

            if manual_only and not manual_info:
                logger.debug(
                    "Skipping chunk on page %d column %s: manual-only mode active and chunk lacks manual metadata.",
                    pieces[0]["page"] + 1,
                    pieces[0]["col"],
                )
                continue

            stem, options, dispute, dispute_site, detected_qnum = (
                extract_qa_from_chunk_text(text, chunk_lines, ch.get("manual"))
            )
            if stem is None or not options:
                snippet = text.strip().replace("\n", " ")
                if len(snippet) > failed_chunk_log_chars:
                    snippet = snippet[: failed_chunk_log_chars].rstrip() + "…"
                logger.warning(
                    "Skipping chunk starting on page %d column %s: no QA detected. Sample: %s",
                    pieces[0]["page"] + 1,
                    pieces[0]["col"],
                    snippet or "<empty>",
                )
                continue

            if manual_qno is not None:
                if (
                    detected_qnum is not None
                    and detected_qnum != manual_qno
                ):
                    logger.debug(
                        "Manual question %s differs from detected %s; using manual label.",
                        manual_qno,
                        detected_qnum,
                    )
                qno = manual_qno
            elif manual_only:
                logger.warning(
                    "Manual-only mode skipping chunk on page %d column %s: no question number assigned.",
                    pieces[0]["page"] + 1,
                    pieces[0]["col"],
                )
                continue
            elif detected_qnum is not None:
                qno = detected_qnum
            elif expected_next is not None:
                qno = expected_next
            else:
                qno = start_num
            global_idx += 1

            preview_path = None
            if preview_dir:
                b = pieces[0]["box"]
                bbox = (b["x0"], b["top"], b["x1"], b["bottom"])
                page = pdf.pages[pieces[0]["page"]]
                preview_path = save_chunk_preview(
                    page,
                    bbox,
                    preview_dir,
                    p1,
                    pieces[0]["col"],
                    1,
                    global_idx,
                    dpi=chunk_preview_dpi,
                    pad=chunk_preview_pad,
                )

            out.append(
                {
                    "year": year,
                    "content": {
                        "question_number": qno,
                        "question_text": stem,
                        "dispute_bool": bool(dispute),
                        "dispute_site": dispute_site,
                        "options": options,
                        "source": {"pieces": pieces, "start_page": p1},
                        **({"preview_image": preview_path} if preview_path else {}),
                    },
                }
            )

            last_assigned_qno = qno
            logger.debug(
                "Accepted chunk %d => question %d (%d options)",
                global_idx,
                qno,
                len(options),
            )

        logger.info("Accepted %d QA items after filtering", len(out))

        page_word_map: Dict[int, List[Dict[str, object]]] = {
            idx: extractor.page_word_boxes.get(idx, [])
            for idx in range(len(pdf.pages))
        }
        crop_report_map: Dict[int, List[Dict[str, Any]]] = {
            idx: extractor.crop_reports.get(idx, [])
            for idx in range(len(pdf.pages))
        }

    return out, page_text_map, page_word_map, crop_report_map


# =========================
# CLI helpers
# =========================

def _auto_detect_margins_for_pdf(
    pdf_path: str, top_frac: float, bottom_frac: float, gutter_frac: float
) -> Tuple[Optional[float], Optional[float]]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                Lb, Rb = two_col_bboxes(pg, top_frac, bottom_frac, gutter_frac)

                def first_x(bbox):
                    sub = pg.within_bbox(bbox)
                    xs = [c["x0"] for c in (sub.chars or []) if c.get("x0") is not None]
                    if not xs:
                        words = sub.extract_words(
                            x_tolerance=3, y_tolerance=3, keep_blank_chars=False
                        )
                        xs = [w["x0"] for w in words if w.get("x0") is not None]
                    return min(xs) if xs else None

                lx = first_x(Lb)
                rx = first_x(Rb)
                if lx is not None and rx is not None:
                    return float(lx - Lb[0]), float(rx - Rb[0])
    except Exception:
        pass
    return None, None


def interactive_chunk_selection(
    pdf_path: str,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    render_dpi: int = 200,
) -> Optional[List[ManualQuestionRegion]]:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Tkinter is required for the chunk selector") from exc

    try:
        from PIL import ImageTk
    except Exception as exc:  # pragma: no cover - Pillow build specific
        raise RuntimeError("Pillow ImageTk is required for the chunk selector") from exc

    pdf = pdfplumber.open(pdf_path)
    try:
        page_count = len(pdf.pages)
        if page_count == 0:
            raise RuntimeError("The PDF has no pages to annotate")

        state: Dict[str, Any] = {
            "zoom": 1.0,
            "page_index": 0,
            "base_img": None,
            "base_w": None,
            "base_h": None,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "L_bbox": None,
            "R_bbox": None,
            "questions": {},
            "selected_qid": None,
            "next_qid": 1,
            "expected_total": None,
            "available_qnums": [],
            "used_qnums": set(),
            "current_qnum": None,
            "drag_start": None,
            "drag_rect_id": None,
            "draw_mode": ("question", None),
            "shift_active": False,
            "active_option_digit": None,
            "result": None,
        }

        root = tk.Tk()
        root.title("Manual QA Chunk Selector")
    
        main_frame = ttk.Frame(root, padding=8)
        main_frame.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
    
        canvas = tk.Canvas(main_frame, background="#1f1f1f", highlightthickness=0)
        hbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        hbar.grid(row=1, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
        controls = ttk.Frame(main_frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(7, weight=1)
        controls.columnconfigure(8, weight=1)
    
        status_var = tk.StringVar(value="Drag to mark questions. Hold Shift for option blocks.")
        selection_var = tk.StringVar(value="None")
        mode_var = tk.StringVar(value="Question")
        page_info_var = tk.StringVar(value="Page 1 / %d" % page_count)
        zoom_var = tk.DoubleVar(value=100.0)
        zoom_display_var = tk.StringVar(value="100%")
        progress_var = tk.StringVar(value="Assigned 0 question(s). Current: -")
        total_questions_var = tk.StringVar()
        current_question_var = tk.StringVar()

        photo_cache: Dict[str, Any] = {}

        def gather_all_questions() -> List[ManualQuestionRegion]:
            items: List[ManualQuestionRegion] = []
            for qlist in state.get("questions", {}).values():
                items.extend(qlist)
            return items

        def recompute_used_numbers():
            used = {region.id for region in gather_all_questions()}
            state["used_qnums"] = used
            max_id = max(used) if used else 0
            state["next_qid"] = max(max_id + 1, state.get("next_qid", 1))

        def find_question_by_number(number: int) -> Optional[ManualQuestionRegion]:
            for qlist in state.get("questions", {}).values():
                for region in qlist:
                    if region.id == number:
                        return region
            return None

        def allocate_unused_question_id(
            skip_region: Optional[ManualQuestionRegion] = None,
        ) -> int:
            used = {
                region.id
                for region in gather_all_questions()
                if skip_region is None or region is not skip_region
            }
            expected = state.get("expected_total")
            if expected:
                for candidate in range(1, expected + 1):
                    if candidate not in used:
                        return candidate
            candidate = max(used) + 1 if used else 1
            while candidate in used:
                candidate += 1
            return candidate

        def remove_question_region(
            region: ManualQuestionRegion, *, quiet: bool = False
        ) -> bool:
            page_regions = questions_for_page(region.page_index)
            try:
                page_regions.remove(region)
            except ValueError:
                return False
            if state.get("selected_qid") == region.id:
                state["selected_qid"] = None
            recompute_used_numbers()
            recalc_number_state()
            update_selection_label()
            draw_overlays()
            if not quiet:
                set_status(
                    "Removed question #%d from page %d."
                    % (region.id, region.page_index + 1)
                )
            return True

        def resolve_number_conflict(
            value: int, keep: Optional[ManualQuestionRegion]
        ) -> Tuple[bool, Optional[str]]:
            existing = find_question_by_number(value)
            if existing is None or existing is keep:
                return True, None
            prompt = (
                "Question number %d is already assigned to page %d. "
                "Replace the existing region with the new one?"
                % (value, existing.page_index + 1)
            )
            if not messagebox.askyesno("Replace question region?", prompt):
                return False, None
            removed = remove_question_region(existing, quiet=True)
            if not removed:
                return False, None
            return True, (
                "Removed previous question #%d from page %d."
                % (existing.id, existing.page_index + 1)
            )

        def sync_current_entry():
            current = state.get("current_qnum")
            if current is None:
                current_question_var.set("")
            else:
                current_question_var.set(str(current))

        def recalc_number_state():
            expected = state.get("expected_total")
            used_set = state.get("used_qnums", set())
            if expected:
                available = [n for n in range(1, expected + 1) if n not in used_set]
                state["available_qnums"] = available
                current = state.get("current_qnum")
                if current is None or current not in available:
                    state["current_qnum"] = available[0] if available else None
                current = state.get("current_qnum")
                progress_var.set(
                    "Assigned %d/%d • Current: %s"
                    % (len(used_set), expected, str(current) if current is not None else "-")
                )
            else:
                state["available_qnums"] = []
                current = state.get("current_qnum")
                if current is None or current in used_set:
                    state["current_qnum"] = state.get("next_qid", 1)
                current = state.get("current_qnum")
                progress_var.set(
                    "Assigned %d question(s). Current: %s"
                    % (len(used_set), str(current) if current is not None else "-")
                )
            sync_current_entry()

        def pick_question_number() -> int:
            used_set = state.get("used_qnums", set())
            current = state.get("current_qnum")
            if current is not None and current not in used_set:
                return current
            expected = state.get("expected_total")
            if expected:
                available = [n for n in range(1, expected + 1) if n not in used_set]
                if available:
                    state["available_qnums"] = available
                    state["current_qnum"] = available[0]
                    sync_current_entry()
                    return available[0]
            hint = state.get("current_qnum")
            if hint is not None and hint not in used_set:
                return hint
            return state.get("next_qid", 1)

        def apply_total_count():
            raw = total_questions_var.get().strip()
            if not raw:
                state["expected_total"] = None
                recompute_used_numbers()
                recalc_number_state()
                set_status("Cleared expected question count; free-form numbering enabled.")
                return
            try:
                total = int(raw)
            except ValueError:
                messagebox.showerror("Invalid input", "Total questions must be an integer.")
                return
            if total <= 0:
                messagebox.showerror(
                    "Invalid count", "Total questions must be a positive integer."
                )
                return
            current_questions = gather_all_questions()
            if len(current_questions) > total:
                if not messagebox.askyesno(
                    "Reduce total?",
                    (
                        "You already have %d questions marked. Setting the total to %d "
                        "will leave some numbers unused. Continue?"
                    )
                    % (len(current_questions), total),
                ):
                    return
            state["expected_total"] = total
            recompute_used_numbers()
            recalc_number_state()
            set_status(f"Expecting {total} total questions; numbering updated.")

        def apply_current_question_number():
            raw = current_question_var.get().strip()
            if not raw:
                state["current_qnum"] = None
                recalc_number_state()
                set_status("Cleared current question number hint.")
                return
            try:
                value = int(raw)
            except ValueError:
                messagebox.showerror(
                    "Invalid input", "Current question number must be an integer."
                )
                return
            if value <= 0:
                messagebox.showerror(
                    "Invalid number", "Question numbers must be positive integers."
                )
                return
            expected = state.get("expected_total")
            if expected and not (1 <= value <= expected):
                messagebox.showerror(
                    "Out of range",
                    f"Question number must be between 1 and {expected}.",
                )
                return
            selected_region = get_selected_question()
            used_set = state.get("used_qnums", set())
            conflict_msg: Optional[str] = None
            if value in used_set and (
                selected_region is None or selected_region.id != value
            ):
                ok, conflict_msg = resolve_number_conflict(value, selected_region)
                if not ok:
                    sync_current_entry()
                    set_status(
                        f"Question number {value} remains assigned to its existing region."
                    )
                    return
                if conflict_msg:
                    recompute_used_numbers()
            state["current_qnum"] = value
            recalc_number_state()
            draw_overlays()
            update_selection_label()
            if conflict_msg:
                set_status(f"{conflict_msg} Current question number set to {value}.")
            else:
                set_status(f"Current question number set to {value}.")

        def assign_selected_question_number():
            q = get_selected_question()
            if q is None:
                messagebox.showwarning(
                    "No selection", "Select a question region before assigning a number."
                )
                return
            raw = current_question_var.get().strip()
            if not raw:
                messagebox.showerror(
                    "Missing number", "Enter a question number to assign to the selection."
                )
                return
            try:
                value = int(raw)
            except ValueError:
                messagebox.showerror("Invalid input", "Question number must be an integer.")
                return
            if value <= 0:
                messagebox.showerror(
                    "Invalid number", "Question numbers must be positive integers."
                )
                return
            expected = state.get("expected_total")
            if expected and not (1 <= value <= expected):
                messagebox.showerror(
                    "Out of range",
                    f"Question number must be between 1 and {expected}.",
                )
                return
            ok, conflict_msg = resolve_number_conflict(value, q)
            if not ok:
                set_status(
                    f"Assignment cancelled; question number {value} remains with its current region."
                )
                return
            q.id = value
            state["selected_qid"] = value
            recompute_used_numbers()
            recalc_number_state()
            draw_overlays()
            update_selection_label()
            if conflict_msg:
                set_status(
                    f"{conflict_msg} Assigned question #{value} to the selected region."
                )
            else:
                set_status(f"Assigned question #{value} to the selected region.")

        def pdf_to_canvas_coords(x: float, y: float) -> Tuple[float, float]:
            zoom = state.get("zoom", 1.0)
            return (
                x * state.get("scale_x", 1.0) * zoom,
                y * state.get("scale_y", 1.0) * zoom,
            )
    
        def canvas_to_pdf_coords(x: float, y: float) -> Tuple[float, float]:
            zoom = state.get("zoom", 1.0) or 1.0
            return (
                x / (state.get("scale_x", 1.0) * zoom),
                y / (state.get("scale_y", 1.0) * zoom),
            )
    
        def questions_for_page(page_index: int) -> List[ManualQuestionRegion]:
            return state.setdefault("questions", {}).setdefault(page_index, [])
    
        def get_selected_question() -> Optional[ManualQuestionRegion]:
            qid = state.get("selected_qid")
            if qid is None:
                return None
            for q in questions_for_page(state["page_index"]):
                if q.id == qid:
                    return q
            return None
    
        def update_selection_label():
            sel = get_selected_question()
            if sel is None:
                selection_var.set("None")
            else:
                selection_var.set(f"Page {sel.page_index + 1} / Question {sel.id}")
    
        def update_mode_label():
            mode, value = state.get("draw_mode", ("question", None))
            active_digit = state.get("active_option_digit")
            if mode == "option" and value:
                mode_var.set(f"Option {value}")
            elif active_digit:
                mode_var.set(f"Option {active_digit}")
            elif mode == "options_block":
                mode_var.set("Options Block")
            else:
                mode_var.set("Question")
    
        def set_status(message: str):
            status_var.set(message)
    
        def draw_overlays():
            canvas.delete("overlay")
            L_bbox = state.get("L_bbox")
            R_bbox = state.get("R_bbox")
            zoom = state.get("zoom", 1.0)
            if L_bbox:
                x0, y0 = pdf_to_canvas_coords(L_bbox[0], L_bbox[1])
                x1, y1 = pdf_to_canvas_coords(L_bbox[2], L_bbox[3])
                canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline="#2a82da",
                    width=1,
                    dash=(4, 4),
                    tags=("overlay", "column"),
                )
            if R_bbox:
                x0, y0 = pdf_to_canvas_coords(R_bbox[0], R_bbox[1])
                x1, y1 = pdf_to_canvas_coords(R_bbox[2], R_bbox[3])
                canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline="#da822a",
                    width=1,
                    dash=(4, 4),
                    tags=("overlay", "column"),
                )
    
            qlist = questions_for_page(state["page_index"])
            for region in qlist:
                rect = region.normalized_rect()
                x0, y0 = pdf_to_canvas_coords(rect[0], rect[1])
                x1, y1 = pdf_to_canvas_coords(rect[2], rect[3])
                width = 3 if region.id == state.get("selected_qid") else 2
                width = max(1, int(round(width * zoom)))
                color = "#5cb85c" if region.id == state.get("selected_qid") else "#3fa34d"
                canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline=color,
                    width=width,
                    tags=("overlay", "question"),
                )
                label = str(region.id)
                font_size = max(9, int(round(11 * zoom)))
                text_x = x0 + 6 * zoom
                text_y = y0 + 6 * zoom
                text_id = canvas.create_text(
                    text_x,
                    text_y,
                    text=label,
                    fill="#ffffff",
                    font=("Helvetica", font_size, "bold"),
                    anchor="nw",
                    tags=("overlay", "label"),
                )
                bbox = canvas.bbox(text_id)
                if bbox:
                    pad = max(2, int(round(4 * zoom)))
                    bg_id = canvas.create_rectangle(
                        bbox[0] - pad,
                        bbox[1] - pad,
                        bbox[2] + pad,
                        bbox[3] + pad,
                        fill="#1a1a1a",
                        outline="",
                        tags=("overlay", "label_bg"),
                    )
                    canvas.tag_lower(bg_id, text_id)

                opt_rect = region.normalized_options_rect()
                if opt_rect:
                    ox0, oy0 = pdf_to_canvas_coords(opt_rect[0], opt_rect[1])
                    ox1, oy1 = pdf_to_canvas_coords(opt_rect[2], opt_rect[3])
                    canvas.create_rectangle(
                        ox0,
                        oy0,
                        ox1,
                        oy1,
                        outline="#2fa8ff",
                        width=max(1, int(round(2 * zoom))),
                        dash=(6, 3),
                        tags=("overlay", "options"),
                    )

                for opt_idx, opt_box in region.iter_option_rects():
                    bx0, by0 = pdf_to_canvas_coords(opt_box[0], opt_box[1])
                    bx1, by1 = pdf_to_canvas_coords(opt_box[2], opt_box[3])
                    canvas.create_rectangle(
                        bx0,
                        by0,
                        bx1,
                        by1,
                        outline="#ff6f61",
                        width=max(1, int(round(2 * zoom))),
                        tags=("overlay", "opt"),
                    )
                    canvas.create_text(
                        bx0 + 6 * zoom,
                        by0 + 6 * zoom,
                        text=str(opt_idx),
                        fill="#ff6f61",
                        font=("Helvetica", max(8, int(round(10 * zoom))), "bold"),
                        anchor="nw",
                        tags=("overlay", "optlabel"),
                    )
    
        def refresh_image(*_):
            base_img = state.get("base_img")
            if base_img is None:
                return
            zoom = max(0.1, zoom_var.get() / 100.0)
            state["zoom"] = zoom
            base_w = state.get("base_w") or base_img.width
            base_h = state.get("base_h") or base_img.height
            disp_w = max(1, int(round(base_w * zoom)))
            disp_h = max(1, int(round(base_h * zoom)))
            resized = base_img.resize((disp_w, disp_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            photo_cache["image"] = photo
            if "image_id" not in photo_cache:
                photo_cache["image_id"] = canvas.create_image(0, 0, image=photo, anchor="nw")
            else:
                canvas.itemconfigure(photo_cache["image_id"], image=photo)
            canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
            zoom_display_var.set(f"{zoom * 100:.0f}%")
            draw_overlays()
    
        def select_question(region: Optional[ManualQuestionRegion]):
            state["selected_qid"] = region.id if region else None
            update_selection_label()
            draw_overlays()
    
        def hit_test_question(pdf_point: Tuple[float, float]) -> Optional[ManualQuestionRegion]:
            qlist = questions_for_page(state["page_index"])
            for region in reversed(qlist):
                if _point_in_rect(pdf_point, region.rect):
                    return region
            return None
    
        def clear_option_mode():
            state["active_option_digit"] = None
            update_mode_label()
    
        def on_button_press(event):
            canvas.focus_set()
            start = (canvas.canvasx(event.x), canvas.canvasy(event.y))
            state["drag_start"] = start
            mode = "question"
            value = None
            if state.get("active_option_digit"):
                mode = "option"
                value = state["active_option_digit"]
            elif state.get("shift_active"):
                mode = "options_block"
            state["draw_mode"] = (mode, value)
            outline = "#3fa34d"
            if mode == "options_block":
                outline = "#2fa8ff"
            elif mode == "option":
                outline = "#ff6f61"
            rect_id = canvas.create_rectangle(
                start[0],
                start[1],
                start[0],
                start[1],
                outline=outline,
                width=2,
                dash=(4, 4),
                tags=("overlay", "preview"),
            )
            state["drag_rect_id"] = rect_id
            update_mode_label()
    
        def on_drag(event):
            rect_id = state.get("drag_rect_id")
            start = state.get("drag_start")
            if rect_id is None or start is None:
                return
            cur = (canvas.canvasx(event.x), canvas.canvasy(event.y))
            canvas.coords(rect_id, start[0], start[1], cur[0], cur[1])

        def on_mousewheel(event):
            if hasattr(event, "delta") and event.delta:
                steps = max(1, abs(int(event.delta / 120)) if event.delta else 1)
                direction = -1 if event.delta > 0 else 1
                if event.state & 0x1:  # Shift for horizontal scroll
                    canvas.xview_scroll(direction * steps, "units")
                else:
                    canvas.yview_scroll(direction * steps, "units")
            elif getattr(event, "num", None) in (4, 5):
                direction = -1 if event.num == 4 else 1
                if event.state & 0x1:
                    canvas.xview_scroll(direction, "units")
                else:
                    canvas.yview_scroll(direction, "units")
            return "break"

        def remove_preview_rect():
            rect_id = state.get("drag_rect_id")
            if rect_id is not None:
                canvas.delete(rect_id)
            state["drag_rect_id"] = None
    
        def on_button_release(event):
            start = state.get("drag_start")
            if start is None:
                remove_preview_rect()
                return
            end = (canvas.canvasx(event.x), canvas.canvasy(event.y))
            remove_preview_rect()
            state["drag_start"] = None
    
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            if dx < 4 and dy < 4:
                pdf_point = canvas_to_pdf_coords(end[0], end[1])
                hit = hit_test_question(pdf_point)
                if hit:
                    select_question(hit)
                    set_status(
                        f"Selected question #{hit.id} on page {state['page_index'] + 1}."
                    )
                else:
                    set_status("No question under cursor; drag to create one.")
                return
    
            start_pdf = canvas_to_pdf_coords(start[0], start[1])
            end_pdf = canvas_to_pdf_coords(end[0], end[1])
            rect = _normalize_rect((start_pdf[0], start_pdf[1], end_pdf[0], end_pdf[1]))
            page_index = state["page_index"]
            mode, value = state.get("draw_mode", ("question", None))
    
            if mode == "question":
                qid = pick_question_number()
                region = ManualQuestionRegion(id=qid, page_index=page_index, rect=rect)
                questions_for_page(page_index).append(region)
                recompute_used_numbers()
                recalc_number_state()
                select_question(region)
                expected_total = state.get("expected_total")
                if expected_total and qid > expected_total:
                    set_status(
                        f"Added question #{qid} on page {page_index + 1} (beyond expected total)."
                    )
                else:
                    set_status(f"Added question #{qid} on page {page_index + 1}.")
            elif mode == "options_block":
                target = get_selected_question()
                if target is None:
                    set_status("Select a question before marking option blocks.")
                else:
                    target.options_rect = rect
                    set_status(
                        f"Set options block for question ID {target.id} on page {page_index + 1}."
                    )
            elif mode == "option" and value is not None:
                target = get_selected_question()
                if target is None:
                    set_status("Select a question before assigning option areas.")
                else:
                    try:
                        opt_idx = int(value)
                    except Exception:
                        opt_idx = None
                    if opt_idx is None:
                        set_status("Invalid option index.")
                    else:
                        target.option_rects[opt_idx] = rect
                        set_status(
                            f"Assigned option {opt_idx} for question ID {target.id}."
                        )
                clear_option_mode()
            else:
                set_status("Unknown drawing mode; nothing recorded.")
    
            draw_overlays()
    
        def delete_selected():
            q = get_selected_question()
            if q is None:
                set_status("No question selected to delete.")
                return
            if remove_question_region(q, quiet=True):
                set_status("Deleted selected question region.")
            else:
                set_status("Unable to delete the selected question region.")
    
        def clear_selected_options():
            q = get_selected_question()
            if q is None:
                set_status("No question selected to clear options.")
                return
            q.options_rect = None
            q.option_rects.clear()
            draw_overlays()
            set_status("Cleared option regions for selected question.")
    
        def on_key_press(event):
            if event.keysym in ("Shift_L", "Shift_R"):
                state["shift_active"] = True
                update_mode_label()
                return
            if event.char and event.char.isdigit() and event.char != "0":
                state["active_option_digit"] = int(event.char)
                update_mode_label()
                set_status(f"Assigning option {event.char}; drag to mark its area.")
                return
            if event.keysym in ("Escape", "Return"):
                clear_option_mode()
                set_status("Exited option assignment mode.")
                return
            if event.keysym in ("Delete", "BackSpace"):
                delete_selected()
                return
            if event.keysym.lower() == "c" and (event.state & 0x4):  # Ctrl+C to clear options
                clear_selected_options()
    
        def on_key_release(event):
            if event.keysym in ("Shift_L", "Shift_R"):
                state["shift_active"] = False
                update_mode_label()
    
        def change_page(delta: int):
            new_index = state["page_index"] + delta
            if not (0 <= new_index < page_count):
                return
            load_page(new_index)
    
        def load_page(idx: int):
            if not (0 <= idx < page_count):
                return
            page = pdf.pages[idx]
            base_img = page.to_image(resolution=int(render_dpi)).original.convert("RGB")
            base_w, base_h = base_img.size
            pdf_w, pdf_h = float(page.width), float(page.height)
            state["base_img"] = base_img
            state["base_w"] = base_w
            state["base_h"] = base_h
            state["scale_x"] = base_w / pdf_w if pdf_w else 1.0
            state["scale_y"] = base_h / pdf_h if pdf_h else 1.0
            state["page_index"] = idx
            state["L_bbox"], state["R_bbox"] = two_col_bboxes(
                page, top_frac, bottom_frac, gutter_frac
            )
            if state.get("selected_qid") not in [q.id for q in questions_for_page(idx)]:
                state["selected_qid"] = None
            page_info_var.set(f"Page {idx + 1} / {page_count}")
            update_selection_label()
            refresh_image()
            set_status("Drag to mark question areas. Use number keys for options.")
    
        def confirm():
            collected: List[ManualQuestionRegion] = []
            for page_idx, qlist in state.get("questions", {}).items():
                ordered = sorted(
                    qlist,
                    key=lambda region: (
                        region.normalized_rect()[1],
                        region.normalized_rect()[0],
                        region.id,
                    ),
                )
                collected.extend(ordered)
            if not collected:
                if not messagebox.askyesno(
                        "No selections", "No question regions were selected. Continue?"
                ):
                    return
            state["result"] = collected
            root.destroy()
    
        def cancel():
            state["result"] = None
            root.destroy()
    
        ttk.Label(controls, text="Zoom:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            controls,
            from_=10,
            to=300,
            orient="horizontal",
            variable=zoom_var,
            command=lambda _evt: refresh_image(),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(controls, textvariable=zoom_display_var, width=6).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Separator(controls, orient="vertical").grid(row=0, column=3, sticky="ns", padx=6)
        ttk.Label(controls, text="Mode:").grid(row=0, column=4, sticky="w")
        ttk.Label(controls, textvariable=mode_var, width=14).grid(row=0, column=5, sticky="w")
        ttk.Label(controls, text="Selected:").grid(row=0, column=6, sticky="e")
        ttk.Label(controls, textvariable=selection_var, width=18).grid(
            row=0, column=7, sticky="w", padx=(4, 0)
        )
        ttk.Button(controls, text="Delete", command=delete_selected).grid(
            row=0, column=8, padx=(8, 4)
        )
        ttk.Button(controls, text="Clear options", command=clear_selected_options).grid(
            row=0, column=9, padx=(4, 4)
        )
        ttk.Button(controls, text="Cancel", command=cancel).grid(row=0, column=10, padx=(12, 4))
        ttk.Button(controls, text="Apply", command=confirm).grid(row=0, column=11)
    
        ttk.Separator(controls, orient="horizontal").grid(
            row=1, column=0, columnspan=12, sticky="ew", pady=(6, 6)
        )
        ttk.Label(controls, textvariable=page_info_var).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Button(controls, text="◀ Prev", command=lambda: change_page(-1)).grid(
            row=2, column=3, padx=(4, 4)
        )
        ttk.Button(controls, text="Next ▶", command=lambda: change_page(1)).grid(
            row=2, column=4, padx=(4, 8)
        )
        ttk.Label(
            controls,
            text="Tips: Left drag = question, Shift+drag = option block, number key + drag = option.",
        ).grid(row=2, column=5, columnspan=7, sticky="w", padx=(8, 0))

        ttk.Label(controls, text="Total questions:").grid(row=3, column=0, sticky="w")
        total_entry = ttk.Entry(controls, textvariable=total_questions_var, width=6)
        total_entry.grid(row=3, column=1, sticky="w")
        ttk.Button(controls, text="Apply total", command=apply_total_count).grid(
            row=3, column=2, padx=(4, 8), sticky="w"
        )
        total_entry.bind("<Return>", lambda _evt: apply_total_count())
        ttk.Label(controls, textvariable=progress_var).grid(
            row=3, column=3, columnspan=2, sticky="w"
        )
        ttk.Label(controls, text="Current number:").grid(row=3, column=5, sticky="e")
        number_entry = ttk.Entry(controls, textvariable=current_question_var, width=6)
        number_entry.grid(row=3, column=6, sticky="w")
        number_entry.bind("<Return>", lambda _evt: apply_current_question_number())
        number_entry.bind("<FocusOut>", lambda _evt: apply_current_question_number())
        ttk.Button(
            controls,
            text="Assign selected",
            command=assign_selected_question_number,
        ).grid(row=3, column=7, padx=(4, 0), sticky="w")
    
        status_label = ttk.Label(main_frame, textvariable=status_var, anchor="w")
        status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
    
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        canvas.bind("<KeyPress>", on_key_press)
        canvas.bind("<KeyRelease>", on_key_release)
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Shift-MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)
        canvas.bind("<Enter>", lambda _evt: canvas.focus_set())
        canvas.focus_set()
        root.bind("<KeyPress>", on_key_press)
        root.bind("<KeyRelease>", on_key_release)
        root.protocol("WM_DELETE_WINDOW", cancel)

        recompute_used_numbers()
        recalc_number_state()
        load_page(0)
        root.mainloop()

        result = state.get("result")
        if result is not None:
            logger.info(
                "Interactive chunk selector captured %d question regions.",
                len(result),
            )
        return result
    finally:
        pdf.close()


def interactive_margin_selection(
    pdf_path: str,
    page_index: int,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    render_dpi: int = 200,
    initial_left: Optional[float] = None,
    initial_right: Optional[float] = None,
) -> Tuple[float, float]:
    """Interactive GUI to pick left/right margins on a preview image."""

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # pragma: no cover - Tkinter availability depends on env
        raise RuntimeError("Tkinter is required for the margin selector") from exc

    try:
        from PIL import ImageTk
    except Exception as exc:  # pragma: no cover - Pillow build specific
        raise RuntimeError("Pillow ImageTk is required for the margin selector") from exc

    if page_index < 0:
        raise ValueError("page_index must be >= 0")

    pdf = pdfplumber.open(pdf_path)
    try:
        page_count = len(pdf.pages)
        if page_index >= page_count:
            raise ValueError("page_index exceeds total page count")
    except Exception:
        pdf.close()
        raise

    state = {
        "zoom": 1.0,
        "page_index": page_index,
        "left_offset": initial_left,
        "right_offset": initial_right,
        "left_pdf_x": None,
        "right_pdf_x": None,
        "base_img": None,
        "base_w": None,
        "base_h": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "L_bbox": None,
        "R_bbox": None,
        "result": None,
    }

    root = tk.Tk()
    root.title("Manual Margin Selector")

    main_frame = ttk.Frame(root, padding=8)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    canvas = tk.Canvas(main_frame, background="#222", highlightthickness=0)
    hbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
    vbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    hbar.grid(row=1, column=0, sticky="ew")
    vbar.grid(row=0, column=1, sticky="ns")
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    controls = ttk.Frame(main_frame)
    controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    controls.columnconfigure(6, weight=1)

    zoom_var = tk.DoubleVar(value=100.0)
    zoom_display_var = tk.StringVar(value="100%")
    selection_var = tk.StringVar(value="L")
    status_var = tk.StringVar(value="Click inside a column to set a margin line.")

    photo_cache = {"image": None}

    def pdf_to_canvas_coords(x: float, y: float) -> Tuple[float, float]:
        zoom = state["zoom"]
        return (
            x * state["scale_x"] * zoom,
            y * state["scale_y"] * zoom,
        )

    def refresh_image(*_):
        base_img = state.get("base_img")
        if base_img is None:
            return
        zoom = zoom_var.get() / 100.0
        if zoom <= 0:
            zoom = 0.1
        state["zoom"] = zoom
        base_w = state.get("base_w") or base_img.width
        base_h = state.get("base_h") or base_img.height
        disp_w = max(1, int(round(base_w * zoom)))
        disp_h = max(1, int(round(base_h * zoom)))
        resized = base_img.resize((disp_w, disp_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        photo_cache["image"] = photo
        if "image_id" not in photo_cache:
            photo_cache["image_id"] = canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            canvas.itemconfigure(photo_cache["image_id"], image=photo)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        zoom_display_var.set(f"{zoom * 100:.0f}%")
        draw_overlays()

    def draw_overlays():
        canvas.delete("overlay")
        zoom = state["zoom"]
        L_bbox = state.get("L_bbox")
        R_bbox = state.get("R_bbox")
        if L_bbox is None or R_bbox is None:
            return
        for tag, bbox, color in (
            ("L", L_bbox, "#33aaff"),
            ("R", R_bbox, "#ff9933"),
        ):
            x0, y0 = pdf_to_canvas_coords(bbox[0], bbox[1])
            x1, y1 = pdf_to_canvas_coords(bbox[2], bbox[3])
            canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=color,
                width=max(1, int(2 * zoom)),
                tags="overlay",
            )
        if state["left_pdf_x"] is not None:
            x = pdf_to_canvas_coords(state["left_pdf_x"], L_bbox[1])[0]
            y0 = pdf_to_canvas_coords(0, L_bbox[1])[1]
            y1 = pdf_to_canvas_coords(0, L_bbox[3])[1]
            canvas.create_line(
                x,
                y0,
                x,
                y1,
                fill="#00ffff",
                width=max(1, int(3 * zoom)),
                tags="overlay",
            )
            canvas.create_text(
                x + 6 * zoom,
                y0 + 6 * zoom,
                text=f"L: {state['left_pdf_x']:.1f} pt",
                anchor="nw",
                fill="#00ffff",
                font=("TkDefaultFont", max(8, int(9 * zoom))),
                tags="overlay",
            )
        if state["right_pdf_x"] is not None:
            x = pdf_to_canvas_coords(state["right_pdf_x"], R_bbox[1])[0]
            y0 = pdf_to_canvas_coords(0, R_bbox[1])[1]
            y1 = pdf_to_canvas_coords(0, R_bbox[3])[1]
            canvas.create_line(
                x,
                y0,
                x,
                y1,
                fill="#ff66aa",
                width=max(1, int(3 * zoom)),
                tags="overlay",
            )
            canvas.create_text(
                x + 6 * zoom,
                y0 + 6 * zoom,
                text=f"R: {state['right_pdf_x']:.1f} pt",
                anchor="nw",
                fill="#ff66aa",
                font=("TkDefaultFont", max(8, int(9 * zoom))),
                tags="overlay",
            )

    def on_click(event):
        canvas.focus_set()
        canvas_x = canvas.canvasx(event.x)
        canvas_y = canvas.canvasy(event.y)
        zoom = state["zoom"]
        pdf_x = canvas_x / (state["scale_x"] * zoom)
        pdf_y = canvas_y / (state["scale_y"] * zoom)
        L_bbox = state.get("L_bbox")
        R_bbox = state.get("R_bbox")
        if L_bbox is None or R_bbox is None:
            return
        side = selection_var.get()
        if side == "L":
            if not (L_bbox[0] <= pdf_x <= L_bbox[2]):
                status_var.set("Click inside the highlighted left column.")
                return
            state["left_pdf_x"] = pdf_x
            state["left_offset"] = pdf_x - L_bbox[0]
            status_var.set(f"Left margin set at x={pdf_x:.2f} pt (y={pdf_y:.2f})")
        else:
            if not (R_bbox[0] <= pdf_x <= R_bbox[2]):
                status_var.set("Click inside the highlighted right column.")
                return
            state["right_pdf_x"] = pdf_x
            state["right_offset"] = pdf_x - R_bbox[0]
            status_var.set(f"Right margin set at x={pdf_x:.2f} pt (y={pdf_y:.2f})")
        draw_overlays()

    def on_key(event):
        if event.char.lower() == "l":
            selection_var.set("L")
        elif event.char.lower() == "r":
            selection_var.set("R")

    def confirm():
        if state["left_offset"] is None or state["right_offset"] is None:
            messagebox.showwarning(
                "Incomplete", "Set both left and right margins before continuing."
            )
            return
        state["result"] = (state["left_offset"], state["right_offset"])
        root.destroy()

    def cancel():
        state["result"] = None
        root.destroy()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Key>", on_key)
    canvas.focus_set()
    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", cancel)

    ttk.Label(controls, text="Zoom:").grid(row=0, column=0, sticky="w")
    zoom_scale = ttk.Scale(
        controls,
        from_=10,
        to=300,
        orient="horizontal",
        variable=zoom_var,
        command=lambda _evt: refresh_image(),
    )
    zoom_scale.grid(row=0, column=1, sticky="ew", padx=(4, 12))
    ttk.Label(controls, textvariable=zoom_display_var, width=6).grid(
        row=0, column=2, sticky="w"
    )

    ttk.Label(controls, text="Select margin:").grid(row=0, column=3, padx=(8, 0))
    ttk.Radiobutton(
        controls, text="Left (L)", value="L", variable=selection_var
    ).grid(row=0, column=4, padx=(4, 4))
    ttk.Radiobutton(
        controls, text="Right (R)", value="R", variable=selection_var
    ).grid(row=0, column=5, padx=(4, 4))

    ttk.Button(controls, text="Reset", command=lambda: reset_lines()).grid(
        row=0, column=6, sticky="w"
    )
    ttk.Button(controls, text="Cancel", command=cancel).grid(
        row=0, column=7, padx=(12, 4)
    )
    ttk.Button(controls, text="Apply", command=confirm).grid(row=0, column=8)

    page_info_var = tk.StringVar()

    def change_page(delta: int):
        new_index = state["page_index"] + delta
        if new_index < 0 or new_index >= page_count:
            return
        load_page(new_index)

    ttk.Separator(controls, orient="horizontal").grid(
        row=1, column=0, columnspan=9, sticky="ew", pady=(6, 6)
    )
    ttk.Label(controls, textvariable=page_info_var).grid(
        row=2, column=0, columnspan=3, sticky="w"
    )
    ttk.Button(controls, text="◀ Prev", command=lambda: change_page(-1)).grid(
        row=2, column=3, padx=(4, 4)
    )
    ttk.Button(controls, text="Next ▶", command=lambda: change_page(1)).grid(
        row=2, column=4, padx=(4, 12)
    )
    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(6, weight=1)

    status = ttk.Label(main_frame, textvariable=status_var, anchor="w")
    status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def reset_lines():
        state["left_pdf_x"] = None
        state["right_pdf_x"] = None
        state["left_offset"] = None
        state["right_offset"] = None
        status_var.set("Margins cleared. Click to set new positions.")
        draw_overlays()

    def load_page(idx: int):
        if not (0 <= idx < page_count):
            return
        page = pdf.pages[idx]
        L_bbox, R_bbox = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        base_img = page.to_image(resolution=int(render_dpi)).original.convert("RGB")
        base_w, base_h = base_img.size
        pdf_w, pdf_h = float(page.width), float(page.height)
        scale_x = base_w / pdf_w if pdf_w else 1.0
        scale_y = base_h / pdf_h if pdf_h else 1.0

        state["page_index"] = idx
        state["base_img"] = base_img
        state["base_w"] = base_w
        state["base_h"] = base_h
        state["scale_x"] = scale_x
        state["scale_y"] = scale_y
        state["L_bbox"] = L_bbox
        state["R_bbox"] = R_bbox
        if state["left_offset"] is not None:
            state["left_pdf_x"] = L_bbox[0] + state["left_offset"]
        else:
            state["left_pdf_x"] = None
        if state["right_offset"] is not None:
            state["right_pdf_x"] = R_bbox[0] + state["right_offset"]
        else:
            state["right_pdf_x"] = None

        page_info_var.set(f"Page {idx + 1} / {page_count}")
        status_var.set(
            "Click inside a column to set a margin line."
            if state["left_offset"] is None or state["right_offset"] is None
            else "Margins loaded. Adjust as needed then click Apply."
        )
        refresh_image()

    try:
        load_page(state["page_index"])

        root.mainloop()

        if state["result"] is None:
            raise RuntimeError("Margin selection cancelled")
        return state["result"]
    finally:
        pdf.close()


def process_single_pdf(
    pdf_path: str,
    out_path: str,
    year: Optional[int],
    start_num: int,
    tol: float,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    clip_mode: str,
    margin_left: Optional[float],
    margin_right: Optional[float],
    preview_dir: Optional[str],
    preview_dpi: int,
    preview_pad: float,
    ocr_settings: OCRSettings,
    margin_ui: bool = False,
    margin_ui_page: int = 1,
    margin_ui_dpi: int = 200,
    chunk_ui: bool = False,
    chunk_ui_dpi: int = 200,
    heartbeat_interval: float = 30.0,
    chunk_debug_dir: Optional[str] = None,
    failed_chunk_log_chars: int = 240,
    raster_pdf_path: Optional[str] = None,
    raster_pdf_dpi: Optional[int] = None,
    searchable_pdf_path: Optional[str] = None,
    searchable_pdf_dpi: Optional[int] = None,
    searchable_pdf_font: Optional[str] = None,
    bbox_overlay_pdf_path: Optional[str] = None,
    bbox_overlay_pdf_dpi: Optional[int] = None,
    text_out_path: Optional[str] = None,
    crop_report_path: Optional[str] = None,
    expected_question_count: Optional[int] = None,
    fail_on_incomplete: bool = False,
    manual_only: bool = False,
    reuse_chunks_from: Optional[str] = None,
):
    if year is None:
        year = infer_year_from_filename(pdf_path) or datetime.now().year

    if margin_ui:
        try:
            margin_left, margin_right = interactive_margin_selection(
                pdf_path=pdf_path,
                page_index=max(0, margin_ui_page - 1),
                top_frac=top_frac,
                bottom_frac=bottom_frac,
                gutter_frac=gutter_frac,
                render_dpi=margin_ui_dpi,
                initial_left=margin_left,
                initial_right=margin_right,
            )
            logger.info(
                "Interactive margins selected: L=%.2f R=%.2f",
                margin_left,
                margin_right,
            )
        except RuntimeError as exc:
            logger.exception("Margin selector failed: %s", exc)
            sys.exit(3)
    elif margin_left is None or margin_right is None:
        auto_l, auto_r = _auto_detect_margins_for_pdf(
            pdf_path, top_frac, bottom_frac, gutter_frac
        )
        logger.debug(
            "Auto-detected margins for %s: L=%s R=%s",
            os.path.basename(pdf_path),
            "None" if auto_l is None else f"{auto_l:.2f}",
            "None" if auto_r is None else f"{auto_r:.2f}",
        )
        margin_left = margin_left if margin_left is not None else auto_l
        margin_right = margin_right if margin_right is not None else auto_r

    logger.info(
        "Processing %s with margins L=%s R=%s",
        os.path.basename(pdf_path),
        "auto" if margin_left is None else f"{margin_left:.2f}",
        "auto" if margin_right is None else f"{margin_right:.2f}",
    )

    manual_regions: Optional[List[ManualQuestionRegion]] = None
    if chunk_ui:
        try:
            manual_regions = interactive_chunk_selection(
                pdf_path=pdf_path,
                top_frac=top_frac,
                bottom_frac=bottom_frac,
                gutter_frac=gutter_frac,
                render_dpi=chunk_ui_dpi,
            )
        except RuntimeError as exc:
            logger.exception("Chunk selector failed: %s", exc)
            sys.exit(4)
        if manual_regions is None:
            logger.info(
                "Manual chunk selection cancelled; continuing with automatic detection."
            )
        elif not manual_regions:
            logger.warning(
                "Manual chunk selector returned zero regions; no questions will be extracted unless regions are added."
            )
        else:
            logger.info(
                "Manual chunk override enabled with %d regions.",
                len(manual_regions),
            )

    if manual_only and not manual_regions:
        logger.warning(
            "Manual-only mode enabled but no manual regions were provided; no questions will be extracted."
        )

    heartbeat = HeartbeatLogger(
        interval=heartbeat_interval,
        message=f"Still processing {os.path.basename(pdf_path)}...",
    )
    heartbeat.start()

    reuse_templates: Optional[List[Dict[str, object]]] = None
    reuse_source: Optional[str] = None
    if reuse_chunks_from:
        reuse_source = os.path.abspath(os.path.expanduser(reuse_chunks_from))
        reuse_templates = load_chunk_templates_from_json(reuse_source)
        logger.info(
            "Loaded %d chunk templates from %s for reuse",
            len(reuse_templates),
            reuse_source,
        )

    crop_report_map: Dict[int, List[Dict[str, Any]]] = {}
    try:
        qa, page_text_map, page_word_map, crop_report_map = pdf_to_qa_flow_chunks(
            pdf_path=pdf_path,
            year=year,
            start_num=start_num,
            L_rel=margin_left,
            R_rel=margin_right,
            tol=tol,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
            gutter_frac=gutter_frac,
            y_tol=3.0,
            clip_mode=clip_mode,
            chunk_preview_dir=preview_dir,
            chunk_preview_dpi=preview_dpi,
            chunk_preview_pad=preview_pad,
            ocr_settings=ocr_settings,
            chunk_debug_dir=chunk_debug_dir,
            failed_chunk_log_chars=failed_chunk_log_chars,
            manual_questions=manual_regions if manual_regions is not None else None,
            manual_only=manual_only,
            reuse_chunk_templates=reuse_templates,
            reuse_chunk_source=reuse_source,
        )
    finally:
        heartbeat.stop()

    sorted_qa = sort_qa_items_by_question_number(qa)
    def _qnum_from_item(entry: Dict[str, Any]) -> Optional[object]:
        if isinstance(entry, dict):
            content = entry.get("content")
            if isinstance(content, dict):
                return content.get("question_number")
        return None

    if [_qnum_from_item(item) for item in qa] != [
        _qnum_from_item(item) for item in sorted_qa
    ]:
        logger.info("Reordered QA output by question number before export")
    qa = sorted_qa

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d QA items to %s", len(qa), out_path)

    if text_out_path:
        ensure_dir(os.path.dirname(os.path.abspath(text_out_path)))
        with open(text_out_path, "w", encoding="utf-8") as fh:
            fh.write(format_qa_items_as_text(qa))
        logger.info("Wrote QA text export to %s", text_out_path)

    if crop_report_path:
        save_crop_report(pdf_path, crop_report_path, crop_report_map)
        logger.info("Wrote column crop report to %s", crop_report_path)

    if raster_pdf_path:
        dpi = raster_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        save_rasterized_pdf(pdf_path, raster_pdf_path, dpi)
        logger.info("Saved rasterized PDF preview to %s", raster_pdf_path)
    if searchable_pdf_path:
        dpi = searchable_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        save_searchable_pdf(
            pdf_path,
            searchable_pdf_path,
            dpi,
            page_text_map,
            font_spec=searchable_pdf_font,
        )
        logger.info("Saved searchable OCR PDF to %s", searchable_pdf_path)
    if bbox_overlay_pdf_path:
        dpi = bbox_overlay_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        overlay_map: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
        for item in qa:
            content = item.get("content") or {}
            qnum = content.get("question_number")
            if qnum is None:
                continue
            try:
                qnum_int = int(qnum)
            except (TypeError, ValueError):
                qnum_int = qnum
            pieces = (content.get("source") or {}).get("pieces") or []
            for piece in pieces:
                try:
                    page_index = int(piece.get("page", 0))
                except (TypeError, ValueError):
                    continue
                box = piece.get("box") or {}
                x0 = _safe_float(box.get("x0"), 0.0)
                top = _safe_float(box.get("top"), 0.0)
                x1 = _safe_float(box.get("x1"), 0.0)
                bottom = _safe_float(box.get("bottom"), 0.0)
                per_page = overlay_map.setdefault(page_index, {})
                prev = per_page.get(qnum_int)
                if prev:
                    per_page[qnum_int] = (
                        min(prev[0], x0),
                        min(prev[1], top),
                        max(prev[2], x1),
                        max(prev[3], bottom),
                    )
                else:
                    per_page[qnum_int] = (x0, top, x1, bottom)
        def _overlay_sort_key(item: Tuple[object, Tuple[float, float, float, float]]):
            key, _bbox = item
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        question_box_map = {
            page: [
                {"question_number": qnum, "box": bbox}
                for qnum, bbox in sorted(entries.items(), key=_overlay_sort_key)
            ]
            for page, entries in overlay_map.items()
        }
        save_bbox_overlay_pdf(
            pdf_path,
            bbox_overlay_pdf_path,
            dpi,
            page_word_map,
            page_text_map,
            question_box_map=question_box_map,
        )
        logger.info("Saved OCR bounding box overlay PDF to %s", bbox_overlay_pdf_path)
    validation = validate_qa_sequence(qa, expected_question_count, start_num)
    log_validation_summary(validation, os.path.basename(pdf_path))
    if fail_on_incomplete and not validation.is_complete():
        raise RuntimeError(
            "Validation failed: gaps, duplicates, or count mismatch detected in parsed QA items"
        )

    return qa, validation


# =========================
# Main CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Parse QA pairs from exam PDFs using PaddleOCR for text extraction."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Input single PDF file")
    g.add_argument("--pdf-dir", help="Folder containing PDFs (non-recursive)")

    ap.add_argument("--out", required=True, help="Output JSON path or folder")
    ap.add_argument("--year", type=int, help="Year of the exam; inferred from filename if omitted")
    ap.add_argument("--start-num", type=int, default=1, help="Starting question number")

    ap.add_argument("--tol", type=float, default=1.0, help="Margin match tolerance (pt)")
    ap.add_argument(
        "--top-frac",
        type=float,
        default=DEFAULT_TOP_FRAC,
        help="Top fraction for the column window (default: 0.10)",
    )
    ap.add_argument(
        "--bottom-frac",
        type=float,
        default=DEFAULT_BOTTOM_FRAC,
        help="Bottom fraction for the column window (default: 0.90)",
    )
    ap.add_argument(
        "--gutter-frac",
        type=float,
        default=DEFAULT_GUTTER_FRAC,
        help="Half-width of the gutter between columns as a fraction of page width",
    )

    ap.add_argument(
        "--clip-mode",
        choices=["none", "band", "ycut"],
        default="none",
        help="Optional header clipping mode",
    )
    ap.add_argument("--margin-left", type=float, help="Left column margin offset (pt)")
    ap.add_argument("--margin-right", type=float, help="Right column margin offset (pt)")

    ap.add_argument("--chunk-preview-dir", help="Save JPEG previews of detected chunks")
    ap.add_argument("--chunk-preview-dpi", type=int, default=220)
    ap.add_argument("--chunk-preview-pad", type=float, default=2.0)
    ap.add_argument(
        "--chunk-debug-dir",
        help="Directory to dump raw and sanitized chunk text for debugging",
    )
    ap.add_argument(
        "--failed-chunk-log-chars",
        type=int,
        default=240,
        help="Max characters of chunk text to include in skip warnings",
    )
    ap.add_argument(
        "--reuse-chunks-from",
        help=(
            "Path to a prior QA JSON or chunk template file whose bounding boxes should be reused instead of re-detecting"
        ),
    )

    ap.add_argument(
        "--paddleocr-dpi", type=int, default=1000, help="Rendering DPI for PaddleOCR"
    )
    ap.add_argument(
        "--paddleocr-langs",
        default="korean",
        help="Comma-separated PaddleOCR language codes",
    )
    ap.add_argument(
        "--paddleocr-gpu",
        action="store_true",
        help="Enable GPU acceleration for PaddleOCR if available",
    )
    ap.add_argument(
        "--paddleocr-column-pad-x",
        type=float,
        default=2.0,
        help="Horizontal padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--paddleocr-column-pad-top",
        type=float,
        default=2.0,
        help="Top padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--paddleocr-column-pad-bottom",
        type=float,
        default=60.0,
        help="Bottom padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--paddleocr-column-filter-top",
        type=float,
        default=4.0,
        help="Extra tolerance (pt) above the column window when keeping OCR words",
    )
    ap.add_argument(
        "--paddleocr-column-filter-bottom",
        type=float,
        default=60.0,
        help="Extra tolerance (pt) below the column window when keeping OCR words",
    )

    ap.add_argument(
        "--paddleocr-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply light denoising/sharpening before OCR (default: enabled)",
    )
    ap.add_argument(
        "--paddleocr-preprocess-contrast",
        type=float,
        default=1.25,
        help="Contrast boost factor applied during OCR preprocessing",
    )
    ap.add_argument(
        "--paddleocr-preprocess-unsharp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply an unsharp mask during OCR preprocessing",
    )
    ap.add_argument(
        "--paddleocr-preprocess-threshold",
        type=int,
        help="Optional binarization threshold (0-255) applied after sharpening/contrast",
    )
    ap.add_argument(
        "--paddleocr-korean-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Korean lexicon-based cleanup to OCR text",
    )
    ap.add_argument(
        "--paddleocr-korean-lexicon",
        help="Path to a newline-delimited lexicon used for Korean OCR cleanup",
    )

    ap.add_argument(
        "--raster-pdf-out",
        help="Path or directory to save a rasterized copy of each processed PDF",
    )
    ap.add_argument(
        "--raster-pdf-dpi",
        type=int,
        help="DPI to use for raster PDF export (defaults to PaddleOCR DPI)",
    )

    ap.add_argument(
        "--searchable-pdf-out",
        help="Path or directory to save a searchable OCR PDF for each input",
    )
    ap.add_argument(
        "--searchable-pdf-dpi",
        type=int,
        help="DPI to use when rasterizing pages for the searchable PDF layer",
    )
    ap.add_argument(
        "--searchable-pdf-font",
        help=(
            "Font name (CID) or path to a TTF/OTF file for the searchable PDF text layer"
        ),
    )
    ap.add_argument(
        "--bbox-overlay-pdf-out",
        help="Path or directory to save a PDF with OCR bounding boxes overlaid",
    )
    ap.add_argument(
        "--bbox-overlay-pdf-dpi",
        type=int,
        help="DPI to use for the bounding-box overlay PDF (defaults to PaddleOCR DPI)",
    )

    ap.add_argument(
        "--text-out",
        help="Path or directory to save a combined plain-text export of parsed QAs",
    )

    ap.add_argument(
        "--crop-report-out",
        help="Path or directory to save a text report describing each column crop",
    )

    ap.add_argument(
        "--expected-question-count",
        type=int,
        help="Expected number of QA items; used for validation reporting",
    )
    ap.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help=(
            "Exit with an error if validation finds missing, duplicate, out-of-order, or mismatched question counts"
        ),
    )

    ap.add_argument(
        "--margin-ui",
        action="store_true",
        help="Launch an interactive preview to pick left/right margins",
    )
    ap.add_argument(
        "--margin-ui-page",
        type=int,
        default=1,
        help="1-indexed page to preview when choosing margins",
    )
    ap.add_argument(
        "--margin-ui-dpi",
        type=int,
        default=200,
        help="Rendering DPI for the margin preview window",
    )

    ap.add_argument(
        "--chunk-ui",
        action="store_true",
        help=(
            "Launch an interactive tool to draw question/option regions that override automatic chunking"
        ),
    )
    ap.add_argument(
        "--chunk-ui-dpi",
        type=int,
        default=200,
        help="Rendering DPI for the manual chunk selection window",
    )

    ap.add_argument(
        "--manual-only",
        action="store_true",
        help="Skip automatic chunk detection and rely solely on manual annotations",
    )

    ap.add_argument(
        "--log-level",
        default=_parse_log_level("INFO"),
        type=_parse_log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    ap.add_argument(
        "--heartbeat-secs",
        type=float,
        default=30.0,
        help="Seconds between progress heartbeat log messages",
    )

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info(
        "Log level set to %s",
        logging.getLevelName(args.log_level),
    )

    if args.pdf:
        pdfs = [args.pdf]
        out_paths = [args.out]
        batch_mode = False
        base_pdf = os.path.splitext(os.path.basename(args.pdf))[0]
        if args.raster_pdf_out:
            if os.path.isdir(args.raster_pdf_out) or args.raster_pdf_out.endswith(os.sep):
                target_dir = args.raster_pdf_out
                ensure_dir(target_dir)
                raster_paths = [
                    os.path.join(target_dir, f"{base_pdf}_raster.pdf")
                ]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.raster_pdf_out)))
                raster_paths = [args.raster_pdf_out]
        else:
            raster_paths = [None]

        if args.searchable_pdf_out:
            if os.path.isdir(args.searchable_pdf_out) or args.searchable_pdf_out.endswith(os.sep):
                target_dir = args.searchable_pdf_out
                ensure_dir(target_dir)
                searchable_paths = [
                    os.path.join(target_dir, f"{base_pdf}_searchable.pdf")
                ]
            else:
                ensure_dir(
                    os.path.dirname(os.path.abspath(args.searchable_pdf_out))
                )
                searchable_paths = [args.searchable_pdf_out]
        else:
            searchable_paths = [None]

        if args.bbox_overlay_pdf_out:
            if os.path.isdir(args.bbox_overlay_pdf_out) or args.bbox_overlay_pdf_out.endswith(os.sep):
                target_dir = args.bbox_overlay_pdf_out
                ensure_dir(target_dir)
                bbox_overlay_paths = [
                    os.path.join(target_dir, f"{base_pdf}_ocr_overlay.pdf")
                ]
            else:
                ensure_dir(
                    os.path.dirname(os.path.abspath(args.bbox_overlay_pdf_out))
                )
                bbox_overlay_paths = [args.bbox_overlay_pdf_out]
        else:
            bbox_overlay_paths = [None]

        if args.text_out:
            if os.path.isdir(args.text_out) or args.text_out.endswith(os.sep):
                target_dir = args.text_out
                ensure_dir(target_dir)
                text_paths = [os.path.join(target_dir, f"{base_pdf}.txt")]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.text_out)))
                text_paths = [args.text_out]
        else:
            text_paths = [None]

        if args.crop_report_out:
            if os.path.isdir(args.crop_report_out) or args.crop_report_out.endswith(os.sep):
                target_dir = args.crop_report_out
                ensure_dir(target_dir)
                crop_report_paths = [
                    os.path.join(target_dir, f"{base_pdf}_crop_report.txt")
                ]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.crop_report_out)))
                crop_report_paths = [args.crop_report_out]
        else:
            crop_report_paths = [None]
    else:
        pdfs = list_pdfs(args.pdf_dir)
        if not pdfs:
            logger.error("No PDFs found in %s", args.pdf_dir)
            sys.exit(2)
        ensure_dir(args.out)
        out_paths = [
            os.path.join(
                args.out,
                os.path.splitext(os.path.basename(p))[0] + ".json",
            )
            for p in pdfs
        ]
        batch_mode = True
        logger.info("Processing %d PDFs from %s", len(pdfs), args.pdf_dir)
        if args.raster_pdf_out:
            ensure_dir(args.raster_pdf_out)
            raster_paths = [
                os.path.join(
                    args.raster_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_raster.pdf",
                )
                for p in pdfs
            ]
        else:
            raster_paths = [None] * len(pdfs)

        if args.searchable_pdf_out:
            ensure_dir(args.searchable_pdf_out)
            searchable_paths = [
                os.path.join(
                    args.searchable_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_searchable.pdf",
                )
                for p in pdfs
            ]
        else:
            searchable_paths = [None] * len(pdfs)

        if args.bbox_overlay_pdf_out:
            ensure_dir(args.bbox_overlay_pdf_out)
            bbox_overlay_paths = [
                os.path.join(
                    args.bbox_overlay_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_ocr_overlay.pdf",
                )
                for p in pdfs
            ]
        else:
            bbox_overlay_paths = [None] * len(pdfs)

        if args.text_out:
            ensure_dir(args.text_out)
            text_paths = [
                os.path.join(
                    args.text_out,
                    os.path.splitext(os.path.basename(p))[0] + ".txt",
                )
                for p in pdfs
            ]
        else:
            text_paths = [None] * len(pdfs)

        if args.crop_report_out:
            ensure_dir(args.crop_report_out)
            crop_report_paths = [
                os.path.join(
                    args.crop_report_out,
                    os.path.splitext(os.path.basename(p))[0]
                    + "_crop_report.txt",
                )
                for p in pdfs
            ]
        else:
            crop_report_paths = [None] * len(pdfs)

    korean_lexicon: Optional[Sequence[str]] = None
    if args.paddleocr_korean_lexicon:
        try:
            with open(
                os.path.abspath(os.path.expanduser(args.paddleocr_korean_lexicon)),
                "r",
                encoding="utf-8",
            ) as fh:
                korean_lexicon = [line.strip() for line in fh if line.strip()]
                logger.info(
                    "Loaded %d lexicon entries from %s",
                    len(korean_lexicon),
                    args.paddleocr_korean_lexicon,
                )
        except OSError as exc:
            logger.warning(
                "Failed to load Korean lexicon from %s: %s",
                args.paddleocr_korean_lexicon,
                exc,
            )

    ocr_settings = OCRSettings(
        dpi=args.paddleocr_dpi,
        languages=[lang.strip() for lang in args.paddleocr_langs.split(",") if lang.strip()],
        gpu=args.paddleocr_gpu,
        column_pad_x=args.paddleocr_column_pad_x,
        column_pad_top=args.paddleocr_column_pad_top,
        column_pad_bottom=args.paddleocr_column_pad_bottom,
        column_filter_top=args.paddleocr_column_filter_top,
        column_filter_bottom=args.paddleocr_column_filter_bottom,
        preprocess_images=args.paddleocr_preprocess,
        preprocess_contrast=args.paddleocr_preprocess_contrast,
        preprocess_unsharp=args.paddleocr_preprocess_unsharp,
        preprocess_threshold=args.paddleocr_preprocess_threshold,
        korean_postprocess=args.paddleocr_korean_postprocess,
        korean_lexicon=korean_lexicon if korean_lexicon is not None else DEFAULT_KOREAN_LEXICON,
    )

    for (
        pdf_path,
        out_path,
        raster_path,
        searchable_path,
        bbox_overlay_path,
        text_path,
        crop_report_path,
    ) in zip(
        pdfs,
        out_paths,
        raster_paths,
        searchable_paths,
        bbox_overlay_paths,
        text_paths,
        crop_report_paths,
    ):
        logger.info("Processing %s -> %s", os.path.basename(pdf_path), out_path)
        if raster_path:
            logger.info(
                "Rasterized PDF output will be saved to %s",
                raster_path,
            )
        if searchable_path:
            logger.info(
                "Searchable OCR PDF will be saved to %s",
                searchable_path,
            )
        if bbox_overlay_path:
            logger.info(
                "OCR bounding-box overlay will be saved to %s",
                bbox_overlay_path,
            )
        if text_path:
            logger.info("QA text export will be saved to %s", text_path)
        if crop_report_path:
            logger.info(
                "Column crop report will be saved to %s", crop_report_path
            )
        debug_dir = None
        if args.chunk_debug_dir:
            base_debug = os.path.abspath(os.path.expanduser(args.chunk_debug_dir))
            if batch_mode:
                ensure_dir(base_debug)
                debug_dir = os.path.join(
                    base_debug, os.path.splitext(os.path.basename(pdf_path))[0]
                )
            else:
                debug_dir = base_debug
            ensure_dir(debug_dir)
        try:
            qa, validation = process_single_pdf(
                pdf_path=pdf_path,
                out_path=out_path,
                year=args.year,
                start_num=args.start_num,
                tol=args.tol,
                top_frac=args.top_frac,
                bottom_frac=args.bottom_frac,
                gutter_frac=args.gutter_frac,
                clip_mode=args.clip_mode,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                preview_dir=args.chunk_preview_dir,
                preview_dpi=args.chunk_preview_dpi,
                preview_pad=args.chunk_preview_pad,
                ocr_settings=ocr_settings,
                margin_ui=args.margin_ui,
                margin_ui_page=args.margin_ui_page,
                margin_ui_dpi=args.margin_ui_dpi,
                chunk_ui=args.chunk_ui,
                chunk_ui_dpi=args.chunk_ui_dpi,
                heartbeat_interval=args.heartbeat_secs,
                chunk_debug_dir=debug_dir,
                failed_chunk_log_chars=args.failed_chunk_log_chars,
                raster_pdf_path=raster_path,
                raster_pdf_dpi=args.raster_pdf_dpi,
                searchable_pdf_path=searchable_path,
                searchable_pdf_dpi=args.searchable_pdf_dpi,
                searchable_pdf_font=args.searchable_pdf_font,
                bbox_overlay_pdf_path=bbox_overlay_path,
                bbox_overlay_pdf_dpi=args.bbox_overlay_pdf_dpi,
                text_out_path=text_path,
                crop_report_path=crop_report_path,
                expected_question_count=args.expected_question_count,
                fail_on_incomplete=args.fail_on_incomplete,
                manual_only=args.manual_only,
                reuse_chunks_from=args.reuse_chunks_from,
            )
        except RuntimeError as exc:
            logger.error("Validation failed for %s: %s", pdf_path, exc)
            if args.fail_on_incomplete:
                sys.exit(4)
            raise
        logger.info("Completed %s with %d QA items", os.path.basename(pdf_path), len(qa))
        if args.expected_question_count and validation.count_mismatch():
            logger.warning(
                "Expected %d QA items but parsed %d from %s",
                args.expected_question_count,
                len(qa),
                os.path.basename(pdf_path),
            )

    if batch_mode:
        logger.info("Batch finished")


if __name__ == "__main__":
    main()