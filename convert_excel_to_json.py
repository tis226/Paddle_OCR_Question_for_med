from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

EXCEL_DIR = Path("2025_exam_excel")
JSON_DIR = Path("json")
OPTION_LABELS = ["①", "②", "③", "④", "⑤"]


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_options(row: pd.Series) -> Tuple[List[Dict[str, str]], Dict[int, str]]:
    options: List[Dict[str, str]] = []
    option_lookup: Dict[int, str] = {}
    for idx, label in enumerate(OPTION_LABELS, start=1):
        text = _normalize_text(row.get(f"option_{idx}"))
        option_lookup[idx] = text
        if text:
            options.append({"index": label, "text": text})
    return options, option_lookup


def _extract_answer(row: pd.Series, option_lookup: Dict[int, str]) -> Optional[Dict[str, Any]]:
    answer_column = "AI_aswer" if "AI_aswer" in row else "AI_answer"
    raw_answer = _normalize_text(row.get(answer_column))
    if not raw_answer:
        return None

    try:
        numeric_index = int(raw_answer)
    except ValueError:
        return {"value": raw_answer}

    label: str
    if 1 <= numeric_index <= len(OPTION_LABELS):
        label = OPTION_LABELS[numeric_index - 1]
    else:
        label = str(numeric_index)

    return {
        "option_number": numeric_index,
        "option_label": label,
        "option_text": option_lookup.get(numeric_index, ""),
    }


def convert_excel_to_entries(excel_path: Path) -> List[Dict[str, Any]]:
    dataframe = pd.read_excel(excel_path)
    entries: List[Dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        options, option_lookup = _build_options(row)
        answer = _extract_answer(row, option_lookup)

        entry = {
            "subject": _normalize_text(row.get("subject")),
            "year": int(row.get("year")),
            "target": _normalize_text(row.get("target")),
            "content": {
                "question_number": int(row.get("question_number")),
                "question_text": _normalize_text(row.get("question_text")),
                "options": options,
                "answer": answer,
            },
        }
        entries.append(entry)

    return entries


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)


def main() -> None:
    all_entries: List[Dict[str, Any]] = []

    for excel_path in sorted(EXCEL_DIR.glob("*.xlsx")):
        entries = convert_excel_to_entries(excel_path)
        json_path = JSON_DIR / f"{excel_path.stem}.json"
        write_json(json_path, entries)
        all_entries.extend(entries)

    write_json(JSON_DIR / "2025_exam_all.json", all_entries)


if __name__ == "__main__":
    main()
