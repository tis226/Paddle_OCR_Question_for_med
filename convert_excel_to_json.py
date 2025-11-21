from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

EXCEL_DIR = Path("2025_exam_excel")
JSON_DIR = Path("json")
CHECKABLE_DIR = JSON_DIR / "2025_exam_checkable"
SUBJECT_ORDER = ["보건의약관계법규", "의학총론", "의학각론1", "의학각론2", "의학각론3"]
OPTION_LABELS = ["①", "②", "③", "④", "⑤"]
_DIGIT_TO_OPTION = {str(idx): label for idx, label in enumerate(OPTION_LABELS, start=1)}


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _read_ai_answer(raw_row: pd.Series) -> Any:
    """Return the raw AI answer value from either spelling of the column."""

    if "AI_answer" in raw_row:
        return raw_row.get("AI_answer")
    return raw_row.get("AI_aswer")


def _build_options(row: pd.Series) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for idx, label in enumerate(OPTION_LABELS, start=1):
        text = _normalize_text(row.get(f"option_{idx}"))
        if text:
            options.append({"index": label, "text": text})
    return options


def _build_ai_answer(row: pd.Series, options: List[Dict[str, str]]) -> Dict[str, str]:
    """Return the AI answer payload expected by evaluators.

    The Excel files store the answer in the ``AI_aswer`` column as a digit from
    1 to 5. We convert that digit into the circled option label used in the JSON
    files and attach the option text for convenience.
    """

    raw_answer = _read_ai_answer(row)
    digit = _normalize_text(raw_answer)
    chosen_index = _DIGIT_TO_OPTION.get(digit)

    chosen_text = ""
    if chosen_index:
        for option in options:
            if option["index"] == chosen_index:
                chosen_text = option["text"]
                break

    return {"chosen_index": chosen_index or "", "chosen_text": chosen_text}


def _build_answer_entry(row: pd.Series) -> Dict[str, str | int]:
    digit_answer = _normalize_text(_read_ai_answer(row))
    chosen_index = _DIGIT_TO_OPTION.get(digit_answer, "")

    chosen_text = ""
    try:
        option_idx = int(digit_answer)
    except ValueError:
        option_idx = None

    if option_idx:
        chosen_text = _normalize_text(row.get(f"option_{option_idx}"))

    return {
        "question_number": int(row.get("question_number")),
        "chosen_index": chosen_index,
        "chosen_text": chosen_text,
        "reasoning": "",
    }


def convert_excel_to_entries(excel_path: Path) -> List[Dict[str, Any]]:
    dataframe = pd.read_excel(excel_path)
    entries: List[Dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        options = _build_options(row)
        entry = {
            "subject": _normalize_text(row.get("subject")),
            "year": int(row.get("year")),
            "target": _normalize_text(row.get("target")),
            "content": {
                "question_number": int(row.get("question_number")),
                "question_text": _normalize_text(row.get("question_text")),
                "options": options,
            },
            "ai_answer": _build_ai_answer(row, options),
        }
        entries.append(entry)

    return entries


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)


def _collect_subject_answers() -> Dict[str, Dict[str, Any]]:
    subject_answers: Dict[str, Dict[str, Any]] = {}

    for excel_path in sorted(EXCEL_DIR.glob("*.xlsx")):
        dataframe = pd.read_excel(excel_path)
        answers = [_build_answer_entry(row) for _, row in dataframe.iterrows()]
        if dataframe.empty:
            continue

        subject = _normalize_text(dataframe.iloc[0].get("subject"))
        year_value = dataframe.iloc[0].get("year")
        year = int(year_value) if not pd.isna(year_value) else None

        subject_answers[subject] = {"year": year, "answers": answers}

    return subject_answers


def _build_checkable_payload(subject: str, subject_answers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for candidate in SUBJECT_ORDER:
        if candidate == subject:
            subject_entry = subject_answers.get(candidate, {"year": None, "answers": []})
            results.append({
                "subject": candidate,
                "year": subject_entry.get("year"),
                "answers": subject_entry.get("answers", []),
            })
        else:
            results.append({"subject": candidate, "year": None, "answers": []})

    return {"results": results}


def main() -> None:
    all_entries: List[Dict[str, Any]] = []

    for excel_path in sorted(EXCEL_DIR.glob("*.xlsx")):
        entries = convert_excel_to_entries(excel_path)
        json_path = JSON_DIR / f"{excel_path.stem}.json"
        write_json(json_path, entries)
        all_entries.extend(entries)

    write_json(JSON_DIR / "2025_exam_all.json", all_entries)

    subject_answers = _collect_subject_answers()
    for subject in SUBJECT_ORDER:
        payload = _build_checkable_payload(subject, subject_answers)
        write_json(CHECKABLE_DIR / f"{subject}.json", payload)


if __name__ == "__main__":
    main()
