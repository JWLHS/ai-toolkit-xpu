#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flag dictionary entries that are code, not UI copy.

``localize_ui.py`` replaces the English key wherever it appears as a quoted
string, so an entry like ``"text"`` or ``"_blank"`` also rewrites
``type="text"`` / ``target="_blank"`` and breaks the UI. Run this after adding
entries:

    python scripts/audit_i18n.py           # list suspicious entries
    python scripts/audit_i18n.py --strict  # exit 1 when anything looks wrong
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT = os.path.join(REPO, "ui_i18n", "zh-CN.json")

# Keys that look like HTML attributes, CSS classes, paths or identifiers.
ATTRS = {
    "text", "password", "email", "number", "checkbox", "radio", "file", "submit",
    "button", "hidden", "search", "tel", "url", "date", "range", "color",
    "_blank", "_self", "_parent", "_top", "noreferrer", "noopener", "button",
    "get", "post", "put", "delete", "json", "utf-8", "true", "false", "null",
}
CODEY = re.compile(
    r"^[a-z0-9_]+$"          # single lowercase identifier / attribute
    r"|^-"                   # leading dash (CSS-ish)
    r"|^[a-z]+-[a-z]+"       # kebab-case
    r"|[\\/]"                # path-ish
    r"|\s{2,}"               # accidental run of spaces
    r"|^[.#]"                # selector
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default=DICT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    with open(args.dict, "r", encoding="utf-8") as f:
        data = json.load(f)

    suspects = []
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.lower() in ATTRS or CODEY.search(key):
            suspects.append((key, value))
        elif re.fullmatch(r"[A-Za-z]+", key) and key[0].isupper() and len(key) <= 3:
            # very short single words are usually identifiers, not labels
            suspects.append((key, value))

    print("entries        :", len(data))
    print("suspect entries:", len(suspects))
    for key, value in suspects:
        print(f"   {key!r} -> {value!r}")
    if args.strict and suspects:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
