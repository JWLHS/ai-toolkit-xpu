#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the Chinese UI dictionary to the upstream ``ui/src`` tree.

Why this exists
---------------
The fork used to keep hand-edited, inline-localized copies of ~20 UI files.  That
made every upstream update a manual merge (and silently dropped new features).
Instead we now keep the upstream UI untouched and re-apply the translations from
``ui_i18n/zh-CN.json`` (English source string -> Chinese) with this script:

    python scripts/localize_ui.py            # apply
    python scripts/localize_ui.py --check    # report only, no writes

``zh-CN.json`` only matches single-quoted strings and one-line JSX text nodes.
Multi-line documentation blocks (``docs.tsx``) are covered by
``ui_i18n/zh-CN.blocks.json``: a list of {file, en, zh} snippets where ``en`` is
the exact upstream source text. Those are matched first, longest first, so a
block can also carry a small upstream repair (see the RamTorch link).

Anything neither dictionary covers is reported, so new upstream strings show up
as a short, reviewable list instead of a silent English leak.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_SRC = os.path.join(REPO, "ui", "src")
DICT = os.path.join(REPO, "ui_i18n", "zh-CN.json")
BLOCKS_DICT = os.path.join(REPO, "ui_i18n", "zh-CN.blocks.json")
DOCS_DICT = os.path.join(REPO, "ui_i18n", "docs-zh.json")
EXT = (".ts", ".tsx")


def load_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k and v and k != v}


def load_blocks(path: str) -> list:
    """Multi-line / multi-node replacements: [{file, en, zh, note?}, ...]."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = []
    for item in data.get("blocks", []):
        en, zh = item.get("en", ""), item.get("zh", "")
        if en and zh and en != zh:
            blocks.append(item)
    # longest first: a block that contains another one must win
    blocks.sort(key=lambda b: -len(b["en"]))
    return blocks


def apply_blocks(text: str, blocks: list, rel_path: str):
    applied, missed = set(), []
    for block in blocks:
        target = block.get("file")
        if target and target != rel_path:
            continue
        en, zh = block["en"], block["zh"]
        if en in text:
            text = text.replace(en, zh)
            applied.add(en)
        elif zh in text:
            applied.add(en)  # already applied on an earlier run
        else:
            missed.append((rel_path, en))
    return text, applied, missed


def load_docs(path: str) -> dict:
    """{doc key: chinese description}; blank line = paragraph break."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        k: v
        for k, v in data.items()
        if isinstance(v, str) and v.strip() and not k.startswith("_")
    }


def render_description(zh: str) -> str:
    """Chinese text -> the JSX fragment used inside docs.tsx descriptions."""
    paragraphs = [p.strip() for p in zh.strip().split("\n\n") if p.strip()]
    rendered = []
    for para in paragraphs:
        if para.startswith("<"):  # raw JSX line (e.g. the guidance diagram)
            rendered.append("        " + para)
        else:
            rendered.append("        " + " ".join(para.split()))
    body = "\n        <br />\n        <br />\n".join(rendered)
    return "description: (\n      <>\n" + body + "\n      </>\n    ),"


def apply_doc_descriptions(text: str, mapping: dict):
    """Overwrite each doc entry's description body with the Chinese version.

    Locating the body by the entry's closing marker (instead of by its English
    text) means an upstream rewrite of the copy does not break the translation.
    Do NOT try to balance parentheses here: English copy contains apostrophes
    ("a person's right side") that a naive scanner mistakes for string quotes.
    """
    applied, missed = set(), []
    END = "\n    ),\n  },"
    for key, zh in mapping.items():
        entry = -1
        for marker in (f"'{key}': {{", f"{key}: {{"):  # some keys are unquoted
            entry = text.find(marker)
            if entry >= 0:
                break
        if entry < 0:
            missed.append(key)
            continue
        start = text.find("description: (", entry)
        if start < 0:
            missed.append(key)
            continue
        close = text.find(END, start)
        if close < 0:
            missed.append(key)
            continue
        stop = close + len("\n    ),")
        text = text[:start] + render_description(zh) + text[stop:]
        applied.add(key)
    return text, applied, missed


# Code/CSS fragments are never UI copy; skip anything carrying them.
BAD_CHARS = re.compile(r"[{}()\[\]=;<>$\\`|]")
# Small words allowed inside an otherwise capitalised label.
LABEL_SMALL = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
ENGLISH_WORD = re.compile(r"[A-Za-z]{2,}")
QUOTED = re.compile(r"(['\"`])((?:\\.|(?!\1).){2,400})\1", re.S)
JSX_TEXT = re.compile(r">\s*([A-Za-z][^<>{}\n]{2,300})\s*<")


def looks_like_copy(text: str) -> bool:
    s = " ".join(text.split())
    if not s or len(s) < 2 or len(s) > 400:
        return False
    if not ENGLISH_WORD.search(s):
        return False
    if s in ("utf-8", "use client", "use server"):
        return False
    if BAD_CHARS.search(s):
        return False
    if "//" in s or s.startswith(("/", "#", ".")):
        return False
    if re.match(r"^[a-z0-9]+(?:[.\-][a-z0-9]+)+$", s):  # css class / dotted path
        return False
    words = s.split()
    if len(words) == 1:
        w = words[0]
        return 2 <= len(w) <= 24 and w.isalpha() and w[:1].isupper()
    if len(words) <= 6 and all(w[:1].isupper() or w.lower() in LABEL_SMALL for w in words):
        return True
    if len(words) >= 4 and re.search(r"[.!?:]$", s):
        return True
    if re.search(r"[.!?]$", s) and len(words) >= 3:
        return True
    if not all(w[:1].isupper() for w in words):
        return False
    return True


def scan(src: str, pairs: dict, blocks: list) -> int:
    """Report English-looking strings with no dictionary entry (review list)."""
    known = set(pairs) | set(pairs.values())
    known |= {b["en"] for b in blocks} | {b["zh"] for b in blocks}
    per_file: dict = {}

    for root, _dirs, names in os.walk(src):
        for name in names:
            if not name.endswith(EXT):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            rel = os.path.relpath(path, REPO).replace(os.sep, "/")
            found = []
            for m in QUOTED.finditer(text):
                s = m.group(2)
                if looks_like_copy(s) and s not in known:
                    found.append(s)
            for m in JSX_TEXT.finditer(text):
                s = m.group(1).strip()
                if looks_like_copy(s) and s not in known:
                    found.append(s)
            # de-dup, keep order
            seen, uniq = set(), []
            for s in found:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            if uniq:
                per_file[rel] = uniq

    total = sum(len(v) for v in per_file.values())
    print(f"untranslated-looking strings: {total} in {len(per_file)} files")
    for rel in sorted(per_file, key=lambda r: -len(per_file[r])):
        print(f"\n== {rel} ({len(per_file[rel])}) ==")
        for s in per_file[rel][:60]:
            print("   ", " ".join(s.split())[:110])
    return 0


def apply_pairs(text: str, pairs: dict) -> tuple[str, set]:
    """Replace whole quoted strings and JSX text nodes."""
    applied = set()
    for en, zh in pairs.items():
        # idempotent: if the translation is already in place, count it as applied
        if zh and zh in text and en not in text:
            applied.add(en)
            continue
        hits = 0
        for q in ('"', "'", "`"):
            needle = f"{q}{en}{q}"
            if needle in text:
                hits += text.count(needle)
                text = text.replace(needle, f"{q}{zh}{q}")
        # JSX text node: >English<
        needle = f">{en}<"
        if needle in text:
            hits += text.count(needle)
            text = text.replace(needle, f">{zh}<")
        # JSX text node that sits on its own line between tags:
        #   <button ...>
        #     Add Validation Image
        #   </button>
        line_re = re.compile(r"(?m)^([ \t]*)" + re.escape(en) + r"([ \t]*)$")
        if line_re.search(text):
            hits += len(line_re.findall(text))
            text = line_re.sub(lambda m: m.group(1) + zh + m.group(2), text)
        if hits:
            applied.add(en)
    return text, applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--src", default=UI_SRC, help="ui/src directory to localize")
    ap.add_argument(
        "--scan",
        action="store_true",
        help="list English-looking UI strings the dictionaries do not cover yet",
    )
    args = ap.parse_args()

    pairs = load_dict(DICT)
    if not pairs:
        print("dictionary is empty:", DICT)
        return 1
    blocks = load_blocks(BLOCKS_DICT)
    doc_descriptions = load_docs(DOCS_DICT)

    if args.scan:
        return scan(args.src, pairs, blocks)

    touched_files, applied_all, block_missed, doc_missing = 0, set(), [], []
    for root, _dirs, names in os.walk(args.src):
        for name in names:
            if not name.endswith(EXT):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    original = f.read()
            except Exception:
                continue
            updated, applied = apply_pairs(original, pairs)
            applied_all |= applied
            rel_path = os.path.relpath(path, REPO).replace(os.sep, "/")
            updated, block_applied, missed = apply_blocks(updated, blocks, rel_path)
            applied_all |= block_applied
            block_missed += missed
            if doc_descriptions and rel_path.endswith("/docs.tsx"):
                updated, doc_applied, doc_missed = apply_doc_descriptions(updated, doc_descriptions)
                applied_all |= doc_applied
                doc_missing += doc_missed
            if updated != original:
                touched_files += 1
                if not args.check:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(updated)

    missing = sorted(set(pairs) - applied_all)
    print(
        f"dictionary entries : {len(pairs)}"
        f"  (+{len(blocks)} block entries, +{len(doc_descriptions)} doc descriptions)"
    )
    print(f"files {'would be ' if args.check else ''}updated : {touched_files}")
    print(f"entries applied    : {len(applied_all)}")
    print(f"entries not found  : {len(missing)}")
    for en in missing[:40]:
        print("   missing:", en)
    if len(missing) > 40:
        print(f"   ... and {len(missing) - 40} more")
    if block_missed:
        print(f"blocks not found   : {len(block_missed)}")
        for rel, en in block_missed[:10]:
            print("   block missing:", rel, "|", " ".join(en.split())[:70])
    if doc_missing:
        print(f"doc keys not found : {len(doc_missing)}")
        for key in doc_missing[:10]:
            print("   doc missing:", key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
