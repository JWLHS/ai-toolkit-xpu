#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HF 缓存里的 blobs 手工重建空的 snapshot（网络不通时的自救手段）。

现象：`snapshots/<rev>/` 下所有文件都是 0 字节，但 `blobs/` 里的真实数据还在
（下载被中断/被 kill 时会出现）。本脚本按内容识别每个 blob 的身份，再用硬链接
（失败则复制）把它们还原成 snapshot 里的正确文件名。

用法: python scripts/repair_hf_snapshot.py <cache_root> <repo_id>
例如: python scripts/repair_hf_snapshot.py D:/Cache/huggingface/hub Qwen/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import json
import os
import sys


def repo_dir(cache_root: str, repo_id: str) -> str:
    return os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))


def snapshot_dir(repo: str) -> str:
    ref = os.path.join(repo, "refs", "main")
    rev = open(ref).read().strip() if os.path.isfile(ref) else None
    snaps = os.path.join(repo, "snapshots")
    if rev and os.path.isdir(os.path.join(snaps, rev)):
        return os.path.join(snaps, rev)
    dirs = [d for d in os.listdir(snaps)] if os.path.isdir(snaps) else []
    if not dirs:
        return os.path.join(snaps, rev or "main")
    return os.path.join(snaps, dirs[0])


def peek(path: str, n: int = 4096) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def classify(path: str) -> str:
    size = os.path.getsize(path)
    head = peek(path, 1 << 20)
    text = None
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        pass

    if text is not None:
        if '"weight_map"' in text:
            return "model.safetensors.index.json"
        if '"model_type"' in text or '"architectures"' in text:
            return "config.json"
        if '"bos_token_id"' in text or '"eos_token_id"' in text or '"generation"' in text.lower():
            if size < 64 * 1024:
                return "generation_config.json"
        if text.lstrip().startswith("{"):
            if '"tokenizer_class"' in text or '"model_max_length"' in text:
                return "tokenizer_config.json"
            if '"version"' in text and '"model"' in text and size > 1 << 20:
                return "tokenizer.json"
            if text.lstrip().startswith("{") and size > 1 << 20:
                return "vocab.json"
    return "binary"


def safetensors_tensor_names(path: str) -> set:
    with open(path, "rb") as f:
        raw = f.read(8)
        n = int.from_bytes(raw, "little")
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


def link_or_copy(src: str, dst: str) -> str:
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        import shutil

        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cache_root, repo_id = sys.argv[1], sys.argv[2]
    repo = repo_dir(cache_root, repo_id)
    blobs_dir = os.path.join(repo, "blobs")
    snap = snapshot_dir(repo)
    if not os.path.isdir(blobs_dir):
        print("no blobs dir:", blobs_dir)
        return 2
    os.makedirs(snap, exist_ok=True)

    blobs = [os.path.join(blobs_dir, b) for b in os.listdir(blobs_dir)]
    blobs = [b for b in blobs if os.path.isfile(b) and os.path.getsize(b) > 0]
    kinds: dict = {}
    for b in blobs:
        kinds[b] = classify(b)

    # 用 index 的 weight_map 把两个 safetensors 分片对上号
    index_blob = next((b for b, k in kinds.items() if k == "model.safetensors.index.json"), None)
    shard_blobs = [b for b in blobs if kinds[b] == "binary" and os.path.getsize(b) > 100 << 20]
    mapping: dict = {}
    if index_blob and shard_blobs:
        index = json.load(open(index_blob, encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        per_file: dict = {}
        for tensor, fname in weight_map.items():
            per_file.setdefault(fname, set()).add(tensor)
        shard_sets = {b: safetensors_tensor_names(b) for b in shard_blobs}
        for fname, tensors in per_file.items():
            for b, names in list(shard_sets.items()):
                if names == tensors:
                    mapping[fname] = b
                    del shard_sets[b]
                    break
        mapping["model.safetensors.index.json"] = index_blob

    for b, kind in kinds.items():
        if kind != "binary" and kind not in mapping:
            mapping.setdefault(kind, b)
    for b in shard_blobs:
        if b not in mapping.values():
            base = os.path.basename(b)[:12]
            mapping[f"unmatched-{base}.bin"] = b

    print(f"snapshot: {snap}")
    for name, src in sorted(mapping.items()):
        dst = os.path.join(snap, name)
        how = link_or_copy(src, dst)
        print(f"  {name:42} <- {os.path.basename(src)[:16]}  ({how})")

    # 验证
    cfg = os.path.join(snap, "config.json")
    if os.path.isfile(cfg):
        try:
            data = json.load(open(cfg, encoding="utf-8"))
            print("config.json 可解析 ✓  model_type =", data.get("model_type"))
        except Exception as exc:  # noqa: BLE001
            print("config.json 仍然不可解析 ✗", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
