"""
Generate iterative ablation sequences by randomly removing one object at a time.

For each labeled scene JSON in an input directory, this script:
- Creates a dedicated output folder named after the input filename (without extension)
- Writes subscene_000.json as the original scene
- Then, iteratively removes ONE random object per step and writes subscene_001.json, subscene_002.json, ...
- Stops when only ONE object remains

Compatible with two schemas:
- New: groups[].objects[] (with room_envelope, etc.)
- Old: functional_areas[].objects[]

Usage (PowerShell):
  python datagen/ablation_generate_sequences.py \
    --in-dir C:\\Users\\<you>\\Documents\\LLMLayout\\datagen\\grouped_layouts_v2_labeled \
    --out-dir C:\\Users\\<you>\\Documents\\LLMLayout\\datagen\\ablation_sequences \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def list_json_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for base, _dirs, fs in os.walk(root_dir):
        for f in fs:
            if f.lower().endswith(".json"):
                files.append(os.path.join(base, f))
    return sorted(files)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_all_objects(doc: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int]]]:
    """Return schema and a list of (container_index, object_index) pairs for all objects.
    schema is either 'groups' or 'functional_areas'.
    """
    if isinstance(doc.get("groups"), list):
        pairs: List[Tuple[int, int]] = []
        for gi, g in enumerate(doc.get("groups") or []):
            for oi, _ in enumerate(g.get("objects") or []):
                pairs.append((gi, oi))
        return "groups", pairs
    if isinstance(doc.get("functional_areas"), list):
        pairs = []
        for ai, a in enumerate(doc.get("functional_areas") or []):
            for oi, _ in enumerate(a.get("objects") or []):
                pairs.append((ai, oi))
        return "functional_areas", pairs
    raise ValueError("JSON lacks 'groups' or 'functional_areas'")


def count_objects(doc: Dict[str, Any]) -> int:
    schema, pairs = iter_all_objects(doc)
    return len(pairs)


def remove_object_in_place(doc: Dict[str, Any], container_idx: int, object_idx: int) -> None:
    """Remove an object in-place from the specified container and clean up empty containers."""
    if isinstance(doc.get("groups"), list):
        groups = doc.get("groups") or []
        if 0 <= container_idx < len(groups):
            objs = groups[container_idx].get("objects") or []
            if 0 <= object_idx < len(objs):
                del objs[object_idx]
                groups[container_idx]["objects"] = objs
        # remove empty groups
        doc["groups"] = [g for g in groups if (g.get("objects") or [])]
        return
    if isinstance(doc.get("functional_areas"), list):
        areas = doc.get("functional_areas") or []
        if 0 <= container_idx < len(areas):
            objs = areas[container_idx].get("objects") or []
            if 0 <= object_idx < len(objs):
                del objs[object_idx]
                areas[container_idx]["objects"] = objs
        # remove empty areas
        doc["functional_areas"] = [a for a in areas if (a.get("objects") or [])]
        return
    raise ValueError("JSON lacks 'groups' or 'functional_areas'")


def generate_sequence_for_file(in_path: str, out_folder: str, rng: random.Random) -> Tuple[int, int]:
    """Generate ablation sequence for a single file. Returns (total_steps_written, initial_object_count)."""
    doc = load_json(in_path)
    obj_count = count_objects(doc)
    if obj_count == 0:
        return 0, 0

    # subscene_000: original
    step_idx = 0
    dump_json(os.path.join(out_folder, f"subscene_{step_idx:03d}.json"), doc)

    # iteratively remove until one object remains
    while True:
        schema, pairs = iter_all_objects(doc)
        if len(pairs) <= 1:
            break
        # choose a random (container, object)
        c_idx, o_idx = rng.choice(pairs)
        remove_object_in_place(doc, c_idx, o_idx)
        step_idx += 1
        dump_json(os.path.join(out_folder, f"subscene_{step_idx:03d}.json"), doc)

    return step_idx + 1, obj_count  # number of files written, initial count


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively remove one random object and write scene snapshots")
    parser.add_argument("--in-dir", required=True, help="Directory of labeled scenes (JSON)")
    parser.add_argument("--out-dir", required=True, help="Directory to write per-scene subfolders with subscene_xxx.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    rng = random.Random(args.seed)

    if not os.path.isdir(in_dir):
        print(f"找不到输入目录: {in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    files = list_json_files(in_dir)
    if not files:
        print("⚠️ 输入目录中未找到任何 JSON 文件", file=sys.stderr)
        sys.exit(2)

    total = len(files)
    ok = 0
    fail = 0
    for i, p in enumerate(files, 1):
        try:
            base = os.path.splitext(os.path.basename(p))[0]
            scene_folder = os.path.join(out_dir, base)
            os.makedirs(scene_folder, exist_ok=True)
            written, n0 = generate_sequence_for_file(p, scene_folder, rng)
            print(f"[{i}/{total}] ✅ {base} | objects_initial={n0} | steps={written}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{total}] ❌ {p} -> {e}", file=sys.stderr)
            fail += 1

    print(f"完成: 成功 {ok} 个, 失败 {fail} 个, 输出目录: {out_dir}")


if __name__ == "__main__":
    main()


