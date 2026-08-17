#!/usr/bin/env python3
"""
demo.py — Quick CLI to test the wildcard engine.

Usage:
    python demo.py                         # 5 random examples
    python demo.py 42 Sakura Hana          # specific seed + names
    python demo.py 42 Sakura Hana --debug  # full breakdown
"""

import sys
import json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from wildcard_engine.core.engine import generate, generate_with_debug


def main():
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]

    if len(args) >= 3:
        seed = int(args[0])
        char1 = args[1]
        char2 = args[2]
        if debug:
            result = generate_with_debug(seed, char1, char2)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(generate(seed, char1, char2))
    else:
        # Demo: 5 seeds
        pairs = [
            (0,   "Sakura", "Hana"),
            (42,  "Yuki",   "Rin"),
            (100, "Ryuu",   "Ken"),
            (777, "Alice",  "Mei"),
            (999, "Nana",   "Sora"),
        ]
        for seed, c1, c2 in pairs:
            prompt = generate(seed, c1, c2)
            print(f"\n[seed={seed}] {c1} x {c2}")
            print(f"  {prompt[:200]}...")


if __name__ == "__main__":
    main()
