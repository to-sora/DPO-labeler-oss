"""
tests/test_engine.py

Run with: python -m pytest tests/ -v
Or:        python tests/test_engine.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wildcard_engine.core.engine import (
    generate,
    generate_with_debug,
    make_rng,
    detect_collisions,
    scenes, actions, appearance,
    _satisfies_scene, _satisfies_dressing, _satisfies_expression,
)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Same seed + names must always produce identical prompt."""
    p1 = generate(42, "Sakura", "Hana")
    p2 = generate(42, "Sakura", "Hana")
    assert p1 == p2, "Non-deterministic output for same seed!"
    print("[PASS] determinism")


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different prompts."""
    results = {generate(i, "Rin", "Yuki") for i in range(20)}
    assert len(results) > 10, "Too little variation across 20 seeds!"
    print(f"[PASS] seed variety: {len(results)} unique prompts from 20 seeds")


def test_char_names_affect_output():
    """Different character names should produce different prompts."""
    p1 = generate(0, "Alice", "Bob")
    p2 = generate(0, "Zoe", "Alex")
    assert p1 != p2
    print("[PASS] character names affect output")


# ---------------------------------------------------------------------------
# 2. Structural integrity
# ---------------------------------------------------------------------------

def test_prompt_contains_quality_prefix():
    p = generate(1, "Yuki", "Mei")
    assert "masterpiece" in p
    assert "best quality" in p
    print("[PASS] quality prefix present")


def test_prompt_contains_anime_suffix():
    p = generate(2, "Ryuu", "Ken")
    assert "anime style" in p
    print("[PASS] anime suffix present")


def test_prompt_contains_char_names():
    p = generate(3, "Sakura", "Nana")
    assert "Sakura" in p
    assert "Nana" in p
    print("[PASS] character names in prompt")


def test_prompt_is_string():
    result = generate(999, "A", "B")
    assert isinstance(result, str)
    assert len(result) > 50
    print("[PASS] output is non-empty string")


# ---------------------------------------------------------------------------
# 3. Tag algebra / constraint satisfaction
# ---------------------------------------------------------------------------

def test_all_scenes_have_required_tags():
    required_keys = {"season", "location_type", "weather", "lighting", "mood"}
    for sc in scenes():
        missing = required_keys - set(sc["tags"].keys())
        assert not missing, f"Scene {sc['id']} missing tags: {missing}"
    print(f"[PASS] all {len(scenes())} scenes have required tag keys")


def test_all_actions_have_requires():
    for ac in actions():
        assert "requires" in ac, f"Action {ac['id']} missing 'requires'"
        assert "location_type" in ac["requires"], f"Action {ac['id']} missing location_type in requires"
    print(f"[PASS] all {len(actions())} actions have 'requires.location_type'")


def test_no_impossible_scene_action_pair():
    """Every action must be satisfiable by at least one scene."""
    sc_list = scenes()
    failures = []
    for ac in actions():
        compatible = [s for s in sc_list if _satisfies_scene(ac, s)]
        if not compatible:
            failures.append(ac["id"])
    assert not failures, f"Actions with no compatible scene: {failures}"
    print("[PASS] every action satisfiable by at least one scene")


def test_every_scene_has_compatible_action():
    """Every scene must have at least one compatible action."""
    ac_list = actions()
    failures = []
    for sc in scenes():
        compatible = [a for a in ac_list if _satisfies_scene(a, sc)]
        if not compatible:
            failures.append(sc["id"])
    assert not failures, f"Scenes with no compatible action: {failures}"
    print("[PASS] every scene has at least one compatible action")


# ---------------------------------------------------------------------------
# 4. Collision detection
# ---------------------------------------------------------------------------

def test_collision_detection_catches_mismatch():
    """Manually assemble a known-bad combination and verify collision detected."""
    sc = next(s for s in scenes() if "beach" in s["tags"]["location_type"])
    ac = next(a for a in actions() if _satisfies_scene(a, sc))
    ap = appearance()

    # Force a winter dressing on a beach/summer scene
    all_dressing = []
    for sets in ap["dressing_sets"].values():
        all_dressing.extend(sets)

    winter_dress = next(d for d in all_dressing
                        if d["tags"].get("season") == ["winter"])
    good_dress = next(d for d in all_dressing
                      if _satisfies_dressing(d, sc, ac))

    exprs = ap["expressions"]
    expr = next(e for e in exprs if _satisfies_expression(e, ac, sc))

    report = detect_collisions(sc, ac, winter_dress, good_dress, expr, expr)
    assert not report.ok, "Expected collision not detected!"
    print(f"[PASS] collision detection: {report.violations[0][:60]}...")


def test_clean_combination_passes():
    """generate_with_debug must return collision_ok=True for normal seeds."""
    failures = 0
    for seed in range(50):
        result = generate_with_debug(seed, "Hana", "Rin")
        if not result.get("collision_ok", False):
            failures += 1
    assert failures == 0, f"{failures}/50 seeds had collisions"
    print("[PASS] 0/50 seeds produced collision in clean generation")


# ---------------------------------------------------------------------------
# 5. Scale test
# ---------------------------------------------------------------------------

def test_1000_seeds_unique():
    """1000 distinct seeds should produce >= 950 unique prompts."""
    results = [generate(i, "Yuki", "Sora") for i in range(1000)]
    unique = len(set(results))
    assert unique >= 950, f"Only {unique}/1000 unique prompts"
    print(f"[PASS] 1000 seed test: {unique} unique prompts")


def test_collision_rate_at_scale():
    """Check collision rate across 200 seeds stays < 5%."""
    collisions = 0
    for seed in range(200):
        result = generate_with_debug(seed, "Mei", "Luna")
        if not result.get("collision_ok"):
            collisions += 1
    rate = collisions / 200
    assert rate < 0.05, f"Collision rate too high: {rate:.1%}"
    print(f"[PASS] collision rate: {rate:.1%} across 200 seeds")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_determinism,
        test_different_seeds_differ,
        test_char_names_affect_output,
        test_prompt_contains_quality_prefix,
        test_prompt_contains_anime_suffix,
        test_prompt_contains_char_names,
        test_prompt_is_string,
        test_all_scenes_have_required_tags,
        test_all_actions_have_requires,
        test_no_impossible_scene_action_pair,
        test_every_scene_has_compatible_action,
        test_collision_detection_catches_mismatch,
        test_clean_combination_passes,
        test_1000_seeds_unique,
        test_collision_rate_at_scale,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
