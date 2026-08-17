"""
wildcard_engine/core/engine.py

Deterministic prompt generator using tag-algebra constraint satisfaction.
seed + char1_name + char2_name -> SDXL anime prompt string
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loader (cached on first import)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _load(name: str) -> Any:
    if name not in _cache:
        with open(DATA_DIR / f"{name}.json", "r", encoding="utf-8") as f:
            _cache[name] = json.load(f)
    return _cache[name]


def scenes():    return _load("scenes")
def actions():   return _load("actions")
def appearance(): return _load("appearance")


# ---------------------------------------------------------------------------
# Seeded RNG
# ---------------------------------------------------------------------------

def make_rng(seed: int, char1: str, char2: str) -> random.Random:
    """
    Derive a deterministic RNG from seed + character names.
    Same (seed, char1, char2) always produces the same random state.
    """
    key = f"{seed}|{char1.strip().lower()}|{char2.strip().lower()}"
    digest = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return random.Random(digest & 0xFFFF_FFFF_FFFF_FFFF)


# ---------------------------------------------------------------------------
# Tag algebra helpers
# ---------------------------------------------------------------------------

def _intersects(a: list, b: list) -> bool:
    """True if any element of a is in b (or either list contains 'any')."""
    if "any" in a or "any" in b:
        return True
    return bool(set(a) & set(b))


def _satisfies_scene(action: dict, scene: dict) -> bool:
    """
    Action is compatible with scene if:
      - required location_type intersects scene location_type
      - required weather (if present) intersects scene weather
      - required season (if present) intersects scene season
    """
    req = action.get("requires", {})

    # location_type is mandatory on every action
    if not _intersects(req.get("location_type", []), scene["tags"]["location_type"]):
        return False

    # weather check (optional requirement)
    if "weather" in req:
        if not _intersects(req["weather"], scene["tags"]["weather"]):
            return False

    # mood check (optional requirement)
    if "mood" in req:
        if not _intersects(req["mood"], scene["tags"]["mood"]):
            return False

    # season check (optional requirement on action — rare)
    if "season" in req:
        if not _intersects(req["season"], scene["tags"]["season"]):
            return False

    return True


def _satisfies_dressing(dset: dict, scene: dict, action: dict) -> bool:
    """
    Dressing set is compatible if:
      - season intersects scene season  (strict — 'any' wildcard honoured)
      - location_type intersects scene location_type
      - energy within ±1 step of action energy
    """
    t = dset["tags"]

    # Season — strict check; 'any' on either side passes
    dressing_seasons = t.get("season", ["any"])
    scene_seasons    = scene["tags"]["season"]
    if not _intersects(dressing_seasons, scene_seasons):
        return False

    # Location type
    if not _intersects(t.get("location_type", []), scene["tags"]["location_type"]):
        return False

    # Energy — allow ±1 step
    energy_order = ["very_low", "low", "medium", "high", "very_high"]
    action_energy    = action.get("energy", "medium")
    dressing_energies = t.get("energy", list(energy_order))
    ae_idx = energy_order.index(action_energy) if action_energy in energy_order else 2
    compatible_range = set(energy_order[max(0, ae_idx - 1): ae_idx + 2])
    if not (compatible_range & set(dressing_energies)):
        return False

    return True


def _satisfies_expression(expr: dict, action: dict, scene: dict) -> bool:
    """
    Expression is compatible if its mood_tags overlap with
    action's compatible_expr_mood OR scene's mood tags.
    """
    compatible = set(action.get("compatible_expr_mood", []))
    compatible |= set(scene["tags"].get("mood", []))
    return bool(set(expr["mood_tags"]) & compatible)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

class CollisionReport:
    def __init__(self):
        self.violations: list[str] = []

    def add(self, msg: str):
        self.violations.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def __str__(self):
        if self.ok:
            return "OK: no collisions"
        return "COLLISIONS:\n" + "\n".join(f"  - {v}" for v in self.violations)


def detect_collisions(scene: dict, action: dict,
                       dress1: dict, dress2: dict,
                       expr1: dict, expr2: dict) -> CollisionReport:
    """
    Full semantic coherence check on a generated combination.
    Returns a CollisionReport; report.ok == True means clean combination.
    """
    r = CollisionReport()

    # Scene <-> Action
    if not _satisfies_scene(action, scene):
        r.add(f"Action '{action['id']}' incompatible with scene '{scene['id']}'")

    # Scene <-> Dressing
    for i, d in enumerate([dress1, dress2], 1):
        if not _satisfies_dressing(d, scene, action):
            r.add(f"Dressing '{d['id']}' (char{i}) incompatible with scene/action")

    # Action <-> Expression
    for i, e in enumerate([expr1, expr2], 1):
        if not _satisfies_expression(e, action, scene):
            r.add(f"Expression '{e['id']}' (char{i}) incompatible with action/scene mood")

    # Season sanity: both dressings should agree on season category
    s1_tags = set(dress1["tags"].get("season", ["any"]))
    s2_tags = set(dress2["tags"].get("season", ["any"]))
    scene_seasons = set(scene["tags"]["season"])
    if "any" not in s1_tags and not (s1_tags & scene_seasons):
        r.add(f"Char1 dressing season {s1_tags} mismatch with scene seasons {scene_seasons}")
    if "any" not in s2_tags and not (s2_tags & scene_seasons):
        r.add(f"Char2 dressing season {s2_tags} mismatch with scene seasons {scene_seasons}")

    return r


# ---------------------------------------------------------------------------
# Character name → appearance hint
# ---------------------------------------------------------------------------

_FEMININE_HINTS = {"sakura","hana","yuki","momo","nana","rin","mei","aoi","sora","luna",
                   "alice","emily","emma","sofia","lily","rose","anna","mia","chloe","ivy"}
_MASCULINE_HINTS = {"ryuu","ken","tatsu","hiroshi","daisuke","shota","tarou","yuto","kei",
                    "james","oliver","liam","ethan","noah","alex","kai","leon","ryo","jun"}


def _gender_hint(name: str) -> str:
    n = name.strip().lower()
    if n in _FEMININE_HINTS:
        return "feminine"
    if n in _MASCULINE_HINTS:
        return "masculine"
    return "neutral"


def _build_char(rng: random.Random, name: str, scene: dict,
                action: dict, slot_key: str) -> dict:
    """Build a character descriptor dict from name + context."""
    ap = appearance()
    hair = rng.choice(ap["appearance_modifiers"]["hair_styles"])
    eye  = rng.choice(ap["appearance_modifiers"]["eye_styles"])
    hint = _gender_hint(name)

    # Flatten all dressing sets into single list, filter by context
    all_dressing: list[dict] = []
    for sets in ap["dressing_sets"].values():
        all_dressing.extend(sets)

    # Filter: compatible with scene + action, prefer gender hint
    compatible = [d for d in all_dressing if _satisfies_dressing(d, scene, action)]
    if not compatible:
        compatible = all_dressing  # fallback: no constraint

    # Prefer gender-matching, but allow neutral
    preferred = [d for d in compatible if d["gender_hint"] in (hint, "neutral")]
    pool = preferred if preferred else compatible

    dressing = rng.choice(pool)
    return {"name": name, "hair": hair, "eye": eye, "dressing": dressing}


# ---------------------------------------------------------------------------
# Prompt assembler
# ---------------------------------------------------------------------------

_QUALITY_PREFIX = (
    "masterpiece, best quality, highly detailed, "
    "ultra-detailed, sharp focus, 8k, "
    "2girls"  # two characters always present
)

_STYLE_SUFFIX = (
    "anime style, anime coloring, cel shading, "
    "clean lineart, vibrant colors, "
    "professional anime illustration"
)


def _assemble_prompt(scene: dict, action: dict,
                     char1: dict, char2: dict,
                     expr1: dict, expr2: dict,
                     *,
                     quality_prefix: str = _QUALITY_PREFIX,
                     style_suffix: str = _STYLE_SUFFIX) -> str:
    parts = [quality_prefix] if quality_prefix else []

    # Scene
    parts.append(scene["desc"])

    # Action
    parts.append(action["desc"])

    # Char1
    c1 = (
        f"{char1['name']}: {char1['hair']['token']}, "
        f"{char1['eye']['token']}, "
        f"{char1['dressing']['prompt_token']}, "
        f"{expr1['prompt_token']}"
    )
    parts.append(c1)

    # Char2
    c2 = (
        f"{char2['name']}: {char2['hair']['token']}, "
        f"{char2['eye']['token']}, "
        f"{char2['dressing']['prompt_token']}, "
        f"{expr2['prompt_token']}"
    )
    parts.append(c2)

    # Lighting hint from scene
    lighting = scene["tags"].get("lighting", [])
    if lighting:
        parts.append(f"lighting: {', '.join(lighting[:2])}")

    if style_suffix:
        parts.append(style_suffix)

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(seed: int, character_name1: str, character_name2: str,
             max_retries: int = 10,
             *,
             quality_prefix: str | None = None,
             style_suffix: str | None = None) -> str:
    """
    Deterministic SDXL anime prompt generator.

    Parameters
    ----------
    seed            : int — controls all random choices
    character_name1 : str — name of first character
    character_name2 : str — name of second character
    max_retries     : int — max attempts to find collision-free combination

    Returns
    -------
    str — fully assembled prompt string
    """
    rng = make_rng(seed, character_name1, character_name2)

    all_scenes  = scenes()
    all_actions = actions()
    ap = appearance()

    all_exprs: list[dict] = ap["expressions"]

    assemble_kwargs: dict = {}
    if quality_prefix is not None:
        assemble_kwargs["quality_prefix"] = quality_prefix
    if style_suffix is not None:
        assemble_kwargs["style_suffix"] = style_suffix

    # Flatten all dressing sets
    all_dressing: list[dict] = []
    for sets in ap["dressing_sets"].values():
        all_dressing.extend(sets)

    for attempt in range(max_retries):
        # Step 1: Pick scene
        scene = rng.choice(all_scenes)

        # Step 2: Pick action compatible with scene
        compatible_actions = [a for a in all_actions if _satisfies_scene(a, scene)]
        if not compatible_actions:
            continue
        action = rng.choice(compatible_actions)

        # Step 3: Build characters (includes dressing selection)
        char1 = _build_char(rng, character_name1, scene, action, "slot_A")
        char2 = _build_char(rng, character_name2, scene, action, "slot_B")

        # Step 4: Pick expressions compatible with action+scene mood
        compatible_exprs = [e for e in all_exprs if _satisfies_expression(e, action, scene)]
        if not compatible_exprs:
            compatible_exprs = all_exprs  # fallback

        expr1 = rng.choice(compatible_exprs)
        expr2 = rng.choice(compatible_exprs)

        # Step 5: Collision check
        report = detect_collisions(scene, action,
                                   char1["dressing"], char2["dressing"],
                                   expr1, expr2)
        if report.ok:
            return _assemble_prompt(scene, action, char1, char2, expr1, expr2, **assemble_kwargs)

        # On collision, rng already advanced — next iteration picks differently

    # Last resort: return whatever we have (best effort)
    return _assemble_prompt(scene, action, char1, char2, expr1, expr2, **assemble_kwargs)


def generate_with_debug(seed: int, character_name1: str, character_name2: str,
                        *,
                        quality_prefix: str | None = None,
                        style_suffix: str | None = None) -> dict:
    """
    Same as generate() but returns full breakdown for inspection/debugging.
    """
    rng = make_rng(seed, character_name1, character_name2)

    all_scenes  = scenes()
    all_actions = actions()
    ap = appearance()
    all_exprs: list[dict] = ap["expressions"]
    all_dressing: list[dict] = []
    for sets in ap["dressing_sets"].values():
        all_dressing.extend(sets)

    assemble_kwargs: dict = {}
    if quality_prefix is not None:
        assemble_kwargs["quality_prefix"] = quality_prefix
    if style_suffix is not None:
        assemble_kwargs["style_suffix"] = style_suffix

    scene = action = char1 = char2 = expr1 = expr2 = report = None
    for _ in range(20):
        scene  = rng.choice(all_scenes)
        compatible_actions = [a for a in all_actions if _satisfies_scene(a, scene)]
        if not compatible_actions:
            continue
        action = rng.choice(compatible_actions)

        char1  = _build_char(rng, character_name1, scene, action, "slot_A")
        char2  = _build_char(rng, character_name2, scene, action, "slot_B")

        compatible_exprs = [e for e in all_exprs if _satisfies_expression(e, action, scene)]
        if not compatible_exprs:
            compatible_exprs = all_exprs

        expr1 = rng.choice(compatible_exprs)
        expr2 = rng.choice(compatible_exprs)

        report = detect_collisions(scene, action,
                                   char1["dressing"], char2["dressing"],
                                   expr1, expr2)
        if report.ok:
            break

    if scene is None:
        return {"error": "Could not find valid combination after 20 attempts"}

    prompt = _assemble_prompt(scene, action, char1, char2, expr1, expr2, **assemble_kwargs)

    return {
        "seed": seed,
        "char1": character_name1,
        "char2": character_name2,
        "scene_id": scene["id"],
        "scene_desc": scene["desc"],
        "action_id": action["id"],
        "action_desc": action["desc"],
        "char1_hair": char1["hair"]["token"],
        "char1_eye": char1["eye"]["token"],
        "char1_dressing": char1["dressing"]["desc"],
        "char1_expr": expr1["label"],
        "char2_hair": char2["hair"]["token"],
        "char2_eye": char2["eye"]["token"],
        "char2_dressing": char2["dressing"]["desc"],
        "char2_expr": expr2["label"],
        "collision_report": str(report),
        "collision_ok": report.ok,
        "prompt": prompt,
    }
