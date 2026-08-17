# wildcard_engine

Deterministic SDXL anime prompt generator with tag-algebra constraint satisfaction.

## Signature

```python
def generate(seed: int, character_name1: str, character_name2: str) -> str
```

Same `(seed, char1, char2)` always returns the same prompt.<br>
Different seeds explore a combinatorial space of **5.4 × 10¹⁶** valid combinations.

---

## Architecture

```
seed + char1 + char2
        │
        ▼
  SHA-256 → seeded RNG
        │
   Step 1: pick scene  (50 scenes × 5 tag dims)
        │
   Step 2: filter actions by scene tags → pick action  (up to 50 compatible)
        │
   Step 3: build char descriptors (hair + eye + dressing, filtered by scene+action)
        │
   Step 4: pick expressions filtered by action.compatible_expr_mood ∪ scene.mood
        │
   Step 5: collision_detect() — retry if violated (max 10 attempts)
        │
   Step 6: assemble prompt string
```

---

## Tag Algebra

Every element carries a `tags` dict. Compatibility is intersection-based:

```
scene      emits  → {season, location_type, weather, lighting, mood}
action   requires → {location_type} ∩ scene.tags ≠ ∅
dressing requires → {season, location_type} ⊆ scene.tags (loose)
expression requires → mood_tags ∩ (action.compatible_expr_mood ∪ scene.mood) ≠ ∅
```

`"any"` in any tag list is a wildcard that satisfies any constraint.

---

## Collision Detection

`detect_collisions(scene, action, dress1, dress2, expr1, expr2)` runs 5 checks:

1. Action location_type ∩ scene location_type
2. Action weather requirement ∩ scene weather
3. Dressing season ∩ scene season (per character)
4. Dressing energy compatibility with action energy
5. Expression mood_tags ∩ action+scene mood

Returns `CollisionReport` with `.ok: bool` and `.violations: list[str]`.

---

## Pool Sizes

| Pool | Count |
|------|-------|
| Scenes | 50 |
| Actions | 50 |
| Dressing sets | 36 |
| Expressions | 20 |
| Hair styles | 15 |
| Eye styles | 10 |

Valid combination lower bound: `50 × 50 × 36² × 20 × 15² × 10² ≈ 2.4 × 10¹²`<br>
Well above 1M image requirement.

---

## Usage

```python
from wildcard_engine.core.engine import generate, generate_with_debug

# Simple
prompt = generate(42, "Sakura", "Hana")

# With full breakdown
info = generate_with_debug(42, "Sakura", "Hana")
print(info["scene_desc"])
print(info["collision_ok"])
print(info["prompt"])
```

### CLI

```bash
python demo.py                          # 5 demo examples
python demo.py 42 Sakura Hana           # specific seed
python demo.py 42 Sakura Hana --debug   # full JSON breakdown
```

### Tests

```bash
python wildcard_engine/tests/test_engine.py
```

---

## Data Files

| File | Contents |
|------|----------|
| `data/scenes.json` | 50 scene bundles with tag sets |
| `data/actions.json` | 50 two-person action pairs with slot descriptors |
| `data/appearance.json` | expressions, dressing sets, hair/eye modifiers |

---

## Extending

To add more scenes/actions/dressing: edit the JSON files.<br>
The engine re-reads files on first import (cached after that).<br>
No code changes needed — the tag algebra adapts automatically.
