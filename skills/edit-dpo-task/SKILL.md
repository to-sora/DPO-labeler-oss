---
name: edit-dpo-task
description: Edit, create, or review Comfy DPO task YAML files, including checkpoint aliases, prompt-family routing, seed controls, image configurations, session counts, LoRA settings, and publication safety. Use when changing files under template/tasks or converting an experiment workflow into a reproducible community task.
---

# Edit DPO Task

Work from the repository root. Keep task edits narrow and validate the compiled
requests before running ComfyUI.

## Workflow

1. Read the target task and `checkpoint_aliases.yaml`.
2. Read [references/task-schema.md](references/task-schema.md) when changing
   task structure, seed behavior, prompt routing, or checkpoint registration.
3. Use exact checkpoint aliases from the registry. Add a registry model entry
   for a new alias; omit `family` only when an ordered keyword rule classifies
   it correctly.
4. Keep private aliases out of publishable tasks. Registry entries with
   `visibility: private` or `publish: false` are local-only.
5. Preserve relative prompt asset roots. Paths in `template/tasks/*.yaml` are
   relative to `template/tasks/`, normally `../prompt_templates`,
   `../prompt_lists`, and `../wildcard`.
6. Run the bundled validator before tests:

```bash
"${VIRTUAL_ENV:-venv}/bin/python" \
  skills/edit-dpo-task/scripts/validate_task.py \
  template/tasks/quality_e2e_10x2.yaml \
  --expected-per-image 10 \
  --publication
```

7. Run the task and workflow unit tests after validation:

```bash
"${VIRTUAL_ENV:-venv}/bin/python" -m unittest \
  tests.unit.test_checkpoint_registry \
  tests.unit.test_quality_e2e_task \
  tests.unit.test_workflows
```

8. Run real generation only when requested and ComfyUI/model prerequisites are
   available. Validate result counts, statuses, files, hashes, and dimensions.

## Editing Rules

- Give every image a unique `image_name`.
- Set `session_count` to the required count per image configuration. Two images
  and `session_count: 10` compile to 20 requests.
- Use `session_seed` for controlled paired values and `image_index_seed` when
  each image should vary independently.
- Provide all canonical prompt-family mappings when a task may select models
  from multiple families: `illustration`, `sdxl_anime_base`, `pony`, and
  `realistic`.
- Set `toggle: false` for a no-LoRA task. The runtime clears all inherited LoRA
  names, but the task should still declare `lora_1_name: None` explicitly.
- Do not hand-edit generated workflow matrices unless the user explicitly asks
  for generated outputs. Edit their generator instead.
- Do not claim a task is publication-ready when validation uses the default
  family fallback or a non-publishable checkpoint.

## Validation Output

The bundled validator reports compiled requests, per-image counts, checkpoint
registry identities, resolved families, family sources, visibility, and
publishability. It exits nonzero for schema errors, unresolved prompt tokens,
unknown aliases, count mismatches, or publication-policy failures.
