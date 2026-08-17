# Checkpoint Alias Registry

## Scope

`checkpoint_aliases.yaml` is the source of truth for checkpoint aliases,
prompt families, visibility, and OSS publishability. It covers all 56 SDXL
checkpoint aliases referenced by the bundled generators and task assets: 49
public/local aliases and seven private training aliases.

The ComfyUI checkpoint loader can also expose SD 1.5, video, audio, Qwen, VAE,
and ControlNet files from a shared model tree. Those are not part of the SDXL
DPO prompt-family inventory and must not be used with the bundled workflows.

## Resolution

Resolution is case-insensitive and follows this order:

1. Exact full alias.
2. Unambiguous basename alias.
3. Explicit `family` on the model entry.
4. First ordered substring match from `keyword_rules`.
5. `default_family`, retained only for backward compatibility.

The canonical families are `illustration`, `sdxl_anime_base`, `pony`, and
`realistic`. The compiler records `ckpt_registry_id`, `ckpt_family_source`,
`ckpt_family_keyword`, `ckpt_visibility`, and `ckpt_publish` in every request.

The current keyword-only audit correctly classifies all 56 known aliases. The
unit gate requires at least 90% and separately requires every known alias to be
registered and resolve to its expected family without the default fallback.

## Model Entries

Aliases are grouped under a stable model ID:

```yaml
models:
  steincustom-v13:
    aliases:
      - sdxl_steincustom_V13.safetensors
      - sdxl_steincustom_V13__2.safetensors
```

An entry can override classification and publication metadata:

```yaml
models:
  private-model:
    family: illustration
    visibility: private
    publish: false
    aliases:
      - private/path/model.ckpt
```

If `family` is absent, the classifier searches the checkpoint path, model ID,
and aliases using the ordered keyword rules. Add an explicit family when a name
contains conflicting or insufficient keywords.

## Local Overlay

`checkpoint_aliases.local.yaml` is ignored and automatically merged over the
tracked registry. An additional overlay can be selected with
`CHECKPOINT_ALIAS_REGISTRY`. Overlay files declare `version: 1` and can add or
replace entries under `models`.

Keep private aliases in a local overlay when they are not already present in
historical generated tasks. Mark every private entry `visibility: private` and
`publish: false`.

## Validation

Run the registry and task gates from an active project environment:

```bash
"${VIRTUAL_ENV:-.venv}/bin/python" -m unittest \
  tests.unit.test_checkpoint_registry \
  tests.unit.test_quality_e2e_task \
  tests.unit.test_task_editing_skill
```

Validate a publication task through the packaged skill:

```bash
"${VIRTUAL_ENV:-.venv}/bin/python" \
  skills/edit-dpo-task/scripts/validate_task.py \
  template/tasks/quality_e2e_10x2.yaml \
  --expected-per-image 10 \
  --publication
```

Publication validation rejects unknown default-family aliases and any model
that is not both public and publishable.
