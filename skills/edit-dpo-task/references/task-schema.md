# Task Schema

## Top Level

Required fields:

```yaml
version: 1
task_name: stable_identifier
global_seed: 20260422
session_count: 10
images: []
```

`session_count` is the number of requests generated for every image entry. A
two-image task with 10 sessions produces 20 requests and 10 session rows.

## Image Entry

Every image requires:

```yaml
- image_name: image1
  workflow_name: SdxlEaseLoraWorkflow
  ckpt: sdxl_novaAnimeXL_ilV150.safetensors
  lora_stack_config: {}
  prompt_generator: {}
  sample: {}
```

`ckpt` can be a scalar or a deterministic weighted option specification:

```yaml
ckpt:
  seed_control: session_seed
  options:
    - value: model_a.safetensors
      weight: 1.0
    - value: model_b.safetensors
      weight: 1.0
```

Valid seed controls are `global_seed`, `session_seed`, and
`image_index_seed`.

## Checkpoint Registry

`checkpoint_aliases.yaml` maps stable model IDs to one or more local aliases:

```yaml
models:
  nova-anime-xl-v150:
    aliases:
      - sdxl_novaAnimeXL_ilV150.safetensors
```

Optional model fields:

```yaml
family: sdxl_anime_base
visibility: public
publish: true
```

Resolution precedence is:

1. Exact full alias, then an unambiguous basename alias.
2. Explicit model `family`.
3. First matching ordered `keyword_rules` substring.
4. `default_family` compatibility fallback.

Do not publish a task that reaches step 4. Add an alias and either a reliable
keyword or explicit family. Put private additions in ignored
`checkpoint_aliases.local.yaml`, or set the `CHECKPOINT_ALIAS_REGISTRY`
environment variable to an overlay YAML.

## Prompt Routing

Wildcard tasks can route templates and negatives by family:

```yaml
prompt_generator:
  name: wildcard_template_generator
  args:
    seed_control: session_seed
    template: research_v8/sfw_hybrid_compact_sdxl_anime_base
    template_by_ckpt_family:
      illustration: research_v8/sfw_hybrid_compact_illustration
      sdxl_anime_base: research_v8/sfw_hybrid_compact_sdxl_anime_base
      pony: research_v8/sfw_hybrid_compact_pony
      realistic: research_v8/sfw_hybrid_compact_realistic
    template_root: ../prompt_templates
    wildcard_root: ../wildcard
    negative_prompt: fallback negative prompt
    negative_prompt_by_ckpt_family:
      illustration: illustration negative prompt
      sdxl_anime_base: anime negative prompt
      pony: pony negative prompt
      realistic: realistic negative prompt
```

`prompt_list_v1` uses `prompt_list_by_ckpt_family`. Non-wildcard family-routed
generators use positive prefix/suffix and negative mappings.

## Sample Settings

Required sample fields are `generation_seed_control`, `steps`, `cfg`, `width`,
and `height`. Each value can be a scalar or weighted option specification.
Use matching `session_seed` controls for controlled pairs. Use
`image_index_seed` for independent candidates.

## LoRA

For no-LoRA tasks:

```yaml
lora_stack_config:
  toggle: false
  mode: simple
  num_loras: 1
  lora_1_name: None
  lora_1_strength: 1.0
  lora_1_model_strength: 1.0
  lora_1_clip_strength: 1.0
```

When enabled, register every intended LoRA slot explicitly and verify the files
exist in ComfyUI.
