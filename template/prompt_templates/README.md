For task YAMLs under `template/tasks/`, use these templates with:

- `prompt_generator.name: wildcard_template_generator`
- `prompt_generator.args.seed_control: session_seed` (or another explicit seed selector)
- `prompt_generator.args.template_root: ../prompt_templates`
- `prompt_generator.args.wildcard_root: ../wildcard`

The current `curated/` prompt pools are the frozen V1 baseline. Versioned prompt research lives
under `research_v2/` through `research_v8/` so the V1 set stays unchanged.

These templates intentionally use full wildcard paths so one prompt can draw from multiple packs under the repo's `template/wildcard/` directory.

Template files may also contain multiple non-empty lines. In that case the wildcard generator
deterministically selects one line per session using `session_id % line_count`, then expands
wildcards inside that line.

The `curated/` subdirectory contains family-specific prompt pools used by the DPO workflow matrix:

- `*_anime_curated.txt`
- `*_pony_curated.txt`
- `*_realistic_curated.txt`

Those files are intended to be selected through checkpoint-family routing in
`prompt_generator.args`:

```yaml
prompt_generator:
  name: wildcard_template_generator
  args:
    seed_control: session_seed
    template_root: ../prompt_templates
    wildcard_root: ../wildcard
    template_by_ckpt_family:
      anime: curated/mix_gpt_anime_curated
      pony: curated/mix_gpt_pony_curated
      realistic: curated/mix_gpt_realistic_curated
    negative_prompt_by_ckpt_family:
      anime: bad anatomy, malformed hands, extra fingers, lowres, blurry
      pony: score_6, score_5, score_4, source_furry, lowres, blurry
      realistic: anime, illustration, cartoon, painting, cgi, 3d, lowres, blurry
```

If checkpoint-family routing is present, the compiler resolves the final `template` and
`negative_prompt` from the selected checkpoint family before wildcard expansion.

Dropout lists are optional shared candidate pools. The selected line is processed first, and any
dropout item that does not appear in that line is ignored.

The `research_v2/` through `research_v8/` subdirectories are reserved for separate experimental or
versioned prompt sets. They can use the same checkpoint-family routing pattern while keeping the
V1 `curated/` pool frozen.

Preview a template from the terminal:

```bash
python3 prompt_generator.py preview-wildcard-template \
  --template illustrij_anime_01_solo_casual \
  --template-root ./template/prompt_templates \
  --wildcard-root ./template/wildcard \
  --count 3 \
  --seed 42
```
