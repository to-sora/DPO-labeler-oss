Pre-quality prompt pool backup:

- `mix_gpt.txt`
- `mix_qwen.txt`

The baseline versions are preserved in git commit `c1d796c` (`Checkpoint DPO workflow baseline`).
The quality pass uses new curated family-specific templates under `prompt_templates/curated/`
instead of editing those original pool files in place.

That curated set is now the frozen V1 baseline. New prompt investigations should be added under
`prompt_templates/research_v2/`.
