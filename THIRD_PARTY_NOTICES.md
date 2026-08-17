# Third-Party Notices

The repository MIT license applies to the project's original source code. It
does not relicense third-party assets or separately obtained model weights.

## WD Tag Metadata

Files under `template/wildcard/research_v6/` and
`template/wildcard/research_v7/` include data derived from:

- `SmilingWolf/wd-vit-tagger-v3/selected_tags.csv`
  (<https://huggingface.co/SmilingWolf/wd-vit-tagger-v3>)
- `deepghs/tags_meta`
  (<https://huggingface.co/datasets/deepghs/tags_meta>)

Both upstream repositories declare the Apache License 2.0. This distribution
includes translated descriptions, semantic classifications, manual rating rows,
and generated category/runtime wildcard lists. See
`LICENSES/Apache-2.0.txt` for the license text.

## Civitai Wildcard Packs

Seven Civitai wildcard downloads are referenced by
`assets/wildcard_sources.json`. Their formal redistribution licenses could not
be established, so the files are not stored in Git and are not licensed under
this repository's MIT license. `install_wildcard.sh` downloads them from their
original Civitai version URLs only after explicit user acceptance.

See `docs/asset_provenance_audit.md` for the exact inventory, checksums, source
links, dependency findings, and required release action.

## Model Weights

Model weights are not distributed by this repository. Names in
`image_grader/config.example.json` are references only. Anyone obtaining those
weights must review and comply with the model repository's current license and
use restrictions.
