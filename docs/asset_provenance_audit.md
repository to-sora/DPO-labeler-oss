# Asset Provenance and Redistribution Audit

Audit date: 2026-08-17

This audit covers tracked prompt assets, wildcard packs referenced by
`assets/wildcard_sources.json`, and model repositories named by the image grader
configuration. It is a release-engineering assessment, not legal advice.

## Release Decision

The public repository stores neither third-party wildcard archives nor their
expanded download directories. Its one-commit history was initialized from the
verified source tree, so superseded development objects are not reachable from
the publication branch. Keep any separately retained local Git backup private.

Seven downloaded Civitai wildcard packs have no attached formal license. The
current Civitai permission fields are not a conventional open-source license,
and the site's UI describes them in terms of generated images, generation
services, and model merges. Civitai's Terms grant public-content rights to users
"through the Service"; they do not clearly grant a right to rehost the exact
download on GitHub.

The two strongest blockers are:

- The uploader of the 5770-character pack says they did not create it and do
  not know who did.
- The mixed-artist pack says it came from a Chinese community and credits an
  unidentified original author. Its retained wildcard text is an exact member
  of that derived archive.

## Publication Path

1. Keep `assets/comfy_prompt_payload.zip*` and all downloaded wildcard
   directories outside every public commit and tag.
2. Retain `assets/wildcard_sources.json` and `install_wildcard.sh`; they pin the
   original Civitai versions and checksums and require explicit user acceptance.
3. Prefer first-party replacement wildcard data under MIT, CC0, or another
   explicit license if a durable fully open distribution is required.
4. If redistribution is later desired, obtain written permission from the
   actual rights holder. Uploader permission is insufficient for packs whose
   uploaders disclaim or do not establish authorship.
5. Retain `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt` for the WD tag
   metadata-derived files.
6. Before publication, the repository owner must confirm authorship or a valid
   license for the unmarked `research_v2` through `research_v5`, `research_v8`,
   custom character, prompt-list, and prompt-template content.

The installer preserves existing behavior without committing the seven packs.
It reproduces path repairs, aliases, duplicate directories, and curated
`research_v5/imported` outputs, then verifies deterministic tree hashes.

## Civitai Wildcard Inventory

All seven model records currently report an empty `licenses` array. Permission
fields below are recorded for provenance only; they are not treated as a grant
to relicense the files under MIT.

| Bundled directory | Civitai model/version | Upstream SHA-256 | Current permission metadata | Finding |
| --- | --- | --- | --- | --- |
| `200WildcardsNSFWAnd_v20` | `20868/532723` | `835866756c867fabf405a3af49f6bbd901616fe4fd7418c0b38d509239faf348` | no credit; derivatives; no commercial uses | Installed directly from the pinned upstream archive. No formal license. |
| `5770AnimeAndGameCharacters_v10` | `321794/360839` | `976c0ce9b24639c6b51c23de78a4e616fc742ae3802d7ceffffbf28410f238cb` | no credit; derivatives; commercial fields enabled | Uploader expressly disclaims authorship. Local copy also has path fixes and three added wrapper files. Remove and replace. |
| `Naiv3IllustriousXLMixedArtist_v10` | `924258/1034550` | `9b6b62770544751d4dfbd3bc7fa5135add46d93c80ec89239d5e898fbbc09b1c` | no credit; derivatives; commercial fields enabled | Derived from an unidentified Chinese-community source. Retained TXT SHA-256 is `2795610f2c3e2f166b5a4cc74fb81b036f58cf50268883c4e1f01a698ba9c511`. Remove and replace. |
| `advancedWildcardsSexyMaidKit_v10` | `76968/81762` | `779d2756d41f31eb148f6b668b928a68be0b5c80a06405c2b18ea4419e6606d2` | no credit; derivatives; no commercial uses | Exact match; duplicated as a second identical local directory. No formal license. Obtain permission or replace. |
| `clothesWildcards_v10` | `73184/77904` | `a7684162f9fbbcad6f9c3f3345e0254db350246418670804ccb9826f532482c5` | no credit; derivatives; commercial fields enabled | Base files match upstream; local tree adds renamed duplicate files used by templates. No formal license. Obtain permission or replace. |
| `organicComposable_v10` | `138152/154598` | `bce04865a70b0182372bb43999358c6f931ead0cd9fa550d6bffefd577d5a285` | no credit; no derivatives; no commercial uses | Installed directly from upstream. The uploader says it was derived from tags on 17,000 Danbooru images. Do not redistribute it from this repository. |
| `wildcardsSexual_v10` | `1177320/1324764` | `341d2d2f71a6e17cac67759cc1e6074fcc558e3760606b4dea61768a45314bde` | credit required; derivatives; Civitai rental only | No formal license and restrictive metadata. Obtain permission or replace. |

Source pages:

- <https://civitai.com/models/20868?modelVersionId=532723>
- <https://civitai.com/models/321794?modelVersionId=360839>
- <https://civitai.com/models/924258?modelVersionId=1034550>
- <https://civitai.com/models/76968?modelVersionId=81762>
- <https://civitai.com/models/73184?modelVersionId=77904>
- <https://civitai.com/models/138152?modelVersionId=154598>
- <https://civitai.com/models/1177320?modelVersionId=1324764>

Civitai terms and permission semantics:

- <https://github.com/civitai/civitai/blob/main/src/static-content/tos.md>
- <https://github.com/civitai/civitai/blob/main/src/components/PermissionIndicator/PermissionIndicator.tsx>

## Licensed Tag Metadata

The duplicated `research_v6` and `research_v7`
`wd14_vit_v3_tags_zh_semantic.csv` files identify these inputs:

- `SmilingWolf/wd-vit-tagger-v3/selected_tags.csv`
- `deepghs/tags_meta`

Both upstream Hugging Face repositories declare Apache-2.0. The local files add
translations, descriptions, semantic classifications, and manual rating rows.
Generated category and runtime wildcard files in those two research directories
are derived from the same data. The required license copy and attribution are
included in this repository.

Sources:

- <https://huggingface.co/SmilingWolf/wd-vit-tagger-v3>
- <https://huggingface.co/datasets/deepghs/tags_meta>

## Referenced Model Weights

No model weight file is tracked in this repository. The grader configuration
names repositories that users must obtain separately. Their licensing therefore
does not block publication of the code-only repository, but users and packagers
must follow each model's terms.

| Model | Reported license/status | Release handling |
| --- | --- | --- |
| `openai/clip-vit-large-patch14` | OpenAI CLIP source is MIT; the model card limits intended use and does not clearly label the HF weights | Do not bundle weights; review the model card for deployment. |
| `Eugeoter/waifu-scorer-v3` | Conflicting metadata: HF front matter says OpenRAIL while model-card text says Apache-2.0 | Do not redistribute until the author resolves the mismatch. |
| `shadowlilac/aesthetic-shadow` | No formal license reported | Local use only unless permission is obtained. |
| `minizhu/aesthetic-anime-v2` | CC-BY-NC-4.0 | Do not use for commercial operation or bundle without complying with attribution and license terms. |
| `Blackroot/Anime-Aesthetic-Predictor-Medium` | MIT | May be obtained separately under its repository terms. |

Sources:

- <https://github.com/openai/CLIP/blob/main/LICENSE>
- <https://huggingface.co/openai/clip-vit-large-patch14>
- <https://huggingface.co/Eugeoter/waifu-scorer-v3>
- <https://huggingface.co/shadowlilac/aesthetic-shadow>
- <https://huggingface.co/minizhu/aesthetic-anime-v2>
- <https://huggingface.co/Blackroot/Anime-Aesthetic-Predictor-Medium>

## Runtime Parity Verification

The installer intentionally excludes:

- `template/wildcard/Naiv3IllustriousXLMixedArtist_v10/mixed_artists_style.html`
- `template/wildcard/Naiv3IllustriousXLMixedArtist_v10/single_artist_style.html`

The wildcard resolver reads the retained `mixed_artists_wildcards.txt`; neither
HTML viewer is a runtime input. A local end-to-end installation reconstructed
all seven source archives, applied every manifest operation, and matched the
validated compatibility layout byte-for-byte for all publishable prompt,
wildcard, and workflow files.
