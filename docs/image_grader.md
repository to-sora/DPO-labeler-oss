# Image Grader

## Scope

The optional grader scores existing generated images. It does not generate
images and does not download model weights. The API and admin/playground share
`image_grader/config.local.json`, while keeping separate runtime state paths.

These models primarily estimate aesthetic or learned quality preferences. They
are not dependable detectors for malformed hands, fingers, anatomy, text, or
every diffusion failure. Human review remains authoritative.

## Dependencies

Install into the Python environment configured as `GRADER_PYTHON_BIN`:

```bash
bash install.sh --non-interactive --skip-wildcards --install-grader-deps cpu
```

For a compatible NVIDIA environment:

```bash
bash install.sh --non-interactive --skip-wildcards --install-grader-deps cuda13
```

The profiles install both `image_grader/requirements.txt` and one runtime file:

```text
image_grader/requirements-cpu.txt
image_grader/requirements-cuda13.txt
```

Set `device` in `image_grader/config.local.json` to `cpu` or `cuda` as
appropriate. A CUDA package installation does not automatically change that
configuration field.

## Model Layout

The default model root is `models/image_eval`.

`waifu_scorer_v3` requires both its MLP head and CLIP backbone:

```text
models/image_eval/Eugeoter/waifu-scorer-v3/model.safetensors
models/image_eval/openai/clip-vit-large-patch14/config.json
models/image_eval/openai/clip-vit-large-patch14/preprocessor_config.json
models/image_eval/openai/clip-vit-large-patch14/model.safetensors
```

The head is approximately 11 MB, but the complete pair is approximately
1.72 GB because the CLIP backbone is about 1.71 GB.

The transformer graders require complete local model directories:

```text
models/image_eval/shadowlilac/aesthetic-shadow/
models/image_eval/minizhu/aesthetic-anime-v2/
```

The `blackroot_medium` example uses a pickle-backed checkpoint and is disabled
by default. Do not set `trusted_pickle: true` unless the exact file source is
trusted.

Remove unavailable model IDs from `enabled_models` or leave them unselected in
the playground. Selecting an unavailable model produces a structured score
error; if every selected score fails, the UI displays a clear terminal error
state and the individual model errors.

## Start

Enable `START_IMAGE_GRADER=1` and/or `START_ADAPTER=1` in `.env.oss`, then run:

```bash
bash start_all_endpoints.sh
```

The defaults are:

```text
Image-grader API:       http://127.0.0.1:8790/
Grader admin/playground http://127.0.0.1:8087/
```

The admin scans `DATASET_ROOT` recursively for collected `sessions.jsonl`
files. Its score cache is keyed by image fingerprint, model, preprocessing
policy, and runtime configuration.

## Preprocessing Policies

- `native`: score the decoded image without forcing a square crop.
- `fit_pad_square`: resize to fit a square canvas and pad the remainder.
- `center_crop_square`: resize and center-crop to a square.

Crop choice can materially change a score because it changes composition,
subject scale, and retained detail. Comparing several policies is useful, but
maxima and minima are sensitive to crop failures. Median or interquartile
formulas are usually more stable than selecting an extreme.

## Ranking Expressions

The canonical direct lookup is:

```text
score["crop_method"]["grade_model"]
```

For example:

```text
score["native"]["waifu_scorer_v3"]
```

The default expression adds every selected model/policy key and divides by the
number of keys:

```text
(score["native"]["waifu_scorer_v3"] + score["fit_pad_square"]["waifu_scorer_v3"] + score["center_crop_square"]["waifu_scorer_v3"]) / 3
```

Supported operators:

```text
+  -  *  /
```

Supported aggregation and helper functions:

```text
avg mean sum min max count abs round
percentile quantile
p10 p25 p50 p75 p90
lower_quartile upper_quartile
scores score model method policy
```

Useful examples:

```text
avg(scores())
percentile(scores(), 50)
(percentile(scores(), 25) + percentile(scores(), 75)) / 2
max(p25(scores()), min(avg(scores()), p75(scores())))
avg(scores()) - abs(score["native"]["waifu_scorer_v3"] - score["center_crop_square"]["waifu_scorer_v3"]) * 0.25
percentile(score["native"]["waifu_scorer_v3"], score["fit_pad_square"]["waifu_scorer_v3"], score["center_crop_square"]["waifu_scorer_v3"], 10)
```

Semicolons are normalized to commas, so this is also accepted:

```text
percentile(value1, value2, value3; 75)
```

`scores()` returns all usable selected scores for the current image. It can be
filtered with a selected model or method, for example:

```text
percentile(scores("waifu_scorer_v3"), 50)
avg(scores("native"))
score("waifu_scorer_v3", "native")
```

The evaluator parses a restricted Python expression AST. It does not support
attribute access, imports, comprehensions, lambdas, arbitrary function calls,
or keyword arguments. Formula errors invalidate that image's rank value instead
of executing general Python.

The top/bottom bucket percentage must be greater than zero and no more than 50.
With a very small number of ranked images, percentile buckets round up to at
least one image.

## Choosing A Robust Formula

When crop variation is moderate but model outliers are extreme, start with a
median within each model rather than a maximum/minimum:

```text
percentile(scores("waifu_scorer_v3"), 50)
```

For several grader models, compute a per-model crop median and then combine the
model medians conservatively:

```text
min(
  percentile(scores("waifu_scorer_v3"), 50),
  percentile(scores("aesthetic_shadow"), 50),
  percentile(scores("aesthetic_anime_v2"), 50)
)
```

That formula filters images when any model's crop-median quality is low. It
still inherits each model's blind spots, so inspect the resulting best and
worst buckets before using it to discard data.

## Verified CPU Result

The release was tested once on a LAN CPU client with Python 3.13,
`torch 2.13.0+cpu`, and `torchvision 0.28.0+cpu`:

- one selected session and two images;
- `waifu_scorer_v3` with `native` preprocessing;
- two successful numeric scores and zero errors;
- browser request duration: 5.295 seconds;
- peak adapter resident memory: approximately 1.62 GiB;
- swap activity reported by the measured process: zero.

The test used one browser click and one score run. It also verified that the UI
rendered ranking cards without null dereferences, `[object Object]`, or
`undefined` placeholders.
