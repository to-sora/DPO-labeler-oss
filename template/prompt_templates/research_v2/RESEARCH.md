# V2 Adult Prompt Research

This directory holds the separate V2 prompt investigation set. It does not replace the frozen V1
baseline under `prompt_templates/curated/`.

## Source Links

- NovelAI Basics: https://docs.novelai.net/image/basics.html?highlight=nsfw+prompts
- NovelAI Undesired Content: https://docs.novelai.net/en/image/undesiredcontent/
- Stable Diffusion Art Prompt Guide: https://stable-diffusion-art.com/prompt-guide/
- Stable Diffusion Art Realistic People: https://stable-diffusion-art.com/realistic-people/
- Civitai Prompt Builder reference: https://civitai.green/models/115347/the-prompt-builder?modelVersionId=1586646

## Rules Used for V2

- Keep the important subject and adult-age anchors in the front half of the prompt.
- Keep each prompt line short and coherent; one line should describe one scene.
- Use a hybrid format: short scene stub first, compact ordered tag tail second.
- Use family-specific negatives instead of one generic negative prompt for every checkpoint.
- Use small wildcard categories for pose, framing, camera, lighting, location, wardrobe state,
  and one detail token rather than large noisy wildcard packs.
- Keep optional sensual modifiers in removable comma-delimited segments so dropout can remove them
  cleanly without breaking the scene structure.
- Keep this V2 set at adult portrait and nudity scope only. Do not add act-centric prompts here.
