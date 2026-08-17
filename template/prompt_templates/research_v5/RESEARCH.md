# V5 Adult Prompt Research

This directory starts as a clone of the V4 prompt investigation set for the next amendment stage.
It does not replace the frozen V1 baseline under `prompt_templates/curated/`, and it leaves V4 untouched.

## Source Links

- NovelAI Basics: https://docs.novelai.net/image/basics.html?highlight=nsfw+prompts
- NovelAI Undesired Content: https://docs.novelai.net/en/image/undesiredcontent/
- NovelAI Character Creation: https://docs.novelai.net/en/image/tutorial-charactercreation
- Stable Diffusion Art Prompt Guide: https://stable-diffusion-art.com/prompt-guide/
- Stable Diffusion Art Realistic People: https://stable-diffusion-art.com/realistic-people/
- Civitai Prompt Builder reference: https://civitai.green/models/115347/the-prompt-builder?modelVersionId=1586646
- PirateDiffusion facial expression guide: https://piratediffusion.com/prompting-a-wide-range-of-facial-expressions-in-stable-diffusion/
- Civitai boudoir expression/posture wildcard reference: https://civitai.green/models/487515/facial-expressions-body-postures-and-boudoir-erotica?modelVersionId=2050486
- Animagine XL 3.1 model card: https://huggingface.co/cagliostrolab/animagine-xl-3.1
- Pony Diffusion V6 XL model card: https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL
- PonyXL prompt guide: https://docs.moescape.ai/image-generation-guide/ponyxl-model-guide
- Illustrious prompting article: https://archive.ph/2025.12.28-112836/https%3A/civitai.com/articles/8380/tips-for-illustrious-xl-prompting-updates

## Rules Used for V5

- Split prompt families into four dialects:
  - `illustration`
  - `sdxl_anime_base`
  - `pony`
  - `realistic`
- Use explicit checkpoint routing for those families instead of one generic `anime` bucket.
- `illustration` uses quality-first aesthetic tags, then subject/body/clothing/pose, then `BREAK`,
  then background and lighting tags.
- `sdxl_anime_base` uses count/identity-first original-character prompting and ends with quality tags.
- `pony` uses score-prefix-first prompting.
- `realistic` keeps photo/cinematic wording.
- Keep the subject block adult-only and move expression, gaze, and face detail before pose so the
  face stays readable.
- Keep optional sensual modifiers in removable comma-delimited segments so dropout can remove them
  cleanly without breaking the scene structure.
