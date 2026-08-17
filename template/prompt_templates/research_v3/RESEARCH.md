# V3 Adult Prompt Research

This directory starts as a clone of the V2 prompt investigation set for the next amendment stage.
It does not replace the frozen V1 baseline under `prompt_templates/curated/`.

## Source Links

- NovelAI Basics: https://docs.novelai.net/image/basics.html?highlight=nsfw+prompts
- NovelAI Undesired Content: https://docs.novelai.net/en/image/undesiredcontent/
- NovelAI Character Creation: https://docs.novelai.net/en/image/tutorial-charactercreation
- Stable Diffusion Art Prompt Guide: https://stable-diffusion-art.com/prompt-guide/
- Stable Diffusion Art Realistic People: https://stable-diffusion-art.com/realistic-people/
- Civitai Prompt Builder reference: https://civitai.green/models/115347/the-prompt-builder?modelVersionId=1586646
- PirateDiffusion facial expression guide: https://piratediffusion.com/prompting-a-wide-range-of-facial-expressions-in-stable-diffusion/
- Civitai boudoir expression/posture wildcard reference: https://civitai.green/models/487515/facial-expressions-body-postures-and-boudoir-erotica?modelVersionId=2050486

## Rules Used for V3

- Keep the important subject and adult-age anchors in the front half of the prompt.
- Keep each prompt line short and coherent; one line should describe one scene.
- Use a hybrid format: short scene stub first, compact ordered tag tail second.
- Use family-specific negatives instead of one generic negative prompt for every checkpoint.
- Use small wildcard categories for pose, framing, camera, lighting, location, wardrobe state,
  identity, body style, restraint-coded styling, and one detail token rather than large noisy wildcard packs.
- Keep optional sensual modifiers in removable comma-delimited segments so dropout can remove them
  cleanly without breaking the scene structure.
- Keep the subject block adult-only and move expression, gaze, and face detail before pose so the
  face stays readable.
- Favor adult androgynous, femboy, non-binary, and voluptuous glamour styling through identity,
  pose, and wardrobe cues rather than explicit act language.
- This V3 set starts from the V2 adult portrait and sexy-clothing baseline and is intended for the
  next amendment pass.
