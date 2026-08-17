Literal prompt-list files live here.

Format:

- one usable positive prompt per line
- commas are allowed
- blank lines are ignored
- lines starting with `#` are ignored
- `__tokens__` and `{a|b}` stay literal and are not expanded

The `prompt_list_demo_v1/` subdirectory is the placeholder/example layout for larger prompt banks.

To batch-generate aligned `illustration.txt`, `anime.txt`, and `pony.txt` files through a localhost OpenAI-compatible API, use:

```bash
python3 generators/generate_prompt_lists_from_local_llm.py \
  --model your-local-model \
  --scene "adult woman in a black coat on a rainy city street"
```
