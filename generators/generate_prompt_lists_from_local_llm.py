from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_LISTS_DIR, WORKFLOWS_DIR
from prompt_generator import load_prompt_list_lines
from generators.generate_prompt_list_demo_v1 import generate_prompt_list_demo_v1


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_API_KEY = "local-openai-compatible"
DEFAULT_EXAMPLE_DIR = PROMPT_LISTS_DIR / "prompt_list_demo_v1"
DEFAULT_OUTPUT_DIR = PROMPT_LISTS_DIR / "prompt_list_demo_v1"
DEFAULT_PROMPT_LIST_ROOT = PROMPT_LISTS_DIR
DEFAULT_WORKFLOW_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow_prompt_list_demo_v1"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 350
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3
PROMPT_FILE_NAMES = {
    "illustration": "illustration.txt",
    "anime": "anime.txt",
    "pony": "pony.txt",
}
FAMILY_ORDER = ("illustration", "anime", "pony")
PONY_PREFIX = "score_9, score_8_up, score_7_up, score_6_up"


def load_family_prompt_lines(example_dir: Path) -> dict[str, list[str]]:
    prompt_lines: dict[str, list[str]] = {}
    for family_name, file_name in PROMPT_FILE_NAMES.items():
        path = example_dir / file_name
        if not path.is_file():
            raise ValueError(f"Missing required family prompt file: {path}")
        lines = load_prompt_list_lines(path)
        if not lines:
            raise ValueError(f"Family prompt file is empty: {path}")
        prompt_lines[family_name] = lines
    _validate_shared_line_count(prompt_lines, source_dir=example_dir)
    return prompt_lines


def load_scene_inputs(*, scenes: Sequence[str] | None = None, scene_file: str | Path | None = None) -> list[str]:
    collected: list[str] = []
    for scene in scenes or ():
        stripped = str(scene).strip()
        if stripped:
            collected.append(stripped)
    if scene_file not in (None, ""):
        collected.extend(load_prompt_list_lines(Path(scene_file)))
    if not collected:
        raise ValueError("Provide at least one scene via --scene or --scene-file")
    return collected


def _validate_shared_line_count(prompt_lines: Mapping[str, Sequence[str]], *, source_dir: Path) -> int:
    counts = {family_name: len(lines) for family_name, lines in prompt_lines.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f"Prompt list files in {source_dir} must have the same usable line count, got {counts}"
        )
    return next(iter(unique_counts))


def format_aligned_examples(prompt_lines: Mapping[str, Sequence[str]]) -> str:
    line_count = _validate_shared_line_count(prompt_lines, source_dir=Path("<memory>"))
    rows: list[str] = []
    for index in range(line_count):
        rows.append(
            "\n".join(
                [
                    f"Example {index + 1}:",
                    f"illustration: {prompt_lines['illustration'][index]}",
                    f"anime: {prompt_lines['anime'][index]}",
                    f"pony: {prompt_lines['pony'][index]}",
                ]
            )
        )
    return "\n\n".join(rows)


def build_system_prompt() -> str:
    return (
        "You rewrite rough scene ideas into aligned family prompts for local image models.\n"
        "Return JSON only with exactly these keys: illustration, anime, pony.\n"
        "All three values must describe the same underlying scene and preserve the user's main facts.\n"
        "You may add coherent camera, lighting, mood, clothing, props, and environmental details.\n"
        "When characters are needed, use the literal placeholders chara and charb instead of names.\n"
        "Never use markdown, bullet lists, code fences, explanations, or wildcard syntax.\n"
        "Each value must be a single-line prompt string.\n"
        "Family rules:\n"
        "- illustration: coherent natural-language or mixed natural-language prompt, readable and composed.\n"
        "- anime: compact anime caption/tag line for the same scene, no pony score prefix.\n"
        f"- pony: must begin with '{PONY_PREFIX}' and remain a comma-separated single line.\n"
        "Keep the three outputs clearly parallel to each other."
    )


def build_chat_messages(*, scene: str, prompt_lines: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    examples = format_aligned_examples(prompt_lines)
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": (
                "Reference aligned family examples:\n"
                f"{examples}\n\n"
                "Generate one new aligned row for this rough scene.\n"
                "Keep the same scene across all families and let the model add coherent detail.\n"
                f"Rough scene: {scene}\n\n"
                "Return JSON only in this shape:\n"
                '{"illustration":"...","anime":"...","pony":"..."}'
            ),
        },
    ]


def _message_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text_value = item.get("text")
                if text_value not in (None, ""):
                    parts.append(str(text_value))
            elif item not in (None, ""):
                parts.append(str(item))
        return "".join(parts).strip()
    return ""


def _extract_message_text_candidates(payload: Mapping[str, Any]) -> list[str]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Chat response missing choices[0].message") from exc
    if not isinstance(message, Mapping):
        raise ValueError("Chat response choices[0].message must be an object")

    candidates: list[str] = []
    for key in ("content", "reasoning_content"):
        text = _message_value_to_text(message.get(key))
        if text:
            candidates.append(text)
    if not candidates:
        raise ValueError("Chat response content and reasoning_content were empty or not text")
    return candidates


def _iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1].strip())
                start_index = None

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _parse_json_objects(text: str) -> list[Mapping[str, Any]]:
    parsed_objects: list[Mapping[str, Any]] = []
    for candidate in _iter_json_object_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            parsed_objects.append(parsed)
    return parsed_objects


def _parse_json_object(text: str) -> Mapping[str, Any]:
    parsed_objects = _parse_json_objects(text)
    if not parsed_objects:
        raise ValueError("Model reply was not valid JSON")
    return parsed_objects[0]


def _coerce_family_response(parsed: Mapping[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for family_name in FAMILY_ORDER:
        if family_name not in parsed:
            raise ValueError(f"Model reply missing required key: {family_name}")
        value = parsed[family_name]
        if not isinstance(value, str):
            raise ValueError(f"Model reply key {family_name} must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"Model reply key {family_name} must not be empty")
        if "\n" in normalized or "\r" in normalized:
            raise ValueError(f"Model reply key {family_name} must be a single line")
        if "```" in normalized:
            raise ValueError(f"Model reply key {family_name} must not contain markdown fences")
        if normalized.startswith("{") or normalized.endswith("}"):
            raise ValueError(f"Model reply key {family_name} must not contain nested JSON")
        outputs[family_name] = normalized
    if not outputs["pony"].startswith(PONY_PREFIX):
        raise ValueError(f"Pony output must start with {PONY_PREFIX!r}")
    return outputs


def parse_family_response(content: str) -> dict[str, str]:
    parsed_objects = _parse_json_objects(content)
    if not parsed_objects:
        raise ValueError("Model reply was not valid JSON")

    last_error: ValueError | None = None
    for parsed in parsed_objects:
        try:
            return _coerce_family_response(parsed)
        except ValueError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("Model reply was not valid JSON")


def _parse_family_response_from_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    errors: list[str] = []
    for text in _extract_message_text_candidates(payload):
        try:
            return parse_family_response(text)
        except ValueError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no text candidates"
    raise ValueError(f"Model reply did not contain valid prompt JSON: {detail}")


def request_family_prompts(
    *,
    base_url: str,
    model: str,
    api_key: str,
    scene: str,
    prompt_lines: Mapping[str, Sequence[str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    session: requests.Session,
) -> dict[str, str]:
    response = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": build_chat_messages(scene=scene, prompt_lines=prompt_lines),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=float(timeout_seconds),
    )
    response.raise_for_status()
    payload = response.json()
    return _parse_family_response_from_payload(payload)


def _resolve_regenerate_workflow(regenerate_workflow: bool | None, output_dir: Path) -> bool:
    if regenerate_workflow is not None:
        return bool(regenerate_workflow)
    return output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()


def resolve_api_key(api_key: str | None) -> str:
    if api_key not in (None, ""):
        return str(api_key)
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    return DEFAULT_API_KEY


def _write_prompt_files(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for family_name, file_name in PROMPT_FILE_NAMES.items():
        path = output_dir / file_name
        content = "\n".join(row[family_name] for row in rows)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")


def _regenerate_prompt_list_demo_workflow(
    *,
    prompt_list_root: Path,
    prompt_list_source_dir: Path,
    workflow_output_dir: Path,
) -> None:
    generate_prompt_list_demo_v1(
        output_dir=workflow_output_dir,
        prompt_list_root=prompt_list_root,
        prompt_list_source_dir=prompt_list_source_dir,
    )


def generate_prompt_lists_from_local_llm(
    *,
    model: str,
    scenes: Sequence[str] | None = None,
    scene_file: str | Path | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    example_dir: str | Path = DEFAULT_EXAMPLE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prompt_list_root: str | Path = DEFAULT_PROMPT_LIST_ROOT,
    workflow_output_dir: str | Path = DEFAULT_WORKFLOW_OUTPUT_DIR,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    regenerate_workflow: bool | None = None,
    dry_run: bool = False,
    api_session: requests.Session | None = None,
    workflow_regenerator: Callable[..., None] | None = None,
) -> dict[str, list[str]]:
    example_dir_path = Path(example_dir)
    output_dir_path = Path(output_dir)
    prompt_list_root_path = Path(prompt_list_root)
    workflow_output_dir_path = Path(workflow_output_dir)
    should_regenerate = _resolve_regenerate_workflow(regenerate_workflow, output_dir_path)
    resolved_scenes = load_scene_inputs(scenes=scenes, scene_file=scene_file)
    example_prompt_lines = load_family_prompt_lines(example_dir_path)

    session = api_session or requests.Session()
    generated_rows: list[dict[str, str]] = []
    try:
        for scene in resolved_scenes:
            last_error: Exception | None = None
            for _attempt_index in range(max(1, int(retries))):
                try:
                    generated_rows.append(
                        request_family_prompts(
                            base_url=base_url,
                            model=model,
                            api_key=resolve_api_key(api_key),
                            scene=scene,
                            prompt_lines=example_prompt_lines,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_seconds=timeout_seconds,
                            session=session,
                        )
                    )
                    last_error = None
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
    finally:
        if api_session is None:
            session.close()

    outputs = {
        family_name: [row[family_name] for row in generated_rows]
        for family_name in FAMILY_ORDER
    }
    if dry_run:
        return outputs

    _write_prompt_files(output_dir_path, generated_rows)
    if should_regenerate:
        if workflow_regenerator is None:
            workflow_regenerator = _regenerate_prompt_list_demo_workflow
        workflow_regenerator(
            prompt_list_root=prompt_list_root_path,
            prompt_list_source_dir=output_dir_path,
            workflow_output_dir=workflow_output_dir_path,
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call a localhost OpenAI-compatible chat/completions API and overwrite the 3 family prompt-list files."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--example-dir", default=str(DEFAULT_EXAMPLE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-list-root", default=str(DEFAULT_PROMPT_LIST_ROOT))
    parser.add_argument("--workflow-output-dir", default=str(DEFAULT_WORKFLOW_OUTPUT_DIR))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--scene-file", default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--regenerate-workflow", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    should_regenerate = _resolve_regenerate_workflow(args.regenerate_workflow, Path(args.output_dir))
    outputs = generate_prompt_lists_from_local_llm(
        model=args.model,
        scenes=args.scene,
        scene_file=args.scene_file,
        base_url=args.base_url,
        api_key=args.api_key,
        example_dir=args.example_dir,
        output_dir=args.output_dir,
        prompt_list_root=args.prompt_list_root,
        workflow_output_dir=args.workflow_output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        retries=args.retries,
        regenerate_workflow=args.regenerate_workflow,
        dry_run=args.dry_run,
    )
    line_count = len(outputs["illustration"])
    print(f"Generated {line_count} aligned prompt rows into {Path(args.output_dir)}")
    if should_regenerate and not args.dry_run:
        print(f"Workflow output dir: {Path(args.workflow_output_dir)}")


if __name__ == "__main__":
    main()
