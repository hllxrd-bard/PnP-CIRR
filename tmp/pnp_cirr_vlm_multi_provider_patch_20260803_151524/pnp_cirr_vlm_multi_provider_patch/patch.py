#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


CORE_FILES = [
    Path("cir/config.py"),
    Path("cir/schemas.py"),
    Path("cir/engine.py"),
    Path("visualize.py"),
    Path("web/templates/index.html"),
    Path("web/static/app.js"),
]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} marker, found {count}. "
            "The repository may have changed; no partial patch was kept."
        )
    return text.replace(old, new, 1)



def patch_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    # Keep the repository default aligned with the actual local Qwen service.
    text = text.replace(
        '        "base_url": "http://127.0.0.1:8000/v1",\n',
        '        "base_url": "http://192.168.20.150:8018/v1",\n',
        1,
    )
    text = text.replace(
        '        "api_key": "EMPTY",\n',
        '        "api_key": None,\n',
        1,
    )

    if '        "default_provider": "qwen",\n' not in text:
        old = '        "enabled_by_default": False,\n'
        new = old + '        "default_provider": "qwen",\n'
        text = replace_once(text, old, new, "vlm.enabled_by_default")

    path.write_text(text, encoding="utf-8")

def patch_schemas(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "vlm_provider:" in text:
        return
    old = "    use_vlm: bool | None = None\n"
    new = (
        old
        + '    vlm_provider: Literal["qwen", "gemini"] | None = None\n'
    )
    text = replace_once(text, old, new, "CIRRequest.use_vlm")
    path.write_text(text, encoding="utf-8")


def patch_engine(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "provider=request.vlm_provider" not in text:
        pattern = re.compile(
            r"(?P<head>vlm_payload\s*=\s*self\.vlm\.rewrite\(\n)"
            r"(?P<indent>\s+)(?P<next>edit_text=)"
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Could not find self.vlm.rewrite call in cir/engine.py.")
        insertion = (
            match.group("head")
            + match.group("indent")
            + "provider=request.vlm_provider,\n"
            + match.group("indent")
            + match.group("next")
        )
        text = text[: match.start()] + insertion + text[match.end() :]

    if '"vlm_provider":' not in text:
        old = '                "used_vlm": bool(vlm_payload),\n'
        new = old + (
            '                "vlm_provider": (\n'
            '                    ((vlm_payload or {}).get("_meta") or {}).get("provider")\n'
            '                ),\n'
            '                "vlm_model": (\n'
            '                    ((vlm_payload or {}).get("_meta") or {}).get("model")\n'
            '                ),\n'
            '                "vlm_http_latency_ms": (\n'
            '                    ((vlm_payload or {}).get("_meta") or {}).get("latency_ms")\n'
            '                ),\n'
        )
        text = replace_once(text, old, new, "query.used_vlm")

    path.write_text(text, encoding="utf-8")


def patch_visualize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if '"default_vlm_provider"' in text:
        return
    old = '            "vlm_model": config.get("vlm.model"),\n'
    new = old + (
        '            "default_vlm_provider": config.get(\n'
        '                "vlm.default_provider", "qwen"\n'
        '            ),\n'
        '            "qwen_vlm_model": config.get(\n'
        '                "vlm.providers.qwen.model",\n'
        '                config.get("vlm.model", "Qwen3.5-9B-Q8_0.gguf"),\n'
        '            ),\n'
        '            "gemini_vlm_model": config.get(\n'
        '                "vlm.providers.gemini.model", "gemini-3.6-flash"\n'
        '            ),\n'
    )
    text = replace_once(text, old, new, "visualize VLM template context")
    path.write_text(text, encoding="utf-8")


def patch_index(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'id="vlm-provider"' not in text:
        pattern = re.compile(
            r'(?P<indent>[ \t]*)<label class="checkbox-label">\s*'
            r'<input id="use-vlm"(?P<input_attrs>[^>]*)>\s*'
            r'(?P<label_text>VLM assist.*?)</label>',
            re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Could not find the VLM checkbox block in index.html.")
        indent = match.group("indent")
        input_attrs = match.group("input_attrs")
        replacement = (
            f'{indent}<label class="checkbox-label">\n'
            f'{indent}  <input id="use-vlm"{input_attrs}>\n'
            f'{indent}  VLM assist cho edit mơ hồ — chậm hơn khoảng vài giây\n'
            f'{indent}</label>\n'
            f'{indent}<label id="vlm-provider-group">VLM provider\n'
            f'{indent}  <select id="vlm-provider">\n'
            f'{indent}    <option value="qwen" '
            '{% if default_vlm_provider == "qwen" %}selected{% endif %}>'
            'Qwen 3.5 9B local ({{ qwen_vlm_model }})</option>\n'
            f'{indent}    <option value="gemini" '
            '{% if default_vlm_provider == "gemini" %}selected{% endif %}>'
            'Gemini API ({{ gemini_vlm_model }})</option>\n'
            f'{indent}  </select>\n'
            f'{indent}  <small>API key Gemini chỉ được đọc từ biến môi trường '
            'GEMINI_API_KEY ở backend.</small>\n'
            f'{indent}</label>'
        )
        text = text[: match.start()] + replacement + text[match.end() :]

    text = re.sub(
        r'(/static/style\.css\?v=)[^"\']+',
        r'\1vlm-providers-v1',
        text,
        count=1,
    )
    text = re.sub(
        r'(/static/app\.js\?v=)[^"\']+',
        r'\1vlm-providers-v1',
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def patch_app_js(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "function updateVlmControls()" not in text:
        marker = "function createPayload() {\n"
        function_text = (
            "function updateVlmControls() {\n"
            '  const provider = $("vlm-provider");\n'
            '  const useVlm = $("use-vlm");\n'
            "  if (!provider || !useVlm) return;\n"
            '  const directional = $("composition-mode").value === "directional";\n'
            "  provider.disabled = !directional || !useVlm.checked;\n"
            "}\n"
        )
        text = replace_once(
            text,
            marker,
            function_text + marker,
            "createPayload function",
        )

    # Keep the provider selector synchronized whenever composition mode changes.
    if "  updateVlmControls();\n}\nfunction updateVlmControls()" not in text:
        boundary = "\n}\nfunction updateVlmControls() {\n"
        text = replace_once(
            text,
            boundary,
            "\n  updateVlmControls();\n}\nfunction updateVlmControls() {\n",
            "updateCompositionMode boundary",
        )

    if "vlm_provider:" not in text:
        pattern = re.compile(
            r'(?P<indent>[ \t]+)use_vlm:\s*mode === "directional" && '
            r'\$\("use-vlm"\)\.checked,\n'
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Could not find use_vlm payload line in app.js.")
        indent = match.group("indent")
        insertion = match.group(0) + (
            f'{indent}vlm_provider: mode === "directional" && '
            '$("use-vlm").checked\n'
            f'{indent}  ? $("vlm-provider").value\n'
            f'{indent}  : null,\n'
        )
        text = text[: match.start()] + insertion + text[match.end() :]

    if "query.vlm_provider" not in text:
        old = "    `used_vlm=${query.used_vlm ?? false}`,\n"
        new = old + (
            '    query.vlm_provider ? `vlm_provider=${query.vlm_provider}` : null,\n'
            '    query.vlm_model ? `vlm_model=${query.vlm_model}` : null,\n'
            '    query.vlm_http_latency_ms != null\n'
            '      ? `vlm_http=${Number(query.vlm_http_latency_ms).toFixed(1)} ms`\n'
            '      : null,\n'
        )
        text = replace_once(text, old, new, "used_vlm query display")

    listener = '$("use-vlm").addEventListener("change", updateVlmControls);\n'
    if listener not in text:
        old = '$("composition-mode").addEventListener("change", updateCompositionMode);\n'
        text = replace_once(text, old, old + listener, "composition mode listener")

    if not text.rstrip().endswith("updateVlmControls();"):
        old = "updateCompositionMode();\n"
        text = replace_once(
            text,
            old,
            old + "updateVlmControls();\n",
            "initial updateCompositionMode call",
        )

    path.write_text(text, encoding="utf-8")


def copy_new_files(repo: Path, payload: Path) -> None:
    files_root = payload / "files"
    for source in sorted(files_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(files_root)
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def validate_repo(repo: Path) -> None:
    required = CORE_FILES + [Path("cir/vlm_client.py")]
    missing = [str(path) for path in required if not (repo / path).is_file()]
    if missing:
        raise RuntimeError("Missing expected repository files: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    payload = args.payload.resolve()
    backup = args.backup.resolve()
    validate_repo(repo)

    backup.mkdir(parents=True, exist_ok=False)
    backup_targets = CORE_FILES + [Path("cir/vlm_client.py")]
    for relative in backup_targets:
        source = repo / relative
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    try:
        copy_new_files(repo, payload)
        patch_config(repo / "cir/config.py")
        patch_schemas(repo / "cir/schemas.py")
        patch_engine(repo / "cir/engine.py")
        patch_visualize(repo / "visualize.py")
        patch_index(repo / "web/templates/index.html")
        patch_app_js(repo / "web/static/app.js")
    except Exception:
        for relative in backup_targets:
            source = backup / relative
            if source.exists():
                destination = repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        shutil.rmtree(repo / "cir/vlm", ignore_errors=True)
        print("Patch failed; original tracked files were restored.", file=sys.stderr)
        raise

    print("Multi-provider VLM patch applied.")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
