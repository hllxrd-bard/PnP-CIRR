from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


PATCH_MARKER = "PNP_CIRR_SLERP_TRAINING_FREE_V1"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one marker, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_payload(project: Path, payload: Path) -> None:
    for source in sorted(payload.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(payload)
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def patch_schemas(project: Path) -> None:
    path = project / "cir" / "schemas.py"
    text = path.read_text(encoding="utf-8")
    if "composition_mode: Literal[\"directional\", \"slerp\"]" in text:
        return
    old = "    reference: ReferenceInput\n"
    new = (
        old
        + "    # Retrieval method selector. Directional remains the backward-compatible default.\n"
        + "    composition_mode: Literal[\"directional\", \"slerp\"] = \"directional\"\n"
        + "    slerp_alpha: float | None = Field(default=None, ge=0.0, le=1.0)\n"
    )
    write(path, replace_once(text, old, new, label="schemas.py"))


def patch_engine(project: Path) -> None:
    path = project / "cir" / "engine.py"
    text = path.read_text(encoding="utf-8")
    if "from .slerp_method.pipeline import search_slerp" not in text:
        marker = "from .schemas import (\n"
        text = replace_once(
            text,
            marker,
            "from .slerp_method.pipeline import search_slerp\n" + marker,
            label="engine.py import",
        )
    dispatch = (
        "    def search(self, request: CIRRequest) -> CIROutput:\n"
        "        if request.composition_mode == \"slerp\":\n"
        "            return search_slerp(self, request)\n\n"
    )
    if "return search_slerp(self, request)" not in text:
        text = replace_once(
            text,
            "    def search(self, request: CIRRequest) -> CIROutput:\n",
            dispatch,
            label="engine.py dispatcher",
        )
    write(path, text)


def patch_config_py(project: Path) -> None:
    path = project / "cir" / "config.py"
    text = path.read_text(encoding="utf-8")
    if '    "slerp": {' in text:
        return
    section = (
        '    "slerp": {\n'
        '        "default_mode": "directional",\n'
        '        "default_alpha": 0.8,\n'
        '        "candidate_k": 300,\n'
        '        "max_candidate_pool": 500,\n'
        '        "epsilon": 1e-8,\n'
        '    },\n'
    )
    write(
        path,
        replace_once(
            text,
            '    "retrieval": {\n',
            section + '    "retrieval": {\n',
            label="config.py",
        ),
    )


def patch_config_yaml(project: Path) -> None:
    path = project / "config.yaml"
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^slerp:\s*$", text):
        return
    section = (
        "slerp:\n"
        "  # Pure training-free paper mode: one SLERP query, then exact local cosine.\n"
        "  default_mode: directional\n"
        "  default_alpha: 0.8\n"
        "  candidate_k: 300\n"
        "  max_candidate_pool: 500\n"
        "  epsilon: 1.0e-8\n\n"
    )
    write(
        path,
        replace_once(text, "retrieval:\n", section + "retrieval:\n", label="config.yaml"),
    )


def patch_visualize(project: Path) -> None:
    path = project / "visualize.py"
    text = path.read_text(encoding="utf-8")
    if '"default_slerp_alpha"' in text:
        return
    marker = (
        '                "default_edit_strength": '
        'config.get("composition.default_edit_strength", 0.95),\n'
    )
    addition = (
        marker
        + '                "default_composition_mode": '
        'config.get("slerp.default_mode", "directional"),\n'
        + '                "default_slerp_alpha": '
        'config.get("slerp.default_alpha", 0.8),\n'
    )
    write(path, replace_once(text, marker, addition, label="visualize.py"))


def patch_index(project: Path) -> None:
    path = project / "web" / "templates" / "index.html"
    text = path.read_text(encoding="utf-8")

    # Always version the static URLs so the mode selector cannot be hidden by an old browser cache.
    text = re.sub(
        r'href="/static/style\.css(?:\?v=[^"]*)?"',
        'href="/static/style.css?v=slerp-training-free-v1"',
        text,
        count=1,
    )
    text = re.sub(
        r'src="/static/app\.js(?:\?v=[^"]*)?"',
        'src="/static/app.js?v=slerp-training-free-v1"',
        text,
        count=1,
    )

    if 'id="composition-mode"' not in text:
        marker = (
            '      <div class="action-row">\n'
            '        <button id="preview-button" type="button">Xem reference</button>\n'
            '      </div>\n\n'
        )
        block = marker + (
            '      <div class="method-grid">\n'
            '        <label>Composition mode\n'
            '          <select id="composition-mode">\n'
            '            <option value="directional" {% if default_composition_mode == "directional" %}selected{% endif %}>Directional Add/Remove (current)</option>\n'
            '            <option value="slerp" {% if default_composition_mode == "slerp" %}selected{% endif %}>SLERP training-free</option>\n'
            '          </select>\n'
            '          <small>Directional giữ nguyên pipeline hiện tại. SLERP dùng một textual intent và một cosine query.</small>\n'
            '        </label>\n'
            '        <div id="slerp-intent-preview" class="intent-preview" hidden>\n'
            '          <strong>SLERP textual intent</strong>\n'
            '          <code id="slerp-intent-text"></code>\n'
            '        </div>\n'
            '      </div>\n\n'
        )
        text = replace_once(text, marker, block, label="index.html method selector")

    if 'id="directional-strength-group"' not in text:
        text = replace_once(
            text,
            "        <label>Edit strength\n",
            '        <label id="directional-strength-group">Edit strength\n',
            label="index.html directional strength",
        )

    if 'id="slerp-alpha-group"' not in text:
        marker = (
            '        </label>\n'
            '        <label class="checkbox-label">\n'
            '          <input id="deduplicate" type="checkbox" checked>\n'
        )
        block = (
            '        </label>\n'
            '        <label id="slerp-alpha-group" hidden>SLERP text balance α\n'
            '          <input id="slerp-alpha" type="number" min="0" max="1" step="0.05" value="{{ default_slerp_alpha }}">\n'
            '          <small>0 = image-only; 1 = text-only; paper thường dùng 0.8–0.9.</small>\n'
            '        </label>\n'
            '        <label class="checkbox-label">\n'
            '          <input id="deduplicate" type="checkbox" checked>\n'
        )
        text = replace_once(text, marker, block, label="index.html slerp alpha")

    if 'id="advanced-options"' not in text:
        text = replace_once(
            text,
            '      <details class="advanced-options">\n',
            '      <details id="advanced-options" class="advanced-options">\n',
            label="index.html advanced",
        )

    write(path, text)


def patch_app_js(project: Path) -> None:
    path = project / "web" / "static" / "app.js"
    text = path.read_text(encoding="utf-8")

    if "function buildSlerpIntent" not in text:
        marker = "function createPayload() {\n"
        helper = r'''function splitRemoveObjects(value) {
  return value
    .split(/[,;|\n\r]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildSlerpIntent(editText, removeText) {
  const edit = editText.trim();
  const remove = splitRemoveObjects(removeText).join(" and ");
  if (edit && remove) return `${edit} without ${remove}`;
  if (edit) return edit;
  if (remove) return `the same scene without ${remove}`;
  return "";
}

function updateCompositionMode() {
  const mode = $("composition-mode").value;
  const isSlerp = mode === "slerp";
  $("directional-strength-group").hidden = isSlerp;
  $("slerp-alpha-group").hidden = !isSlerp;
  $("advanced-options").hidden = isSlerp;
  $("slerp-intent-preview").hidden = !isSlerp;
  if (isSlerp) {
    $("use-vlm").checked = false;
    $("slerp-intent-text").textContent = buildSlerpIntent(
      $("edit-text").value,
      $("remove-text").value,
    ) || "(nhập Edit/Add hoặc Remove)";
  }
}

'''
        text = replace_once(text, marker, helper + marker, label="app.js helpers")

    old_payload = r'''function createPayload() {
  const editText = $("edit-text").value.trim();
  const removeText = $("remove-text").value.trim();
  if (!editText && !removeText) {
    throw new Error("Hãy nhập ít nhất một ô Edit/Add hoặc Remove.");
  }

  const strengthText = $("edit-strength").value.trim();
  if (!strengthText) throw new Error("Hãy chọn Edit strength.");
  const strength = Number(strengthText);
  if (!Number.isFinite(strength) || strength < -3 || strength > 5) {
    throw new Error("Edit strength phải là số trong khoảng -3 đến 5.");
  }

  return {
    reference: parseReference(),
    edit_text: editText,
    remove_text: removeText,
    top_k: Number($("top-k").value || 60),
    use_vlm: $("use-vlm").checked,
    edit_strength: strength,
    deduplication: { enabled: $("deduplicate").checked },
  };
}
'''
    new_payload = r'''function createPayload() {
  const editText = $("edit-text").value.trim();
  const removeText = $("remove-text").value.trim();
  if (!editText && !removeText) {
    throw new Error("Hãy nhập ít nhất một ô Edit/Add hoặc Remove.");
  }

  const mode = $("composition-mode").value;
  let strength = null;
  let slerpAlpha = null;

  if (mode === "directional") {
    const strengthText = $("edit-strength").value.trim();
    if (!strengthText) throw new Error("Hãy chọn Edit strength.");
    strength = Number(strengthText);
    if (!Number.isFinite(strength) || strength < -3 || strength > 5) {
      throw new Error("Edit strength phải là số trong khoảng -3 đến 5.");
    }
  } else {
    slerpAlpha = Number($("slerp-alpha").value);
    if (!Number.isFinite(slerpAlpha) || slerpAlpha < 0 || slerpAlpha > 1) {
      throw new Error("SLERP alpha phải là số trong khoảng 0 đến 1.");
    }
  }

  return {
    reference: parseReference(),
    composition_mode: mode,
    edit_text: editText,
    remove_text: removeText,
    top_k: Number($("top-k").value || 60),
    use_vlm: mode === "directional" && $("use-vlm").checked,
    edit_strength: strength,
    slerp_alpha: slerpAlpha,
    deduplication: { enabled: $("deduplicate").checked },
  };
}
'''
    if "composition_mode: mode" not in text:
        text = replace_once(text, old_payload, new_payload, label="app.js payload")

    old_query = r'''  $("query-info").textContent = [
    `edit_add=${query.edit_text || "none"}`,
    `remove=${(query.remove_objects || []).join(", ") || "none"}`,
    `expanded_remove=${(query.expanded_remove_objects || []).join(", ") || "none"}`,
    `operation=${query.operation || "edit"}`,
    `strength=${query.selected_strength ?? "n/a"}`,
    `candidate_pool=${query.candidate_pool_size ?? "n/a"}`,
    `used_vlm=${query.used_vlm ?? false}`,
  ].join(" | ");
'''
    new_query = r'''  $("query-info").textContent = [
    `mode=${query.composition_mode || output.request?.composition_mode || "directional"}`,
    query.intent_text ? `intent=${query.intent_text}` : null,
    query.slerp_alpha != null ? `alpha=${query.slerp_alpha}` : null,
    `edit_add=${query.edit_text || "none"}`,
    `remove=${(query.remove_objects || []).join(", ") || "none"}`,
    `expanded_remove=${(query.expanded_remove_objects || []).join(", ") || "none"}`,
    `operation=${query.operation || "edit"}`,
    `strength=${query.selected_strength ?? "n/a"}`,
    `candidate_pool=${query.candidate_pool_size ?? "n/a"}`,
    `used_vlm=${query.used_vlm ?? false}`,
  ].filter(Boolean).join(" | ");
'''
    if "query.intent_text ?" not in text:
        text = replace_once(text, old_query, new_query, label="app.js query info")

    if '$("composition-mode").addEventListener' not in text:
        marker = '$("preview-button").addEventListener("click", previewReference);\n'
        listeners = (
            '$("composition-mode").addEventListener("change", updateCompositionMode);\n'
            '$("edit-text").addEventListener("input", updateCompositionMode);\n'
            '$("remove-text").addEventListener("input", updateCompositionMode);\n'
            + marker
        )
        text = replace_once(text, marker, listeners, label="app.js listeners")
        text += "\nupdateCompositionMode();\n"

    write(path, text)


def patch_style(project: Path) -> None:
    path = project / "web" / "static" / "style.css"
    text = path.read_text(encoding="utf-8")
    if ".method-grid" in text:
        return
    addition = (
        "\n.method-grid { display: grid; grid-template-columns: minmax(260px, 1fr) 2fr; gap: 12px; margin-top: 14px; align-items: end; }\n"
        "select { border-radius: 8px; border: 1px solid #3a465b; background: #0f141d; color: #eef2f7; padding: 10px; }\n"
        ".intent-preview { border: 1px solid #30394a; border-radius: 8px; padding: 10px 12px; background: #0f141d; }\n"
        ".intent-preview code { display: block; margin-top: 6px; color: #dfe6ff; white-space: pre-wrap; overflow-wrap: anywhere; }\n"
        "[hidden] { display: none !important; }\n"
    )
    # Extend the existing responsive rule without relying on exact whitespace.
    text = text.replace(
        ".reference-inputs, .settings-grid, .edit-fields-grid, .reference-panel { grid-template-columns: 1fr; }",
        ".reference-inputs, .settings-grid, .edit-fields-grid, .reference-panel, .method-grid { grid-template-columns: 1fr; }",
    )
    write(path, text.rstrip() + addition)


def patch_readme(project: Path) -> None:
    path = project / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return
    section = f'''\n\n---\n\n## Training-free SLERP mode\n\n<!-- {PATCH_MARKER} -->\n\nSLERP được triển khai tách biệt trong `cir/slerp_method/`. Pipeline directional cũ vẫn là mặc định và không đổi hành vi.\n\nRequest chọn method bằng:\n\n```json\n{{\n  "composition_mode": "slerp",\n  "edit_text": "pond",\n  "remove_text": "lotus flowers",\n  "slerp_alpha": 0.8\n}}\n```\n\nSLERP mode thực hiện:\n\n```text\nreference image embedding\n+ one deterministic textual-intent embedding\n→ spherical linear interpolation\n→ one Milvus cosine search\n→ exact local cosine reranking\n→ deduplication\n→ Top-K\n```\n\nNó không gọi VLM, không dùng directional `add-remove`, không dùng removal penalty, edit gate hoặc TAT/LoRA.\n'''
    write(path, text.rstrip() + section)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--payload-root", required=True)
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    payload = Path(args.payload_root).resolve()

    required = [
        project / "cir" / "engine.py",
        project / "cir" / "schemas.py",
        project / "cir" / "config.py",
        project / "config.yaml",
        project / "visualize.py",
        project / "web" / "templates" / "index.html",
        project / "web" / "static" / "app.js",
        project / "web" / "static" / "style.css",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing project files:\n" + "\n".join(missing))

    copy_payload(project, payload)
    patch_schemas(project)
    patch_engine(project)
    patch_config_py(project)
    patch_config_yaml(project)
    patch_visualize(project)
    patch_index(project)
    patch_app_js(project)
    patch_style(project)
    patch_readme(project)

    print("SLERP integration applied successfully.")


if __name__ == "__main__":
    main()
