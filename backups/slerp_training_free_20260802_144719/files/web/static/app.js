const state = {
  output: null,
  visibleCount: 0,
  pageSize: Number(document.body.dataset.pageSize || 30),
};

const $ = (id) => document.getElementById(id);
const statusBox = $("status");
const gallery = $("gallery");
const loadMoreButton = $("load-more");

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function parseReference() {
  const idValue = $("reference-id").value.trim();
  const videoName = $("video-name").value.trim();
  const frameName = $("frame-name").value.trim();
  const path = $("reference-path").value.trim();
  const modes = [Boolean(idValue), Boolean(videoName && frameName), Boolean(path)].filter(Boolean).length;
  if (modes !== 1) {
    throw new Error("Hãy nhập đúng một kiểu reference: ID; video + frame; hoặc local path.");
  }
  if (idValue) return { id: idValue };
  if (path) return { path };
  return { video_name: videoName, frame_name: frameName };
}

function renderReference(reference) {
  if (!reference) return;
  $("reference-meta").textContent = [
    `id=${reference.id ?? "n/a"}`,
    `video=${reference.video_name ?? "n/a"}`,
    `frame=${reference.frame_name ?? "n/a"}`,
    `timestamp=${reference.timestamp ?? "n/a"}`,
    reference.image_path ? `path=${reference.image_path}` : "",
  ].filter(Boolean).join(" | ");
  const image = $("reference-image");
  if (reference.image_url) {
    image.src = reference.image_url;
    image.style.display = "block";
  } else {
    image.removeAttribute("src");
    image.style.display = "none";
  }
}

async function previewReference() {
  try {
    setStatus("Đang lấy reference...");
    const response = await fetch("/api/reference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parseReference()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Không thể lấy reference.");
    renderReference(data);
    setStatus("Đã tải reference.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function createPayload() {
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

function lazyLoadImages() {
  const observer = new IntersectionObserver((entries, currentObserver) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const image = entry.target;
      image.src = image.dataset.src;
      image.removeAttribute("data-src");
      currentObserver.unobserve(image);
    });
  }, { rootMargin: "300px" });
  document.querySelectorAll("img.lazy[data-src]").forEach((image) => observer.observe(image));
}

function renderMore() {
  if (!state.output) return;
  const results = state.output.results || [];
  const next = Math.min(results.length, state.visibleCount + state.pageSize);
  const template = $("card-template");
  for (let index = state.visibleCount; index < next; index += 1) {
    const item = results[index];
    const node = template.content.cloneNode(true);
    node.querySelector(".rank-badge").textContent = `#${item.rank}`;
    const image = node.querySelector(".result-image");
    image.dataset.src = item.image_url || "";
    node.querySelector(".result-title").textContent = `${item.video_name || "unknown"} / ${item.frame_name || item.id}`;
    node.querySelector(".result-meta").textContent = `id=${item.id} | t=${item.timestamp ?? "n/a"} | cluster=${item.cluster_id ?? "n/a"}`;
    const composedQuery = item.best_composed_query || item.matched_query || "";
    node.querySelector(".score").textContent = `score=${Number(item.score).toFixed(5)} | ${composedQuery}`;
    node.querySelector(".score-details").textContent = JSON.stringify({
      normalized: item.scores,
      raw: item.raw_scores,
      best_ann_query: item.best_ann_query,
      retrieved_by: item.retrieved_by,
    }, null, 2);
    node.querySelector(".reuse-button").addEventListener("click", () => {
      $("reference-id").value = item.id;
      $("video-name").value = "";
      $("frame-name").value = "";
      $("reference-path").value = "";
      renderReference(item);
      window.scrollTo({ top: 0, behavior: "smooth" });
      setStatus(`Đã chọn #${item.rank} làm reference mới.`);
    });
    gallery.appendChild(node);
  }
  state.visibleCount = next;
  loadMoreButton.hidden = next >= results.length;
  lazyLoadImages();
}

function renderOutput(output) {
  state.output = output;
  state.visibleCount = 0;
  gallery.innerHTML = "";
  renderReference(output.reference);
  $("timings").textContent = Object.entries(output.timings_ms || {})
    .map(([key, value]) => `${key}: ${Number(value).toFixed(1)} ms`)
    .join(" | ");
  $("warnings").textContent = (output.warnings || []).join(" | ");
  const query = output.query || {};
  $("query-info").textContent = [
    `edit_add=${query.edit_text || "none"}`,
    `remove=${(query.remove_objects || []).join(", ") || "none"}`,
    `expanded_remove=${(query.expanded_remove_objects || []).join(", ") || "none"}`,
    `operation=${query.operation || "edit"}`,
    `strength=${query.selected_strength ?? "n/a"}`,
    `candidate_pool=${query.candidate_pool_size ?? "n/a"}`,
    `used_vlm=${query.used_vlm ?? false}`,
  ].join(" | ");
  $("download-json").disabled = false;
  renderMore();
}

async function runSearch() {
  const button = $("search-button");
  try {
    button.disabled = true;
    setStatus("Đang chạy CIR...");
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(createPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "CIR request failed.");
    renderOutput(data);
    setStatus(`Hoàn tất: ${data.results.length} kết quả, total=${Number(data.timings_ms.total).toFixed(1)} ms.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

function downloadJson() {
  if (!state.output) return;
  const blob = new Blob([JSON.stringify(state.output, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cir_output.json";
  link.click();
  URL.revokeObjectURL(url);
}

$("preview-button").addEventListener("click", previewReference);
$("search-button").addEventListener("click", runSearch);
$("download-json").addEventListener("click", downloadJson);
loadMoreButton.addEventListener("click", renderMore);
