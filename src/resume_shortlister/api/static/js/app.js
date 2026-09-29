// Smart Resume Shortlisting: progressive enhancement for the server-rendered pages.
"use strict";

(() => {
  const STAGE_LABELS = {
    queued: "Waiting in queue",
    extracting: "Reading resumes",
    retrieving: "Semantic retrieval",
    matching: "Matching requirements",
    profiles: "Checking public profiles",
    insights: "Writing AI summaries",
    done: "Done",
  };
  // Keep in sync with api/presenters.py::_STAGE_SPAN
  const STAGE_SPAN = {
    queued: [0, 0], extracting: [0, 50], retrieving: [50, 58], matching: [58, 80],
    profiles: [80, 88], insights: [88, 99], done: [100, 100],
  };
  const STAGE_ORDER = Object.keys(STAGE_LABELS);

  const formatSize = (bytes) =>
    bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

  async function errorMessage(response) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
    } catch (_) { /* not JSON */ }
    return `Request failed (${response.status}).`;
  }

  // ------------------------------------------------------------ upload form
  function initUploadForm(form) {
    const input = form.querySelector("[data-file-input]");
    const zone = form.querySelector("[data-dropzone]");
    const list = form.querySelector("[data-file-list]");
    const errorBox = form.querySelector("[data-form-error]");
    const submit = form.querySelector("[data-submit]");
    const maxFiles = Number(form.dataset.maxFiles);
    const maxBytes = Number(form.dataset.maxFileMb) * 1024 * 1024;
    let files = [];

    const showError = (message) => {
      errorBox.textContent = message;
      errorBox.hidden = !message;
    };

    const render = () => {
      list.replaceChildren(
        ...files.map((file, index) => {
          const item = document.createElement("li");
          const name = document.createElement("span");
          name.className = "file-name";
          name.textContent = file.name;
          const size = document.createElement("span");
          size.className = "file-size";
          size.textContent = formatSize(file.size);
          name.append(size);
          const remove = document.createElement("button");
          remove.type = "button";
          remove.setAttribute("aria-label", `Remove ${file.name}`);
          remove.textContent = "×";
          remove.addEventListener("click", () => {
            files.splice(index, 1);
            render();
          });
          item.append(name, remove);
          return item;
        }),
      );
    };

    const addFiles = (incoming) => {
      const known = new Set(files.map((f) => `${f.name}:${f.size}`));
      const rejected = [];
      for (const file of incoming) {
        if (file.size > maxBytes) rejected.push(`${file.name} is larger than ${form.dataset.maxFileMb} MB`);
        else if (!known.has(`${file.name}:${file.size}`)) files.push(file);
      }
      if (files.length > maxFiles) {
        rejected.push(`only the first ${maxFiles} files were kept`);
        files = files.slice(0, maxFiles);
      }
      showError(rejected.length ? `Note: ${rejected.join("; ")}.` : "");
      render();
    };

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      addFiles(input.files);
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.add("dragover");
      }),
    );
    ["dragleave", "drop"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.remove("dragover");
      }),
    );
    zone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const description = form.elements.job_description.value.trim();
      if (!description) return showError("Please paste a job description.");
      if (!files.length) return showError("Please add at least one resume.");

      const data = new FormData();
      data.append("job_description", description);
      files.forEach((file) => data.append("files", file, file.name));
      if (form.elements.shortlist_size.value) data.append("shortlist_size", form.elements.shortlist_size.value);
      data.append("include_social", form.elements.include_social.checked ? "true" : "false");
      const insights = form.elements.include_insights;
      data.append("include_insights", insights && insights.checked ? "true" : "false");

      showError("");
      submit.disabled = true;
      submit.innerHTML = '<span class="spinner" aria-hidden="true"></span> Uploading…';
      try {
        const response = await fetch("/api/v1/shortlists", { method: "POST", body: data });
        if (!response.ok) throw new Error(await errorMessage(response));
        const created = await response.json();
        window.location.assign(created.ui_url);
      } catch (error) {
        showError(error.message || "Upload failed.");
        submit.disabled = false;
        submit.textContent = "Rank resumes";
      }
    });
  }

  // ------------------------------------------------------------ run polling
  function initRunPolling(card) {
    const runId = card.dataset.pollRun;
    const fill = card.querySelector(".progress-fill");
    const bar = card.querySelector("[data-progress-bar]");
    const text = card.querySelector("[data-progress-text]");
    const stageItems = card.querySelectorAll("[data-stage]");

    const update = (stage, done, total) => {
      const [start, end] = STAGE_SPAN[stage] || [0, 0];
      const fraction = total ? Math.min(1, done / total) : 0;
      const percent = Math.round(start + (end - start) * fraction);
      fill.style.width = `${percent}%`;
      bar.setAttribute("aria-valuenow", String(percent));
      const current = STAGE_ORDER.indexOf(stage);
      stageItems.forEach((item) => {
        const index = STAGE_ORDER.indexOf(item.dataset.stage);
        item.classList.toggle("done", index < current);
        item.classList.toggle("current", index === current);
      });
      const counter = total ? ` (${done}/${total})` : "";
      text.textContent = `${STAGE_LABELS[stage] || stage}${counter}…`;
    };

    update(text.dataset.initialStage, Number(text.dataset.initialDone), Number(text.dataset.initialTotal));

    const poll = async () => {
      try {
        const response = await fetch(`/api/v1/shortlists/${encodeURIComponent(runId)}`, { cache: "no-store" });
        if (response.status === 404) return window.location.assign("/shortlists");
        if (response.ok) {
          const run = await response.json();
          if (run.status === "completed" || run.status === "failed") return window.location.reload();
          update(run.progress.stage, run.progress.done, run.progress.total);
        }
      } catch (_) { /* transient network error: keep polling */ }
      window.setTimeout(poll, 1500);
    };
    window.setTimeout(poll, 800);
  }

  // ------------------------------------------------------------ delete run
  function initDelete(button) {
    button.addEventListener("click", async () => {
      if (!window.confirm("Delete this shortlist? This cannot be undone.")) return;
      button.disabled = true;
      const response = await fetch(`/api/v1/shortlists/${encodeURIComponent(button.dataset.deleteRun)}`, {
        method: "DELETE",
      });
      if (response.ok || response.status === 404) window.location.assign("/shortlists");
      else {
        button.disabled = false;
        window.alert(await errorMessage(response));
      }
    });
  }

  // ------------------------------------------------------------ OCR status chip
  async function initSystemStatus(chip) {
    const dot = chip.querySelector("[data-status-dot]");
    const text = chip.querySelector("[data-status-text]");
    try {
      const response = await fetch("/api/v1/status", { cache: "no-store" });
      const status = await response.json();
      const engines = (status.ocr.engines || []).filter((e) => e.available);
      if (engines.some((e) => e.name.startsWith("chandra"))) {
        dot.className = "dot ok";
        text.textContent = "Scans, photos and handwriting supported in 90+ languages";
      } else if (engines.length) {
        dot.className = "dot warn";
        text.textContent = "Scans supported (basic quality)";
      } else if (status.ocr.mode === "none") {
        dot.className = "dot warn";
        text.textContent = "Only digital PDFs, Word and text files can be read";
      } else {
        dot.className = "dot bad";
        text.textContent = "Scanned resumes can't be read right now";
      }
    } catch (_) {
      text.textContent = "Could not check scan support";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("[data-upload-form]");
    if (form) initUploadForm(form);
    const card = document.querySelector("[data-poll-run]");
    if (card) initRunPolling(card);
    document.querySelectorAll("[data-delete-run]").forEach(initDelete);
    const chip = document.querySelector("[data-system-status]");
    if (chip) initSystemStatus(chip);
  });
})();
