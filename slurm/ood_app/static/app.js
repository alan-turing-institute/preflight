"use strict";
// The server renders every page (all values escaped there); this script only refreshes the live parts of the queue
// page by fetching the page again, shows chart tooltips, runs the job actions, and loads the log (textContent only).

const ROOT = document.body.dataset.root || "./";
const REFRESH_MS = 10000;

async function refreshLive() {
  const note = document.getElementById("live-note");
  try {
    const reply = await fetch(location.href, { headers: { Accept: "text/html" }, cache: "no-store" });
    if (!reply.ok) throw new Error(String(reply.status));
    const fresh = new DOMParser().parseFromString(await reply.text(), "text/html");
    // new code was installed: only the live parts are swapped below, so the page's own stylesheet would stay the old one
    const sheet = 'link[rel="stylesheet"]';
    if (fresh.querySelector(sheet)?.getAttribute("href") !== document.querySelector(sheet)?.getAttribute("href")) {
      location.reload();
      return;
    }
    // a list the viewer opened (details with data-key) stays open across the swap
    const open = new Set([...document.querySelectorAll("details[data-key][open]")].map((d) => d.dataset.key));
    for (const part of document.querySelectorAll("[data-live]")) {
      const next = fresh.querySelector('[data-live="' + part.dataset.live + '"]');
      if (next) part.replaceChildren(...next.childNodes);
    }
    for (const d of document.querySelectorAll("details[data-key]")) if (open.has(d.dataset.key)) d.open = true;
    if (note) note.textContent = "Updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) + " · refreshes every 10 s";
  } catch (_) {
    if (note) note.textContent = "Could not refresh; retrying in 10 s";
  }
}

function tooltips() {
  let tip = null;
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest && event.target.closest("[data-tip]");
    if (!target) return;
    const [project, when, value] = target.dataset.tip.split("|");
    tip = tip || Object.assign(document.createElement("div"), { className: "tip", role: "tooltip" });
    const head = document.createElement("b");
    head.textContent = project;
    tip.replaceChildren(head, document.createTextNode(when + ": " + value));
    document.body.append(tip);
  });
  document.addEventListener("mousemove", (event) => {
    if (!tip || !tip.isConnected) return;
    tip.style.left = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = event.clientY + 14 + "px";
  });
  document.addEventListener("mouseout", (event) => {
    if (tip && event.target.closest && event.target.closest("[data-tip]")) tip.remove();
  });
}

async function loadLog(form) {
  const params = new URLSearchParams();
  const text = form.querySelector("[name=text]").value;
  if (text) params.set("text", text);
  if (form.querySelector("[name=errors]").checked) params.set("errors", "1");
  const out = document.getElementById("log-output");
  try {
    const reply = await fetch(ROOT + "api/job/" + form.dataset.job + "/log?" + params, { headers: { Accept: "application/json" } });
    const body = reply.ok ? (await reply.json()).log : "Could not load the log (HTTP " + reply.status + ").";
    out.textContent = body || "The log is empty" + (text || params.has("errors") ? " for this filter." : " so far.");
    out.scrollTop = out.scrollHeight;
  } catch (_) {
    out.textContent = "Could not load the log.";
  }
}

async function post(job, action, payload) {
  return postJson(ROOT + "api/job/" + job + "/" + action, payload);
}

async function postJson(url, payload) {
  const reply = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  try {
    return await reply.json();
  } catch (_) {
    return { ok: false, output: "unexpected reply (HTTP " + reply.status + ")" };
  }
}

function confirmInPage(title, detail, needsLoseProgress) {
  const dialog = document.getElementById("confirm");
  document.getElementById("confirm-title").textContent = title;
  document.getElementById("confirm-detail").textContent = detail;
  const box = document.getElementById("confirm-lose");
  const label = document.getElementById("confirm-lose-label");
  box.checked = false;
  label.hidden = !needsLoseProgress;
  const accept = document.getElementById("confirm-yes");
  accept.disabled = needsLoseProgress;
  box.onchange = () => { accept.disabled = needsLoseProgress && !box.checked; };
  return new Promise((resolve) => {
    dialog.onclose = () => resolve(dialog.returnValue === "yes" ? { lose: box.checked } : null);
    dialog.returnValue = "";
    dialog.showModal();
    document.getElementById("confirm-no").focus();
  });
}

async function act(button) {
  const job = button.dataset.job;
  const action = button.dataset.action;
  const result = document.getElementById("action-result");
  let payload = {};
  if (action === "pause") {
    const preview = await (await fetch(ROOT + "api/job/" + job + "/pause-preview")).json();
    const answer = await confirmInPage("Pause job " + job + "?", preview.output, button.dataset.steady === "1");
    if (!answer) return;
    payload = { lose_progress: answer.lose };
  } else if (action === "kill") {
    if (!(await confirmInPage("Cancel job " + job + "?", "Slurm stops it now. A cancelled job cannot be resumed.", false))) return;
  }
  for (const b of document.querySelectorAll("button[data-action]")) b.disabled = true;
  const done = await post(job, action, payload);
  result.className = "notice " + (done.ok ? "ok" : "err");
  result.textContent = (done.ok ? "Done: " : "Failed: ") + (done.output || done.error || action);
  if (done.ok) setTimeout(() => location.reload(), 1500);
  else for (const b of document.querySelectorAll("button[data-action]")) b.disabled = false;
}

// The admin card for part-time machines: each control leaves one request for the agent, which applies it.
async function partTime(node, action, payload) {
  const result = document.getElementById("part-time-result");
  const done = await postJson(ROOT + "api/part-time/" + encodeURIComponent(node) + "/" + action, payload);
  result.className = "notice " + (done.ok ? "ok" : "err");
  result.textContent = (done.ok ? "Done: " : "Failed: ") + (done.output || done.error || action);
  if (done.ok) setTimeout(() => location.reload(), 3000);
}

function partTimeForm(form) {
  const payload = {};
  for (const input of form.querySelectorAll("input[name]")) {
    if (input.type === "number") {
      if (input.value !== "") payload[input.name] = Number(input.value);
      else if (input.name === "jobs_memory_gb") payload[input.name] = null;
    } else {
      payload[input.name] = input.value || null;
    }
  }
  partTime(form.dataset.node, form.dataset.partTime, payload);
}

function partTimeControls() {
  for (const form of document.querySelectorAll("form[data-part-time]")) {
    form.addEventListener("submit", (event) => { event.preventDefault(); partTimeForm(form); });
  }
  for (const box of document.querySelectorAll("input[data-part-time=switch]")) {
    box.addEventListener("change", () => partTime(box.dataset.node, box.checked ? "accept" : "close", {}));
  }
  for (const button of document.querySelectorAll("button[data-part-time=give_back]")) {
    button.addEventListener("click", async () => {
      const jobs = Number(button.dataset.jobs);
      const detail = "This closes " + button.dataset.node + " and sends back " + jobs + " running job" + (jobs === 1 ? "" : "s")
        + " to the queue; each starts again from its checkpoint elsewhere.";
      if (await confirmInPage("Give " + button.dataset.node + " back?", detail, false)) partTime(button.dataset.node, "give_back", {});
    });
  }
}

// The admins' maintenance banner (D43): each button leaves one request for the site's restart slot, which decides.
async function maintenance(action) {
  const result = document.getElementById("maintenance-result");
  const done = await postJson(ROOT + "api/maintenance/" + action, {});
  result.className = "notice " + (done.ok ? "ok" : "err");
  result.textContent = (done.ok ? "" : "Failed: ") + (done.output || done.error || action);
}

function maintenanceControls() {
  for (const button of document.querySelectorAll("button[data-maintenance]")) {
    button.addEventListener("click", async () => {
      const action = button.dataset.maintenance;
      if (action === "restart_now" && !(await confirmInPage("Restart the machine now?", button.dataset.detail, false))) return;
      maintenance(action);
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  partTimeControls();
  maintenanceControls();
  if (document.querySelector("[data-live]")) setInterval(refreshLive, REFRESH_MS);
  if (document.querySelector("[data-tip]")) tooltips();
  const log = document.getElementById("log-viewer");
  if (log) {
    log.addEventListener("submit", (event) => { event.preventDefault(); loadLog(log); });
    loadLog(log);
  }
  for (const button of document.querySelectorAll("button[data-action]")) {
    button.addEventListener("click", () => act(button));
  }
});
