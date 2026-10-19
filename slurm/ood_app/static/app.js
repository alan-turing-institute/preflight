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
    for (const part of document.querySelectorAll("[data-live]")) {
      const next = fresh.querySelector('[data-live="' + part.dataset.live + '"]');
      if (next) part.replaceChildren(...next.childNodes);
    }
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
  const reply = await fetch(ROOT + "api/job/" + job + "/" + action, {
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

document.addEventListener("DOMContentLoaded", () => {
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
