"use strict";
// Every value from the server goes in through textContent; every URL is relative (OOD serves this app under /pun/sys/<name>/).

const JOB_COLUMNS = ["id", "user", "project", "class", "name", "description", "state", "waiting", "elapsed", "gpus", "mem"];

function renderJobs(jobs) {
  const body = document.querySelector("#jobs tbody");
  if (!body) return;
  body.replaceChildren();
  for (const job of jobs) {
    const row = document.createElement("tr");
    for (const column of JOB_COLUMNS) {
      const cell = document.createElement("td");
      if (column === "id") {
        const link = document.createElement("a");
        link.href = "job/" + encodeURIComponent(job.id);
        link.textContent = job.id;
        cell.append(link);
      } else {
        cell.textContent = String(job[column] ?? "");
      }
      row.append(cell);
    }
    body.append(row);
  }
  const status = document.getElementById("jobs-status");
  if (status) status.textContent = jobs.length + " job(s), updated " + new Date().toLocaleTimeString();
}

async function pollJobs() {
  try {
    const reply = await fetch("api/jobs", { headers: { Accept: "application/json" } });
    if (reply.ok) renderJobs((await reply.json()).jobs);
  } catch (_) {
    const status = document.getElementById("jobs-status");
    if (status) status.textContent = "could not refresh; retrying";
  }
}

async function loadLog(root) {
  const params = new URLSearchParams();
  const text = root.querySelector("[name=text]").value;
  if (text) params.set("text", text);
  if (root.querySelector("[name=errors]").checked) params.set("errors", "1");
  const reply = await fetch("../api/job/" + root.dataset.job + "/log?" + params, { headers: { Accept: "application/json" } });
  const body = reply.ok ? (await reply.json()).log : "could not load the log (" + reply.status + ")";
  document.getElementById("log-output").textContent = body;
}

async function post(job, action, payload) {
  const reply = await fetch("../api/job/" + job + "/" + action, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  try {
    return await reply.json();
  } catch (_) {
    return { ok: false, output: "unexpected reply (" + reply.status + ")" };
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
    const preview = await (await fetch("../api/job/" + job + "/pause-preview")).json();
    const answer = await confirmInPage("Pause job " + job + "?", preview.output, button.dataset.steady === "1");
    if (!answer) return;
    payload = { lose_progress: answer.lose };
  } else if (action === "kill") {
    if (!(await confirmInPage("Kill job " + job + "?", "It is cancelled and cannot be resumed.", false))) return;
  }
  const done = await post(job, action, payload);
  result.textContent = (done.ok ? action + " done: " : action + " failed: ") + (done.output || done.error || "");
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("jobs")) {
    pollJobs();
    setInterval(pollJobs, 10000);
  }
  const log = document.getElementById("log-viewer");
  if (log) {
    log.addEventListener("submit", (event) => { event.preventDefault(); loadLog(log); });
    loadLog(log);
  }
  for (const button of document.querySelectorAll("button[data-action]")) {
    button.addEventListener("click", () => act(button));
  }
});
