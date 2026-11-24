// 안태현 PR #8 대시보드(MIT)를 바탕으로 이 엔진에 맞게 다시 작성 (라이선스: web/fl_server/LICENSE-dashboard)
// 조회형(결과 폴더 보기)과 실행형(/api/start·status·stop) 화면 제어. DOM은 createElement·textContent로만 만든다.
"use strict";

const POLL_MS = 1000;
const POLL_RETRY_MS = 3000;   // 상태 조회가 실패했을 때 다시 시도하는 간격
const STATUS_TEXT = { idle: "● 실행 대기", running: "◉ 실행 중", completed: "✓ 완료", stopped: "■ 중지됨", error: "▲ 오류" };
const METRIC_COLUMNS = [
  { metric: "attack_damage", defense: "", label: "공격 피해" },
  { metric: "defense_gain", defense: "clip_fedavg", label: "방어 개선 (클리핑)" },
  { metric: "defense_gain", defense: "median", label: "방어 개선 (중앙값)" },
  { metric: "normal_loss", defense: "clip_fedavg", label: "정상 손실 (클리핑)" },
  { metric: "normal_loss", defense: "median", label: "정상 손실 (중앙값)" },
];
const MSE_LOG_MIN = -6;   // σ별 MSE 막대: 10^-6 ~ 10^0을 로그 눈금으로

let allRuns = [];
let selectedRunId = null;
let pollTimer = null;      // 다음 상태 조회 예약 (setTimeout). 조회는 하나씩만 돈다
let pollSeq = 0;           // 상태 조회 번호: 나중에 보낸 조회가 있으면 늦게 온 이전 응답은 버린다
let openSeq = 0;           // 결과 상세 조회 번호: 빠르게 다른 결과를 고르면 이전 응답은 버린다
let currentMode = "browse";
let liveRunId = null;      // 시작했거나 실행 중으로 본 실행 중 아직 끝 처리를 안 한 것
let statusReadFailed = false;
let dlgRuns = [];

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  renderRun({ id: "-", summary: null, plan: null, replay: null, rounds: [], clients: [], class_acc: [], client_acc: [], exclusions: [] });
  setText("view-title", "왼쪽에서 결과 폴더를 고르거나 시연 실행을 시작하세요.");
  loadRuns();
  loadDlg();
  loadComparison();
  pollStatus();
});

function bindEvents() {
  document.querySelectorAll(".mode-btn").forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    byId(btn.dataset.tab).classList.add("active");
  }));
  byId("group-select").addEventListener("change", renderRunList);
  byId("btn-refresh").addEventListener("click", loadRuns);
  byId("btn-start").addEventListener("click", startLive);
  byId("btn-stop").addEventListener("click", stopLive);
  byId("dlg-select").addEventListener("change", () => renderDlgRun(byId("dlg-select").value));
}

function setMode(mode) {
  currentMode = mode;
  if (mode === "live") pollStatus();   // 실행형으로 오면 진행 중인 실행을 바로 그린다
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  byId("panel-browse").classList.toggle("hidden", mode !== "browse");
  byId("panel-live").classList.toggle("hidden", mode !== "live");
}

function showError(message) {
  const banner = byId("error-banner");
  banner.textContent = message;
  banner.classList.toggle("hidden", !message);
}

// ---- 조회형: 결과 폴더 목록 ----
async function loadRuns() {
  try {
    allRuns = await api("/api/runs");
  } catch (err) {
    clear(byId("run-list")).appendChild(el("div", { className: "empty-state", text: `목록을 읽지 못했습니다: ${err.message}` }));
    return;
  }
  const select = byId("group-select");
  const current = select.value;
  clear(select).appendChild(el("option", { text: `전체 (${allRuns.length})`, attrs: { value: "" } }));
  [...new Set(allRuns.map((r) => r.group))].sort().forEach((group) => {
    const count = allRuns.filter((r) => r.group === group).length;
    select.appendChild(el("option", { text: `${group || "(최상위)"} (${count})`, attrs: { value: group } }));
  });
  select.value = [...select.options].some((o) => o.value === current) ? current : "";
  renderRunList();
  renderReplayList();
}

function runMeta(run) {
  const attack = run.attack === true ? "공격" : run.attack === false ? "정상" : null;
  const acc = isNum(run.final_test_acc) ? `테스트 ${pct(run.final_test_acc)}` : isNum(run.final_val_acc) ? `검증 ${pct(run.final_val_acc)}` : null;
  const diverged = isNum(run.diverged_round) ? `R${run.diverged_round} 발산` : null;
  return [run.env, attack, run.aggregator, isNum(run.seed) ? `s${run.seed}` : null, acc, diverged, run.replay ? "재실행" : null]
    .filter(Boolean).join(" · ");
}

function renderRunList() {
  const group = byId("group-select").value;
  const list = clear(byId("run-list"));
  const runs = allRuns.filter((r) => !group || r.group === group);
  if (!runs.length) {
    list.appendChild(el("div", { className: "empty-state", text: "summary.json이 있는 결과 폴더가 없습니다." }));
    return;
  }
  runs.forEach((run) => {
    const item = el("button", { className: `run-item${run.id === selectedRunId ? " active" : ""}`, attrs: { type: "button" } }, [
      el("div", { className: "run-item-name", text: run.id }),
      el("div", { className: "run-item-meta", text: runMeta(run) || "-" }),
    ]);
    item.addEventListener("click", () => openRun(run.id));
    list.appendChild(item);
  });
}

async function openRun(runId) {
  selectedRunId = runId;
  const seq = ++openSeq;
  renderRunList();
  try {
    const detail = await api(`/api/runs/${encodePath(runId)}`);
    if (seq !== openSeq || selectedRunId !== runId) return;
    renderRun(detail, { live: runId.startsWith("live/") });
    document.querySelector('.tab-btn[data-tab="tab-trends"]').click();
  } catch (err) {
    if (seq !== openSeq || selectedRunId !== runId) return;
    setText("view-title", `결과를 읽지 못했습니다: ${err.message}`);
  }
}

function renderReplayList() {
  const rows = allRuns.filter((r) => r.replay).map((run) => {
    const row = cellsRow([run.id, run.replay.source ?? "-", run.replay.from_round ?? "-",
      JSON.stringify(run.replay.change ?? {}), pct(run.final_val_acc), pct(run.final_test_acc)]);
    row.addEventListener("click", () => {
      setMode("browse");
      openRun(run.id);
    });
    return row;
  });
  fillRows("replay-tbody", rows, "replay.json이 있는 결과 폴더가 아직 없습니다 (experiments/replay/).", 6);
}

// ---- DLG 탭 ----
async function loadDlg() {
  try {
    dlgRuns = await api("/api/dlg");
  } catch (err) {
    fillRows("dlg-tbody", [], `DLG 결과를 읽지 못했습니다: ${err.message}`, 8);
    return;
  }
  const select = clear(byId("dlg-select"));
  dlgRuns.forEach((run) => select.appendChild(el("option", { text: run.id, attrs: { value: run.id } })));
  if (dlgRuns.length) renderDlgRun(dlgRuns[0].id);
  else fillRows("dlg-tbody", [], "experiments/dlg/에 결과가 없습니다.", 8);
}

function dlgImage(runId, name, alt, className = "dlg-thumb") {
  if (!name) return el("span", { className: "helper-text", text: "없음" });
  return el("img", { className, attrs: { src: `/api/dlg/${encodeURIComponent(runId)}/images/${encodeURIComponent(name)}`, alt } });
}

function renderDlgRun(runId) {
  const run = dlgRuns.find((r) => r.id === runId);
  if (!run) return;
  const rows = run.results.map((r) => cellsRow([
    `#${r.image_index}`, r.class ?? r.label, r.sigma, r.iterations, isNum(r.mse) ? r.mse.toExponential(2) : "-",
    r.label_correct ? tag("성공", "bad") : tag("실패", "ok"),
    dlgImage(run.id, r.original, `원본 ${r.image_index}`), dlgImage(run.id, r.reconstructed, `복원 ${r.image_index} σ=${r.sigma}`),
  ]));
  fillRows("dlg-tbody", rows, "results.csv가 비었습니다.", 8);

  const sigmas = [...new Set(run.results.map((r) => r.sigma))].filter(isNum).sort((a, b) => a - b);
  const items = sigmas.map((sigma) => {
    const m = mean(run.results.filter((r) => r.sigma === sigma).map((r) => r.mse));
    const scaled = isNum(m) && m > 0 ? (Math.log10(m) - MSE_LOG_MIN) / -MSE_LOG_MIN : 0;
    return { label: `σ = ${sigma}`, value: scaled, text: isNum(m) ? m.toExponential(2) : "-" };
  });
  const bars = clear(byId("dlg-mse-bars"));
  bars.appendChild(barRows("막대는 로그 눈금 (10⁻⁶ ~ 1), 값은 이미지 평균 MSE", items));

  renderDlgGallery(run);
}

// 이미지마다 한 줄: 원본 | σ별 복원 (낱장 이미지 + 아래 설명)
function dlgCard(runId, name, alt, captions) {
  return el("figure", { className: "dlg-card" }, [
    dlgImage(runId, name, alt, "dlg-card-img"),
    el("figcaption", {}, captions.map((text) => el("div", { text }))),
  ]);
}

function renderDlgGallery(run) {
  const gallery = clear(byId("dlg-summary"));
  const indices = [...new Set(run.results.map((r) => r.image_index))];
  if (!indices.length) return;
  const manyIterations = new Set(run.results.map((r) => r.iterations)).size > 1;
  gallery.appendChild(el("div", { className: "section-subtitle", text: "원본과 복원 이미지" }));
  indices.forEach((index) => {
    const rows = run.results.filter((r) => r.image_index === index);
    const first = rows[0];
    const recon = rows.map((r) => dlgCard(run.id, r.reconstructed, `복원 ${index} σ=${r.sigma}`, [
      `σ = ${r.sigma}`,
      ...(manyIterations ? [`반복 ${r.iterations}`] : []),
      isNum(r.mse) ? `MSE ${r.mse.toPrecision(3)}` : "MSE -",
    ]));
    gallery.appendChild(el("div", { className: "dlg-gallery-row" }, [
      dlgCard(run.id, first.original, `원본 ${index}`, ["original", `#${index} ${first.class ?? first.label}`]),
      el("div", { className: "dlg-gallery-recon" }, recon),
    ]));
  });
}

// ---- 비교표 탭 ----
function seedText(values, diverged) {
  const parts = values.map((v, i) => (isNum(v) ? v : `R${diverged[i] ?? "?"}`));
  return parts.join(" / ");
}

function metricCell(group) {
  if (!group) return "-";
  const values = group.seeds.map((s) => s.value_pp);
  if (values.every(isNum)) return `${values.map(signedPp).join(" / ")} (${signedPp(group.mean_value_pp)})`;
  const diverged = group.seeds.map((s) => s.diverged_rounds);
  const ref = group.seeds.map((s) => s.reference_pp);
  const refText = ref.every(isNum) ? ` · 참고 ${ref.map(signedPp).join(" / ")} (${signedPp(group.mean_reference_pp)})` : "";
  return `발산 (${seedText(values.map(() => null), diverged)})${refText}`;
}

function accuracyCell(cell) {
  if (!cell) return "-";
  const seeds = cell.seeds.slice().sort((a, b) => a.seed - b.seed);
  if (seeds.some((s) => isNum(s.diverged_round))) {
    return `발산 (${seeds.map((s) => (isNum(s.diverged_round) ? `R${s.diverged_round}` : pct(s.acc))).join(" / ")})`;
  }
  const accs = seeds.map((s) => s.acc);
  const m = mean(accs);
  return `${accs.map((a) => (isNum(a) ? (a * 100).toFixed(2) : "-")).join(" / ")} (${isNum(m) ? (m * 100).toFixed(2) : "-"})`;
}

function headRow(theadId, labels) {
  clear(byId(theadId)).appendChild(el("tr", {}, labels.map((label) => el("th", { text: label }))));
}

async function loadComparison() {
  let data;
  try {
    data = await api("/api/comparison");
  } catch (err) {
    byId("compare-empty").textContent = `비교표를 읽지 못했습니다: ${err.message}`;
    byId("compare-empty").classList.remove("hidden");
    return;
  }
  byId("compare-empty").classList.toggle("hidden", data.available);
  const envs = data.env_order.filter((env) => data.metrics.some((g) => g.env === env) || data.accuracy.some((c) => c.env === env));

  headRow("metric-thead", ["환경", ...METRIC_COLUMNS.map((c) => c.label)]);
  const metricRows = envs.map((env) => cellsRow([env, ...METRIC_COLUMNS.map((col) => metricCell(
    data.metrics.find((g) => g.env === env && g.metric === col.metric && g.defense === col.defense)))]));
  fillRows("metric-tbody", metricRows, "comparison.csv가 없습니다.", 6);

  const conditionLabel = (c) => `${data.names.attack[c.attack] ?? c.attack} ${data.names.aggregator[c.aggregator] ?? c.aggregator}`;
  headRow("acc-thead", ["환경", ...data.conditions.map(conditionLabel)]);
  const accRows = envs.map((env) => cellsRow([env, ...data.conditions.map((c) => accuracyCell(
    data.accuracy.find((cell) => cell.env === env && cell.attack === c.attack && cell.aggregator === c.aggregator)))]));
  fillRows("acc-tbody", accRows, "experiments/matrix/runs.csv가 없습니다.", 7);

  const chart = clear(byId("compare-chart"));
  if (data.has_chart) {
    chart.appendChild(el("div", { className: "section-subtitle", text: "환경별 최종 테스트 정확도 (accuracy.svg)" }));
    chart.appendChild(el("img", { className: "dlg-summary-img", attrs: { src: "/api/comparison/accuracy.svg", alt: "환경별 최종 테스트 정확도 막대그래프" } }));
  }
}

// ---- 실행형 ----
function readInt(id) {
  const value = Number(byId(id).value);
  return Number.isInteger(value) ? value : NaN;
}

async function startLive() {
  showError("");
  const body = {
    env: document.querySelector('input[name="env"]:checked').value,
    attack: document.querySelector('input[name="attack"]:checked').value,
    aggregator: byId("aggregator-select").value,
    seed: readInt("seed-input"),
    rounds: readInt("rounds-input"),
  };
  if (!Number.isInteger(body.seed) || !Number.isInteger(body.rounds)) {
    showError("Seed와 Rounds는 정수여야 합니다.");
    return;
  }
  byId("btn-start").disabled = true;
  try {
    const started = await api("/api/start", body);
    selectedRunId = started.run_id;
    liveRunId = started.run_id;   // 첫 응답이 이미 완료여도 끝 처리를 하도록
    document.querySelector('.tab-btn[data-tab="tab-trends"]').click();
    pollStatus();
  } catch (err) {
    showError(err.message);
    byId("btn-start").disabled = false;
  }
}

async function stopLive() {
  try {
    await api("/api/stop", {});
  } catch (err) {
    showError(err.message);
  }
  pollStatus();
}

function renderStatusPill(status) {
  const pill = byId("status-pill");
  pill.className = `status-pill status-${status.state}`;
  const progress = status.run_id ? ` (Round ${status.rounds_done} / ${status.rounds_total})` : "";
  pill.textContent = `${STATUS_TEXT[status.state] || status.state}${progress}`;
  byId("btn-start").disabled = status.state === "running";
  byId("btn-stop").disabled = status.state !== "running";
}

function renderLogTail(status) {
  const box = byId("log-tail");
  const show = status.state === "error" || status.state === "stopped";
  box.textContent = show ? (status.log_tail || []).join("\n") : "";
  box.classList.toggle("hidden", !show || !box.textContent);
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(pollStatus, delay);
}

async function pollStatus() {
  clearTimeout(pollTimer);
  const seq = ++pollSeq;
  let status;
  try {
    status = await api("/api/status");
  } catch (err) {
    if (seq !== pollSeq) return;
    statusReadFailed = true;
    showError(`상태를 읽지 못했습니다: ${err.message} (잠시 뒤 다시 시도합니다)`);
    schedulePoll(POLL_RETRY_MS);
    return;
  }
  if (seq !== pollSeq) return;
  if (statusReadFailed) {
    statusReadFailed = false;
    showError("");
  }
  renderStatusPill(status);
  renderLogTail(status);
  const running = status.state === "running";
  // 조회형으로 다른 결과를 보는 중이면 화면을 바꾸지 않는다 (상태 표시만 갱신)
  if (currentMode === "live" && status.run_id) showLiveRun(status, running);
  if (status.state === "error") showError(`실행이 오류로 끝났습니다 (코드 ${status.returncode}). 아래 출력을 확인하세요.`);
  if (running) {
    liveRunId = status.run_id;
    schedulePoll(POLL_MS);
  } else if (status.run_id && status.run_id === liveRunId) {
    liveRunId = null;
    await loadRuns();   // 끝난 실행을 목록에 넣고, 클래스별 정확도까지 있는 상세로 다시 그린다
    // 목록을 읽는 동안 새 실행을 시작했으면 이전 실행을 열지 않는다
    if (seq === pollSeq && liveRunId === null && currentMode === "live" && status.rounds_done > 0) openRun(status.run_id);
  }
}

// 실행형 화면: 진행 중이면 상태 응답으로 그리고, 끝난 실행은 상세 결과로 연다 (조회형에서 고른 결과와 무관)
function showLiveRun(status, running) {
  // 진행 중이거나, 한 라운드도 못 끝내고 멈춘 실행(상세 결과가 없음)은 상태 응답으로 그린다
  if (running || status.rounds_done === 0) {
    selectedRunId = status.run_id;
    openSeq += 1;   // 대기 중인 이전 결과 상세 응답이 이 화면을 덮지 않게
    renderRun({ id: status.run_id, summary: status.summary, plan: status.plan, replay: null, rounds: status.rounds,
      clients: status.clients, class_acc: [], client_acc: [], exclusions: [] }, { live: true, running });
    setText("real-time", `${status.elapsed_sec.toFixed(1)}s`);
  } else if (selectedRunId !== status.run_id && status.rounds_done > 0) {
    openRun(status.run_id);
  }
}
