const CIFAR_CLASSES = [
  "비행기(0)", "자동차(1)", "새(2)", "고양이(3)", "사슴(4)",
  "강아지(5)", "개구리(6)", "말(7)", "배(8)", "트럭(9)"
];

let pollTimer = null;
let currentExpId = null;

document.addEventListener("DOMContentLoaded", () => {
  initInitialClientGrid();
  bindEvents();
  fetchStatus();
});

function initInitialClientGrid() {
  const grid = document.getElementById("client-grid");
  grid.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const card = document.createElement("div");
    card.className = "client-card";
    card.innerHTML = `
      <div class="client-card-top">
        <span class="client-name">Client #${i}</span>
      </div>
      <div class="client-status-text">○ 대기 중</div>
    `;
    grid.appendChild(card);
  }
}

function bindEvents() {
  // 공격 유형 변경 시 방어 UI 자동 전환
  document.querySelectorAll('input[name="attack"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const val = e.target.value;
      const flWrap = document.getElementById("fl-defense-wrapper");
      const dlgWrap = document.getElementById("dlg-defense-wrapper");
      const title = document.getElementById("defense-group-title");

      if (val === "dlg") {
        flWrap.classList.add("hidden");
        dlgWrap.classList.remove("hidden");
        title.textContent = "3. 모델 역추적 방어 (노이즈)";
      } else {
        flWrap.classList.remove("hidden");
        dlgWrap.classList.add("hidden");
        title.textContent = "3. 방어 기법 선택";
      }
    });
  });

  // 탭 전환
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // 실행 버튼
  document.getElementById("btn-start").addEventListener("click", startSimulation);

  // 중지 버튼
  document.getElementById("btn-stop").addEventListener("click", stopSimulation);

  // 라운드 재실행 버튼
  document.getElementById("btn-rerun").addEventListener("click", rerunRound);

  // CSV 다운로드 버튼
  document.getElementById("btn-download-csv").addEventListener("click", () => {
    if (!currentExpId) return;
    window.location.href = `/outputs/logs/${currentExpId}_client_audit.csv`;
  });
}

async function startSimulation() {
  const environment = document.querySelector('input[name="environment"]:checked').value;
  const attack = document.querySelector('input[name="attack"]:checked').value;
  const defense = document.getElementById("defense-select").value;
  const dlgNoiseMode = document.getElementById("dlg-noise-select").value;
  const seed = parseInt(document.getElementById("seed-input").value || "1", 10);
  const epochs = parseInt(document.getElementById("epochs-input").value || "10", 10);

  const errBanner = document.getElementById("error-banner");
  errBanner.classList.add("hidden");

  // DLG 선택 시 자동으로 DLG 탭 열기
  if (attack === "dlg") {
    document.querySelector('.tab-btn[data-tab="tab-dlg"]').click();
  } else {
    document.querySelector('.tab-btn[data-tab="tab-trends"]').click();
  }

  try {
    const resp = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        environment,
        attack,
        defense,
        dlg_noise_mode: dlgNoiseMode,
        seed,
        epochs,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "시뮬레이션 시작 실패");

    currentExpId = data.exp_id;
    document.getElementById("rerun-exp-id").value = currentExpId;
    startPolling(500);
  } catch (err) {
    errBanner.textContent = err.message;
    errBanner.classList.remove("hidden");
  }
}

async function stopSimulation() {
  await fetch("/api/stop", { method: "POST" });
  fetchStatus();
}

function startPolling(intervalMs) {
  if (pollTimer) clearInterval(pollTimer);
  fetchStatus();
  pollTimer = setInterval(fetchStatus, intervalMs);
}

async function fetchStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    renderDashboard(data);

    if (data.status === "running") {
      if (!pollTimer) startPolling(500);
    } else if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) {
    console.error("상태 조회 오류:", e);
  }
}

function renderDashboard(data) {
  if (data.exp_id) {
    currentExpId = data.exp_id;
    document.getElementById("btn-download-csv").disabled = false;
    if (!document.getElementById("rerun-exp-id").value) {
      document.getElementById("rerun-exp-id").value = data.exp_id;
    }
  }

  // 상단 상태 배지 및 버튼 제어
  const pill = document.getElementById("status-pill");
  const btnStart = document.getElementById("btn-start");
  const btnStop = document.getElementById("btn-stop");

  pill.className = `status-pill status-${data.status}`;
  const statusMap = {
    idle: "● 대기 중",
    running: "◉ 실행 중",
    completed: "✓ 완료",
    stopped: "■ 중지됨",
    error: "▲ 오류 발생",
  };
  pill.textContent = `${statusMap[data.status] || data.status} (Round ${data.current_round} / ${data.total_rounds})`;

  btnStart.disabled = data.status === "running";
  btnStop.disabled = data.status !== "running";

  if (data.status === "error" && data.error_message) {
    const errBanner = document.getElementById("error-banner");
    errBanner.textContent = data.error_message;
    errBanner.classList.remove("hidden");
  }

  document.getElementById("virtual-time").textContent = `${data.virtual_elapsed_sec.toFixed(1)}s`;
  document.getElementById("real-time").textContent = `${data.real_elapsed_sec.toFixed(1)}s`;

  // KPI 카드 업데이트
  const latestVal = data.val_acc_history.length > 0
    ? `${data.val_acc_history[data.val_acc_history.length - 1].toFixed(2)}%`
    : "-";
  document.getElementById("kpi-test-acc").textContent =
    data.test_acc !== null ? `${data.test_acc.toFixed(2)}%` : latestVal;
  document.getElementById("kpi-val-acc").textContent = `최근 검증 정확도: ${latestVal}`;

  const kpis = data.latest_kpis || {};
  document.getElementById("kpi-adoption").textContent =
    `${kpis.selected_count || 0} / ${kpis.accepted_count || 0} / ${kpis.rejected_count || 0}`;
  document.getElementById("kpi-mal-ratio").textContent = kpis.accepted_malicious_ratio || "N/A";
  document.getElementById("kpi-total-mal").textContent = `전체 악성 비율: ${kpis.total_malicious_ratio || "0.0%"}`;

  const cmp = data.comparison || {};
  const fmtPp = (v) => (v !== null && v !== undefined ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%p` : "-");
  document.getElementById("kpi-sec-gain").textContent = `방어 개선: ${fmtPp(cmp.defense_gain_pp)}`;
  document.getElementById("kpi-sec-damage").textContent =
    `공격 피해: ${fmtPp(cmp.attack_damage_pp)} | 정상 손실: ${fmtPp(cmp.normal_loss_pp)}`;

  // 10개 클라이언트 카드 렌더링
  if (data.client_cards && data.client_cards.length > 0) {
    renderClientCards(data.client_cards);
  }

  // 차트 및 분석 탭 렌더링
  renderTrendChart(data.val_acc_history, data.val_loss_history);
  renderBreakdownBars(data.per_class_acc, data.per_client_acc);
  renderDlgSection(data.dlg_result);
  renderComparisonTable(cmp);
  renderAuditLog(data.audit_logs);
}

function renderClientCards(cards) {
  const grid = document.getElementById("client-grid");
  grid.innerHTML = "";
  cards.forEach((c) => {
    const card = document.createElement("div");
    card.className = `client-card state-${c.status}`;

    let badges = "";
    if (c.is_malicious) badges += `<span class="role-badge badge-malicious">악성</span>`;
    if (c.is_delay_target) badges += `<span class="role-badge badge-delay">지연대상</span>`;

    const iconMap = {
      training: "◉",
      accepted: "●",
      delayed: "▲",
      waiting: "○",
      unselected: "—",
    };

    card.innerHTML = `
      <div class="client-card-top">
        <span class="client-name">Client #${c.client_id}</span>
        <div>${badges}</div>
      </div>
      <div class="client-status-text">${iconMap[c.status] || "○"} ${c.sub_text}</div>
    `;
    grid.appendChild(card);
  });
}

function renderTrendChart(accHist, lossHist) {
  const container = document.getElementById("trend-chart-container");
  if (!accHist || accHist.length === 0) return;

  const w = 540, h = 210, pad = 34;
  const n = accHist.length;
  const maxLoss = Math.max(...lossHist, 2.5);

  const getX = (idx) => (n === 1 ? w / 2 : pad + (idx / (n - 1)) * (w - pad * 2));
  const getYAcc = (val) => h - pad - (val / 100) * (h - pad * 2);
  const getYLoss = (val) => h - pad - (Math.min(val, maxLoss) / maxLoss) * (h - pad * 2);

  const accPoints = accHist.map((v, i) => `${getX(i)},${getYAcc(v)}`).join(" ");
  const lossPoints = lossHist.map((v, i) => `${getX(i)},${getYLoss(v)}`).join(" ");

  let circles = "";
  accHist.forEach((v, i) => {
    circles += `<circle cx="${getX(i)}" cy="${getYAcc(v)}" r="3.5" fill="#2563eb" />`;
    circles += `<text x="${getX(i)}" y="${h - 10}" font-size="10" text-anchor="middle" fill="#64748b">R${i + 1}</text>`;
  });
  lossHist.forEach((v, i) => {
    circles += `<circle cx="${getX(i)}" cy="${getYLoss(v)}" r="3.5" fill="#10b981" />`;
  });

  container.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" width="100%" height="210">
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#cbd5e1" />
      <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#cbd5e1" />
      <text x="6" y="${pad + 4}" font-size="10" fill="#64748b">100%</text>
      <text x="12" y="${h - pad}" font-size="10" fill="#64748b">0%</text>
      <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="${accPoints}" />
      <polyline fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="4,3" points="${lossPoints}" />
      ${circles}
    </svg>
  `;
}

function renderBreakdownBars(perClass, perClient) {
  const container = document.getElementById("breakdown-container");
  if (!perClass || Object.keys(perClass).length === 0) return;

  let html = `<div style="font-size:12px;font-weight:700;margin-bottom:4px;">클래스별 테스트 정확도</div>`;
  for (let c = 0; c < 10; c++) {
    const val = perClass[String(c)] ?? 0;
    html += `
      <div class="mini-bar-row">
        <span class="mini-bar-label">${CIFAR_CLASSES[c]}</span>
        <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${val}%"></div></div>
        <span style="width:42px;text-align:right;font-weight:600;">${val}%</span>
      </div>
    `;
  }
  container.innerHTML = html;
}

function renderDlgSection(dlg) {
  const emptyEl = document.getElementById("dlg-empty");
  const viewEl = document.getElementById("dlg-result-view");
  if (!dlg) {
    emptyEl.classList.remove("hidden");
    viewEl.classList.add("hidden");
    return;
  }

  emptyEl.classList.add("hidden");
  viewEl.classList.remove("hidden");

  const ts = Date.now();
  document.getElementById("dlg-gt-img").src = `${dlg.gt_url}?t=${ts}`;
  document.getElementById("dlg-rec-img").src = `${dlg.rec_url}?t=${ts}`;
  document.getElementById("dlg-gt-meta").textContent = `정답 클래스: ${CIFAR_CLASSES[dlg.gt_label]}`;
  document.getElementById("dlg-rec-meta").textContent = `추론 클래스: ${CIFAR_CLASSES[dlg.inferred_label]}`;
  document.getElementById("dlg-noise-info").textContent = `${dlg.noise_mode} (σ = ${dlg.noise_std})`;
  document.getElementById("dlg-iter-info").textContent = `${dlg.iterations} 회 (L-BFGS)`;
  document.getElementById("dlg-mse-info").textContent = dlg.mse.toFixed(6);
  document.getElementById("dlg-idx-info").textContent = `Client #${dlg.client_id} / Sample #${dlg.sample_index}`;
}

function renderComparisonTable(cmp) {
  const fmtPp = (v) => (v !== null && v !== undefined ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}%p` : "-");
  document.getElementById("cmp-damage").textContent = fmtPp(cmp.attack_damage_pp);
  document.getElementById("cmp-gain").textContent = fmtPp(cmp.defense_gain_pp);
  document.getElementById("cmp-loss").textContent = fmtPp(cmp.normal_loss_pp);

  const tbody = document.getElementById("compare-tbody");
  const rows = cmp.rows || [];
  if (rows.length === 0) return;

  tbody.innerHTML = rows.slice().reverse().map((r) => `
    <tr>
      <td>${r.exp_id}</td>
      <td>${r.environment}</td>
      <td>${r.attack}</td>
      <td>${r.defense}</td>
      <td>${r.seed}</td>
      <td><strong>${(r.test_acc * 100).toFixed(2)}%</strong></td>
      <td>${(r.final_val_acc * 100).toFixed(2)}%</td>
      <td>${r.accepted_malicious_ratio}</td>
    </tr>
  `).join("");
}

function renderAuditLog(logs) {
  const tbody = document.getElementById("audit-tbody");
  if (!logs || logs.length === 0) return;

  tbody.innerHTML = logs.slice().reverse().map((row) => {
    const roleTag = row.is_malicious_role
      ? `<span class="tag-pill tag-bad">악성</span>`
      : `<span class="tag-pill tag-gray">정상</span>`;

    let decClass = "tag-ok";
    if (row.decision === "제외") decClass = "tag-warn";
    if (row.decision === "미선택") decClass = "tag-gray";

    const fmtNum = (n) => (n !== null && n !== undefined ? Number(n).toFixed(3) : "-");

    return `
      <tr>
        <td>R${row.round}</td>
        <td><strong>Client #${row.client_id}</strong></td>
        <td>${roleTag}</td>
        <td>${row.num_samples}</td>
        <td>${fmtNum(row.pre_attack_norm)}</td>
        <td>${fmtNum(row.post_attack_norm)}</td>
        <td>${fmtNum(row.post_defense_norm)}</td>
        <td><span class="tag-pill ${decClass}">${row.decision}</span></td>
        <td>${row.reason}</td>
        <td>${row.virtual_time_sec.toFixed(1)}s / ${row.real_time_sec.toFixed(2)}s</td>
      </tr>
    `;
  }).join("");
}

async function rerunRound() {
  const expId = document.getElementById("rerun-exp-id").value.trim();
  const roundNum = parseInt(document.getElementById("rerun-round").value || "1", 10);
  const newDefense = document.getElementById("rerun-defense").value;
  const resBox = document.getElementById("rerun-result-box");

  try {
    const resp = await fetch("/api/rerun_round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exp_id: expId,
        round: roundNum,
        defense: newDefense,
        unmask_client_ids: [],
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "재실행 오류");

    resBox.classList.remove("hidden");
    resBox.innerHTML = `
      <strong>[Round ${data.round} 재실행 비교 결과]</strong><br />
      방어 기법 변경: <strong>${data.new_defense}</strong> |
      라운드 시작 검증 정확도: <strong>${data.before_round_val_acc}%</strong> ➔
      재집계 후 검증 정확도: <strong>${data.after_rerun_val_acc}%</strong>
      (변화량: <strong>${data.diff_pp >= 0 ? "+" : ""}${data.diff_pp}%p</strong>)
    `;
  } catch (err) {
    resBox.classList.remove("hidden");
    resBox.innerHTML = `<span style="color:#dc2626;">오류: ${err.message}</span>`;
  }
}