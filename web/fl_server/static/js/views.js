// 안태현 PR #8 대시보드(MIT)를 바탕으로 이 엔진에 맞게 다시 작성 (라이선스: web/fl_server/LICENSE-dashboard)
// 결과 폴더 한 개(조회형 상세 또는 실행형 상태)를 화면에 그린다. 모든 값은 textContent로 넣는다.
"use strict";

const CIFAR_CLASSES = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"];
const NUM_CLIENTS = 10;
const AUDIT_ROW_LIMIT = 300;
const SUMMARY_LABELS = {
  experiment: "실험 ID", env: "환경", attack: "공격", aggregator: "집계", seed: "시드",
  rounds_completed: "완료 라운드", final_val_acc: "최종 검증 정확도", final_test_acc: "최종 테스트 정확도",
  final_test_loss: "최종 테스트 손실", diverged_round: "발산 라운드", rounds_not_updated: "갱신 안 된 라운드",
  total_virtual_time: "가상 시간 합", total_sec: "실제 시간 합(초)", mean_comm_sec: "평균 통신 시간(초)", runner: "실행기",
};

function lastRow(rows) {
  return rows.length ? rows[rows.length - 1] : null;
}

function sumOf(rows, key) {
  return rows.reduce((acc, r) => acc + (isNum(r[key]) ? r[key] : 0), 0);
}

function finalAccuracy(summary, rounds) {
  if (summary && isNum(summary.final_test_acc)) return { value: summary.final_test_acc, label: "테스트" };
  if (summary && isNum(summary.final_val_acc)) return { value: summary.final_val_acc, label: "검증" };
  const last = lastRow(rounds);
  return last && isNum(last.val_acc) ? { value: last.val_acc, label: "현재 검증" } : null;
}

function renderKpis(detail) {
  const { summary, rounds } = detail;
  const acc = finalAccuracy(summary, rounds);
  setText("kpi-acc", acc ? pct(acc.value) : "-");
  const last = lastRow(rounds);
  setText("kpi-acc-sub", acc ? `${acc.label} 정확도 · 마지막 검증 ${last ? pct(last.val_acc) : "-"}` : "결과 없음");

  setText("kpi-participation", last ? `${last.num_selected ?? "-"} / ${last.num_accepted ?? "-"} / ${last.num_late ?? "-"}` : "-");
  setText("kpi-participation-sub", rounds.length
    ? `전체 ${rounds.length}라운드 합계: ${sumOf(rounds, "num_selected")} / ${sumOf(rounds, "num_accepted")} / ${sumOf(rounds, "num_late")}`
    : "전체 라운드 합계: -");

  setText("kpi-mal-ratio", last && isNum(last.accepted_malicious_ratio) ? pct(last.accepted_malicious_ratio, 1) : "N/A");
  const meanRatio = mean(rounds.map((r) => r.accepted_malicious_ratio));
  setText("kpi-mal-sub", `전체 악성 비율: ${last ? pct(last.overall_malicious_ratio, 1) : "-"} · 라운드 평균 채택 비율: ${pct(meanRatio, 1)}`);

  const s = summary || {};
  const headline = isNum(s.diverged_round)
    ? `R${s.diverged_round}에서 발산`
    : `${s.rounds_completed ?? rounds.length}라운드 · ${s.aggregator ?? (last ? last.aggregator : "-")}`;
  setText("kpi-summary", summary ? headline : `${rounds.length}라운드 진행 중`);
  const attack = s.attack === true ? "공격" : s.attack === false ? "정상" : "-";
  setText("kpi-summary-sub", `환경 ${s.env ?? "-"} · ${attack} · 시드 ${s.seed ?? "-"}`);
}

function clientState(row, running) {
  if (!row) return running ? { cls: "state-training", text: "◉ 학습·제출 중" } : { cls: "", text: "○ 기록 없음" };
  if (!row.selected) return { cls: "state-unselected", text: "— 미선택" };
  if (row.accepted) return { cls: "state-accepted", text: `● 채택 (노름 ${num(row.norm_defended, 2)})` };
  return { cls: "state-delayed", text: `▲ 제외: ${row.reason ?? "-"}` };
}

function renderClientGrid(detail, running) {
  const grid = clear(byId("client-grid"));
  const plan = detail.plan || {};
  const malicious = new Set(plan.malicious || []);
  const delayed = new Set(plan.delayed || []);
  const lastRound = lastRow(detail.rounds);
  const rows = new Map(detail.clients
    .filter((r) => lastRound && r.round === lastRound.round)
    .map((r) => [r.client, r]));
  for (let cid = 0; cid < NUM_CLIENTS; cid += 1) {
    const state = clientState(rows.get(cid), running);
    const badges = el("div");
    if (malicious.has(cid)) badges.appendChild(el("span", { className: "role-badge badge-malicious", text: "악성" }));
    if (delayed.has(cid)) badges.appendChild(el("span", { className: "role-badge badge-delay", text: "지연" }));
    grid.appendChild(el("div", { className: `client-card ${state.cls}` }, [
      el("div", { className: "client-card-top" }, [el("span", { className: "client-name", text: `Client #${cid}` }), badges]),
      el("div", { className: "client-status-text", text: state.text }),
    ]));
  }
}

function renderRoundsTable(rounds) {
  const rows = rounds.slice().reverse().map((r) => cellsRow([
    `R${r.round}`, pct(r.val_acc), num(r.val_loss, 4), r.num_selected, r.num_accepted, r.num_late,
    r.malicious_selected, r.malicious_accepted, pct(r.accepted_malicious_ratio, 1),
    tag(r.updated ? "갱신" : `안 함 (${r.skip_reason ?? "-"})`, r.updated ? "ok" : "warn"),
    num(r.aggregate_norm, 3), num(r.virtual_elapsed, 1),
  ]));
  fillRows("rounds-tbody", rows, "rounds.csv 기록이 없습니다.", 12);
}

function renderBreakdown(detail) {
  const box = clear(byId("breakdown"));
  const testRows = detail.class_acc.filter((r) => r.split === "test");
  const classRows = testRows.length ? testRows : detail.class_acc.filter((r) => r.split === "val");
  const split = testRows.length ? "테스트" : "검증";
  if (!classRows.length) {
    box.appendChild(el("div", { className: "empty-state", text: "class_acc.csv가 없습니다." }));
    return;
  }
  const classItems = classRows.slice().sort((a, b) => a.class - b.class).map((r) => ({
    label: `${CIFAR_CLASSES[r.class] ?? r.class}(${r.class})`, value: r.acc, text: pct(r.acc, 1),
  }));
  box.appendChild(barRows(`클래스별 ${split} 정확도 (R${classRows[0].round})`, classItems));
  const clientTest = detail.client_acc.filter((r) => r.split === (testRows.length ? "test" : "val"));
  if (clientTest.length) {
    const clientItems = clientTest.slice().sort((a, b) => a.client - b.client).map((r) => ({
      label: `Client #${r.client}`, value: r.weighted_acc, text: pct(r.weighted_acc, 1),
    }));
    box.appendChild(barRows(`클라이언트 분포 가중 ${split} 정확도`, clientItems));
  }
}

function clientAggregates(detail) {
  const plan = detail.plan || {};
  const malicious = new Set(plan.malicious || []);
  const delayed = new Set(plan.delayed || []);
  const lastAcc = new Map(detail.client_acc.filter((r) => r.split === "val").map((r) => [r.client, r.weighted_acc]));
  const ids = [...new Set(detail.clients.map((r) => r.client))].filter(isNum).sort((a, b) => a - b);
  return ids.map((cid) => {
    const rows = detail.clients.filter((r) => r.client === cid);
    const selected = rows.filter((r) => r.selected);
    const role = malicious.has(cid) || rows.some((r) => r.malicious) ? tag("악성", "bad")
      : delayed.has(cid) ? tag("지연", "warn") : tag("정상", "gray");
    return cellsRow([
      `Client #${cid}`, role, rows[0] ? rows[0].num_samples : "-", selected.length,
      selected.filter((r) => r.accepted).length, selected.filter((r) => !r.accepted).length,
      num(mean(selected.map((r) => r.update_norm)), 3), num(mean(selected.map((r) => r.norm_submitted)), 3),
      num(mean(selected.map((r) => r.norm_defended)), 3), pct(lastAcc.get(cid), 1),
    ]);
  });
}

function renderSummaryTable(summary) {
  const rows = summary ? Object.entries(summary).map(([key, value]) => {
    const shown = typeof value === "number" && key.endsWith("_acc") ? pct(value)
      : typeof value === "object" && value !== null ? JSON.stringify(value) : value;
    return el("tr", {}, [el("th", { text: SUMMARY_LABELS[key] || key }), el("td", { text: shown ?? "-" })]);
  }) : [];
  fillRows("summary-tbody", rows, "summary.json이 아직 없습니다 (진행 중이거나 중지됨).", 2);
}

function renderExclusions(exclusions) {
  const rows = exclusions.map((r) => cellsRow([`R${r.round}`, r.client_id ?? "-", r.reason ?? "-", r.detail ?? "-"]));
  fillRows("exclusions-tbody", rows, "제외 기록이 없습니다.", 4);
}

function renderAuditLog(clients) {
  const rows = clients.slice().reverse().slice(0, AUDIT_ROW_LIMIT).map((r) => {
    const decision = !r.selected ? tag("미선택", "gray") : r.accepted ? tag("채택", "ok") : tag("제외", "warn");
    return cellsRow([
      `R${r.round}`, `Client #${r.client}`, r.malicious ? tag("악성", "bad") : tag("정상", "gray"), r.num_samples ?? "-",
      num(r.update_norm), num(r.norm_submitted), num(r.norm_defended), decision, r.reason ?? "-",
      `${num(r.virtual_finish, 1)} / ${num(r.train_sec, 2)}s`,
    ]);
  });
  fillRows("audit-tbody", rows, "기록된 로그가 없습니다.", 10);
}

function renderReplayBox(replay) {
  const box = clear(byId("replay-box"));
  if (!replay) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  box.appendChild(el("strong", { text: "재실행 결과 " }));
  box.appendChild(document.createTextNode(
    `원래 실행: ${replay.source ?? "-"} · 시작 라운드: ${replay.from_round ?? "-"} · 바꾼 조건: ${JSON.stringify(replay.change ?? {})}`));
}

// detail: {id, summary, plan, replay, rounds, clients, class_acc, client_acc, exclusions}
function renderRun(detail, options = {}) {
  setText("view-title", `${options.live ? "실행형" : "조회형"} · experiments/${detail.id}`);
  renderReplayBox(detail.replay);
  renderKpis(detail);
  renderClientGrid(detail, Boolean(options.running));
  renderTrendChart(byId("trend-chart"), detail.rounds);
  renderRoundsTable(detail.rounds);
  renderBreakdown(detail);
  fillRows("clients-tbody", clientAggregates(detail), "clients.csv 기록이 없습니다.", 10);
  renderSummaryTable(detail.summary);
  renderExclusions(detail.exclusions || []);
  renderAuditLog(detail.clients);
  const virtual = sumOf(detail.rounds, "virtual_elapsed");
  setText("virtual-time", detail.rounds.length ? `${virtual.toFixed(1)}s` : "-");
  setText("real-time", detail.rounds.length ? `${sumOf(detail.rounds, "total_sec").toFixed(1)}s` : "-");
}
