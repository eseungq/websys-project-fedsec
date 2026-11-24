// 안태현 PR #8 대시보드(MIT)를 바탕으로 이 엔진에 맞게 다시 작성 (라이선스: web/LICENSE-dashboard)
// SVG 꺾은선·막대. createElementNS로만 만든다 (외부 차트 라이브러리 없음).
"use strict";

const CHART_W = 540;
const CHART_H = 220;
const CHART_PAD = 36;
const ACC_COLOR = "#2563eb";
const LOSS_COLOR = "#10b981";
const MAX_X_LABELS = 10;

// rounds: rounds.csv 행 목록 (round, val_acc 0~1, val_loss)
function renderTrendChart(container, rounds) {
  clear(container);
  const points = rounds.filter((r) => isNum(r.round));
  if (!points.length) {
    container.appendChild(el("div", { className: "empty-state", text: "아직 끝난 라운드가 없습니다." }));
    return;
  }
  const n = points.length;
  const losses = points.map((r) => r.val_loss).filter(isNum);
  const maxLoss = Math.max(2.5, ...losses);
  const x = (i) => (n === 1 ? CHART_W / 2 : CHART_PAD + (i / (n - 1)) * (CHART_W - CHART_PAD * 2));
  const yAcc = (v) => CHART_H - CHART_PAD - v * (CHART_H - CHART_PAD * 2);
  const yLoss = (v) => CHART_H - CHART_PAD - (Math.min(v, maxLoss) / maxLoss) * (CHART_H - CHART_PAD * 2);

  const svg = svgEl("svg", { viewBox: `0 0 ${CHART_W} ${CHART_H}`, class: "chart-svg", role: "img",
    "aria-label": "라운드별 검증 정확도와 손실" });
  svg.appendChild(svgEl("line", { x1: CHART_PAD, y1: CHART_PAD, x2: CHART_PAD, y2: CHART_H - CHART_PAD, stroke: "#cbd5e1" }));
  svg.appendChild(svgEl("line", { x1: CHART_PAD, y1: CHART_H - CHART_PAD, x2: CHART_W - CHART_PAD, y2: CHART_H - CHART_PAD, stroke: "#cbd5e1" }));
  svg.appendChild(svgEl("text", { x: 4, y: CHART_PAD + 4, "font-size": 10, fill: "#64748b" }, "100%"));
  svg.appendChild(svgEl("text", { x: 12, y: CHART_H - CHART_PAD, "font-size": 10, fill: "#64748b" }, "0%"));
  svg.appendChild(svgEl("text", { x: CHART_W - CHART_PAD + 4, y: CHART_PAD + 4, "font-size": 10, fill: LOSS_COLOR },
    maxLoss.toFixed(1)));

  const line = (values, y, attrs) => {
    const coords = values.map((v, i) => (isNum(v) ? `${x(i)},${y(v)}` : null)).filter(Boolean).join(" ");
    if (coords) svg.appendChild(svgEl("polyline", { fill: "none", points: coords, ...attrs }));
  };
  line(points.map((r) => r.val_acc), yAcc, { stroke: ACC_COLOR, "stroke-width": 2.5 });
  line(points.map((r) => r.val_loss), yLoss, { stroke: LOSS_COLOR, "stroke-width": 2, "stroke-dasharray": "4,3" });

  const step = Math.max(1, Math.ceil(n / MAX_X_LABELS));
  points.forEach((r, i) => {
    if (isNum(r.val_acc)) {
      const dot = svgEl("circle", { cx: x(i), cy: yAcc(r.val_acc), r: 3, fill: ACC_COLOR });
      dot.appendChild(svgEl("title", {}, `R${r.round} 정확도 ${pct(r.val_acc)} · 손실 ${num(r.val_loss)}`));
      svg.appendChild(dot);
    }
    if (i % step === 0 || i === n - 1) {
      svg.appendChild(svgEl("text", { x: x(i), y: CHART_H - 12, "font-size": 10, "text-anchor": "middle", fill: "#64748b" }, `R${r.round}`));
    }
  });
  container.appendChild(svg);
}

// items: [{label, value(0~1 막대 길이), text(오른쪽 표시)}]
function barRows(title, items) {
  const box = el("div", {}, [el("div", { className: "section-subtitle", text: title })]);
  items.forEach((item) => {
    const fill = el("div", { className: "mini-bar-fill" });
    fill.style.width = `${Math.max(0, Math.min(1, isNum(item.value) ? item.value : 0)) * 100}%`;
    box.appendChild(el("div", { className: "mini-bar-row" }, [
      el("span", { className: "mini-bar-label", text: item.label }),
      el("div", { className: "mini-bar-track" }, [fill]),
      el("span", { className: "mini-bar-value", text: item.text }),
    ]));
  });
  return box;
}
