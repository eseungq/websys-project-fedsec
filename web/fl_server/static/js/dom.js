// 안태현 PR #8 대시보드(MIT)를 바탕으로 이 엔진에 맞게 다시 작성 (라이선스: web/fl_server/LICENSE-dashboard)
// DOM·API 도구. 서버·파일 데이터는 textContent로만 넣는다 (innerHTML·템플릿 문자열 사용 금지, XSS 방지).
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

// el("td", {className: "x", text: "값"}, [자식...])
function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
  Object.entries(options.attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
  children.forEach((child) => node.appendChild(child));
  return node;
}

function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function byId(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  byId(id).textContent = text === null || text === undefined ? "-" : String(text);
}

function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function pct(v, digits = 2) {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : "-";
}

function num(v, digits = 3) {
  return isNum(v) ? v.toFixed(digits) : "-";
}

function signedPp(v) {
  if (!isNum(v)) return "-";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}`;
}

function mean(values) {
  const nums = values.filter(isNum);
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
}

// 표 tbody에 행을 채운다. rows가 비면 colspan 안내 한 줄.
function fillRows(tbodyId, rows, emptyText, colspan) {
  const tbody = clear(byId(tbodyId));
  if (!rows.length) {
    tbody.appendChild(el("tr", {}, [el("td", { className: "text-center", text: emptyText, attrs: { colspan } })]));
    return;
  }
  rows.forEach((row) => tbody.appendChild(row));
}

function cellsRow(values) {
  return el("tr", {}, values.map((v) => (v instanceof Node ? el("td", {}, [v]) : el("td", { text: v }))));
}

function tag(text, kind) {
  return el("span", { className: `tag-pill tag-${kind}`, text });
}

// 응답 형식 {"ok", "data", "error"}를 풀어 data를 돌려준다. POST는 JSON으로만 보낸다 (서버가 다른 형식을 거부).
async function api(path, body) {
  const options = body === undefined
    ? { method: "GET", headers: { Accept: "application/json" } }
    : { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body) };
  const resp = await fetch(path, options);
  let envelope;
  try {
    envelope = await resp.json();
  } catch (e) {
    throw new Error(`응답을 읽지 못했습니다 (HTTP ${resp.status})`);
  }
  if (!resp.ok || !envelope.ok) throw new Error(envelope.error || `요청 실패 (HTTP ${resp.status})`);
  return envelope.data;
}

// 경로 조각을 URL에 넣을 때 쓴다 (결과 폴더 ID는 "/"를 포함하므로 조각마다 인코딩)
function encodePath(id) {
  return id.split("/").map(encodeURIComponent).join("/");
}
