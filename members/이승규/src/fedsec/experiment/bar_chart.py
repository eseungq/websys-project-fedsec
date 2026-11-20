"""묶음 막대그래프를 SVG 문자열로 만든다 (표준 라이브러리만. 비교 그래프용).

값이 없는 칸(예: 발산해 40라운드 정확도가 없는 실행)은 막대를 그리지 않고 메모만 세로로 적는다.
"""

from collections.abc import Sequence
from xml.sax.saxutils import escape

# 계열 색: 정상 3개(파랑 계열), 공격 3개(주황 계열) 순서
PALETTE = ("#1f4e79", "#4f81bd", "#9dc3e6", "#c55a11", "#f4a261", "#f7d08a")
BAR_WIDTH = 16
GROUP_GAP = 28
PLOT_HEIGHT = 240
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 48, 16, 72, 36
TICK_STEP = 20
LEGEND_ITEM_WIDTH = 104


def _check(groups: Sequence[str], series: Sequence[str], values, notes, y_max: float) -> None:
    if len(series) > len(PALETTE):
        raise ValueError(f"계열은 {len(PALETTE)}개까지 그릴 수 있습니다: {len(series)}개")
    shapes = [len(values), len(notes)] == [len(groups)] * 2 and all(
        len(v) == len(n) == len(series) for v, n in zip(values, notes))
    if not shapes:
        raise ValueError("values·notes의 개수가 그룹 × 계열과 다릅니다.")
    if any(v is not None and not 0 <= v <= y_max for row in values for v in row):
        raise ValueError(f"값은 0~{y_max} 범위여야 합니다.")


def _text(x: float, y: float, content: str, size: int = 11, anchor: str = "middle", extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{extra}>'
            f"{escape(content)}</text>")


def _axis(width: float, base: float, y_max: float) -> list[str]:
    parts = []
    for tick in range(0, int(y_max) + 1, TICK_STEP):
        y = base - PLOT_HEIGHT * tick / y_max
        parts.append(f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{width - MARGIN_RIGHT}" y2="{y:.1f}" '
                     'stroke="#d9d9d9" stroke-width="1"/>')
        parts.append(_text(MARGIN_LEFT - 6, y + 4, str(tick), size=10, anchor="end"))
    return parts


def _legend(series: Sequence[str]) -> list[str]:
    parts = []
    for i, name in enumerate(series):
        x = MARGIN_LEFT + i * LEGEND_ITEM_WIDTH
        parts.append(f'<rect class="legend" x="{x}" y="34" width="10" height="10" fill="{PALETTE[i]}"/>')
        parts.append(_text(x + 14, 43, name, size=11, anchor="start"))
    return parts


def _group(index: int, name: str, row, notes, base: float, y_max: float) -> list[str]:
    group_width = len(row) * BAR_WIDTH
    left = MARGIN_LEFT + GROUP_GAP / 2 + index * (group_width + GROUP_GAP)
    parts = [_text(left + group_width / 2, base + 20, name, size=12)]
    for i, (value, note) in enumerate(zip(row, notes)):
        x = left + i * BAR_WIDTH
        if value is not None:
            height = PLOT_HEIGHT * value / y_max
            parts.append(f'<rect class="bar" x="{x:.1f}" y="{base - height:.1f}" width="{BAR_WIDTH - 2}" '
                         f'height="{height:.1f}" fill="{PALETTE[i]}"/>')
        if note:
            cx, cy = x + BAR_WIDTH / 2, base - 6
            parts.append(_text(cx, cy, note, size=10, anchor="start",
                               extra=f' transform="rotate(-90 {cx:.1f} {cy:.1f})" fill="#c00000"'))
    return parts


def grouped_bar_svg(title: str, groups: Sequence[str], series: Sequence[str],
                    values: Sequence[Sequence[float | None]], notes: Sequence[Sequence[str]],
                    y_label: str = "", y_max: float = 100) -> str:
    """values[그룹][계열]. None이면 막대 없이 notes[그룹][계열]만 적는다."""
    _check(groups, series, values, notes, y_max)
    plot_width = len(groups) * (len(series) * BAR_WIDTH + GROUP_GAP)
    width = max(MARGIN_LEFT + plot_width + MARGIN_RIGHT, MARGIN_LEFT + len(series) * LEGEND_ITEM_WIDTH)
    height = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM
    base = MARGIN_TOP + PLOT_HEIGHT
    parts = [f'<rect class="background" x="0" y="0" width="{width}" height="{height}" fill="white"/>',
             _text(MARGIN_LEFT, 20, title, size=14, anchor="start", extra=' font-weight="bold"'),
             *_legend(series), *_axis(width, base, y_max)]
    if y_label:
        parts.append(_text(12, MARGIN_TOP - 8, y_label, size=10, anchor="start"))
    for index, (name, row, row_notes) in enumerate(zip(groups, values, notes)):
        parts += _group(index, name, row, row_notes, base, y_max)
    body = "\n".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="sans-serif">\n{body}\n</svg>\n')
