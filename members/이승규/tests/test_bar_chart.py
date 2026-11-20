"""묶음 막대그래프 SVG (비교 그래프)."""

import xml.etree.ElementTree as ET

import pytest

from fedsec.experiment.bar_chart import PALETTE, grouped_bar_svg

SVG = "{http://www.w3.org/2000/svg}"


def _chart(values, notes=None, **kwargs):
    notes = notes or [["" for _ in row] for row in values]
    return ET.fromstring(grouped_bar_svg("제목", ["A", "B"], ["정상", "공격"], values, notes, **kwargs))


def _texts(root):
    return [t.text for t in root.iter(f"{SVG}text")]


def test_draws_one_bar_per_value():
    root = _chart([[70.0, 60.0], [50.0, 40.0]])

    bars = [r for r in root.iter(f"{SVG}rect") if r.get("class") == "bar"]

    assert len(bars) == 4


def test_bar_height_is_proportional_to_value():
    root = _chart([[100.0, 50.0], [25.0, 0.0]])

    heights = [float(r.get("height")) for r in root.iter(f"{SVG}rect") if r.get("class") == "bar"]

    assert heights[1] == pytest.approx(heights[0] / 2)
    assert heights[2] == pytest.approx(heights[0] / 4)
    assert heights[3] == 0


def test_missing_value_has_no_bar_but_shows_note():
    root = _chart([[70.0, None], [50.0, 40.0]], notes=[["", "발산"], ["", ""]])

    bars = [r for r in root.iter(f"{SVG}rect") if r.get("class") == "bar"]

    assert len(bars) == 3
    assert "발산" in _texts(root)


def test_shows_title_groups_and_legend():
    texts = _texts(_chart([[70.0, 60.0], [50.0, 40.0]]))

    assert {"제목", "A", "B", "정상", "공격"} <= set(texts)


def test_escapes_text():
    svg = grouped_bar_svg("a<b & c", ["A"], ["정상"], [[1.0]], [[""]])

    assert ET.fromstring(svg).find(f"{SVG}text").text == "a<b & c"


def test_rejects_too_many_series():
    series = [str(i) for i in range(len(PALETTE) + 1)]
    with pytest.raises(ValueError, match="계열"):
        grouped_bar_svg("t", ["A"], series, [[1.0] * len(series)], [[""] * len(series)])


def test_rejects_value_shape_mismatch():
    with pytest.raises(ValueError, match="개수"):
        grouped_bar_svg("t", ["A", "B"], ["정상"], [[1.0]], [[""]])


def test_rejects_value_outside_axis():
    with pytest.raises(ValueError, match="범위"):
        grouped_bar_svg("t", ["A"], ["정상"], [[120.0]], [[""]])
