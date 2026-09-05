from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from backend.jianpu_score.svg_long import merge_svg_pages


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
<defs><linearGradient id="shared-gradient"><stop offset="0" stop-color="#111"/><stop offset="1" stop-color="#eee"/></linearGradient><clipPath id="shared-clip"><rect width="{width}" height="{height}"/></clipPath></defs>
<g id="shared-group" clip-path="url(#shared-clip)"><rect id="shared-rect" width="{width}" height="{height}" fill="url(#shared-gradient)"/><use id="shared-use" href="#shared-rect"/></g>
</svg>"""


def test_merge_svg_pages_namespaces_ids_and_preserves_vector_order(tmp_path: Path) -> None:
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    first.write_text(SVG.format(width=100, height=50), encoding="utf-8")
    second.write_text(SVG.format(width=80, height=40), encoding="utf-8")
    destination = merge_svg_pages([first, second], tmp_path / "long.svg", gap=10)

    root = ET.parse(destination).getroot()
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    assert view_box[:2] == [0.0, 0.0]
    assert view_box[2] == 100.0
    assert view_box[3] == 110.0  # 50 + 10 + (40 scaled to target width 100)
    groups = list(root)
    assert [group.attrib["id"] for group in groups] == ["page1", "page2"]
    ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    assert "page1-shared-gradient" in ids
    assert "page2-shared-gradient" in ids
    rects = [node for node in root.iter() if node.tag.endswith("rect") and node.attrib.get("id")]
    groups_with_clips = [node for node in root.iter() if node.tag.endswith("g") and node.attrib.get("clip-path")]
    assert rects[0].attrib["fill"] == "url(#page1-shared-gradient)"
    assert groups_with_clips[0].attrib["clip-path"] == "url(#page1-shared-clip)"
    assert rects[1].attrib["fill"] == "url(#page2-shared-gradient)"
    uses = [node for node in root.iter() if node.tag.endswith("use")]
    assert uses[0].attrib["href"] == "#page1-shared-rect"
    assert uses[1].attrib["href"] == "#page2-shared-rect"


def test_merge_svg_pages_rejects_scripts_and_page_limits(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.svg"
    unsafe.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><script>alert(1)</script></svg>', encoding="utf-8")
    with pytest.raises(ValueError, match="不允许"):
        merge_svg_pages([unsafe], tmp_path / "long.svg")
