from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from backend.jianpu_score.high_accuracy_service import _collect_artifacts
from backend.jianpu_score.render import natural_svg_sort_key
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


def test_twelve_pages_keep_natural_order_in_render_service_and_long_svg(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    page_paths = []
    for page in range(12, 0, -1):
        path = destination / f"piano.score-{page}.svg"
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><text id="page-{page}">PAGE {page}</text></svg>',
            encoding="utf-8",
        )
        page_paths.append(path)
    long_path = destination / "piano.score.long.svg"
    long_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><text>long</text></svg>',
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text("{}", encoding="utf-8")

    ordered = sorted(page_paths, key=natural_svg_sort_key)
    assert [path.name for path in ordered] == [f"piano.score-{page}.svg" for page in range(1, 13)]
    artifacts = _collect_artifacts(destination, manifest_path=destination / "manifest.json")
    service_pages = [item.path.name for item in artifacts if item.kind == "svg"]
    assert service_pages == [f"piano.score-{page}.svg" for page in range(1, 13)] + [long_path.name]

    merge_svg_pages(ordered, long_path, gap=0)
    groups = list(ET.parse(long_path).getroot())
    assert [group.attrib["id"] for group in groups] == [f"page{page}" for page in range(1, 13)]
