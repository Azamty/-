"""Safe, vector-preserving vertical SVG page compositor."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
_TAG = lambda name: f"{{{SVG_NS}}}{name}"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_VIEWBOX = re.compile(rf"^\s*({_NUMBER})[\s,]+({_NUMBER})[\s,]+({_NUMBER})[\s,]+({_NUMBER})\s*$")
_LENGTH = re.compile(rf"^\s*({_NUMBER})(?:px|pt|pc|mm|cm|in)?\s*$", re.IGNORECASE)
_URL_REF = re.compile(r"url\(\s*#([^\)\s]+)\s*\)")
_EXTERNAL_URL = re.compile(r"url\(\s*([^#][^\)]*)\)", re.IGNORECASE)

MAX_PAGES = 128
MAX_WIDTH = 20_000.0
MAX_HEIGHT = 300_000.0
PAGE_GAP = 28.0
FORBIDDEN_TAGS = {"script", "foreignObject", "iframe", "object", "embed"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _dimension(value: str | None, *, fallback: float | None = None) -> float:
    if value is None:
        if fallback is None:
            raise ValueError("SVG 页面缺少尺寸")
        return fallback
    match = _LENGTH.fullmatch(value)
    if not match:
        raise ValueError(f"SVG 页面尺寸无效：{value!r}")
    number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("SVG 页面尺寸必须是有限正数")
    return number


def _page_geometry(root: ET.Element) -> tuple[float, float, float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        match = _VIEWBOX.fullmatch(view_box)
        if not match:
            raise ValueError("SVG viewBox 无效")
        x, y, width, height = (float(value) for value in match.groups())
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError("SVG viewBox 必须包含有限正尺寸")
        return x, y, width, height
    width = _dimension(root.attrib.get("width"))
    height = _dimension(root.attrib.get("height"))
    return 0.0, 0.0, width, height


def _reject_unsafe_content(root: ET.Element) -> None:
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in FORBIDDEN_TAGS:
            raise ValueError(f"SVG 含不允许的元素：{tag}")
        for attribute, value in element.attrib.items():
            name = _local_name(attribute).lower()
            text = str(value).strip()
            # LilyPond adds a harmless link to its project homepage around
            # some page metadata.  It is stripped from the composed copy in
            # ``_rewrite_references``; external image/resource URLs remain
            # forbidden because removing them could change visible content.
            if name in {"href", "xlink:href"} and text and not text.startswith(("#", "data:")):
                if _local_name(element.tag) not in {"a", "title", "desc"}:
                    raise ValueError("SVG 含外部 href，拒绝合成长图")
            if _EXTERNAL_URL.search(text):
                raise ValueError("SVG 含外部资源引用，拒绝合成长图")
            if name.lower().startswith("on") or text.lower().lstrip().startswith("javascript:"):
                raise ValueError("SVG 含脚本事件，拒绝合成长图")


def _rewrite_references(element: ET.Element, ids: dict[str, str]) -> None:
    for node in element.iter():
        if "id" in node.attrib and node.attrib["id"] in ids:
            node.attrib["id"] = ids[node.attrib["id"]]
        if node.text:
            node.text = _URL_REF.sub(lambda match: f"url(#{ids.get(match.group(1), match.group(1))})", node.text)
        for attribute, value in list(node.attrib.items()):
            name = _local_name(attribute).lower()
            if name in {"href", "xlink:href"} and value.strip() and not value.strip().startswith(("#", "data:")):
                del node.attrib[attribute]
                continue
            rewritten = _URL_REF.sub(lambda match: f"url(#{ids.get(match.group(1), match.group(1))})", value)
            if name in {"href", "xlink:href"} and rewritten.startswith("#"):
                rewritten = "#" + ids.get(rewritten[1:], rewritten[1:])
            node.attrib[attribute] = rewritten


def _page_clone(
    root: ET.Element,
    page_index: int,
    source_x: float,
    source_y: float,
    target_y: float,
    width: float,
    height: float,
    target_width: float,
) -> ET.Element:
    prefix = f"page{page_index + 1}-"
    ids: dict[str, str] = {}
    for node in root.iter():
        identifier = node.attrib.get("id")
        if identifier:
            if identifier in ids:
                raise ValueError(f"SVG 第 {page_index + 1} 页含重复 id：{identifier}")
            ids[identifier] = prefix + identifier
    scale = target_width / width
    group = ET.Element(
        _TAG("g"),
        {
            "id": f"{prefix.rstrip('-')}",
            "transform": f"translate(0 {target_y:g}) scale({scale:.10g}) translate({-source_x:.10g} {-source_y:.10g})",
        },
    )
    for child in list(root):
        clone = deepcopy(child)
        _rewrite_references(clone, ids)
        group.append(clone)
    return group


def merge_svg_pages(
    page_paths: list[str | Path],
    destination: str | Path,
    *,
    gap: float = PAGE_GAP,
) -> Path:
    """Compose pages into one vertical SVG without rasterization or string splicing."""

    if not page_paths:
        raise ValueError("没有可合并的 SVG 页面")
    if len(page_paths) > MAX_PAGES:
        raise ValueError(f"SVG 页面数超过限制（最多 {MAX_PAGES} 页）")
    if not math.isfinite(gap) or gap < 0 or gap > 500:
        raise ValueError("SVG 页面间距无效")
    pages: list[tuple[ET.Element, float, float, float, float]] = []
    for page_path in page_paths:
        source = Path(page_path).resolve()
        if not source.is_file():
            raise ValueError(f"SVG 页面不存在：{page_path}")
        try:
            root = ET.parse(source).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"SVG 页面无法解析：{page_path}") from exc
        if _local_name(root.tag) != "svg":
            raise ValueError("SVG 页面根元素无效")
        _reject_unsafe_content(root)
        x, y, width, height = _page_geometry(root)
        pages.append((root, x, y, width, height))
    target_width = min(MAX_WIDTH, max(page[3] for page in pages))
    if target_width <= 0:
        raise ValueError("SVG 合并目标宽度无效")
    total_height = sum(height * target_width / width for _root, _x, _y, width, height in pages)
    total_height += gap * max(0, len(pages) - 1)
    if not math.isfinite(total_height) or total_height <= 0 or total_height > MAX_HEIGHT:
        raise ValueError(f"SVG 长图尺寸超过限制（最大高度 {MAX_HEIGHT:g}px）")

    canvas = ET.Element(
        _TAG("svg"),
        {
            "version": "1.1",
            "viewBox": f"0 0 {target_width:g} {total_height:g}",
            "width": f"{target_width:g}",
            "height": f"{total_height:g}",
            "role": "img",
            "aria-label": "纵向合并谱面",
        },
    )
    cursor = 0.0
    for index, (root, source_x, source_y, width, height) in enumerate(pages):
        canvas.append(_page_clone(root, index, source_x, source_y, cursor, width, height, target_width))
        cursor += height * target_width / width
        if index + 1 < len(pages):
            cursor += gap
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.ElementTree(canvas)
    tree.write(destination_path, encoding="utf-8", xml_declaration=True)
    return destination_path
