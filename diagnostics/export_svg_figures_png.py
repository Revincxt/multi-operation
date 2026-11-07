"""Export the project's simple SVG chart figures as PNG using macOS sips."""

from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURE_DIR = (Path(__file__).resolve().parents[1] / "experiments"
              / "lstm_fixed_random_20261009" / "grid_policy_timelines")
FONT_WIDTHS = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 278, ";": 278,
    "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015,
    "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722,
    "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722,
    "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556,
    "o": 556, "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556,
    "v": 500, "w": 722, "x": 500, "y": 500, "z": 500, "{": 334, "|": 260,
    "}": 334, "~": 584,
}


def rgb(value: str) -> tuple[float, float, float] | None:
    if not value or value == "none":
        return None
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return None


def pdf_text(value: str) -> str:
    value = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return value.encode("cp1252", errors="replace").decode("cp1252")


def text_width(value: str, size: float, bold: bool) -> float:
    # Helvetica-Bold is slightly wider on average; this only affects centered labels.
    weight = 1.015 if bold else 1.0
    return sum(FONT_WIDTHS.get(char, 556) for char in value) * size * weight / 1000


def parse_svg(path: Path) -> tuple[float, float, str]:
    root = ET.parse(path).getroot()
    width, height = float(root.attrib["width"]), float(root.attrib["height"])
    commands: list[str] = []
    style_node = root.find("{http://www.w3.org/2000/svg}style")
    styles: dict[str, dict[str, str]] = {}
    if style_node is not None and style_node.text:
        for class_name, raw in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", style_node.text):
            styles[class_name] = dict(
                item.strip().split(":", 1) for item in raw.split(";") if ":" in item)

    def set_color(color: tuple[float, float, float] | None, mode: str) -> None:
        if color is not None:
            commands.append(f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} {mode}")

    def common_shape(element: ET.Element) -> tuple[dict[str, str], tuple[float, float, float] | None,
                                                    tuple[float, float, float] | None, float]:
        attrs = element.attrib
        return attrs, rgb(attrs.get("fill")), rgb(attrs.get("stroke")), float(attrs.get("stroke-width", 1))

    ns = "{http://www.w3.org/2000/svg}"
    for element in root:
        tag = element.tag.removeprefix(ns)
        if tag in ("style",):
            continue
        if tag == "text":
            value = "".join(element.itertext())
            attrs = element.attrib
            class_name = attrs.get("class", "")
            style = styles.get(class_name, {})
            size = float(style.get("font-size", "12px").removesuffix("px"))
            bold = style.get("font-weight", "normal") == "700"
            color = rgb(style.get("fill", "#202b38")) or (0, 0, 0)
            x = float(attrs.get("x", 0))
            y = height - float(attrs.get("y", 0))
            anchor = attrs.get("text-anchor", "start")
            if anchor == "middle":
                x -= text_width(value, size, bold) / 2
            elif anchor == "end":
                x -= text_width(value, size, bold)
            set_color(color, "rg")
            commands.append(f"BT /{'F2' if bold else 'F1'} {size:g} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({pdf_text(value)}) Tj ET")
        elif tag == "rect":
            attrs, fill, stroke, stroke_width = common_shape(element)
            x, top = float(attrs.get("x", 0)), float(attrs.get("y", 0))
            w = width if attrs.get("width") == "100%" else float(attrs["width"])
            h = height if attrs.get("height") == "100%" else float(attrs["height"])
            y = height - top - h
            commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
            if fill:
                set_color(fill, "rg")
            if stroke:
                set_color(stroke, "RG")
                commands.append(f"{stroke_width:g} w")
            commands.append("B" if fill and stroke else "f" if fill else "S" if stroke else "n")
        elif tag == "line":
            attrs, _fill, stroke, stroke_width = common_shape(element)
            if not stroke:
                continue
            set_color(stroke, "RG")
            commands.append(f"{stroke_width:g} w")
            dash = attrs.get("stroke-dasharray")
            commands.append(f"[{dash.replace(',', ' ')}] 0 d" if dash else "[] 0 d")
            x1, y1 = float(attrs["x1"]), height - float(attrs["y1"])
            x2, y2 = float(attrs["x2"]), height - float(attrs["y2"])
            commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        elif tag == "path":
            attrs, fill, stroke, stroke_width = common_shape(element)
            set_color(fill, "rg")
            set_color(stroke, "RG")
            commands.append(f"{stroke_width:g} w")
            commands.append("1 j 1 J")
            tokens = re.findall(r"[ML]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", attrs["d"])
            mode = None
            index = 0
            while index < len(tokens):
                if tokens[index] in ("M", "L"):
                    mode = tokens[index]
                    index += 1
                x = float(tokens[index])
                y = height - float(tokens[index + 1])
                op = "m" if mode == "M" else "l"
                commands.append(f"{x:.2f} {y:.2f} {op}")
                index += 2
            commands.append("B" if fill and stroke else "f" if fill else "S" if stroke else "n")
        elif tag == "polygon":
            attrs, fill, stroke, stroke_width = common_shape(element)
            points = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)", attrs["points"])]
            if len(points) < 4:
                continue
            set_color(fill, "rg")
            set_color(stroke, "RG")
            commands.append(f"{stroke_width:g} w")
            commands.append("q /GS1 gs" if float(attrs.get("fill-opacity", 1)) < 1 else "q")
            commands.append(f"{points[0]:.2f} {height-points[1]:.2f} m")
            for index in range(2, len(points), 2):
                commands.append(f"{points[index]:.2f} {height-points[index+1]:.2f} l")
            commands.append("h")
            commands.append("B" if fill and stroke else "f" if fill else "S" if stroke else "n")
            commands.append("Q")
    return width, height, "\n".join(commands)


def make_pdf(width: float, height: float, stream: str) -> bytes:
    content = stream.encode("cp1252", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}] "
         "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> /ExtGState << /GS1 7 0 R >> >> "
         "/Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Type /ExtGState /ca 0.18 /CA 0.18 >>",
    ]
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(result)


def main() -> None:
    paths = sorted(FIGURE_DIR.glob("*.svg"))
    if not paths:
        raise FileNotFoundError
    with tempfile.TemporaryDirectory(prefix="svg_pdf_") as temp_dir:
        for svg in paths:
            width, height, stream = parse_svg(svg)
            pdf = Path(temp_dir) / f"{svg.stem}.pdf"
            png = svg.with_suffix(".png")
            pdf.write_bytes(make_pdf(width, height, stream))
            subprocess.run(["sips", "-s", "format", "png", str(pdf), "--out", str(png)],
                           check=True, stdout=subprocess.DEVNULL)
            print(f"{png} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
