"""使用既有边界绘制论文用网格对比图，不重新划分或修改仿真输入。

使用方式:
    python3 -m diagnostics.plot_grid_partition

输出为 my_data/grid6/paper_figures 下的 PDF、SVG、600 dpi PNG 和英文图注。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import Patch, PathPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from shapely import transform
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from data_pipeline.regrid_hex import load_region


COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def polygon_path(polygon):
    """外环逆时针、内环顺时针，矢量导出时保留原边界内部空隙。"""
    polygon = orient(polygon, sign=1.0)
    vertices, codes = [], []
    for ring in (polygon.exterior, *polygon.interiors):
        coordinates = np.asarray(ring.coords)
        vertices.extend(coordinates)
        codes.extend([DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(coordinates) - 2)
                     + [DrawingPath.CLOSEPOLY])
    return DrawingPath(vertices, codes)


def _local_geometries(path, projection):
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    result = {}

    for feature in features:
        grid_id = feature["properties"]["grid_id"]
        geometry = transform(shape(feature["geometry"]), projection.forward)
        if (grid_id in result or geometry.is_empty or not geometry.is_valid
                or geometry.geom_type not in ("Polygon", "MultiPolygon")):
            raise ValueError
        result[grid_id] = geometry
    return result


def load_comparison(source_info, grid_dir):
    """核对网格数、投影和覆盖范围，直接读取生成仿真输入时保存的几何。"""
    source_info, grid_dir = Path(source_info), Path(grid_dir)
    manifest = json.loads((grid_dir / "grid_manifest.json").read_text(encoding="utf-8"))
    region, old_cells, projection = load_region(source_info)
    expected_projection = manifest["projection"]
    if (len(old_cells) != manifest["source_grid_num"]
            or expected_projection["method"] != "local_equirectangular_km"
            or not np.allclose([projection.longitude, projection.latitude],
                               [expected_projection["longitude"], expected_projection["latitude"]],
                               rtol=0, atol=1e-10)):
        raise ValueError
    regions = _local_geometries(grid_dir / "grid_regions.geojson", projection)
    lattice = _local_geometries(grid_dir / "hex_lattice.geojson", projection)
    saved_region = _local_geometries(grid_dir / "study_region.geojson", projection)
    expected_ids = set(range(manifest["grid_num"]))
    combined = unary_union(list(regions.values()))
    difference = region.symmetric_difference(combined).area
    overlap = abs(sum(cell.area for cell in regions.values()) - combined.area)
    if (set(regions) != expected_ids or set(lattice) != expected_ids
            or difference > 1e-7 or overlap > 1e-7
            or region.symmetric_difference(unary_union(list(saved_region.values()))).area > 1e-7
            or any(regions[key].symmetric_difference(lattice[key].intersection(region)).area > 1e-7
                   for key in regions)):
        raise ValueError
    return region, old_cells, regions, lattice, projection, manifest, difference


def _font_family():
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise ValueError


def _fill(ax, geometry, color, edgecolor="none", linewidth=0):
    for polygon in polygon_parts(geometry):
        ax.add_patch(PathPatch(polygon_path(polygon), facecolor=color,
                               edgecolor=edgecolor, linewidth=linewidth, zorder=2))


def _map_decoration(ax, extent):
    xmin, ymin, xmax, ymax = extent
    ax.annotate("", xy=(0.10, 0.94), xytext=(0.10, 0.845), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 0.8,
                            "mutation_scale": 10})
    ax.text(0.10, 0.96, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    # 两幅图采用同一公里坐标系，在右下方空白处放置 2 km 比例尺。
    x, y = xmax - 2.7, ymin + 0.75
    ax.plot([x, x + 2], [y, y], color="#333333", linewidth=1.8, solid_capstyle="butt", zorder=6)
    for offset in (0, 1, 2):
        ax.plot([x + offset, x + offset], [y - 0.11, y + 0.11], color="#333333", lw=0.7, zorder=6)
    ax.text(x + 1, y + 0.24, "2 km", fontsize=8, ha="center", va="bottom")


def plot_grid_partition(source_info="my_data/hex_updated_r300_info.csv", grid_dir="my_data/grid6",
                        output_dir=None):
    source_info, grid_dir = Path(source_info), Path(grid_dir)
    output_dir = Path(output_dir) if output_dir is not None else grid_dir / "paper_figures"
    region, old_cells, regions, lattice, projection, source_manifest, difference = load_comparison(
        source_info, grid_dir)
    if len(regions) > len(COLORS):
        raise ValueError
    output_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = region.bounds
    extent = (xmin - 0.85, ymin - 0.85, xmax + 0.85, ymax + 0.85)
    family = _font_family()
    style = {
        "font.family": family, "font.size": 8.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "path.simplify": False,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 6.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.09, right=0.975, bottom=0.135, top=0.93, wspace=0.18)
        for polygon in old_cells:
            _fill(axes[0], polygon, "#E5EDF2", edgecolor="#738898", linewidth=0.30)
        for grid_id, polygon in sorted(regions.items()):
            pastel = 0.45 * np.array(to_rgb(COLORS[grid_id])) + 0.55
            _fill(axes[1], polygon, pastel)
        # 彩色填充直接显示真实覆盖范围，不用粗描边放大原有细小空隙。
        lattice_edges = unary_union([polygon.boundary for polygon in lattice.values()])
        for line in getattr(lattice_edges, "geoms", [lattice_edges]):
            axes[1].plot(*np.asarray(line.coords).T, color="#68737D", linewidth=0.65,
                         linestyle=(0, (4, 2.5)), alpha=0.8, zorder=4)
        for grid_id, polygon in sorted(regions.items()):
            point = max(polygon_parts(polygon), key=lambda part: part.area).representative_point()
            axes[1].text(point.x, point.y, str(grid_id), ha="center", va="center", fontsize=10,
                         weight="bold", zorder=6,
                         bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                               "edgecolor": "none", "alpha": 0.88})
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
            ax.set_xlabel("Local easting (km)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.tick_params(direction="out", length=3, width=0.55)
            ax.set_axisbelow(True)
            ax.grid(color="#DFE3E6", linewidth=0.3, alpha=0.75)
            _map_decoration(ax, extent)
        axes[0].set_ylabel("Local northing (km)")
        axes[0].set_title(f"(a) Original partition ({len(old_cells)} cells)", loc="left", fontsize=9, pad=9)
        axes[1].set_title(f"(b) Revised partition ({len(regions)} cells)", loc="left", fontsize=9, pad=9)
        handles = [
            Patch(facecolor="#E5EDF2", edgecolor="#738898", lw=0.5, label="Original grid cell"),
            Patch(facecolor="#9CCADD", edgecolor="none", label="Revised grid region"),
            Line2D([], [], color="#68737D", lw=0.7, linestyle=(0, (4, 2.5)),
                   label="Coarse hexagonal lattice"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.027),
                   ncol=3, frameon=False, handlelength=2.3, columnspacing=1.7)
        files = []
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"fig_grid_partition_comparison.{extension}"
            fig.savefig(path, dpi=600, facecolor="white")
            files.append(path.name)
        plt.close(fig)
    caption = (
        f"Spatial partition comparison over the same study footprint. (a) Original partition "
        f"with {len(old_cells)} hexagonal cells. (b) Revised partition with {len(regions)} coarser "
        "hexagonal cells clipped to the original footprint; labels denote the grid IDs used by "
        "the simulator. Colors identify regions and do not encode demand or performance. "
        "Dashed lines show the underlying coarse hexagonal lattice, not an extension of the "
        "study area. Both panels use identical bounds, scale and a north-up orientation.\n\n"
        f"The footprint is the union of the original cells (approximately {region.area:.2f} km²), "
        "not an administrative boundary. Original internal gaps and disconnected components "
        "are retained. Geometry is read directly from the saved simulation inputs without "
        "repartitioning, smoothing or changing order/driver assignments. Coordinates use the "
        f"same local equirectangular approximation as preprocessing, with origin "
        f"({projection.longitude:.8f}°, {projection.latitude:.8f}°); distances and areas are approximate. "
        f"The full coarse hexagons have a center-to-vertex radius of "
        f"{source_manifest['radius_km']:.2f} km.\n"
    )
    (output_dir / "captions.txt").write_text(caption, encoding="utf-8")
    sources = [source_info] + [grid_dir / name for name in (
        "grid_regions.geojson", "hex_lattice.geojson", "study_region.geojson", "grid_manifest.json")]
    manifest = {
        "original_grid_count": len(old_cells), "revised_grid_count": len(regions),
        "grid_ids": sorted(regions), "study_area_km2": region.area,
        "symmetric_difference_km2": difference, "shared_extent_km": extent,
        "projection": source_manifest["projection"], "geometry_smoothing": False,
        "font_family": family, "figure_size_inches": [7.16, 6.6], "png_dpi": 600,
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources}, "files": files,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"论文网格对比图已保存至：{output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="绘制既有网格划分的论文版对比图")
    parser.add_argument("--source-info", default="my_data/hex_updated_r300_info.csv")
    parser.add_argument("--grid-dir", default="my_data/grid6")
    parser.add_argument("--output", dest="output_dir")
    plot_grid_partition(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
