"""在原研究区域内重新铺设六边形网格，并同步订单、司机和节点归属。

使用方式:
    python3 -m pip install -r requirements-grid.txt
    python3 -m data_pipeline.regrid_hex --grid-count 6 --output my_data/grid6

只改变空间分区，不采样订单，不修改经纬度、时间、行程距离或原始文件。
研究区域取已有 263 个六边形的并集，不另换行政边界或矩形包围框。
"""

import argparse
from dataclasses import dataclass
from datetime import date as calendar_date
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from shapely import intersects_xy
from shapely.geometry import Polygon, mapping
from shapely.ops import transform, unary_union

from src.env.simulator_pattern import input_data_paths


@dataclass(frozen=True)
class LocalProjection:
    """局部等距近似，统一在公里坐标内铺网，避免直接按经纬度量距离。"""

    longitude: float
    latitude: float

    @property
    def scale(self):
        return np.array([np.cos(np.deg2rad(self.latitude)), 1.0]) * 6371 * np.pi / 180

    def forward(self, coordinates):
        return (np.asarray(coordinates, dtype=float) - [self.longitude, self.latitude]) * self.scale

    def inverse(self, coordinates):
        return np.asarray(coordinates, dtype=float) / self.scale + [self.longitude, self.latitude]

    def geometry_to_lonlat(self, geometry):
        def inverse(x, y, z=None):
            return (np.asarray(x) / self.scale[0] + self.longitude,
                    np.asarray(y) / self.scale[1] + self.latitude)
        return transform(inverse, geometry)


def load_region(info_path):
    """完整保留旧网格的覆盖范围，包括边界和内部空隙。"""
    info = pd.read_csv(info_path)
    projection = LocalProjection(float(info.center_lon.mean()), float(info.center_lat.mean()))
    polygons = []
    for row in info.itertuples(index=False):
        coordinates = [[getattr(row, f'vertex_{i}_lon'), getattr(row, f'vertex_{i}_lat')]
                       for i in range(1, 7)]
        polygon = Polygon(projection.forward(coordinates))
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError
        polygons.append(polygon)
    region = unary_union(polygons)
    return region, polygons, projection


def hexagon(center, radius):
    """正六边形，尖顶向北，与旧网格方向一致；radius 为中心到顶点距离。"""
    angles = np.deg2rad(np.arange(30, 390, 60))
    vertices = radius * np.column_stack([np.cos(angles), np.sin(angles)]) + center
    return Polygon(vertices)


def lattice_cells(region, radius, phase):
    """生成与原区域相交的完整六边形及其裁切结果，不截取前 N 个格子。"""
    xmin, ymin, xmax, ymax = region.bounds
    fx, fy = phase
    cells = []
    for row in range(int(np.floor(ymin / (1.5 * radius) - fy)) - 1,
                     int(np.ceil(ymax / (1.5 * radius) - fy)) + 2):
        qmin = int(np.floor(xmin / (np.sqrt(3) * radius) - row / 2 - fx)) - 1
        qmax = int(np.ceil(xmax / (np.sqrt(3) * radius) - row / 2 - fx)) + 1
        for column in range(qmin, qmax + 1):
            center = np.array([np.sqrt(3) * radius * (column + row / 2 + fx),
                               1.5 * radius * (row + fy)])
            full = hexagon(center, radius)
            clipped = full.intersection(region)
            if clipped.area > 1e-9:
                cells.append({'axial_q': column, 'axial_r': row, 'center': center,
                              'full': full, 'region': clipped})
    return cells


def build_layout(region, grid_count):
    """搜索尺度与平移量，得到恰好 N 个有效格子；不按订单量调整边界。"""
    if isinstance(grid_count, bool) or int(grid_count) != grid_count or grid_count <= 0:
        raise ValueError
    nominal_radius = np.sqrt(region.area / (grid_count * 3 * np.sqrt(3) / 2))
    start = max(0.1, np.floor(nominal_radius * 0.75 * 10) / 10)
    phases = [(x, y) for y in (0.0, 0.5) for x in (0.0, 0.25, 0.5, 0.75)]
    best = None
    for radius in np.arange(start, nominal_radius * 3 + 0.05, 0.1):
        for phase in phases:
            cells = lattice_cells(region, float(radius), phase)
            if len(cells) != grid_count:
                continue
            # 避免边缘只有极小面积的空壳格子，保持各区空间规模相对均衡。
            score = min(cell['region'].area for cell in cells)
            if best is None or score > best[0]:
                best = score, float(radius), phase, cells
    if best is None:
        raise ValueError
    _, radius, phase, cells = best
    combined = unary_union([cell['region'] for cell in cells])
    if region.symmetric_difference(combined).area > 1e-7:
        raise ValueError
    if abs(sum(cell['region'].area for cell in cells) - combined.area) > 1e-7:
        raise ValueError
    return radius, phase, cells


def assign_grids(coordinates, projection, centers):
    """六边形铺网的最近中心归属；边界外点沿用旧方案的最近中心规则。"""
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError
    xy = projection.forward(coordinates)
    distances = ((xy[:, None, 0] - centers[:, 0]) ** 2
                 + (xy[:, None, 1] - centers[:, 1]) ** 2)
    return np.argmin(distances, axis=1)


def remap_orders(orders, projection, centers, region):
    """只改原始记录中的起终点 grid_id（第 9、10 列）。"""
    records = [record for batch in orders.values() for record in batch]
    origins = np.array([[row[3], row[2]] for row in records]).reshape(-1, 2)
    destinations = np.array([[row[6], row[5]] for row in records]).reshape(-1, 2)
    origin_ids = assign_grids(origins, projection, centers)
    destination_ids = assign_grids(destinations, projection, centers)
    result = {}
    offset = 0
    for timestamp, batch in orders.items():
        result[timestamp] = []
        for record in batch:
            new_record = list(record)
            new_record[9] = int(origin_ids[offset])
            new_record[10] = int(destination_ids[offset])
            result[timestamp].append(new_record)
            offset += 1
    outside_counts = {}
    for name, coordinates in [('origin', origins), ('destination', destinations)]:
        xy = projection.forward(coordinates)
        outside_counts[f'{name}_outside_original_region'] = int(
            (~intersects_xy(region, xy[:, 0], xy[:, 1])).sum())
    counts = {
        'order_count': len(records), 'retained_order_count': offset,
        'origin_counts_by_grid': np.bincount(origin_ids, minlength=len(centers)).tolist(),
        'destination_counts_by_grid': np.bincount(destination_ids, minlength=len(centers)).tolist(),
        **outside_counts,
    }
    return result, counts


def remap_drivers(drivers, projection, centers):
    result = drivers.copy(deep=True)
    result['grid_id'] = assign_grids(result[['lng', 'lat']].to_numpy(), projection, centers)
    if {'target_loc_lng', 'target_loc_lat'}.issubset(result.columns):
        result['target_grid_id'] = assign_grids(
            result[['target_loc_lng', 'target_loc_lat']].to_numpy(), projection, centers)
    elif 'target_grid_id' in result:
        result['target_grid_id'] = result['grid_id']
    return result


def layout_tables(cells, projection):
    rows = []
    for grid_id, cell in enumerate(cells):
        # 调位坐标必须在裁切区域内部。它不参与订单归属，避免改变六边形边界。
        point = cell['region'].centroid
        if not cell['region'].covers(point):
            point = cell['region'].representative_point()
        center = projection.inverse([point.x, point.y])
        lattice_center = projection.inverse(cell['center'])
        rows.append({'grid_id': grid_id, 'center_lon': center[0], 'center_lat': center[1],
                     'lattice_center_lon': lattice_center[0], 'lattice_center_lat': lattice_center[1],
                     'axial_q': cell['axial_q'], 'axial_r': cell['axial_r'],
                     'area_km2': cell['region'].area})
    adjacency = np.zeros((len(cells), len(cells)), dtype=int)
    for i, left in enumerate(cells):
        for j in range(i + 1, len(cells)):
            # 以裁切后区域的公共边为邻接关系；1 mm 容差仅处理浮点拼接误差。
            shared = left['region'].buffer(1e-6).intersection(cells[j]['region'].boundary)
            if shared.length > 1e-5:
                adjacency[i, j] = adjacency[j, i] = 1
    reached = {0}
    while True:
        expanded = reached | set(np.flatnonzero(adjacency[list(reached)].any(axis=0)))
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != len(cells):
        raise ValueError
    return pd.DataFrame(rows), pd.DataFrame(adjacency)


def save_geojson(path, geometries, projection):
    features = [{'type': 'Feature', 'properties': {'grid_id': grid_id},
                 'geometry': mapping(projection.geometry_to_lonlat(geometry))}
                for grid_id, geometry in enumerate(geometries)]
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}), encoding='utf-8')


def plot_layout(path, old_cells, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    def parts(geometry):
        if geometry.geom_type == 'Polygon':
            yield geometry
        elif hasattr(geometry, 'geoms'):
            for sub_geometry in geometry.geoms:
                yield from parts(sub_geometry)

    fig, axes = plt.subplots(1, 2, figsize=(12, 11), layout='constrained')
    for polygon in old_cells:
        axes[0].add_patch(PolygonPatch(np.asarray(polygon.exterior.coords),
                                      facecolor='#dceaf4', edgecolor='#53778c', linewidth=0.4))
    colors = plt.get_cmap('tab10')
    for grid_id, cell in enumerate(cells):
        for part in parts(cell['region']):
            axes[1].add_patch(PolygonPatch(np.asarray(part.exterior.coords),
                                          facecolor=colors(grid_id % 10), edgecolor='none', alpha=0.75))
        axes[1].plot(*np.asarray(cell['full'].exterior.coords).T,
                     color=colors(grid_id % 10), linestyle='--', linewidth=1.2)
        point = cell['region'].representative_point()
        axes[1].text(point.x, point.y, str(grid_id), ha='center', va='center',
                     fontsize=17, weight='bold', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    bounds = unary_union(old_cells).bounds
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0] - 1, bounds[2] + 1)
        ax.set_ylim(bounds[1] - 1, bounds[3] + 1)
        ax.set_xlabel('Local east-west distance (km)')
        ax.set_ylabel('Local north-south distance (km)')
        ax.grid(alpha=0.15)
    axes[0].set_title(f'Original footprint: {len(old_cells)} hexagonal cells')
    axes[1].set_title(f'Same footprint: {len(cells)} coarser hexagonal cells\n'
                      'Dashed: full hex lattice; colored: clipped study region')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def prepare_hex_inputs(source_data='my_data', output='my_data/grid6',
                       dates=('2015-05-18',), grid_count=6):
    source_data, output = Path(source_data), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError
    dates = list(dates)
    if not dates or len(dates) != len(set(dates)):
        raise ValueError
    paths = [source_data / 'hex_updated_r300_info.csv']
    for date in dates:
        calendar_date.fromisoformat(date)
        paths.extend(input_data_paths(date, source_data))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError
    fingerprints = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in set(paths)}
    region, original_cells, projection = load_region(paths[0])
    print('正在搜索覆盖原区域的六边形划分...', flush=True)
    radius, phase, cells = build_layout(region, grid_count)
    centers = np.asarray([cell['center'] for cell in cells])
    grid_info, adjacency = layout_tables(cells, projection)
    output.mkdir(parents=True, exist_ok=True)
    grid_info.to_csv(output / 'grid_info.csv', index=False)
    adjacency.to_csv(output / 'grid_adjacency.csv', index=False)
    save_geojson(output / 'grid_regions.geojson', [cell['region'] for cell in cells], projection)
    save_geojson(output / 'hex_lattice.geojson', [cell['full'] for cell in cells], projection)
    save_geojson(output / 'study_region.geojson', [region], projection)
    plot_layout(output / 'grid_preview.png', original_cells, cells)

    order_counts = {}
    for date in dates:
        source_orders, source_drivers = input_data_paths(date, source_data)
        with source_orders.open('rb') as file:
            orders = pickle.load(file)
        remapped, counts = remap_orders(orders, projection, centers, region)
        with (output / f'orders_grid{grid_count}_{date}.pkl').open('wb') as file:
            pickle.dump(remapped, file, protocol=pickle.HIGHEST_PROTOCOL)
        order_counts[date] = counts
        print(f'{date}: 保留 {counts["order_count"]} 条订单', flush=True)
    with source_drivers.open('rb') as file:
        drivers = pickle.load(file)
    remapped_drivers = remap_drivers(drivers, projection, centers)
    with (output / 'drivers.pickle').open('wb') as file:
        pickle.dump(remapped_drivers, file, protocol=pickle.HIGHEST_PROTOCOL)
    driver_xy = projection.forward(drivers[['lng', 'lat']].to_numpy())
    node_path = source_data / 'new_grids_263.csv'
    if node_path.is_file():
        nodes = pd.read_csv(node_path)
        nodes['grid_id'] = assign_grids(nodes[['lng', 'lat']].to_numpy(), projection, centers)
        nodes.to_csv(output / 'node_grid_map.csv', index=False)
        fingerprints[str(node_path.resolve())] = hashlib.sha256(node_path.read_bytes()).hexdigest()
    combined = unary_union([cell['region'] for cell in cells])
    manifest = {
        'schema_version': 'hex_grid_inputs_v1', 'grid_num': int(grid_count),
        'grid_shape': 'hexagonal_clipped_to_original_region', 'dates': dates,
        'source_grid_num': len(original_cells), 'radius_km': radius, 'lattice_phase': phase,
        'projection': {'method': 'local_equirectangular_km',
                       'longitude': projection.longitude, 'latitude': projection.latitude},
        'region_definition': 'union of original 263 hexagon polygons; not a new rectangle or administrative boundary',
        'original_area_km2': region.area, 'new_area_km2': combined.area,
        'symmetric_difference_km2': region.symmetric_difference(combined).area,
        'overlap_area_km2': max(0.0, sum(cell['region'].area for cell in cells) - combined.area),
        'grid_areas_km2': grid_info['area_km2'].tolist(),
        'assignment': 'nearest regular-hex lattice center in local km coordinates; same rule for outside points',
        'reposition_centers': 'inside clipped region, separate from lattice centers used for assignment',
        'orders': order_counts, 'driver_count': len(drivers),
        'driver_counts_by_grid': np.bincount(remapped_drivers['grid_id'], minlength=grid_count).tolist(),
        'drivers_outside_original_region': int((~intersects_xy(region, driver_xy[:, 0], driver_xy[:, 1])).sum()),
        'source_sha256': fingerprints,
        'grid_geometry_sha256': hashlib.sha256((output / 'grid_regions.geojson').read_bytes()).hexdigest(),
    }
    (output / 'grid_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'已生成 {len(cells)} 个网格，完整六边形半径 {radius:.2f} km，输出至 {output}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', default='my_data')
    parser.add_argument('--output', default='my_data/grid6')
    parser.add_argument('--grid-count', type=int, default=6)
    parser.add_argument('--dates', nargs='+', default=['2015-05-18'])
    args = parser.parse_args()
    prepare_hex_inputs(args.source_data, args.output, args.dates, args.grid_count)


if __name__ == '__main__':
    main()
