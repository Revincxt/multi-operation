"""Plot per-grid metric and action timelines for the small-density policy run."""

from __future__ import annotations

import csv
import gzip
import html
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "lstm_fixed_random_20261009"
OUT_DIR = RUN_DIR / "grid_policy_timelines"
METRICS = [
    ("baseline_request_num", "Bubble requests / 5 min", "count"),
    ("request_rate", "Order call rate", "rate"),
    ("driver_accept_rate", "Driver acceptance rate", "rate"),
    ("order_completion_rate", "Order completion rate", "rate"),
    ("mean_wait_to_match_seconds", "Mean matched wait (s)", "wait"),
    ("platform_revenue", "Platform revenue / 5 min", "revenue"),
]
GRID_COLORS = ["#2369a8", "#d05a47", "#31936e", "#8a62a8", "#c58a1b", "#258d9b"]
PRICE_COLOR = "#2369a8"
RADIUS_COLOR = "#d05a47"


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def clock_label(minute: float) -> str:
    total = int(minute)
    return f"{total // 60:02d}:{total % 60:02d}"


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202b38}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#536273}.panel{font-size:13px;font-weight:600}.tick{font-size:10px;fill:#536273}.rowlabel{font-size:13px;font-weight:600}.note{font-size:11px;fill:#536273}</style>',
        f'<text x="45" y="38" class="title">{html.escape(title)}</text>',
        f'<text x="45" y="62" class="subtitle">{html.escape(subtitle)}</text>',
    ]


def path_for(values: list[float | None], x0: float, y0: float, w: float, h: float,
             y_min: float, y_max: float, step: bool = False) -> str:
    commands: list[str] = []
    previous: tuple[float, float] | None = None
    for index, value in enumerate(values):
        if value is None or not math.isfinite(value):
            previous = None
            continue
        x = x0 + w * index / max(1, len(values) - 1)
        y = y0 + h - (value - y_min) / (y_max - y_min) * h
        if previous is None:
            commands.append(f"M {x:.1f},{y:.1f}")
        elif step:
            commands.append(f"L {x:.1f},{previous[1]:.1f} L {x:.1f},{y:.1f}")
        else:
            commands.append(f"L {x:.1f},{y:.1f}")
        previous = (x, y)
    return " ".join(commands)


def draw_metrics(policy: str, title: str, subtitle: str, series: dict,
                 metric_ranges: dict[str, tuple[float, float]], out_path: Path) -> None:
    left, top, cell_w, cell_h = 178, 100, 250, 157
    width, height = left + cell_w * 6 + 35, top + cell_h * 6 + 55
    parts = svg_start(width, height, title, subtitle)
    x_ticks = [0, 12, 24, 36, 48, 59]
    for metric_i, (key, label, _kind) in enumerate(METRICS):
        y0 = top + metric_i * cell_h
        parts.append(f'<text x="{left - 12}" y="{y0 + 12}" text-anchor="end" class="rowlabel">{html.escape(label)}</text>')
        ymin, ymax = metric_ranges[key]
        for grid in range(6):
            x0 = left + grid * cell_w
            plot_x, plot_y, plot_w, plot_h = x0 + 27, y0 + 25, cell_w - 38, cell_h - 55
            if metric_i == 0:
                parts.append(f'<text x="{plot_x + plot_w/2:.1f}" y="{y0 + 15}" text-anchor="middle" class="panel">Grid {grid}</text>')
            warmup_x = plot_x + plot_w * 20 / 300
            parts.append(f'<rect x="{plot_x}" y="{plot_y}" width="{warmup_x-plot_x:.1f}" height="{plot_h}" fill="#eef1f4"/>')
            for frac in (0, .5, 1):
                gy = plot_y + plot_h * frac
                parts.append(f'<line x1="{plot_x}" y1="{gy:.1f}" x2="{plot_x+plot_w}" y2="{gy:.1f}" stroke="#dce2e8" stroke-width="1"/>')
                val = ymax - (ymax-ymin)*frac
                fmt = f"{val:.0f}" if ymax-ymin >= 15 else f"{val:.2f}" if ymax <= 2.1 else f"{val:.1f}"
                parts.append(f'<text x="{plot_x-4}" y="{gy+3:.1f}" text-anchor="end" class="tick">{fmt}</text>')
            parts.append(f'<line x1="{warmup_x:.1f}" y1="{plot_y}" x2="{warmup_x:.1f}" y2="{plot_y+plot_h}" stroke="#8b949e" stroke-dasharray="3,3"/>')
            values = series[(policy, grid, key)]
            d = path_for(values, plot_x, plot_y, plot_w, plot_h, ymin, ymax)
            if d:
                parts.append(f'<path d="{d}" fill="none" stroke="{GRID_COLORS[grid]}" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/>')
            if metric_i == len(METRICS)-1:
                for tick in x_ticks:
                    x = plot_x + plot_w * tick / 59
                    parts.append(f'<line x1="{x:.1f}" y1="{plot_y+plot_h}" x2="{x:.1f}" y2="{plot_y+plot_h+3}" stroke="#536273"/>')
                    labelx = x
                    anchor = "middle"
                    if tick == 0:
                        anchor = "start"
                    elif tick == 59:
                        anchor = "end"
                    parts.append(f'<text x="{labelx:.1f}" y="{plot_y+plot_h+16}" text-anchor="{anchor}" class="tick">{clock_label(300+tick*5)}</text>')
            parts.append(f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" stroke="#536273"/>')
            parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y+plot_h}" stroke="#536273"/>')
    parts.append(f'<text x="{left + cell_w*3}" y="{height-13}" text-anchor="middle" class="note">Simulation time (HH:MM); dashed line marks end of common 20-min warm-up</text>')
    parts.append('</svg>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def draw_actions(actions: dict, action_ranges: dict[str, tuple[float, float]], out_path: Path) -> None:
    left, top, cell_w, cell_h = 153, 105, 545, 157
    width, height = left + cell_w * 2 + 45, top + cell_h * 6 + 58
    parts = svg_start(width, height, "LSTM policy: per-grid action selection over simulation time",
                      "10 seeds; solid line = mean action, shaded band = interquartile range; actions are discrete, so means may not be selectable values")
    columns = [("price_multiplier", "Price multiplier", PRICE_COLOR),
               ("matching_radius_km", "Matching radius (km)", RADIUS_COLOR)]
    ticks = [0, 12, 24, 36, 48, 59]
    for grid in range(6):
        y0 = top + grid * cell_h
        for col, (key, label, color) in enumerate(columns):
            x0 = left + col * cell_w
            plot_x, plot_y, plot_w, plot_h = x0 + 47, y0 + 25, cell_w - 62, cell_h - 54
            if grid == 0:
                parts.append(f'<text x="{plot_x+plot_w/2:.1f}" y="{y0+13}" text-anchor="middle" class="panel">{label}</text>')
            else:
                pass
            if col == 0:
                parts.append(f'<text x="{left-12}" y="{plot_y+plot_h/2:.1f}" text-anchor="end" class="rowlabel">Grid {grid}</text>')
            ymin, ymax = action_ranges[key]
            warmup_x = plot_x + plot_w * 20/300
            parts.append(f'<rect x="{plot_x}" y="{plot_y}" width="{warmup_x-plot_x:.1f}" height="{plot_h}" fill="#eef1f4"/>')
            for frac in (0, .5, 1):
                gy = plot_y + plot_h * frac
                parts.append(f'<line x1="{plot_x}" y1="{gy:.1f}" x2="{plot_x+plot_w}" y2="{gy:.1f}" stroke="#dce2e8"/>')
                val = ymax - (ymax-ymin)*frac
                parts.append(f'<text x="{plot_x-5}" y="{gy+3:.1f}" text-anchor="end" class="tick">{val:.2f}</text>')
            data = [actions[(grid, step, key)] for step in range(60)]
            xs = [plot_x + plot_w * i / 59 for i in range(60)]
            ys_lo = [plot_y + plot_h - (d[0]-ymin)/(ymax-ymin)*plot_h for d in data]
            ys_hi = [plot_y + plot_h - (d[2]-ymin)/(ymax-ymin)*plot_h for d in data]
            upper = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_hi)]
            lower = [f"{x:.1f},{y:.1f}" for x, y in reversed(list(zip(xs, ys_lo)))]
            parts.append(f'<polygon points="{" ".join(upper+lower)}" fill="{color}" fill-opacity="0.18" stroke="none"/>')
            mean_vals = [d[1] for d in data]
            dpath = path_for(mean_vals, plot_x, plot_y, plot_w, plot_h, ymin, ymax, step=True)
            parts.append(f'<path d="{dpath}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>')
            parts.append(f'<line x1="{warmup_x:.1f}" y1="{plot_y}" x2="{warmup_x:.1f}" y2="{plot_y+plot_h}" stroke="#8b949e" stroke-dasharray="3,3"/>')
            parts.append(f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" stroke="#536273"/>')
            parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y+plot_h}" stroke="#536273"/>')
            if grid == 5:
                for tick in ticks:
                    x = plot_x + plot_w * tick / 59
                    anchor = "start" if tick == 0 else "end" if tick == 59 else "middle"
                    parts.append(f'<text x="{x:.1f}" y="{plot_y+plot_h+16}" text-anchor="{anchor}" class="tick">{clock_label(300+tick*5)}</text>')
    parts.append(f'<text x="{left + cell_w}" y="{height-12}" text-anchor="middle" class="note">Simulation time (HH:MM); dashed line marks end of common warm-up</text>')
    parts.append('</svg>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    seeds = manifest["seeds"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    action_grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    expected_rows = 60 * 6
    for seed in seeds:
        for policy in ("fixed_3", "lstm"):
            folder = RUN_DIR / "runs" / f"seed{seed}" / policy
            rows = read_rows(folder / "decision_metrics.csv.gz")
            if len(rows) != expected_rows:
                raise ValueError(f"Unexpected decision row count: {folder}")
            for row in rows:
                grouped[(policy, int(row["grid_id"]), int(row["decision_step"]))].append(row)
        rows = read_rows(RUN_DIR / "runs" / f"seed{seed}" / "lstm" / "action_plan.csv.gz")
        if len(rows) != expected_rows:
            raise ValueError(f"Unexpected action row count for seed {seed}")
        for row in rows:
            step, grid = int(row["decision_step"]), int(row["grid_id"])
            action_grouped[(grid, step, "price_multiplier")].append(number(row, "price_multiplier"))
            action_grouped[(grid, step, "matching_radius_km")].append(number(row, "matching_radius_km"))

    series: dict[tuple[str, int, str], list[float | None]] = {}
    for policy in ("fixed_3", "lstm"):
        for grid in range(6):
            for key, _label, kind in METRICS:
                values: list[float | None] = []
                for step in range(60):
                    rows = grouped[(policy, grid, step)]
                    if kind in ("count", "revenue"):
                        value = sum(number(r, key) for r in rows) / len(rows)
                    elif kind == "rate":
                        if key == "request_rate":
                            num, den = "submitted_request_num", "baseline_request_num"
                        elif key == "driver_accept_rate":
                            num, den = "driver_accept_num", "driver_offer_num"
                        else:
                            num, den = "matched_request_num", None
                        numerator = sum(number(r, num) for r in rows)
                        # candidate_request_num is stored as a per-minute mean in
                        # decision_metrics, unlike the summed event counters.
                        denominator = (sum(number(r, "candidate_request_num") * number(r, "interval_minutes")
                                           + number(r, "unmatched_cancel_num") for r in rows)
                                       if den is None else sum(number(r, den) for r in rows))
                        value = numerator / denominator if denominator else None
                    else:
                        matched = sum(number(r, "matched_request_num") for r in rows)
                        value = (sum(number(r, key) * number(r, "matched_request_num") for r in rows) / matched
                                 if matched else None)
                    values.append(value)
                series[(policy, grid, key)] = values

    metric_ranges = {}
    for key, _label, _kind in METRICS:
        numeric = [v for (p, g, k), vals in series.items() if k == key for v in vals if v is not None]
        low, high = min(numeric), max(numeric)
        if high == low:
            pad = max(0.1, abs(high) * 0.1)
        else:
            pad = (high-low) * 0.08
        if key in ("request_rate", "driver_accept_rate", "order_completion_rate"):
            metric_ranges[key] = (max(0.0, low-pad), min(1.0, high+pad))
        else:
            metric_ranges[key] = (max(0.0, low-pad), high+pad)

    draw_metrics("fixed_3", "Fixed policy F3: per-grid operational metrics",
                 "F3 = price multiplier 1.6, matching radius 2.0 km; metric aggregation across 10 seeds",
                 series, metric_ranges, OUT_DIR / "fixed_f3_grid_metrics.svg")
    draw_metrics("lstm", "LSTM policy: per-grid operational metrics",
                 "Metric aggregation across the same 10 seeds; common 20-min warm-up uses multiplier 1.5 and radius 3.0 km",
                 series, metric_ranges, OUT_DIR / "lstm_grid_metrics.svg")

    actions = {}
    for grid in range(6):
        for step in range(60):
            for key in ("price_multiplier", "matching_radius_km"):
                vals = action_grouped[(grid, step, key)]
                actions[(grid, step, key)] = (quantile(vals, .25), sum(vals)/len(vals), quantile(vals, .75))
    action_ranges = {}
    for key in ("price_multiplier", "matching_radius_km"):
        vals = [v for (g, s, k), values in action_grouped.items() if k == key for v in values]
        action_ranges[key] = (min(vals)-0.1, max(vals)+0.1)
    draw_actions(actions, action_ranges, OUT_DIR / "lstm_grid_actions.svg")

    (OUT_DIR / "figure_notes.txt").write_text(
        "Small-density scenario: 6 grids, 250 drivers, 10 paired seeds, 60 five-minute decision intervals (05:00–10:00).\n"
        "F3 is (price multiplier=1.6, matching radius=2.0 km); both policies share the first 20-minute warm-up (1.5, 3.0 km).\n"
        "Request rate = submitted / baseline requests; driver acceptance = accepted / offers; completion = retained matches / (candidate attempts + unmatched cancellations). candidate_request_num is a per-minute mean in decision_metrics, so it is multiplied by interval length before pooling counts across seeds.\n"
        "Wait is matched-count-weighted across seeds; bubbles and revenue are arithmetic means across seeds. Wait is omitted where there were no matches.\n"
        "Action chart shows mean discrete action and 25th–75th percentile across seeds; the mean may lie between available action candidates.\n",
        encoding="utf-8")
    print(f"Generated 3 SVG figures in {OUT_DIR}")
    for path in sorted(OUT_DIR.glob("*.svg")):
        print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
