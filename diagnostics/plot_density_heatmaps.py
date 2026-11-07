"""用已完成的动作扫描结果绘制六指标热力图，不重新运行仿真。

使用方式:
    python3 -m diagnostics.plot_density_heatmaps --input experiments/density_action_sweep_20261008

只读取完整候选网格的 screening 数据，不混入仅复测收益优选动作的 confirmation。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "baseline_request_num": ("冒泡请求数量", "= 外生输入，无优劣方向", "单"),
    "request_rate": ("订单呼出率", "↑ 服务指标越高越好", "%"),
    "driver_accept_rate": ("司机接单率", "↑ 越高越好", "%"),
    "order_completion_rate": ("订单完成率", "↑ 越高越好", "%"),
    "mean_wait_to_match_seconds": ("乘客平均等待时间", "↓ 越低越好；仅成功匹配乘客", "秒"),
    "platform_revenue": ("平台收益", "↑ 越高越好", "USD"),
}
SCENARIO_NAMES = {"low": "小规模", "medium": "中规模", "high": "大规模"}
KEYS = ["scenario", "price_multiplier", "matching_radius_km"]
NOTES = (
    "只展示完整候选筛选，不混入收益 shortlist 的复测；单种子热力图不代表稳定最优。",
    "冒泡数为分钟表已记录的原始需求事件；最后生成批次尚未进入下一分钟统计，所以略少于实际输入订单数。",
    "订单呼出率 = 呼出数 / 已记录冒泡数；观测需求重采样可放大需求，因此可能超过 100%，不是发单概率。",
    "订单完成率 = 最终匹配数 / (候选匹配次数 + 未匹配取消数)；重试重复计数，不是唯一订单的最终完成比例。",
    "等待时间按成功匹配数加权，只含匹配前排队时间，不含未服务乘客和接驾时间。",
    "平台收益为成功匹配订单车费的抽成，不是扣除成本后的利润。",
    "箭头表示单项服务方向，不意味着六指标可以同时最优；高完成率、低等待可能伴随价格抑制需求。",
    "横轴是决策用匹配半径，不是订单实际接驾距离；全城、全时段使用同一固定动作。",
)


def prepare_metrics(frame, config):
    """校验完整配对网格，保留原始比率与缺失值，再跨 seed 等权平均。"""
    if (frame.empty or set(frame.phase) != {"screening"}
            or frame.duplicated(KEYS + ["seed"]).any()):
        raise ValueError
    expected = pd.MultiIndex.from_product(
        [config[name] for name in ("scenarios", "prices", "radii", "screening_seeds")],
        names=KEYS + ["seed"])
    actual = pd.MultiIndex.from_frame(frame[KEYS + ["seed"]])
    if len(actual.difference(expected)) or len(expected.difference(actual)):
        raise ValueError
    for _, group in frame.groupby(["scenario", "seed"]):
        for column in ("sampled_orders_sha256", "initial_drivers_sha256",
                       "generated_baseline_requests", "baseline_request_num"):
            if group[column].isna().any() or group[column].nunique() != 1:
                raise ValueError
    # 使用累计计数的比率，而不是将各分钟的比率直接平均；不把呼出率截到 100%。
    for metric, numerator, denominator in (
        ("request_rate", "submitted_request_num", "baseline_request_num"),
        ("driver_accept_rate", "driver_accept_num", "driver_offer_num"),
        ("order_completion_rate", "matched_request_num", "completion_denominator"),
    ):
        values = frame[numerator] / frame[denominator].where(frame[denominator] > 0)
        np.testing.assert_allclose(frame[metric], values, equal_nan=True)
    np.testing.assert_allclose(frame.completion_denominator,
                               frame.candidate_request_num + frame.unmatched_cancel_num)
    if (np.isinf(frame[list(METRICS)].to_numpy()).any()
            or (frame[list(METRICS)] < 0).any().any()
            or frame.loc[frame.matched_request_num == 0, "mean_wait_to_match_seconds"].notna().any()):
        raise ValueError
    long = frame.melt(id_vars=KEYS + ["seed"], value_vars=list(METRICS), var_name="metric")
    return long.groupby(KEYS + ["metric"], sort=True)["value"].agg(
        value="mean", seed_std="std", seed_count="size", valid_seed_count="count").reset_index()


def color_limits(values, metric):
    """相同指标三种规模共用色标，避免颜色相同却代表不同数值。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    if metric in ("driver_accept_rate", "order_completion_rate"):
        return 0.0, 1.0
    lower = float(finite.min()) if metric == "mean_wait_to_match_seconds" else 0.0
    upper = max(float(finite.max()), 1.0)
    if metric == "request_rate":
        upper = np.ceil(upper * 10) / 10
    return lower, upper if upper > lower else lower + 1.0


def draw_heatmap(fig, ax, data, metric, prices, radii, limits, title):
    """离散动作格逐格显示真实数值，不插值；灰格表示未定义。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    values = data[data.metric == metric].pivot(
        index="price_multiplier", columns="matching_radius_km", values="value").reindex(
            index=prices, columns=radii).to_numpy()
    cmap = plt.get_cmap("viridis_r" if metric == "mean_wait_to_match_seconds" else "viridis").with_extremes(
        bad="#d1d5db")
    im = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto",
                   interpolation="nearest", cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set_xticks(range(len(radii)), [f"{v:g}" for v in radii])
    ax.set_yticks(range(len(prices)), [f"{v:g}" for v in prices])
    ax.set(xlabel="匹配半径（公里）", ylabel="价格乘子", title=title)
    for (row, column), value in np.ndenumerate(values):
        if np.isnan(value):
            label, color = "—", "#374151"
        else:
            label = (f"{value:.1%}" if metric.endswith("rate") else
                     f"{value:.1f}" if metric == "mean_wait_to_match_seconds" else f"{value:,.0f}")
            red, green, blue, _ = im.cmap(im.norm(value))
            color = "#111827" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55 else "white"
        ax.text(column, row, label, ha="center", va="center", fontsize=8, color=color)
    if 1.5 in prices and 3.0 in radii:
        ax.add_patch(Rectangle((radii.index(3.0) - 0.5, prices.index(1.5) - 0.5), 1, 1,
                               fill=False, edgecolor="#f97316", linewidth=2.2, linestyle="--"))
    colorbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label(METRICS[metric][2])
    if metric.endswith("rate"):
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    return im


def plot_heatmaps(input_dir, output=None):
    """另外保存图片及绘图口径，不改写仿真实验或其源码指纹。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    source = Path(input_dir)
    report = json.loads((source / "manifest.json").read_text())
    config = report["config"]
    data = prepare_metrics(pd.read_csv(source / "screening_runs.csv"), config)
    output = Path(output) if output else source / "six_metric_heatmaps"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "heatmap_values.csv", index=False)
    font_names = ["sans-serif"]
    for path in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            font_names.insert(0, font_manager.FontProperties(fname=path).get_name())
            break
    prices, radii = sorted(config["prices"]), sorted(config["radii"])
    limits = {metric: color_limits(data.loc[data.metric == metric, "value"], metric) for metric in METRICS}
    seed_count = len(config["screening_seeds"])
    start_hour, start_minute = map(int, config["start_time"].split(":"))
    end_minute = start_hour * 60 + start_minute + config["simulation_steps"]
    window = f"{config['start_time']}–{end_minute // 60:02d}:{end_minute % 60:02d}"
    subtitle = (f"{config['date']}  {window} | 6 网格 | 全城固定动作 | 完整筛选：每格 {seed_count} 个种子"
                "\n同一指标三规模共用色标；橙色虚线框：半径 3 公里 / 价格 1.5")
    footnote = ("冒泡数为已记录事件（末批次尚未入表）；呼出率可因需求重采样超过 100%；完成率分母包含重复匹配尝试。\n"
                "等待时间只统计成功匹配者，不含接驾；单指标改善不等于收益提高。"
                "灰格为未定义，不当作零；筛选图不代表多种子复测结论。")
    names = {scenario: f"{SCENARIO_NAMES[scenario]}｜约 {report['scenario_settings'][scenario]['actual_sampled_orders']:,} 单 / "
             f"{report['scenario_settings'][scenario]['driver_num']} 名司机" for scenario in config["scenarios"]}
    files = []
    with plt.rc_context({"font.family": font_names, "font.size": 11, "axes.unicode_minus": False}):
        for scenario in config["scenarios"]:
            fig, axes = plt.subplots(2, 3, figsize=(19, 15))
            for ax, (metric, (name, direction, _)) in zip(axes.flat, METRICS.items()):
                draw_heatmap(fig, ax, data[data.scenario == scenario], metric, prices, radii,
                             limits[metric], f"{name}\n{direction}")
            fig.suptitle(f"{names[scenario]}：六指标动作热力图\n{subtitle}", fontsize=16, y=0.985)
            fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=10)
            fig.subplots_adjust(left=0.045, right=0.945, bottom=0.08, top=0.88, hspace=0.3, wspace=0.3)
            filename = f"{scenario}_six_metrics.png"
            fig.savefig(output / filename, dpi=180, bbox_inches="tight")
            plt.close(fig)
            files.append(filename)
        for metric, (name, direction, _) in METRICS.items():
            fig, axes = plt.subplots(1, len(names), figsize=(6.4 * len(names), 8.8), squeeze=False)
            for ax, scenario in zip(axes.flat, names):
                draw_heatmap(fig, ax, data[data.scenario == scenario], metric, prices, radii,
                             limits[metric], names[scenario])
            fig.suptitle(f"{name}｜{direction}\n{subtitle}", fontsize=15, y=0.98)
            fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=9)
            fig.subplots_adjust(left=0.045, right=0.945, bottom=0.17, top=0.80, wspace=0.3)
            filename = f"{metric}_comparison.png"
            fig.savefig(output / filename, dpi=180, bbox_inches="tight")
            plt.close(fig)
            files.append(filename)
    manifest = {
        "source": str(source.resolve()), "phase": "screening", "seeds": config["screening_seeds"],
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("screening_runs.csv", "manifest.json")},
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {key: {"title": value[0], "direction": value[1], "unit": value[2]}
                    for key, value in METRICS.items()},
        "color_limits_shared_across_scenarios": limits, "notes": NOTES,
        "files": files + ["heatmap_values.csv"],
    }
    (output / "heatmap_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="已有动作扫描的六指标热力图")
    parser.add_argument("--input", dest="input_dir", default="experiments/density_action_sweep_20261008")
    parser.add_argument("--output")
    print(plot_heatmaps(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
