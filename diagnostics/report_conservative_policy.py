"""汇总保守策略迭代试验的配对收益、初始化差异及研究限制。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.compare_policies import METRICS
from offline_rl.data import save_json, sha256
from offline_rl.optimize import paired_statistics
from offline_rl.train_conservative import ROOT, REFERENCES


def plot_training(root, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    joint = plan["algorithm"].get("policy_gap_limit") is not None
    metrics = ["td_loss", "cf_gap", "q_auxiliary_loss", "actor_kl"]+(["policy_q_gap", "joint_conservative_weight"] if joint else [])
    titles = ["Joint-Q TD loss", "Counterfactual Q gap", "Dynamics auxiliary loss", "Actor KL to behavior",
              "Policy-action Q gap (USD)", "Learned joint penalty weight"]
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8)) if joint else plt.subplots(1, 4, figsize=(12, 3.2))
        for seed, color in zip((72, 73, 74), ("#0072B2", "#D55E00", "#009E73")):
            frame = pd.read_csv(root/f"train_seed{seed}/training_history.csv")
            for ax, metric in zip(axes.flat, metrics):
                part = frame[frame.phase == "policy_improvement"] if metric == "actor_kl" else frame[frame.phase != "behavior_initialization"]
                ax.plot(part.step, part[metric]*(100 if metric == "policy_q_gap" else 1), color=color, lw=.8, label=f"Seed {seed}")
        for index, (ax, title) in enumerate(zip(axes.flat, titles)):
            ax.set_title(f"({chr(97+index)}) {title}", loc="left", fontsize=9)
            ax.set_xlabel("Updates per initialization")
            ax.grid(alpha=.2)
        if joint:
            axes.flat[4].axhline(plan["algorithm"]["policy_gap_limit"]*100, color=".4", lw=.8, ls="--", label="Gap target")
            axes.flat[4].legend(frameon=False, fontsize=7)
        axes.flat[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/f"training_curves.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def plot_results(root, frame, selected, statistics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    order = ["fixed_f3", "eq_joint", "joint_start", selected]
    labels = ["F3", "EQ-Joint", "STQ-JS", "Conservative AC"]
    colors = ["#888888", "#0072B2", "#D55E00", "#009E73"]
    titles = ("Bubble requests =", "Order request rate ↑", "Driver acceptance rate ↑",
              "Order completion rate ↑", "Matched-passenger wait (s) ↓", "Platform revenue (USD) ↑")
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
        for index, (ax, metric, title) in enumerate(zip(axes.flat, METRICS, titles)):
            values = [frame.loc[frame.policy == name, metric] for name in order]
            scale = 100 if metric in ("request_rate", "driver_accept_rate", "order_completion_rate") else 1
            ax.bar(np.arange(4), [x.mean()*scale for x in values], yerr=[x.std(ddof=1)*scale for x in values],
                   color=colors, width=.65, error_kw={"linewidth": .7, "capsize": 2})
            ax.set_title(f"({chr(97+index)}) {title}", loc="left", fontsize=10)
            ax.set_xticks(np.arange(4), labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            if scale == 100:
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.text(.5, .015, "Means ± SD over 100 paired test seeds; initialization selected using separate validation seeds.", ha="center", fontsize=8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/"test"/f"paper_comparison.{ext}", dpi=600, facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        seeds = [72, 73, 74]
        for offset, reference, label, color in ((-.15, "fixed_f3", "vs F3", "#D55E00"), (.15, "eq_joint", "vs EQ-Joint", "#0072B2")):
            rows = [statistics[f"ac_seed{seed}"][reference] for seed in seeds]
            means = np.array([row["mean_difference_usd"] for row in rows])
            bounds = np.array([[row["ci95_low"], row["ci95_high"]] for row in rows])
            ax.errorbar(np.arange(3)+offset, means, yerr=np.stack((means-bounds[:, 0], bounds[:, 1]-means)),
                        fmt="o", ms=4, capsize=4, color=color, label=label)
        ax.axhline(0, color=".4", lw=.8)
        ax.set_xticks(np.arange(3), [f"Initialization {seed}" for seed in seeds])
        ax.set_ylabel("Paired revenue difference (USD)")
        ax.set_title("Revenue gain ↑; paired 95% seed bootstrap intervals", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(root/"test"/f"paired_revenue_gain.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def run(root=ROOT):
    root = Path(root)
    status = json.loads((root/"status.json").read_text())
    assert status["status"] == "complete"
    selected = json.loads((root/"selection.json").read_text())["selected"]
    frame = pd.read_csv(root/"test/run_summary.csv")
    validation = pd.read_csv(root/"validation/run_summary.csv")
    plan = json.loads((root/"protocol.json").read_text())
    assert not frame.duplicated(["seed", "policy"]).any()
    assert set(frame.seed) == set(plan["blocks"]["test"]["seeds"])
    assert not set(frame.seed).intersection(validation.seed)
    assert selected == validation[validation.policy.str.startswith("ac_seed")].groupby("policy").platform_revenue.mean().idxmax()
    statistics = {f"ac_seed{seed}": {reference: paired_statistics(frame, f"ac_seed{seed}", reference)
                  for reference in REFERENCES} for seed in (72, 73, 74)}
    save_json(root/"test/all_initializations.json", statistics)
    plot_training(root, plan)
    plot_results(root, frame, selected, statistics)
    means = frame.groupby("policy").platform_revenue.mean()
    gate = all(statistics[selected][reference]["ci95_low"] > 0 for reference in ("fixed_f3", "eq_joint"))
    save_json(root/"adoption_test_gate.json", {"selected": selected, "passed": gate,
        "criterion": "positive paired 95% CI versus both F3 and EQ-Joint; independent confirmation still required"})
    checks = json.loads((root/"verification.json").read_text())
    checks.update({"training_source_snapshot_verified": all(sha256(root/"code_snapshot"/path) == value for path, value in plan["source_sha256"].items()),
        "initial_critic_checkpoints_unchanged": all(sha256(path) == value for path, value in plan["critic_initial_sha256"].items()),
        "validation_test_disjoint_and_locked": True, "validation_selection_reproduced": True})
    for seed in (72, 73, 74):
        directory = root/f"train_seed{seed}"
        record = json.loads((directory/"complete.json").read_text())
        checks[f"policy_seed{seed}_unchanged"] = sha256(directory/"policy.pt") == record["checkpoint_sha256"]
    for column in ("initial_driver_sha256", "warmup_metrics_sha256", "sampled_orders_sha256", "generated_baseline_requests"):
        checks[f"paired_{column}"] = bool(frame.groupby("seed")[column].nunique().eq(1).all())
    assert all(checks.values())
    save_json(root/"verification.json", checks)
    lines = ["保守actor-critic的算法改进及配对试验", "",
        "本轮更改：当前目标策略联合MAP构造TD；训练期反事实保守Q正则；actor与Q物理动态辅助任务；以全城Q作逐区域改进；末尾固定目标actor再做价值评估。",
        "保留六网格、原全部110个单网格动作及四步历史；没有固定F3覆盖、禁止半径或后处理搜索。actor为四步LSTM48、网格注意力及价格条件半径双头。",
        "这是CFCQL式适配，未实现长期递归burn-in或自回归跨网格行为先验；共享actor编码器使单区域目标更新也会影响其他区域，不能引用严格InSPO单调改进保证。熵/KL仅用于actor投影，TD评估确定性佣金回报。", "",
        "Q从三个独立STQ训练成员初始化并继续更新，actor重新训练。训练集70,000条轨迹、3,920,000条城市转移；全部训练转移可抽样，非逐条整遍训练。",
        "三组72/73/74，每组1000行为初始化+2000初始策略评估+6000联合学习+1000固定最终策略评估；batch128、MPS，Q学习率1e-4，actor3e-4余弦降至3e-5，gamma1，奖励/100。",
        "反事实系数0.05、温度0.1；行为KL系数0.0001、投影熵0.002、辅助权重0.02。每批枚举一个轮换网格的110个动作，保留记录中的队友动作；每6次更新随机排列六网格顺序。没有为本轮结果调参。", ""]
    for seed in (72, 73, 74):
        record = json.loads((root/f"train_seed{seed}/complete.json").read_text())
        lines.append(f"初始化{seed}：{record['elapsed_seconds']:.1f}秒，访问{record['training_trajectories_visited']}条训练轨迹、{record['training_transitions_visited']}条不同转移；actor {record['actor_parameters']}参数，Q {record['critic_parameters']}参数。")
    limit = plan["algorithm"].get("policy_gap_limit")
    if limit is not None:
        lines += ["", f"联合策略保守项：当前/目标actor的实际联合MAP动作均纳入Q正则，限制其相对记录动作的平均估值差，目标为{limit*100:.2f}美元；非负系数从0.05开始，按拉格朗日上升学习，学习率0.003。该阈值是价值正则的研究超参数，不是平台收益保证或动作禁令。",
            "后续训练设计依据前一轮开发验证结果；开发验证池可重复用于方法设计，最终测试重新抽取未使用种子。没有根据前一轮测试调参。"]
    else:
        lines += ["", "本轮仅使用逐网格反事实正则，没有额外联合策略动作保守项；后续版本的code_snapshot与本轮分别保存。"]
    lines += ["", "6网格、250司机、同一2015-05-18订单日期，05:00—10:00，统一热身20分钟，05:20—10:00评分。F3价格1.6、半径2公里。",
        "30个新开发验证种子选择初始化，锁定为"+selected+"；100个新配对测试种子未参与选择。所有六种策略在每个种子上独立仿真，共享外生订单、司机初态和预热。动作后随机调用可能分岔。",
        "配对bootstrap区间只描述给定模型的环境种子变异，不是多个训练初始化总体的置信区间。相同日期的重复种子不能证明跨日期泛化。", "", "测试平台收益均值（USD）："]
    lines += [f"{name}: {value:.2f}" for name, value in means.items()]
    lines += ["", "验证选定模型的配对收益差："]
    for reference, row in statistics[selected].items():
        lines.append(f"{selected} − {reference}: {row['mean_difference_usd']:+.2f}USD ({row['relative_percent']:+.2f}%), CI[{row['ci95_low']:+.2f},{row['ci95_high']:+.2f}], {row['wins']}/100胜")
    lines += ["", f"通过100种子采用初筛：{gate}。采用标准还包含另50种子确认；本统计脚本不改默认策略。",
        "训练TD误差和辅助损失不等于真实动作排序准确度。没有单项消融，不能把总体结果唯一归因于某个损失或结构。",
        "订单完成率分母含跨周期重复候选订单；等待仅统计已匹配乘客排队，不等于所有乘客总等待。",
        "代码：offline_rl/conservative_policy.py、offline_rl/train_conservative.py；复现/续训使用本轮code_snapshot的源码和原参数加--resume，当前源码可能已进入下一版本。模型完整断点包含网络、目标网络、优化器、批次与网格顺序RNG。",
        "文献：CQL https://arxiv.org/abs/2006.04779；CFCQL https://arxiv.org/abs/2309.12696；InSPO https://arxiv.org/abs/2412.07639。文献支持设计动机，不为本适配提供直接性能保证。"]
    if (root/"additional_control_comparison.json").exists():
        row = json.loads((root/"additional_control_comparison.json").read_text())
        lines += ["", "预先登记的额外对照（同一100测试种子）：",
            f"相对首轮验证选优actor-critic：{row['mean_difference_usd']:+.2f}美元，{row['relative_percent']:+.2f}%，CI[{row['ci95_low']:+.2f},{row['ci95_high']:+.2f}]，{row['wins']}/100胜。",
            "双方各使用自己验证选中的初始化，这是完整训练/选优流程对照，不能把差值全部解释为固定同一初始化的单组件因果效应。"]
    if (root/"memory_diagnostic.json").exists():
        memory = json.loads((root/"memory_diagnostic.json").read_text())["results"]
        lines += ["", "开发诊断集的历史半径敏感性（不是独立泛化测试，也不用于初始化选择）："]
        for name, row in memory.items():
            lines.append(f"{name}: 36对输入中{row['changed_joint_actions']}对改变联合动作，216个网格动作中{row['changed_grid_actions']}个改变。")
        lines.append("低敏感性本身不能证明控制更好，常数策略也可以零敏感；应结合真实收益判断。")
    if (root/"offline_validation.json").exists():
        lines += ["", "同一2048条验证转移上的诊断："]
        for row in json.loads((root/"offline_validation.json").read_text()):
            lines.append(f"初始化{row['seed']}: TD MAE {row['validation_td_mae_usd']:.2f}美元，部署动作相对记录动作的预测Q差{row['policy_minus_logged_q_usd']:+.2f}美元。")
        lines.append("预测Q差不是实测高估误差；该保守约束也不保证每个状态满足阈值。")
    if (root/"adoption.json").exists():
        adoption = json.loads((root/"adoption.json").read_text())
        lines += ["", "独立50种子确认（固定模型、固定样本量、无可选停止）："]
        for row in adoption["confirmation"]:
            lines.append(f"相对{row['reference']}: {row['mean_difference_usd']:+.2f}美元，{row['relative_percent']:+.2f}%，CI[{row['ci95_low']:+.2f},{row['ci95_high']:+.2f}]，{row['wins']}/50胜。")
        lines.append(f"最终采用：{adoption['adopted']}。收益优势仍只针对当前日期和低密度场景。")
        lines.append("其他初始化仍较弱，当前结果不等于跨初始化稳定优势；比较对象为预先指定F3，尚未验证优于经验证选择的最优固定动作。")
    if (root/"combined_150_run_summary.csv").exists():
        combined = pd.read_csv(root/"combined_150_run_summary.csv")
        means = combined.groupby("policy")[["platform_revenue", "request_rate", "driver_accept_rate", "order_completion_rate", "mean_wait_to_match_seconds", "matched_request_num", "pickup_cancel_num"]].mean()
        lines += ["", "150种子合并描述性汇总（采用门槛仍分别检查100与50两个预定区块）：", means.to_string(),
            "请求率及成功匹配总数有下降；本方法优化平台佣金，不能把口径中的完成率上升解释为更多原始订单得到服务。"]
    (root/"experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": selected, "test_gate_passed": gate, "means": means.to_dict(), "comparisons": statistics[selected]}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    run(parser.parse_args().root)
