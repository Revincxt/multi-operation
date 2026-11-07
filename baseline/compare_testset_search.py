"""直接在冻结的监督测试状态上比较单轮贪心和 BCS，不运行仿真。

使用方式:
    python3 -m baseline.compare_testset_search --device mps

只读取测试缓存中的历史、上下文和网格编号；候选动作的真实反事实标签
不可由日志直接获得，因此所有搜索收益均为同一个模型的预测值。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from baseline.data import DEFAULT_COLLECTION, DEFAULT_MODEL, SequenceDataset
from baseline.lstm import LSTMPredictor
from baseline.sequence import TARGET_COLUMNS
from data_pipeline.collect_density import save_json, sha256


_ACTIVE_OUTPUT = None


class FrozenStateScorer:
    """候选动作不改变历史编码；仅复用 LSTM 编码，仍调用完整预测头。"""

    def __init__(self, predictor, prices, radii):
        self.predictor = predictor
        self.model = predictor.model
        self.processor = predictor.preprocessor
        self.device = predictor.device
        self.grid_count = len(self.processor.grid_ids)
        self.pairs = np.array([(p, r) for p in prices for r in radii], dtype=np.float32)
        self.candidate_count = len(self.pairs)
        columns = self.processor.context_columns
        expected = [name for grid in self.processor.grid_ids
                    for name in (f"joint_price_grid_{grid}", f"joint_radius_grid_{grid}")]
        self.joint_start = len(columns) - len(expected)
        if (columns[2:4] != ["price_multiplier", "matching_radius_km"]
                or columns[self.joint_start:] != expected):
            raise ValueError
        context = self.processor.scalers["context"]
        self.own_normalized = torch.as_tensor(
            (self.pairs - context["mean"][2:4]) / context["scale"][2:4], device=self.device)
        joint = np.stack([
            (self.pairs - context["mean"][self.joint_start+2*g:self.joint_start+2*g+2])
            / context["scale"][self.joint_start+2*g:self.joint_start+2*g+2]
            for g in range(self.grid_count)])
        self.joint_normalized = torch.as_tensor(joint, device=self.device)
        self.target_index = TARGET_COLUMNS.index("target_platform_revenue")
        self.revenue_mean = self.processor.scalers["targets"]["mean"][self.target_index]
        self.revenue_scale = self.processor.scalers["targets"]["scale"][self.target_index]

    def encode(self, history, grid_indices):
        shape = history.shape
        x = torch.as_tensor(history.reshape(-1, *shape[2:]), device=self.device)
        grids = torch.as_tensor(grid_indices.reshape(-1), device=self.device)
        hidden = self.model.encoder(x)[1][0][-1]
        return torch.cat([hidden, self.model.grid_embedding(grids)], dim=1).reshape(
            shape[0], shape[1], -1)

    def contexts(self, context, actions):
        count, candidates, grids = actions.shape
        ids = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        grid = torch.arange(grids, device=self.device)[None, None, :]
        joint = self.joint_normalized[grid, ids].reshape(count, candidates, 2*grids)
        result = context[:, None].expand(-1, candidates, -1, -1).clone()
        result[..., 2:4] = self.own_normalized[ids]
        result[..., self.joint_start:] = joint[:, :, None, :]
        return result

    def score(self, encoded, context, actions):
        count, candidates, grids = actions.shape
        current = self.contexts(context, actions)
        features = torch.cat([encoded[:, None].expand(-1, candidates, -1, -1), current], dim=-1)
        predictions = self.model.head(features.reshape(-1, features.shape[-1])).reshape(
            count, candidates, grids, -1)
        normalized = predictions[..., self.target_index].cpu().numpy()
        # 沿用推理接口的 float32 反标准化与 float64 全网格求和口径。
        revenue = normalized * self.revenue_scale + self.revenue_mean
        totals = revenue.sum(axis=2, dtype=np.float64)
        if not np.isfinite(totals).all():
            raise FloatingPointError
        return totals

    def initial_actions(self, history):
        stats = self.processor.scalers["history"]
        indices = [self.processor.history_features.index(name)
                   for name in ("price_multiplier", "matching_radius_km")]
        previous = history[:, :, -1, :][..., indices] * stats["scale"][indices] + stats["mean"][indices]
        distance = np.max(np.abs(previous[..., None, :] - self.pairs), axis=-1)
        action_ids = distance.argmin(axis=-1)
        if np.max(np.take_along_axis(distance, action_ids[..., None], axis=-1)) > 1e-4:
            raise ValueError
        return action_ids


def search_batch(scorer, encoded, context, initial, max_sweeps=5):
    """同一初始动作向量：第一轮为贪心，后续轮次仅更新尚未稳定的状态。"""
    if max_sweeps <= 0:
        raise ValueError
    actions = initial.copy()
    stable = np.zeros(len(actions), dtype=bool)
    sweeps = np.zeros(len(actions), dtype=np.uint8)
    first_seconds = extra_seconds = 0.
    greedy = None
    for sweep in range(max_sweeps):
        active = np.flatnonzero(~stable)
        if not len(active):
            break
        started = perf_counter()
        local = actions[active].copy()
        before = local.copy()
        prefix, states = encoded[active], context[active]
        for grid in range(scorer.grid_count):
            vectors = np.repeat(local[:, None, :], scorer.candidate_count, axis=1)
            vectors[:, :, grid] = np.arange(scorer.candidate_count)
            totals = scorer.score(prefix, states, vectors)
            tied = np.isclose(totals, totals.max(axis=1, keepdims=True), rtol=1e-10, atol=1e-10)
            chosen = tied.argmax(axis=1)
            keep = tied[np.arange(len(local)), local[:, grid]]
            local[:, grid] = np.where(keep, local[:, grid], chosen)
        actions[active] = local
        sweeps[active] += 1
        stable[active] = (local == before).all(axis=1)
        seconds = perf_counter() - started
        if sweep == 0:
            greedy, first_seconds = actions.copy(), seconds
        else:
            extra_seconds += seconds
    return greedy, actions, stable, sweeps, first_seconds, extra_seconds


def load_test_groups(dataset, processor):
    """仅读取行键以恢复六网格同一时刻的状态，不读取当前标签。"""
    keys = pd.read_csv(dataset.keys_path, usecols=["run_id", "grid_id", "decision_step"],
                       dtype={"run_id": "category", "grid_id": "int16", "decision_step": "int16"})
    grids = len(processor.grid_ids)
    runs = len(dataset.run_ids)
    periods = len(keys) // runs // grids
    if len(keys) != runs * grids * periods:
        raise ValueError
    shape = (runs, grids, periods)
    codes = keys.run_id.cat.codes.to_numpy().reshape(shape)
    if not np.all(codes == codes[:, :1, :1]):
        raise ValueError
    run_ids = keys.run_id.cat.categories.take(codes[:, 0, 0]).tolist()
    if run_ids != dataset.run_ids:
        raise ValueError
    steps = keys.decision_step.to_numpy().reshape(shape)
    decisions = steps[0, 0].astype(int)
    if (not np.all(steps == decisions[None, None, :])
            or not np.all(keys.grid_id.to_numpy().reshape(shape)
                          == np.array(processor.grid_ids)[None, :, None])):
        raise ValueError
    return run_ids, decisions


def bootstrap_summary(greedy, bcs, runs, periods):
    """整条测试轨迹重采样，保留同一 run 内各个决策步的相关性。"""
    gain = (bcs-greedy).reshape(runs, periods)
    trajectory = np.column_stack([greedy.reshape(runs, periods).mean(1),
                                  bcs.reshape(runs, periods).mean(1), gain.mean(1)])
    values = np.column_stack([trajectory, gain])
    bootstrap = np.empty((10000, values.shape[1]))
    rng = np.random.default_rng(42)
    for start in range(0, len(bootstrap), 128):
        count = min(128, len(bootstrap)-start)
        indices = rng.integers(runs, size=(count, runs))
        weights = np.stack([np.bincount(row, minlength=runs) for row in indices]).astype(float)
        bootstrap[start:start+count] = weights @ values / runs
    low, high = np.quantile(bootstrap, [.025, .975], axis=0)
    return trajectory, low, high


def plot_summary(output, summary, by_time, sweeps, stable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from baseline.plot_paper_figures import _font_family
    with plt.rc_context({"font.family": _font_family(), "font.size": 8.5,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6}):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.1))
        fig.subplots_adjust(left=.095, right=.98, bottom=.14, top=.8, wspace=.32, hspace=.52)
        colors = ("#0072B2", "#D55E00")
        ax=axes[0,0]
        for index, policy in enumerate(("greedy", "bcs")):
            mean=summary[f"mean_{policy}_predicted_revenue"]
            lo,hi=summary[f"{policy}_mean_ci95"]
            ax.bar(index,mean,color=colors[index],width=.55)
            ax.errorbar(index,mean,yerr=[[mean-lo],[hi-mean]],fmt="none",color="#222222",capsize=2)
            ax.annotate(f"{mean:.3f}",(index,hi),xytext=(0,5),textcoords="offset points",ha="center",fontsize=8)
        ax.set_xticks([0,1],["Greedy","BCS"])
        ax.set_ylim(0, max(summary['mean_greedy_predicted_revenue'],summary['mean_bcs_predicted_revenue'])*1.25)
        ax.set_ylabel("Predicted revenue (USD / 5 min)")
        ax.set_title("(a) Mean model score",loc="left",fontsize=9)
        ax=axes[0,1]
        x=by_time.decision_step.to_numpy()
        ax.plot(x,by_time.mean_predicted_gain,color=colors[1],linewidth=1)
        ax.fill_between(x,by_time.gain_ci95_low,by_time.gain_ci95_high,color=colors[1],alpha=.2)
        ax.axhline(0,color="#777777",linewidth=.7,linestyle="--")
        ax.set_xlabel("Decision step")
        ax.set_ylabel("Predicted gain: BCS − greedy (USD)")
        ax.set_title("(b) Predicted gain over time",loc="left",fontsize=9)
        ax=axes[1,0]
        counts=[summary[k] for k in ("improved_states","equivalent_states","lower_score_states")]
        ax.bar([0,1,2],np.array(counts)/1000,color=[colors[1],"#8894A0","#555555"],width=.6)
        for pos,count in enumerate(counts):
            ax.annotate(f"{count/summary['state_count']:.1%}",(pos,count/1000),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
        ax.set_xticks([0,1,2],["Higher","Equivalent","Lower"])
        ax.set_ylim(0,max(counts)/1000*1.2)
        ax.set_ylabel(r"States ($10^3$)")
        ax.set_title("(c) BCS score versus greedy",loc="left",fontsize=9)
        ax=axes[1,1]
        n=summary['max_sweeps'];bins=np.arange(1,n+1)
        good=np.array([np.sum((sweeps==step)&stable) for step in bins])/1000
        capped=np.array([np.sum((sweeps==step)&~stable) for step in bins])/1000
        ax.bar(bins,good,color=colors[0],label="Stable")
        ax.bar(bins,capped,bottom=good,color=colors[1],label="Budget reached")
        ax.set_xticks(bins);ax.set_xlabel("BCS sweeps");ax.set_ylabel(r"States ($10^3$)")
        ax.set_title("(d) Coordinate convergence",loc="left",fontsize=9)
        ax.legend(frameon=False,fontsize=7.5)
        for ax in axes.flat:
            ax.grid(axis="y",color="#E1E5E8",linewidth=.45);ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.tick_params(length=2.5,width=.55)
        fig.suptitle("Offline test-set search: model-predicted revenue",fontsize=11,y=.985)
        fig.text(.5,.936,f"{summary['trajectory_count']:,} trajectories  |  {summary['state_count']:,} frozen global states  |  Selected epoch {summary['best_epoch']}",ha="center",fontsize=8.4)
        fig.text(.5,.875,"Same histories, pre-action states and previous-action initialization; 110 candidates per grid",ha="center",fontsize=8)
        fig.text(.5,.045,r"Predicted gain $\uparrow$: higher is better  |  Intervals: 95% trajectory-cluster bootstrap",ha="center",fontsize=8)
        for ext in ("png","pdf","svg"):
            fig.savefig(output/f"testset_greedy_vs_bcs.{ext}",dpi=600,facecolor="white")
        plt.close(fig)


def main():
    global _ACTIVE_OUTPUT
    parser=argparse.ArgumentParser(description="在完整监督测试集上比较贪心和 BCS 的预测目标")
    parser.add_argument("--model",default=DEFAULT_MODEL)
    parser.add_argument("--collection-root",default=DEFAULT_COLLECTION)
    parser.add_argument("--output",default="experiments/lstm_mixed47k_testset_search_20261029")
    parser.add_argument("--device",choices=("cpu","mps"),default="mps")
    parser.add_argument("--batch-states",type=int,default=256)
    parser.add_argument("--max-sweeps",type=int,default=5)
    args=parser.parse_args()
    output=Path(args.output)
    if args.batch_states<=0 or args.max_sweeps<=0 or output.exists():
        raise ValueError
    torch.set_num_threads(2)
    model_dir=Path(args.model)
    config=json.loads((model_dir/"training_config.json").read_text())
    model_metrics=json.loads((model_dir/"metrics.json").read_text())
    cache=Path(config['data_cache'])
    metadata=json.loads((cache/"cache_manifest.json").read_text())
    if metadata['source_sha256']!=config['source_sha256']:
        raise ValueError
    predictor=LSTMPredictor.load(model_dir/"best_model.pt",args.device)
    if predictor.preprocessor.state_dict()!=metadata['preprocessor']:
        raise ValueError
    dataset=SequenceDataset(cache,"test",metadata,predictor.preprocessor)
    collection=Path(args.collection_root)/"collection_config.json"
    source=json.loads(collection.read_text())
    scorer=FrozenStateScorer(predictor,source['price_action_values'],source['matching_radius_values'])
    run_ids,decisions=load_test_groups(dataset,predictor.preprocessor)
    grids=scorer.grid_count;periods=len(decisions);total=len(run_ids)*periods
    output.mkdir(parents=True)
    report={"status":"running","scope":"model-predicted objective on fixed test states; no simulator outcomes",
            "checkpoint":str((model_dir/'best_model.pt').resolve()),"checkpoint_sha256":sha256(model_dir/'best_model.pt'),
            "dataset":config['input_dir'],"cache":str(cache),"test_grid_rows":len(dataset),
            "trajectory_count":len(run_ids),"state_count":total,"processed_states":0,
            "best_epoch":int(model_metrics['best_epoch']),"decision_steps":decisions.tolist(),
            "grid_count":grids,"max_sweeps":args.max_sweeps,"device":args.device,
            "batch_states":args.batch_states,"initialization":"recorded previous-period actions",
            "action_pairs":scorer.pairs.tolist(),"started_at":datetime.now(timezone.utc).isoformat(),
            "source_code_sha256":sha256(__file__)}
    save_json(output/'manifest.json',report)
    _ACTIVE_OUTPUT=output
    scores=np.empty((total,2));sweeps=np.empty(total,dtype=np.uint8);stable=np.empty(total,dtype=bool)
    selected=np.lib.format.open_memmap(output/'selected_action_ids.npy',mode='w+',dtype='int16',shape=(total,2,grids))
    timings={"encode_seconds":0.,"greedy_sweep_seconds":0.,"bcs_extra_sweep_seconds":0.,"final_score_seconds":0.}
    started=perf_counter()
    with (output/'state_results.csv').open('w',encoding='utf-8',newline='') as stream,torch.inference_mode():
        for offset in range(0,total,args.batch_states):
            ids=np.arange(offset,min(offset+args.batch_states,total));run_index=ids//periods;time_index=ids%periods
            rows=run_index[:,None]*(grids*periods)+np.arange(grids)[None,:]*periods+time_index[:,None]
            history=np.array(dataset.arrays['history'][rows]);context=torch.as_tensor(np.array(dataset.arrays['context'][rows]),device=predictor.device)
            grid_indices=np.array(dataset.arrays['grid_indices'][rows])
            t=perf_counter();encoded=scorer.encode(history,grid_indices)
            if args.device=='mps':torch.mps.synchronize()
            timings['encode_seconds']+=perf_counter()-t
            initial=scorer.initial_actions(history)
            greedy,bcs,converged,used,first,extra=search_batch(scorer,encoded,context,initial,args.max_sweeps)
            timings['greedy_sweep_seconds']+=first;timings['bcs_extra_sweep_seconds']+=extra
            t=perf_counter();values=scorer.score(encoded,context,np.stack([greedy,bcs],axis=1))
            timings['final_score_seconds']+=perf_counter()-t
            scores[ids]=values;stable[ids]=converged;sweeps[ids]=used
            selected[ids]=np.stack([greedy,bcs],axis=1)
            pd.DataFrame({"state_index":ids,"run_id":[run_ids[i] for i in run_index],
                          "decision_step":decisions[time_index],"greedy_predicted_revenue":values[:,0],
                          "bcs_predicted_revenue":values[:,1],"predicted_gain":values[:,1]-values[:,0],
                          "changed_grids":(greedy!=bcs).sum(1),"bcs_sweeps":used,"bcs_converged":converged}).to_csv(
                              stream,index=False,header=offset==0)
            if (offset//args.batch_states+1)%20==0 or offset+len(ids)==total:
                report.update(processed_states=int(offset+len(ids)),elapsed_seconds=perf_counter()-started)
                save_json(output/'manifest.json',report)
                print(f"已比较 {offset+len(ids):,}/{total:,} 个全网格测试状态",flush=True)
    selected.flush()
    trajectory,low,high=bootstrap_summary(scores[:,0],scores[:,1],len(run_ids),periods)
    gain=scores[:,1]-scores[:,0];tolerance=1e-4
    summary={"trajectory_count":len(run_ids),"state_count":total,"test_grid_rows":len(dataset),
             "best_epoch":int(model_metrics['best_epoch']),
             "mean_greedy_predicted_revenue":float(scores[:,0].mean()),"mean_bcs_predicted_revenue":float(scores[:,1].mean()),
             "mean_predicted_gain":float(gain.mean()),"relative_predicted_gain_percent":float(100*gain.mean()/scores[:,0].mean()),
             "greedy_mean_ci95":[float(low[0]),float(high[0])],"bcs_mean_ci95":[float(low[1]),float(high[1])],
             "gain_mean_ci95":[float(low[2]),float(high[2])],
             "improved_states":int((gain>tolerance).sum()),"equivalent_states":int((np.abs(gain)<=tolerance).sum()),
             "lower_score_states":int((gain< -tolerance).sum()),"score_tolerance_usd":tolerance,
             "coordinate_converged_states":int(stable.sum()),"mean_bcs_sweeps":float(sweeps.mean()),
             "max_sweeps":args.max_sweeps,"timings":timings,"elapsed_seconds":perf_counter()-started}
    by_time=pd.DataFrame({"decision_step":decisions,"mean_greedy_predicted_revenue":scores[:,0].reshape(-1,periods).mean(0),
                          "mean_bcs_predicted_revenue":scores[:,1].reshape(-1,periods).mean(0),"mean_predicted_gain":gain.reshape(-1,periods).mean(0),
                          "gain_ci95_low":low[3:],"gain_ci95_high":high[3:]})
    by_time.to_csv(output/'decision_step_summary.csv',index=False)
    pd.DataFrame(trajectory,columns=['greedy_mean_score','bcs_mean_score','mean_predicted_gain']).assign(run_id=run_ids).to_csv(output/'trajectory_summary.csv',index=False)
    save_json(output/'summary.json',summary)
    if sha256(model_dir/'best_model.pt')!=report['checkpoint_sha256']:
        raise ValueError
    plot_summary(output,summary,by_time,sweeps,stable)
    (output/'captions.txt').write_text(
        "Offline model-objective comparison over all 7048 held-out trajectories, comprising 394688 "
        "six-grid decision states and 2368128 grid rows. Both searches use identical historical "
        "sequences, pre-action states and recorded previous-period action initialization. Greedy "
        "uses one ordered grid sweep; BCS continues from that first sweep up to five total sweeps, "
        "retaining current actions on ties. Histories are encoded once per state and reused while "
        "the full prediction head evaluates candidate contexts. Candidate revenues are inverse-" 
        "standardized in float32 and summed across grids in float64. No current-period labels "
        "enter action selection. Reported revenues are model predictions for counterfactual actions, "
        "not actual simulator outcomes or closed-loop policy returns. Bands and intervals are "
        "pointwise 95% trajectory-cluster percentile bootstrap intervals (10000 resamples, RNG "
        "seed 42). Equivalent scores use absolute tolerance 0.0001 USD. Coordinate stability does "
        "not guarantee global optimality. Timing is for batched inference with shared historical "
        "encodings and first sweeps, rather than separately timed standalone policy runs.\n",encoding='utf-8')
    report.update(status='complete',processed_states=total,elapsed_seconds=perf_counter()-started,
                  completed_at=datetime.now(timezone.utc).isoformat(),summary=summary,
                  code_sha256=sha256(__file__))
    save_json(output/'manifest.json',report)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        if _ACTIVE_OUTPUT is not None:
            path=_ACTIVE_OUTPUT/'manifest.json'
            report=json.loads(path.read_text())
            report.update(status='interrupted_or_failed',error_type=type(error).__name__,error=str(error))
            save_json(path,report)
        raise
