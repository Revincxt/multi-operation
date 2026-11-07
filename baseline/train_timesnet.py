"""训练联合动作条件下的 TimesNet 六指标基线。

使用方式:
    python3 -m baseline.train_timesnet --prepare-only --epochs 1000
    python3 -m baseline.train_timesnet --epochs 1000

与 LSTM 共用 AdamW + weight_decay=1e-3，结果按优化器分目录保存。
原始对照使用 --optimizer adam --weight-decay 0。
大数据默认每 65536 行分块打乱；--shuffle-block-size 0 使用全局随机打乱。
"""

from baseline.data import DEFAULT_CHUNK_SIZE, DEFAULT_INPUT
from baseline.training import train_baseline, training_parser


def train_timesnet(input_dir=DEFAULT_INPUT, output_dir=None,
                   epochs=50, batch_size=512, hidden_size=64, embedding_size=16,
                   num_layers=2, dropout=0.1, learning_rate=0.001,
                   seed=42, device="cpu", threads=4, workers=0, loss_weighting="wesm",
                   wesm_gamma=0.1, wesm_history_steps=10, wesm_update_interval=10,
                   top_k=3, d_ff=64, num_kernels=3,
                   chunk_size=DEFAULT_CHUNK_SIZE, prepare_only=False,
                   optimizer_name="adamw", weight_decay=1e-3, shuffle_block_size=65536,
                   prepared_cache=None):
    """沿用现有数据划分和 WESM，不更改仿真时段或标签定义。"""
    return train_baseline(
        "timesnet", {"hidden_size": hidden_size, "embedding_size": embedding_size,
                     "num_layers": num_layers, "dropout": dropout,
                     "top_k": top_k, "d_ff": d_ff, "num_kernels": num_kernels},
        input_dir, output_dir, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed, device=device, threads=threads,
        workers=workers,
        loss_weighting=loss_weighting, wesm_gamma=wesm_gamma,
        wesm_history_steps=wesm_history_steps, wesm_update_interval=wesm_update_interval,
        chunk_size=chunk_size, prepare_only=prepare_only,
        optimizer_name=optimizer_name, weight_decay=weight_decay,
        shuffle_block_size=shuffle_block_size, prepared_cache=prepared_cache)


def main():
    parser = training_parser("timesnet")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--num-kernels", type=int, default=3)
    train_timesnet(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
