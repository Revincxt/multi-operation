"""训练 LSTM 六指标基线，保留原有命令与函数入口。

使用方式:
    python3 -m baseline.train_lstm --prepare-only --epochs 1000
    python3 -m baseline.train_lstm --epochs 1000

默认 AdamW + weight_decay=1e-3，结果保存到 models/lstm_grid6_low_full_adamw。
原始对照使用 --optimizer adam --weight-decay 0，既有模型的加载方式不变。
大数据默认每 65536 行分块打乱；--shuffle-block-size 0 使用全局随机打乱。
"""

# 保留已有脚本使用的数据读取、评估入口；实现统一放在 training 中。
from baseline.data import DEFAULT_CHUNK_SIZE, DEFAULT_INPUT
from baseline.training import (
    _make_loader,
    _run_epoch,
    load_dataset,
    regression_metrics,
    train_baseline,
    training_parser,
)


def train_lstm(input_dir=DEFAULT_INPUT, output_dir=None,
               epochs=50, batch_size=512, hidden_size=64, embedding_size=16,
               num_layers=1, dropout=0.1, learning_rate=0.001,
               seed=42, device="cpu", threads=4, workers=0, loss_weighting="wesm",
               wesm_gamma=0.1, wesm_history_steps=10, wesm_update_interval=10,
               chunk_size=DEFAULT_CHUNK_SIZE, prepare_only=False,
               optimizer_name="adamw", weight_decay=1e-3, shuffle_block_size=65536,
               prepared_cache=None):
    """完整训练指定轮数，按验证集选择模型，再评估测试集。"""
    return train_baseline(
        "lstm", {"hidden_size": hidden_size, "embedding_size": embedding_size,
                 "num_layers": num_layers, "dropout": dropout},
        input_dir, output_dir, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed, device=device, threads=threads,
        workers=workers,
        loss_weighting=loss_weighting, wesm_gamma=wesm_gamma,
        wesm_history_steps=wesm_history_steps, wesm_update_interval=wesm_update_interval,
        chunk_size=chunk_size, prepare_only=prepare_only,
        optimizer_name=optimizer_name, weight_decay=weight_decay,
        shuffle_block_size=shuffle_block_size, prepared_cache=prepared_cache)


def main():
    train_lstm(**vars(training_parser("lstm").parse_args()))


if __name__ == "__main__":
    main()
