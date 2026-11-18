# DL-MRD 权重

六网格联合价格／半径、六指标预测适配版，使用47,000条完整轨迹。

- `best_model.pt`：验证集选中的第58轮模型，Min-Max MSE为0.02888919，包含网络配置及训练集Min-Max参数。
- `latest.pt`：第799轮完整断点，包含模型、Adam、WESM、799轮历史及随机状态。它不是验证最优模型。
- `training_config.json`、`normalization.json`：原样保存的训练配置与归一化参数，供来源核验和恢复。
- `training_status.json`：目标1000轮，已完成799轮，第800轮因`FloatingPointError`中断；完整测试集尚未评估。

```python
from baseline.dlmrd.model import DLMRDPredictor

model = DLMRDPredictor.load("checkpoints/dlmrd/best_model.pt", device="cpu")
predictions = model.predict(features)
```

`features`包含四步历史、当前可观测状态及联合候选动作；预测不需要当前标签。模型结构、输入列和Min-Max参数已包含在最佳权重中，加载不需要训练数据。训练缓存与仿真输入仍需自行准备。

`latest.pt`可恢复已保存的训练状态，但恢复仍需相同的数据缓存和匹配的冻结配置／源码；配置中的本地路径是原实验来源记录。已确认断点中的数值有限，不保证恢复后自动解决第800轮的异常。本次发布未重新训练，也没有使用测试集选择权重。文件SHA-256见上级[manifest.json](../manifest.json)。
