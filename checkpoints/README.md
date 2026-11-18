# 已发布模型

[DL-MRD权重与加载说明](dlmrd/README.md)包含第58轮验证最优模型及第799轮完整续训断点。两份权重与 `ZHYXNJD/refactor_simulator` 分支中发布的文件逐字节一致，SHA-256见[manifest.json](manifest.json)。

训练目标为1000轮，在完成799轮后因数值异常中断，完整测试集尚未评估。本次同步不启动或恢复训练。
