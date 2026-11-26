# DL-MRD-JF 权重

best_model.pt 是 DL-MRD 联合融合版的验证集最优检查点。模型、预处理参数和六项输出头均保存在权重中，可直接加载：

    from baseline.dlmrd.model import DLMRDPredictor

    model = DLMRDPredictor.load("checkpoints/dlmrd/best_model.pt", device="cpu")
    predictions = model.predict(features)

该模型从随机初始化训练100轮，按验证集等权标准化误差选中第33轮。训练配置、测试结果、曲线和闭环评估见 [实验结果](../../results/dlmrd_joint_fusion_20261126/README.md)。不上传训练缓存、原始轨迹或优化器断点。
