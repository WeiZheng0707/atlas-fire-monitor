# 演示/测试图片（不是训练数据集）

这里的 4 张图片只用于：

- 没有摄像头时预览网页；
- `tests/` 中的接口/界面测试；
- `training/benchmark.py` 的单图 ONNX 基准。

**不要把它们当作训练集**，也不要把 `samples/` 误认为 `dataset/`。
板端打包默认不包含该目录（`--with-samples` 仅用于 PC 联调）。

- `fire_phone.jpg`：课程 PPT 中的手机火焰实拍
- `fire_synthetic.jpg`：程序化生成火焰图
- `normal_room.jpg`：无火房间
- `normal_warm_lamp.jpg`：暖色灯负样本
