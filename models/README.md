# 模型目录

训练完成后，这里只放**板端推理需要的模型**：

```text
models/
├─ fire_yolov8n.onnx     # PC 训练导出，CPU / 板端 ONNX 兜底
├─ fire_yolov8n.om       # 在板端用 ATC 转换出的昇腾模型（推荐）
├─ labels.txt            # 类别顺序，必须与训练时一致
└─ fire_yolov8n.meta.json # 导出脚本自动生成（输入尺寸/类别/哈希）
```

## 获取顺序

1. PC 训练：`python training/train.py`
2. 导出 ONNX：`python training/export_onnx.py --weights runs/fire/yolov8n_fire/weights/best.pt`
3. 拷贝到板端并转 OM：

```bash
scp models/fire_yolov8n.onnx root@192.168.137.100:/home/root/atlas_fire/models/
ssh root@192.168.137.100
cd /home/root/atlas_fire
bash deployment/convert_om.sh
```

4. 运行：`bash deployment/start.sh`

## 说明

- `.onnx` 和 `.om` 都不包含数据集，体积通常几 MB 到几十 MB；
- 如果板端 `ais_bench` 不可用，程序会自动回退到 ONNX Runtime CPU；
- 类别数必须与 `labels.txt` 行数一致；默认只有 `fire` 一类。

本目录初始没有模型权重。缺失时网页会显示“模型未就绪”，而不是直接崩溃。
