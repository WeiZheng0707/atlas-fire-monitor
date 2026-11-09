# 板端部署脚本

| 文件 | 用途 |
|---|---|
| `convert_om.sh` | 在板端调用 ATC，把 ONNX 转为 Ascend310B4 的 OM |
| `install_board.sh` | 安装板端运行依赖，不安装训练框架 |
| `start.sh` | 启动 Web 服务（自动 source CANN 环境） |
| `package_for_board.py` | 在 PC 上打板端包，排除数据集/训练代码 |
| `atlas-fire.service` | systemd 开机自启模板 |

最短流程：

```bash
# PC
python training/train.py
python training/export_onnx.py --weights runs/fire/yolov8n_fire/weights/best.pt
python deployment/package_for_board.py --model models/fire_yolov8n.onnx

# 板端
scp dist/atlas_fire_board_*.tar.gz root@192.168.137.100:/home/root/
ssh root@192.168.137.100
cd /home/root && tar -xzf atlas_fire_board_*.tar.gz && cd atlas_fire
bash deployment/install_board.sh
bash deployment/convert_om.sh
bash deployment/start.sh
```

完整说明（含 Type-C/网口 IP、root 密码、CANN、FAQ）见
[`../docs/板端部署.md`](../docs/板端部署.md)。

> 板端不接收 `dataset/`；打包脚本会检查并排除它。
