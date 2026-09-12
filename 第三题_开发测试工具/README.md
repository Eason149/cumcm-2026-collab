# 第三题开发与模拟器测试工具

本目录保存官方模拟器入口、客户端适配器、回归测试和运行日志；正式算法从相邻的 `D:\Cumcm\第三题最终版\q3_algorithm` 加载。

只检查依赖与本机端口，不调用 `/enter`：

```powershell
python .\官方模拟器接口.py --check-only --probe-port
```

正式测试会调用 `/enter`，可能消耗一次演练或正式测试机会：

```powershell
python .\官方模拟器接口.py --robot-id "你的队号" --confirm-official-run
```

运行结果保存在 `logs` 的独立时间戳目录中。
