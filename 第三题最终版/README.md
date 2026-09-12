# 第三题最终版算法代码

本目录是论文代码附件使用的精简版本，只保留第三题 v5 算法与运行所需依赖说明，不包含官方模拟器入口、测试脚本或运行日志。

## 目录结构

```text
第三题最终版/
├── q3_algorithm/
│   ├── __init__.py          # 对外公开入口
│   ├── config.py            # 题设常量和采用参数
│   ├── models.py            # 在线状态数据结构
│   ├── geometry_kernel.py   # 完整保守几何内核
│   ├── geometry_backend.py  # 几何内核适配层
│   ├── geometry.py          # 扇区和定位点几何
│   ├── route.py             # 开放路线与搜索站优化
│   ├── planner.py           # 搜索—清除联合规划
│   ├── actions.py           # 测量、扫描和清除状态更新
│   ├── solver.py            # v5 顶层闭环
│   └── result.py            # 结果组装
├── requirements.txt
└── README.md
```

所有算法 Python 文件均位于 `q3_algorithm` 内部，包内不再引用外部的 `q3_optimizer_v4.py`。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 调用方法

```python
from q3_algorithm import solve_optimized_v5

result = solve_optimized_v5(client)
```

`client` 需要提供与题目运行环境对应的以下接口：

```python
client.position
client.virtual
client.rows
client.act("/measure", position, channel)
client.act("/clear", position, channel)
```

算法不会读取源的真实位置等仿真内部信息。采用版本为 `q3_joint_search_clear_route_v5`。

## 开发与测试材料

官方模拟器连接器、兼容入口、回归测试和旧日志已移出本交付目录，保存在：

```text
D:\Cumcm\第三题_开发测试工具
```

这些文件不属于论文算法代码附件。
