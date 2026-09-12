# 第三题 v5 分层代码

本目录将原 `第三问_核心算法.py` 的采用路径拆成独立模块。重构目标是改善层级结构，不改变 v5 的动作决策口径。

## 模块职责

| 文件 | 职责 |
|---|---|
| `config.py` | 题设常量、采用参数和版本号 |
| `models.py` | 已发现源与求解器在线状态 |
| `geometry_backend.py` | 历史几何内核的唯一适配入口 |
| `geometry.py` | 六扇区构造和偏置定位点选择 |
| `route.py` | 开放路线 2-opt 与可移动搜索站 |
| `planner.py` | 初始共享探测和联合任务规划 |
| `actions.py` | 测量、扫描、共享测向、清除及状态更新 |
| `solver.py` | 顶层闭环与停止条件 |
| `result.py` | 返回结果组装 |

## 使用

确保 `q3_optimizer_v4.py` 所在目录可被 Python 导入，然后调用：

```python
from q3_algorithm import solve_optimized_v5

result = solve_optimized_v5(client)
```

也可以沿用单文件式入口：

```python
from 第三问_核心算法_分层版 import solve_optimized_v5
```

`client` 接口与原附件相同。本包没有复制几何内核，也没有包含模拟器客户端。

## 与原文件的边界

分层版只保留 `SELECTED_PARAMETERS` 对应的正式 v5 路径。原文件中的 `v4`、`near_joint`、`scan_first`、`sweep`、`arc_scan`、`exact_positive` 等实验分支没有混入正式求解器；如需继续比较，应在独立实验模块中实现共同的规划器接口。
