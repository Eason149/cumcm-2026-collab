"""第三题核心算法分层版兼容入口。

正式实现位于 ``q3_algorithm`` 包；保留本文件是为了让原调用方只需修改模块名。
"""

from q3_algorithm import (
    Q3Config,
    SELECTED_PARAMETERS,
    VERSION,
    solve,
    solve_optimized_v5,
)

__all__ = [
    "Q3Config",
    "SELECTED_PARAMETERS",
    "VERSION",
    "solve",
    "solve_optimized_v5",
]
