"""第三题 v5 分层实现的公开入口。"""

from .config import SELECTED_PARAMETERS, VERSION, Q3Config
from .solver import solve, solve_optimized_v5

__all__ = [
    "Q3Config",
    "SELECTED_PARAMETERS",
    "VERSION",
    "solve",
    "solve_optimized_v5",
]
