import ast
import math
from typing import ClassVar, Any

from pydantic import BaseModel, Field

from backend.tools.base_tool import BaseAgentTool


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
    )


class CalculatorTool(BaseAgentTool):
    name: str = "calculator"
    description: str = (
        "安全的数学计算器，支持加减乘除、幂运算、三角函数、对数等。"
        "输入数学表达式字符串，返回计算结果。"
    )
    args_schema: type[BaseModel] = CalculatorInput

    # 白名单：只允许安全操作
    SAFE_NAMES: ClassVar[dict[str, Any]] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": math.pow,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "exp": math.exp, "e": math.e, "pi": math.pi,
        "floor": math.floor, "ceil": math.ceil,
    }

    def _safe_eval(self, expr: str) -> float:
        """使用 AST 安全地求值数学表达式（防止代码注入）"""
        # 允许的 AST 节点类型
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
            ast.Constant, ast.Name, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
        )
        tree = ast.parse(expr.strip(), mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"不允许的操作: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in self.SAFE_NAMES:
                raise ValueError(f"不允许的函数/变量: {node.id}")


        return eval(
            compile(tree, "<string>", "eval"),
            {"__builtins__": {}},
            self.SAFE_NAMES,
        )

    def _run(self, expression: str) -> str:
        try:
            result = self._safe_eval(expression)
            # 整数结果去掉小数点
            if isinstance(result, float) and result.is_integer():
                return f"{expression} = {int(result)}"
            return f"{expression} = {result:.10g}"
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except ValueError as e:
            return f"表达式错误：{e}"
        except Exception as e:
            return f"计算失败：{e}"

