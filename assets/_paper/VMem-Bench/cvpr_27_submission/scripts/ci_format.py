"""Format frozen confidence intervals as gray subscript widths."""
from decimal import Decimal


def ci_cell(value: str, lo: str, hi: str, *, bold: bool = False, underline: bool = False, width_digits: int | None = None) -> str:
    """Display total CI width (upper minus lower), not a standard deviation."""
    down, up = Decimal(value) - Decimal(lo), Decimal(hi) - Decimal(value)
    if min(down, up) < 0:
        raise ValueError("Point estimate lies outside displayed interval")
    point = value
    if bold:
        point = rf"\textbf{{{point}}}"
    if underline:
        point = rf"\underline{{{point}}}"
    width = Decimal(hi) - Decimal(lo)
    width_text = f"{width:f}" if width_digits is None else f"{width:.{width_digits}f}"
    return rf"\ci{{{point}}}{{{width_text}}}"
