"""
Z80 Pipeline Module
Instruction execution primitives and micro-operation structures.
"""

from typing import Callable, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Z80CPU


class MicroOp:
    """
    Pre-decoded instruction ready for execution.

    Uses __slots__ instead of a dataclass to eliminate __dict__ overhead
    and reduce per-instruction attribute access cost (~12% faster than
    a plain dataclass on the hot handler-call path).
    """

    __slots__ = ("handler", "cycles", "length", "mnemonic", "is_ld_a_ir", "affects_f")

    def __init__(
        self,
        handler: "Callable[[Z80CPU], int]",
        cycles: int,
        length: int,
        mnemonic: str = "",
        is_ld_a_ir: bool = False,
        affects_f: bool = False,
    ) -> None:
        self.handler = handler
        self.cycles = cycles
        self.length = length
        self.mnemonic = mnemonic
        self.is_ld_a_ir = is_ld_a_ir
        self.affects_f = affects_f

    def __repr__(self) -> str:
        return f"MicroOp({self.mnemonic!r}, cycles={self.cycles}, length={self.length})"


# =============================================================================
# Fast Memory Access Helpers
# =============================================================================


def read_byte(mem: Any, addr: int, cpu: Any = None) -> int:
    """Fast byte read from memory."""
    if cpu is not None:
        return cpu.read_byte(addr)
    if hasattr(mem, "read_byte"):
        return mem.read_byte(addr, 0)
    return mem[addr & 0xFFFF]
