"""
Z80 Pipeline Module
Instruction execution primitives and micro-operation structures.
"""

from typing import Callable, TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from .cpu import Z80CPU


@dataclass
class MicroOp:
    """
    Micro-operation structure.
    Pre-decoded instruction ready for execution.
    """

    handler: Callable[["Z80CPU"], int]
    cycles: int
    length: int
    mnemonic: str = ""


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


def write_byte(mem: Any, addr: int, value: int, cpu: Any = None) -> None:
    """Fast byte write to memory."""
    if cpu is not None:
        cpu.write_byte(addr, value)
    elif hasattr(mem, "write_byte"):
        mem.write_byte(addr, value, 0)
    else:
        mem[addr & 0xFFFF] = value & 0xFF


def read_word(mem: Any, addr: int, cpu: Any = None) -> int:
    """Fast word read (little-endian)."""
    if cpu is not None:
        low = cpu.read_byte(addr)
        high = cpu.read_byte((addr + 1) & 0xFFFF)
        return low | (high << 8)

    if hasattr(mem, "read_byte"):
        low = mem.read_byte(addr & 0xFFFF, 0)
        high = mem.read_byte((addr + 1) & 0xFFFF, 0)
        return low | (high << 8)

    addr &= 0xFFFF
    return mem[addr] | (mem[(addr + 1) & 0xFFFF] << 8)


def write_word(mem: Any, addr: int, value: int, cpu: Any = None) -> None:
    """Fast word write (little-endian)."""
    if cpu is not None:
        cpu.write_byte(addr, value & 0xFF)
        cpu.write_byte((addr + 1) & 0xFFFF, (value >> 8) & 0xFF)
    elif hasattr(mem, "write_byte"):
        mem.write_byte(addr & 0xFFFF, value & 0xFF, 0)
        mem.write_byte((addr + 1) & 0xFFFF, (value >> 8) & 0xFF, 0)
    else:
        addr &= 0xFFFF
        mem[addr] = value & 0xFF
        mem[(addr + 1) & 0xFFFF] = (value >> 8) & 0xFF


def push_word(mem: Any, sp: int, value: int, cpu: Any = None) -> int:
    """Push word to stack."""
    sp = (sp - 2) & 0xFFFF
    write_word(mem, sp, value, cpu)
    return sp


def pop_word(mem: Any, sp: int, cpu: Any = None) -> tuple[int, int]:
    """Pop word from stack."""
    value = read_word(mem, sp, cpu)
    return value, (sp + 2) & 0xFFFF
