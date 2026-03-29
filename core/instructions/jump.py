"""
Z80 Jump and Call Instructions

This module implements all control flow instructions:
    - Absolute jumps: JP nn, JP cc,nn
    - Relative jumps: JR e, JR cc,e
    - Indexed jumps: JP (HL), JP (IX), JP (IY)
    - Loop: DJNZ e
    - Subroutines: CALL nn, CALL cc,nn, RET, RET cc
    - Restarts: RST p
    - Returns from interrupt: RETI, RETN
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU

from .ld8 import _read_addr_from_pc
from .ld16 import _push_word, _pop_word
from ..flags import COND_TABLE


_COND_TABLE = COND_TABLE


def _cond(cpu, condition):
    # Inline lookup to reduce call overhead for the most frequent branch checks.
    return _COND_TABLE[(cpu.regs.F << 3) | (condition & 0x07)]


def jp_nn(cpu: "Z80CPU") -> int:
    """JP nn - Jump to 16-bit address (10 T-states)"""
    cpu.regs.PC = _read_addr_from_pc(cpu, 1)
    cpu._pc_modified = True
    return 10


def jp_cc_nn(cpu: "Z80CPU", condition: int) -> int:
    """JP cc,nn - Conditional jump (10 T-states)"""
    if _cond(cpu, condition):
        cpu.regs.PC = _read_addr_from_pc(cpu, 1)
        cpu._pc_modified = True
    return 10


def jr_e(cpu: "Z80CPU") -> int:
    """JR e - Relative jump (12 T-states)"""
    regs = cpu.regs
    pc = regs.PC
    offset = cpu._bus_read((pc + 1) & 0xFFFF, cpu.cycles + 1)
    if offset >= 128:
        offset -= 256
    regs.PC = (pc + 2 + offset) & 0xFFFF
    cpu._pc_modified = True
    return 12


def jr_cc_e(cpu: "Z80CPU", condition: int) -> int:
    """JR cc,e - Conditional relative jump (7/12 T-states)"""
    if _COND_TABLE[(cpu.regs.F << 3) | (condition & 0x07)]:
        regs = cpu.regs
        pc = regs.PC
        offset = cpu._bus_read((pc + 1) & 0xFFFF, cpu.cycles + 1)
        if offset >= 128:
            offset -= 256
        regs.PC = (pc + 2 + offset) & 0xFFFF
        cpu._pc_modified = True
        return 12
    return 7


def jp_hl(cpu: "Z80CPU") -> int:
    """JP (HL) - Jump to HL (4 T-states)"""
    cpu.regs.PC = cpu.regs.HL
    cpu._pc_modified = True
    return 4


def djnz_e(cpu: "Z80CPU") -> int:
    """DJNZ e - Decrement B and jump if not zero (8/13 T-states)"""
    regs = cpu.regs
    b = (regs.B - 1) & 0xFF
    regs.B = b
    if b != 0:
        pc = regs.PC
        offset = cpu._bus_read((pc + 1) & 0xFFFF, cpu.cycles + 1)
        if offset >= 128:
            offset -= 256
        regs.PC = (pc + 2 + offset) & 0xFFFF
        cpu._pc_modified = True
        return 13
    return 8


def call_nn(cpu: "Z80CPU") -> int:
    """CALL nn - Call subroutine (17 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    _push_word(cpu, (regs.PC + 3) & 0xFFFF)
    regs.PC = addr
    cpu._pc_modified = True
    return 17


def call_cc_nn(cpu: "Z80CPU", condition: int) -> int:
    """CALL cc,nn - Conditional call (10/17 T-states)"""
    if _cond(cpu, condition):
        return call_nn(cpu)
    return 10


def ret(cpu: "Z80CPU") -> int:
    """RET - Return from subroutine (10 T-states)"""
    value, _ = _pop_word(cpu)
    cpu.regs.PC = value
    cpu._pc_modified = True
    return 10


def ret_cc(cpu: "Z80CPU", condition: int) -> int:
    """RET cc - Conditional return (5/11 T-states)"""
    if _cond(cpu, condition):
        ret(cpu)
        return 11
    return 5


def rst_p(cpu: "Z80CPU", addr: int) -> int:
    """RST p - Restart (call to page 0) (11 T-states)"""
    regs = cpu.regs
    _push_word(cpu, (regs.PC + 1) & 0xFFFF)
    regs.PC = addr
    cpu._pc_modified = True
    return 11


def reti(cpu: "Z80CPU") -> int:
    """RETI - Return from interrupt (14 T-states)"""
    ret(cpu)
    cpu.regs.IFF1 = cpu.regs.IFF2
    return 14


def retn(cpu: "Z80CPU") -> int:
    """RETN - Return from NMI (14 T-states)"""
    ret(cpu)
    cpu.regs.IFF1 = cpu.regs.IFF2
    return 14


def jp_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """JP (IX/IY) (8 T-states)"""
    regs = cpu.regs
    target = regs.IY if is_iy else regs.IX
    regs.Memptr = target
    regs.PC = target
    cpu._pc_modified = True
    return 8
