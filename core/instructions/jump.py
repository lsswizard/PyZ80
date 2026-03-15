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


def jp_nn(cpu: "Z80CPU") -> int:
    """JP nn - Jump to 16-bit address (10 T-states)"""
    cpu.regs.PC = _read_addr_from_pc(cpu, 1)
    cpu._pc_modified = True
    return 10


def jp_cc_nn(cpu: "Z80CPU", condition: int) -> int:
    """JP cc,nn - Conditional jump (10 T-states)"""
    if cpu.check_condition(condition):
        cpu.regs.PC = _read_addr_from_pc(cpu, 1)
        cpu._pc_modified = True
    return 10


def jr_e(cpu: "Z80CPU") -> int:
    """JR e - Relative jump (12 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    offset = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    offset = offset if offset < 128 else offset - 256
    cpu.advance_cycles(3)
    cpu.regs.PC = (pc + 2 + offset) & 0xFFFF
    cpu._pc_modified = True
    return 12


def jr_cc_e(cpu: "Z80CPU", condition: int) -> int:
    """JR cc,e - Conditional relative jump (7/12 T-states)"""
    if cpu.check_condition(condition):
        pc = cpu.regs.PC
        cycles = cpu.cycles
        offset = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
        offset = offset if offset < 128 else offset - 256
        cpu.advance_cycles(3)
        cpu.regs.PC = (pc + 2 + offset) & 0xFFFF
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
    regs.B = (regs.B - 1) & 0xFF
    if regs.B != 0:
        pc = regs.PC
        cycles = cpu.cycles
        offset = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
        offset = offset if offset < 128 else offset - 256
        cpu.advance_cycles(3)
        regs.PC = (pc + 2 + offset) & 0xFFFF
        cpu._pc_modified = True
        return 13
    return 8


def call_nn(cpu: "Z80CPU") -> int:
    """CALL nn - Call subroutine (17 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    next_pc = (regs.PC + 3) & 0xFFFF
    _push_word(cpu, next_pc)
    regs.PC = addr
    cpu._pc_modified = True
    return 17


def call_cc_nn(cpu: "Z80CPU", condition: int) -> int:
    """CALL cc,nn - Conditional call (10/17 T-states)"""
    if cpu.check_condition(condition):
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
    if cpu.check_condition(condition):
        ret(cpu)
        return 11
    return 5


def rst_p(cpu: "Z80CPU", addr: int) -> int:
    """RST p - Restart (call to page 0) (11 T-states)"""
    regs = cpu.regs
    return_addr = (regs.PC + 1) & 0xFFFF
    _push_word(cpu, return_addr)
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
    regs.Memptr = regs.IY if is_iy else regs.IX
    regs.PC = regs.IY if is_iy else regs.IX
    cpu._pc_modified = True
    return 8
