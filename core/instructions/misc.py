from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_S, FLAG_Z, FLAG_PV, FLAG_H, FLAG_N, FLAG_C, FLAG_F3, FLAG_F5,
    get_daa_result, get_sub_flags
)

def daa(cpu: "Z80CPU") -> int:
    """DAA - Decimal Adjust Accumulator (4 T-states)"""
    cpu.regs.A, cpu.regs.F = get_daa_result(cpu.regs.A, cpu.regs.F)
    return 4

def cpl(cpu: "Z80CPU") -> int:
    """CPL - Complement A (4 T-states)"""
    cpu.regs.A = (~cpu.regs.A) & 0xFF
    cpu.regs.F = (
        (cpu.regs.F & (FLAG_S | FLAG_Z | FLAG_PV | FLAG_C))
        | FLAG_H
        | FLAG_N
        | (cpu.regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4

def ccf(cpu: "Z80CPU") -> int:
    """CCF - Complement Carry Flag (4 T-states)"""
    old_c = cpu.regs.F & FLAG_C
    cpu.regs.F = (cpu.regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | (
        cpu.regs.A & (FLAG_F3 | FLAG_F5)
    )
    if old_c:
        cpu.regs.F |= FLAG_H
    if not old_c:
        cpu.regs.F |= FLAG_C
    return 4

def scf(cpu: "Z80CPU") -> int:
    """SCF - Set Carry Flag (4 T-states)"""
    cpu.regs.F = (
        (cpu.regs.F & (FLAG_S | FLAG_Z | FLAG_PV))
        | FLAG_C
        | (cpu.regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4

def nop(cpu: "Z80CPU") -> int:
    """NOP - No operation (4 T-states)"""
    return 4

def halt(cpu: "Z80CPU") -> int:
    """HALT - Halt CPU (4 T-states)"""
    cpu.halted = True
    cpu._pc_modified = True
    return 4

def di(cpu: "Z80CPU") -> int:
    """DI - Disable interrupts (4 T-states)"""
    cpu.regs.IFF1 = False
    cpu.regs.IFF2 = False
    cpu.regs.EI_PENDING = False
    return 4

def ei(cpu: "Z80CPU") -> int:
    """EI - Enable interrupts (delayed by one instruction) (4 T-states)"""
    cpu.regs.EI_PENDING = True
    cpu.regs.EI_JUST_RESOLVED = False
    return 4

def neg(cpu: "Z80CPU") -> int:
    """NEG - Negate A (8 T-states)"""
    a = cpu.regs.A
    cpu.regs.A = (-a) & 0xFF
    cpu.regs.F = get_sub_flags(0, a)
    return 8

def im_0(cpu: "Z80CPU") -> int:
    """IM 0 - Set interrupt mode 0 (8 T-states)"""
    cpu.regs.IM = 0
    return 8

def im_1(cpu: "Z80CPU") -> int:
    """IM 1 - Set interrupt mode 1 (8 T-states)"""
    cpu.regs.IM = 1
    return 8

def im_2(cpu: "Z80CPU") -> int:
    """IM 2 - Set interrupt mode 2 (8 T-states)"""
    cpu.regs.IM = 2
    return 8
