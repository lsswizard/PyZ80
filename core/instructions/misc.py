"""
Z80 Miscellaneous Instructions

This module implements miscellaneous CPU control instructions:
    - DAA: Decimal Adjust Accumulator
    - CPL, NEG: Complement/Negate accumulator
    - SCF, CCF: Carry flag operations
    - NOP, HALT: No-operation and halt
    - DI, EI: Enable/Disable interrupts
    - IM 0/1/2: Set interrupt mode
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_S,
    FLAG_Z,
    FLAG_PV,
    FLAG_H,
    FLAG_N,
    FLAG_C,
    FLAG_F3,
    FLAG_F5,
    get_daa_result,
    SUB_FLAGS,
)


def daa(cpu: "Z80CPU") -> int:
    """DAA - Decimal Adjust Accumulator (4 T-states)"""
    regs = cpu.regs
    regs.A, regs.F = get_daa_result(regs.A, regs.F)
    regs.Q = regs.F
    return 4


def cpl(cpu: "Z80CPU") -> int:
    """CPL - Complement A (4 T-states)"""
    regs = cpu.regs
    regs.A = (~regs.A) & 0xFF
    regs.F = (
        (regs.F & (FLAG_S | FLAG_Z | FLAG_PV | FLAG_C))
        | FLAG_H
        | FLAG_N
        | (regs.A & (FLAG_F3 | FLAG_F5))
    )
    regs.Q = regs.F
    return 4


def ccf(cpu: "Z80CPU") -> int:
    """CCF - Complement Carry Flag (4 T-states)"""
    regs = cpu.regs
    old_f = regs.F  # Save BEFORE modifying
    old_c = old_f & FLAG_C
    # CCF: C=!C, N=0, H=oldC, S/Z/PV preserved, F3/F5 from (last_Q^F)|A
    f = old_f & (FLAG_S | FLAG_Z | FLAG_PV)
    # Q factor: F3/F5 = ((last_Q ^ F) | A)[3/5] - use OLD flags
    result = (regs.last_Q ^ old_f) | regs.A
    f |= result & (FLAG_F3 | FLAG_F5)
    if old_c:
        f |= FLAG_H  # old carry becomes H, new C is 0
    else:
        f |= FLAG_C  # new C is 1, H stays 0
    regs.F = f
    regs.Q = f
    return 4


def scf(cpu: "Z80CPU") -> int:
    """SCF - Set Carry Flag (4 T-states)"""
    regs = cpu.regs
    old_f = regs.F  # Save BEFORE modifying
    # SCF: C=1, N=0, H=0, S/Z/PV preserved, F3/F5 from (last_Q^F)|A
    f = (old_f & (FLAG_S | FLAG_Z | FLAG_PV)) | FLAG_C
    # Q factor: F3/F5 = ((last_Q ^ F) | A)[3/5] - use OLD flags
    result = (regs.last_Q ^ old_f) | regs.A
    f |= result & (FLAG_F3 | FLAG_F5)
    regs.F = f
    regs.Q = f
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
    regs = cpu.regs
    regs.IFF1 = False
    regs.IFF2 = False
    regs.EI_PENDING = False
    return 4


def ei(cpu: "Z80CPU") -> int:
    """EI - Enable interrupts (delayed by one instruction) (4 T-states)"""
    regs = cpu.regs
    regs.EI_PENDING = True
    regs.EI_JUST_RESOLVED = False
    cpu._needs_slow_step = True
    return 4


def neg(cpu: "Z80CPU") -> int:
    """NEG - Negate A (8 T-states)"""
    regs = cpu.regs
    a = regs.A
    regs.A = (-a) & 0xFF
    regs.F = SUB_FLAGS[a]  # 0 - a = -a, so we use a as the operand
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
