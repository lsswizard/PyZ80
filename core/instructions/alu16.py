"""
Z80 16-bit Arithmetic Instructions

This module implements 16-bit arithmetic operations:
    - Addition: ADD HL,rr, ADD IX,rr, ADD IY,rr
    - Addition with carry: ADC HL,rr, SBC HL,rr
    - Increment: INC rr, INC IX, INC IY
    - Decrement: DEC rr, DEC IX, DEC IY
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_C,
    FLAG_S,
    FLAG_Z,
    FLAG_PV,
    get_add16_flags,
    get_adc16_flags,
    get_sbc16_flags,
)


def add_hl_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """ADD HL,rr - Add register pair to HL (11 T-states)"""
    regs = cpu.regs
    hl = regs.HL
    operand = regs.get_reg16(reg_pair)
    regs.HL = (hl + operand) & 0xFFFF
    regs.F = (regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | get_add16_flags(hl, operand, regs.F)
    return 11


def inc_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """INC rr - Increment register pair (6 T-states, no flags)"""
    regs = cpu.regs
    regs.set_reg16(reg_pair, (regs.get_reg16(reg_pair) + 1) & 0xFFFF)
    return 6


def dec_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """DEC rr - Decrement register pair (6 T-states, no flags)"""
    regs = cpu.regs
    regs.set_reg16(reg_pair, (regs.get_reg16(reg_pair) - 1) & 0xFFFF)
    return 6


def adc_hl_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """ADC HL,rr - Consolidated (15 T-states)"""
    regs = cpu.regs
    hl = regs.HL
    carry = regs.F & FLAG_C
    # Use get_reg16 to avoid a throwaway list literal built on every call
    src = regs.get_reg16(reg_pair)
    regs.HL = (hl + src + carry) & 0xFFFF
    regs.F = get_adc16_flags(hl, src, carry)
    return 15


def sbc_hl_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """SBC HL,rr - Consolidated (15 T-states)"""
    regs = cpu.regs
    hl = regs.HL
    carry = regs.F & FLAG_C
    src = regs.get_reg16(reg_pair)
    regs.HL = (hl - src - carry) & 0xFFFF
    regs.F = get_sbc16_flags(hl, src, carry)
    return 15


def add_ix_rr(cpu: "Z80CPU", reg_pair: int, is_iy: bool = False) -> int:
    """ADD IX/IY,rr (15 T-states)"""
    regs = cpu.regs
    if is_iy:
        ix = regs.IY
        operand = regs.IY if reg_pair == 2 else regs.get_reg16(reg_pair)
        result = (ix + operand) & 0xFFFF
        regs.IY = result
    else:
        ix = regs.IX
        operand = regs.IX if reg_pair == 2 else regs.get_reg16(reg_pair)
        result = (ix + operand) & 0xFFFF
        regs.IX = result
    regs.F = (regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | get_add16_flags(ix, operand, regs.F)
    return 15


def inc_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """INC IX/IY (10 T-states)"""
    regs = cpu.regs
    if is_iy:
        regs.IY = (regs.IY + 1) & 0xFFFF
    else:
        regs.IX = (regs.IX + 1) & 0xFFFF
    return 10


def dec_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """DEC IX/IY (10 T-states)"""
    regs = cpu.regs
    if is_iy:
        regs.IY = (regs.IY - 1) & 0xFFFF
    else:
        regs.IX = (regs.IX - 1) & 0xFFFF
    return 10


def adc_ix_rr(cpu: "Z80CPU", reg_pair: int, is_iy: bool = False) -> int:
    """ADC IX/IY,rr - 16-bit add with carry to index register (15 T-states)"""
    regs = cpu.regs
    carry = regs.F & FLAG_C
    if is_iy:
        ix = regs.IY
        src = regs.IY if reg_pair == 2 else regs.get_reg16(reg_pair)
        regs.IY = (ix + src + carry) & 0xFFFF
    else:
        ix = regs.IX
        src = regs.IX if reg_pair == 2 else regs.get_reg16(reg_pair)
        regs.IX = (ix + src + carry) & 0xFFFF
    regs.F = get_adc16_flags(ix, src, carry)
    return 15


def sbc_ix_rr(cpu: "Z80CPU", reg_pair: int, is_iy: bool = False) -> int:
    """SBC IX/IY,rr - 16-bit subtract with carry from index register (15 T-states)"""
    regs = cpu.regs
    carry = regs.F & FLAG_C
    if is_iy:
        ix = regs.IY
        src = regs.IY if reg_pair == 2 else regs.get_reg16(reg_pair)
        regs.IY = (ix - src - carry) & 0xFFFF
    else:
        ix = regs.IX
        src = regs.IX if reg_pair == 2 else regs.get_reg16(reg_pair)
        regs.IX = (ix - src - carry) & 0xFFFF
    regs.F = get_sbc16_flags(ix, src, carry)
    return 15
