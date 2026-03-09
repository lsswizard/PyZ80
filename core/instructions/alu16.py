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
    operand = cpu.regs.get_reg16(reg_pair)
    hl = cpu.regs.HL
    cpu.regs.HL = (hl + operand) & 0xFFFF
    cpu.regs.F = (cpu.regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | get_add16_flags(
        hl, operand, cpu.regs.F
    )
    return 11


def inc_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """INC rr - Increment register pair (6 T-states, no flags)"""
    value = cpu.regs.get_reg16(reg_pair)
    cpu.regs.set_reg16(reg_pair, (value + 1) & 0xFFFF)
    return 6


def dec_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """DEC rr - Decrement register pair (6 T-states, no flags)"""
    value = cpu.regs.get_reg16(reg_pair)
    cpu.regs.set_reg16(reg_pair, (value - 1) & 0xFFFF)
    return 6


def adc_hl_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """ADC HL,rr - Consolidated (15 T-states)"""
    hl = cpu.regs.HL
    carry = cpu.regs.F & FLAG_C
    src = [cpu.regs.BC, cpu.regs.DE, cpu.regs.HL, cpu.regs.SP][reg_pair]
    cpu.regs.HL = (hl + src + carry) & 0xFFFF
    cpu.regs.F = get_adc16_flags(hl, src, carry)
    return 15


def sbc_hl_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """SBC HL,rr - Consolidated (15 T-states)"""
    hl = cpu.regs.HL
    carry = cpu.regs.F & FLAG_C
    src = [cpu.regs.BC, cpu.regs.DE, cpu.regs.HL, cpu.regs.SP][reg_pair]
    cpu.regs.HL = (hl - src - carry) & 0xFFFF
    cpu.regs.F = get_sbc16_flags(hl, src, carry)
    return 15


def add_ix_rr(cpu: "Z80CPU", reg_pair: int, is_iy: bool = False) -> int:
    """ADD IX/IY,rr (15 T-states)"""
    ix = cpu.regs.IY if is_iy else cpu.regs.IX
    if is_iy and reg_pair == 2:
        operand = cpu.regs.IY
    elif not is_iy and reg_pair == 2:
        operand = cpu.regs.IX
    else:
        operand = cpu.regs.get_reg16(reg_pair)

    result = (ix + operand) & 0xFFFF
    if is_iy:
        cpu.regs.IY = result
    else:
        cpu.regs.IX = result
    cpu.regs.F = (cpu.regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | get_add16_flags(
        ix, operand, cpu.regs.F
    )
    return 15


def inc_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """INC IX/IY (10 T-states)"""
    if is_iy:
        cpu.regs.IY = (cpu.regs.IY + 1) & 0xFFFF
    else:
        cpu.regs.IX = (cpu.regs.IX + 1) & 0xFFFF
    return 10


def dec_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """DEC IX/IY (10 T-states)"""
    if is_iy:
        cpu.regs.IY = (cpu.regs.IY - 1) & 0xFFFF
    else:
        cpu.regs.IX = (cpu.regs.IX - 1) & 0xFFFF
    return 10
