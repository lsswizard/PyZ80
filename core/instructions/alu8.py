"""
Z80 8-bit Arithmetic and Logic Instructions

This module implements all 8-bit ALU operations:
    - Addition: ADD, ADC
    - Subtraction: SUB, SBC, CP
    - Logical: AND, OR, XOR
    - Increment/Decrement: INC, DEC
    - Indexed variants with IX/IY registers
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_C,
    FLAG_H,
    FLAG_N,
    FLAG_PV,
    FLAG_S,
    FLAG_Z,
    # Precomputed tables — single bytearray index per ALU op (2-5x vs function calls)
    _ADD_PAIR,  # (ADD_FLAGS, ADC_FLAGS); indexed by carry (0 or 1)
    _SUB_PAIR,  # (SUB_FLAGS, SBC_FLAGS); indexed by carry (0 or 1)
    ADD_FLAGS,  # 65536-entry bytearray: add_flags[(a<<8)|b]
    ADC_FLAGS,
    SUB_FLAGS,
    SBC_FLAGS,
    CP_FLAGS,  # F3/F5 from operand b, not result
    INC_FLAGS,  # 256-entry: inc_flags[old_value]
    DEC_FLAGS_TBL,  # 256-entry: dec_flags[old_value]
    SZHZP_TABLE,  # AND flags (H always set)
    SZ53P_TABLE,  # OR/XOR flags
)
from .ld8 import _get_indexed_addr


def add_a_r(cpu: "Z80CPU", src: int) -> int:
    """ADD A,r - Add register to A (4 T-states)"""
    b = cpu.get_reg8(src)
    a = cpu.regs.A
    cpu.regs.A = (a + b) & 0xFF
    cpu.regs.F = ADD_FLAGS[(a << 8) | b]
    return 4


def add_a_n(cpu: "Z80CPU") -> int:
    """ADD A,n - Add immediate to A (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    b = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    a = cpu.regs.A
    cpu.regs.A = (a + b) & 0xFF
    cpu.regs.F = ADD_FLAGS[(a << 8) | b]
    return 7


def add_a_hl(cpu: "Z80CPU") -> int:
    """ADD A,(HL) - Add memory to A (7 T-states)"""
    cycles = cpu.cycles
    b = cpu._bus_read(cpu.regs.HL, cycles)
    a = cpu.regs.A
    cpu.regs.A = (a + b) & 0xFF
    cpu.regs.F = ADD_FLAGS[(a << 8) | b]
    return 7


def adc_a_r(cpu: "Z80CPU", src: int) -> int:
    """ADC A,r - Add register with carry to A (4 T-states)"""
    b = cpu.get_reg8(src)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a + b + c) & 0xFF
    regs.F = _ADD_PAIR[c][(a << 8) | b]
    return 4


def adc_a_n(cpu: "Z80CPU") -> int:
    """ADC A,n - Add immediate with carry to A (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    b = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a + b + c) & 0xFF
    regs.F = _ADD_PAIR[c][(a << 8) | b]
    return 7


def adc_a_hl(cpu: "Z80CPU") -> int:
    """ADC A,(HL) - Add memory with carry to A (7 T-states)"""
    cycles = cpu.cycles
    b = cpu._bus_read(cpu.regs.HL, cycles)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a + b + c) & 0xFF
    regs.F = _ADD_PAIR[c][(a << 8) | b]
    return 7


def sub_r(cpu: "Z80CPU", src: int) -> int:
    """SUB r - Subtract register from A (4 T-states)"""
    b = cpu.get_reg8(src)
    a = cpu.regs.A
    cpu.regs.A = (a - b) & 0xFF
    cpu.regs.F = SUB_FLAGS[(a << 8) | b]
    return 4


def sub_n(cpu: "Z80CPU") -> int:
    """SUB n - Subtract immediate from A (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    b = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    a = cpu.regs.A
    cpu.regs.A = (a - b) & 0xFF
    cpu.regs.F = SUB_FLAGS[(a << 8) | b]
    return 7


def sub_hl(cpu: "Z80CPU") -> int:
    """SUB (HL) - Subtract memory from A (7 T-states)"""
    cycles = cpu.cycles
    b = cpu._bus_read(cpu.regs.HL, cycles)
    a = cpu.regs.A
    cpu.regs.A = (a - b) & 0xFF
    cpu.regs.F = SUB_FLAGS[(a << 8) | b]
    return 7


def sbc_a_r(cpu: "Z80CPU", src: int) -> int:
    """SBC A,r - Subtract register with carry from A (4 T-states)"""
    b = cpu.get_reg8(src)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a - b - c) & 0xFF
    regs.F = _SUB_PAIR[c][(a << 8) | b]
    return 4


def sbc_a_n(cpu: "Z80CPU") -> int:
    """SBC A,n - Subtract immediate with carry from A (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    b = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a - b - c) & 0xFF
    regs.F = _SUB_PAIR[c][(a << 8) | b]
    return 7


def sbc_a_hl(cpu: "Z80CPU") -> int:
    """SBC A,(HL) - Subtract memory with carry from A (7 T-states)"""
    cycles = cpu.cycles
    b = cpu._bus_read(cpu.regs.HL, cycles)
    regs = cpu.regs
    a = regs.A
    c = regs.F & FLAG_C
    regs.A = (a - b - c) & 0xFF
    regs.F = _SUB_PAIR[c][(a << 8) | b]
    return 7


def and_r(cpu: "Z80CPU", src: int) -> int:
    """AND r - Logical AND with register (4 T-states)"""
    cpu.regs.A &= cpu.get_reg8(src)
    cpu.regs.F = SZHZP_TABLE[cpu.regs.A]
    return 4


def and_n(cpu: "Z80CPU") -> int:
    """AND n - Logical AND with immediate (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    cpu.regs.A &= cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    cpu.regs.F = SZHZP_TABLE[cpu.regs.A]
    return 7


def and_hl(cpu: "Z80CPU") -> int:
    """AND (HL) - Logical AND with memory (7 T-states)"""
    cycles = cpu.cycles
    cpu.regs.A &= cpu._bus_read(cpu.regs.HL, cycles)
    cpu.regs.F = SZHZP_TABLE[cpu.regs.A]
    return 7


def or_r(cpu: "Z80CPU", src: int) -> int:
    """OR r - Logical OR with register (4 T-states)"""
    cpu.regs.A |= cpu.get_reg8(src)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 4


def or_n(cpu: "Z80CPU") -> int:
    """OR n - Logical OR with immediate (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    cpu.regs.A |= cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 7


def or_hl(cpu: "Z80CPU") -> int:
    """OR (HL) - Logical OR with memory (7 T-states)"""
    cycles = cpu.cycles
    cpu.regs.A |= cpu._bus_read(cpu.regs.HL, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 7


def xor_r(cpu: "Z80CPU", src: int) -> int:
    """XOR r - Logical XOR with register (4 T-states)"""
    cpu.regs.A ^= cpu.get_reg8(src)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 4


def xor_n(cpu: "Z80CPU") -> int:
    """XOR n - Logical XOR with immediate (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    cpu.regs.A ^= cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 7


def xor_hl(cpu: "Z80CPU") -> int:
    """XOR (HL) - Logical XOR with memory (7 T-states)"""
    cycles = cpu.cycles
    cpu.regs.A ^= cpu._bus_read(cpu.regs.HL, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 7


def cp_r(cpu: "Z80CPU", src: int) -> int:
    """CP r - Compare register with A (4 T-states)"""
    b = cpu.get_reg8(src)
    cpu.regs.F = CP_FLAGS[(cpu.regs.A << 8) | b]
    return 4


def cp_n(cpu: "Z80CPU") -> int:
    """CP n - Compare immediate with A (7 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    b = cpu._bus_read((pc + 1) & 0xFFFF, cycles)
    cpu.regs.F = CP_FLAGS[(cpu.regs.A << 8) | b]
    return 7


def cp_hl(cpu: "Z80CPU") -> int:
    """CP (HL) - Compare memory with A (7 T-states)"""
    cycles = cpu.cycles
    b = cpu._bus_read(cpu.regs.HL, cycles)
    cpu.regs.F = CP_FLAGS[(cpu.regs.A << 8) | b]
    return 7


def inc_r(cpu: "Z80CPU", dest: int) -> int:
    """INC r - Increment register (4 T-states)"""
    value = cpu.get_reg8(dest)
    new_value = (value + 1) & 0xFF
    cpu.set_reg8(dest, new_value)
    cpu.regs.F = (cpu.regs.F & FLAG_C) | INC_FLAGS[value]
    return 4


def inc_hl(cpu: "Z80CPU") -> int:
    """INC (HL) - Increment memory (11 T-states)"""
    addr = cpu.regs.HL
    cycles = cpu.cycles
    value = cpu._bus_read(addr, cycles + 1)
    new_value = (value + 1) & 0xFF
    cpu._bus_write(addr, new_value, cycles + 4)
    cpu.cycles += 11
    cpu.regs.F = (cpu.regs.F & FLAG_C) | INC_FLAGS[value]
    return 11


def dec_r(cpu: "Z80CPU", dest: int) -> int:
    """DEC r - Decrement register (4 T-states)"""
    value = cpu.get_reg8(dest)
    new_value = (value - 1) & 0xFF
    cpu.set_reg8(dest, new_value)
    cpu.regs.F = (cpu.regs.F & FLAG_C) | DEC_FLAGS_TBL[value]
    return 4


def dec_hl(cpu: "Z80CPU") -> int:
    """DEC (HL) - Decrement memory (11 T-states)"""
    addr = cpu.regs.HL
    cycles = cpu.cycles
    value = cpu._bus_read(addr, cycles + 1)
    new_value = (value - 1) & 0xFF
    cpu._bus_write(addr, new_value, cycles + 4)
    cpu.cycles += 11
    cpu.regs.F = (cpu.regs.F & FLAG_C) | DEC_FLAGS_TBL[value]
    return 11


def add_a_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADD A,IXH/IYH (8 T-states)"""
    regs = cpu.regs
    b = regs.IYh if is_iy else regs.IXh
    a = regs.A
    regs.A = (a + b) & 0xFF
    regs.F = ADD_FLAGS[(a << 8) | b]
    return 8


def add_a_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADD A,IXL/IYL (8 T-states)"""
    regs = cpu.regs
    b = regs.IYl if is_iy else regs.IXl
    a = regs.A
    regs.A = (a + b) & 0xFF
    regs.F = ADD_FLAGS[(a << 8) | b]
    return 8


def add_a_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADD A,(IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    b = cpu._bus_read(addr, cycles)
    a = cpu.regs.A
    cpu.regs.A = (a + b) & 0xFF
    cpu.regs.F = ADD_FLAGS[(a << 8) | b]
    return 19


def adc_a_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADC A,(IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    b = cpu._bus_read(addr, cycles)
    regs = cpu.regs
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a + b + carry) & 0xFF
    regs.F = _ADD_PAIR[carry][(a << 8) | b]
    return 19


def adc_a_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADC A,IXH/IYH (8 T-states) - Undocumented"""
    regs = cpu.regs
    b = regs.IYh if is_iy else regs.IXh
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a + b + carry) & 0xFF
    regs.F = _ADD_PAIR[carry][(a << 8) | b]
    return 8


def adc_a_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """ADC A,IXL/IYL (8 T-states) - Undocumented"""
    regs = cpu.regs
    b = regs.IYl if is_iy else regs.IXl
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a + b + carry) & 0xFF
    regs.F = _ADD_PAIR[carry][(a << 8) | b]
    return 8


def sub_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SUB IXH/IYH (8 T-states)"""
    regs = cpu.regs
    b = regs.IYh if is_iy else regs.IXh
    regs.F = SUB_FLAGS[(regs.A << 8) | b]
    regs.A = (regs.A - b) & 0xFF
    return 8


def sub_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SUB IXL/IYL (8 T-states)"""
    regs = cpu.regs
    b = regs.IYl if is_iy else regs.IXl
    regs.F = SUB_FLAGS[(regs.A << 8) | b]
    regs.A = (regs.A - b) & 0xFF
    return 8


def sub_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SUB (IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    b = cpu._bus_read(addr, cycles)
    cpu.regs.F = SUB_FLAGS[(cpu.regs.A << 8) | b]
    cpu.regs.A = (cpu.regs.A - b) & 0xFF
    return 19


def sbc_a_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SBC A,(IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    b = cpu._bus_read(addr, cycles)
    regs = cpu.regs
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a - b - carry) & 0xFF
    regs.F = _SUB_PAIR[carry][(a << 8) | b]
    return 19


def sbc_a_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SBC A,IXH/IYH (8 T-states) - Undocumented"""
    regs = cpu.regs
    b = regs.IYh if is_iy else regs.IXh
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a - b - carry) & 0xFF
    regs.F = _SUB_PAIR[carry][(a << 8) | b]
    return 8


def sbc_a_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """SBC A,IXL/IYL (8 T-states) - Undocumented"""
    regs = cpu.regs
    b = regs.IYl if is_iy else regs.IXl
    a = regs.A
    carry = regs.F & FLAG_C
    regs.A = (a - b - carry) & 0xFF
    regs.F = _SUB_PAIR[carry][(a << 8) | b]
    return 8


def and_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """AND IXH/IYH (8 T-states)"""
    regs = cpu.regs
    regs.A &= regs.IYh if is_iy else regs.IXh
    regs.F = SZHZP_TABLE[regs.A]
    return 8


def and_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """AND IXL/IYL (8 T-states)"""
    regs = cpu.regs
    regs.A &= regs.IYl if is_iy else regs.IXl
    regs.F = SZHZP_TABLE[regs.A]
    return 8


def and_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """AND (IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    cpu.regs.A &= cpu._bus_read(addr, cycles)
    cpu.regs.F = SZHZP_TABLE[cpu.regs.A]
    return 19


def or_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """OR IXH/IYH (8 T-states)"""
    regs = cpu.regs
    regs.A |= regs.IYh if is_iy else regs.IXh
    regs.F = SZ53P_TABLE[regs.A]
    return 8


def or_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """OR IXL/IYL (8 T-states)"""
    regs = cpu.regs
    regs.A |= regs.IYl if is_iy else regs.IXl
    regs.F = SZ53P_TABLE[regs.A]
    return 8


def or_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """OR (IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    cpu.regs.A |= cpu._bus_read(addr, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 19


def xor_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """XOR IXH/IYH (8 T-states)"""
    regs = cpu.regs
    regs.A ^= regs.IYh if is_iy else regs.IXh
    regs.F = SZ53P_TABLE[regs.A]
    return 8


def xor_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """XOR IXL/IYL (8 T-states)"""
    regs = cpu.regs
    regs.A ^= regs.IYl if is_iy else regs.IXl
    regs.F = SZ53P_TABLE[regs.A]
    return 8


def xor_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """XOR (IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    cpu.regs.A ^= cpu._bus_read(addr, cycles)
    cpu.regs.F = SZ53P_TABLE[cpu.regs.A]
    return 19


def cp_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """CP IXH/IYH (8 T-states)"""
    regs = cpu.regs
    b = regs.IYh if is_iy else regs.IXh
    regs.F = CP_FLAGS[(regs.A << 8) | b]
    return 8


def cp_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """CP IXL/IYL (8 T-states)"""
    regs = cpu.regs
    b = regs.IYl if is_iy else regs.IXl
    regs.F = CP_FLAGS[(regs.A << 8) | b]
    return 8


def cp_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """CP (IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    b = cpu._bus_read(addr, cycles)
    cpu.regs.F = CP_FLAGS[(cpu.regs.A << 8) | b]
    return 19


def inc_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """INC IXH/IYH (8 T-states)"""
    regs = cpu.regs
    value = regs.IYh if is_iy else regs.IXh
    new_value = (value + 1) & 0xFF
    if is_iy:
        regs.IYh = new_value
    else:
        regs.IXh = new_value
    regs.F = (regs.F & FLAG_C) | INC_FLAGS[value]
    return 8


def inc_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """INC IXL/IYL (8 T-states)"""
    regs = cpu.regs
    value = regs.IYl if is_iy else regs.IXl
    new_value = (value + 1) & 0xFF
    if is_iy:
        regs.IYl = new_value
    else:
        regs.IXl = new_value
    regs.F = (regs.F & FLAG_C) | INC_FLAGS[value]
    return 8


def inc_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """INC (IX/IY+d) (23 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    value = cpu._bus_read(addr, cycles + 4)
    new_value = (value + 1) & 0xFF
    cpu._bus_write(addr, new_value, cycles + 7)
    cpu.cycles += 23
    cpu.regs.F = (cpu.regs.F & FLAG_C) | INC_FLAGS[value]
    return 23


def dec_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """DEC IXH/IYH (8 T-states)"""
    regs = cpu.regs
    value = regs.IYh if is_iy else regs.IXh
    new_value = (value - 1) & 0xFF
    if is_iy:
        regs.IYh = new_value
    else:
        regs.IXh = new_value
    regs.F = (regs.F & FLAG_C) | DEC_FLAGS_TBL[value]
    return 8


def dec_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """DEC IXL/IYL (8 T-states)"""
    regs = cpu.regs
    value = regs.IYl if is_iy else regs.IXl
    new_value = (value - 1) & 0xFF
    if is_iy:
        regs.IYl = new_value
    else:
        regs.IXl = new_value
    regs.F = (regs.F & FLAG_C) | DEC_FLAGS_TBL[value]
    return 8


def dec_ixd(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """DEC (IX/IY+d) (23 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    value = cpu._bus_read(addr, cycles + 4)
    new_value = (value - 1) & 0xFF
    cpu._bus_write(addr, new_value, cycles + 7)
    cpu.cycles += 23
    cpu.regs.F = (cpu.regs.F & FLAG_C) | DEC_FLAGS_TBL[value]
    return 23
