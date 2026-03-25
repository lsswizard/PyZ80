"""
Z80 Block Transfer and I/O Instructions

This module implements block data transfer and I/O operations:
    - Block transfers: LDI, LDD, LDIR, LDDR
    - Block comparisons: CPI, CPD, CPIR, CPDR
    - Block I/O: INI, IND, INIR, INIR, OUTI, OUTD, OTIR, OTDR
    - Rotates: RLD, RRD (digit rotate)
    - Direct I/O: IN A,(n), OUT (n),A, IN r,(C), OUT (C),r
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_C,
    FLAG_F3,
    FLAG_F5,
    FLAG_H,
    FLAG_N,
    FLAG_PV,
    FLAG_S,
    FLAG_Z,
    PARITY_TABLE,
)


def _compute_ld_block_flags(regs, a_value: int, bc_after: int) -> None:
    """Compute flags for LDI/LDD instructions."""
    n = (a_value) & 0xFF
    regs.F = (regs.F & (FLAG_S | FLAG_Z | FLAG_C)) | (n & FLAG_F3)
    if n & 0x02:
        regs.F |= FLAG_F5
    if bc_after != 0:
        regs.F |= FLAG_PV


def ldi(cpu: "Z80CPU") -> int:
    """LDI - Load and increment (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    cpu._bus_write(regs.DE, value, cycles + 4)
    cpu.cycles += 16
    regs.HL = (regs.HL + 1) & 0xFFFF
    regs.DE = (regs.DE + 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    _compute_ld_block_flags(regs, regs.A + value, regs.BC)
    return 16


def ldir(cpu: "Z80CPU") -> int:
    """LDIR - Load, increment, repeat (16/21 T-states)
    When BC=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.BC == 0:
        cpu.cycles += 16
        return 16
    ldi(cpu)
    if cpu.regs.BC != 0:
        cpu._pc_modified = True
        return 21
    return 16


def ldd(cpu: "Z80CPU") -> int:
    """LDD - Load and decrement (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    cpu._bus_write(regs.DE, value, cycles + 4)
    cpu.cycles += 16
    regs.HL = (regs.HL - 1) & 0xFFFF
    regs.DE = (regs.DE - 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    _compute_ld_block_flags(regs, regs.A + value, regs.BC)
    return 16


def lddr(cpu: "Z80CPU") -> int:
    """LDDR - Load, decrement, repeat (16/21 T-states)
    When BC=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.BC == 0:
        cpu.cycles += 16
        return 16
    ldd(cpu)
    if cpu.regs.BC != 0:
        cpu._pc_modified = True
        return 21
    return 16


def cpi(cpu: "Z80CPU") -> int:
    """CPI - Compare and increment (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    cpu.cycles += 16
    result = (regs.A - value) & 0xFF
    regs.HL = (regs.HL + 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    half_carry = ((regs.A & 0x0F) - (value & 0x0F)) < 0
    n = (regs.A - value - (1 if half_carry else 0)) & 0xFF
    regs.F = (regs.F & FLAG_C) | FLAG_N | (n & FLAG_F3)
    if n & 0x02:
        regs.F |= FLAG_F5
    if result == 0:
        regs.F |= FLAG_Z
    if result & 0x80:
        regs.F |= FLAG_S
    if half_carry:
        regs.F |= FLAG_H
    if regs.BC != 0:
        regs.F |= FLAG_PV
    return 16


def _compute_cpi_flags(regs, a: int, value: int, bc_after: int) -> None:
    """Compute flags for CPI/CPD without side effects."""
    result = (a - value) & 0xFF
    half_carry = ((a & 0x0F) - (value & 0x0F)) < 0
    n = (a - value - (1 if half_carry else 0)) & 0xFF
    regs.F = (regs.F & FLAG_C) | FLAG_N | (n & FLAG_F3)
    if n & 0x02:
        regs.F |= FLAG_F5
    if result == 0:
        regs.F |= FLAG_Z
    if result & 0x80:
        regs.F |= FLAG_S
    if half_carry:
        regs.F |= FLAG_H
    if bc_after != 0:
        regs.F |= FLAG_PV


def cpir(cpu: "Z80CPU") -> int:
    """CPIR - Compare, increment, repeat (16/21 T-states).
    Inlined to avoid double bus_read that cpi()+check would cause."""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    a = regs.A
    result = (a - value) & 0xFF
    regs.HL = (regs.HL + 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    cpu.cycles += 16
    _compute_cpi_flags(regs, a, value, regs.BC)
    if regs.BC != 0 and result != 0:
        cpu._pc_modified = True
        return 21
    return 16


def cpd(cpu: "Z80CPU") -> int:
    """CPD - Compare and decrement (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)  # fix: was missing +1, matching cpi
    cpu.cycles += 16  # fix: was missing entirely
    result = (regs.A - value) & 0xFF
    regs.HL = (regs.HL - 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    half_carry = ((regs.A & 0x0F) - (value & 0x0F)) < 0
    n = (regs.A - value - (1 if half_carry else 0)) & 0xFF
    regs.F = (regs.F & FLAG_C) | FLAG_N | (n & FLAG_F3)
    if n & 0x02:
        regs.F |= FLAG_F5
    if result == 0:
        regs.F |= FLAG_Z
    if result & 0x80:
        regs.F |= FLAG_S
    if half_carry:
        regs.F |= FLAG_H
    if regs.BC != 0:
        regs.F |= FLAG_PV
    return 16


def cpdr(cpu: "Z80CPU") -> int:
    """CPDR - Compare, decrement, repeat (16/21 T-states).
    Inlined to avoid double bus_read that cpd()+check would cause."""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)  # fix: was missing +1, matching cpir
    a = regs.A
    result = (a - value) & 0xFF
    regs.HL = (regs.HL - 1) & 0xFFFF
    regs.BC = (regs.BC - 1) & 0xFFFF
    cpu.cycles += 16
    _compute_cpi_flags(regs, a, value, regs.BC)
    if regs.BC != 0 and result != 0:
        cpu._pc_modified = True
        return 21
    return 16


def _compute_in_out_flags(regs, value: int, old_b: int, new_b: int) -> None:
    """Compute flags for INI/IND/OUTI/OUTD instructions.
    N flag = bit 7 of the transferred value."""
    f = FLAG_C if regs.F & FLAG_C else 0
    if value & 0x80:
        f |= FLAG_N
    if new_b & 0x80:
        f |= FLAG_S
    if new_b == 0:
        f |= FLAG_Z
    if (old_b & 0x0F) == 0:
        f |= FLAG_H
    # PV = parity of (new_b & 0x07) XOR (value & 0x02)/2 ??? No.
    # Simplified: PV = 1 if old_b == 0x80 (wrapping to 0xFF)
    if old_b == 0x80:
        f |= FLAG_PV
    regs.F = f


def ini(cpu: "Z80CPU") -> int:
    """INI - Input and increment (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_io_read(regs.BC, cycles + 1)
    cpu._bus_write(regs.HL, value, cycles + 4)
    cpu.cycles += 16
    old_b = regs.B
    regs.B = (regs.B - 1) & 0xFF
    regs.HL = (regs.HL + 1) & 0xFFFF
    _compute_in_out_flags(regs, value, old_b, regs.B)
    return 16


def inir(cpu: "Z80CPU") -> int:
    """INIR - Input, increment, repeat (16/21 T-states)
    When B=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.B == 0:
        cpu.cycles += 16
        return 16
    old_b = cpu.regs.B
    ini(cpu)
    if old_b != 1:
        cpu._pc_modified = True
        return 21
    return 16


def ind(cpu: "Z80CPU") -> int:
    """IND - Input and decrement (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_io_read(regs.BC, cycles + 1)
    cpu._bus_write(regs.HL, value, cycles + 4)
    cpu.cycles += 16
    old_b = regs.B
    regs.B = (regs.B - 1) & 0xFF
    regs.HL = (regs.HL - 1) & 0xFFFF
    _compute_in_out_flags(regs, value, old_b, regs.B)
    return 16


def indr(cpu: "Z80CPU") -> int:
    """INDR - Input, decrement, repeat (16/21 T-states)
    When B=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.B == 0:
        cpu.cycles += 16
        return 16
    old_b = cpu.regs.B
    ind(cpu)
    if old_b != 1:
        cpu._pc_modified = True
        return 21
    return 16


def outi(cpu: "Z80CPU") -> int:
    """OUTI - Output and increment (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    old_b = regs.B
    regs.B = (regs.B - 1) & 0xFF
    cpu._bus_io_write(regs.BC, value, cycles + 4)
    cpu.cycles += 16
    regs.HL = (regs.HL + 1) & 0xFFFF
    _compute_in_out_flags(regs, value, old_b, regs.B)
    return 16


def otir(cpu: "Z80CPU") -> int:
    """OTIR - Output, increment, repeat (16/21 T-states)
    When B=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.B == 0:
        cpu.cycles += 16
        return 16
    old_b = cpu.regs.B
    outi(cpu)
    if old_b != 1:
        cpu._pc_modified = True
        return 21
    return 16


def outd(cpu: "Z80CPU") -> int:
    """OUTD - Output and decrement (16 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    old_b = regs.B
    regs.B = (regs.B - 1) & 0xFF
    cpu._bus_io_write(regs.BC, value, cycles + 4)
    cpu.cycles += 16
    regs.HL = (regs.HL - 1) & 0xFFFF
    _compute_in_out_flags(regs, value, old_b, regs.B)
    return 16


def otdr(cpu: "Z80CPU") -> int:
    """OTDR - Output, decrement, repeat (16/21 T-states)
    When B=0 before execution, acts as a 2-byte NOP (16 T-states)."""
    if cpu.regs.B == 0:
        cpu.cycles += 16
        return 16
    old_b = cpu.regs.B
    outd(cpu)
    if old_b != 1:
        cpu._pc_modified = True
        return 21
    return 16


def in_a_n(cpu: "Z80CPU") -> int:
    """IN A,(n) - Input from port to A (11 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    port = cpu._bus_read((pc + 1) & 0xFFFF, cycles + 1)
    addr = (cpu.regs.A << 8) | port
    cpu.regs.A = cpu._bus_io_read(addr, cycles + 4)
    cpu.cycles += 11
    return 11


def out_n_a(cpu: "Z80CPU") -> int:
    """OUT (n),A - Output A to port (11 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    port = cpu._bus_read((pc + 1) & 0xFFFF, cycles + 1)
    addr = (cpu.regs.A << 8) | port
    cpu._bus_io_write(addr, cpu.regs.A, cycles + 4)
    cpu.cycles += 11
    return 11


def in_reg_c(cpu: "Z80CPU", reg: int) -> int:
    """IN r,(C) - Consolidated (12 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_io_read(regs.BC, cycles + 1)
    cpu.cycles += 12
    if reg != 6:
        cpu.set_reg8(reg, value)
    regs.F = (regs.F & FLAG_C) | (value & (FLAG_S | FLAG_F3 | FLAG_F5))
    if value == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[value]:
        regs.F |= FLAG_PV
    return 12


def in_a_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 7)


def in_b_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 0)


def in_c_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 1)


def in_d_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 2)


def in_e_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 3)


def in_h_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 4)


def in_l_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 5)


def in_f_c(cpu: "Z80CPU") -> int:
    return in_reg_c(cpu, 6)


def out_c_reg(cpu: "Z80CPU", reg: int) -> int:
    """OUT (C),r - Consolidated (12 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    cpu._bus_io_write(regs.BC, cpu.get_reg8(reg), cycles)
    return 12


def out_c_a(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 7)


def out_c_b(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 0)


def out_c_c(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 1)


def out_c_d(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 2)


def out_c_e(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 3)


def out_c_h(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 4)


def out_c_l(cpu: "Z80CPU") -> int:
    return out_c_reg(cpu, 5)


def out_c_0(cpu: "Z80CPU") -> int:
    """OUT (C),0 - Output 0 to port (undocumented) (12 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    cpu._bus_io_write(regs.BC, 0, cycles)
    return 12


def rld(cpu: "Z80CPU") -> int:
    """RLD - Rotate digit left (BCD) (18 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    low_nibble = value & 0x0F
    high_nibble = value & 0xF0
    a_low = regs.A & 0x0F
    new_value = (low_nibble << 4) | a_low
    new_a = (regs.A & 0xF0) | (high_nibble >> 4)
    cpu._bus_write(regs.HL, new_value, cycles + 4)
    cpu.cycles += 18
    regs.A = new_a
    regs.F = regs.F & FLAG_C
    if regs.A == 0:
        regs.F |= FLAG_Z
    if regs.A & 0x80:
        regs.F |= FLAG_S
    if PARITY_TABLE[regs.A]:
        regs.F |= FLAG_PV
    regs.F |= regs.A & (FLAG_F3 | FLAG_F5)
    return 18


def rrd(cpu: "Z80CPU") -> int:
    """RRD - Rotate digit right (BCD) (18 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read(regs.HL, cycles + 1)
    low_nibble = value & 0x0F
    high_nibble = value & 0xF0
    a_low = regs.A & 0x0F
    new_value = (a_low << 4) | (high_nibble >> 4)
    new_a = (regs.A & 0xF0) | low_nibble
    cpu._bus_write(regs.HL, new_value, cycles + 4)
    cpu.cycles += 18
    regs.A = new_a
    regs.F = regs.F & FLAG_C
    if regs.A == 0:
        regs.F |= FLAG_Z
    if regs.A & 0x80:
        regs.F |= FLAG_S
    if PARITY_TABLE[regs.A]:
        regs.F |= FLAG_PV
    regs.F |= regs.A & (FLAG_F3 | FLAG_F5)
    return 18
