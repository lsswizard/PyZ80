"""
Z80 Load Instructions

This module implements all LD (Load) instructions:
    - 8-bit loads: LD r,r', LD r,n, LD r,(HL), LD (HL),r, LD (HL),n
    - 16-bit loads: LD rr,nn, LD (nn),HL, LD HL,(nn)
    - Special: LD A,(BC/DE/nn), LD (BC/DE/nn),A
    - Indexed: LD r,(IX+d), LD (IX+d),r, LD (IX+d),n
    - Registers: LD A,I, LD A,R, LD I,A, LD R,A
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU

# Flags used by ld_a_i / ld_a_r — imported once at module level, not per-call
from ..flags import FLAG_C, FLAG_S, FLAG_F3, FLAG_F5, FLAG_Z, FLAG_PV


def _read_addr_from_pc(cpu: "Z80CPU", offset: int = 1) -> int:
    """Read 16-bit address from PC at given offset (little-endian)."""
    regs = cpu.regs
    pc = regs.PC
    cycles = cpu.cycles
    low = cpu._bus_read((pc + offset) & 0xFFFF, cycles + 1)
    high = cpu._bus_read((pc + offset + 1) & 0xFFFF, cycles + 4)
    cpu.cycles += 7
    return low | (high << 8)


def _get_indexed_addr(cpu: "Z80CPU", is_iy: bool, offset_pos: int = 2) -> int:
    """Calculate indexed address (IX+d) or (IY+d).
    Also sets MEMPTR, which is used for flag calculations in BIT instructions.
    """
    regs = cpu.regs
    pc = regs.PC
    cycles = cpu.cycles
    displacement = cpu._bus_read((pc + offset_pos) & 0xFFFF, cycles + 1)
    disp = displacement if displacement < 128 else displacement - 256
    addr = ((regs.IY if is_iy else regs.IX) + disp) & 0xFFFF
    regs.Memptr = addr
    cpu.cycles += 4
    return addr


def ld_r_r(cpu: "Z80CPU", dest: int, src: int) -> int:
    """LD r,r' - Load register to register (4 T-states)

    Register encoding: 0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A
    """
    regs = cpu.regs

    # Inline register read (avoid method call overhead)
    if src == 7:
        val = regs.A
    elif src == 0:
        val = regs.B
    elif src == 1:
        val = regs.C
    elif src == 2:
        val = regs.D
    elif src == 3:
        val = regs.E
    elif src == 4:
        val = regs.H
    elif src == 5:
        val = regs.L
    else:  # src == 6: (HL)
        val = cpu._bus_read(regs.HL, cpu.cycles)

    # Inline register write (avoid method call overhead)
    if dest == 7:
        regs.A = val
    elif dest == 0:
        regs.B = val
    elif dest == 1:
        regs.C = val
    elif dest == 2:
        regs.D = val
    elif dest == 3:
        regs.E = val
    elif dest == 4:
        regs.H = val
    elif dest == 5:
        regs.L = val
    else:  # dest == 6: (HL)
        cpu._bus_write(regs.HL, val, cpu.cycles)

    return 4


def ld_r_n(cpu: "Z80CPU", dest: int) -> int:
    """LD r,n - Load immediate to register (7 T-states)"""
    val = cpu._bus_read((cpu.regs.PC + 1) & 0xFFFF, cpu.cycles)
    regs = cpu.regs
    if dest == 7:
        regs.A = val
    elif dest == 0:
        regs.B = val
    elif dest == 1:
        regs.C = val
    elif dest == 2:
        regs.D = val
    elif dest == 3:
        regs.E = val
    elif dest == 4:
        regs.H = val
    elif dest == 5:
        regs.L = val
    else:  # dest == 6: (HL)
        cpu._bus_write(regs.HL, val, cpu.cycles + 3)
    return 7


def ld_r_hl(cpu: "Z80CPU", dest: int) -> int:
    """LD r,(HL) - Load from memory to register (7 T-states)"""
    val = cpu._bus_read(cpu.regs.HL, cpu.cycles)
    regs = cpu.regs
    if dest == 7:
        regs.A = val
    elif dest == 0:
        regs.B = val
    elif dest == 1:
        regs.C = val
    elif dest == 2:
        regs.D = val
    elif dest == 3:
        regs.E = val
    elif dest == 4:
        regs.H = val
    elif dest == 5:
        regs.L = val
    return 7


def ld_hl_r(cpu: "Z80CPU", src: int) -> int:
    """LD (HL),r - Store register to memory (7 T-states)"""
    regs = cpu.regs
    if src == 7:
        val = regs.A
    elif src == 0:
        val = regs.B
    elif src == 1:
        val = regs.C
    elif src == 2:
        val = regs.D
    elif src == 3:
        val = regs.E
    elif src == 4:
        val = regs.H
    else:  # src == 5
        val = regs.L
    cpu._bus_write(regs.HL, val, cpu.cycles)
    return 7


def ld_hl_n(cpu: "Z80CPU") -> int:
    """LD (HL),n - Store immediate to memory (10 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    value = cpu._bus_read((regs.PC + 1) & 0xFFFF, cycles + 1)
    cpu._bus_write(regs.HL, value, cycles + 4)
    cpu.cycles += 10
    return 10


def ld_a_bc(cpu: "Z80CPU") -> int:
    """LD A,(BC) - Load A from address BC (7 T-states)"""
    regs = cpu.regs
    bc = regs.BC
    regs.A = cpu._bus_read(bc, cpu.cycles)
    regs.Memptr = (bc + 1) & 0xFFFF
    return 7


def ld_a_de(cpu: "Z80CPU") -> int:
    """LD A,(DE) - Load A from address DE (7 T-states)"""
    regs = cpu.regs
    de = regs.DE
    regs.A = cpu._bus_read(de, cpu.cycles)
    regs.Memptr = (de + 1) & 0xFFFF
    return 7


def ld_a_nn(cpu: "Z80CPU") -> int:
    """LD A,(nn) - Load A from 16-bit address (13 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    regs.A = cpu._bus_read(addr, cpu.cycles + 3)
    cpu.cycles += 13
    regs.Memptr = (addr + 1) & 0xFFFF
    return 13


def ld_bc_a(cpu: "Z80CPU") -> int:
    """LD (BC),A - Store A to address BC (7 T-states)"""
    regs = cpu.regs
    bc = regs.BC
    a = regs.A
    cpu._bus_write(bc, a, cpu.cycles)
    regs.Memptr = ((a << 8) | ((bc + 1) & 0xFF)) & 0xFFFF
    return 7


def ld_de_a(cpu: "Z80CPU") -> int:
    """LD (DE),A - Store A to address DE (7 T-states)"""
    regs = cpu.regs
    de = regs.DE
    a = regs.A
    cpu._bus_write(de, a, cpu.cycles)
    regs.Memptr = ((a << 8) | ((de + 1) & 0xFF)) & 0xFFFF
    return 7


def ld_nn_a(cpu: "Z80CPU") -> int:
    """LD (nn),A - Store A to 16-bit address (13 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    a = regs.A
    cpu._bus_write(addr, a, cpu.cycles + 3)
    cpu.cycles += 13
    regs.Memptr = ((a << 8) | ((addr + 1) & 0xFF)) & 0xFFFF
    return 13


def _ld_a_ir(cpu: "Z80CPU", reg_value: int) -> int:
    """Shared implementation for LD A,I and LD A,R (9 T-states)."""
    regs = cpu.regs
    a = reg_value
    regs.A = a
    f = regs.F & FLAG_C
    f |= a & (FLAG_S | FLAG_F3 | FLAG_F5)
    if a == 0:
        f |= FLAG_Z
    if regs.IFF2:
        f |= FLAG_PV
    regs.F = f
    return 9


def ld_a_i(cpu: "Z80CPU") -> int:
    """LD A,I - Load I register to A (9 T-states)"""
    return _ld_a_ir(cpu, cpu.regs.I)


def ld_a_r(cpu: "Z80CPU") -> int:
    """LD A,R - Load R register to A (9 T-states)"""
    return _ld_a_ir(cpu, cpu.regs.R)


def ld_i_a(cpu: "Z80CPU") -> int:
    """LD I,A - Load A to I register (9 T-states)"""
    cpu.regs.I = cpu.regs.A
    return 9


def ld_r_a(cpu: "Z80CPU") -> int:
    """LD R,A - Load A to R register (9 T-states)"""
    cpu.regs.R = cpu.regs.A
    return 9


def ld_r_ixd(cpu: "Z80CPU", dest: int, is_iy: bool = False) -> int:
    """LD r,(IX/IY+d) (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cpu.set_reg8(dest, cpu._bus_read(addr, cpu.cycles + 4))
    return 19


def ld_ixd_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD (IX/IY+d),r (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cpu._bus_write(addr, cpu.get_reg8(src), cpu.cycles + 4)
    return 19


def ld_ixd_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD (IX/IY+d),n (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cycles = cpu.cycles
    value = cpu._bus_read((cpu.regs.PC + 3) & 0xFFFF, cycles + 4)
    cpu._bus_write(addr, value, cycles + 7)
    cpu.cycles += 19
    return 19


def ld_ixh_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXH/IYH,n (11 T-states)"""
    regs = cpu.regs
    value = cpu._bus_read((regs.PC + 2) & 0xFFFF, cpu.cycles + 1)
    if is_iy:
        regs.IYh = value
    else:
        regs.IXh = value
    cpu.cycles += 11
    return 11


def ld_ixl_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXL/IYL,n (11 T-states)"""
    regs = cpu.regs
    value = cpu._bus_read((regs.PC + 2) & 0xFFFF, cpu.cycles + 1)
    if is_iy:
        regs.IYl = value
    else:
        regs.IXl = value
    cpu.cycles += 11
    return 11


def ld_a_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD A,IXH/IYH - Undocumented (8 T-states)"""
    regs = cpu.regs
    regs.A = regs.IYh if is_iy else regs.IXh
    return 8


def ld_a_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD A,IXL/IYL - Undocumented (8 T-states)"""
    regs = cpu.regs
    regs.A = regs.IYl if is_iy else regs.IXl
    return 8


def ld_r_ixh(cpu: "Z80CPU", dest: int, is_iy: bool = False) -> int:
    """LD r,IXH/IYH - Load IXH/IYH into register (8 T-states)"""
    regs = cpu.regs
    cpu.set_reg8(dest, regs.IYh if is_iy else regs.IXh)
    return 8


def ld_r_ixl(cpu: "Z80CPU", dest: int, is_iy: bool = False) -> int:
    """LD r,IXL/IYL - Load IXL/IYL into register (8 T-states)"""
    regs = cpu.regs
    cpu.set_reg8(dest, regs.IYl if is_iy else regs.IXl)
    return 8


def ld_ixh_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXH,IXL / LD IYH,IYL (8 T-states)"""
    regs = cpu.regs
    if is_iy:
        regs.IYh = regs.IYl
    else:
        regs.IXh = regs.IXl
    return 8


def ld_ixl_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXL,IXH / LD IYL,IYH (8 T-states)"""
    regs = cpu.regs
    if is_iy:
        regs.IYl = regs.IYh
    else:
        regs.IXl = regs.IXh
    return 8


def ld_ixh_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXH,IXH / LD IYH,IYH - Undocumented self-copy no-op (8 T-states)"""
    return 8


def ld_ixl_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXL,IXL / LD IYL,IYL - Undocumented self-copy no-op (8 T-states)"""
    return 8


def ld_ixh_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD IXH/IYH,r - Undocumented (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(src)
    if is_iy:
        regs.IYh = value
    else:
        regs.IXh = value
    return 8


def ld_ixl_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD IXL/IYL,r - Undocumented (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(src)
    if is_iy:
        regs.IYl = value
    else:
        regs.IXl = value
    return 8
