from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU


def _read_addr_from_pc(cpu: "Z80CPU", offset: int = 1) -> int:
    """Read 16-bit address from PC at given offset (little-endian)."""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    low = cpu._bus_read((pc + offset) & 0xFFFF, cycles)
    high = cpu._bus_read((pc + offset + 1) & 0xFFFF, cycles)
    return low | (high << 8)


def _get_indexed_addr(cpu: "Z80CPU", is_iy: bool, offset_pos: int = 2) -> int:
    """Calculate indexed address (IX+d) or (IY+d)
    Also sets MEMPTR which is used for flag calculations in BIT instructions.
    """
    pc = cpu.regs.PC
    cycles = cpu.cycles
    displacement = cpu._bus_read((pc + offset_pos) & 0xFFFF, cycles)
    disp = displacement if displacement < 128 else displacement - 256
    base = cpu.regs.IY if is_iy else cpu.regs.IX
    addr = (base + disp) & 0xFFFF
    cpu.regs.Memptr = addr
    return addr


def ld_r_r(cpu: "Z80CPU", dest: int, src: int) -> int:
    """LD r,r' - Load register to register (4 T-states)"""
    cpu.set_reg8(dest, cpu.get_reg8(src))
    return 4


def ld_r_n(cpu: "Z80CPU", dest: int) -> int:
    """LD r,n - Load immediate to register (7 T-states)"""
    value = cpu._bus_read((cpu.regs.PC + 1) & 0xFFFF, cpu.cycles)
    cpu.set_reg8(dest, value)
    return 7


def ld_r_hl(cpu: "Z80CPU", dest: int) -> int:
    """LD r,(HL) - Load from memory to register (7 T-states)"""
    value = cpu._bus_read(cpu.regs.HL, cpu.cycles)
    cpu.set_reg8(dest, value)
    return 7


def ld_hl_r(cpu: "Z80CPU", src: int) -> int:
    """LD (HL),r - Store register to memory (7 T-states)"""
    value = cpu.get_reg8(src)
    cpu._bus_write(cpu.regs.HL, value, cpu.cycles)
    return 7


def ld_hl_n(cpu: "Z80CPU") -> int:
    """LD (HL),n - Store immediate to memory (10 T-states)"""
    value = cpu._bus_read((cpu.regs.PC + 1) & 0xFFFF, cpu.cycles)
    cpu._bus_write(cpu.regs.HL, value, cpu.cycles)
    return 10


def ld_a_bc(cpu: "Z80CPU") -> int:
    """LD A,(BC) - Load A from address BC (7 T-states)"""
    bc = cpu.regs.BC
    cpu.regs.A = cpu._bus_read(bc, cpu.cycles)
    cpu.regs.Memptr = (bc + 1) & 0xFFFF
    return 7


def ld_a_de(cpu: "Z80CPU") -> int:
    """LD A,(DE) - Load A from address DE (7 T-states)"""
    de = cpu.regs.DE
    cpu.regs.A = cpu._bus_read(de, cpu.cycles)
    cpu.regs.Memptr = (de + 1) & 0xFFFF
    return 7


def ld_a_nn(cpu: "Z80CPU") -> int:
    """LD A,(nn) - Load A from 16-bit address (13 T-states)"""
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    cpu.regs.A = cpu._bus_read(addr, cycles)
    cpu.regs.Memptr = (addr + 1) & 0xFFFF
    return 13


def ld_bc_a(cpu: "Z80CPU") -> int:
    """LD (BC),A - Store A to address BC (7 T-states)"""
    bc = cpu.regs.BC
    cpu._bus_write(bc, cpu.regs.A, cpu.cycles)
    cpu.regs.Memptr = ((cpu.regs.A << 8) | ((bc + 1) & 0xFF)) & 0xFFFF
    return 7


def ld_de_a(cpu: "Z80CPU") -> int:
    """LD (DE),A - Store A to address DE (7 T-states)"""
    de = cpu.regs.DE
    cpu._bus_write(de, cpu.regs.A, cpu.cycles)
    cpu.regs.Memptr = ((cpu.regs.A << 8) | ((de + 1) & 0xFF)) & 0xFFFF
    return 7


def ld_nn_a(cpu: "Z80CPU") -> int:
    """LD (nn),A - Store A to 16-bit address (13 T-states)"""
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    cpu._bus_write(addr, cpu.regs.A, cycles)
    cpu.regs.Memptr = ((cpu.regs.A << 8) | ((addr + 1) & 0xFF)) & 0xFFFF
    return 13


def ld_a_i(cpu: "Z80CPU") -> int:
    """LD A,I - Load I register to A (9 T-states)"""
    from ..tables import FLAG_C, FLAG_S, FLAG_F3, FLAG_F5, FLAG_Z, FLAG_PV

    cpu.regs.A = cpu.regs.I
    cpu.regs.F = (cpu.regs.F & FLAG_C) | (cpu.regs.A & (FLAG_S | FLAG_F3 | FLAG_F5))
    if cpu.regs.A == 0:
        cpu.regs.F |= FLAG_Z
    if cpu.regs.IFF2:
        cpu.regs.F |= FLAG_PV
    return 9


def ld_a_r(cpu: "Z80CPU") -> int:
    """LD A,R - Load R register to A (9 T-states)"""
    from ..tables import FLAG_C, FLAG_S, FLAG_F3, FLAG_F5, FLAG_Z, FLAG_PV

    cpu.regs.A = cpu.regs.R
    cpu.regs.F = (cpu.regs.F & FLAG_C) | (cpu.regs.A & (FLAG_S | FLAG_F3 | FLAG_F5))
    if cpu.regs.A == 0:
        cpu.regs.F |= FLAG_Z
    if cpu.regs.IFF2:
        cpu.regs.F |= FLAG_PV
    return 9


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
    cpu.set_reg8(dest, cpu._bus_read(addr, cpu.cycles))
    return 19


def ld_ixd_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD (IX/IY+d),r (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    cpu._bus_write(addr, cpu.get_reg8(src), cpu.cycles)
    return 19


def ld_ixd_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD (IX/IY+d),n (19 T-states)"""
    addr = _get_indexed_addr(cpu, is_iy)
    value = cpu._bus_read((cpu.regs.PC + 3) & 0xFFFF, cpu.cycles)
    cpu._bus_write(addr, value, cpu.cycles)
    return 19


def ld_ixh_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXH/IYH,n (11 T-states)"""
    value = cpu._bus_read((cpu.regs.PC + 2) & 0xFFFF, cpu.cycles)
    if is_iy:
        cpu.regs.IYh = value
    else:
        cpu.regs.IXh = value
    return 11


def ld_ixl_n(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXL/IYL,n (11 T-states)"""
    value = cpu._bus_read((cpu.regs.PC + 2) & 0xFFFF, cpu.cycles)
    if is_iy:
        cpu.regs.IYl = value
    else:
        cpu.regs.IXl = value
    return 11


def ld_a_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD A,IXH/IYH - Undocumented (8 T-states)"""
    cpu.regs.A = cpu.regs.IYh if is_iy else cpu.regs.IXh
    return 8


def ld_a_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD A,IXL/IYL - Undocumented (8 T-states)"""
    cpu.regs.A = cpu.regs.IYl if is_iy else cpu.regs.IXl
    return 8


def ld_r_ixh(cpu: "Z80CPU", dest: int, is_iy: bool = False) -> int:
    """LD r,IXH/IYH - Load IXH/IYH into register (8 T-states)"""
    cpu.set_reg8(dest, cpu.regs.IYh if is_iy else cpu.regs.IXh)
    return 8


def ld_r_ixl(cpu: "Z80CPU", dest: int, is_iy: bool = False) -> int:
    """LD r,IXL/IYL - Load IXL/IYL into register (8 T-states)"""
    cpu.set_reg8(dest, cpu.regs.IYl if is_iy else cpu.regs.IXl)
    return 8


def ld_ixh_ixl(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXH,IXL / LD IYH,IYL (8 T-states)"""
    if is_iy:
        cpu.regs.IYh = cpu.regs.IYl
    else:
        cpu.regs.IXh = cpu.regs.IXl
    return 8


def ld_ixl_ixh(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IXL,IXH / LD IYL,IYH (8 T-states)"""
    if is_iy:
        cpu.regs.IYl = cpu.regs.IYh
    else:
        cpu.regs.IXl = cpu.regs.IXh
    return 8


def ld_ixh_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD IXH/IYH,r - Undocumented (8 T-states)"""
    value = cpu.get_reg8(src)
    if is_iy:
        cpu.regs.IYh = value
    else:
        cpu.regs.IXh = value
    return 8


def ld_ixl_r(cpu: "Z80CPU", src: int, is_iy: bool = False) -> int:
    """LD IXL/IYL,r - Undocumented (8 T-states)"""
    value = cpu.get_reg8(src)
    if is_iy:
        cpu.regs.IYl = value
    else:
        cpu.regs.IXl = value
    return 8
