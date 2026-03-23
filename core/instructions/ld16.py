"""
Z80 16-bit Load Instructions

This module implements 16-bit load operations:
    - Immediate: LD rr,nn, LD IX,nn, LD IY,nn
    - Memory: LD (nn),HL, LD HL,(nn), LD (nn),IX, LD IX,(nn), etc.
    - Stack: PUSH rr, POP rr, PUSH IX, POP IX
    - Exchange: EX DE,HL, EX AF,AF', EXX, EX (SP),HL
    - Stack pointer: LD SP,HL, LD SP,IX, LD SP,IY
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU

from .ld8 import _read_addr_from_pc


def _push_word(cpu: "Z80CPU", value: int) -> int:
    """Push 16-bit value to stack. Returns new SP."""
    cycles = cpu.cycles
    sp = (cpu.regs.SP - 2) & 0xFFFF
    cpu._bus_write(sp, value & 0xFF, cycles + 1)
    cpu._bus_write((sp + 1) & 0xFFFF, (value >> 8) & 0xFF, cycles + 4)
    cpu.cycles += 7
    cpu.regs.SP = sp
    return sp


def _pop_word(cpu: "Z80CPU") -> tuple[int, int]:
    """Pop 16-bit value from stack. Returns (value, new SP)."""
    regs = cpu.regs
    sp = regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles + 1)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles + 4)
    new_sp = (sp + 2) & 0xFFFF
    cpu.cycles += 10
    regs.SP = new_sp
    return (low | (high << 8)), new_sp


def ld_rr_nn(cpu: "Z80CPU", reg_pair: int) -> int:
    """LD rr,nn - Load 16-bit immediate to register pair (10 T-states)"""
    cpu.regs.set_reg16(reg_pair, _read_addr_from_pc(cpu, 1))
    return 10


def ld_hl_nn(cpu: "Z80CPU") -> int:
    """LD HL,(nn) - Load HL from 16-bit address (16 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles + 3)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles + 6)
    cpu.cycles += 16
    regs.HL = low | (high << 8)
    return 16


def ld_hl_nn_ed(cpu: "Z80CPU") -> int:
    """ED LD HL,(nn) - Load HL from 16-bit address (20 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles + 4)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles + 7)
    cpu.cycles += 20
    regs.HL = low | (high << 8)
    return 20


def ld_nn_hl(cpu: "Z80CPU") -> int:
    """LD (nn),HL - Store HL to 16-bit address (16 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    cpu._bus_write(addr, regs.L, cycles + 3)
    cpu._bus_write((addr + 1) & 0xFFFF, regs.H, cycles + 6)
    cpu.cycles += 16
    return 16


def ld_nn_hl_ed(cpu: "Z80CPU") -> int:
    """ED LD (nn),HL - Store HL to 16-bit address (20 T-states)"""
    regs = cpu.regs
    pc = regs.PC
    cycles = cpu.cycles
    low_addr = cpu._bus_read((pc + 2) & 0xFFFF, cycles + 1)
    high_addr = cpu._bus_read((pc + 3) & 0xFFFF, cycles + 4)
    addr = low_addr | (high_addr << 8)
    cpu._bus_write(addr, regs.L, cycles + 7)
    cpu._bus_write((addr + 1) & 0xFFFF, regs.H, cycles + 10)
    cpu.cycles += 20
    return 20


def ld_sp_hl(cpu: "Z80CPU") -> int:
    """LD SP,HL - Load HL to SP (6 T-states)"""
    cpu.regs.SP = cpu.regs.HL
    return 6


def push_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """PUSH rr - Push register pair to stack (11 T-states)"""
    _push_word(cpu, cpu.regs.get_reg16_push(reg_pair))
    return 11


def pop_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """POP rr - Pop register pair from stack (10 T-states)"""
    value, _ = _pop_word(cpu)
    cpu.regs.set_reg16_push(reg_pair, value)
    return 10


def ex_de_hl(cpu: "Z80CPU") -> int:
    """EX DE,HL - Exchange DE and HL (4 T-states)"""
    regs = cpu.regs
    regs.DE, regs.HL = regs.HL, regs.DE
    return 4


def ex_af_afp(cpu: "Z80CPU") -> int:
    """EX AF,AF' - Exchange AF with shadow AF (4 T-states)"""
    cpu.regs.swap_shadow()
    return 4


def exx(cpu: "Z80CPU") -> int:
    """EXX - Exchange all main registers with shadow (4 T-states)"""
    cpu.regs.swap_shadow_all()
    return 4


def ex_sp_hl(cpu: "Z80CPU") -> int:
    """EX (SP),HL - Exchange HL with top of stack (19 T-states)"""
    regs = cpu.regs
    sp = regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles + 1)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles + 4)
    temp = low | (high << 8)
    cpu._bus_write(sp, regs.L, cycles + 7)
    cpu._bus_write((sp + 1) & 0xFFFF, regs.H, cycles + 10)
    cpu.cycles += 19
    regs.HL = temp
    return 19


_REG16_IND_GETTERS = {
    "BC": lambda r: (r.B << 8) | r.C,
    "DE": lambda r: (r.D << 8) | r.E,
    "SP": lambda r: r.SP,
}

_REG16_IND_SETTERS = {
    "BC": lambda r, v: setattr(r, "BC", v),
    "DE": lambda r, v: setattr(r, "DE", v),
    "SP": lambda r, v: setattr(r, "SP", v),
}


def ld_rr_nn_ind(cpu: "Z80CPU", dest_attr: str) -> int:
    """LD rr,(nn) - Consolidated (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles + 4)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles + 7)
    cpu.cycles += 20
    _REG16_IND_SETTERS[dest_attr](cpu.regs, low | (high << 8))
    return 20


def ld_nn_rr(cpu: "Z80CPU", src_attr: str) -> int:
    """LD (nn),rr - Consolidated (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    val = _REG16_IND_GETTERS[src_attr](cpu.regs)
    cpu._bus_write(addr, val & 0xFF, cycles + 4)
    cpu._bus_write((addr + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles + 7)
    cpu.cycles += 20
    return 20


def ld_ix_nn(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IX/IY,nn (14 T-states)"""
    regs = cpu.regs
    val = _read_addr_from_pc(cpu, 2)
    if is_iy:
        regs.IY = val
    else:
        regs.IX = val
    return 14


def ld_ix_nn_ind(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IX/IY,(nn) (20 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles + 4)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles + 7)
    val = low | (high << 8)
    cpu.cycles += 20
    if is_iy:
        regs.IY = val
    else:
        regs.IX = val
    return 20


def ld_nn_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD (nn),IX/IY (20 T-states)"""
    regs = cpu.regs
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    val = regs.IY if is_iy else regs.IX
    cpu._bus_write(addr, val & 0xFF, cycles + 4)
    cpu._bus_write((addr + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles + 7)
    cpu.cycles += 20
    return 20


def ld_sp_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD SP,IX/IY (10 T-states)"""
    regs = cpu.regs
    regs.SP = regs.IY if is_iy else regs.IX
    return 10


def push_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """PUSH IX/IY (15 T-states)"""
    regs = cpu.regs
    _push_word(cpu, regs.IY if is_iy else regs.IX)
    return 15


def pop_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """POP IX/IY (14 T-states)"""
    regs = cpu.regs
    val, _ = _pop_word(cpu)
    if is_iy:
        regs.IY = val
    else:
        regs.IX = val
    return 14


def ex_sp_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """EX (SP),IX/IY (23 T-states)"""
    regs = cpu.regs
    sp = regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles + 1)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles + 4)
    temp = low | (high << 8)
    regs.Memptr = temp
    val = regs.IY if is_iy else regs.IX
    cpu._bus_write(sp, val & 0xFF, cycles + 7)
    cpu._bus_write((sp + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles + 10)
    cpu.cycles += 23
    if is_iy:
        regs.IY = temp
    else:
        regs.IX = temp
    return 23
