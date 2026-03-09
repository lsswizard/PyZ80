from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU

from .ld8 import _read_addr_from_pc


def _push_word(cpu: "Z80CPU", value: int) -> int:
    """Push 16-bit value to stack. Returns new SP."""
    cycles = cpu.cycles
    sp = (cpu.regs.SP - 2) & 0xFFFF
    cpu._bus_write(sp, value & 0xFF, cycles)
    cpu._bus_write((sp + 1) & 0xFFFF, (value >> 8) & 0xFF, cycles)
    cpu.regs.SP = sp
    return sp


def _pop_word(cpu: "Z80CPU") -> tuple[int, int]:
    """Pop 16-bit value from stack. Returns (value, new SP)."""
    sp = cpu.regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles)
    new_sp = (sp + 2) & 0xFFFF
    cpu.regs.SP = new_sp
    return (low | (high << 8)), new_sp


def ld_rr_nn(cpu: "Z80CPU", reg_pair: int) -> int:
    """LD rr,nn - Load 16-bit immediate to register pair (10 T-states)"""
    addr = _read_addr_from_pc(cpu, 1)
    cpu.regs.set_reg16(reg_pair, addr)
    return 10


def ld_hl_nn(cpu: "Z80CPU") -> int:
    """LD HL,(nn) - Load HL from 16-bit address (16 T-states)"""
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles)
    cpu.regs.HL = low | (high << 8)
    return 16


def ld_hl_nn_ed(cpu: "Z80CPU") -> int:
    """ED LD HL,(nn) - Load HL from 16-bit address (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles)
    cpu.regs.HL = low | (high << 8)
    return 20


def ld_nn_hl_ed(cpu: "Z80CPU") -> int:
    """ED LD (nn),HL - Store HL to 16-bit address (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    cpu._bus_write(addr, cpu.regs.L, cycles)
    cpu._bus_write((addr + 1) & 0xFFFF, cpu.regs.H, cycles)
    return 20


def ld_nn_hl(cpu: "Z80CPU") -> int:
    """LD (nn),HL - Store HL to 16-bit address (16 T-states)"""
    addr = _read_addr_from_pc(cpu, 1)
    cycles = cpu.cycles
    cpu._bus_write(addr, cpu.regs.L, cycles)
    cpu._bus_write((addr + 1) & 0xFFFF, cpu.regs.H, cycles)
    return 16


def ld_nn_hl_ed(cpu: "Z80CPU") -> int:
    """ED LD (nn),HL - Store HL to 16-bit address (20 T-states)"""
    pc = cpu.regs.PC
    cycles = cpu.cycles
    low_addr = cpu._bus_read((pc + 2) & 0xFFFF, cycles)
    high_addr = cpu._bus_read((pc + 3) & 0xFFFF, cycles)
    addr = low_addr | (high_addr << 8)
    cpu._bus_write(addr, cpu.regs.L, cycles)
    cpu._bus_write((addr + 1) & 0xFFFF, cpu.regs.H, cycles)
    return 20


def ld_sp_hl(cpu: "Z80CPU") -> int:
    """LD SP,HL - Load HL to SP (6 T-states)"""
    cpu.regs.SP = cpu.regs.HL
    return 6


def push_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """PUSH rr - Push register pair to stack (11 T-states)"""
    value = cpu.regs.get_reg16_push(reg_pair)
    _push_word(cpu, value)
    return 11


def pop_rr(cpu: "Z80CPU", reg_pair: int) -> int:
    """POP rr - Pop register pair from stack (10 T-states)"""
    value, _ = _pop_word(cpu)
    cpu.regs.set_reg16_push(reg_pair, value)
    return 10


def ex_de_hl(cpu: "Z80CPU") -> int:
    """EX DE,HL - Exchange DE and HL (4 T-states)"""
    cpu.regs.DE, cpu.regs.HL = cpu.regs.HL, cpu.regs.DE
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
    sp = cpu.regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles)
    temp = low | (high << 8)
    cpu._bus_write(sp, cpu.regs.L, cycles)
    cpu._bus_write((sp + 1) & 0xFFFF, cpu.regs.H, cycles)
    cpu.regs.HL = temp
    return 19


def ld_rr_nn_ind(cpu: "Z80CPU", dest_attr: str) -> int:
    """LD rr,(nn) - Consolidated (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles)
    setattr(cpu.regs, dest_attr, low | (high << 8))
    return 20


def ld_nn_rr(cpu: "Z80CPU", src_attr: str) -> int:
    """LD (nn),rr - Consolidated (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    val = getattr(cpu.regs, src_attr)
    cpu._bus_write(addr, val & 0xFF, cycles)
    cpu._bus_write((addr + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles)
    return 20


def ld_ix_nn(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IX/IY,nn (14 T-states)"""
    val = _read_addr_from_pc(cpu, 2)
    if is_iy:
        cpu.regs.IY = val
    else:
        cpu.regs.IX = val
    return 14


def ld_ix_nn_ind(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD IX/IY,(nn) (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    low = cpu._bus_read(addr, cycles)
    high = cpu._bus_read((addr + 1) & 0xFFFF, cycles)
    val = low | (high << 8)
    if is_iy:
        cpu.regs.IY = val
    else:
        cpu.regs.IX = val
    return 20


def ld_nn_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD (nn),IX/IY (20 T-states)"""
    addr = _read_addr_from_pc(cpu, 2)
    cycles = cpu.cycles
    val = cpu.regs.IY if is_iy else cpu.regs.IX
    cpu._bus_write(addr, val & 0xFF, cycles)
    cpu._bus_write((addr + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles)
    return 20


def ld_sp_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """LD SP,IX/IY (10 T-states)"""
    cpu.regs.SP = cpu.regs.IY if is_iy else cpu.regs.IX
    return 10


def push_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """PUSH IX/IY (15 T-states)"""
    val = cpu.regs.IY if is_iy else cpu.regs.IX
    _push_word(cpu, val)
    return 15


def pop_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """POP IX/IY (14 T-states)"""
    val, _ = _pop_word(cpu)
    if is_iy:
        cpu.regs.IY = val
    else:
        cpu.regs.IX = val
    return 14


def ex_sp_ix(cpu: "Z80CPU", is_iy: bool = False) -> int:
    """EX (SP),IX/IY (23 T-states)"""
    sp = cpu.regs.SP
    cycles = cpu.cycles
    low = cpu._bus_read(sp, cycles)
    high = cpu._bus_read((sp + 1) & 0xFFFF, cycles)
    temp = low | (high << 8)
    cpu.regs.Memptr = temp
    val = cpu.regs.IY if is_iy else cpu.regs.IX
    cpu._bus_write(sp, val & 0xFF, cycles)
    cpu._bus_write((sp + 1) & 0xFFFF, (val >> 8) & 0xFF, cycles)
    if is_iy:
        cpu.regs.IY = temp
    else:
        cpu.regs.IX = temp
    return 23
