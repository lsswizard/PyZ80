from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Z80CPU
from ..flags import (
    FLAG_C,
    FLAG_F3,
    FLAG_F5,
    FLAG_H,
    FLAG_PV,
    FLAG_S,
    FLAG_Z,
    PARITY_TABLE,
)
from .ld8 import _get_indexed_addr


def _rot_op_0(value: int, carry_flag: int) -> tuple[int, int]:
    carry = (value >> 7) & 1
    return (((value << 1) | carry) & 0xFF, carry)


def _rot_op_1(value: int, carry_flag: int) -> tuple[int, int]:
    carry = value & 1
    return (((value >> 1) | (carry << 7)) & 0xFF, carry)


def _rot_op_2(value: int, carry_flag: int) -> tuple[int, int]:
    carry = (value >> 7) & 1
    return (((value << 1) | carry_flag) & 0xFF, carry)


def _rot_op_3(value: int, carry_flag: int) -> tuple[int, int]:
    carry = value & 1
    return (((value >> 1) | (carry_flag << 7)) & 0xFF, carry)


def _rot_op_4(value: int, carry_flag: int) -> tuple[int, int]:
    carry = (value >> 7) & 1
    return ((value << 1) & 0xFF, carry)


def _rot_op_5(value: int, carry_flag: int) -> tuple[int, int]:
    carry = value & 1
    return (((value >> 1) | (value & 0x80)) & 0xFF, carry)


def _rot_op_6(value: int, carry_flag: int) -> tuple[int, int]:
    carry = (value >> 7) & 1
    return (((value << 1) | 1) & 0xFF, carry)


def _rot_op_7(value: int, carry_flag: int) -> tuple[int, int]:
    carry = value & 1
    return (value >> 1, carry)


_ROT_OPS = (
    _rot_op_0,
    _rot_op_1,
    _rot_op_2,
    _rot_op_3,
    _rot_op_4,
    _rot_op_5,
    _rot_op_6,
    _rot_op_7,
)

__all__ = [
    "rlca",
    "rrca",
    "rla",
    "rra",
    "rlc_r",
    "rlc_hl",
    "rrc_r",
    "rrc_hl",
    "rl_r",
    "rl_hl",
    "rr_r",
    "rr_hl",
    "sla_r",
    "sla_hl",
    "sra_r",
    "sra_hl",
    "sll_r",
    "sll_hl",
    "srl_r",
    "srl_hl",
    "bit_n_r",
    "bit_n_hl",
    "set_n_r",
    "set_n_hl",
    "res_n_r",
    "res_n_hl",
    "_ixycb_bit_n",
    "_ixycb_res_n",
    "_ixycb_set_n",
    "_ixycb_rot",
]


def rlca(cpu: "Z80CPU") -> int:
    """RLCA - Rotate A left circular (4 T-states)"""
    regs = cpu.regs
    a = regs.A
    carry = (a >> 7) & 1
    regs.A = ((a << 1) | carry) & 0xFF
    regs.F = (
        (regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | carry | (regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4


def rrca(cpu: "Z80CPU") -> int:
    """RRCA - Rotate A right circular (4 T-states)"""
    regs = cpu.regs
    a = regs.A
    carry = a & 1
    regs.A = ((a >> 1) | (carry << 7)) & 0xFF
    regs.F = (
        (regs.F & (FLAG_S | FLAG_Z | FLAG_PV)) | carry | (regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4


def rla(cpu: "Z80CPU") -> int:
    """RLA - Rotate A left through carry (4 T-states)"""
    regs = cpu.regs
    a = regs.A
    old_carry = regs.F & FLAG_C
    new_carry = (a >> 7) & 1
    regs.A = ((a << 1) | old_carry) & 0xFF
    regs.F = (
        (regs.F & (FLAG_S | FLAG_Z | FLAG_PV))
        | new_carry
        | (regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4


def rra(cpu: "Z80CPU") -> int:
    """RRA - Rotate A right through carry (4 T-states)"""
    regs = cpu.regs
    a = regs.A
    old_carry = regs.F & FLAG_C
    new_carry = a & 1
    regs.A = ((a >> 1) | (old_carry << 7)) & 0xFF
    regs.F = (
        (regs.F & (FLAG_S | FLAG_Z | FLAG_PV))
        | new_carry
        | (regs.A & (FLAG_F3 | FLAG_F5))
    )
    return 4


def rlc_r(cpu: "Z80CPU", dest: int) -> int:
    """RLC r - Rotate left circular (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = (value >> 7) & 1
    result = ((value << 1) | carry) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def rlc_hl(cpu: "Z80CPU") -> int:
    """RLC (HL) - Rotate left circular memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = (value >> 7) & 1
    result = ((value << 1) | carry) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def rrc_r(cpu: "Z80CPU", dest: int) -> int:
    """RRC r - Rotate right circular (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = value & 1
    result = ((value >> 1) | (carry << 7)) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def rrc_hl(cpu: "Z80CPU") -> int:
    """RRC (HL) - Rotate right circular memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = value & 1
    result = ((value >> 1) | (carry << 7)) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def rl_r(cpu: "Z80CPU", dest: int) -> int:
    """RL r - Rotate left through carry (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    old_carry = regs.F & FLAG_C
    new_carry = (value >> 7) & 1
    result = ((value << 1) | old_carry) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if new_carry:
        regs.F |= FLAG_C
    return 8


def rl_hl(cpu: "Z80CPU") -> int:
    """RL (HL) - Rotate left through carry memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    old_carry = regs.F & FLAG_C
    new_carry = (value >> 7) & 1
    result = ((value << 1) | old_carry) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if new_carry:
        regs.F |= FLAG_C
    return 15


def rr_r(cpu: "Z80CPU", dest: int) -> int:
    """RR r - Rotate right through carry (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    old_carry = (regs.F & FLAG_C) << 7
    new_carry = value & 1
    result = ((value >> 1) | old_carry) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if new_carry:
        regs.F |= FLAG_C
    return 8


def rr_hl(cpu: "Z80CPU") -> int:
    """RR (HL) - Rotate right through carry memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    old_carry = (regs.F & FLAG_C) << 7
    new_carry = value & 1
    result = ((value >> 1) | old_carry) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if new_carry:
        regs.F |= FLAG_C
    return 15


def sla_r(cpu: "Z80CPU", dest: int) -> int:
    """SLA r - Shift left arithmetic (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = (value >> 7) & 1
    result = (value << 1) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def sla_hl(cpu: "Z80CPU") -> int:
    """SLA (HL) - Shift left arithmetic memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = (value >> 7) & 1
    result = (value << 1) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def sra_r(cpu: "Z80CPU", dest: int) -> int:
    """SRA r - Shift right arithmetic (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = value & 1
    result = ((value >> 1) | (value & 0x80)) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def sra_hl(cpu: "Z80CPU") -> int:
    """SRA (HL) - Shift right arithmetic memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = value & 1
    result = ((value >> 1) | (value & 0x80)) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def sll_r(cpu: "Z80CPU", dest: int) -> int:
    """SLL r - Shift left logical (undocumented) (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = (value >> 7) & 1
    result = ((value << 1) | 1) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def sll_hl(cpu: "Z80CPU") -> int:
    """SLL (HL) - Shift left logical memory (undocumented) (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = (value >> 7) & 1
    result = ((value << 1) | 1) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def srl_r(cpu: "Z80CPU", dest: int) -> int:
    """SRL r - Shift right logical (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(dest)
    carry = value & 1
    result = (value >> 1) & 0xFF
    cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 8


def srl_hl(cpu: "Z80CPU") -> int:
    """SRL (HL) - Shift right logical memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    carry = value & 1
    result = (value >> 1) & 0xFF
    cpu._bus_write(addr, result, cycles)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 15


def bit_n_r(cpu: "Z80CPU", bit: int, src: int) -> int:
    """BIT n,r - Test bit n in register (8 T-states)"""
    regs = cpu.regs
    value = cpu.get_reg8(src)
    test_result = value & (1 << bit)
    regs.F = FLAG_H | (regs.F & FLAG_C)
    if test_result == 0:
        regs.F |= FLAG_Z | FLAG_PV
    if bit == 7 and test_result:
        regs.F |= FLAG_S
    regs.F |= value & (FLAG_F3 | FLAG_F5)
    return 8


def bit_n_hl(cpu: "Z80CPU", bit: int) -> int:
    """BIT n,(HL) - Test bit n in memory (12 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    test_result = value & (1 << bit)
    regs.Memptr = addr
    regs.F = FLAG_H | (regs.F & FLAG_C)
    if test_result == 0:
        regs.F |= FLAG_Z | FLAG_PV
    if bit == 7 and test_result:
        regs.F |= FLAG_S
    regs.F |= (addr >> 8) & (FLAG_F3 | FLAG_F5)
    return 12


def set_n_r(cpu: "Z80CPU", bit: int, dest: int) -> int:
    """SET n,r - Set bit n in register (8 T-states)"""
    value = cpu.get_reg8(dest)
    cpu.set_reg8(dest, value | (1 << bit))
    return 8


def set_n_hl(cpu: "Z80CPU", bit: int) -> int:
    """SET n,(HL) - Set bit n in memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    cpu._bus_write(addr, value | (1 << bit), cycles)
    return 15


def res_n_r(cpu: "Z80CPU", bit: int, dest: int) -> int:
    """RES n,r - Reset bit n in register (8 T-states)"""
    value = cpu.get_reg8(dest)
    cpu.set_reg8(dest, value & ~(1 << bit))
    return 8


def res_n_hl(cpu: "Z80CPU", bit: int) -> int:
    """RES n,(HL) - Reset bit n in memory (15 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = regs.HL
    value = cpu._bus_read(addr, cycles)
    cpu._bus_write(addr, value & ~(1 << bit), cycles)
    return 15


def _ixycb_bit_n(cpu: "Z80CPU", bit: int, is_iy: bool) -> int:
    """BIT n,(IX/IY+d) - Test bit n in indexed memory (20 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = _get_indexed_addr(cpu, is_iy)
    value = cpu._bus_read(addr, cycles)
    test_result = value & (1 << bit)
    regs.F = FLAG_H | (regs.F & FLAG_C)
    if test_result == 0:
        regs.F |= FLAG_Z | FLAG_PV
    if bit == 7 and test_result:
        regs.F |= FLAG_S
    regs.F |= (addr >> 8) & (FLAG_F3 | FLAG_F5)
    return 20


def _ixycb_res_n(cpu: "Z80CPU", bit: int, dest: int, is_iy: bool) -> int:
    """RES n,(IX/IY+d) - Reset bit n in indexed memory (23 T-states)"""
    cycles = cpu.cycles
    addr = _get_indexed_addr(cpu, is_iy)
    value = cpu._bus_read(addr, cycles)
    result = value & ~(1 << bit)
    cpu._bus_write(addr, result, cycles)
    if dest != 6:
        cpu.set_reg8(dest, result)
    return 23


def _ixycb_set_n(cpu: "Z80CPU", bit: int, dest: int, is_iy: bool) -> int:
    """SET n,(IX/IY+d) - Set bit n in indexed memory (23 T-states)"""
    cycles = cpu.cycles
    addr = _get_indexed_addr(cpu, is_iy)
    value = cpu._bus_read(addr, cycles)
    result = value | (1 << bit)
    cpu._bus_write(addr, result, cycles)
    if dest != 6:
        cpu.set_reg8(dest, result)
    return 23


def _ixycb_rot(cpu: "Z80CPU", dest: int, op_idx: int, is_iy: bool) -> int:
    """RLC/RRC/RL/RR/SLA/SRA/SLL/SRL (IX/IY+d)[,r] (23 T-states)"""
    regs = cpu.regs
    cycles = cpu.cycles
    addr = _get_indexed_addr(cpu, is_iy)
    value = cpu._bus_read(addr, cycles)
    old_carry = regs.F & FLAG_C
    result, carry = _ROT_OPS[op_idx](value, old_carry)
    cpu._bus_write(addr, result, cycles)
    if dest != 6:
        cpu.set_reg8(dest, result)
    regs.F = result & (FLAG_S | FLAG_F3 | FLAG_F5)
    if result == 0:
        regs.F |= FLAG_Z
    if PARITY_TABLE[result]:
        regs.F |= FLAG_PV
    if carry:
        regs.F |= FLAG_C
    return 23
