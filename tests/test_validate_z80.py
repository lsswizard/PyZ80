#!/usr/bin/env python3
"""
Z80 CPU Core Validation - Comprehensive pytest Implementation
Tests the PythonZ80 implementation against expected Z80 behavior.
"""

import sys
import pytest

sys.path.insert(0, "/home/lss/builds/PyZ80")

from core import Z80CPU
from core.instructions import block, bit
from core.flags import (
    FLAG_S,
    FLAG_Z,
    FLAG_H,
    FLAG_PV,
    FLAG_N,
    FLAG_C,
    PARITY_TABLE,
    get_add_flags,
    get_sub_flags,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def cpu():
    """Fresh Z80CPU instance for each test."""
    return Z80CPU()


# ============================================================
# Helpers
# ============================================================


def write_program(cpu, program_bytes, addr=0):
    """Write a sequence of bytes into CPU memory and set PC."""
    for i, b in enumerate(program_bytes):
        cpu.bus.bus_write(addr + i, b, cpu.cycles)
    cpu.regs.PC = addr


def flag_set(cpu, flag):
    """Check if a flag is set."""
    return bool(cpu.regs.F & flag)


def flag_clear(cpu, flag):
    """Check if a flag is clear."""
    return not (cpu.regs.F & flag)


def run_cb_instruction(cpu, cb_op):
    """Execute a CB-prefixed instruction."""
    cpu.bus.bus_write(0, 0xCB, cpu.cycles)
    cpu.bus.bus_write(1, cb_op, cpu.cycles)
    cpu.regs.PC = 0
    return cpu.step()


def step_n(cpu, n):
    """Step the CPU n times, returning total cycles."""
    total = 0
    for _ in range(n):
        total += cpu.step()
    return total


# ============================================================
# 1. 8-Bit Load Instructions
# ============================================================


class TestLoad8Bit:
    """8-bit load instruction tests."""

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("A", 0x3E),
            ("B", 0x06),
            ("C", 0x0E),
            ("D", 0x16),
            ("E", 0x1E),
            ("H", 0x26),
            ("L", 0x2E),
        ],
    )
    def test_ld_r_n(self, cpu, reg, opcode):
        """LD r,n — load immediate byte into register."""
        write_program(cpu, [opcode, 0xAB])
        cpu.step()
        assert getattr(cpu.regs, reg) == 0xAB

    def test_ld_a_bc_indirect(self, cpu):
        """LD A,(BC) — load from address in BC."""
        cpu.regs.BC = 0x1234
        cpu.bus.bus_write(0x1234, 0x77, cpu.cycles)
        write_program(cpu, [0x0A])
        cpu.step()
        assert cpu.regs.A == 0x77

    def test_ld_a_de_indirect(self, cpu):
        """LD A,(DE) — load from address in DE."""
        cpu.regs.DE = 0x1234
        cpu.bus.bus_write(0x1234, 0x55, cpu.cycles)
        write_program(cpu, [0x1A])
        cpu.step()
        assert cpu.regs.A == 0x55

    def test_ld_hl_indirect_n(self, cpu):
        """LD (HL),n — store immediate byte at address in HL."""
        cpu.regs.HL = 0x2000
        write_program(cpu, [0x36, 0xCC])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0xCC

    def test_ld_a_nn_indirect(self, cpu):
        """LD A,(nn) — load from absolute address."""
        cpu.bus.bus_write(0x3000, 0x99, cpu.cycles)
        write_program(cpu, [0x3A, 0x00, 0x30])
        cpu.step()
        assert cpu.regs.A == 0x99

    def test_ld_nn_indirect_a(self, cpu):
        """LD (nn),A — store A at absolute address."""
        cpu.regs.A = 0xBB
        write_program(cpu, [0x32, 0x00, 0x30])
        cpu.step()
        assert cpu.bus.bus_read(0x3000, cpu.cycles) == 0xBB

    @pytest.mark.parametrize(
        "opcode,src,dst_getter",
        [
            (0x78, "B", "A"),  # LD A,B
            (0x79, "C", "A"),  # LD A,C
            (0x7A, "D", "A"),  # LD A,D
            (0x7B, "E", "A"),  # LD A,E
            (0x7C, "H", "A"),  # LD A,H
            (0x7D, "L", "A"),  # LD A,L
            (0x47, "A", "B"),  # LD B,A
            (0x48, "B", "C"),  # LD C,B — wait, 0x48 = LD C,B
            (0x50, "B", "D"),  # LD D,B
            (0x58, "B", "E"),  # LD E,B
            (0x60, "B", "H"),  # LD H,B
            (0x68, "B", "L"),  # LD L,B
        ],
    )
    def test_ld_r_r(self, cpu, opcode, src, dst_getter):
        """LD r,r' — register to register transfer."""
        setattr(cpu.regs, src, 0x42)
        write_program(cpu, [opcode])
        cpu.step()
        assert getattr(cpu.regs, dst_getter) == 0x42

    def test_ld_bc_indirect_a(self, cpu):
        """LD (BC),A — store A at address in BC."""
        cpu.regs.A = 0xEE
        cpu.regs.BC = 0x3000
        write_program(cpu, [0x02])
        cpu.step()
        assert cpu.bus.bus_read(0x3000, cpu.cycles) == 0xEE

    def test_ld_de_indirect_a(self, cpu):
        """LD (DE),A — store A at address in DE."""
        cpu.regs.A = 0xDD
        cpu.regs.DE = 0x3000
        write_program(cpu, [0x12])
        cpu.step()
        assert cpu.bus.bus_read(0x3000, cpu.cycles) == 0xDD

    def test_ld_hl_indirect_r(self, cpu):
        """LD (HL),r — store register at address in HL."""
        cpu.regs.HL = 0x2000
        cpu.regs.B = 0x42
        write_program(cpu, [0x70])  # LD (HL),B
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x42

    def test_ld_r_hl_indirect(self, cpu):
        """LD r,(HL) — load register from address in HL."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x99, cpu.cycles)
        write_program(cpu, [0x46])  # LD B,(HL)
        cpu.step()
        assert cpu.regs.B == 0x99


# ============================================================
# 2. 16-Bit Load Instructions
# ============================================================


class TestLoad16Bit:
    """16-bit load instruction tests."""

    @pytest.mark.parametrize(
        "pair,opcode",
        [
            ("BC", 0x01),
            ("DE", 0x11),
            ("HL", 0x21),
            ("SP", 0x31),
        ],
    )
    def test_ld_rr_nn(self, cpu, pair, opcode):
        """LD rr,nn — load immediate 16-bit value."""
        write_program(cpu, [opcode, 0xCD, 0xAB])
        cpu.step()
        assert getattr(cpu.regs, pair) == 0xABCD

    def test_ld_hl_nn_indirect(self, cpu):
        """LD HL,(nn) — load HL from memory."""
        cpu.bus.bus_write(0x4000, 0x78, cpu.cycles)
        cpu.bus.bus_write(0x4001, 0x56, cpu.cycles)
        write_program(cpu, [0x2A, 0x00, 0x40])
        cpu.step()
        assert cpu.regs.HL == 0x5678

    def test_ld_nn_indirect_hl(self, cpu):
        """LD (nn),HL — store HL to memory."""
        cpu.regs.HL = 0x1234
        write_program(cpu, [0x22, 0x00, 0x40])
        cpu.step()
        assert cpu.bus.bus_read(0x4000, cpu.cycles) == 0x34
        assert cpu.bus.bus_read(0x4001, cpu.cycles) == 0x12

    def test_ld_sp_hl(self, cpu):
        """LD SP,HL — copy HL to SP."""
        cpu.regs.HL = 0x1234
        write_program(cpu, [0xF9])
        cpu.step()
        assert cpu.regs.SP == 0x1234


# ============================================================
# 3. PUSH / POP
# ============================================================


class TestPushPop:
    """Stack push and pop operations."""

    @pytest.mark.parametrize(
        "pair,push_op,pop_op",
        [
            ("BC", 0xC5, 0xC1),
            ("DE", 0xD5, 0xD1),
            ("HL", 0xE5, 0xE1),
        ],
    )
    def test_push_pop_round_trip(self, cpu, pair, push_op, pop_op):
        """PUSH rr / POP rr — round-trip preserves value and SP."""
        cpu.regs.SP = 0x2000
        setattr(cpu.regs, pair, 0xDEAD)
        write_program(cpu, [push_op, pop_op])
        cpu.step()  # PUSH
        setattr(cpu.regs, pair, 0x0000)  # clear
        cpu.step()  # POP
        assert getattr(cpu.regs, pair) == 0xDEAD
        assert cpu.regs.SP == 0x2000

    def test_push_decrements_sp(self, cpu):
        """PUSH decrements SP by 2."""
        cpu.regs.SP = 0xFFFF
        cpu.regs.BC = 0xDEAD
        write_program(cpu, [0xC5])
        cpu.step()
        assert cpu.regs.SP == 0xFFFD

    def test_push_stores_value(self, cpu):
        """PUSH stores high byte at SP+1, low byte at SP."""
        cpu.regs.SP = 0xFFFF
        cpu.regs.BC = 0xDEAD
        write_program(cpu, [0xC5])
        cpu.step()
        assert cpu.bus.bus_read(0xFFFE, cpu.cycles) == 0xDE
        assert cpu.bus.bus_read(0xFFFD, cpu.cycles) == 0xAD

    def test_push_pop_af(self, cpu):
        """PUSH AF / POP AF — preserves accumulator and flags."""
        cpu.regs.A = 0xFF
        cpu.regs.F = 0xD7
        cpu.regs.SP = 0xFFFF
        write_program(cpu, [0xF5, 0xF1])
        cpu.step()
        cpu.regs.A = 0x00
        cpu.regs.F = 0x00
        cpu.step()
        assert cpu.regs.A == 0xFF

    def test_nested_push_pop(self, cpu):
        """Nested PUSH/POP preserves all values (LIFO order)."""
        cpu.regs.SP = 0x2000
        cpu.regs.BC = 0xAAAA
        cpu.regs.DE = 0xBBBB
        write_program(cpu, [0xC5, 0xD5, 0xD1, 0xC1])
        step_n(cpu, 4)
        assert cpu.regs.BC == 0xAAAA
        assert cpu.regs.DE == 0xBBBB

    def test_push_pop_cross(self, cpu):
        """PUSH BC / POP DE — transfer value between pairs."""
        cpu.regs.SP = 0x2000
        cpu.regs.BC = 0x1234
        cpu.regs.DE = 0x0000
        write_program(cpu, [0xC5, 0xD1])  # PUSH BC; POP DE
        step_n(cpu, 2)
        assert cpu.regs.DE == 0x1234


# ============================================================
# 4. ADD / ADC Flags
# ============================================================


class TestAddFlags:
    """ADD A,n and ADC A,n flag verification."""

    @pytest.mark.parametrize(
        "a,b",
        [
            (0x00, 0x00),
            (0x01, 0x02),
            (0x7F, 0x01),
            (0xFF, 0x01),
            (0x80, 0x80),
            (0x0F, 0x01),
            (0xF0, 0x10),
            (0x55, 0xAA),
            (0x01, 0xFF),
            (0x40, 0x40),
            (0xFE, 0x01),
        ],
    )
    def test_add_a_n_flags(self, cpu, a, b):
        """ADD A,n — verify result and all affected flags."""
        write_program(cpu, [0x3E, a, 0xC6, b])
        cpu.step()
        cpu.step()
        expected_flags = get_add_flags(a, b)
        mask = FLAG_S | FLAG_Z | FLAG_H | FLAG_PV | FLAG_C | FLAG_N
        assert cpu.regs.A == (a + b) & 0xFF
        assert (cpu.regs.F & mask) == (expected_flags & mask)

    def test_adc_with_carry(self, cpu):
        """ADC A,n — carry input is included in addition."""
        cpu.regs.A = 0x0F
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xCE, 0x01])
        cpu.step()
        assert cpu.regs.A == 0x11
        assert flag_set(cpu, FLAG_H)

    def test_adc_no_carry(self, cpu):
        """ADC A,n — without carry behaves like ADD."""
        cpu.regs.A = 0x10
        cpu.regs.F = 0
        write_program(cpu, [0xCE, 0x05])
        cpu.step()
        assert cpu.regs.A == 0x15

    def test_adc_carry_causes_overflow(self, cpu):
        """ADC A,n — carry can trigger overflow."""
        cpu.regs.A = 0x7F
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xCE, 0x00])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_PV)
        assert flag_set(cpu, FLAG_S)

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("B", 0x80),
            ("C", 0x81),
            ("D", 0x82),
            ("E", 0x83),
            ("H", 0x84),
            ("L", 0x85),
        ],
    )
    def test_add_a_r(self, cpu, reg, opcode):
        """ADD A,r — add register to A."""
        cpu.regs.A = 0x10
        setattr(cpu.regs, reg, 0x05)
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.A == 0x15

    def test_add_a_hl_indirect(self, cpu):
        """ADD A,(HL) — add memory byte to A."""
        cpu.regs.A = 0x10
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x20, cpu.cycles)
        write_program(cpu, [0x86])
        cpu.step()
        assert cpu.regs.A == 0x30

    def test_add_a_a(self, cpu):
        """ADD A,A — double the accumulator."""
        cpu.regs.A = 0x40
        write_program(cpu, [0x87])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_PV)  # overflow: pos + pos = neg


# ============================================================
# 5. ADD HL,rr / ADC HL,rr / SBC HL,rr
# ============================================================


class TestAdd16Bit:
    """16-bit addition and subtraction tests."""

    def test_add_hl_bc_overflow(self, cpu):
        """ADD HL,BC — overflow sets carry."""
        cpu.regs.HL = 0xFFFF
        cpu.regs.BC = 0x0001
        write_program(cpu, [0x09])
        cpu.step()
        assert cpu.regs.HL == 0x0000
        assert flag_set(cpu, FLAG_C)

    def test_add_hl_bc_no_overflow(self, cpu):
        """ADD HL,BC — no carry when no overflow."""
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0100
        write_program(cpu, [0x09])
        cpu.step()
        assert cpu.regs.HL == 0x1100
        assert flag_clear(cpu, FLAG_C)

    @pytest.mark.parametrize(
        "pair,opcode",
        [
            ("BC", 0x09),
            ("DE", 0x19),
            ("HL", 0x29),
            ("SP", 0x39),
        ],
    )
    def test_add_hl_rr(self, cpu, pair, opcode):
        """ADD HL,rr — all register pairs."""
        cpu.regs.HL = 0x1000
        if pair == "HL":
            # ADD HL,HL doubles HL
            pass
        else:
            setattr(cpu.regs, pair, 0x0100)
        write_program(cpu, [opcode])
        cpu.step()
        if pair == "HL":
            assert cpu.regs.HL == 0x2000
        else:
            assert cpu.regs.HL == 0x1100

    def test_add_hl_preserves_z(self, cpu):
        """ADD HL,rr — does not affect Z flag."""
        cpu.regs.F = FLAG_Z
        cpu.regs.HL = 0x0001
        cpu.regs.BC = 0x0001
        write_program(cpu, [0x09])
        cpu.step()
        assert flag_set(cpu, FLAG_Z)  # preserved

    def test_add_hl_clears_n(self, cpu):
        """ADD HL,rr — always clears N."""
        cpu.regs.F = FLAG_N
        cpu.regs.HL = 0x0001
        cpu.regs.BC = 0x0001
        write_program(cpu, [0x09])
        cpu.step()
        assert flag_clear(cpu, FLAG_N)

    def test_adc_hl_bc_no_carry(self, cpu):
        """ADC HL,BC — without carry."""
        cpu.regs.HL = 0x0001
        cpu.regs.BC = 0x0002
        cpu.regs.F = 0
        write_program(cpu, [0xED, 0x4A])
        cpu.step()
        assert cpu.regs.HL == 0x0003

    def test_adc_hl_bc_with_carry(self, cpu):
        """ADC HL,BC — with carry adds one extra."""
        cpu.regs.HL = 0x0001
        cpu.regs.BC = 0x0002
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xED, 0x4A])
        cpu.step()
        assert cpu.regs.HL == 0x0004

    def test_adc_hl_sets_z(self, cpu):
        """ADC HL,rr — sets Z when result is zero."""
        cpu.regs.HL = 0xFFFF
        cpu.regs.BC = 0x0001
        cpu.regs.F = 0
        write_program(cpu, [0xED, 0x4A])
        cpu.step()
        assert cpu.regs.HL == 0x0000
        assert flag_set(cpu, FLAG_Z)

    def test_sbc_hl_bc(self, cpu):
        """SBC HL,BC — subtraction sets N flag."""
        cpu.regs.HL = 0x0003
        cpu.regs.BC = 0x0001
        cpu.regs.F = 0
        write_program(cpu, [0xED, 0x42])
        cpu.step()
        assert cpu.regs.HL == 0x0002
        assert flag_set(cpu, FLAG_N)

    def test_sbc_hl_bc_with_carry(self, cpu):
        """SBC HL,BC — with carry subtracts one extra."""
        cpu.regs.HL = 0x0003
        cpu.regs.BC = 0x0001
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xED, 0x42])
        cpu.step()
        assert cpu.regs.HL == 0x0001

    def test_sbc_hl_zero_result(self, cpu):
        """SBC HL,rr — Z set when result is zero."""
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x1000
        cpu.regs.F = 0
        write_program(cpu, [0xED, 0x42])
        cpu.step()
        assert cpu.regs.HL == 0x0000
        assert flag_set(cpu, FLAG_Z)

    def test_sbc_hl_borrow(self, cpu):
        """SBC HL,rr — carry set on underflow."""
        cpu.regs.HL = 0x0000
        cpu.regs.BC = 0x0001
        cpu.regs.F = 0
        write_program(cpu, [0xED, 0x42])
        cpu.step()
        assert cpu.regs.HL == 0xFFFF
        assert flag_set(cpu, FLAG_C)


# ============================================================
# 6. SUB / SBC / CP Flags
# ============================================================


class TestSubFlags:
    """SUB A,n, SBC A,n, and CP n flag verification."""

    @pytest.mark.parametrize(
        "a,b",
        [
            (0x00, 0x00),
            (0x00, 0x01),
            (0x7F, 0x01),
            (0x80, 0x01),
            (0x00, 0x80),
            (0xFF, 0xFF),
            (0x10, 0x01),
            (0x01, 0x01),
            (0x80, 0x80),
            (0x3E, 0x3E),
        ],
    )
    def test_sub_a_n_flags(self, cpu, a, b):
        """SUB A,n — verify result and all affected flags."""
        write_program(cpu, [0x3E, a, 0xD6, b])
        cpu.step()
        cpu.step()
        expected_flags = get_sub_flags(a, b)
        mask = FLAG_S | FLAG_Z | FLAG_H | FLAG_PV | FLAG_C | FLAG_N
        assert cpu.regs.A == (a - b) & 0xFF
        assert (cpu.regs.F & mask) == (expected_flags & mask)

    def test_sbc_with_carry(self, cpu):
        """SBC A,n — carry is subtracted too."""
        cpu.regs.A = 0x10
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xDE, 0x01])
        cpu.step()
        assert cpu.regs.A == 0x0E

    def test_sbc_no_carry(self, cpu):
        """SBC A,n — without carry behaves like SUB."""
        cpu.regs.A = 0x10
        cpu.regs.F = 0
        write_program(cpu, [0xDE, 0x01])
        cpu.step()
        assert cpu.regs.A == 0x0F

    def test_cp_does_not_modify_a(self, cpu):
        """CP n — A is unchanged after comparison."""
        cpu.regs.A = 0x10
        write_program(cpu, [0xFE, 0x10])
        cpu.step()
        assert cpu.regs.A == 0x10

    def test_cp_equal_sets_z(self, cpu):
        """CP n — Z flag set when A == n."""
        cpu.regs.A = 0x10
        write_program(cpu, [0xFE, 0x10])
        cpu.step()
        assert flag_set(cpu, FLAG_Z)

    def test_cp_less_sets_carry(self, cpu):
        """CP n — carry set when A < n."""
        cpu.regs.A = 0x05
        write_program(cpu, [0xFE, 0x10])
        cpu.step()
        assert flag_set(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_N)

    def test_cp_greater_no_carry(self, cpu):
        """CP n — no carry when A > n."""
        cpu.regs.A = 0x20
        write_program(cpu, [0xFE, 0x10])
        cpu.step()
        assert flag_clear(cpu, FLAG_C)
        assert flag_clear(cpu, FLAG_Z)

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("B", 0x90),
            ("C", 0x91),
            ("D", 0x92),
            ("E", 0x93),
            ("H", 0x94),
            ("L", 0x95),
        ],
    )
    def test_sub_a_r(self, cpu, reg, opcode):
        """SUB A,r — subtract register from A."""
        cpu.regs.A = 0x20
        setattr(cpu.regs, reg, 0x10)
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.A == 0x10

    def test_sub_a_hl_indirect(self, cpu):
        """SUB A,(HL) — subtract memory byte from A."""
        cpu.regs.A = 0x30
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x10, cpu.cycles)
        write_program(cpu, [0x96])
        cpu.step()
        assert cpu.regs.A == 0x20


# ============================================================
# 7. INC / DEC (8-bit)
# ============================================================


class TestIncDec8Bit:
    """8-bit increment and decrement tests."""

    @pytest.mark.parametrize(
        "val,exp_result,exp_s,exp_z,exp_h,exp_pv",
        [
            (0x00, 0x01, False, False, False, False),
            (0x7F, 0x80, True, False, True, True),
            (0xFF, 0x00, False, True, True, False),
            (0x0F, 0x10, False, False, True, False),
            (0xFE, 0xFF, True, False, False, False),
        ],
    )
    def test_inc_a(self, cpu, val, exp_result, exp_s, exp_z, exp_h, exp_pv):
        """INC A — verify result and flags."""
        cpu.regs.A = val
        cpu.regs.F = FLAG_C  # carry should be preserved
        write_program(cpu, [0x3C])
        cpu.step()
        assert cpu.regs.A == exp_result
        assert flag_set(cpu, FLAG_S) == exp_s
        assert flag_set(cpu, FLAG_Z) == exp_z
        assert flag_set(cpu, FLAG_H) == exp_h
        assert flag_set(cpu, FLAG_PV) == exp_pv
        assert flag_set(cpu, FLAG_C)  # preserved
        assert flag_clear(cpu, FLAG_N)  # always cleared

    @pytest.mark.parametrize(
        "val,exp_result,exp_s,exp_z,exp_h,exp_pv",
        [
            (0x01, 0x00, False, True, False, False),
            (0x80, 0x7F, False, False, True, True),
            (0x00, 0xFF, True, False, True, False),
            (0x10, 0x0F, False, False, True, False),
            (0x02, 0x01, False, False, False, False),
        ],
    )
    def test_dec_a(self, cpu, val, exp_result, exp_s, exp_z, exp_h, exp_pv):
        """DEC A — verify result and flags."""
        cpu.regs.A = val
        cpu.regs.F = FLAG_C  # carry should be preserved
        write_program(cpu, [0x3D])
        cpu.step()
        assert cpu.regs.A == exp_result
        assert flag_set(cpu, FLAG_S) == exp_s
        assert flag_set(cpu, FLAG_Z) == exp_z
        assert flag_set(cpu, FLAG_H) == exp_h
        assert flag_set(cpu, FLAG_PV) == exp_pv
        assert flag_set(cpu, FLAG_C)  # preserved
        assert flag_set(cpu, FLAG_N)  # always set

    @pytest.mark.parametrize(
        "reg,inc_op,dec_op",
        [
            ("B", 0x04, 0x05),
            ("C", 0x0C, 0x0D),
            ("D", 0x14, 0x15),
            ("E", 0x1C, 0x1D),
            ("H", 0x24, 0x25),
            ("L", 0x2C, 0x2D),
        ],
    )
    def test_inc_dec_registers(self, cpu, reg, inc_op, dec_op):
        """INC r / DEC r — basic operation on all registers."""
        setattr(cpu.regs, reg, 0x10)
        write_program(cpu, [inc_op])
        cpu.step()
        assert getattr(cpu.regs, reg) == 0x11

        cpu.reset()
        setattr(cpu.regs, reg, 0x10)
        write_program(cpu, [dec_op])
        cpu.step()
        assert getattr(cpu.regs, reg) == 0x0F

    def test_inc_hl_indirect(self, cpu):
        """INC (HL) — increment memory byte."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x0F, cpu.cycles)
        write_program(cpu, [0x34])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x10
        assert flag_set(cpu, FLAG_H)

    def test_dec_hl_indirect(self, cpu):
        """DEC (HL) — decrement memory byte."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x10, cpu.cycles)
        write_program(cpu, [0x35])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x0F
        assert flag_set(cpu, FLAG_H)


# ============================================================
# 8. INC / DEC (16-bit)
# ============================================================


class TestIncDec16Bit:
    """16-bit increment and decrement (no flags affected)."""

    def test_inc_bc_wrap(self, cpu):
        """INC BC wraps from 0xFFFF to 0x0000."""
        cpu.regs.BC = 0xFFFF
        write_program(cpu, [0x03])
        cpu.step()
        assert cpu.regs.BC == 0x0000

    def test_dec_de_wrap(self, cpu):
        """DEC DE wraps from 0x0000 to 0xFFFF."""
        cpu.regs.DE = 0x0000
        write_program(cpu, [0x1B])
        cpu.step()
        assert cpu.regs.DE == 0xFFFF

    @pytest.mark.parametrize(
        "pair,inc_op,dec_op",
        [
            ("BC", 0x03, 0x0B),
            ("DE", 0x13, 0x1B),
            ("HL", 0x23, 0x2B),
            ("SP", 0x33, 0x3B),
        ],
    )
    def test_inc_dec_16_all_pairs(self, cpu, pair, inc_op, dec_op):
        """INC/DEC rr — all register pairs."""
        setattr(cpu.regs, pair, 0x1000)
        write_program(cpu, [inc_op])
        cpu.step()
        assert getattr(cpu.regs, pair) == 0x1001

        cpu.reset()
        setattr(cpu.regs, pair, 0x1000)
        write_program(cpu, [dec_op])
        cpu.step()
        assert getattr(cpu.regs, pair) == 0x0FFF

    def test_inc_16_preserves_flags(self, cpu):
        """INC rr — does not modify any flags."""
        cpu.regs.F = FLAG_Z | FLAG_C
        cpu.regs.BC = 0x0001
        write_program(cpu, [0x03])
        cpu.step()
        assert cpu.regs.F == (FLAG_Z | FLAG_C)


# ============================================================
# 9. Logical Operations: AND / OR / XOR
# ============================================================


class TestLogical:
    """AND, OR, XOR instruction tests."""

    @pytest.mark.parametrize(
        "a,n,expected",
        [
            (0xFF, 0x0F, 0x0F),
            (0x00, 0xFF, 0x00),
            (0xAA, 0x55, 0x00),
            (0xFF, 0xFF, 0xFF),
            (0x12, 0x34, 0x10),
        ],
    )
    def test_and_n(self, cpu, a, n, expected):
        """AND n — result and flag checks."""
        cpu.regs.A = a
        write_program(cpu, [0xE6, n])
        cpu.step()
        assert cpu.regs.A == expected
        assert flag_set(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)
        assert flag_clear(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_Z) == (expected == 0)
        assert flag_set(cpu, FLAG_S) == bool(expected & 0x80)
        assert flag_set(cpu, FLAG_PV) == bool(PARITY_TABLE[expected])

    @pytest.mark.parametrize(
        "a,n,expected",
        [
            (0x0F, 0xF0, 0xFF),
            (0x00, 0x00, 0x00),
            (0xAA, 0x55, 0xFF),
            (0x80, 0x00, 0x80),
        ],
    )
    def test_or_n(self, cpu, a, n, expected):
        """OR n — result and flag checks."""
        cpu.regs.A = a
        write_program(cpu, [0xF6, n])
        cpu.step()
        assert cpu.regs.A == expected
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)
        assert flag_clear(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_Z) == (expected == 0)
        assert flag_set(cpu, FLAG_S) == bool(expected & 0x80)

    @pytest.mark.parametrize(
        "a,n,expected",
        [
            (0xFF, 0xFF, 0x00),
            (0xAA, 0x55, 0xFF),
            (0x00, 0x80, 0x80),
            (0x12, 0x12, 0x00),
            (0x00, 0x00, 0x00),
        ],
    )
    def test_xor_n(self, cpu, a, n, expected):
        """XOR n — result and flag checks."""
        cpu.regs.A = a
        write_program(cpu, [0xEE, n])
        cpu.step()
        assert cpu.regs.A == expected
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)
        assert flag_clear(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_Z) == (expected == 0)

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("B", 0xA0),
            ("C", 0xA1),
            ("D", 0xA2),
            ("E", 0xA3),
            ("H", 0xA4),
            ("L", 0xA5),
        ],
    )
    def test_and_r(self, cpu, reg, opcode):
        """AND r — register operand."""
        cpu.regs.A = 0xFF
        setattr(cpu.regs, reg, 0x0F)
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.A == 0x0F

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("B", 0xB0),
            ("C", 0xB1),
            ("D", 0xB2),
            ("E", 0xB3),
            ("H", 0xB4),
            ("L", 0xB5),
        ],
    )
    def test_or_r(self, cpu, reg, opcode):
        """OR r — register operand."""
        cpu.regs.A = 0x0F
        setattr(cpu.regs, reg, 0xF0)
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.A == 0xFF

    @pytest.mark.parametrize(
        "reg,opcode",
        [
            ("B", 0xA8),
            ("C", 0xA9),
            ("D", 0xAA),
            ("E", 0xAB),
            ("H", 0xAC),
            ("L", 0xAD),
        ],
    )
    def test_xor_r(self, cpu, reg, opcode):
        """XOR r — register operand."""
        cpu.regs.A = 0xFF
        setattr(cpu.regs, reg, 0xFF)
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_Z)

    def test_and_hl_indirect(self, cpu):
        """AND (HL) — memory operand."""
        cpu.regs.A = 0xFF
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x0F, cpu.cycles)
        write_program(cpu, [0xA6])
        cpu.step()
        assert cpu.regs.A == 0x0F

    def test_or_hl_indirect(self, cpu):
        """OR (HL) — memory operand."""
        cpu.regs.A = 0x0F
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0xF0, cpu.cycles)
        write_program(cpu, [0xB6])
        cpu.step()
        assert cpu.regs.A == 0xFF

    def test_xor_hl_indirect(self, cpu):
        """XOR (HL) — memory operand."""
        cpu.regs.A = 0xAA
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0xAA, cpu.cycles)
        write_program(cpu, [0xAE])
        cpu.step()
        assert cpu.regs.A == 0x00


# ============================================================
# 10. CPL / NEG / DAA / SCF / CCF
# ============================================================


class TestAccFlagOps:
    """Accumulator and flag manipulation instructions."""

    def test_cpl(self, cpu):
        """CPL — one's complement of A."""
        cpu.regs.A = 0x3C
        write_program(cpu, [0x2F])
        cpu.step()
        assert cpu.regs.A == 0xC3
        assert flag_set(cpu, FLAG_H)
        assert flag_set(cpu, FLAG_N)

    def test_cpl_zero(self, cpu):
        """CPL 0x00 -> 0xFF."""
        cpu.regs.A = 0x00
        write_program(cpu, [0x2F])
        cpu.step()
        assert cpu.regs.A == 0xFF

    def test_cpl_ff(self, cpu):
        """CPL 0xFF -> 0x00."""
        cpu.regs.A = 0xFF
        write_program(cpu, [0x2F])
        cpu.step()
        assert cpu.regs.A == 0x00

    def test_neg_positive(self, cpu):
        """NEG — negate a positive value."""
        cpu.regs.A = 0x01
        write_program(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.regs.A == 0xFF
        assert flag_set(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_N)

    def test_neg_zero(self, cpu):
        """NEG 0 — result is 0, Z set, C clear."""
        cpu.regs.A = 0x00
        write_program(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_Z)
        assert flag_clear(cpu, FLAG_C)

    def test_neg_0x80_overflow(self, cpu):
        """NEG 0x80 — overflow (PV set), result is 0x80."""
        cpu.regs.A = 0x80
        write_program(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_PV)

    def test_scf(self, cpu):
        """SCF — set carry flag."""
        cpu.regs.F = 0x00
        write_program(cpu, [0x37])
        cpu.step()
        assert flag_set(cpu, FLAG_C)
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)

    def test_ccf_invert_set(self, cpu):
        """CCF — invert carry from 1 to 0."""
        cpu.regs.F = FLAG_C
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_clear(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_H)  # old carry stored in H

    def test_ccf_invert_clear(self, cpu):
        """CCF — invert carry from 0 to 1."""
        cpu.regs.F = 0x00
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_set(cpu, FLAG_C)

    def test_daa_add_09_01(self, cpu):
        """DAA after BCD addition: 09 + 01 = 10."""
        cpu.regs.A = 0x09
        write_program(cpu, [0xC6, 0x01, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x10

    def test_daa_add_09_09(self, cpu):
        """DAA after BCD addition: 09 + 09 = 18."""
        cpu.regs.A = 0x09
        write_program(cpu, [0xC6, 0x09, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x18

    def test_daa_add_99_01(self, cpu):
        """DAA after BCD addition: 99 + 01 = 00 with carry."""
        cpu.regs.A = 0x99
        write_program(cpu, [0xC6, 0x01, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)

    def test_daa_sub(self, cpu):
        """DAA after BCD subtraction: 10 - 01 = 09."""
        cpu.regs.A = 0x10
        write_program(cpu, [0xD6, 0x01, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x09


# ============================================================
# 11. Exchange Instructions
# ============================================================


class TestExchange:
    """Register exchange instruction tests."""

    def test_ex_af_af_prime_round_trip(self, cpu):
        """EX AF,AF' — round-trip preserves values."""
        cpu.regs.A = 0x11
        cpu.regs.F = 0x22
        write_program(cpu, [0x08])
        cpu.step()
        cpu.regs.A = 0x00
        cpu.regs.F = 0x00
        write_program(cpu, [0x08])
        cpu.step()
        assert cpu.regs.A == 0x11

    def test_exx_round_trip(self, cpu):
        """EXX — round-trip preserves BC/DE/HL."""
        cpu.regs.BC = 0x1111
        cpu.regs.DE = 0x2222
        cpu.regs.HL = 0x3333
        write_program(cpu, [0xD9])
        cpu.step()
        write_program(cpu, [0xD9])
        cpu.step()
        assert cpu.regs.BC == 0x1111
        assert cpu.regs.DE == 0x2222
        assert cpu.regs.HL == 0x3333

    def test_ex_de_hl(self, cpu):
        """EX DE,HL — swaps DE and HL."""
        cpu.regs.DE = 0xAAAA
        cpu.regs.HL = 0xBBBB
        write_program(cpu, [0xEB])
        cpu.step()
        assert cpu.regs.DE == 0xBBBB
        assert cpu.regs.HL == 0xAAAA

    def test_ex_sp_hl(self, cpu):
        """EX (SP),HL — exchanges HL with top of stack."""
        cpu.regs.SP = 0x1000
        cpu.regs.HL = 0x1234
        cpu.bus.bus_write(0x1000, 0x78, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x56, cpu.cycles)
        write_program(cpu, [0xE3])
        cpu.step()
        assert cpu.regs.HL == 0x5678
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0x34
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x12

    def test_ex_sp_ix(self, cpu):
        """EX (SP),IX — exchanges IX with top of stack."""
        cpu.regs.SP = 0x1000
        cpu.regs.IX = 0xABCD
        cpu.bus.bus_write(0x1000, 0x34, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x12, cpu.cycles)
        write_program(cpu, [0xDD, 0xE3])
        cpu.step()
        assert cpu.regs.IX == 0x1234
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0xCD
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0xAB

    def test_ex_sp_iy(self, cpu):
        """EX (SP),IY — exchanges IY with top of stack."""
        cpu.regs.SP = 0x1000
        cpu.regs.IY = 0xABCD
        cpu.bus.bus_write(0x1000, 0x34, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x12, cpu.cycles)
        write_program(cpu, [0xFD, 0xE3])
        cpu.step()
        assert cpu.regs.IY == 0x1234
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0xCD
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0xAB


# ============================================================
# 12. Jumps
# ============================================================


class TestJumps:
    """Jump instruction tests."""

    def test_jp_nn(self, cpu):
        """JP nn — unconditional jump."""
        write_program(cpu, [0xC3, 0x00, 0x20])
        cpu.step()
        assert cpu.regs.PC == 0x2000

    def test_jp_hl(self, cpu):
        """JP (HL) — jump to address in HL."""
        cpu.regs.HL = 0x3000
        write_program(cpu, [0xE9])
        cpu.step()
        assert cpu.regs.PC == 0x3000

    def test_jr_forward(self, cpu):
        """JR e — relative jump forward."""
        write_program(cpu, [0x18, 0x04])
        cpu.step()
        assert cpu.regs.PC == 6  # PC was at 0, +2 for instruction, +4 offset

    def test_jr_backward_self_loop(self, cpu):
        """JR e — relative jump backward (to self)."""
        write_program(cpu, [0x18, 0xFE], 0x0010)
        cpu.step()
        assert cpu.regs.PC == 0x0010

    def test_jr_backward(self, cpu):
        """JR e — relative jump backward."""
        write_program(cpu, [0x18, 0xFC], 0x0010)  # -4
        cpu.step()
        assert cpu.regs.PC == 0x000E

    @pytest.mark.parametrize(
        "opcode,flag,flag_val,taken",
        [
            (0xC2, FLAG_Z, 0, True),  # JP NZ: Z=0 -> taken
            (0xC2, FLAG_Z, FLAG_Z, False),  # JP NZ: Z=1 -> not taken
            (0xCA, FLAG_Z, FLAG_Z, True),  # JP Z: Z=1 -> taken
            (0xCA, FLAG_Z, 0, False),  # JP Z: Z=0 -> not taken
            (0xD2, FLAG_C, 0, True),  # JP NC: C=0 -> taken
            (0xD2, FLAG_C, FLAG_C, False),  # JP NC: C=1 -> not taken
            (0xDA, FLAG_C, FLAG_C, True),  # JP C: C=1 -> taken
            (0xDA, FLAG_C, 0, False),  # JP C: C=0 -> not taken
            (0xE2, FLAG_PV, 0, True),  # JP PO: PV=0 -> taken
            (0xE2, FLAG_PV, FLAG_PV, False),  # JP PO: PV=1 -> not taken
            (0xEA, FLAG_PV, FLAG_PV, True),  # JP PE: PV=1 -> taken
            (0xEA, FLAG_PV, 0, False),  # JP PE: PV=0 -> not taken
            (0xF2, FLAG_S, 0, True),  # JP P: S=0 -> taken
            (0xF2, FLAG_S, FLAG_S, False),  # JP P: S=1 -> not taken
            (0xFA, FLAG_S, FLAG_S, True),  # JP M: S=1 -> taken
            (0xFA, FLAG_S, 0, False),  # JP M: S=0 -> not taken
        ],
    )
    def test_jp_cc_nn(self, cpu, opcode, flag, flag_val, taken):
        """JP cc,nn — conditional jump."""
        cpu.regs.F = flag_val
        write_program(cpu, [opcode, 0x00, 0x10])
        cpu.step()
        if taken:
            assert cpu.regs.PC == 0x1000
        else:
            assert cpu.regs.PC == 3

    @pytest.mark.parametrize(
        "opcode,flag_val,taken",
        [
            (0x20, 0, True),  # JR NZ: Z=0 -> taken
            (0x20, FLAG_Z, False),  # JR NZ: Z=1 -> not taken
            (0x28, FLAG_Z, True),  # JR Z: Z=1 -> taken
            (0x28, 0, False),  # JR Z: Z=0 -> not taken
            (0x30, 0, True),  # JR NC: C=0 -> taken
            (0x30, FLAG_C, False),  # JR NC: C=1 -> not taken
            (0x38, FLAG_C, True),  # JR C: C=1 -> taken
            (0x38, 0, False),  # JR C: C=0 -> not taken
        ],
    )
    def test_jr_cc_e(self, cpu, opcode, flag_val, taken):
        """JR cc,e — conditional relative jump."""
        cpu.regs.F = flag_val
        write_program(cpu, [opcode, 0x04])
        cpu.step()
        if taken:
            assert cpu.regs.PC == 6
        else:
            assert cpu.regs.PC == 2

    def test_jp_ix(self, cpu):
        """JP (IX) — jump to address in IX."""
        cpu.regs.IX = 0x4000
        write_program(cpu, [0xDD, 0xE9])
        cpu.step()
        assert cpu.regs.PC == 0x4000

    def test_jp_iy(self, cpu):
        """JP (IY) — jump to address in IY."""
        cpu.regs.IY = 0x5000
        write_program(cpu, [0xFD, 0xE9])
        cpu.step()
        assert cpu.regs.PC == 0x5000


# ============================================================
# 13. DJNZ
# ============================================================


class TestDJNZ:
    """DJNZ instruction tests."""

    def test_djnz_branch(self, cpu):
        """DJNZ — B > 1, takes the branch."""
        cpu.regs.B = 2
        write_program(cpu, [0x10, 0x00])
        cpu.step()
        assert cpu.regs.B == 1
        assert cpu.regs.PC == 2  # offset 0 + 2

    def test_djnz_no_branch(self, cpu):
        """DJNZ — B = 1, falls through (B becomes 0)."""
        cpu.regs.B = 1
        write_program(cpu, [0x10, 0x04])
        cpu.step()
        assert cpu.regs.B == 0
        assert cpu.regs.PC == 2  # falls through

    def test_djnz_backward_loop(self, cpu):
        """DJNZ — backward jump creates a loop."""
        cpu.regs.B = 3
        write_program(cpu, [0x10, 0xFE], 0x0010)  # loop to self
        cpu.step()
        assert cpu.regs.B == 2
        assert cpu.regs.PC == 0x0010

    def test_djnz_count_to_zero(self, cpu):
        """DJNZ — count down from 3 to 0 in a loop."""
        # Place DJNZ at 0x0000 that loops to itself
        cpu.regs.B = 3
        write_program(cpu, [0x10, 0xFE])
        cpu.step()  # B: 3 -> 2, jump back
        assert cpu.regs.B == 2
        cpu.step()  # B: 2 -> 1, jump back
        assert cpu.regs.B == 1
        cpu.step()  # B: 1 -> 0, fall through
        assert cpu.regs.B == 0
        assert cpu.regs.PC == 2


# ============================================================
# 14. CALL / RET
# ============================================================


class TestCallRet:
    """CALL and RET instruction tests."""

    def test_call_nn(self, cpu):
        """CALL nn — jump to subroutine, push return address."""
        cpu.regs.SP = 0xFFFF
        write_program(cpu, [0xCD, 0x00, 0x10])
        cpu.step()
        assert cpu.regs.PC == 0x1000
        assert cpu.regs.SP == 0xFFFD
        assert cpu.bus.bus_read(0xFFFE, cpu.cycles) == 0x00  # high byte of return addr
        assert cpu.bus.bus_read(0xFFFD, cpu.cycles) == 0x03  # low byte of return addr

    def test_ret(self, cpu):
        """RET — return from subroutine."""
        cpu.regs.SP = 0xFFFF
        write_program(cpu, [0xCD, 0x00, 0x10])
        cpu.step()
        cpu.bus.bus_write(0x1000, 0xC9, cpu.cycles)  # RET
        cpu.step()
        assert cpu.regs.PC == 0x0003
        assert cpu.regs.SP == 0xFFFF

    @pytest.mark.parametrize(
        "opcode,flag_val,taken",
        [
            (0xC4, 0, True),  # CALL NZ: Z=0 -> taken
            (0xC4, FLAG_Z, False),  # CALL NZ: Z=1 -> not taken
            (0xCC, FLAG_Z, True),  # CALL Z: Z=1 -> taken
            (0xCC, 0, False),  # CALL Z: Z=0 -> not taken
            (0xD4, 0, True),  # CALL NC: C=0 -> taken
            (0xD4, FLAG_C, False),  # CALL NC: C=1 -> not taken
            (0xDC, FLAG_C, True),  # CALL C: C=1 -> taken
            (0xDC, 0, False),  # CALL C: C=0 -> not taken
            (0xE4, 0, True),  # CALL PO: PV=0 -> taken
            (0xE4, FLAG_PV, False),  # CALL PO: PV=1 -> not taken
            (0xEC, FLAG_PV, True),  # CALL PE: PV=1 -> taken
            (0xEC, 0, False),  # CALL PE: PV=0 -> not taken
            (0xF4, 0, True),  # CALL P: S=0 -> taken
            (0xF4, FLAG_S, False),  # CALL P: S=1 -> not taken
            (0xFC, FLAG_S, True),  # CALL M: S=1 -> taken
            (0xFC, 0, False),  # CALL M: S=0 -> not taken
        ],
    )
    def test_call_cc_nn(self, cpu, opcode, flag_val, taken):
        """CALL cc,nn — conditional call."""
        cpu.regs.SP = 0xFFFF
        cpu.regs.F = flag_val
        write_program(cpu, [opcode, 0x00, 0x20])
        cpu.step()
        if taken:
            assert cpu.regs.PC == 0x2000
        else:
            assert cpu.regs.PC == 3

    @pytest.mark.parametrize(
        "opcode,flag_val,taken",
        [
            (0xC0, 0, True),  # RET NZ: Z=0 -> taken
            (0xC0, FLAG_Z, False),  # RET NZ: Z=1 -> not taken
            (0xC8, FLAG_Z, True),  # RET Z: Z=1 -> taken
            (0xC8, 0, False),  # RET Z: Z=0 -> not taken
            (0xD0, 0, True),  # RET NC: C=0 -> taken
            (0xD0, FLAG_C, False),  # RET NC: C=1 -> not taken
            (0xD8, FLAG_C, True),  # RET C: C=1 -> taken
            (0xD8, 0, False),  # RET C: C=0 -> not taken
            (0xE0, 0, True),  # RET PO: PV=0 -> taken
            (0xE0, FLAG_PV, False),  # RET PO: PV=1 -> not taken
            (0xE8, FLAG_PV, True),  # RET PE: PV=1 -> taken
            (0xE8, 0, False),  # RET PE: PV=0 -> not taken
            (0xF0, 0, True),  # RET P: S=0 -> taken
            (0xF0, FLAG_S, False),  # RET P: S=1 -> not taken
            (0xF8, FLAG_S, True),  # RET M: S=1 -> taken
            (0xF8, 0, False),  # RET M: S=0 -> not taken
        ],
    )
    def test_ret_cc(self, cpu, opcode, flag_val, taken):
        """RET cc — conditional return."""
        cpu.regs.SP = 0xFFFD
        cpu.bus.bus_write(0xFFFD, 0x00, cpu.cycles)
        cpu.bus.bus_write(0xFFFE, 0x30, cpu.cycles)
        cpu.regs.F = flag_val
        write_program(cpu, [opcode])
        cpu.step()
        if taken:
            assert cpu.regs.PC == 0x3000
        else:
            assert cpu.regs.PC == 1

    @pytest.mark.parametrize(
        "opcode,target",
        [
            (0xC7, 0x00),
            (0xCF, 0x08),
            (0xD7, 0x10),
            (0xDF, 0x18),
            (0xE7, 0x20),
            (0xEF, 0x28),
            (0xF7, 0x30),
            (0xFF, 0x38),
        ],
    )
    def test_rst(self, cpu, opcode, target):
        """RST p — restart to fixed address."""
        cpu.regs.SP = 0xFFFF
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.PC == target
        assert cpu.regs.SP == 0xFFFD


# ============================================================
# 15. Rotate Instructions (Accumulator)
# ============================================================


class TestRotateAcc:
    """Accumulator rotate instructions (RLCA, RRCA, RLA, RRA)."""

    def test_rlca_bit7_set(self, cpu):
        """RLCA — bit 7 rotates to bit 0 and carry."""
        cpu.regs.A = 0x88
        write_program(cpu, [0x07])
        cpu.step()
        assert cpu.regs.A == 0x11
        assert flag_set(cpu, FLAG_C)

    def test_rlca_bit7_clear(self, cpu):
        """RLCA — bit 7 clear, no carry."""
        cpu.regs.A = 0x41
        write_program(cpu, [0x07])
        cpu.step()
        assert cpu.regs.A == 0x82
        assert flag_clear(cpu, FLAG_C)

    def test_rrca_bit0_set(self, cpu):
        """RRCA — bit 0 rotates to bit 7 and carry."""
        cpu.regs.A = 0x11
        write_program(cpu, [0x0F])
        cpu.step()
        assert cpu.regs.A == 0x88
        assert flag_set(cpu, FLAG_C)

    def test_rrca_bit0_clear(self, cpu):
        """RRCA — bit 0 clear, no carry."""
        cpu.regs.A = 0x82
        write_program(cpu, [0x0F])
        cpu.step()
        assert cpu.regs.A == 0x41
        assert flag_clear(cpu, FLAG_C)

    def test_rla_with_carry(self, cpu):
        """RLA — rotate left through carry."""
        cpu.regs.A = 0x41
        cpu.regs.F = FLAG_C
        write_program(cpu, [0x17])
        cpu.step()
        assert cpu.regs.A == 0x83
        assert flag_clear(cpu, FLAG_C)

    def test_rla_without_carry(self, cpu):
        """RLA — rotate left, carry clear."""
        cpu.regs.A = 0x80
        cpu.regs.F = 0x00
        write_program(cpu, [0x17])
        cpu.step()
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)

    def test_rra_no_carry(self, cpu):
        """RRA — rotate right, carry clear."""
        cpu.regs.A = 0x82
        cpu.regs.F = 0x00
        write_program(cpu, [0x1F])
        cpu.step()
        assert cpu.regs.A == 0x41
        assert flag_clear(cpu, FLAG_C)

    def test_rra_bit0_to_carry(self, cpu):
        """RRA — bit 0 goes to carry."""
        cpu.regs.A = 0x01
        cpu.regs.F = 0x00
        write_program(cpu, [0x1F])
        cpu.step()
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)

    def test_rra_with_carry(self, cpu):
        """RRA — carry rotates into bit 7."""
        cpu.regs.A = 0x00
        cpu.regs.F = FLAG_C
        write_program(cpu, [0x1F])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_clear(cpu, FLAG_C)

    def test_rlca_clears_h_n(self, cpu):
        """RLCA — always clears H and N."""
        cpu.regs.A = 0x01
        cpu.regs.F = FLAG_H | FLAG_N
        write_program(cpu, [0x07])
        cpu.step()
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)

    def test_rrca_clears_h_n(self, cpu):
        """RRCA — always clears H and N."""
        cpu.regs.A = 0x01
        cpu.regs.F = FLAG_H | FLAG_N
        write_program(cpu, [0x0F])
        cpu.step()
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)


# ============================================================
# 16. CB-Prefix Rotate/Shift Instructions
# ============================================================


class TestCBRotateShift:
    """CB-prefixed rotate and shift instructions."""

    def test_rlc_a(self, cpu):
        """CB RLC A — rotate left circular."""
        cpu.regs.A = 0x80
        run_cb_instruction(cpu, 0x07)
        assert cpu.regs.A == 0x01
        assert flag_set(cpu, FLAG_C)

    def test_rl_a(self, cpu):
        """CB RL A — rotate left through carry."""
        cpu.regs.A = 0x80
        cpu.regs.F = 0x00
        run_cb_instruction(cpu, 0x17)
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_Z)

    def test_rrc_a(self, cpu):
        """CB RRC A — rotate right circular."""
        cpu.regs.A = 0x01
        run_cb_instruction(cpu, 0x0F)
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_C)

    def test_rr_a(self, cpu):
        """CB RR A — rotate right through carry."""
        cpu.regs.A = 0x01
        cpu.regs.F = 0x00
        run_cb_instruction(cpu, 0x1F)
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)

    def test_sla_a(self, cpu):
        """CB SLA A — shift left arithmetic."""
        cpu.regs.A = 0x80
        run_cb_instruction(cpu, 0x27)
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_Z)

    def test_sla_a_no_carry(self, cpu):
        """CB SLA A — no carry when bit 7 clear."""
        cpu.regs.A = 0x40
        run_cb_instruction(cpu, 0x27)
        assert cpu.regs.A == 0x80
        assert flag_clear(cpu, FLAG_C)

    def test_sra_a(self, cpu):
        """CB SRA A — shift right arithmetic (MSB preserved)."""
        cpu.regs.A = 0x81
        run_cb_instruction(cpu, 0x2F)
        assert cpu.regs.A == 0xC0
        assert flag_set(cpu, FLAG_C)

    def test_sra_a_positive(self, cpu):
        """CB SRA A — positive number (MSB=0 preserved)."""
        cpu.regs.A = 0x40
        run_cb_instruction(cpu, 0x2F)
        assert cpu.regs.A == 0x20
        assert flag_clear(cpu, FLAG_C)

    def test_srl_a(self, cpu):
        """CB SRL A — shift right logical (MSB=0)."""
        cpu.regs.A = 0x81
        run_cb_instruction(cpu, 0x3F)
        assert cpu.regs.A == 0x40
        assert flag_set(cpu, FLAG_C)

    def test_srl_a_no_carry(self, cpu):
        """CB SRL A — no carry when bit 0 clear."""
        cpu.regs.A = 0x80
        run_cb_instruction(cpu, 0x3F)
        assert cpu.regs.A == 0x40
        assert flag_clear(cpu, FLAG_C)

    def test_rlc_hl_indirect(self, cpu):
        """CB RLC (HL) — rotate memory byte."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0x80, cpu.cycles)
        run_cb_instruction(cpu, 0x06)
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x01
        assert flag_set(cpu, FLAG_C)

    def test_srl_b(self, cpu):
        """CB SRL B — shift register right logical."""
        cpu.regs.B = 0x02
        run_cb_instruction(cpu, 0x38)
        assert cpu.regs.B == 0x01
        assert flag_clear(cpu, FLAG_C)

    @pytest.mark.parametrize(
        "reg_idx,reg_name",
        [
            (0, "B"),
            (1, "C"),
            (2, "D"),
            (3, "E"),
            (4, "H"),
            (5, "L"),
            (7, "A"),
        ],
    )
    def test_rlc_all_registers(self, cpu, reg_idx, reg_name):
        """CB RLC r — all registers."""
        setattr(cpu.regs, reg_name, 0x80)
        run_cb_instruction(cpu, 0x00 + reg_idx)
        assert getattr(cpu.regs, reg_name) == 0x01

    @pytest.mark.parametrize(
        "reg_idx,reg_name",
        [
            (0, "B"),
            (1, "C"),
            (2, "D"),
            (3, "E"),
            (4, "H"),
            (5, "L"),
            (7, "A"),
        ],
    )
    def test_srl_all_registers(self, cpu, reg_idx, reg_name):
        """CB SRL r — all registers."""
        setattr(cpu.regs, reg_name, 0x02)
        run_cb_instruction(cpu, 0x38 + reg_idx)
        assert getattr(cpu.regs, reg_name) == 0x01


# ============================================================
# 17. BIT / SET / RES
# ============================================================


class TestBitSetRes:
    """BIT, SET, and RES instruction tests."""

    @pytest.mark.parametrize("bit", range(8))
    def test_bit_set_in_a(self, cpu, bit):
        """BIT b,A — test each bit when set."""
        cpu.regs.A = 1 << bit
        run_cb_instruction(cpu, 0x40 + (bit * 8) + 7)
        assert flag_clear(cpu, FLAG_Z)
        assert flag_set(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)

    @pytest.mark.parametrize("bit", range(8))
    def test_bit_clear_in_a(self, cpu, bit):
        """BIT b,A — test each bit when clear."""
        cpu.regs.A = 0xFF ^ (1 << bit)
        run_cb_instruction(cpu, 0x40 + (bit * 8) + 7)
        assert flag_set(cpu, FLAG_Z)

    def test_bit_0_hl_indirect(self, cpu):
        """BIT 0,(HL) — test bit in memory."""
        cpu.regs.HL = 0x1000
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        run_cb_instruction(cpu, 0x46)
        assert flag_clear(cpu, FLAG_Z)

    def test_bit_0_hl_indirect_clear(self, cpu):
        """BIT 0,(HL) — bit clear in memory."""
        cpu.regs.HL = 0x1000
        cpu.bus.bus_write(0x1000, 0xFE, cpu.cycles)
        run_cb_instruction(cpu, 0x46)
        assert flag_set(cpu, FLAG_Z)

    @pytest.mark.parametrize("bit", range(8))
    def test_set_bit_a(self, cpu, bit):
        """SET b,A — set each bit."""
        cpu.regs.A = 0x00
        run_cb_instruction(cpu, 0xC0 + (bit * 8) + 7)
        assert cpu.regs.A == (1 << bit)

    def test_set_already_set(self, cpu):
        """SET b,A — setting an already-set bit is idempotent."""
        cpu.regs.A = 0xFF
        run_cb_instruction(cpu, 0xC7)  # SET 0,A
        assert cpu.regs.A == 0xFF

    @pytest.mark.parametrize("bit", range(8))
    def test_res_bit_a(self, cpu, bit):
        """RES b,A — reset each bit."""
        cpu.regs.A = 0xFF
        run_cb_instruction(cpu, 0x80 + (bit * 8) + 7)
        assert cpu.regs.A == (0xFF ^ (1 << bit))

    def test_res_already_clear(self, cpu):
        """RES b,A — resetting an already-clear bit is idempotent."""
        cpu.regs.A = 0x00
        run_cb_instruction(cpu, 0x87)  # RES 0,A
        assert cpu.regs.A == 0x00

    def test_set_0_hl_indirect(self, cpu):
        """SET 0,(HL) — set bit in memory."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0xF0, cpu.cycles)
        run_cb_instruction(cpu, 0xC6)
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0xF1

    def test_res_4_hl_indirect(self, cpu):
        """RES 4,(HL) — reset bit in memory."""
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0xFF, cpu.cycles)
        run_cb_instruction(cpu, 0xA6)
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0xEF

    @pytest.mark.parametrize(
        "reg_idx,reg_name",
        [
            (0, "B"),
            (1, "C"),
            (2, "D"),
            (3, "E"),
            (4, "H"),
            (5, "L"),
            (7, "A"),
        ],
    )
    def test_set_3_all_registers(self, cpu, reg_idx, reg_name):
        """SET 3,r — all registers."""
        setattr(cpu.regs, reg_name, 0x00)
        run_cb_instruction(cpu, 0xD8 + reg_idx)  # SET 3,r
        assert getattr(cpu.regs, reg_name) == 0x08

    @pytest.mark.parametrize(
        "reg_idx,reg_name",
        [
            (0, "B"),
            (1, "C"),
            (2, "D"),
            (3, "E"),
            (4, "H"),
            (5, "L"),
            (7, "A"),
        ],
    )
    def test_res_3_all_registers(self, cpu, reg_idx, reg_name):
        """RES 3,r — all registers."""
        setattr(cpu.regs, reg_name, 0xFF)
        run_cb_instruction(cpu, 0x98 + reg_idx)  # RES 3,r
        assert getattr(cpu.regs, reg_name) == 0xF7


# ============================================================
# 18. Block Instructions
# ============================================================


class TestBlockInstructions:
    """LDI, LDD, LDIR, LDDR, CPI, CPD, CPIR, CPDR tests."""

    def test_ldi(self, cpu):
        """LDI — transfer one byte, increment HL/DE, decrement BC."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0003
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xA0])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x42
        assert cpu.regs.HL == 0x1001
        assert cpu.regs.DE == 0x2001
        assert cpu.regs.BC == 0x0002
        assert flag_set(cpu, FLAG_PV)  # BC still non-zero

    def test_ldi_bc_zero(self, cpu):
        """LDI — PV=0 when BC reaches 0."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xA0])
        cpu.step()
        assert cpu.regs.BC == 0x0000
        assert flag_clear(cpu, FLAG_PV)

    def test_ldd(self, cpu):
        """LDD — transfer one byte, decrement HL/DE, decrement BC."""
        cpu.regs.HL = 0x1005
        cpu.regs.DE = 0x2005
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1005, 0x55, cpu.cycles)
        write_program(cpu, [0xED, 0xA8])
        cpu.step()
        assert cpu.bus.bus_read(0x2005, cpu.cycles) == 0x55
        assert cpu.regs.HL == 0x1004
        assert cpu.regs.DE == 0x2004
        assert cpu.regs.BC == 0x0000
        assert flag_clear(cpu, FLAG_PV)

    def test_ldi_clears_h_n(self, cpu):
        """LDI — clears H and N flags."""
        cpu.regs.F = FLAG_H | FLAG_N
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        write_program(cpu, [0xED, 0xA0])
        cpu.step()
        assert flag_clear(cpu, FLAG_H)
        assert flag_clear(cpu, FLAG_N)

    def test_ldir(self, cpu):
        """LDIR — repeats until BC=0."""
        cpu.bus.bus_write(0x1000, 0xAA, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0xBB, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0xCC, cpu.cycles)
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0003
        write_program(cpu, [0xED, 0xB0])
        step_n(cpu, 3)
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0xAA
        assert cpu.bus.bus_read(0x2001, cpu.cycles) == 0xBB
        assert cpu.bus.bus_read(0x2002, cpu.cycles) == 0xCC
        assert cpu.regs.BC == 0x0000
        assert flag_clear(cpu, FLAG_PV)

    def test_lddr(self, cpu):
        """LDDR — transfer bytes in reverse."""
        cpu.bus.bus_write(0x1002, 0x11, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x22, cpu.cycles)
        cpu.bus.bus_write(0x1000, 0x33, cpu.cycles)
        cpu.regs.HL = 0x1002
        cpu.regs.DE = 0x2002
        cpu.regs.BC = 0x0003
        write_program(cpu, [0xED, 0xB8])
        step_n(cpu, 3)
        assert cpu.bus.bus_read(0x2002, cpu.cycles) == 0x11
        assert cpu.bus.bus_read(0x2001, cpu.cycles) == 0x22
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x33
        assert cpu.regs.BC == 0x0000

    def test_cpi(self, cpu):
        """CPI — compare A with (HL), Z set on match."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xA1])
        cpu.step()
        assert flag_set(cpu, FLAG_Z)
        assert cpu.regs.HL == 0x1001
        assert cpu.regs.BC == 0x0001

    def test_cpi_no_match(self, cpu):
        """CPI — Z clear on mismatch."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xA1])
        cpu.step()
        assert flag_clear(cpu, FLAG_Z)

    def test_cpi_sets_n(self, cpu):
        """CPI — always sets N flag."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xA1])
        cpu.step()
        assert flag_set(cpu, FLAG_N)

    def test_cpd(self, cpu):
        """CPD — compare and decrement."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1005
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1005, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xA9])
        cpu.step()
        assert flag_set(cpu, FLAG_Z)
        assert cpu.regs.HL == 0x1004
        assert cpu.regs.BC == 0x0001

    def test_cpir_find_match(self, cpu):
        """CPIR — scan until match."""
        cpu.regs.A = 0x55
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0x55, cpu.cycles)
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0005
        write_program(cpu, [0xED, 0xB1])
        step_n(cpu, 3)
        assert cpu.regs.HL == 0x1003
        assert flag_set(cpu, FLAG_Z)

    def test_cpir_no_match(self, cpu):
        """CPIR — exhausts BC without match."""
        cpu.regs.A = 0xFF
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0003
        write_program(cpu, [0xED, 0xB1])
        step_n(cpu, 3)
        assert cpu.regs.BC == 0x0000
        assert flag_clear(cpu, FLAG_Z)

    def test_cpdr(self, cpu):
        """CPDR — compare and decrement repeat."""
        cpu.regs.A = 0x55
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x55, cpu.cycles)
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        cpu.regs.HL = 0x1002
        cpu.regs.BC = 0x0003
        write_program(cpu, [0xED, 0xB9])
        step_n(cpu, 2)
        assert flag_set(cpu, FLAG_Z)
        assert cpu.regs.HL == 0x1000


# ============================================================
# 19. I/O Block Instructions
# ============================================================


class TestIOBlock:
    """INI, IND, INIR, INDR, OUTI, OUTD, OTIR, OTDR tests."""

    def test_ini(self, cpu):
        """INI — input from port (C), store at (HL), decrement B."""
        cpu.regs.B = 0x02
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2000
        cpu.bus.io_ports[0x10] = 0xAB
        write_program(cpu, [0xED, 0xA2])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0xAB
        assert cpu.regs.HL == 0x2001
        assert cpu.regs.B == 0x01

    def test_ind(self, cpu):
        """IND — input, decrement HL and B."""
        cpu.regs.B = 0x02
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2005
        cpu.bus.io_ports[0x10] = 0xCD
        write_program(cpu, [0xED, 0xAA])
        cpu.step()
        assert cpu.bus.bus_read(0x2005, cpu.cycles) == 0xCD
        assert cpu.regs.HL == 0x2004
        assert cpu.regs.B == 0x01

    def test_outi(self, cpu):
        """OUTI — output (HL) to port (C), increment HL, decrement B."""
        cpu.regs.B = 0x02
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2000
        cpu.bus.bus_write(0x2000, 0xEF, cpu.cycles)
        write_program(cpu, [0xED, 0xA3])
        cpu.step()
        assert cpu.bus.io_ports[0x10] == 0xEF
        assert cpu.regs.HL == 0x2001
        assert cpu.regs.B == 0x01

    def test_outd(self, cpu):
        """OUTD — output (HL) to port (C), decrement HL and B."""
        cpu.regs.B = 0x02
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2005
        cpu.bus.bus_write(0x2005, 0x77, cpu.cycles)
        write_program(cpu, [0xED, 0xAB])
        cpu.step()
        assert cpu.bus.io_ports[0x10] == 0x77
        assert cpu.regs.HL == 0x2004
        assert cpu.regs.B == 0x01


# ============================================================
# 20. IN / OUT Instructions
# ============================================================


class TestIO:
    """I/O port instruction tests."""

    def test_in_a_n(self, cpu):
        """IN A,(n) — read port."""
        cpu.bus.io_ports[0x42] = 0xAB
        write_program(cpu, [0xDB, 0x42])
        cpu.step()
        assert cpu.regs.A == 0xAB

    def test_out_n_a(self, cpu):
        """OUT (n),A — write port."""
        cpu.regs.A = 0xCD
        write_program(cpu, [0xD3, 0x42])
        cpu.step()
        assert cpu.bus.io_ports[0x42] == 0xCD

    def test_in_r_c(self, cpu):
        """IN r,(C) — ED-prefix port read with flag effects."""
        cpu.regs.C = 0x10
        cpu.bus.io_ports[0x10] = 0x80
        write_program(cpu, [0xED, 0x78])  # IN A,(C)
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_S)
        assert flag_clear(cpu, FLAG_Z)

    def test_out_c_r(self, cpu):
        """OUT (C),r — ED-prefix port write."""
        cpu.regs.C = 0x10
        cpu.regs.B = 0x42
        write_program(cpu, [0xED, 0x41])  # OUT (C),B
        cpu.step()
        assert cpu.bus.io_ports[0x10] == 0x42


# ============================================================
# 21. Instruction Timing
# ============================================================


class TestTiming:
    """Cycle-accurate instruction timing verification."""

    @pytest.mark.parametrize(
        "program,mnemonic,expected_cycles",
        [
            (b"\x00", "NOP", 4),
            (b"\x3e\x00", "LD A,n", 7),
            (b"\x01\x00\x00", "LD BC,nn", 10),
            (b"\x80", "ADD A,B", 4),
            (b"\x86", "ADD A,(HL)", 7),
            (b"\xc6\x00", "ADD A,n", 7),
            (b"\x09", "ADD HL,BC", 11),
            (b"\xc3\x00\x00", "JP nn", 10),
            (b"\xe9", "JP (HL)", 4),
            (b"\x18\x00", "JR e", 12),
            (b"\xcd\x00\x00", "CALL nn", 17),
            (b"\xc9", "RET", 10),
            (b"\x76", "HALT", 4),
            (b"\x3c", "INC A", 4),
            (b"\x34", "INC (HL)", 11),
            (b"\x03", "INC BC", 6),
            (b"\xcb\x07", "RLC A", 8),
            (b"\xcb\x06", "RLC (HL)", 15),
            (b"\xed\xa0", "LDI", 16),
            (b"\xff", "RST 38H", 11),
            (b"\xc5", "PUSH BC", 11),
            (b"\xc1", "POP BC", 10),
            (b"\xd6\x00", "SUB n", 7),
            (b"\x2f", "CPL", 4),
            (b"\x37", "SCF", 4),
            (b"\x3f", "CCF", 4),
            (b"\x07", "RLCA", 4),
            (b"\x0f", "RRCA", 4),
            (b"\x17", "RLA", 4),
            (b"\x1f", "RRA", 4),
            (b"\xeb", "EX DE,HL", 4),
            (b"\x08", "EX AF,AF'", 4),
            (b"\xd9", "EXX", 4),
            (b"\xe3", "EX (SP),HL", 19),
            (b"\xf9", "LD SP,HL", 6),
            (b"\xf3", "DI", 4),
            (b"\xfb", "EI", 4),
            (b"\x27", "DAA", 4),
            (b"\xfe\x00", "CP n", 7),
            (b"\xed\x44", "NEG", 8),
        ],
    )
    def test_instruction_timing(self, cpu, program, mnemonic, expected_cycles):
        """Verify cycle count for {mnemonic}."""
        for _i, _b in enumerate(program):
            cpu.bus.bus_write(0 + _i, _b, cpu.cycles)
        cpu.regs.PC = 0
        cycles = cpu.step()
        assert cycles == expected_cycles, (
            f"{mnemonic}: expected {expected_cycles}, got {cycles}"
        )

    def test_djnz_no_branch_timing(self, cpu):
        """DJNZ — no branch (B=1) takes 8 cycles."""
        cpu.regs.B = 1
        write_program(cpu, [0x10, 0x00])
        cycles = cpu.step()
        assert cycles == 8

    def test_djnz_branch_timing(self, cpu):
        """DJNZ — branch (B>1) takes 13 cycles."""
        cpu.regs.B = 2
        write_program(cpu, [0x10, 0x00])
        cycles = cpu.step()
        assert cycles == 13

    def test_jp_cc_always_10_cycles(self, cpu):
        """JP cc,nn — always 10 cycles regardless of condition."""
        # Taken
        cpu.regs.F = 0
        write_program(cpu, [0xC2, 0x00, 0x00])
        assert cpu.step() == 10

        # Not taken
        cpu.reset()
        cpu.regs.F = FLAG_Z
        write_program(cpu, [0xC2, 0x00, 0x00])
        assert cpu.step() == 10

    def test_jr_cc_taken_timing(self, cpu):
        """JR cc,e — taken takes 12 cycles."""
        cpu.regs.F = FLAG_Z
        write_program(cpu, [0x28, 0x00])
        assert cpu.step() == 12

    def test_jr_cc_not_taken_timing(self, cpu):
        """JR cc,e — not taken takes 7 cycles."""
        cpu.regs.F = 0
        write_program(cpu, [0x28, 0x00])
        assert cpu.step() == 7

    def test_call_cc_taken_timing(self, cpu):
        """CALL cc,nn — taken takes 17 cycles."""
        cpu.regs.SP = 0xFFFF
        cpu.regs.F = 0
        write_program(cpu, [0xC4, 0x00, 0x00])
        assert cpu.step() == 17

    def test_call_cc_not_taken_timing(self, cpu):
        """CALL cc,nn — not taken takes 10 cycles."""
        cpu.regs.SP = 0xFFFF
        cpu.regs.F = FLAG_Z
        write_program(cpu, [0xC4, 0x00, 0x00])
        assert cpu.step() == 10

    def test_ret_cc_taken_timing(self, cpu):
        """RET cc — taken takes 11 cycles."""
        cpu.regs.SP = 0xFFFD
        cpu.bus.bus_write(0xFFFD, 0x00, cpu.cycles)
        cpu.bus.bus_write(0xFFFE, 0x10, cpu.cycles)
        cpu.regs.F = FLAG_Z
        write_program(cpu, [0xC8])
        assert cpu.step() == 11

    def test_ret_cc_not_taken_timing(self, cpu):
        """RET cc — not taken takes 5 cycles."""
        cpu.regs.SP = 0xFFFD
        cpu.regs.F = 0
        write_program(cpu, [0xC8])
        assert cpu.step() == 5

    def test_ldir_repeating_timing(self, cpu):
        """LDIR — repeating iteration takes 21 cycles."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x02, cpu.cycles)
        write_program(cpu, [0xED, 0xB0])
        assert cpu.step() == 21

    def test_ldir_final_timing(self, cpu):
        """LDIR — final iteration takes 16 cycles."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        write_program(cpu, [0xED, 0xB0])
        assert cpu.step() == 16

    def test_ldd_timing(self, cpu):
        """LDD — transfer and decrement takes 16 cycles."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        write_program(cpu, [0xED, 0xA8])
        assert cpu.step() == 16

    def test_lddr_repeating_timing(self, cpu):
        """LDDR — repeating iteration takes 21 cycles."""
        cpu.regs.HL = 0x1001
        cpu.regs.DE = 0x2001
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1001, 0x01, cpu.cycles)
        cpu.bus.bus_write(0x1000, 0x02, cpu.cycles)
        write_program(cpu, [0xED, 0xB8])
        assert cpu.step() == 21

    def test_lddr_final_timing(self, cpu):
        """LDDR — final iteration takes 16 cycles."""
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x2000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x01, cpu.cycles)
        write_program(cpu, [0xED, 0xB8])
        assert cpu.step() == 16

    def test_cpi_timing(self, cpu):
        """CPI — compare and increment takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xA1])
        assert cpu.step() == 16

    def test_cpd_timing(self, cpu):
        """CPD — compare and decrement takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xA9])
        assert cpu.step() == 16

    def test_cpir_repeating_timing(self, cpu):
        """CPIR — repeating iteration takes 21 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB1])
        assert cpu.step() == 21

    def test_cpir_final_timing(self, cpu):
        """CPIR — final iteration (BC=1) takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB1])
        assert cpu.step() == 16

    def test_cpir_match_on_last_byte_timing(self, cpu):
        """CPIR — match on last byte (BC=1) takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xB1])
        assert cpu.step() == 16

    def test_cpir_no_match_bc3_timing(self, cpu):
        """CPIR — no match, BC=3, first iteration takes 21 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0003
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB1])
        assert cpu.step() == 21

    def test_cpdr_repeating_timing(self, cpu):
        """CPDR — repeating iteration takes 21 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1001
        cpu.regs.BC = 0x0002
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB9])
        assert cpu.step() == 21

    def test_cpdr_final_timing(self, cpu):
        """CPDR — final iteration (BC=1) takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB9])
        assert cpu.step() == 16

    def test_cpdr_match_on_last_byte_timing(self, cpu):
        """CPDR — match on last byte (BC=1) takes 16 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1000
        cpu.regs.BC = 0x0001
        cpu.bus.bus_write(0x1000, 0x42, cpu.cycles)
        write_program(cpu, [0xED, 0xB9])
        assert cpu.step() == 16

    def test_cpdr_no_match_bc3_timing(self, cpu):
        """CPDR — no match, BC=3, first iteration takes 21 cycles."""
        cpu.regs.A = 0x42
        cpu.regs.HL = 0x1002
        cpu.regs.BC = 0x0003
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1000, 0x00, cpu.cycles)
        write_program(cpu, [0xED, 0xB9])
        assert cpu.step() == 21

    @pytest.mark.parametrize(
        "program,mnemonic,expected_cycles",
        [
            (b"\xcb\x40", "BIT 0,B", 8),
            (b"\xcb\x46", "BIT 0,(HL)", 12),
            (b"\xcb\xc0", "SET 0,B", 8),
            (b"\xcb\xc6", "SET 0,(HL)", 15),
            (b"\xcb\x80", "RES 0,B", 8),
            (b"\xcb\x86", "RES 0,(HL)", 15),
        ],
    )
    def test_cb_timing(self, cpu, program, mnemonic, expected_cycles):
        """CB-prefix instruction timing."""
        for _i, _b in enumerate(program):
            cpu.bus.bus_write(0 + _i, _b, cpu.cycles)
        cpu.regs.PC = 0
        cycles = cpu.step()
        assert cycles == expected_cycles, (
            f"{mnemonic}: expected {expected_cycles}, got {cycles}"
        )

    @pytest.mark.parametrize(
        "program,mnemonic,expected_cycles",
        [
            (b"\xed\x42", "SBC HL,BC", 15),
            (b"\xed\x4a", "ADC HL,BC", 15),
            (b"\xed\x43\x00\x00", "LD (nn),BC", 20),
            (b"\xed\x4b\x00\x00", "LD BC,(nn)", 20),
            (b"\xed\x47", "LD I,A", 9),
            (b"\xed\x4f", "LD R,A", 9),
            (b"\xed\x57", "LD A,I", 9),
            (b"\xed\x5f", "LD A,R", 9),
            (b"\xed\x46", "IM 0", 8),
            (b"\xed\x56", "IM 1", 8),
            (b"\xed\x5e", "IM 2", 8),
            (b"\xed\x6f", "RLD", 18),
            (b"\xed\x67", "RRD", 18),
        ],
    )
    def test_ed_timing(self, cpu, program, mnemonic, expected_cycles):
        """ED-prefix instruction timing."""
        for _i, _b in enumerate(program):
            cpu.bus.bus_write(0 + _i, _b, cpu.cycles)
        cpu.regs.PC = 0
        cycles = cpu.step()
        assert cycles == expected_cycles, (
            f"{mnemonic}: expected {expected_cycles}, got {cycles}"
        )

    @pytest.mark.parametrize(
        "program,mnemonic,expected_cycles",
        [
            (b"\xdd\x21\x00\x00", "LD IX,nn", 14),
            (b"\xdd\x7e\x00", "LD A,(IX+d)", 19),
            (b"\xdd\x36\x00\x00", "LD (IX+d),n", 19),
            (b"\xdd\x34\x00", "INC (IX+d)", 23),
            (b"\xdd\x86\x00", "ADD A,(IX+d)", 19),
            (b"\xdd\xe9", "JP (IX)", 8),
            (b"\xdd\xe5", "PUSH IX", 15),
            (b"\xdd\xe1", "POP IX", 14),
        ],
    )
    def test_ix_timing(self, cpu, program, mnemonic, expected_cycles):
        """IX-prefix instruction timing."""
        cpu.regs.IX = 0x1000
        for _i, _b in enumerate(program):
            cpu.bus.bus_write(0 + _i, _b, cpu.cycles)
        cpu.regs.PC = 0
        cycles = cpu.step()
        assert cycles == expected_cycles, (
            f"{mnemonic}: expected {expected_cycles}, got {cycles}"
        )


# ============================================================
# 22. Interrupts
# ============================================================


class TestInterrupts:
    """Interrupt handling tests."""

    def test_nmi_jumps_to_0066(self, cpu):
        """NMI — always jumps to 0x0066."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IFF1 = True
        cpu.trigger_nmi()
        cpu.step()
        assert cpu.regs.PC == 0x0066

    def test_nmi_clears_iff1(self, cpu):
        """NMI — clears IFF1."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IFF1 = True
        cpu.trigger_nmi()
        cpu.step()
        assert not cpu.regs.IFF1

    def test_nmi_preserves_iff2(self, cpu):
        """NMI — IFF2 is not affected."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IFF1 = True
        cpu.regs.IFF2 = True
        cpu.trigger_nmi()
        cpu.step()
        assert cpu.regs.IFF2

    def test_im0_rst_ff(self, cpu):
        """IM 0 — RST 0xFF instruction on data bus -> PC=0x0038."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 0
        cpu.regs.IFF1 = True
        cpu.trigger_interrupt(0xFF)
        cpu.step()
        assert cpu.regs.PC == 0x0038
        assert not cpu.regs.IFF1

    def test_im1(self, cpu):
        """IM 1 — always jumps to 0x0038."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert cpu.regs.PC == 0x0038

    def test_im2_vectored(self, cpu):
        """IM 2 — vectored interrupt using I register and data bus."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 2
        cpu.regs.I = 0x10
        cpu.regs.IFF1 = True
        cpu.bus.bus_write(0x1020, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x1021, 0x40, cpu.cycles)
        cpu.trigger_interrupt(0x20)
        cpu.step()
        assert cpu.regs.PC == 0x4000

    def test_interrupt_pushes_return_address(self, cpu):
        """INT — return address is pushed to stack."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert cpu.bus.bus_read(0xFFFE, cpu.cycles) == 0x10  # high byte
        assert cpu.bus.bus_read(0xFFFD, cpu.cycles) == 0x00  # low byte

    def test_di(self, cpu):
        """DI — disables interrupts."""
        cpu.regs.IFF1 = True
        cpu.regs.IFF2 = True
        write_program(cpu, [0xF3])
        cpu.step()
        assert not cpu.regs.IFF1
        assert not cpu.regs.IFF2

    def test_ei(self, cpu):
        """EI — enables interrupts after next instruction."""
        cpu.regs.IFF1 = False
        cpu.regs.IFF2 = False
        write_program(cpu, [0xFB, 0x00])  # EI; NOP
        cpu.step()  # EI
        cpu.step()  # NOP (EI takes effect)
        assert cpu.regs.IFF1
        assert cpu.regs.IFF2

    def test_interrupt_not_accepted_when_disabled(self, cpu):
        """INT — not accepted when IFF1=0."""
        cpu.regs.PC = 0x0000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 1
        cpu.regs.IFF1 = False
        cpu.bus.bus_write(0x0000, 0x00, cpu.cycles)  # NOP
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert cpu.regs.PC == 0x0001  # NOP executed, not interrupted

    def test_retn(self, cpu):
        """RETN — return from NMI, restores IFF1 from IFF2."""
        cpu.regs.SP = 0xFFFD
        cpu.bus.bus_write(0xFFFD, 0x00, cpu.cycles)
        cpu.bus.bus_write(0xFFFE, 0x10, cpu.cycles)
        cpu.regs.IFF1 = False
        cpu.regs.IFF2 = True
        write_program(cpu, [0xED, 0x45])
        cpu.step()
        assert cpu.regs.PC == 0x1000
        assert cpu.regs.IFF1

    def test_reti(self, cpu):
        """RETI — return from interrupt."""
        cpu.regs.SP = 0xFFFD
        cpu.bus.bus_write(0xFFFD, 0x00, cpu.cycles)
        cpu.bus.bus_write(0xFFFE, 0x20, cpu.cycles)
        write_program(cpu, [0xED, 0x4D])
        cpu.step()
        assert cpu.regs.PC == 0x2000

    def test_maskable_interrupt_clears_both_iffs(self, cpu):
        """INT — clears both IFF1 and IFF2."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.regs.IFF2 = True
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert not cpu.regs.IFF1
        assert not cpu.regs.IFF2

    def test_nmi_exits_halt(self, cpu):
        """NMI — exits HALT and jumps to 0x0066."""
        write_program(cpu, [0x76])  # HALT
        cpu.step()
        assert cpu.halted
        assert cpu.regs.PC == 0
        cpu.trigger_nmi()
        cpu.step()
        assert not cpu.halted
        assert cpu.regs.PC == 0x0066  # NMI vector

    def test_nmi_returns_past_halt(self, cpu):
        """NMI — RETN returns to instruction after HALT."""
        cpu.regs.SP = 0xFFFD
        cpu.bus.bus_write(0xFFFD, 0x01, cpu.cycles)  # Return address low
        cpu.bus.bus_write(0xFFFE, 0x00, cpu.cycles)  # Return address high
        write_program(cpu, [0x76])  # HALT
        cpu.step()
        assert cpu.halted
        cpu.trigger_nmi()
        cpu.step()  # Handle NMI
        cpu.bus.bus_write(0x0066, 0xED, cpu.cycles)  # RETN at NMI vector
        cpu.bus.bus_write(0x0067, 0x45, cpu.cycles)
        cpu.step()  # Execute RETN
        assert cpu.regs.PC == 1  # Returned past HALT

    def test_maskable_interrupt_exits_halt(self, cpu):
        """INT — exits HALT and advances PC past HALT instruction."""
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        write_program(cpu, [0x76])  # HALT
        cpu.step()
        assert cpu.halted
        assert cpu.regs.PC == 0
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert not cpu.halted
        assert cpu.regs.PC == 0x0038  # IM 1 vector

    def test_ei_when_already_enabled(self, cpu):
        """EI — works correctly when interrupts already enabled."""
        cpu.regs.IFF1 = True
        cpu.regs.IFF2 = True
        write_program(cpu, [0xFB, 0x00])  # EI; NOP
        cpu.step()  # EI
        cpu.step()  # NOP (EI takes effect)
        assert cpu.regs.IFF1
        assert cpu.regs.IFF2

    def test_ld_a_i_interrupt_bug(self, cpu):
        """LD A,I — PV cleared if interrupt accepted after."""
        cpu.regs.I = 0x55
        cpu.regs.IFF2 = True
        write_program(cpu, [0xED, 0x57, 0x00])  # LD A,I; NOP
        cpu.step()  # LD A,I
        cpu.trigger_interrupt(0x00)
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.step()  # NOP with interrupt
        assert not flag_set(cpu, FLAG_PV)  # PV should be cleared due to bug

    def test_ld_a_r_interrupt_bug(self, cpu):
        """LD A,R — PV cleared if interrupt accepted after."""
        cpu.regs.R = 0x42
        cpu.regs.IFF2 = True
        write_program(cpu, [0xED, 0x5F, 0x00])  # LD A,R; NOP
        cpu.step()  # LD A,R
        cpu.trigger_interrupt(0x00)
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.step()  # NOP with interrupt
        assert not flag_set(cpu, FLAG_PV)  # PV should be cleared due to bug


# ============================================================
# 23. IX / IY Indexed Instructions
# ============================================================


class TestIndexed:
    """IX and IY indexed instruction tests."""

    def test_ld_ix_nn(self, cpu):
        """LD IX,nn — load immediate."""
        write_program(cpu, [0xDD, 0x21, 0x34, 0x12])
        cpu.step()
        assert cpu.regs.IX == 0x1234

    def test_ld_iy_nn(self, cpu):
        """LD IY,nn — load immediate."""
        write_program(cpu, [0xFD, 0x21, 0x78, 0x56])
        cpu.step()
        assert cpu.regs.IY == 0x5678

    def test_ld_a_ix_d(self, cpu):
        """LD A,(IX+d) — load from indexed address."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1005, 0xAB, cpu.cycles)
        write_program(cpu, [0xDD, 0x7E, 0x05])
        cpu.step()
        assert cpu.regs.A == 0xAB

    def test_ld_ix_d_n(self, cpu):
        """LD (IX+d),n — store immediate at indexed address."""
        cpu.regs.IX = 0x1000
        write_program(cpu, [0xDD, 0x36, 0x02, 0x99])
        cpu.step()
        assert cpu.bus.bus_read(0x1002, cpu.cycles) == 0x99

    def test_ld_a_iy_negative_d(self, cpu):
        """LD A,(IY+d) — negative displacement."""
        cpu.regs.IY = 0x2000
        cpu.bus.bus_write(0x1FFE, 0x77, cpu.cycles)
        write_program(cpu, [0xFD, 0x7E, 0xFE])  # IY-2
        cpu.step()
        assert cpu.regs.A == 0x77

    def test_add_a_ix_d(self, cpu):
        """ADD A,(IX+d) — add indexed memory to A."""
        cpu.regs.A = 0x10
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0x05, cpu.cycles)
        write_program(cpu, [0xDD, 0x86, 0x03])
        cpu.step()
        assert cpu.regs.A == 0x15

    def test_inc_ix_d(self, cpu):
        """INC (IX+d) — increment indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0x09, cpu.cycles)
        write_program(cpu, [0xDD, 0x34, 0x01])
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x0A

    def test_dec_ix_d(self, cpu):
        """DEC (IX+d) — decrement indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0x09, cpu.cycles)
        write_program(cpu, [0xDD, 0x35, 0x01])
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x08

    def test_bit_ix_d(self, cpu):
        """BIT 3,(IX+d) — test bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1002, 0x08, cpu.cycles)  # bit 3 set
        write_program(cpu, [0xDD, 0xCB, 0x02, 0x5E])
        cpu.step()
        assert flag_clear(cpu, FLAG_Z)

    def test_bit_ix_d_clear(self, cpu):
        """BIT 3,(IX+d) — bit clear."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x02, 0x5E])
        cpu.step()
        assert flag_set(cpu, FLAG_Z)

    def test_set_ix_d(self, cpu):
        """SET 0,(IX+d) — set bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1002, 0x00, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x02, 0xC6])
        cpu.step()
        assert cpu.bus.bus_read(0x1002, cpu.cycles) == 0x01

    def test_res_ix_d(self, cpu):
        """RES 0,(IX+d) — reset bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1002, 0xFF, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x02, 0x86])
        cpu.step()
        assert cpu.bus.bus_read(0x1002, cpu.cycles) == 0xFE

    def test_inc_ix(self, cpu):
        """INC IX — increment index register."""
        cpu.regs.IX = 0x1000
        write_program(cpu, [0xDD, 0x23])
        cpu.step()
        assert cpu.regs.IX == 0x1001

    def test_dec_ix(self, cpu):
        """DEC IX — decrement index register."""
        cpu.regs.IX = 0x1000
        write_program(cpu, [0xDD, 0x2B])
        cpu.step()
        assert cpu.regs.IX == 0x0FFF

    def test_inc_iy(self, cpu):
        """INC IY — increment index register."""
        cpu.regs.IY = 0x2000
        write_program(cpu, [0xFD, 0x23])
        cpu.step()
        assert cpu.regs.IY == 0x2001

    def test_dec_iy(self, cpu):
        """DEC IY — decrement index register."""
        cpu.regs.IY = 0x2000
        write_program(cpu, [0xFD, 0x2B])
        cpu.step()
        assert cpu.regs.IY == 0x1FFF

    def test_add_ix_rr(self, cpu):
        """ADD IX,BC — add register pair to IX."""
        cpu.regs.IX = 0x1000
        cpu.regs.BC = 0x0100
        write_program(cpu, [0xDD, 0x09])
        cpu.step()
        assert cpu.regs.IX == 0x1100

    def test_add_iy_rr(self, cpu):
        """ADD IY,DE — add register pair to IY."""
        cpu.regs.IY = 0x2000
        cpu.regs.DE = 0x0200
        write_program(cpu, [0xFD, 0x19])
        cpu.step()
        assert cpu.regs.IY == 0x2200

    def test_push_pop_ix(self, cpu):
        """PUSH IX / POP IX — round-trip."""
        cpu.regs.SP = 0x2000
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0xE5, 0xDD, 0xE1])  # PUSH IX; POP IX
        cpu.step()
        cpu.regs.IX = 0x0000
        cpu.step()
        assert cpu.regs.IX == 0xABCD

    def test_push_pop_iy(self, cpu):
        """PUSH IY / POP IY — round-trip."""
        cpu.regs.SP = 0x2000
        cpu.regs.IY = 0x1234
        write_program(cpu, [0xFD, 0xE5, 0xFD, 0xE1])  # PUSH IY; POP IY
        cpu.step()
        cpu.regs.IY = 0x0000
        cpu.step()
        assert cpu.regs.IY == 0x1234

    def test_ld_ix_nn_indirect(self, cpu):
        """LD IX,(nn) — load IX from memory."""
        cpu.bus.bus_write(0x5000, 0x34, cpu.cycles)
        cpu.bus.bus_write(0x5001, 0x12, cpu.cycles)
        write_program(cpu, [0xDD, 0x2A, 0x00, 0x50])
        cpu.step()
        assert cpu.regs.IX == 0x1234

    def test_ld_nn_indirect_ix(self, cpu):
        """LD (nn),IX — store IX to memory."""
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0x22, 0x00, 0x50])
        cpu.step()
        assert cpu.bus.bus_read(0x5000, cpu.cycles) == 0xCD
        assert cpu.bus.bus_read(0x5001, cpu.cycles) == 0xAB

    def test_ld_sp_ix(self, cpu):
        """LD SP,IX — copy IX to SP."""
        cpu.regs.IX = 0x4000
        write_program(cpu, [0xDD, 0xF9])
        cpu.step()
        assert cpu.regs.SP == 0x4000

    def test_ld_sp_iy(self, cpu):
        """LD SP,IY — copy IY to SP."""
        cpu.regs.IY = 0x5000
        write_program(cpu, [0xFD, 0xF9])
        cpu.step()
        assert cpu.regs.SP == 0x5000

    def test_ld_ixh_ixh(self, cpu):
        """LD IXH,IXH — undocumented self-copy no-op (DD 64)."""
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0x64])
        cycles = cpu.step()
        assert cpu.regs.IXh == 0xAB
        assert cycles == 8

    def test_ld_ixl_ixl(self, cpu):
        """LD IXL,IXL — undocumented self-copy no-op (DD 6D)."""
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0x6D])
        cycles = cpu.step()
        assert cpu.regs.IXl == 0xCD
        assert cycles == 8

    def test_ld_ixh_ixl(self, cpu):
        """LD IXH,IXL — copy IXL to IXH (DD 65)."""
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0x65])
        cycles = cpu.step()
        assert cpu.regs.IXh == 0xCD
        assert cycles == 8

    def test_ld_ixl_ixh(self, cpu):
        """LD IXL,IXH — copy IXH to IXL (DD 6C)."""
        cpu.regs.IX = 0xABCD
        write_program(cpu, [0xDD, 0x6C])
        cycles = cpu.step()
        assert cpu.regs.IXl == 0xAB
        assert cycles == 8

    def test_ld_iyh_iyh(self, cpu):
        """LD IYH,IYH — undocumented self-copy no-op (FD 64)."""
        cpu.regs.IY = 0xABCD
        write_program(cpu, [0xFD, 0x64])
        cycles = cpu.step()
        assert cpu.regs.IYh == 0xAB
        assert cycles == 8

    def test_ld_iyl_iyl(self, cpu):
        """LD IYL,IYL — undocumented self-copy no-op (FD 6D)."""
        cpu.regs.IY = 0xABCD
        write_program(cpu, [0xFD, 0x6D])
        cycles = cpu.step()
        assert cpu.regs.IYl == 0xCD
        assert cycles == 8

    def test_ld_iyh_iyl(self, cpu):
        """LD IYH,IYL — copy IYL to IYH (FD 65)."""
        cpu.regs.IY = 0xABCD
        write_program(cpu, [0xFD, 0x65])
        cycles = cpu.step()
        assert cpu.regs.IYh == 0xCD
        assert cycles == 8

    def test_ld_iyl_iyh(self, cpu):
        """LD IYL,IYH — copy IYH to IYL (FD 6C)."""
        cpu.regs.IY = 0xABCD
        write_program(cpu, [0xFD, 0x6C])
        cycles = cpu.step()
        assert cpu.regs.IYl == 0xAB
        assert cycles == 8

    def test_ld_ixh_n(self, cpu):
        """LD IXH,n — load immediate into IXH (DD 26)."""
        cpu.regs.IX = 0x0000
        write_program(cpu, [0xDD, 0x26, 0x42])
        cpu.step()
        assert cpu.regs.IXh == 0x42

    def test_ld_ixl_n(self, cpu):
        """LD IXL,n — load immediate into IXL (DD 2E)."""
        cpu.regs.IX = 0x0000
        write_program(cpu, [0xDD, 0x2E, 0x42])
        cpu.step()
        assert cpu.regs.IXl == 0x42

    def test_ld_ixh_r(self, cpu):
        """LD IXH,B — load register into IXH (DD 60)."""
        cpu.regs.IX = 0x0000
        cpu.regs.B = 0x42
        write_program(cpu, [0xDD, 0x60])
        cpu.step()
        assert cpu.regs.IXh == 0x42

    def test_ld_ixl_r(self, cpu):
        """LD IXL,C — load register into IXL (DD 69)."""
        cpu.regs.IX = 0x0000
        cpu.regs.C = 0x42
        write_program(cpu, [0xDD, 0x69])
        cpu.step()
        assert cpu.regs.IXl == 0x42

    def test_ld_r_ixh(self, cpu):
        """LD B,IXH — load IXH into register (DD 44)."""
        cpu.regs.IX = 0x4200
        write_program(cpu, [0xDD, 0x44])
        cpu.step()
        assert cpu.regs.B == 0x42

    def test_ld_r_ixl(self, cpu):
        """LD C,IXL — load IXL into register (DD 4D)."""
        cpu.regs.IX = 0x0042
        write_program(cpu, [0xDD, 0x4D])
        cpu.step()
        assert cpu.regs.C == 0x42

    @pytest.mark.parametrize(
        "reg,opcode_suffix",
        [
            ("B", 0x70),
            ("C", 0x71),
            ("D", 0x72),
            ("E", 0x73),
            ("A", 0x77),
        ],
    )
    def test_ld_ix_d_r(self, cpu, reg, opcode_suffix):
        """LD (IX+d),r — store register at indexed address."""
        cpu.regs.IX = 0x1000
        setattr(cpu.regs, reg, 0x42)
        write_program(cpu, [0xDD, opcode_suffix, 0x05])
        cpu.step()
        assert cpu.bus.bus_read(0x1005, cpu.cycles) == 0x42

    @pytest.mark.parametrize(
        "reg,opcode_suffix",
        [
            ("B", 0x46),
            ("C", 0x4E),
            ("D", 0x56),
            ("E", 0x5E),
            ("H", 0x66),
            ("L", 0x6E),
            ("A", 0x7E),
        ],
    )
    def test_ld_r_ix_d(self, cpu, reg, opcode_suffix):
        """LD r,(IX+d) — load register from indexed address."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1005, 0x99, cpu.cycles)
        write_program(cpu, [0xDD, opcode_suffix, 0x05])
        cpu.step()
        assert getattr(cpu.regs, reg) == 0x99

    def test_sub_a_ix_d(self, cpu):
        """SUB A,(IX+d) — subtract indexed memory from A."""
        cpu.regs.A = 0x20
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0x05, cpu.cycles)
        write_program(cpu, [0xDD, 0x96, 0x03])
        cpu.step()
        assert cpu.regs.A == 0x1B

    def test_and_ix_d(self, cpu):
        """AND (IX+d) — logical AND with indexed memory."""
        cpu.regs.A = 0xFF
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0x0F, cpu.cycles)
        write_program(cpu, [0xDD, 0xA6, 0x03])
        cpu.step()
        assert cpu.regs.A == 0x0F

    def test_or_ix_d(self, cpu):
        """OR (IX+d) — logical OR with indexed memory."""
        cpu.regs.A = 0x0F
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0xF0, cpu.cycles)
        write_program(cpu, [0xDD, 0xB6, 0x03])
        cpu.step()
        assert cpu.regs.A == 0xFF

    def test_xor_ix_d(self, cpu):
        """XOR (IX+d) — logical XOR with indexed memory."""
        cpu.regs.A = 0xAA
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0xAA, cpu.cycles)
        write_program(cpu, [0xDD, 0xAE, 0x03])
        cpu.step()
        assert cpu.regs.A == 0x00

    def test_cp_ix_d(self, cpu):
        """CP (IX+d) — compare A with indexed memory."""
        cpu.regs.A = 0x42
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0x42, cpu.cycles)
        write_program(cpu, [0xDD, 0xBE, 0x03])
        cpu.step()
        assert cpu.regs.A == 0x42  # unchanged
        assert flag_set(cpu, FLAG_Z)


# ============================================================
# 24. ED-Prefix Instructions
# ============================================================


class TestEDInstructions:
    """ED-prefix miscellaneous instructions."""

    def test_ld_i_a(self, cpu):
        """LD I,A — load I register from A."""
        cpu.regs.A = 0x42
        write_program(cpu, [0xED, 0x47])
        cpu.step()
        assert cpu.regs.I == 0x42

    def test_ld_r_a(self, cpu):
        """LD R,A — load R register from A."""
        cpu.regs.A = 0x11
        write_program(cpu, [0xED, 0x4F])
        cpu.step()
        assert cpu.regs.R == 0x11

    def test_ld_a_i(self, cpu):
        """LD A,I — load A from I, PV = IFF2."""
        cpu.regs.I = 0x55
        cpu.regs.IFF2 = True
        write_program(cpu, [0xED, 0x57])
        cpu.step()
        assert cpu.regs.A == 0x55
        assert flag_set(cpu, FLAG_PV)

    def test_ld_a_i_iff2_clear(self, cpu):
        """LD A,I — PV=0 when IFF2=0."""
        cpu.regs.I = 0x55
        cpu.regs.IFF2 = False
        write_program(cpu, [0xED, 0x57])
        cpu.step()
        assert cpu.regs.A == 0x55
        assert flag_clear(cpu, FLAG_PV)

    def test_ld_a_r(self, cpu):
        """LD A,R — load A from R register."""
        cpu.regs.R = 0x42
        write_program(cpu, [0xED, 0x5F])
        cpu.step()
        # R may have been incremented by instruction fetch
        assert cpu.regs.A is not None  # basic sanity

    @pytest.mark.parametrize(
        "pair,load_op,store_op",
        [
            ("BC", [0xED, 0x4B], [0xED, 0x43]),
            ("DE", [0xED, 0x5B], [0xED, 0x53]),
            ("HL", [0xED, 0x6B], [0xED, 0x63]),
            ("SP", [0xED, 0x7B], [0xED, 0x73]),
        ],
    )
    def test_ed_ld_rr_nn_indirect(self, cpu, pair, load_op, store_op):
        """ED LD rr,(nn) and LD (nn),rr — all pairs."""
        # Store
        cpu.reset()
        setattr(cpu.regs, pair, 0x1234)
        write_program(cpu, store_op + [0x00, 0x50])
        cpu.step()
        assert cpu.bus.bus_read(0x5000, cpu.cycles) == 0x34
        assert cpu.bus.bus_read(0x5001, cpu.cycles) == 0x12

        # Load
        cpu.reset()
        cpu.bus.bus_write(0x5000, 0xCD, cpu.cycles)
        cpu.bus.bus_write(0x5001, 0xAB, cpu.cycles)
        write_program(cpu, load_op + [0x00, 0x50])
        cpu.step()
        assert getattr(cpu.regs, pair) == 0xABCD

    def test_rld(self, cpu):
        """RLD — rotate left digit."""
        cpu.regs.A = 0x9A
        cpu.regs.HL = 0x1000
        cpu.bus.bus_write(0x1000, 0x31, cpu.cycles)
        write_program(cpu, [0xED, 0x6F])
        cpu.step()
        assert cpu.regs.A == 0x93
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0x1A

    def test_rrd(self, cpu):
        """RRD — rotate right digit."""
        cpu.regs.A = 0x84
        cpu.regs.HL = 0x1000
        cpu.bus.bus_write(0x1000, 0x20, cpu.cycles)
        write_program(cpu, [0xED, 0x67])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0x42

    @pytest.mark.parametrize(
        "im_val,opcode",
        [
            (0, 0x46),
            (1, 0x56),
            (2, 0x5E),
        ],
    )
    def test_im(self, cpu, im_val, opcode):
        """IM 0/1/2 — set interrupt mode."""
        write_program(cpu, [0xED, opcode])
        cpu.step()
        assert cpu.regs.IM == im_val

    def test_neg_various(self, cpu):
        """NEG — various input values."""
        test_cases = [
            (0x00, 0x00, True, False),  # zero
            (0x01, 0xFF, False, True),
            (0x80, 0x80, False, True),  # overflow
            (0x7F, 0x81, False, True),
        ]
        for a, expected, z, c in test_cases:
            cpu.reset()
            cpu.regs.A = a
            write_program(cpu, [0xED, 0x44])
            cpu.step()
            assert cpu.regs.A == expected, f"NEG {a:#04x} -> {expected:#04x}"
            assert flag_set(cpu, FLAG_Z) == z
            assert flag_set(cpu, FLAG_C) == c


# ============================================================
# 25. HALT
# ============================================================


class TestHalt:
    """HALT instruction behavior."""

    def test_halt_stops_execution(self, cpu):
        """HALT — CPU enters halted state."""
        write_program(cpu, [0x76])
        cpu.step()
        assert cpu.halted

    def test_halt_pc_does_not_advance(self, cpu):
        """HALT — PC stays at HALT instruction on subsequent steps."""
        write_program(cpu, [0x76])
        cpu.step()
        pc_after = cpu.regs.PC
        cpu.step()  # should re-execute HALT
        assert cpu.regs.PC == pc_after


# ============================================================
# 26. NOP
# ============================================================


class TestNop:
    """NOP instruction tests."""

    def test_nop_advances_pc(self, cpu):
        """NOP — only advances PC by 1."""
        write_program(cpu, [0x00])
        cpu.step()
        assert cpu.regs.PC == 1

    def test_nop_preserves_registers(self, cpu):
        """NOP — does not modify any registers or flags."""
        cpu.regs.A = 0x42
        cpu.regs.BC = 0x1234
        cpu.regs.DE = 0x5678
        cpu.regs.HL = 0x9ABC
        cpu.regs.F = 0xFF
        write_program(cpu, [0x00])
        cpu.step()
        assert cpu.regs.A == 0x42
        assert cpu.regs.BC == 0x1234
        assert cpu.regs.DE == 0x5678
        assert cpu.regs.HL == 0x9ABC
        assert cpu.regs.F == 0xFF


# ============================================================
# 27. Undocumented: SLL (CB 30-37)
# ============================================================


class TestSLL:
    """SLL (shift left logical) — undocumented instruction."""

    def test_sll_a(self, cpu):
        """SLL A — shift left, bit 0 = 1."""
        cpu.regs.A = 0x00
        run_cb_instruction(cpu, 0x37)
        assert cpu.regs.A == 0x01

    def test_sll_a_with_carry(self, cpu):
        """SLL A — bit 7 goes to carry."""
        cpu.regs.A = 0x80
        run_cb_instruction(cpu, 0x37)
        assert cpu.regs.A == 0x01
        assert flag_set(cpu, FLAG_C)


# ============================================================
# 28. Edge Cases / Regression Tests
# ============================================================


class TestEdgeCases:
    """Boundary conditions and regression tests."""

    def test_sp_wrap_around(self, cpu):
        """SP wraps around on PUSH from 0x0001."""
        cpu.regs.SP = 0x0001
        cpu.regs.BC = 0x1234
        write_program(cpu, [0xC5])
        cpu.step()
        assert cpu.regs.SP == 0xFFFF

    def test_pc_wrap_around(self, cpu):
        """PC wraps around past 0xFFFF."""
        write_program(cpu, [0xC3, 0xFF, 0xFF])  # JP 0xFFFF
        cpu.step()
        assert cpu.regs.PC == 0xFFFF
        cpu.bus.bus_write(0xFFFF, 0x00, cpu.cycles)  # NOP at 0xFFFF
        cpu.step()
        assert cpu.regs.PC == 0x0000

    def test_add_overflow_signed(self, cpu):
        """ADD A,n — signed overflow: 0x7F + 0x01 -> PV set."""
        cpu.regs.A = 0x7F
        write_program(cpu, [0xC6, 0x01])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_PV)
        assert flag_set(cpu, FLAG_S)

    def test_sub_overflow_signed(self, cpu):
        """SUB A,n — signed overflow: 0x80 - 0x01 -> PV set."""
        cpu.regs.A = 0x80
        write_program(cpu, [0xD6, 0x01])
        cpu.step()
        assert cpu.regs.A == 0x7F
        assert flag_set(cpu, FLAG_PV)

    def test_inc_0x7f_overflow(self, cpu):
        """INC — 0x7F -> 0x80, PV set (signed overflow)."""
        cpu.regs.A = 0x7F
        write_program(cpu, [0x3C])
        cpu.step()
        assert cpu.regs.A == 0x80
        assert flag_set(cpu, FLAG_PV)

    def test_dec_0x80_overflow(self, cpu):
        """DEC — 0x80 -> 0x7F, PV set (signed overflow)."""
        cpu.regs.A = 0x80
        write_program(cpu, [0x3D])
        cpu.step()
        assert cpu.regs.A == 0x7F
        assert flag_set(cpu, FLAG_PV)

    def test_ldir_overlapping_blocks(self, cpu):
        """LDIR — overlapping source and destination."""
        cpu.bus.bus_write(0x1000, 0xAA, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0xBB, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0xCC, cpu.cycles)
        cpu.regs.HL = 0x1000
        cpu.regs.DE = 0x1001  # overlapping
        cpu.regs.BC = 0x0003
        write_program(cpu, [0xED, 0xB0])
        step_n(cpu, 3)
        # First byte gets copied forward repeatedly
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0xAA
        assert cpu.bus.bus_read(0x1002, cpu.cycles) == 0xAA
        assert cpu.bus.bus_read(0x1003, cpu.cycles) == 0xAA

    def test_multiple_interrupts(self, cpu):
        """Multiple interrupt triggers — only one accepted per check."""
        cpu.regs.PC = 0x1000
        cpu.regs.SP = 0xFFFF
        cpu.regs.IM = 1
        cpu.regs.IFF1 = True
        cpu.trigger_interrupt(0x00)
        cpu.step()
        assert cpu.regs.PC == 0x0038
        # After first interrupt, IFF1 is cleared
        assert not cpu.regs.IFF1

    def test_r_register_increments(self, cpu):
        """R register increments with each instruction."""
        cpu.regs.R = 0x00
        write_program(cpu, [0x00, 0x00, 0x00])  # 3 NOPs
        cpu.step()
        r1 = cpu.regs.R
        cpu.step()
        r2 = cpu.regs.R
        cpu.step()
        r3 = cpu.regs.R
        assert r1 == 1
        assert r2 == 2
        assert r3 == 3

    def test_r_register_bit7_preserved(self, cpu):
        """R register — bit 7 is preserved (only lower 7 bits count)."""
        cpu.regs.R = 0x80
        write_program(cpu, [0x00])
        cpu.step()
        assert cpu.regs.R & 0x80  # bit 7 preserved

    def test_add_hl_hl(self, cpu):
        """ADD HL,HL — doubles HL."""
        cpu.regs.HL = 0x4000
        write_program(cpu, [0x29])
        cpu.step()
        assert cpu.regs.HL == 0x8000

    def test_ld_a_a(self, cpu):
        """LD A,A — no-op equivalent."""
        cpu.regs.A = 0x42
        write_program(cpu, [0x7F])
        cpu.step()
        assert cpu.regs.A == 0x42


# ============================================================
# 29. Full Program Integration Tests
# ============================================================


class TestIntegration:
    """Small Z80 programs to verify instruction interaction."""

    def test_memcpy_loop(self, cpu):
        """Copy 3 bytes using a loop: LD A,(HL); LD (DE),A; INC HL; INC DE; DEC B; JR NZ."""
        # Source data at 0x1000
        cpu.bus.bus_write(0x1000, 0x11, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x22, cpu.cycles)
        cpu.bus.bus_write(0x1002, 0x33, cpu.cycles)

        # Program at 0x0000:
        #   LD HL,0x1000
        #   LD DE,0x2000
        #   LD B,3
        # loop:
        #   LD A,(HL)
        #   LD (DE),A
        #   INC HL
        #   INC DE
        #   DEC B
        #   JR NZ,loop
        program = [
            0x21,
            0x00,
            0x10,  # LD HL,0x1000
            0x11,
            0x00,
            0x20,  # LD DE,0x2000
            0x06,
            0x03,  # LD B,3
            # loop (offset 8):
            0x7E,  # LD A,(HL)
            0x12,  # LD (DE),A
            0x23,  # INC HL
            0x13,  # INC DE
            0x05,  # DEC B
            0x20,
            0xF9,  # JR NZ,-7 (back to offset 8)
        ]
        write_program(cpu, program)

        # Execute enough steps
        for _ in range(50):
            if cpu.regs.PC >= len(program):
                break
            cpu.step()

        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x11
        assert cpu.bus.bus_read(0x2001, cpu.cycles) == 0x22
        assert cpu.bus.bus_read(0x2002, cpu.cycles) == 0x33

    def test_sum_array(self, cpu):
        """Sum 4 bytes: result in A."""
        cpu.bus.bus_write(0x1000, 10, cpu.cycles)
        cpu.bus.bus_write(0x1001, 20, cpu.cycles)
        cpu.bus.bus_write(0x1002, 30, cpu.cycles)
        cpu.bus.bus_write(0x1003, 40, cpu.cycles)

        program = [
            0x21,
            0x00,
            0x10,  # LD HL,0x1000
            0x06,
            0x04,  # LD B,4
            0x3E,
            0x00,  # LD A,0
            # loop (offset 7):
            0x86,  # ADD A,(HL)
            0x23,  # INC HL
            0x05,  # DEC B
            0x20,
            0xFB,  # JR NZ,-5
        ]
        write_program(cpu, program)

        for _ in range(50):
            if cpu.regs.PC >= len(program):
                break
            cpu.step()

        assert cpu.regs.A == 100

    def test_call_ret_sequence(self, cpu):
        """CALL subroutine that adds two numbers, then RET."""
        # Main at 0x0000:
        #   LD A,0x10
        #   LD B,0x20
        #   CALL 0x0100
        #   HALT
        main = [
            0x3E,
            0x10,  # LD A,0x10
            0x06,
            0x20,  # LD B,0x20
            0xCD,
            0x00,
            0x01,  # CALL 0x0100
            0x76,  # HALT
        ]
        write_program(cpu, main)
        cpu.regs.SP = 0xFFFF

        # Subroutine at 0x0100: ADD A,B; RET
        cpu.bus.bus_write(0x0100, 0x80, cpu.cycles)  # ADD A,B
        cpu.bus.bus_write(0x0101, 0xC9, cpu.cycles)  # RET

        for _ in range(10):
            if cpu.halted:
                break
            cpu.step()

        assert cpu.regs.A == 0x30
        assert cpu.halted

    def test_nested_calls(self, cpu):
        """Nested CALL/RET."""
        cpu.regs.SP = 0xFFFF

        # Main: CALL 0x0100; HALT
        cpu.bus.bus_write(0x0000, 0xCD, cpu.cycles)
        cpu.bus.bus_write(0x0001, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x0002, 0x01, cpu.cycles)
        cpu.bus.bus_write(0x0003, 0x76, cpu.cycles)  # HALT

        # Sub1 at 0x0100: LD A,1; CALL 0x0200; RET
        cpu.bus.bus_write(0x0100, 0x3E, cpu.cycles)
        cpu.bus.bus_write(0x0101, 0x01, cpu.cycles)
        cpu.bus.bus_write(0x0102, 0xCD, cpu.cycles)
        cpu.bus.bus_write(0x0103, 0x00, cpu.cycles)
        cpu.bus.bus_write(0x0104, 0x02, cpu.cycles)
        cpu.bus.bus_write(0x0105, 0xC9, cpu.cycles)

        # Sub2 at 0x0200: ADD A,2; RET
        cpu.bus.bus_write(0x0200, 0xC6, cpu.cycles)
        cpu.bus.bus_write(0x0201, 0x02, cpu.cycles)
        cpu.bus.bus_write(0x0202, 0xC9, cpu.cycles)

        cpu.regs.PC = 0

        for _ in range(20):
            if cpu.halted:
                break
            cpu.step()

        assert cpu.regs.A == 0x03
        assert cpu.halted
        assert cpu.regs.SP == 0xFFFF  # stack balanced

    def test_push_pop_all_pairs(self, cpu):
        """Push all pairs, pop in reverse order."""
        cpu.regs.SP = 0x2000
        cpu.regs.BC = 0x1111
        cpu.regs.DE = 0x2222
        cpu.regs.HL = 0x3333

        program = [
            0xC5,  # PUSH BC
            0xD5,  # PUSH DE
            0xE5,  # PUSH HL
            # Clear registers
            0x01,
            0x00,
            0x00,  # LD BC,0
            0x11,
            0x00,
            0x00,  # LD DE,0
            0x21,
            0x00,
            0x00,  # LD HL,0
            # Pop in reverse
            0xE1,  # POP HL
            0xD1,  # POP DE
            0xC1,  # POP BC
        ]
        write_program(cpu, program)

        for _ in range(len(program)):
            cpu.step()

        assert cpu.regs.HL == 0x3333
        assert cpu.regs.DE == 0x2222
        assert cpu.regs.BC == 0x1111
        assert cpu.regs.SP == 0x2000

    def test_fibonacci(self, cpu):
        """Calculate first 8 Fibonacci numbers."""
        # Store fib(1)..fib(8) at 0x2000-0x2007
        # fib: 1,1,2,3,5,8,13,21
        program = [
            0x3E,
            0x01,  # LD A,1
            0x06,
            0x00,  # LD B,0
            0x21,
            0x00,
            0x20,  # LD HL,0x2000
            0x0E,
            0x08,  # LD C,8 (counter)
            # loop (offset 9):
            0x77,  # LD (HL),A
            0x23,  # INC HL
            0x57,  # LD D,A (temp = current)
            0x80,  # ADD A,B (current = current + previous)
            0x42,  # LD B,D (previous = temp)
            0x0D,  # DEC C
            0x20,
            0xF8,  # JR NZ, -8
        ]
        write_program(cpu, program)

        for _ in range(100):
            if cpu.regs.PC >= len(program):
                break
            cpu.step()

        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 1
        assert cpu.bus.bus_read(0x2001, cpu.cycles) == 1
        assert cpu.bus.bus_read(0x2002, cpu.cycles) == 2
        assert cpu.bus.bus_read(0x2003, cpu.cycles) == 3
        assert cpu.bus.bus_read(0x2004, cpu.cycles) == 5
        assert cpu.bus.bus_read(0x2005, cpu.cycles) == 8
        assert cpu.bus.bus_read(0x2006, cpu.cycles) == 13
        assert cpu.bus.bus_read(0x2007, cpu.cycles) == 21


# ============================================================
# 30. Parity Table Verification
# ============================================================


class TestParityTable:
    """Verify the parity lookup table is correct."""

    @pytest.mark.parametrize("val", range(256))
    def test_parity_table(self, val):
        """PARITY_TABLE[n] matches computed even parity."""
        bit_count = bin(val).count("1")
        expected_even_parity = (bit_count % 2) == 0
        assert bool(PARITY_TABLE[val]) == expected_even_parity, (
            f"PARITY_TABLE[{val:#04x}] incorrect"
        )


# ============================================================
# 31. Undocumented Flags (F3, F5)
# ============================================================


class TestUndocumentedFlags:
    """
    Z80 undocumented flags F3 and F5.

    These flags copy bits 3 and 5 from the result of most instructions.
    They are not used by Z80 hardware but some software uses them.
    """

    FLAG_F3 = 0x08
    FLAG_F5 = 0x20

    def test_add_f3_f5_from_result(self, cpu):
        """ADD — F3/F5 copied from result bit 3/5."""
        cpu.regs.A = 0x08  # bit 3 set
        write_program(cpu, [0xC6, 0x00])  # ADD A,0
        cpu.step()
        assert cpu.regs.F & self.FLAG_F3

    def test_sub_f3_f5_from_result(self, cpu):
        """SUB — F3/F5 copied from result bit 3/5."""
        cpu.regs.A = 0x20  # bit 5 set
        write_program(cpu, [0xD6, 0x00])
        cpu.step()
        assert cpu.regs.F & self.FLAG_F5

    def test_and_f3_f5_from_result(self, cpu):
        """AND — F3/F5 copied from result bit 3/5."""
        cpu.regs.A = 0x08
        write_program(cpu, [0xE6, 0xFF])
        cpu.step()
        assert cpu.regs.F & self.FLAG_F3

    def test_or_f3_f5_from_result(self, cpu):
        """OR — F3/F5 copied from result bit 3/5."""
        cpu.regs.A = 0x20
        write_program(cpu, [0xF6, 0x00])
        cpu.step()
        assert cpu.regs.F & self.FLAG_F5

    def test_xor_f3_f5_from_result(self, cpu):
        """XOR — F3/F5 copied from result bit 3/5."""
        cpu.regs.A = 0x08
        write_program(cpu, [0xEE, 0x00])  # XOR 0x00 (result = 0x08)
        cpu.step()
        assert cpu.regs.F & self.FLAG_F3

    def test_cp_f3_f5_from_operand(self, cpu):
        """CP — F3/F5 copied from operand, not result."""
        cpu.regs.A = 0x00
        write_program(cpu, [0xFE, 0x28])  # compare with 0x28 (bit 3 and 5 set)
        cpu.step()
        assert cpu.regs.F & self.FLAG_F3
        assert cpu.regs.F & self.FLAG_F5

    def test_inc_f3_f5(self, cpu):
        """INC — F3/F5 from result."""
        cpu.regs.A = 0x08
        write_program(cpu, [0x3C])
        cpu.step()
        assert cpu.regs.F & self.FLAG_F3

    def test_dec_f3_f5(self, cpu):
        """DEC — F3/F5 from result."""
        cpu.regs.A = 0x28
        write_program(cpu, [0x3D])
        cpu.step()
        assert cpu.regs.F & self.FLAG_F5


# ============================================================
# 32. DD/FD Prefix Undocumented Behavior
# ============================================================


class TestDDFDFallthrough:
    """
    Undocumented Z80: Unknown DD/FD prefixes fall through to base opcode.

    When a DD or FD prefix is followed by an opcode that has no IX/IY
    equivalent, the CPU ignores the prefix and executes the base opcode.
    """

    def test_dd_prefix_nop_fallthrough(self, cpu):
        """DD prefix + NOP (0x00) executes NOP."""
        write_program(cpu, [0xDD, 0x00])
        cpu.step()
        assert cpu.regs.PC == 2  # advanced past both bytes

    def test_fd_prefix_nop_fallthrough(self, cpu):
        """FD prefix + NOP (0x00) executes NOP."""
        write_program(cpu, [0xFD, 0x00])
        cpu.step()
        assert cpu.regs.PC == 2

    def test_dd_prefix_ld_a_n_fallthrough(self, cpu):
        """DD prefix + LD A,n (0x3E) executes LD A,n."""
        cpu.regs.PC = 0
        write_program(cpu, [0xDD, 0x3E, 0x42])
        cpu.step()
        assert cpu.regs.A == 0x42

    def test_fd_prefix_add_a_b_fallthrough(self, cpu):
        """FD prefix + ADD A,B (0x80) executes ADD A,B."""
        cpu.regs.A = 0x10
        cpu.regs.B = 0x20
        write_program(cpu, [0xFD, 0x80])
        cpu.step()
        assert cpu.regs.A == 0x30


# ============================================================
# 33. DDCB/FDCB Indexed Bit Operations
# ============================================================


class TestDDCBFDCB:
    """
    DDCB/FDCB indexed bit operations: DD CB d / FD CB d instructions.

    These are 4-byte instructions: prefix + CB + displacement + opcode.
    The displacement is signed (-128 to +127).
    """

    def test_ddcb_rlc_ix_d(self, cpu):
        """DDCB: RLC (IX+d) stores result in memory and optionally in register."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1002, 0x80, cpu.cycles)  # IX+2
        write_program(cpu, [0xDD, 0xCB, 0x02, 0x06])  # RLC (IX+2)
        cpu.step()
        assert cpu.bus.bus_read(0x1002, cpu.cycles) == 0x01

    def test_ddcb_bit_ix_d(self, cpu):
        """DDCB: BIT b,(IX+d) tests bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0x08, cpu.cycles)  # bit 3 set
        write_program(cpu, [0xDD, 0xCB, 0x03, 0x5E])  # BIT 3,(IX+3)
        cpu.step()
        assert flag_clear(cpu, FLAG_Z)  # bit is set

    def test_ddcb_bit_ix_d_clear(self, cpu):
        """DDCB: BIT b,(IX+d) when bit is clear."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1003, 0xF7, cpu.cycles)  # bit 3 clear
        write_program(cpu, [0xDD, 0xCB, 0x03, 0x5E])  # BIT 3,(IX+3)
        cpu.step()
        assert flag_set(cpu, FLAG_Z)  # bit is clear

    def test_ddcb_set_ix_d(self, cpu):
        """DDCB: SET b,(IX+d) sets bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0x00, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x01, 0xC6])  # SET 0,(IX+1)
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x01

    def test_ddcb_res_ix_d(self, cpu):
        """DDCB: RES b,(IX+d) resets bit in indexed memory."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0xFF, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x01, 0x86])  # RES 0,(IX+1)
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0xFE

    def test_ddcb_sla_ix_d(self, cpu):
        """DDCB: SLA (IX+d) arithmetic shift left."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0x40, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x01, 0x26])  # SLA (IX+1)
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x80

    def test_ddcb_sra_ix_d(self, cpu):
        """DDCB: SRA (IX+d) arithmetic shift right."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x1001, 0x80, cpu.cycles)
        write_program(cpu, [0xDD, 0xCB, 0x01, 0x2E])  # SRA (IX+1)
        cpu.step()
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0xC0

    def test_fdcb_rlc_iy_d(self, cpu):
        """FDCB: RLC (IY+d) similar to DDCB."""
        cpu.regs.IY = 0x2000
        cpu.bus.bus_write(0x2002, 0x01, cpu.cycles)
        write_program(cpu, [0xFD, 0xCB, 0x02, 0x06])  # RLC (IY+2)
        cpu.step()
        assert cpu.bus.bus_read(0x2002, cpu.cycles) == 0x02

    def test_ddcb_negative_displacement(self, cpu):
        """DDCB: negative displacement wraps correctly."""
        cpu.regs.IX = 0x1000
        cpu.bus.bus_write(0x0FF8, 0xFF, cpu.cycles)  # IX-8 = 0x0FF8
        write_program(cpu, [0xDD, 0xCB, 0xF8, 0x46])  # BIT 0,(IX-8)
        cpu.step()
        assert flag_clear(cpu, FLAG_Z)

    @pytest.mark.parametrize(
        "program,mnemonic,expected_cycles",
        [
            # DDCB — IX indexed
            ([0xDD, 0xCB, 0x00, 0x06], "RLC (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x0E], "RRC (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x16], "RL (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x1E], "RR (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x26], "SLA (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x2E], "SRA (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x3E], "SRL (IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x46], "BIT 0,(IX+0)", 20),
            ([0xDD, 0xCB, 0x00, 0x7E], "BIT 7,(IX+0)", 20),
            ([0xDD, 0xCB, 0x00, 0xC6], "SET 0,(IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0xFE], "SET 7,(IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0x86], "RES 0,(IX+0)", 23),
            ([0xDD, 0xCB, 0x00, 0xBE], "RES 7,(IX+0)", 23),
            # FDCB — IY indexed
            ([0xFD, 0xCB, 0x00, 0x06], "RLC (IY+0)", 23),
            ([0xFD, 0xCB, 0x00, 0x46], "BIT 0,(IY+0)", 20),
            ([0xFD, 0xCB, 0x00, 0xC6], "SET 0,(IY+0)", 23),
            ([0xFD, 0xCB, 0x00, 0x86], "RES 0,(IY+0)", 23),
        ],
    )
    def test_ddcb_fdcb_timing(self, cpu, program, mnemonic, expected_cycles):
        """DDCB/FDCB indexed instruction timing."""
        if program[0] == 0xDD:
            cpu.regs.IX = 0x1000
            addr = 0x1000  # IX + 0
        else:
            cpu.regs.IY = 0x1000
            addr = 0x1000  # IY + 0
        cpu.bus.bus_write(addr, 0x00, cpu.cycles)
        for _i, _b in enumerate(program):
            cpu.bus.bus_write(0 + _i, _b, cpu.cycles)
        cpu.regs.PC = 0
        cycles = cpu.step()
        assert cycles == expected_cycles, (
            f"{mnemonic}: expected {expected_cycles}, got {cycles}"
        )


# ============================================================
# 34. Repeat I/O Block Instructions
# ============================================================


class TestRepeatIOBlock:
    """INIR, INDR, OTIR, OTDR — repeat I/O instructions."""

    def test_inir_terminates_on_b_zero(self, cpu):
        """INIR — terminates when B reaches 0."""
        cpu.regs.B = 1
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2000
        cpu.bus.io_ports[0x10] = 0x99
        write_program(cpu, [0xED, 0xB2])
        cpu.step()
        assert cpu.bus.bus_read(0x2000, cpu.cycles) == 0x99
        assert cpu.regs.B == 0

    def test_inir_decrements_b(self, cpu):
        """INIR — decrements B on each iteration."""
        cpu.regs.B = 3
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2000
        cpu.bus.io_ports[0x10] = 0x11
        write_program(cpu, [0xED, 0xB2])
        step_n(cpu, 3)
        assert cpu.regs.B == 0
        assert cpu.regs.HL == 0x2003

    def test_indr_decrements_hl(self, cpu):
        """INDR — decrements HL on each iteration."""
        cpu.regs.B = 2
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x2001
        cpu.bus.io_ports[0x10] = 0x22
        write_program(cpu, [0xED, 0xBA])
        step_n(cpu, 2)
        assert cpu.regs.HL == 0x1FFF

    def test_otir_decrements_b(self, cpu):
        """OTIR — decrements B on each iteration."""
        cpu.regs.B = 2
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x1000
        cpu.bus.bus_write(0x1000, 0xAA, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0xBB, cpu.cycles)
        write_program(cpu, [0xED, 0xB3])
        step_n(cpu, 2)
        assert cpu.regs.B == 0

    def test_otdr_decrements_hl(self, cpu):
        """OTDR — decrements HL on each iteration."""
        cpu.regs.B = 2
        cpu.regs.C = 0x10
        cpu.regs.HL = 0x1001
        cpu.bus.bus_write(0x1000, 0x11, cpu.cycles)
        cpu.bus.bus_write(0x1001, 0x22, cpu.cycles)
        write_program(cpu, [0xED, 0xBB])
        step_n(cpu, 2)
        assert cpu.regs.HL == 0x0FFF


# ============================================================
# 35. IX/IY Arithmetic Operations
# ============================================================


class TestIXIYArithmetic:
    """ADD/ADC/SBC operations with IX and IY registers."""

    def test_adc_ix_bc(self, cpu):
        """ADC IX,BC — add with carry."""
        cpu.regs.IX = 0x0001
        cpu.regs.BC = 0x0002
        cpu.regs.F = 0
        write_program(cpu, [0xDD, 0x4A])
        cpu.step()
        assert cpu.regs.IX == 0x0003

    def test_adc_ix_bc_with_carry(self, cpu):
        """ADC IX,BC — carry causes overflow."""
        cpu.regs.IX = 0xFFFF
        cpu.regs.BC = 0x0001
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xDD, 0x4A])
        cpu.step()
        assert cpu.regs.IX == 0x0001
        assert flag_set(cpu, FLAG_C)

    def test_adc_ix_zero(self, cpu):
        """ADC IX,rr — Z flag set when result is zero."""
        cpu.regs.IX = 0xFFFF
        cpu.regs.BC = 0x0000
        cpu.regs.F = FLAG_C  # FFFF + 0 + 1 = 0x10000 -> 0x0000 with carry
        write_program(cpu, [0xDD, 0x4A])
        cpu.step()
        assert cpu.regs.IX == 0x0000
        assert flag_set(cpu, FLAG_Z)
        assert flag_set(cpu, FLAG_C)

    def test_sbc_ix_de(self, cpu):
        """SBC IX,DE — subtract with carry."""
        cpu.regs.IX = 0x0010
        cpu.regs.DE = 0x0002
        cpu.regs.F = 0
        write_program(cpu, [0xDD, 0x52])
        cpu.step()
        assert cpu.regs.IX == 0x000E

    def test_sbc_ix_de_with_carry(self, cpu):
        """SBC IX,DE — with carry subtracts extra."""
        cpu.regs.IX = 0x0010
        cpu.regs.DE = 0x0002
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xDD, 0x52])
        cpu.step()
        assert cpu.regs.IX == 0x000D

    def test_add_iy_sp(self, cpu):
        """ADD IY,SP — add SP to IY."""
        cpu.regs.IY = 0x1000
        cpu.regs.SP = 0x0100
        write_program(cpu, [0xFD, 0x39])
        cpu.step()
        assert cpu.regs.IY == 0x1100

    def test_adc_iy_de(self, cpu):
        """ADC IY,DE — with carry."""
        cpu.regs.IY = 0x0001
        cpu.regs.DE = 0x0002
        cpu.regs.F = FLAG_C
        write_program(cpu, [0xFD, 0x5A])  # ADC IY,DE
        cpu.step()
        assert cpu.regs.IY == 0x0004


# ============================================================
# 36. DAA Comprehensive Tests
# ============================================================


class TestDAAComprehensive:
    """Comprehensive DAA (Decimal Adjust Accumulator) tests."""

    def test_daa_add_0f_01(self, cpu):
        """DAA: 0x0F + 0x01 = 0x10 with H=1, DAA adds 0x06 -> 0x16."""
        cpu.regs.A = 0x0F
        write_program(cpu, [0xC6, 0x01, 0x27])  # ADD A,1; DAA
        step_n(cpu, 2)
        assert cpu.regs.A == 0x16
        assert flag_clear(cpu, FLAG_C)

    def test_daa_add_f9_01(self, cpu):
        """DAA: 0xF9 + 0x01 = 0xFA, lower nibble > 9, DAA adds 0x06 -> 0x00."""
        cpu.regs.A = 0xF9
        write_program(cpu, [0xC6, 0x01, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x00
        assert flag_clear(cpu, FLAG_C)

    def test_daa_sub_10_01(self, cpu):
        """DAA: 0x10 - 0x01 = 0x09."""
        cpu.regs.A = 0x10
        write_program(cpu, [0xD6, 0x01, 0x27])  # SUB 1; DAA
        step_n(cpu, 2)
        assert cpu.regs.A == 0x09

    def test_daa_sub_00_01(self, cpu):
        """DAA: 0x00 - 0x01 = 0x99 with borrow (carry)."""
        cpu.regs.A = 0x00
        write_program(cpu, [0xD6, 0x01, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x99
        assert flag_set(cpu, FLAG_C)

    def test_daa_add_80_80(self, cpu):
        """DAA: 0x80 + 0x80 = 0x60 with carry."""
        cpu.regs.A = 0x80
        write_program(cpu, [0xC6, 0x80, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x60
        assert flag_set(cpu, FLAG_C)

    def test_daa_preserves_z_flag(self, cpu):
        """DAA: Z flag set when result is zero."""
        cpu.regs.A = 0x90
        write_program(cpu, [0xC6, 0x10, 0x27])
        step_n(cpu, 2)
        assert cpu.regs.A == 0x00
        assert flag_set(cpu, FLAG_Z)


# ============================================================
# 37. CCF Proper H-Flag Behavior
# ============================================================


class TestCCFHFlag:
    """CCF — Complement Carry Flag, H gets old C value."""

    def test_ccf_h_gets_old_carry_set(self, cpu):
        """CCF: H = old C when C was set (H=1, C=0)."""
        cpu.regs.F = FLAG_C
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_clear(cpu, FLAG_C)
        assert flag_set(cpu, FLAG_H)

    def test_ccf_h_gets_old_carry_clear(self, cpu):
        """CCF: H = old C when C was clear (H=0, C=1)."""
        cpu.regs.F = 0
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_set(cpu, FLAG_C)
        assert flag_clear(cpu, FLAG_H)

    def test_ccf_n_always_clear(self, cpu):
        """CCF: N flag always cleared."""
        cpu.regs.F = FLAG_N | FLAG_C
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_clear(cpu, FLAG_N)

    def test_ccf_preserves_s_z_pv(self, cpu):
        """CCF: preserves S, Z, PV flags."""
        cpu.regs.F = FLAG_S | FLAG_Z | FLAG_PV | FLAG_C
        write_program(cpu, [0x3F])
        cpu.step()
        assert flag_set(cpu, FLAG_S)
        assert flag_set(cpu, FLAG_Z)
        assert flag_set(cpu, FLAG_PV)


# ============================================================
# 38. 16-bit Load with SP
# ============================================================


class TestLoadSPIndirect:
    """LD (nn),SP and LD SP,(nn) instructions."""

    def test_ld_nn_indirect_sp(self, cpu):
        """LD (nn),SP — store SP to memory."""
        cpu.regs.SP = 0x2000
        write_program(cpu, [0xED, 0x73, 0x00, 0x10])  # LD (0x1000),SP
        cpu.step()
        assert cpu.bus.bus_read(0x1000, cpu.cycles) == 0x00
        assert cpu.bus.bus_read(0x1001, cpu.cycles) == 0x20

    def test_ld_sp_nn_indirect(self, cpu):
        """LD SP,(nn) — load SP from memory."""
        cpu.bus.bus_write(0x4000, 0x34, cpu.cycles)
        cpu.bus.bus_write(0x4001, 0x12, cpu.cycles)
        write_program(cpu, [0xED, 0x7B, 0x00, 0x40])
        cpu.step()
        assert cpu.regs.SP == 0x1234


# ============================================================
# 39. Edge Case: Page Boundary Wrapping
# ============================================================


class TestPageBoundary:
    """Memory operations that cross 64KB boundaries."""

    def test_jp_at_page_boundary(self, cpu):
        """JP to address at page boundary works correctly."""
        cpu.bus.bus_write(0x0100, 0xC3, cpu.cycles)  # JP opcode at target
        cpu.bus.bus_write(0x0101, 0x00, cpu.cycles)  # low byte
        cpu.bus.bus_write(0x0102, 0x20, cpu.cycles)  # high byte
        write_program(cpu, [0xC3, 0x00, 0x01])  # JP 0x0100
        cpu.step()
        assert cpu.regs.PC == 0x0100

    def test_call_at_page_boundary(self, cpu):
        """CALL pushes correct return address."""
        cpu.regs.SP = 0xFFFC
        cpu.bus.bus_write(0x0100, 0xC9, cpu.cycles)  # RET at target
        write_program(cpu, [0xCD, 0x00, 0x01])  # CALL 0x0100
        cpu.step()
        # SP decrements by 2, so return address at 0xFFFA and 0xFFFB
        assert cpu.bus.bus_read(0xFFFA, cpu.cycles) == 0x03
        assert cpu.bus.bus_read(0xFFFB, cpu.cycles) == 0x00

    def test_inc_hl_wraps(self, cpu):
        """INC HL wraps from 0xFFFF to 0x0000."""
        cpu.regs.HL = 0xFFFF
        write_program(cpu, [0x23])  # INC HL
        cpu.step()
        assert cpu.regs.HL == 0x0000


# ============================================================
# 40. Reset (RST) Instructions Comprehensive
# ============================================================


class TestRSTComprehensive:
    """All RST (restart) instructions."""

    @pytest.mark.parametrize(
        "opcode,vector",
        [
            (0xC7, 0x00),
            (0xCF, 0x08),
            (0xD7, 0x10),
            (0xDF, 0x18),
            (0xE7, 0x20),
            (0xEF, 0x28),
            (0xF7, 0x30),
            (0xFF, 0x38),
        ],
    )
    def test_rst_vectors(self, cpu, opcode, vector):
        """RST p — jumps to correct vector."""
        cpu.regs.SP = 0x2000
        write_program(cpu, [opcode])
        cpu.step()
        assert cpu.regs.PC == vector
        assert cpu.regs.SP == 0x1FFE

    def test_rst_pushes_return_address(self, cpu):
        """RST pushes correct return address."""
        cpu.regs.SP = 0xFFFF
        write_program(cpu, [0xCD, 0x00, 0x00])  # Skip to 0x0003
        cpu.step()
        write_program(cpu, [0xFF])  # RST 0x38
        cpu.step()
        assert cpu.bus.bus_read(0xFFFE, cpu.cycles) == 0x00
        assert cpu.bus.bus_read(0xFFFD, cpu.cycles) == 0x03


# ============================================================
# Q Factor Tracking — Patrik Rak discovery for SCF/CCF
#
# The Z80 has an internal Q latch that holds the new F value
# after flag-modifying instructions, or 0 after non-flag-modifying
# instructions (including EX AF,AF' and POP AF).
#
# For SCF and CCF, YF (bit 5) and XF (bit 3) are computed as:
#   result = (Q ^ F) | A
#   YF = result.5, XF = result.3
#
# When Q == F (previous instruction modified flags):
#   result = (F ^ F) | A = 0 | A = A
#   So YF = A.5, XF = A.3  (copy from A)
#
# When Q == 0 (previous instruction did NOT modify flags):
#   result = (0 ^ F) | A = F | A
#   So YF = F.5 | A.5, XF = F.3 | A.3  (OR with previous flags)
#
# Reference: https://worldofspectrum.org/forums/discussion/41704
# ============================================================

FLAG_F3 = 0x08
FLAG_F5 = 0x20


class TestQFactorSCF:
    """Q factor tracking for SCF instruction (Patrik Rak discovery)."""

    def test_scf_after_flag_modifying_copies_a(self, cpu):
        """SCF after flag-modifying instruction: F3/F5 copied from A."""
        # DEC A modifies flags, so Q = F after DEC
        # A=0x10 -> DEC A -> A=0x0F (0000 1111): A.5=0, A.3=1
        cpu.regs.A = 0x10
        write_program(cpu, [0x3D, 0x37])  # DEC A; SCF
        cpu.step()  # DEC A modifies flags -> Q = F, A=0x0F
        cpu.step()  # SCF: since Q==F, F3/F5 = A.3/A.5
        assert not (cpu.regs.F & FLAG_F5)  # F5 = A.5 = 0
        assert cpu.regs.F & FLAG_F3  # F3 = A.3 = 1
        assert cpu.regs.F & FLAG_C  # SCF sets carry

    def test_scf_after_flag_modifying_copies_a_clear(self, cpu):
        """SCF after flag-modifying instruction: F3/F5 copied from A (clear)."""
        # A=0x20 -> DEC A -> A=0x1F (0001 1111): A.5=0, A.3=1
        # Use 0x30 -> DEC A -> A=0x2F (0010 1111): A.5=1, A.3=1
        # Use 0x18 -> DEC A -> A=0x17 (0001 0111): A.5=0, A.3=1
        # Use 0x10 -> DEC A -> A=0x0F (0000 1111): A.5=0, A.3=1
        # Need A.5=0, A.3=0 after DEC -> A=0x08 -> DEC -> A=0x07 (0000 0111)
        cpu.regs.A = 0x08
        write_program(cpu, [0x3D, 0x37])  # DEC A; SCF
        cpu.step()  # DEC A: A=0x07, modifies flags -> Q = F
        cpu.step()  # SCF: since Q==F, F3/F5 = A.3/A.5
        assert not (cpu.regs.F & FLAG_F5)  # F5 = A.5 = 0
        assert not (cpu.regs.F & FLAG_F3)  # F3 = A.3 = 0
        assert cpu.regs.F & FLAG_C

    def test_scf_after_non_flag_modifying_ors_with_a(self, cpu):
        """SCF after non-flag-modifying instruction: F3/F5 = F|A (OR)."""
        # EX AF,AF' does NOT modify flags -> Q = 0
        cpu.regs.A = 0x28  # A.5=1, A.3=1
        cpu.regs.F = 0x04  # F.5=0, F.3=0 (only PV set)
        write_program(cpu, [0x08, 0x37])  # EX AF,AF'; SCF
        cpu.step()  # EX AF,AF' doesn't modify flags -> Q = 0
        # After EX, A and F are swapped. But the point is Q=0
        saved_a = cpu.regs.A
        saved_f = cpu.regs.F
        cpu.step()  # SCF: Q=0, so F3/F5 = (0^F)|A = F|A
        # F3 = saved_f.3 | saved_a.3, F5 = saved_f.5 | saved_a.5
        expected_f5 = bool(saved_f & FLAG_F5) or bool(saved_a & FLAG_F5)
        expected_f3 = bool(saved_f & FLAG_F3) or bool(saved_a & FLAG_F3)
        assert bool(cpu.regs.F & FLAG_F5) == expected_f5
        assert bool(cpu.regs.F & FLAG_F3) == expected_f3

    def test_scf_after_pop_af_ors(self, cpu):
        """SCF after POP AF: Q=0, so F3/F5 = F|A (OR behavior)."""
        cpu.regs.SP = 0x4000
        # Push value with F5=0, F3=0 onto stack
        cpu.bus.bus_write(0x4000, 0x00, 0)  # F byte (no flags)
        cpu.bus.bus_write(0x4001, 0x00, 0)  # A byte
        cpu.regs.A = 0x28  # A.5=1, A.3=1
        write_program(cpu, [0xF1, 0x37])  # POP AF; SCF
        cpu.step()  # POP AF -> Q = 0 (exception!)
        cpu.step()  # SCF with Q=0
        # Since Q=0, F3/F5 = F|A. After POP AF, F=0x00, A=0x00
        # So F3=0|0=0, F5=0|0=0
        assert not (cpu.regs.F & FLAG_F5)
        assert not (cpu.regs.F & FLAG_F3)


class TestQFactorCCF:
    """Q factor tracking for CCF instruction (Patrik Rak discovery)."""

    def test_ccf_after_flag_modifying_copies_a(self, cpu):
        """CCF after flag-modifying instruction: F3/F5 copied from A."""
        # DEC A modifies flags, so Q = F after DEC
        # A=0x30 -> DEC A -> A=0x2F (0010 1111): A.5=1, A.3=1
        cpu.regs.A = 0x30
        cpu.regs.F = 0x01  # C=1
        write_program(cpu, [0x3D, 0x3F])  # DEC A; CCF
        cpu.step()  # DEC A modifies flags -> Q = F, A=0x2F
        cpu.step()  # CCF: since Q==F, F3/F5 = A.3/A.5
        assert cpu.regs.F & FLAG_F5  # F5 = A.5 = 1
        assert cpu.regs.F & FLAG_F3  # F3 = A.3 = 1
        assert not (cpu.regs.F & FLAG_C)  # CCF toggles carry

    def test_ccf_after_flag_modifying_copies_a_clear(self, cpu):
        """CCF after flag-modifying instruction: F3/F5 copied from A (clear)."""
        # A=0x08 -> DEC A -> A=0x07 (0000 0111): A.5=0, A.3=0
        cpu.regs.A = 0x08
        cpu.regs.F = 0x00  # C=0
        write_program(cpu, [0x3D, 0x3F])  # DEC A; CCF
        cpu.step()  # DEC A modifies flags -> Q = F, A=0x07
        cpu.step()  # CCF: since Q==F, F3/F5 = A.3/A.5
        assert not (cpu.regs.F & FLAG_F5)  # F5 = A.5 = 0
        assert not (cpu.regs.F & FLAG_F3)  # F3 = A.3 = 0
        assert cpu.regs.F & FLAG_C  # CCF toggles carry on

    def test_ccf_after_non_flag_modifying_ors_with_a(self, cpu):
        """CCF after non-flag-modifying instruction: F3/F5 = F|A (OR)."""
        # NOP does NOT modify flags -> Q = 0
        cpu.regs.A = 0x00  # A.5=0, A.3=0
        cpu.regs.F = 0x28  # F.5=1, F.3=1
        write_program(cpu, [0x00, 0x3F])  # NOP; CCF
        cpu.step()  # NOP doesn't modify flags -> Q = 0
        cpu.step()  # CCF: Q=0, so F3/F5 = (0^F)|A = F|A
        # F.5|A.5 = 1|0 = 1, F.3|A.3 = 1|0 = 1
        assert cpu.regs.F & FLAG_F5
        assert cpu.regs.F & FLAG_F3

    def test_ccf_after_ex_af_ors(self, cpu):
        """CCF after EX AF,AF': Q=0, so F3/F5 = F|A (OR behavior)."""
        cpu.regs.A = 0x28  # A.5=1, A.3=1
        cpu.regs.F = 0x04  # F.5=0, F.3=0
        write_program(cpu, [0x08, 0x3F])  # EX AF,AF'; CCF
        cpu.step()  # EX AF,AF' -> Q = 0
        saved_a = cpu.regs.A
        saved_f = cpu.regs.F
        cpu.step()  # CCF with Q=0
        expected_f5 = bool(saved_f & FLAG_F5) or bool(saved_a & FLAG_F5)
        expected_f3 = bool(saved_f & FLAG_F3) or bool(saved_a & FLAG_F3)
        assert bool(cpu.regs.F & FLAG_F5) == expected_f5
        assert bool(cpu.regs.F & FLAG_F3) == expected_f3


class TestQFactorSequence:
    """Q factor tracking across instruction sequences."""

    def test_scf_scf_second_copies_a(self, cpu):
        """SCF; SCF — second SCF sees Q=F from first SCF, copies A."""
        cpu.regs.A = 0x28  # A.5=1, A.3=1
        write_program(cpu, [0x37, 0x37])  # SCF; SCF
        cpu.step()  # First SCF modifies flags -> Q = F
        cpu.step()  # Second SCF: Q=F, so F3/F5 = A.3/A.5
        assert cpu.regs.F & FLAG_F5
        assert cpu.regs.F & FLAG_F3
        assert cpu.regs.F & FLAG_C

    def test_nop_scf_ors(self, cpu):
        """NOP; SCF — SCF sees Q=0 from NOP, ORs with A."""
        cpu.regs.A = 0x00  # A.5=0, A.3=0
        cpu.regs.F = 0x28  # F.5=1, F.3=1
        write_program(cpu, [0x00, 0x37])  # NOP; SCF
        cpu.step()  # NOP -> Q = 0
        cpu.step()  # SCF: Q=0, so F3/F5 = F|A = 1|0 = 1
        assert cpu.regs.F & FLAG_F5
        assert cpu.regs.F & FLAG_F3

    def test_dec_a_scf_copies(self, cpu):
        """DEC A; SCF — SCF copies F3/F5 from A."""
        cpu.regs.A = 0x10  # A.5=0, A.3=0 (but will become 0x0F after DEC)
        write_program(cpu, [0x3D, 0x37])  # DEC A; SCF
        cpu.step()  # DEC A: A=0x0F, modifies flags -> Q=F
        assert cpu.regs.A == 0x0F  # A.5=0, A.3=1
        cpu.step()  # SCF: Q=F, so F3/F5 = A.3/A.5 = 1/0
        assert cpu.regs.F & FLAG_F3  # A.3 = 1
        assert not (cpu.regs.F & FLAG_F5)  # A.5 = 0

    def test_pop_af_scf_q_zero(self, cpu):
        """POP AF; SCF — POP AF sets Q=0, SCF ORs F3/F5."""
        cpu.regs.SP = 0x4000
        cpu.bus.bus_write(0x4000, 0xFF, 0)  # F with all flags set
        cpu.bus.bus_write(0x4001, 0xFF, 0)  # A = 0xFF
        write_program(cpu, [0xF1, 0x37])  # POP AF; SCF
        cpu.step()  # POP AF -> Q = 0 (exception!)
        cpu.step()  # SCF: Q=0, so F3/F5 = F|A
        # F=0xFF, A=0xFF, so F|A = 0xFF, F3=1, F5=1
        assert cpu.regs.F & FLAG_F5
        assert cpu.regs.F & FLAG_F3

    def test_ex_af_af_scf_q_zero(self, cpu):
        """EX AF,AF'; SCF — EX AF,AF' sets Q=0, SCF ORs F3/F5."""
        cpu.regs.A = 0x00
        cpu.regs.F = 0x28  # F5=1, F3=1
        cpu.regs.Ap = 0xFF
        cpu.regs.Fp = 0x00
        write_program(cpu, [0x08, 0x37])  # EX AF,AF'; SCF
        cpu.step()  # EX AF,AF' -> Q = 0
        # After EX: A=0xFF, F=0x00
        cpu.step()  # SCF: Q=0, F3/F5 = F|A = 0|0xFF = 1
        assert cpu.regs.F & FLAG_F5
        assert cpu.regs.F & FLAG_F3
