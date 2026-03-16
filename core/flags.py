"""
Z80 Flag Computation Module - Optimized with Numba JIT

This module provides high-performance flag calculation functions using Numba JIT
compilation for maximum speed. It also includes spyccy-style lookup tables.

Features:
- Numba JIT-compiled flag calculations (~10x faster)
- Pre-computed lookup tables for parity, half-carry, overflow
- Seamless fallback to pure Python if Numba unavailable
"""

from typing import Optional

# Flag bit definitions
FLAG_S = 0x80  # Sign (negative result)
FLAG_Z = 0x40  # Zero (result is zero)
FLAG_F5 = 0x20  # Undocumented: copy of bit 5
FLAG_H = 0x10  # Half Carry (carry from bit 3)
FLAG_F3 = 0x08  # Undocumented: copy of bit 3
FLAG_PV = 0x04  # Parity/Overflow
FLAG_N = 0x02  # Add/Subtract (1 = subtraction)
FLAG_C = 0x01  # Carry flag

# Try to import numba for JIT compilation
NUMBA_AVAILABLE = False
try:
    from numba import njit
    import numpy as np

    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    np = None


# =============================================================================
# Spyccy-style Lookup Tables
# =============================================================================

# Precomputed parity table (1 = even parity, 0 = odd)
PARITY_TABLE = bytearray(256)
for i in range(256):
    bits = 0
    n = i
    while n:
        bits += 1
        n &= n - 1
    PARITY_TABLE[i] = 0 if bits % 2 else 1

# Half-carry lookup tables (from spyccy)
# Index: (lower_nibble_of_a << 3) | lower_nibble_of_operand
HALFCARRY_ADD_TABLE = bytearray(8)
HALFCARRY_ADD_TABLE[0] = 0
HALFCARRY_ADD_TABLE[1] = FLAG_H
HALFCARRY_ADD_TABLE[2] = FLAG_H
HALFCARRY_ADD_TABLE[3] = FLAG_H
HALFCARRY_ADD_TABLE[4] = 0
HALFCARRY_ADD_TABLE[5] = 0
HALFCARRY_ADD_TABLE[6] = 0
HALFCARRY_ADD_TABLE[7] = FLAG_H

HALFCARRY_SUB_TABLE = bytearray(8)
HALFCARRY_SUB_TABLE[0] = 0
HALFCARRY_SUB_TABLE[1] = 0
HALFCARRY_SUB_TABLE[2] = FLAG_H
HALFCARRY_SUB_TABLE[3] = 0
HALFCARRY_SUB_TABLE[4] = FLAG_H
HALFCARRY_SUB_TABLE[5] = 0
HALFCARRY_SUB_TABLE[6] = FLAG_H
HALFCARRY_SUB_TABLE[7] = FLAG_H

# Overflow lookup tables (from spyccy)
OVERFLOW_ADD_TABLE = bytearray(8)
OVERFLOW_ADD_TABLE[0] = 0
OVERFLOW_ADD_TABLE[1] = 0
OVERFLOW_ADD_TABLE[2] = 0
OVERFLOW_ADD_TABLE[3] = FLAG_PV
OVERFLOW_ADD_TABLE[4] = FLAG_PV
OVERFLOW_ADD_TABLE[5] = 0
OVERFLOW_ADD_TABLE[6] = 0
OVERFLOW_ADD_TABLE[7] = 0

OVERFLOW_SUB_TABLE = bytearray(8)
OVERFLOW_SUB_TABLE[0] = 0
OVERFLOW_SUB_TABLE[1] = FLAG_PV
OVERFLOW_SUB_TABLE[2] = 0
OVERFLOW_SUB_TABLE[3] = 0
OVERFLOW_SUB_TABLE[4] = 0
OVERFLOW_SUB_TABLE[5] = 0
OVERFLOW_SUB_TABLE[6] = FLAG_PV
OVERFLOW_SUB_TABLE[7] = 0

# SZ (Sign Zero) table - combines S and Z flags
SZ_TABLE = bytearray(256)
for i in range(256):
    sz = 0
    if i & 0x80:
        sz |= FLAG_S
    if i == 0:
        sz |= FLAG_Z
    SZ_TABLE[i] = sz

# SZP table - combines SZ with Parity
SZP_TABLE = bytearray(256)
for i in range(256):
    szp = 0
    if i & 0x80:
        szp |= FLAG_S
    if i == 0:
        szp |= FLAG_Z
    szp |= PARITY_TABLE[i] << 2  # P/V = parity
    SZP_TABLE[i] = szp

# SZHZP table - combines SZ with Half-carry, Parity (for AND)
SZHZP_TABLE = bytearray(256)
for i in range(256):
    szhzp = FLAG_H  # AND always sets H
    if i & 0x80:
        szhzp |= FLAG_S
    if i == 0:
        szhzp |= FLAG_Z
    szhzp |= PARITY_TABLE[i] << 2  # P/V = parity
    SZHZP_TABLE[i] = szhzp


# =============================================================================
# Numba JIT-compiled Flag Functions
# =============================================================================

if NUMBA_AVAILABLE:
    # Create numpy versions of lookup tables for JIT
    _PARITY_TABLE = np.array(PARITY_TABLE, dtype=np.uint8)
    _HALFCARRY_ADD_TABLE = np.array(HALFCARRY_ADD_TABLE, dtype=np.uint8)
    _HALFCARRY_SUB_TABLE = np.array(HALFCARRY_SUB_TABLE, dtype=np.uint8)
    _OVERFLOW_ADD_TABLE = np.array(OVERFLOW_ADD_TABLE, dtype=np.uint8)
    _OVERFLOW_SUB_TABLE = np.array(OVERFLOW_SUB_TABLE, dtype=np.uint8)
    _SZ_TABLE = np.array(SZ_TABLE, dtype=np.uint8)
    _SZP_TABLE = np.array(SZP_TABLE, dtype=np.uint8)
    _SZHZP_TABLE = np.array(SZHZP_TABLE, dtype=np.uint8)

    @njit(cache=True)
    def _parity_fast(val: int) -> int:
        """Fast parity calculation using lookup table."""
        return _PARITY_TABLE[val]

    @njit(cache=True)
    def _sz_flags(val: int) -> int:
        """Get Sign and Zero flags from lookup table."""
        return _SZ_TABLE[val]

    @njit(cache=True)
    def _szp_flags(val: int) -> int:
        """Get Sign, Zero, Parity flags from lookup table."""
        return _SZP_TABLE[val]

    @njit(cache=True)
    def _add_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        """JIT-compiled ADD flags calculation."""
        flags = _SZ_TABLE[result]
        # Half-carry from bit 3
        if ((a & 0x0F) + (b & 0x0F)) & 0x10:
            flags |= FLAG_H
        # Carry from bit 7
        if full_result > 0xFF:
            flags |= FLAG_C
        # Overflow: same sign operands, different sign result
        if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        # Copy bits 3 and 5 from result
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _adc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled ADC flags calculation."""
        flags = _SZ_TABLE[result]
        # Half-carry including carry
        if ((a & 0x0F) + (b & 0x0F) + carry) & 0x10:
            flags |= FLAG_H
        # Carry from bit 7
        if full_result > 0xFF:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sub_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        """JIT-compiled SUB flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow)
        if (a & 0x0F) < (b & 0x0F):
            flags |= FLAG_H
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sbc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled SBC flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow) including carry
        if (a & 0x0F) < ((b & 0x0F) + carry):
            flags |= FLAG_H
        # Carry (borrow)
        if full_result < 0:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _inc_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled INC flags calculation."""
        flags = carry_flag & FLAG_C | _SZ_TABLE[new_val]
        if (old_val & 0x0F) == 0x0F:
            flags |= FLAG_H
        if old_val == 0x7F:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _dec_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled DEC flags calculation."""
        flags = (carry_flag & FLAG_C) | FLAG_N | _SZ_TABLE[new_val]
        if (old_val & 0x0F) == 0:
            flags |= FLAG_H
        if old_val == 0x80:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _and_flags_jit(result: int) -> int:
        """JIT-compiled AND flags calculation using lookup table."""
        return _SZHZP_TABLE[result] | (result & (FLAG_F3 | FLAG_F5))

    @njit(cache=True)
    def _or_flags_jit(result: int) -> int:
        """JIT-compiled OR flags calculation using lookup table."""
        flags = _SZP_TABLE[result]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _xor_flags_jit(result: int) -> int:
        """JIT-compiled XOR flags calculation using lookup table."""
        flags = _SZP_TABLE[result]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _cp_flags_jit(a: int, b: int, result: int) -> int:
        """JIT-compiled CP (compare) flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow)
        if (a & 0x0F) < (b & 0x0F):
            flags |= FLAG_H
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        # CP copies bits 3 and 5 from operand, not result
        flags |= b & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _adc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled ADC flags calculation."""
        flags = _SZ_TABLE[result]
        # Half-carry including carry - use lookup table
        flags |= _HALFCARRY_ADD_TABLE[((a & 0x0F) + (b & 0x0F) + carry) & 0x07]
        # Carry from bit 7
        if full_result > 0xFF:
            flags |= FLAG_C
        # Overflow - use lookup table
        flags |= _OVERFLOW_ADD_TABLE[
            ((a >> 6) & 0x07) ^ ((b >> 6) & 0x07) ^ ((result >> 6) & 0x07)
        ]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sub_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        """JIT-compiled SUB flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow) - use lookup table
        flags |= _HALFCARRY_SUB_TABLE[((a & 0x0F) - (b & 0x0F)) & 0x07]
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow - use lookup table
        flags |= _OVERFLOW_SUB_TABLE[
            ((a >> 6) & 0x07) ^ ((b >> 6) & 0x07) ^ ((result >> 6) & 0x07)
        ]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sbc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled SBC flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow) including carry - use lookup table
        flags |= _HALFCARRY_SUB_TABLE[((a & 0x0F) - (b & 0x0F) - carry) & 0x07]
        # Carry (borrow)
        if full_result < 0:
            flags |= FLAG_C
        # Overflow - use lookup table
        flags |= _OVERFLOW_SUB_TABLE[
            ((a >> 6) & 0x07) ^ ((b >> 6) & 0x07) ^ ((result >> 6) & 0x07)
        ]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _inc_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled INC flags calculation."""
        flags = carry_flag & FLAG_C | _SZ_TABLE[new_val]
        if (old_val & 0x0F) == 0x0F:
            flags |= FLAG_H
        if old_val == 0x7F:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _dec_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled DEC flags calculation."""
        flags = (carry_flag & FLAG_C) | FLAG_N | _SZ_TABLE[new_val]
        if (old_val & 0x0F) == 0:
            flags |= FLAG_H
        if old_val == 0x80:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _and_flags_jit(result: int) -> int:
        """JIT-compiled AND flags calculation using lookup table."""
        return _SZHZP_TABLE[result] | (result & (FLAG_F3 | FLAG_F5))

    @njit(cache=True)
    def _or_flags_jit(result: int) -> int:
        """JIT-compiled OR flags calculation using lookup table."""
        flags = _SZP_TABLE[result]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _xor_flags_jit(result: int) -> int:
        """JIT-compiled XOR flags calculation using lookup table."""
        flags = _SZP_TABLE[result]
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _cp_flags_jit(a: int, b: int, result: int) -> int:
        """JIT-compiled CP (compare) flags calculation."""
        flags = FLAG_N | _SZ_TABLE[result]
        # Half-carry (borrow)
        if (a & 0x0F) < (b & 0x0F):
            flags |= FLAG_H
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow
        flags |= _OVERFLOW_SUB_TABLE[
            ((a >> 6) & 0x07) ^ ((b >> 6) & 0x07) ^ ((result >> 6) & 0x07)
        ]
        # CP copies bits 3 and 5 from operand, not result
        flags |= b & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _adc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled ADC flags calculation."""
        flags = 0
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        # Half-carry including carry
        if ((a & 0x0F) + (b & 0x0F) + carry) & 0x10:
            flags |= FLAG_H
        # Carry from bit 7
        if full_result > 0xFF:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sub_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        """JIT-compiled SUB flags calculation."""
        flags = FLAG_N
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        # Half-carry (borrow)
        if (a & 0x0F) < (b & 0x0F):
            flags |= FLAG_H
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _sbc_flags_jit(
        a: int, b: int, carry: int, result: int, full_result: int
    ) -> int:
        """JIT-compiled SBC flags calculation."""
        flags = FLAG_N
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        # Half-carry including carry
        if (a & 0x0F) < ((b & 0x0F) + carry):
            flags |= FLAG_H
        # Carry (borrow)
        if full_result < 0:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _inc_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled INC flags calculation."""
        flags = carry_flag & FLAG_C  # Preserve carry
        if new_val == 0:
            flags |= FLAG_Z
        if new_val & 0x80:
            flags |= FLAG_S
        if (old_val & 0x0F) == 0x0F:
            flags |= FLAG_H
        if old_val == 0x7F:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _dec_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        """JIT-compiled DEC flags calculation."""
        flags = (carry_flag & FLAG_C) | FLAG_N
        if new_val == 0:
            flags |= FLAG_Z
        if new_val & 0x80:
            flags |= FLAG_S
        if (old_val & 0x0F) == 0:
            flags |= FLAG_H
        if old_val == 0x80:
            flags |= FLAG_PV
        flags |= new_val & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _and_flags_jit(result: int) -> int:
        """JIT-compiled AND flags calculation."""
        flags = FLAG_H  # Always set for AND
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        flags |= _parity_fast(result) << 2
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _or_flags_jit(result: int) -> int:
        """JIT-compiled OR flags calculation."""
        flags = 0
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        flags |= _parity_fast(result) << 2
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _xor_flags_jit(result: int) -> int:
        """JIT-compiled XOR flags calculation."""
        flags = 0
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        flags |= _parity_fast(result) << 2
        flags |= result & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _cp_flags_jit(a: int, b: int, result: int) -> int:
        """JIT-compiled CP (compare) flags calculation."""
        flags = FLAG_N
        if result & 0x80:
            flags |= FLAG_S
        if result == 0:
            flags |= FLAG_Z
        # Half-carry (borrow)
        if (a & 0x0F) < (b & 0x0F):
            flags |= FLAG_H
        # Carry (borrow)
        if a < b:
            flags |= FLAG_C
        # Overflow
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
            flags |= FLAG_PV
        # CP copies bits 3 and 5 from operand, not result
        flags |= b & (FLAG_F3 | FLAG_F5)
        return flags

    @njit(cache=True)
    def _add16_flags_jit(hl: int, reg: int, current_f: int) -> int:
        """JIT-compiled ADD HL,ss flags calculation."""
        # Keep S, Z, PV from current flags
        flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)

        # N is always reset
        # H is set if carry from bit 11
        if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF:
            flags |= FLAG_H
        # C is set if carry from bit 15
        if (hl + reg) > 0xFFFF:
            flags |= FLAG_C
        return flags

    @njit(cache=True)
    def _adc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
        """JIT-compiled ADC HL,ss flags calculation."""
        flags = 0
        if result == 0:
            flags |= FLAG_Z
        if result & 0x8000:
            flags |= FLAG_S
        if ((hl & 0x0FFF) + (reg & 0x0FFF) + carry) > 0x0FFF:
            flags |= FLAG_H
        if (hl + reg + carry) > 0xFFFF:
            flags |= FLAG_C
        # Overflow
        if ((hl ^ reg) & 0x8000) == 0 and ((result ^ hl) & 0x8000) != 0:
            flags |= FLAG_PV
        return flags

    @njit(cache=True)
    def _sbc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
        """JIT-compiled SBC HL,ss flags calculation."""
        flags = FLAG_N
        if result == 0:
            flags |= FLAG_Z
        if result & 0x8000:
            flags |= FLAG_S
        if (hl & 0x0FFF) < (reg & 0x0FFF) + carry:
            flags |= FLAG_H
        if hl < reg + carry:
            flags |= FLAG_C
        # Overflow
        if (hl ^ reg) & (hl ^ result) & 0x8000:
            flags |= FLAG_PV
        return flags


# =============================================================================
# Python Fallback Functions (when Numba unavailable)
# =============================================================================


def _parity_python(val: int) -> int:
    """Pure Python parity calculation."""
    return PARITY_TABLE[val & 0xFF]


def _add_flags_python(
    a: int, b: int, result: int, full_result: int, current_f: int
) -> int:
    """Pure Python ADD flags."""
    flags = 0
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    if ((a & 0x0F) + (b & 0x0F)) & 0x10:
        flags |= FLAG_H
    if full_result > 0xFF:
        flags |= FLAG_C
    if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0:
        flags |= FLAG_PV
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _adc_flags_python(
    a: int, b: int, carry: int, result: int, full_result: int, current_f: int
) -> int:
    """Pure Python ADC flags."""
    flags = 0
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    if ((a & 0x0F) + (b & 0x0F) + carry) & 0x10:
        flags |= FLAG_H
    if full_result > 0xFF:
        flags |= FLAG_C
    if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0:
        flags |= FLAG_PV
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _sub_flags_python(
    a: int, b: int, result: int, full_result: int, current_f: int
) -> int:
    """Pure Python SUB flags."""
    flags = FLAG_N
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    if (a & 0x0F) < (b & 0x0F):
        flags |= FLAG_H
    if a < b:
        flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
        flags |= FLAG_PV
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _sbc_flags_python(
    a: int, b: int, carry: int, result: int, full_result: int, current_f: int
) -> int:
    """Pure Python SBC flags."""
    flags = FLAG_N
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    if (a & 0x0F) < ((b & 0x0F) + carry):
        flags |= FLAG_H
    if full_result < 0:
        flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
        flags |= FLAG_PV
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _inc_flags_python(old_val: int, new_val: int, current_f: int) -> int:
    """Pure Python INC flags."""
    flags = current_f & FLAG_C
    if new_val == 0:
        flags |= FLAG_Z
    if new_val & 0x80:
        flags |= FLAG_S
    if (old_val & 0x0F) == 0x0F:
        flags |= FLAG_H
    if old_val == 0x7F:
        flags |= FLAG_PV
    flags |= new_val & (FLAG_F3 | FLAG_F5)
    return flags


def _dec_flags_python(old_val: int, new_val: int, current_f: int) -> int:
    """Pure Python DEC flags."""
    flags = (current_f & FLAG_C) | FLAG_N
    if new_val == 0:
        flags |= FLAG_Z
    if new_val & 0x80:
        flags |= FLAG_S
    if (old_val & 0x0F) == 0:
        flags |= FLAG_H
    if old_val == 0x80:
        flags |= FLAG_PV
    flags |= new_val & (FLAG_F3 | FLAG_F5)
    return flags


def _and_flags_python(result: int) -> int:
    """Pure Python AND flags."""
    flags = FLAG_H
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    flags |= PARITY_TABLE[result] << 2
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _or_flags_python(result: int) -> int:
    """Pure Python OR flags."""
    flags = 0
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    flags |= PARITY_TABLE[result] << 2
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _xor_flags_python(result: int) -> int:
    """Pure Python XOR flags."""
    flags = 0
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    flags |= PARITY_TABLE[result] << 2
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def _cp_flags_python(a: int, b: int, result: int) -> int:
    """Pure Python CP flags."""
    flags = FLAG_N
    if result & 0x80:
        flags |= FLAG_S
    if result == 0:
        flags |= FLAG_Z
    if (a & 0x0F) < (b & 0x0F):
        flags |= FLAG_H
    if a < b:
        flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0:
        flags |= FLAG_PV
    flags |= b & (FLAG_F3 | FLAG_F5)
    return flags


def _add16_flags_python(hl: int, reg: int, current_f: int) -> int:
    """Pure Python ADD HL,ss flags."""
    flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)
    if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF:
        flags |= FLAG_H
    if (hl + reg) > 0xFFFF:
        flags |= FLAG_C
    return flags


def _adc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
    """Pure Python ADC HL,ss flags."""
    flags = 0
    if result == 0:
        flags |= FLAG_Z
    if result & 0x8000:
        flags |= FLAG_S
    if ((hl & 0x0FFF) + (reg & 0x0FFF) + carry) > 0x0FFF:
        flags |= FLAG_H
    if (hl + reg + carry) > 0xFFFF:
        flags |= FLAG_C
    if ((hl ^ reg) & 0x8000) == 0 and ((result ^ hl) & 0x8000) != 0:
        flags |= FLAG_PV
    return flags


def _sbc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
    """Pure Python SBC HL,ss flags."""
    flags = FLAG_N
    if result == 0:
        flags |= FLAG_Z
    if result & 0x8000:
        flags |= FLAG_S
    if (hl & 0x0FFF) < (reg & 0x0FFF) + carry:
        flags |= FLAG_H
    if hl < reg + carry:
        flags |= FLAG_C
    if (hl ^ reg) & (hl ^ result) & 0x8000:
        flags |= FLAG_PV
    return flags


# =============================================================================
# Public API - Selects JIT or Python based on availability
# =============================================================================

if NUMBA_AVAILABLE:
    # Use JIT-compiled versions
    parity = _parity_fast
    add_flags = lambda a, b, carry=0, f=0: _add_flags_jit(
        a, b, (a + b + carry) & 0xFF, a + b + carry
    )
    adc_flags = lambda a, b, carry, f=0: _adc_flags_jit(
        a, b, carry, (a + b + carry) & 0xFF, a + b + carry
    )
    sub_flags = lambda a, b, carry=0, f=0: _sub_flags_jit(
        a, b, (a - b - carry) & 0xFF, a - b - carry
    )
    sbc_flags = lambda a, b, carry, f=0: _sbc_flags_jit(
        a, b, carry, (a - b - carry) & 0xFF, a - b - carry
    )
    inc_flags = lambda old, new, f=0: _inc_flags_jit(old, new, f)
    dec_flags = lambda old, new, f=0: _dec_flags_jit(old, new, f)
    and_flags = _and_flags_jit
    or_flags = _or_flags_jit
    xor_flags = _xor_flags_jit
    cp_flags = lambda a, b, f=0: _cp_flags_jit(a, b, (a - b) & 0xFF)
    add16_flags = _add16_flags_jit
    adc16_flags = _adc16_flags_jit
    sbc16_flags = _sbc16_flags_jit
else:
    # Fallback to Python functions with proper wrappers
    parity = _parity_python
    add_flags = lambda a, b, carry=0, f=0: _add_flags_python(
        a, b, (a + b + carry) & 0xFF, a + b + carry, f
    )
    adc_flags = lambda a, b, carry, f=0: _adc_flags_python(
        a, b, carry, (a + b + carry) & 0xFF, a + b + carry, f
    )
    sub_flags = lambda a, b, carry=0, f=0: _sub_flags_python(
        a, b, (a - b - carry) & 0xFF, a - b - carry, f
    )
    sbc_flags = lambda a, b, carry, f=0: _sbc_flags_python(
        a, b, carry, (a - b - carry) & 0xFF, a - b - carry, f
    )
    inc_flags = lambda old, new, f=0: _inc_flags_python(old, new, f)
    dec_flags = lambda old, new, f=0: _dec_flags_python(old, new, f)
    and_flags = _and_flags_python
    or_flags = _or_flags_python
    xor_flags = _xor_flags_python
    cp_flags = lambda a, b, f=0: _cp_flags_python(a, b, (a - b) & 0xFF)
    add16_flags = _add16_flags_python
    adc16_flags = _adc16_flags_python
    sbc16_flags = _sbc16_flags_python

# =============================================================================
# Legacy API compatibility with original flags.py
# =============================================================================


def get_add_flags(a: int, b: int, carry: int = 0) -> int:
    """Get flags after ADD/ADC operation."""
    if carry:
        full = a + b + carry
        return adc_flags(a, b, carry)
    else:
        full = a + b
        return add_flags(a, b)


def get_sub_flags(a: int, b: int, carry: int = 0) -> int:
    """Get flags after SUB/SBC/CMP operation."""
    return sub_flags(a, b, carry)


def get_and_flags(result: int) -> int:
    """Get flags after AND operation."""
    return and_flags(result)


def get_or_flags(result: int) -> int:
    """Get flags after OR operation."""
    return or_flags(result)


def get_xor_flags(result: int) -> int:
    """Get flags after XOR operation."""
    return xor_flags(result)


def get_inc_flags(a: int) -> int:
    """Get flags after INC operation."""
    return inc_flags(a, (a + 1) & 0xFF)


def get_dec_flags(a: int) -> int:
    """Get flags after DEC operation."""
    return dec_flags(a, (a - 1) & 0xFF)


def get_cp_flags(a: int, b: int) -> int:
    """Get flags after CP operation."""
    return cp_flags(a, b)


def get_add16_flags(hl: int, reg: int, current_f: int) -> int:
    """Get flags after ADD HL,ss."""
    return add16_flags(hl, reg, current_f)


def get_adc16_flags(hl: int, reg: int, carry: int) -> int:
    """Get flags after ADC HL,ss."""
    result = (hl + reg + carry) & 0xFFFF
    return adc16_flags(hl, reg, carry, result)


def get_sbc16_flags(hl: int, reg: int, carry: int) -> int:
    """Get flags after SBC HL,ss."""
    result = (hl - reg - carry) & 0xFFFF
    return sbc16_flags(hl, reg, carry, result)


def get_daa_result(a: int, f: int) -> tuple:
    """DAA - Decimal Adjust Accumulator."""
    flags = f & ~(FLAG_PV | FLAG_F3 | FLAG_F5 | FLAG_H | FLAG_Z | FLAG_S)
    lo_correction = False

    if not (f & FLAG_N):
        if (f & FLAG_H) or (a & 0x0F) > 9:
            a = (a + 0x06) & 0xFF
            lo_correction = True
        if (f & FLAG_C) or a > 0x9F:
            a = (a + 0x60) & 0xFF
            flags |= FLAG_C
        if lo_correction:
            flags |= FLAG_H
    else:
        if f & FLAG_H:
            lo_correction = True
            a = (a - 0x06) & 0xFF
        if f & FLAG_C:
            a = (a - 0x60) & 0xFF
        if lo_correction and (a & 0x0F) >= 6:
            flags |= FLAG_H

    if a == 0:
        flags |= FLAG_Z
    if a & 0x80:
        flags |= FLAG_S
    flags |= a & (FLAG_F3 | FLAG_F5)
    if PARITY_TABLE[a]:
        flags |= FLAG_PV

    return a, flags


# Condition table for fast condition checking
def _build_cond_table():
    table = [[False] * 8 for _ in range(256)]
    for f in range(256):
        table[f][0] = not (f & FLAG_Z)  # NZ
        table[f][1] = bool(f & FLAG_Z)  # Z
        table[f][2] = not (f & FLAG_C)  # NC
        table[f][3] = bool(f & FLAG_C)  # C
        table[f][4] = not (f & FLAG_PV)  # PO
        table[f][5] = bool(f & FLAG_PV)  # PE
        table[f][6] = not (f & FLAG_S)  # P
        table[f][7] = bool(f & FLAG_S)  # M
    return table


COND_TABLE = _build_cond_table()
