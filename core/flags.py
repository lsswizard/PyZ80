"""
Z80 Flag Computation Module

This module provides flag calculation functions for Z80 arithmetic and logic
operations. It includes precomputed parity tables for performance.

Z80 Flag Layout (F register):
    Bit 7: S - Sign flag (1 if result is negative)
    Bit 6: Z - Zero flag (1 if result is zero)
    Bit 5: F5 - Undocumented (copy of bit 5 of result)
    Bit 4: H - Half Carry flag (carry from bit 3 to bit 4)
    Bit 3: F3 - Undocumented (copy of bit 3 of result)
    Bit 2: P/V - Parity/Overflow flag
    Bit 1: N - Add/Subtract flag (0 after ADD, 1 after SUB)
    Bit 0: C - Carry flag
"""

# Flag bit definitions
FLAG_S = 0x80  # Sign (negative result)
FLAG_Z = 0x40  # Zero (result is zero)
FLAG_F5 = 0x20  # Undocumented: copy of bit 5
FLAG_H = 0x10  # Half Carry (carry from bit 3)
FLAG_F3 = 0x08  # Undocumented: copy of bit 3
FLAG_PV = 0x04  # Parity/Overflow
FLAG_N = 0x02  # Add/Subtract (1 = subtraction)
FLAG_C = 0x01  # Carry flag

# Precomputed parity table for performance
# PARITY_TABLE[n] = 1 if n has even parity, 0 if odd
PARITY_TABLE = bytearray(256)

for i in range(256):
    bits = 0
    n = i
    while n:
        bits += 1
        n &= n - 1
    PARITY_TABLE[i] = 0 if bits % 2 else 1


def parity(n: int) -> int:
    """Calculate parity of 8-bit value (1 = even parity, 0 = odd parity)."""
    return PARITY_TABLE[n & 0xFF]


def get_add_flags(a: int, b: int, carry: int = 0) -> int:
    """Get flags after ADD/ADC operation"""
    result = (a + b + carry) & 0xFF

    flags = 0
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    if ((a & 0x0F) + (b & 0x0F) + carry) > 0x0F:
        flags |= FLAG_H
    if (a + b + carry) > 0xFF:
        flags |= FLAG_C
    # Overflow: set if operands have same sign but result has different sign
    if (~(a ^ b)) & (a ^ result) & 0x80:
        flags |= FLAG_PV
    # Undocumented: bits 3 and 5 copied from result
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def get_sub_flags(a: int, b: int, carry: int = 0) -> int:
    """Get flags after SUB/SBC/CMP operation"""
    result = (a - b - carry) & 0xFF

    flags = FLAG_N
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    if (a & 0x0F) < (b & 0x0F) + carry:
        flags |= FLAG_H
    if a < b + carry:
        flags |= FLAG_C
    # Overflow: ((a ^ b) & (a ^ result)) & 0x80
    if ((a ^ b) & (a ^ result)) & 0x80:
        flags |= FLAG_PV
    # Undocumented: bits 3 and 5 copied from result
    flags |= result & (FLAG_F3 | FLAG_F5)
    return flags


def get_and_flags(result: int) -> int:
    """Get flags after AND operation - flags based on result only"""
    flags = FLAG_H  # Always set for AND
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    flags |= PARITY_TABLE[result] << 2  # P/V = parity
    return flags


def get_or_flags(result: int) -> int:
    """Get flags after OR operation - flags based on result only"""
    flags = 0
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    flags |= PARITY_TABLE[result] << 2  # P/V = parity
    return flags


def get_xor_flags(result: int) -> int:
    """Get flags after XOR operation - flags based on result only"""
    flags = 0
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    flags |= PARITY_TABLE[result] << 2  # P/V = parity
    return flags


def get_inc_flags(a: int) -> int:
    """Get flags after INC operation"""
    result = (a + 1) & 0xFF

    flags = 0
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    if (a & 0x0F) == 0x0F:
        flags |= FLAG_H
    if a == 0x7F:
        flags |= FLAG_PV
    return flags


def get_dec_flags(a: int) -> int:
    """Get flags after DEC operation"""
    result = (a - 1) & 0xFF

    flags = FLAG_N
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S
    if (a & 0x0F) == 0x00:
        flags |= FLAG_H
    if a == 0x80:
        flags |= FLAG_PV
    return flags


def get_cp_flags(a: int, b: int) -> int:
    """Get flags after CP operation (compare)"""
    result = (a - b) & 0xFF

    flags = FLAG_N
    if result == 0:
        flags |= FLAG_Z
    if result & 0x80:
        flags |= FLAG_S

    # Half-carry (borrow from bit 4)
    if (a & 0x0F) < (b & 0x0F):
        flags |= FLAG_H
    # Carry (borrow from bit 8)
    if a < b:
        flags |= FLAG_C

    # FIX: P/V is Overflow, not bit 4.
    # Logic: operands have different signs, and result sign differs from minuend
    if ((a ^ b) & (a ^ result)) & 0x80:
        flags |= FLAG_PV

    # Undocumented bits 3 and 5 are copied from the subtrahend 'b' in CP!
    flags |= b & (FLAG_F3 | FLAG_F5)

    return flags


def get_daa_result(a: int, f: int) -> tuple[int, int]:
    """Decimal Adjust Accumulator after ADD/SUB.

    Returns (new_a, new_f)
    """
    # Clear flags that will be recalculated (including H)
    flags = f & ~(FLAG_PV | FLAG_F3 | FLAG_F5 | FLAG_H | FLAG_Z | FLAG_S)

    lo_correction = False

    if not (f & FLAG_N):
        # After addition
        if (f & FLAG_H) or (a & 0x0F) > 9:
            a += 0x06
            lo_correction = True
        if (f & FLAG_C) or a > 0x9F:
            a += 0x60
            flags |= FLAG_C
        # H set if low nibble correction was applied
        if lo_correction:
            flags |= FLAG_H
    else:
        # After subtraction
        if f & FLAG_H:
            lo_correction = True
            a = (a - 0x06) & 0xFF
        if f & FLAG_C:
            a = (a - 0x60) & 0xFF
        # H set if half-borrow correction was applied and the nibble needed it
        if lo_correction and (a & 0x0F) >= 6:
            flags |= FLAG_H

    a &= 0xFF

    if a == 0:
        flags |= FLAG_Z
    if a & 0x80:
        flags |= FLAG_S
    # Undocumented: F3 = bit 3, F5 = bit 5 of result (direct copy)
    flags |= a & (FLAG_F3 | FLAG_F5)
    if PARITY_TABLE[a]:
        flags |= FLAG_PV

    return a, flags


def get_add16_flags(hl: int, reg: int, current_f: int) -> int:
    """Get flags after ADD HL,ss (S, Z, PV are NOT affected)"""
    # Keep S, Z, and PV from the current flag register
    flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)

    # N is always reset
    # H is set if carry from bit 11
    if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF:
        flags |= FLAG_H
    # C is set if carry from bit 15
    if (hl + reg) > 0xFFFF:
        flags |= FLAG_C

    return flags


def get_adc16_flags(hl: int, reg: int, carry: int) -> int:
    """Get flags after ADC HL,ss"""
    result = (hl + reg + carry) & 0xFFFF

    flags = 0
    if (result & 0xFFFF) == 0:
        flags |= FLAG_Z
    if result & 0x8000:
        flags |= FLAG_S
    if ((hl & 0x0FFF) + (reg & 0x0FFF) + carry) > 0x0FFF:
        flags |= FLAG_H
    if (hl + reg + carry) > 0xFFFF:
        flags |= FLAG_C
    # Overflow: same sign operands produce opposite sign result
    if (~(hl ^ reg)) & (hl ^ result) & 0x8000:
        flags |= FLAG_PV
    return flags


def get_sbc16_flags(hl: int, reg: int, carry: int) -> int:
    """Get flags after SBC HL,ss"""
    result = (hl - reg - carry) & 0xFFFF

    flags = FLAG_N
    if (result & 0xFFFF) == 0:
        flags |= FLAG_Z
    if result & 0x8000:
        flags |= FLAG_S
    if (hl & 0x0FFF) < (reg & 0x0FFF) + carry:
        flags |= FLAG_H
    if hl < reg + carry:
        flags |= FLAG_C
    # Overflow: operands have different sign and result sign differs from minuend
    if (hl ^ reg) & (hl ^ result) & 0x8000:
        flags |= FLAG_PV
    return flags


def _build_cond_table():
    FLAG_S, FLAG_Z, FLAG_PV, FLAG_C = 0x80, 0x40, 0x04, 0x01
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
