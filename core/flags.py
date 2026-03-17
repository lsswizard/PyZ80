"""
Z80 Flag Computation Module

Provides high-performance flag calculation functions with optional Numba JIT
acceleration. Uses pre-computed lookup tables to reduce branching on the hot path.

Key lookup tables:
    SZ_TABLE    — Sign + Zero flags
    SZ53_TABLE  — Sign + Zero + undocumented bits 3 and 5
    SZP_TABLE   — Sign + Zero + Parity
    SZ53P_TABLE — Sign + Zero + Parity + undocumented bits 3/5  (OR / XOR result)
    SZHZP_TABLE — SZ53P with H always set                        (AND result)
"""

# ---------------------------------------------------------------------------
# Flag bit definitions
# ---------------------------------------------------------------------------
FLAG_S  = 0x80  # Sign
FLAG_Z  = 0x40  # Zero
FLAG_F5 = 0x20  # Undocumented: copy of result bit 5
FLAG_H  = 0x10  # Half Carry
FLAG_F3 = 0x08  # Undocumented: copy of result bit 3
FLAG_PV = 0x04  # Parity / Overflow
FLAG_N  = 0x02  # Add/Subtract (1 = subtraction)
FLAG_C  = 0x01  # Carry

_F53 = FLAG_F3 | FLAG_F5  # combined undocumented-bits mask

# ---------------------------------------------------------------------------
# Optional Numba JIT
# ---------------------------------------------------------------------------
NUMBA_AVAILABLE = False
try:
    from numba import njit
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    np = None


# ===========================================================================
# Pre-computed lookup tables
# ===========================================================================

# Parity: 1 = even (PV set), 0 = odd
PARITY_TABLE = bytearray(256)
for _i in range(256):
    _b, _n = 0, _i
    while _n:
        _b += 1
        _n &= _n - 1
    PARITY_TABLE[_i] = 0 if _b % 2 else 1

# Spyccy-style half-carry / overflow lookup tables
HALFCARRY_ADD_TABLE = bytearray([0, FLAG_H, FLAG_H, FLAG_H, 0, 0, 0, FLAG_H])
HALFCARRY_SUB_TABLE = bytearray([0, 0, FLAG_H, 0, FLAG_H, 0, FLAG_H, FLAG_H])
OVERFLOW_ADD_TABLE  = bytearray([0, 0, 0, FLAG_PV, FLAG_PV, 0, 0, 0])
OVERFLOW_SUB_TABLE  = bytearray([0, FLAG_PV, 0, 0, 0, 0, FLAG_PV, 0])

# Sign + Zero
SZ_TABLE = bytearray(256)
for _i in range(256):
    _v = 0
    if _i & 0x80: _v |= FLAG_S
    if _i == 0:   _v |= FLAG_Z
    SZ_TABLE[_i] = _v

# Sign + Zero + undocumented bits 3/5
SZ53_TABLE = bytearray(SZ_TABLE[_i] | (_i & _F53) for _i in range(256))

# Sign + Zero + Parity
SZP_TABLE = bytearray(SZ_TABLE[_i] | (PARITY_TABLE[_i] << 2) for _i in range(256))

# Sign + Zero + Parity + undocumented bits 3/5  — single lookup for OR / XOR
SZ53P_TABLE = bytearray(SZP_TABLE[_i] | (_i & _F53) for _i in range(256))

# As above with H always set — single lookup for AND
SZHZP_TABLE = bytearray(SZ53P_TABLE[_i] | FLAG_H for _i in range(256))


# ===========================================================================
# Numba JIT functions  (single definition each — no duplicate compilations)
# ===========================================================================

if NUMBA_AVAILABLE:
    _NP_PARITY   = np.array(PARITY_TABLE,  dtype=np.uint8)
    _NP_SZ       = np.array(SZ_TABLE,      dtype=np.uint8)
    _NP_SZ53P    = np.array(SZ53P_TABLE,   dtype=np.uint8)
    _NP_SZHZP    = np.array(SZHZP_TABLE,   dtype=np.uint8)

    @njit(cache=True)
    def _parity_fast(val: int) -> int:
        return int(_NP_PARITY[val & 0xFF])

    @njit(cache=True)
    def _add_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        flags = int(_NP_SZ[result])
        if ((a & 0x0F) + (b & 0x0F)) & 0x10: flags |= FLAG_H
        if full_result > 0xFF: flags |= FLAG_C
        if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
        flags |= result & _F53
        return flags

    @njit(cache=True)
    def _adc_flags_jit(a: int, b: int, carry: int, result: int, full_result: int) -> int:
        flags = 0
        if result & 0x80: flags |= FLAG_S
        if result == 0:   flags |= FLAG_Z
        if ((a & 0x0F) + (b & 0x0F) + carry) & 0x10: flags |= FLAG_H
        if full_result > 0xFF: flags |= FLAG_C
        if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
        flags |= result & _F53
        return flags

    @njit(cache=True)
    def _sub_flags_jit(a: int, b: int, result: int, full_result: int) -> int:
        flags = FLAG_N
        if result & 0x80: flags |= FLAG_S
        if result == 0:   flags |= FLAG_Z
        if (a & 0x0F) < (b & 0x0F): flags |= FLAG_H
        if a < b: flags |= FLAG_C
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
        flags |= result & _F53
        return flags

    @njit(cache=True)
    def _sbc_flags_jit(a: int, b: int, carry: int, result: int, full_result: int) -> int:
        flags = FLAG_N
        if result & 0x80: flags |= FLAG_S
        if result == 0:   flags |= FLAG_Z
        if (a & 0x0F) < ((b & 0x0F) + carry): flags |= FLAG_H
        if full_result < 0: flags |= FLAG_C
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
        flags |= result & _F53
        return flags

    @njit(cache=True)
    def _inc_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        flags = carry_flag & FLAG_C
        if new_val == 0:   flags |= FLAG_Z
        if new_val & 0x80: flags |= FLAG_S
        if (old_val & 0x0F) == 0x0F: flags |= FLAG_H
        if old_val == 0x7F: flags |= FLAG_PV
        flags |= new_val & _F53
        return flags

    @njit(cache=True)
    def _dec_flags_jit(old_val: int, new_val: int, carry_flag: int) -> int:
        flags = (carry_flag & FLAG_C) | FLAG_N
        if new_val == 0:   flags |= FLAG_Z
        if new_val & 0x80: flags |= FLAG_S
        if (old_val & 0x0F) == 0: flags |= FLAG_H
        if old_val == 0x80: flags |= FLAG_PV
        flags |= new_val & _F53
        return flags

    @njit(cache=True)
    def _and_flags_jit(result: int) -> int:
        """AND always sets H; F3/F5 from result. Single table lookup."""
        return int(_NP_SZHZP[result])

    @njit(cache=True)
    def _or_flags_jit(result: int) -> int:
        """OR clears H/N/C; P from parity; F3/F5 from result. Single table lookup."""
        return int(_NP_SZ53P[result])

    @njit(cache=True)
    def _xor_flags_jit(result: int) -> int:
        """XOR clears H/N/C; P from parity; F3/F5 from result. Single table lookup."""
        return int(_NP_SZ53P[result])

    @njit(cache=True)
    def _cp_flags_jit(a: int, b: int, result: int) -> int:
        """CP: F3/F5 come from the *operand* b, not the result."""
        flags = FLAG_N
        if result & 0x80: flags |= FLAG_S
        if result == 0:   flags |= FLAG_Z
        if (a & 0x0F) < (b & 0x0F): flags |= FLAG_H
        if a < b: flags |= FLAG_C
        if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
        flags |= b & _F53
        return flags

    @njit(cache=True)
    def _add16_flags_jit(hl: int, reg: int, current_f: int) -> int:
        """ADD HL,ss: preserves S/Z/PV; N cleared; H/C/F3/F5 updated."""
        full = hl + reg
        r16  = full & 0xFFFF
        flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)
        if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF: flags |= FLAG_H
        if full > 0xFFFF: flags |= FLAG_C
        flags |= (r16 >> 8) & _F53   # undocumented: from high byte of result
        return flags

    @njit(cache=True)
    def _adc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
        flags = 0
        if result == 0:      flags |= FLAG_Z
        if result & 0x8000:  flags |= FLAG_S
        if ((hl & 0x0FFF) + (reg & 0x0FFF) + carry) > 0x0FFF: flags |= FLAG_H
        if (hl + reg + carry) > 0xFFFF: flags |= FLAG_C
        if ((hl ^ reg) & 0x8000) == 0 and ((result ^ hl) & 0x8000) != 0: flags |= FLAG_PV
        flags |= (result >> 8) & _F53
        return flags

    @njit(cache=True)
    def _sbc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
        flags = FLAG_N
        if result == 0:      flags |= FLAG_Z
        if result & 0x8000:  flags |= FLAG_S
        if (hl & 0x0FFF) < (reg & 0x0FFF) + carry: flags |= FLAG_H
        if hl < reg + carry: flags |= FLAG_C
        if (hl ^ reg) & (hl ^ result) & 0x8000: flags |= FLAG_PV
        flags |= (result >> 8) & _F53
        return flags


# ===========================================================================
# Pure-Python fallback functions
# ===========================================================================

def _parity_python(val: int) -> int:
    return PARITY_TABLE[val & 0xFF]

def _add_flags_python(a: int, b: int, result: int, full_result: int) -> int:
    flags = 0
    if result & 0x80: flags |= FLAG_S
    if result == 0:   flags |= FLAG_Z
    if ((a & 0x0F) + (b & 0x0F)) & 0x10: flags |= FLAG_H
    if full_result > 0xFF: flags |= FLAG_C
    if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
    return flags | (result & _F53)

def _adc_flags_python(a: int, b: int, carry: int, result: int, full_result: int) -> int:
    flags = 0
    if result & 0x80: flags |= FLAG_S
    if result == 0:   flags |= FLAG_Z
    if ((a & 0x0F) + (b & 0x0F) + carry) & 0x10: flags |= FLAG_H
    if full_result > 0xFF: flags |= FLAG_C
    if ((a ^ b) & 0x80) == 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
    return flags | (result & _F53)

def _sub_flags_python(a: int, b: int, result: int, full_result: int) -> int:
    flags = FLAG_N
    if result & 0x80: flags |= FLAG_S
    if result == 0:   flags |= FLAG_Z
    if (a & 0x0F) < (b & 0x0F): flags |= FLAG_H
    if a < b: flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
    return flags | (result & _F53)

def _sbc_flags_python(a: int, b: int, carry: int, result: int, full_result: int) -> int:
    flags = FLAG_N
    if result & 0x80: flags |= FLAG_S
    if result == 0:   flags |= FLAG_Z
    if (a & 0x0F) < ((b & 0x0F) + carry): flags |= FLAG_H
    if full_result < 0: flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
    return flags | (result & _F53)

def _inc_flags_python(old_val: int, new_val: int, carry_flag: int) -> int:
    flags = carry_flag & FLAG_C
    if new_val == 0:   flags |= FLAG_Z
    if new_val & 0x80: flags |= FLAG_S
    if (old_val & 0x0F) == 0x0F: flags |= FLAG_H
    if old_val == 0x7F: flags |= FLAG_PV
    return flags | (new_val & _F53)

def _dec_flags_python(old_val: int, new_val: int, carry_flag: int) -> int:
    flags = (carry_flag & FLAG_C) | FLAG_N
    if new_val == 0:   flags |= FLAG_Z
    if new_val & 0x80: flags |= FLAG_S
    if (old_val & 0x0F) == 0: flags |= FLAG_H
    if old_val == 0x80: flags |= FLAG_PV
    return flags | (new_val & _F53)

def _and_flags_python(result: int) -> int:
    return SZHZP_TABLE[result]   # includes H, F3, F5 — no extra OR needed

def _or_flags_python(result: int) -> int:
    return SZ53P_TABLE[result]   # includes F3, F5

def _xor_flags_python(result: int) -> int:
    return SZ53P_TABLE[result]   # includes F3, F5

def _cp_flags_python(a: int, b: int, result: int) -> int:
    """CP: F3/F5 from operand b, not the subtraction result."""
    flags = FLAG_N
    if result & 0x80: flags |= FLAG_S
    if result == 0:   flags |= FLAG_Z
    if (a & 0x0F) < (b & 0x0F): flags |= FLAG_H
    if a < b: flags |= FLAG_C
    if ((a ^ b) & 0x80) != 0 and ((result ^ a) & 0x80) != 0: flags |= FLAG_PV
    return flags | (b & _F53)

def _add16_flags_python(hl: int, reg: int, current_f: int) -> int:
    """ADD HL,ss: S/Z/PV preserved; N cleared; H/C/F3/F5 updated."""
    full  = hl + reg
    r16   = full & 0xFFFF
    flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)
    if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF: flags |= FLAG_H
    if full > 0xFFFF: flags |= FLAG_C
    flags |= (r16 >> 8) & _F53
    return flags

def _adc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
    flags = 0
    if result == 0:      flags |= FLAG_Z
    if result & 0x8000:  flags |= FLAG_S
    if ((hl & 0x0FFF) + (reg & 0x0FFF) + carry) > 0x0FFF: flags |= FLAG_H
    if (hl + reg + carry) > 0xFFFF: flags |= FLAG_C
    if ((hl ^ reg) & 0x8000) == 0 and ((result ^ hl) & 0x8000) != 0: flags |= FLAG_PV
    flags |= (result >> 8) & _F53
    return flags

def _sbc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
    flags = FLAG_N
    if result == 0:      flags |= FLAG_Z
    if result & 0x8000:  flags |= FLAG_S
    if (hl & 0x0FFF) < (reg & 0x0FFF) + carry: flags |= FLAG_H
    if hl < reg + carry: flags |= FLAG_C
    if (hl ^ reg) & (hl ^ result) & 0x8000: flags |= FLAG_PV
    flags |= (result >> 8) & _F53
    return flags


# ===========================================================================
# Public API — selects JIT or Python
# ===========================================================================

if NUMBA_AVAILABLE:
    parity      = _parity_fast
    add_flags   = lambda a, b, carry=0, f=0: _add_flags_jit(a, b, (a+b+carry)&0xFF, a+b+carry)
    adc_flags   = lambda a, b, carry,  f=0: _adc_flags_jit(a, b, carry, (a+b+carry)&0xFF, a+b+carry)
    sub_flags   = lambda a, b, carry=0, f=0: _sub_flags_jit(a, b, (a-b-carry)&0xFF, a-b-carry)
    sbc_flags   = lambda a, b, carry,  f=0: _sbc_flags_jit(a, b, carry, (a-b-carry)&0xFF, a-b-carry)
    inc_flags   = lambda old, new, f=0: _inc_flags_jit(old, new, f)
    dec_flags   = lambda old, new, f=0: _dec_flags_jit(old, new, f)
    and_flags   = _and_flags_jit
    or_flags    = _or_flags_jit
    xor_flags   = _xor_flags_jit
    cp_flags    = lambda a, b, f=0: _cp_flags_jit(a, b, (a-b)&0xFF)
    add16_flags = _add16_flags_jit
    adc16_flags = _adc16_flags_jit
    sbc16_flags = _sbc16_flags_jit
else:
    parity      = _parity_python
    add_flags   = lambda a, b, carry=0, f=0: _add_flags_python(a, b, (a+b+carry)&0xFF, a+b+carry)
    adc_flags   = lambda a, b, carry,  f=0: _adc_flags_python(a, b, carry, (a+b+carry)&0xFF, a+b+carry)
    sub_flags   = lambda a, b, carry=0, f=0: _sub_flags_python(a, b, (a-b-carry)&0xFF, a-b-carry)
    sbc_flags   = lambda a, b, carry,  f=0: _sbc_flags_python(a, b, carry, (a-b-carry)&0xFF, a-b-carry)
    inc_flags   = lambda old, new, f=0: _inc_flags_python(old, new, f)
    dec_flags   = lambda old, new, f=0: _dec_flags_python(old, new, f)
    and_flags   = _and_flags_python
    or_flags    = _or_flags_python
    xor_flags   = _xor_flags_python
    cp_flags    = lambda a, b, f=0: _cp_flags_python(a, b, (a-b)&0xFF)
    add16_flags = _add16_flags_python
    adc16_flags = _adc16_flags_python
    sbc16_flags = _sbc16_flags_python


# ===========================================================================
# Legacy compatibility wrappers
# ===========================================================================

def get_add_flags(a: int, b: int, carry: int = 0) -> int:
    return adc_flags(a, b, carry) if carry else add_flags(a, b)

def get_sub_flags(a: int, b: int, carry: int = 0) -> int:
    return sbc_flags(a, b, carry) if carry else sub_flags(a, b)

def get_and_flags(result: int) -> int:   return and_flags(result)
def get_or_flags(result: int) -> int:    return or_flags(result)
def get_xor_flags(result: int) -> int:   return xor_flags(result)
def get_cp_flags(a: int, b: int) -> int: return cp_flags(a, b)

def get_inc_flags(a: int) -> int:
    return inc_flags(a, (a + 1) & 0xFF)

def get_dec_flags(a: int) -> int:
    return dec_flags(a, (a - 1) & 0xFF)

def get_add16_flags(hl: int, reg: int, current_f: int) -> int:
    return add16_flags(hl, reg, current_f)

def get_adc16_flags(hl: int, reg: int, carry: int) -> int:
    return adc16_flags(hl, reg, carry, (hl + reg + carry) & 0xFFFF)

def get_sbc16_flags(hl: int, reg: int, carry: int) -> int:
    return sbc16_flags(hl, reg, carry, (hl - reg - carry) & 0xFFFF)


# ===========================================================================
# DAA — Decimal Adjust Accumulator
# ===========================================================================

def get_daa_result(a: int, f: int) -> tuple:
    """Apply DAA correction; return (new_a, new_flags)."""
    flags = f & ~(FLAG_PV | FLAG_F3 | FLAG_F5 | FLAG_H | FLAG_Z | FLAG_S)
    lo_correction = False

    if not (f & FLAG_N):                           # after addition
        if (f & FLAG_H) or (a & 0x0F) > 9:
            a = (a + 0x06) & 0xFF
            lo_correction = True
        if (f & FLAG_C) or a > 0x9F:
            a = (a + 0x60) & 0xFF
            flags |= FLAG_C
        if lo_correction:
            flags |= FLAG_H
    else:                                           # after subtraction
        if f & FLAG_H:
            lo_correction = True
            a = (a - 0x06) & 0xFF
        if f & FLAG_C:
            a = (a - 0x60) & 0xFF
            flags |= FLAG_C
        if lo_correction and (a & 0x0F) >= 6:
            flags |= FLAG_H

    if a == 0:    flags |= FLAG_Z
    if a & 0x80:  flags |= FLAG_S
    flags |= a & _F53
    if PARITY_TABLE[a]: flags |= FLAG_PV

    return a, flags


# ===========================================================================
# Condition table  (indexed by [flags_byte][condition_code])
# ===========================================================================

def _build_cond_table():
    table = [[False] * 8 for _ in range(256)]
    for f in range(256):
        table[f][0] = not (f & FLAG_Z)    # NZ
        table[f][1] = bool(f & FLAG_Z)    # Z
        table[f][2] = not (f & FLAG_C)    # NC
        table[f][3] = bool(f & FLAG_C)    # C
        table[f][4] = not (f & FLAG_PV)   # PO
        table[f][5] = bool(f & FLAG_PV)   # PE
        table[f][6] = not (f & FLAG_S)    # P (positive)
        table[f][7] = bool(f & FLAG_S)    # M (minus)
    return table

COND_TABLE = _build_cond_table()
