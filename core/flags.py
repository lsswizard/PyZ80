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
    ROT_RESULT  — Rotate/Shift results (op x 256)
    ROT_CARRY   — Rotate/Shift carry out (op x 256)
    RL_CARRY_0/1, RR_CARRY_0/1 — RL/RR through carry results
"""

__all__ = [
    "FLAG_S",
    "FLAG_Z",
    "FLAG_H",
    "FLAG_PV",
    "FLAG_N",
    "FLAG_C",
    "FLAG_F5",
    "FLAG_F3",
    "PARITY_TABLE",
    "SZ_TABLE",
    "SZ53_TABLE",
    "SZP_TABLE",
    "SZ53P_TABLE",
    "SZHZP_TABLE",
    "ROT_RESULT",
    "ROT_CARRY",
    "RL_CARRY_0",
    "RL_CARRY_1",
    "RR_CARRY_0",
    "RR_CARRY_1",
    "ADD_FLAGS",
    "ADC_FLAGS",
    "SUB_FLAGS",
    "SBC_FLAGS",
    "CP_FLAGS",
    "INC_FLAGS",
    "DEC_FLAGS_TBL",
    "COND_TABLE",
    "_ADD_PAIR",
    "_SUB_PAIR",
    "DAA_TABLE",
    "DAA_FULL_FLAGS",
    "get_daa_result",
    "get_adc16_flags",
    "get_sbc16_flags",
    "NUMBA_AVAILABLE",
]

# ---------------------------------------------------------------------------
# Flag bit definitions
# ---------------------------------------------------------------------------
FLAG_S = 0x80  # Sign
FLAG_Z = 0x40  # Zero
FLAG_F5 = 0x20  # Undocumented: copy of result bit 5
FLAG_H = 0x10  # Half Carry
FLAG_F3 = 0x08  # Undocumented: copy of result bit 3
FLAG_PV = 0x04  # Parity / Overflow
FLAG_N = 0x02  # Add/Subtract (1 = subtraction)
FLAG_C = 0x01  # Carry

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

# Sign + Zero
SZ_TABLE = bytearray(256)
for _i in range(256):
    _v = 0
    if _i & 0x80:
        _v |= FLAG_S
    if _i == 0:
        _v |= FLAG_Z
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
# Rotate/Shift result tables  (256 x 8 = 2KB)
# op: 0=RLC, 1=RRC, 2=RL, 3=RR, 4=SLA, 5=SRA, 6=SLL, 7=SRL
# ===========================================================================


def _build_rot_tables():
    rot_result = [bytearray(256) for _ in range(8)]
    rot_carry = [bytearray(256) for _ in range(8)]
    for v in range(256):
        rot_result[0][v] = ((v << 1) | (v >> 7)) & 0xFF  # RLC
        rot_carry[0][v] = (v >> 7) & 1
        rot_result[1][v] = ((v >> 1) | (v << 7)) & 0xFF  # RRC
        rot_carry[1][v] = v & 1
        rot_result[4][v] = (v << 1) & 0xFF  # SLA
        rot_carry[4][v] = (v >> 7) & 1
        rot_result[5][v] = (v >> 1) | (v & 0x80)  # SRA
        rot_carry[5][v] = v & 1
        rot_result[6][v] = ((v << 1) | 1) & 0xFF  # SLL
        rot_carry[6][v] = (v >> 7) & 1
        rot_result[7][v] = v >> 1  # SRL
        rot_carry[7][v] = v & 1
    return rot_result, rot_carry


ROT_RESULT, ROT_CARRY = _build_rot_tables()

# RL/RR through carry — result depends on carry in (0 or 1)
RL_CARRY_0 = bytearray(((v << 1) & 0xFF) for v in range(256))
RL_CARRY_1 = bytearray(((v << 1) | 1) & 0xFF for v in range(256))
RR_CARRY_0 = bytearray(v >> 1 for v in range(256))
RR_CARRY_1 = bytearray(((v >> 1) | 0x80) for v in range(256))

# ===========================================================================
# Precomputed 8-bit ALU flag tables
# Eliminates 5-8 branch operations per ALU instruction.
# Total: ~320 KB; fits in L2 cache and stays warm during emulation.
#
# ADD_FLAGS[(a<<8)|b]  — ADD A,x   result flags (no carry in)
# ADC_FLAGS[(a<<8)|b]  — ADC A,x   result flags (carry in = 1)
# SUB_FLAGS[(a<<8)|b]  — SUB/CP x  result flags (no carry in)
# SBC_FLAGS[(a<<8)|b]  — SBC A,x   result flags (carry in = 1)
# CP_FLAGS[(a<<8)|b]   — CP x      result flags (F3/F5 from operand b)
# INC_FLAGS[a]         — INC a     result flags (C not included)
# DEC_FLAGS_TBL[a]     — DEC a     result flags (C not included)
#
# _ADD_PAIR / _SUB_PAIR allow carry dispatch without branching:
#   carry = regs.F & FLAG_C   (either 0 or 1, since FLAG_C == 1)
#   regs.F = _ADD_PAIR[carry][(a << 8) | b]
# ===========================================================================


def _build_alu_flag_tables():
    """Build all 8-bit ALU flag tables.

    ADD/SUB/CP use *overflow* detection for the PV flag (sign change on
    arithmetic).  AND/OR/XOR use *parity* — those continue to use the
    existing SZ53P_TABLE / SZHZP_TABLE lookups.
    """
    add = bytearray(65536)
    adc = bytearray(65536)
    sub = bytearray(65536)
    sbc = bytearray(65536)
    inc = bytearray(256)
    dec = bytearray(256)

    for a in range(256):
        a8 = a << 8

        # INC: PV set only when old value is 0x7F (positive max → negative)
        nv = (a + 1) & 0xFF
        f = nv & (FLAG_S | _F53)
        if nv == 0:
            f |= FLAG_Z
        if (a & 0x0F) == 0x0F:
            f |= FLAG_H
        if a == 0x7F:
            f |= FLAG_PV  # overflow, not parity
        inc[a] = f

        # DEC: PV set only when old value is 0x80 (negative min → positive)
        nv = (a - 1) & 0xFF
        f = FLAG_N | (nv & (FLAG_S | _F53))
        if nv == 0:
            f |= FLAG_Z
        if (a & 0x0F) == 0x00:
            f |= FLAG_H
        if a == 0x80:
            f |= FLAG_PV  # overflow
        dec[a] = f

        for b in range(256):
            idx = a8 | b

            # ADD (carry=0): PV when both operands same sign but result differs
            r = a + b
            r8 = r & 0xFF
            f = r8 & (FLAG_S | _F53)
            if r8 == 0:
                f |= FLAG_Z
            if ((a & 0x0F) + (b & 0x0F)) & 0x10:
                f |= FLAG_H
            if r > 0xFF:
                f |= FLAG_C
            if ((a ^ b) & 0x80) == 0 and ((r8 ^ a) & 0x80) != 0:
                f |= FLAG_PV
            add[idx] = f

            # ADC (carry=1)
            r = a + b + 1
            r8 = r & 0xFF
            f = r8 & (FLAG_S | _F53)
            if r8 == 0:
                f |= FLAG_Z
            if ((a & 0x0F) + (b & 0x0F) + 1) & 0x10:
                f |= FLAG_H
            if r > 0xFF:
                f |= FLAG_C
            if ((a ^ b) & 0x80) == 0 and ((r8 ^ a) & 0x80) != 0:
                f |= FLAG_PV
            adc[idx] = f

            # SUB (carry=0): PV when operands differ in sign and result differs from a
            r = a - b
            r8 = r & 0xFF
            f = FLAG_N | (r8 & (FLAG_S | _F53))
            if r8 == 0:
                f |= FLAG_Z
            if (a & 0x0F) < (b & 0x0F):
                f |= FLAG_H
            if r < 0:
                f |= FLAG_C
            if ((a ^ b) & 0x80) != 0 and ((r8 ^ a) & 0x80) != 0:
                f |= FLAG_PV
            sub[idx] = f

            # SBC (carry=1)
            r = a - b - 1
            r8 = r & 0xFF
            f = FLAG_N | (r8 & (FLAG_S | _F53))
            if r8 == 0:
                f |= FLAG_Z
            if (a & 0x0F) < ((b & 0x0F) + 1):
                f |= FLAG_H
            if r < 0:
                f |= FLAG_C
            if ((a ^ b) & 0x80) != 0 and ((r8 ^ a) & 0x80) != 0:
                f |= FLAG_PV
            sbc[idx] = f

    return add, adc, sub, sbc, inc, dec


(ADD_FLAGS, ADC_FLAGS, SUB_FLAGS, SBC_FLAGS, INC_FLAGS, DEC_FLAGS_TBL) = (
    _build_alu_flag_tables()
)


class _CPFlags(bytearray):
    """CP flags proxy — derives from SUB_FLAGS at lookup time.

    CP differs from SUB only in F3/F5 sourcing: operand b instead of result.
    This avoids a separate 64KB table by patching SUB_FLAGS on the fly.
    """

    __slots__ = ()

    def __getitem__(self, idx):
        return (SUB_FLAGS[idx] & ~_F53) | (idx & _F53)


CP_FLAGS = _CPFlags(65536)

# Pair tables for carry-indexed dispatch (FLAG_C == 1, so carry is 0 or 1)
_ADD_PAIR = (ADD_FLAGS, ADC_FLAGS)  # _ADD_PAIR[carry][(a<<8)|b]
_SUB_PAIR = (SUB_FLAGS, SBC_FLAGS)  # _SUB_PAIR[carry][(a<<8)|b]


# ===========================================================================
# Numba JIT functions  (single definition each — no duplicate compilations)
# ===========================================================================

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _add16_flags_jit(hl: int, reg: int, current_f: int) -> int:
        """ADD HL,ss: preserves S/Z/PV; N cleared; H/C/F3/F5 updated."""
        full = hl + reg
        r16 = full & 0xFFFF
        flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)
        if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF:
            flags |= FLAG_H
        if full > 0xFFFF:
            flags |= FLAG_C
        flags |= (r16 >> 8) & _F53
        return flags

    @njit(cache=True)
    def _adc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
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
        flags |= (result >> 8) & _F53
        return flags

    @njit(cache=True)
    def _sbc16_flags_jit(hl: int, reg: int, carry: int, result: int) -> int:
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
        flags |= (result >> 8) & _F53
        return flags


# ===========================================================================
# Pure-Python fallback functions
# ===========================================================================


def _add16_flags_python(hl: int, reg: int, current_f: int) -> int:
    """ADD HL,ss: S/Z/PV preserved; N cleared; H/C/F3/F5 updated."""
    full = hl + reg
    r16 = full & 0xFFFF
    flags = current_f & (FLAG_S | FLAG_Z | FLAG_PV)
    if ((hl & 0x0FFF) + (reg & 0x0FFF)) > 0x0FFF:
        flags |= FLAG_H
    if full > 0xFFFF:
        flags |= FLAG_C
    flags |= (r16 >> 8) & _F53
    return flags


def _adc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
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
    flags |= (result >> 8) & _F53
    return flags


def _sbc16_flags_python(hl: int, reg: int, carry: int, result: int) -> int:
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
    flags |= (result >> 8) & _F53
    return flags


# ===========================================================================
# Backend selection — only 16-bit flag helpers are used externally
# ===========================================================================

if NUMBA_AVAILABLE:
    _bf_add16 = _add16_flags_jit
    _bf_adc16 = _adc16_flags_jit
    _bf_sbc16 = _sbc16_flags_jit
else:
    _bf_add16 = _add16_flags_python
    _bf_adc16 = _adc16_flags_python
    _bf_sbc16 = _sbc16_flags_python

add16_flags = _bf_add16


def get_adc16_flags(hl: int, reg: int, carry: int) -> int:
    result = (hl + reg + carry) & 0xFFFF
    return _bf_adc16(hl, reg, carry, result)


def get_sbc16_flags(hl: int, reg: int, carry: int) -> int:
    result = (hl - reg - carry) & 0xFFFF
    return _bf_sbc16(hl, reg, carry, result)


# ===========================================================================
# DAA — Decimal Adjust Accumulator
# ===========================================================================


def get_daa_result(a: int, f: int) -> tuple:
    """Apply DAA correction; return (new_a, new_flags)."""
    n = (f >> 1) & 1
    h = (f >> 4) & 1
    c = f & 1
    idx = (n << 10) | (h << 9) | (c << 8) | a
    return DAA_FULL_FLAGS[idx]


# ===========================================================================
# Condition table  (indexed by (flags_byte << 3) | condition_code)
# ===========================================================================


def _build_cond_table():
    table = [False] * 2048
    for f in range(256):
        base = f << 3
        table[base + 0] = not (f & FLAG_Z)  # NZ
        table[base + 1] = bool(f & FLAG_Z)  # Z
        table[base + 2] = not (f & FLAG_C)  # NC
        table[base + 3] = bool(f & FLAG_C)  # C
        table[base + 4] = not (f & FLAG_PV)  # PO
        table[base + 5] = bool(f & FLAG_PV)  # PE
        table[base + 6] = not (f & FLAG_S)  # P (positive)
        table[base + 7] = bool(f & FLAG_S)  # M (minus)
    return table


COND_TABLE = _build_cond_table()


# ===========================================================================
# DAA — Decimal Adjust Accumulator lookup tables
# index = (N << 10) | (H << 9) | (C << 8) | A  = 2048 entries
# Sequential correction: low nibble first, then high nibble on corrected value
#
# DAA_TABLE[index]      = (corrected_A, c_flag, h_flag)  — raw correction
# DAA_FULL_FLAGS[index] = (corrected_A, full_flags_byte) — ready to use
# ===========================================================================


def _build_daa_tables():
    raw = [(0, 0, 0)] * 2048
    full = [(0, 0)] * 2048

    for n in range(2):
        for h in range(2):
            for c in range(2):
                base_idx = (n << 10) | (h << 9) | (c << 8)
                input_f = (n << 1) | c  # N and C from original flags
                for orig_a in range(256):
                    a = orig_a
                    new_c = c
                    new_h = 0

                    if not n:  # after addition
                        if h or (a & 0x0F) > 9:
                            a = a + 0x06
                            new_h = 1
                        if c or a > 0x9F:
                            a = (a + 0x60) & 0xFF
                            new_c = 1
                        else:
                            a = a & 0xFF
                    else:  # after subtraction
                        if h:
                            a = (a - 0x06) & 0xFF
                            new_h = 1
                        if c:
                            a = (a - 0x60) & 0xFF
                            new_c = 1

                    idx = base_idx | orig_a
                    raw[idx] = (a, new_c, new_h)

                    # Build full flags byte
                    flags = input_f & FLAG_N  # preserve N
                    if new_c:
                        flags |= FLAG_C
                    if new_h:
                        flags |= FLAG_H
                    if a == 0:
                        flags |= FLAG_Z
                    if a & 0x80:
                        flags |= FLAG_S
                    flags |= a & _F53
                    if PARITY_TABLE[a]:
                        flags |= FLAG_PV
                    full[idx] = (a, flags)

    return raw, full


DAA_TABLE, DAA_FULL_FLAGS = _build_daa_tables()
