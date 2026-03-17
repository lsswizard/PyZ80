"""
Z80 Instruction Decoder Module

Pre-decodes Z80 opcodes into MicroOps for fast execution.  Results are cached
by address so that a given instruction is decoded at most once per memory state.

Opcode table layout:
    Base  0x00-0xFF  Standard instructions
    CB    bit-manipulation instructions
    ED    extended instructions
    DD    IX-register instructions
    FD    IY-register instructions
    DDCB  IX-indexed bit instructions  (4 bytes: DD CB d op)
    FDCB  IY-indexed bit instructions  (4 bytes: FD CB d op)

Cache invalidation:
    The cache is keyed by the 16-bit PC address of the first opcode byte.
    Any bus_write to RAM must call invalidate_cache(addr) so that stale
    MicroOps are not re-executed.  For banked memory, call invalidate_range()
    across the entire bank window after every bank switch.
"""

from typing import Optional
from .primitives import MicroOp, read_byte
from .instructions import (
    get_base_opcode,
    get_cb_opcode,
    get_ed_opcode,
    get_dd_opcode,
    get_fd_opcode,
    get_ddcb_opcode,
    get_fdcb_opcode,
    nop,
)


class InstructionDecoder:
    """
    Pre-decoding instruction cache.

    Cache design:
      - Flat list of 65 536 slots (one per possible 16-bit address).
      - O(1) lookup and O(1) invalidation.
      - No eviction policy — assumes a flat or paged 64 KB memory model.
        For banked systems call invalidate_range() on every bank switch.
    """

    MAX_CACHE_SIZE = 65536

    def __init__(self):
        self.cache: list = [None] * 65536

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------

    def decode_at(self, memory, addr: int) -> MicroOp:
        """Decode instruction at *addr* without consulting the cache."""
        opcode = read_byte(memory, addr)

        if opcode == 0xCB:
            cb_op = read_byte(memory, addr + 1)
            entry = get_cb_opcode(cb_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 8, 2, f"NOP* (CB {cb_op:02X})")

        elif opcode == 0xED:
            ed_op = read_byte(memory, addr + 1)
            entry = get_ed_opcode(ed_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 8, 2, f"NOP* (ED {ed_op:02X})")

        elif opcode == 0xDD:
            dd_op = read_byte(memory, addr + 1)
            if dd_op == 0xCB:
                # DDCB: DD CB d op  (4 bytes total)
                # The displacement is read at execution time by the handler
                # via _get_indexed_addr(); we must NOT capture it here or
                # the bit-number field of the opcode gets clobbered.
                d      = read_byte(memory, addr + 2)
                if d >= 128:
                    d -= 256
                cb_op  = read_byte(memory, addr + 3)
                entry  = get_ddcb_opcode(cb_op)
                if entry:
                    handler, cycles, _, mnemonic = entry
                    return MicroOp(lambda cpu, h=handler: h(cpu), cycles, 4, mnemonic)
                return MicroOp(nop, 23, 4, "NOP* (DDCB)")
            entry = get_dd_opcode(dd_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            # Unknown DD prefix: fall through to base opcode (undocumented behaviour)
            entry = get_base_opcode(dd_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles + 4, 2, f"(DD) {mnemonic}")
            return MicroOp(nop, 4, 2, f"NOP* (DD {dd_op:02X})")

        elif opcode == 0xFD:
            fd_op = read_byte(memory, addr + 1)
            if fd_op == 0xCB:
                # FDCB: FD CB d op  (4 bytes total)
                d      = read_byte(memory, addr + 2)
                if d >= 128:
                    d -= 256
                cb_op  = read_byte(memory, addr + 3)
                entry  = get_fdcb_opcode(cb_op)
                if entry:
                    handler, cycles, _, mnemonic = entry
                    return MicroOp(lambda cpu, h=handler: h(cpu), cycles, 4, mnemonic)
                return MicroOp(nop, 23, 4, "NOP* (FDCB)")
            entry = get_fd_opcode(fd_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            # Unknown FD prefix: fall through to base opcode
            entry = get_base_opcode(fd_op)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles + 4, 2, f"(FD) {mnemonic}")
            return MicroOp(nop, 4, 2, f"NOP* (FD {fd_op:02X})")

        else:
            entry = get_base_opcode(opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 4, 1, f"NOP* ({opcode:02X})")

    def decode(self, memory, addr: int) -> MicroOp:
        """Return a (possibly cached) MicroOp for the instruction at *addr*."""
        op = self.cache[addr]
        if op is not None:
            return op
        op = self.decode_at(memory, addr)
        self.cache[addr] = op
        return op

    # -------------------------------------------------------------------------
    # Cache invalidation
    # -------------------------------------------------------------------------

    def invalidate_cache(self, addr: Optional[int] = None) -> None:
        """Invalidate one address (± 3 bytes for multi-byte instructions) or the
        entire cache when *addr* is None.

        Call after every memory write so that modified code is re-decoded
        on the next fetch.  For bank switches use invalidate_range().
        """
        if addr is None:
            for i in range(65536):
                self.cache[i] = None
        else:
            # An instruction starting up to 3 bytes before *addr* could
            # span the written location, so flush the surrounding window.
            addr &= 0xFFFF
            for i in range(4):
                self.cache[(addr - i) & 0xFFFF] = None

    def invalidate_range(self, start: int, end: int) -> None:
        """Invalidate all cached entries covering addresses in [start, end).

        An instruction whose first byte is up to 3 bytes *before* the range
        boundary could straddle it, so the flush window is extended by 4 on
        the left.

        This must be called for *every* bank switch, regardless of range size.
        The previous optimisation that silently skipped ranges ≤ 4 096 bytes
        has been removed — it was a correctness bug for small bank windows
        such as the 8 KB ROM/RAM pages used by many Z80 systems.
        """
        flush_start = max(0,     start - 4)
        flush_end   = min(65536, end)
        range_size  = flush_end - flush_start

        if range_size <= 0:
            return

        if range_size >= 65536:
            # Whole address space — fastest path
            for i in range(65536):
                self.cache[i] = None
        else:
            self.cache[flush_start:flush_end] = [None] * range_size

    def cache_stats(self) -> dict:
        filled = sum(1 for x in self.cache if x is not None)
        return {"size": filled, "capacity": 65536}
