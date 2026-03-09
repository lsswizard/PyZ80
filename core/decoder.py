"""
Z80 Instruction Decoder Module
Pre-decodes instructions into MicroOps for fast execution.

Note on DDCB/FDCB instructions:
    These indexed bit operations (DDCB/FDCB prefixes) require special handling
    because they operate on (IX+d)/(IY+d) instead of (HL), and they have
    an additional displacement byte. The CB opcode's low 3 bits determine
    whether the result is also stored in a register (undocumented Z80 behavior).

    Format: DD CB d cb  (4 bytes)
            FD CB d cb  (4 bytes)

    Where:
    - d is the signed displacement byte
    - cb is the CB opcode determining the operation and target register
"""

from typing import Optional
from .pipeline import MicroOp, read_byte
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
    Pre-decoding instruction decoder.
    Converts opcodes into MicroOps for fast execution.

    Cache Behavior:
        - Unbounded cache (max 65536 entries for 64KB address space)
        - No eviction policy - assumes flat memory model
        - For banked memory systems, call invalidate_cache() on bank switches
    """

    # Maximum cache size (full 64KB address space)
    MAX_CACHE_SIZE = 65536

    def __init__(self):
        # OPT5: Pre-allocated list for O(1) integer-indexed cache lookup
        self.cache: list = [None] * 65536

    def decode_at(self, memory, addr: int) -> MicroOp:
        """Decode instruction at address, return MicroOp."""
        opcode = read_byte(memory, addr)

        if opcode == 0xCB:
            cb_opcode = read_byte(memory, addr + 1)
            entry = get_cb_opcode(cb_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 8, 2, f"NOP* (CB {cb_opcode:02X})")

        elif opcode == 0xED:
            ed_opcode = read_byte(memory, addr + 1)
            entry = get_ed_opcode(ed_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 8, 2, f"NOP* (ED {ed_opcode:02X})")

        elif opcode == 0xDD:
            dd_opcode = read_byte(memory, addr + 1)
            if dd_opcode == 0xCB:
                # DDCB indexed bit operations (4 bytes): DD CB d cb
                # d = signed displacement at addr+2
                # cb = CB opcode at addr+3
                displacement = read_byte(memory, addr + 2)
                # Sign-extend displacement to signed byte (-128 to 127)
                if displacement >= 128:
                    displacement -= 256
                cb_opcode = read_byte(memory, addr + 3)
                entry = get_ddcb_opcode(cb_opcode)
                if entry:
                    handler, cycles, _, mnemonic = entry
                    # FIX Bug4: do NOT pass displacement as positional arg.
                    # Handlers call _get_indexed_addr() which reads cpu.regs.PC
                    # at execution time; passing d here clobbered the bit number.
                    return MicroOp(lambda cpu, h=handler: h(cpu), cycles, 4, mnemonic)
                return MicroOp(nop, 23, 4, "NOP* (DDCB)")
            entry = get_dd_opcode(dd_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            # FIX Bug8: Unknown DD prefix falls through to base opcode
            entry = get_base_opcode(dd_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                # Use DD prefix cycles (4 extra) and length 2
                return MicroOp(handler, cycles + 4, 2, f"(DD) {mnemonic}")
            return MicroOp(nop, 4, 2, f"NOP* (DD {dd_opcode:02X})")

        elif opcode == 0xFD:
            fd_opcode = read_byte(memory, addr + 1)
            if fd_opcode == 0xCB:
                # FDCB indexed bit operations (4 bytes): FD CB d cb
                # d = signed displacement at addr+2
                # cb = CB opcode at addr+3
                displacement = read_byte(memory, addr + 2)
                # Sign-extend displacement to signed byte (-128 to 127)
                if displacement >= 128:
                    displacement -= 256
                cb_opcode = read_byte(memory, addr + 3)
                entry = get_fdcb_opcode(cb_opcode)
                if entry:
                    handler, cycles, _, mnemonic = entry
                    # FIX Bug4: do NOT pass displacement as positional arg.
                    # Handlers call _get_indexed_addr() which reads cpu.regs.PC
                    # at execution time; passing d here clobbered the bit number.
                    return MicroOp(lambda cpu, h=handler: h(cpu), cycles, 4, mnemonic)
                return MicroOp(nop, 23, 4, "NOP* (FDCB)")
            entry = get_fd_opcode(fd_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            # FIX Bug8: Unknown FD prefix falls through to base opcode
            entry = get_base_opcode(fd_opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                # Use FD prefix cycles (4 extra) and length 2
                return MicroOp(handler, cycles + 4, 2, f"(FD) {mnemonic}")
            return MicroOp(nop, 4, 2, f"NOP* (FD {fd_opcode:02X})")

        else:
            entry = get_base_opcode(opcode)
            if entry:
                handler, cycles, length, mnemonic = entry
                return MicroOp(handler, cycles, length, mnemonic)
            return MicroOp(nop, 4, 1, f"NOP* ({opcode:02X})")

    def decode(self, memory, addr: int) -> MicroOp:
        """Decode with caching (OPT5: direct list index, no hash overhead)."""
        op = self.cache[addr]
        if op is not None:
            return op
        op = self.decode_at(memory, addr)
        self.cache[addr] = op
        return op

    def invalidate_cache(self, addr: Optional[int] = None) -> None:
        """Invalidate cache entry or entire cache.

        Args:
            addr: Specific address to invalidate, or None to clear all.

        Note:
            For banked memory systems, call invalidate_cache() without
            arguments when switching memory banks to prevent stale cache
            entries from causing misexecution.
        """
        if addr is None:
            # OPT5: reset list in-place (faster than building new list)
            for i in range(65536):
                self.cache[i] = None
        else:
            # FIX Bug5+OPT5: invalidate multi-byte instruction range
            for a in range(max(0, addr - 3), addr + 1):
                self.cache[a] = None

    def invalidate_range(self, start: int, end: int) -> None:
        """Invalidate all cached entries in [start, end).

        Called by MemoryPager after a bank switch to flush only the
        address range whose physical contents changed, rather than
        nuking the entire 64 KB cache every time.

        An instruction starting up to 3 bytes before ``start`` could
        straddle the bank boundary, so we extend the flush by 4 bytes
        on the left.
        """
        flush_start = max(0, start - 4)
        flush_end = min(65536, end)
        self.cache[flush_start:flush_end] = [None] * (flush_end - flush_start)

    def cache_stats(self) -> dict:
        """Get cache statistics (OPT5: counts non-None slots)."""
        filled = sum(1 for x in self.cache if x is not None)
        return {"size": filled, "capacity": 65536}
