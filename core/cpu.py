"""
Z80 CPU Core Module

This module provides a cycle-exact Z80 CPU emulator that is fully machine-independent.
The CPU communicates with the outside world exclusively through the Z80Bus interface,
making it suitable for emulating various Z80-based systems (ZX Spectrum, Amstrad CPC, etc.).

Features:
- Cycle-exact instruction timing with proper M-cycle sequencing
- Support for all documented and undocumented Z80 instructions
- Interrupt handling (NMI and maskable interrupts via INT pin)
- Optional instruction tracing for debugging
"""

from typing import Optional
import atexit

from .decoder import InstructionDecoder
from .state import CPUState
from .flags import FLAG_PV, COND_TABLE
from .registers import Registers
from .bus import Z80Bus, SimpleBus

_IS_PREFIXED = bytearray(256)
for _op in (0xCB, 0xDD, 0xFD, 0xED):
    _IS_PREFIXED[_op] = 1

# Opcodes that modify F but should NOT set Q (FUSE convention)
_IS_Q_EXCEPT = bytearray(256)
for _op in (0x08, 0xF1):  # EX AF,AF' and POP AF
    _IS_Q_EXCEPT[_op] = 1


def _gB(cpu, r):
    return r.B


def _gC(cpu, r):
    return r.C


def _gD(cpu, r):
    return r.D


def _gE(cpu, r):
    return r.E


def _gH(cpu, r):
    return r.H


def _gL(cpu, r):
    return r.L


def _gM(cpu, r):
    return cpu._bus_read(r.HL, cpu.cycles)


def _gA(cpu, r):
    return r.A


_GET_REG8 = (_gB, _gC, _gD, _gE, _gH, _gL, _gM, _gA)


def _sB(cpu, r, v):
    r.B = v


def _sC(cpu, r, v):
    r.C = v


def _sD(cpu, r, v):
    r.D = v


def _sE(cpu, r, v):
    r.E = v


def _sH(cpu, r, v):
    r.H = v


def _sL(cpu, r, v):
    r.L = v


def _sM(cpu, r, v):
    cpu._bus_write(r.HL, v, cpu.cycles)


def _sA(cpu, r, v):
    r.A = v


_SET_REG8 = (_sB, _sC, _sD, _sE, _sH, _sL, _sM, _sA)


class Z80CPU:
    """
    Z80 CPU emulator core.

    This class implements a cycle-accurate Z80 processor. All memory and I/O operations
    are delegated to a Z80Bus implementation, allowing the emulator to run on any
    platform with appropriate memory mapping.

    Attributes:
        regs: Register file (A, B, C, D, E, H, L, F, IX, IY, SP, PC, etc.)
        bus: Bus interface for memory/IO operations
        cycles: Total T-states executed since reset
        halted: Whether the CPU is in HALT state
    """

    __slots__ = (
        "regs",
        "bus",
        "decoder",
        "_bus_read",
        "_bus_write_direct",
        "_bus_write",
        "_bus_io_read",
        "_bus_io_write",
        "_decode",
        "_cache_list",
        "_mem",
        "_is_simple_bus",
        "halted",
        "cycles",
        "instruction_count",
        "interrupt_pending",
        "interrupt_data",
        "nmi_pending",
        "bus_request",
        "_needs_slow_step",
        "_pc_modified",
        "_is_ld_a_ir",
        "_trace_enabled",
        "_trace_file",
    )

    def __init__(self, bus: Optional[Z80Bus] = None):
        """Initialize CPU with given bus interface."""
        self.regs = Registers()
        self.bus = bus or SimpleBus()
        self.decoder = InstructionDecoder()

        self._bus_read = self.bus.bus_read
        self._bus_write_direct = self.bus.bus_write
        self._bus_write = self._cache_write
        self._bus_io_read = self.bus.bus_io_read
        self._bus_io_write = self.bus.bus_io_write
        self._decode = self.decoder.decode
        self._cache_list = self.decoder.cache

        self._mem = getattr(self.bus, "memory", None)
        if self._mem is None:
            self._mem = self.bus
        self._is_simple_bus = isinstance(self.bus, SimpleBus)

        self.halted = False
        self.cycles = 0
        self.instruction_count = 0

        self.interrupt_pending = False
        self.interrupt_data = 0xFF
        self.nmi_pending = False
        self.bus_request = False
        self._needs_slow_step = False

        self._pc_modified = False
        self._is_ld_a_ir = False

        self._trace_enabled = False
        self._trace_file = None

    def reset(self) -> None:
        """Reset CPU to initial power-on state."""
        self.regs.reset()
        self.halted = False
        self.cycles = 0
        self.instruction_count = 0
        self.interrupt_pending = False
        self.interrupt_data = 0xFF
        self.nmi_pending = False
        self.bus_request = False
        self._needs_slow_step = False
        self._pc_modified = False
        self._is_ld_a_ir = False
        self.decoder.invalidate_cache()

    # -------------------------------------------------------------------------
    # Bus Interaction
    # -------------------------------------------------------------------------

    def read_byte(self, addr: int) -> int:
        """Read byte from memory via bus."""
        return self._bus_read(addr, self.cycles)

    def write_byte(self, addr: int, value: int) -> None:
        """Write byte to memory via bus, and invalidate decoder cache."""
        self._cache_write(addr, value, self.cycles)

    def io_read(self, port: int) -> int:
        """Read from I/O port via bus."""
        return self._bus_io_read(port, self.cycles)

    def io_write(self, port: int, value: int) -> None:
        """Write to I/O port via bus."""
        self._bus_io_write(port, value, self.cycles)

    def read16_at(self, addr: int, base_t: int) -> int:
        """Read 16-bit value from memory at explicit T-states (addr, addr+1)."""
        lo = self._bus_read(addr, base_t)
        hi = self._bus_read((addr + 1) & 0xFFFF, base_t + 1)
        return hi << 8 | lo

    def write16_at(self, addr: int, value: int, base_t: int) -> None:
        """Write 16-bit value to memory at explicit T-states."""
        self._cache_write(addr, value & 0xFF, base_t)
        self._cache_write((addr + 1) & 0xFFFF, value >> 8, base_t + 1)

    def _cache_write(self, addr: int, value: int, cycles: int) -> None:
        """Internal write wrapper that invalidates decoder cache on change."""
        addr &= 0xFFFF
        if self._mem[addr] != value:
            self._bus_write_direct(addr, value, cycles)
            cache = self._cache_list
            cache[addr] = None
            cache[(addr - 1) & 0xFFFF] = None
            cache[(addr - 2) & 0xFFFF] = None
            cache[(addr - 3) & 0xFFFF] = None
        else:
            self._bus_write_direct(addr, value, cycles)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def check_condition(self, condition: int) -> bool:
        """Check if condition CC is met."""
        return COND_TABLE[(self.regs.F << 3) | (condition & 0x07)]

    def trigger_interrupt(self, data: int = 0xFF) -> None:
        """Signal a maskable interrupt."""
        self.interrupt_pending = True
        self.interrupt_data = data
        self._needs_slow_step = True

    def trigger_nmi(self) -> None:
        """Signal a non-maskable interrupt."""
        self.nmi_pending = True
        self._needs_slow_step = True

    def handle_interrupt(self) -> int:
        """Handle pending interrupts, returning cycles consumed."""
        if self.nmi_pending:
            return self._handle_nmi()
        if self.interrupt_pending and self.regs.IFF1 and not self.regs.UnresolvedPrefix:
            return self._handle_maskable_interrupt()
        return 0

    def _push_pc_and_get_cycles(self) -> None:
        """Push PC to stack."""
        regs = self.regs
        if self.halted:
            self.halted = False
            regs.PC = (regs.PC + 1) & 0xFFFF
        cycles = self.cycles
        sp = (regs.SP - 2) & 0xFFFF
        pc = regs.PC
        # Direct call to avoid double invalidation check (stack is rarely code)
        self._bus_write_direct(sp, pc & 0xFF, cycles + 1)
        self._bus_write_direct((sp + 1) & 0xFFFF, pc >> 8, cycles + 4)
        regs.SP = sp

    def _handle_nmi(self) -> int:
        self.nmi_pending = False
        regs = self.regs
        regs.IFF2 = regs.IFF1
        regs.IFF1 = False
        self._push_pc_and_get_cycles()
        self.cycles += 11
        regs.PC = 0x0066
        self._pc_modified = True
        self._needs_slow_step = False
        return 11

    def _handle_maskable_interrupt(self) -> int:
        self.interrupt_pending = False
        regs = self.regs
        regs.IFF1 = False
        regs.IFF2 = False
        self._push_pc_and_get_cycles()
        self._needs_slow_step = False

        if regs.IM == 0:
            opcode = self.interrupt_data
            if (opcode & 0xC7) == 0xC7:
                regs.PC = opcode & 0x38
            else:
                regs.PC = 0x0038
            self._pc_modified = True
            self.cycles += 13
            return 13
        elif regs.IM == 1:
            regs.PC = 0x0038
            self._pc_modified = True
            self.cycles += 13
            return 13
        else:
            cycles = self.cycles
            vector = (regs.I << 8) | (self.interrupt_data & 0xFE)
            low = self._bus_read(vector, cycles + 10)
            high = self._bus_read((vector + 1) & 0xFFFF, cycles + 13)
            regs.PC = low | (high << 8)
            self._pc_modified = True
            self.cycles += 19
            return 19

    def step(self) -> int:
        """Execute one instruction with optimized attribute access."""
        if self.bus_request:
            self.cycles += 1
            return 1

        regs = self.regs
        cycles = self.cycles

        # --- Interrupt / HALT fast check ---
        if self._needs_slow_step:
            result = self._step_interrupt()
            if result:
                # _step_interrupt already updated self.cycles via handle_interrupt
                return result
        elif self.halted:
            r = regs.R
            regs.R = (r & 0x80) | ((r + 1) & 0x7F)
            self.cycles = cycles + 4
            return 4

        pc = regs.PC
        mem = self._mem
        cache_list = self._cache_list

        # --- Opcode fetch: _mem_fast for SimpleBus, bus_read otherwise ---
        if self._is_simple_bus:
            opcode = mem[pc]
        else:
            opcode = self._bus_read(pc, self.cycles)

        # --- Decode (cache hit is the common case) ---
        # Accessing cache_list directly avoids method call overhead
        op = cache_list[pc]
        if op is None:
            op = self._decode(mem, pc)

        if self._trace_enabled:
            self._write_trace(pc, opcode, op)

        # --- R register refresh ---
        self._pc_modified = False
        r = regs.R
        regs.R = (r & 0x80) | ((r + 1 + _IS_PREFIXED[opcode]) & 0x7F)

        # --- Q factor tracking for SCF/CCF (Patrik Rak discovery) ---
        regs.last_Q = regs.Q
        regs.Q = 0  # Preemptive clear
        old_f = regs.F

        # --- Execute instruction ---
        # M1 T1-T4 are always executed. Handler returns TOTAL base cycles.
        self.cycles += 4
        op_cycles = op.handler(self)

        # Set Q = F if this instruction modified flags
        # (EX AF,AF' and POP AF are exceptions that don't set Q)
        if op.affects_f:
            new_f = regs.F
            if new_f != old_f and not _IS_Q_EXCEPT[opcode]:
                regs.Q = new_f

        if not self._pc_modified:
            regs.PC = (pc + op.length) & 0xFFFF

        # Final cycles update - base instruction timing only (no contention)
        total_cycles = cycles + op_cycles
        self.cycles = total_cycles
        self.instruction_count += 1

        self._is_ld_a_ir = op.is_ld_a_ir
        if self.interrupt_pending:
            regs.UnresolvedPrefix = op.length == 1 and _IS_PREFIXED[opcode]

        return op_cycles

    def _step_interrupt(self) -> int:
        """Service pending interrupt or EI deferral. Returns cycles consumed, or 0 if none."""
        regs = self.regs
        if regs.EI_PENDING:
            regs.EI_PENDING = False
            regs.EI_JUST_RESOLVED = True
            regs.IFF1 = regs.IFF2 = True
        elif regs.EI_JUST_RESOLVED:
            regs.EI_JUST_RESOLVED = False
        cycles = self.handle_interrupt()
        if cycles:
            if self._is_ld_a_ir:
                regs.F &= ~FLAG_PV
            return cycles
        self._needs_slow_step = (
            self.interrupt_pending
            or self.nmi_pending
            or regs.EI_PENDING
            or regs.EI_JUST_RESOLVED
        )
        return 0

    def execute(self, target_cycles: int) -> int:
        """Execute until target cycles reached."""
        start_cycles = self.cycles
        while (self.cycles - start_cycles) < target_cycles:
            self.step()
        return self.cycles - start_cycles

    def get_reg8(self, reg: int) -> int:
        """Get 8-bit register by index (0-7). Single table lookup, no if-chain."""
        return _GET_REG8[reg](self, self.regs)

    def set_reg8(self, reg: int, value: int) -> None:
        """Set 8-bit register by index. Single table lookup, no if-chain."""
        _SET_REG8[reg](self, self.regs, value & 0xFF)

    def get_state(self) -> CPUState:
        """Return snapshot of current state."""
        s = self.regs.get_state()
        return CPUState(
            **s,
            halted=self.halted,
            cycles=self.cycles,
            instruction_count=self.instruction_count,
            interrupt_pending=self.interrupt_pending,
            interrupt_data=self.interrupt_data,
            nmi_pending=self.nmi_pending,
            bus_request=self.bus_request,
        )

    def set_state(self, state: CPUState) -> None:
        """Restore state from snapshot."""
        self.regs.set_state(state.to_dict())
        self.halted = state.halted
        self.cycles = state.cycles
        self.instruction_count = state.instruction_count
        self.interrupt_pending = state.interrupt_pending
        self.interrupt_data = state.interrupt_data
        self.nmi_pending = state.nmi_pending
        self.bus_request = state.bus_request
        self.decoder.invalidate_cache()

    def _write_trace(self, pc, opcode, op):
        if not self._trace_file:
            return
        regs = self.regs
        self._trace_file.write(
            f"{pc:04X} {opcode:02X} {op.mnemonic:<16} "
            f"A:{regs.A:02X} F:{regs.F:02X} BC:{regs.BC:04X} "
            f"DE:{regs.DE:04X} HL:{regs.HL:04X} IX:{regs.IX:04X} "
            f"IY:{regs.IY:04X} SP:{regs.SP:04X} CYC:{self.cycles}\n"
        )

    def enable_trace(self, filename: str = "cpu_trace.log") -> None:
        """Start logging CPU state to file."""
        self.disable_trace()
        self._trace_file = open(filename, "w")
        self._trace_enabled = True
        atexit.register(self.disable_trace)

    def disable_trace(self) -> None:
        """Stop logging CPU state."""
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
        self._trace_enabled = False
