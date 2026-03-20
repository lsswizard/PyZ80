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

_PREFIXED_OPCODES = frozenset((0xCB, 0xDD, 0xFD, 0xED))


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

    def __init__(self, bus: Optional[Z80Bus] = None):
        """Initialize CPU with given bus interface."""
        self.regs = Registers()
        self.bus = bus or SimpleBus()
        self.decoder = InstructionDecoder()

        # Cache frequently used attributes locally for performance
        self._bus_read = self.bus.bus_read
        self._bus_write = self._internal_write
        self._bus_io_read = self.bus.bus_io_read
        self._bus_io_write = self.bus.bus_io_write
        self._decode = self.decoder.decode
        self._cache_list = self.decoder.cache

        # Optimized memory view for decoder (if bus provides one)
        self._mem = getattr(self.bus, "memory", None)
        if self._mem is None:
            self._mem = self.bus

        self.halted = False
        self.cycles = 0
        self.instruction_count = 0

        self.interrupt_pending = False
        self.interrupt_data = 0xFF
        self.nmi_pending = False
        self.bus_request = False

        self._pc_modified = False
        self._is_ld_a_ir = False

        # Tracing
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
        self._bus_write(addr, value, self.cycles)
        self.decoder.invalidate_cache(addr)

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
        self._bus_write(addr, value & 0xFF, base_t)
        self._bus_write((addr + 1) & 0xFFFF, value >> 8, base_t + 1)
        self.decoder.invalidate_cache(addr)
        self.decoder.invalidate_cache((addr + 1) & 0xFFFF)

    def push16_at(self, value: int, base_t: int) -> int:
        """Push 16-bit value at explicit T-states. Returns new SP."""
        sp = (self.regs.SP - 2) & 0xFFFF
        self._bus_write(sp, value & 0xFF, base_t)
        self._bus_write((sp + 1) & 0xFFFF, value >> 8, base_t + 1)
        self.regs.SP = sp
        return sp

    def pop16_at(self, base_t: int) -> int:
        """Pop 16-bit value at explicit T-states. Returns value."""
        sp = self.regs.SP
        lo = self._bus_read(sp, base_t)
        hi = self._bus_read((sp + 1) & 0xFFFF, base_t + 1)
        self.regs.SP = (sp + 2) & 0xFFFF
        return hi << 8 | lo

    def _internal_write(self, addr: int, value: int, cycles: int) -> None:
        """Internal write wrapper that ensures the decoder cache is always invalidated."""
        self.bus.bus_write(addr, value, cycles)
        self.decoder.invalidate_cache(addr)

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

    def trigger_nmi(self) -> None:
        """Signal a non-maskable interrupt."""
        self.nmi_pending = True

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
        self._bus_write(sp, pc & 0xFF, cycles + 1)
        self._bus_write((sp + 1) & 0xFFFF, pc >> 8, cycles + 4)
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
        return 11

    def _handle_maskable_interrupt(self) -> int:
        self.interrupt_pending = False
        regs = self.regs
        regs.IFF1 = False
        regs.IFF2 = False
        self._push_pc_and_get_cycles()

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
            # IM 2: 7(push) + 2(wait) + 2(read low) + 2(read high) = 19 T-states
            # Vector low read at T10, high read at T13
            cycles = self.cycles
            vector = (regs.I << 8) | (self.interrupt_data & 0xFE)
            low = self._bus_read(vector, cycles + 10)
            high = self._bus_read((vector + 1) & 0xFFFF, cycles + 13)
            regs.PC = low | (high << 8)
            self._pc_modified = True
            self.cycles += 19
            return 19

    def _dispatch(self, opcode: int, pc: int) -> int:
        """Execute one instruction and return total T-states."""
        regs = self.regs
        cache = self._cache_list
        mem = self._mem

        op = cache[pc] or self._decode(mem, pc)

        if self._trace_enabled:
            self._write_trace(pc, opcode, op)

        self._pc_modified = False
        is_prefixed = opcode in _PREFIXED_OPCODES
        self._increment_r(2 if is_prefixed else 1)

        op_cycles = op.handler(self)

        if not self._pc_modified:
            regs.PC = (pc + op.length) & 0xFFFF

        self.cycles += op_cycles
        self.instruction_count += 1
        regs.UnresolvedPrefix = op.length == 1 and is_prefixed

        # Only check mnemonic for ED-prefixed instructions (rare)
        self._is_ld_a_ir = opcode == 0xED and op.mnemonic in ("LD A,I", "LD A,R")

        return op_cycles

    def step(self) -> int:
        """Execute one instruction."""
        if self.bus_request:
            self.cycles += 1
            return 1

        regs = self.regs
        pc = regs.PC
        cycles = self.cycles

        # Fetch first byte (at current T-state)
        opcode = self._bus_read(pc, cycles)

        # Advance T-state by M1 duration (4 cycles)
        self.cycles = cycles + 4

        # Fast path: no interrupt or EI deferral expected
        if not (
            self.interrupt_pending
            or self.nmi_pending
            or regs.EI_PENDING
            or regs.EI_JUST_RESOLVED
        ):
            if self.halted:
                self._increment_r(1)
                return 4
        else:
            # Interrupt or EI deferral — handle via slow path
            self.cycles -= 4
            result = self._step_interrupt()
            if result:
                return result
            # No interrupt serviced (e.g. EI deferral), fall through to dispatch

        op_cycles = self._dispatch(opcode, pc)
        self.cycles -= 4
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
        return 0

    def _increment_r(self, amount: int) -> None:
        """Increment the 7-bit portion of R."""
        regs = self.regs
        r = (regs.R + amount) & 0x7F
        regs.R = (regs.R & 0x80) | r

    def execute(self, target_cycles: int) -> int:
        """Execute until target cycles reached."""
        start_cycles = self.cycles
        while (self.cycles - start_cycles) < target_cycles:
            self.step()
        return self.cycles - start_cycles

    def get_reg8(self, reg: int) -> int:
        """Get 8-bit register by index (0-7)."""
        regs = self.regs
        if reg < 6:
            return (regs.B, regs.C, regs.D, regs.E, regs.H, regs.L)[reg]
        if reg == 6:
            return self._bus_read(regs.HL, self.cycles)
        return regs.A

    def set_reg8(self, reg: int, value: int) -> None:
        """Set 8-bit register by index."""
        regs = self.regs
        value &= 0xFF
        if reg == 0:
            regs.B = value
        elif reg == 1:
            regs.C = value
        elif reg == 2:
            regs.D = value
        elif reg == 3:
            regs.E = value
        elif reg == 4:
            regs.H = value
        elif reg == 5:
            regs.L = value
        elif reg == 6:
            self._bus_write(regs.HL, value, self.cycles)
        else:
            regs.A = value

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
