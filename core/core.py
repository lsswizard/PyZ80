"""
Z80 CPU Core Module
High-performance, machine-independent Z80 CPU emulator.
"""

from typing import Optional, Tuple, Protocol, Dict, Any
import array

from .decoder import InstructionDecoder
from .state import CPUState
from .tables import (
    FLAG_PV,
)

# =============================================================================
# Bus Interface
# =============================================================================


class Z80Bus(Protocol):
    """
    Protocol defining the interface between the Z80 CPU and the host machine.
    Implement this to provide memory and I/O access.
    """

    def bus_read(self, addr: int, t_state: int) -> int:
        """Read a byte from memory at the given address and T-state."""
        ...

    def bus_write(self, addr: int, value: int, t_state: int) -> None:
        """Write a byte to memory at the given address and T-state."""
        ...

    def bus_io_read(self, port: int, t_state: int) -> int:
        """Read a byte from an I/O port at the given address and T-state."""
        ...

    def bus_io_write(self, port: int, value: int, t_state: int) -> None:
        """Write a byte to an I/O port at the given address and T-state."""
        ...

    def __getitem__(self, addr: int) -> int: ...
    def __setitem__(self, addr: int, value: int) -> None: ...


class SimpleBus:
    """
    A basic implementation of Z80Bus for standalone CPU testing.
    """

    def __init__(self, memory: Optional[bytearray] = None):
        self.memory = memory if memory is not None else bytearray(65536)
        self.io_ports = bytearray([0xFF] * 256)

    def bus_read(self, addr: int, t_state: int) -> int:
        return self.memory[addr & 0xFFFF]

    def bus_write(self, addr: int, value: int, t_state: int) -> None:
        self.memory[addr & 0xFFFF] = value & 0xFF

    def bus_io_read(self, port: int, t_state: int) -> int:
        return self.io_ports[port & 0xFF]

    def bus_io_write(self, port: int, value: int, t_state: int) -> None:
        self.io_ports[port & 0xFF] = value & 0xFF

    def __getitem__(self, addr: int) -> int:
        return self.bus_read(addr, 0)

    def __setitem__(self, addr: int, value: int) -> None:
        self.bus_write(addr, value, 0)


# =============================================================================
# Registers Class
# =============================================================================


class Registers:
    """
    Z80 Register Set with fast access patterns.
    """

    __slots__ = (
        "A", "F", "B", "C", "D", "E", "H", "L",
        "Ap", "Fp", "Bp", "Cp", "Dp", "Ep", "Hp", "Lp",
        "IX", "IY", "SP", "PC", "I", "R",
        "IFF1", "IFF2", "IM", "EI_PENDING", "EI_JUST_RESOLVED",
        "UnresolvedPrefix", "Memptr",
    )

    def __init__(self):
        self.reset()

    @property
    def BC(self) -> int: return (self.B << 8) | self.C
    @BC.setter
    def BC(self, value: int) -> None:
        self.B = (value >> 8) & 0xFF
        self.C = value & 0xFF

    @property
    def DE(self) -> int: return (self.D << 8) | self.E
    @DE.setter
    def DE(self, value: int) -> None:
        self.D = (value >> 8) & 0xFF
        self.E = value & 0xFF

    @property
    def HL(self) -> int: return (self.H << 8) | self.L
    @HL.setter
    def HL(self, value: int) -> None:
        self.H = (value >> 8) & 0xFF
        self.L = value & 0xFF

    @property
    def AF(self) -> int: return (self.A << 8) | self.F
    @AF.setter
    def AF(self, value: int) -> None:
        self.A = (value >> 8) & 0xFF
        self.F = value & 0xFF

    @property
    def IXh(self) -> int: return (self.IX >> 8) & 0xFF
    @IXh.setter
    def IXh(self, value: int) -> None:
        self.IX = (self.IX & 0x00FF) | ((value & 0xFF) << 8)

    @property
    def IXl(self) -> int: return self.IX & 0xFF
    @IXl.setter
    def IXl(self, value: int) -> None:
        self.IX = (self.IX & 0xFF00) | (value & 0xFF)

    @property
    def IYh(self) -> int: return (self.IY >> 8) & 0xFF
    @IYh.setter
    def IYh(self, value: int) -> None:
        self.IY = (self.IY & 0x00FF) | ((value & 0xFF) << 8)

    @property
    def IYl(self) -> int: return self.IY & 0xFF
    @IYl.setter
    def IYl(self, value: int) -> None:
        self.IY = (self.IY & 0xFF00) | (value & 0xFF)

    def swap_shadow(self) -> None:
        self.A, self.Ap = self.Ap, self.A
        self.F, self.Fp = self.Fp, self.F

    def swap_shadow_all(self) -> None:
        self.B, self.Bp = self.Bp, self.B
        self.C, self.Cp = self.Cp, self.C
        self.D, self.Dp = self.Dp, self.D
        self.E, self.Ep = self.Ep, self.E
        self.H, self.Hp = self.Hp, self.H
        self.L, self.Lp = self.Lp, self.L

    def reset(self) -> None:
        self.A = 0xFF
        self.F = 0xFF
        self.B = self.C = 0
        self.D = self.E = 0
        self.H = self.L = 0
        self.Ap = self.Fp = self.Bp = self.Cp = 0
        self.Dp = self.Ep = self.Hp = self.Lp = 0
        self.IX = 0xFFFF
        self.IY = 0xFFFF
        self.SP = 0xFFFF
        self.PC = 0
        self.I = 0x00
        self.R = 0
        self.IFF1 = self.IFF2 = False
        self.IM = 0
        self.EI_PENDING = False
        self.EI_JUST_RESOLVED = False
        self.UnresolvedPrefix = False
        self.Memptr = 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "A": self.A, "F": self.F, "B": self.B, "C": self.C,
            "D": self.D, "E": self.E, "H": self.H, "L": self.L,
            "Ap": self.Ap, "Fp": self.Fp, "Bp": self.Bp, "Cp": self.Cp,
            "Dp": self.Dp, "Ep": self.Ep, "Hp": self.Hp, "Lp": self.Lp,
            "IX": self.IX, "IY": self.IY, "SP": self.SP, "PC": self.PC,
            "I": self.I, "R": self.R, "IFF1": self.IFF1, "IFF2": self.IFF2,
            "IM": self.IM, "EI_PENDING": self.EI_PENDING, "Memptr": self.Memptr,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_reg16(self, pair: int) -> int:
        if pair == 0: return self.BC
        if pair == 1: return self.DE
        if pair == 2: return self.HL
        return self.SP

    def set_reg16(self, pair: int, value: int) -> None:
        value &= 0xFFFF
        if pair == 0: self.BC = value
        elif pair == 1: self.DE = value
        elif pair == 2: self.HL = value
        else: self.SP = value

    def get_reg16_push(self, pair: int) -> int:
        if pair == 0: return self.BC
        if pair == 1: return self.DE
        if pair == 2: return self.HL
        return self.AF

    def set_reg16_push(self, pair: int, value: int) -> None:
        value &= 0xFFFF
        if pair == 0: self.BC = value
        elif pair == 1: self.DE = value
        elif pair == 2: self.HL = value
        else: self.AF = value


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

_COND_TABLE = _build_cond_table()

# =============================================================================
# Z80 CPU Class
# =============================================================================


class Z80CPU:
    """
    Machine-independent Z80 CPU core.
    Interacts with its environment strictly through the Z80Bus interface.
    """

    def __init__(self, bus: Optional[Z80Bus] = None):
        self.regs = Registers()
        self.bus = bus or SimpleBus()
        self.decoder = InstructionDecoder()

        # Cache frequently used attributes locally for performance
        self._bus_read = self.bus.bus_read
        self._bus_write = self.bus.bus_write
        self._bus_io_read = self.bus.bus_io_read
        self._bus_io_write = self.bus.bus_io_write
        self._decode = self.decoder.decode
        
        # Optimized memory view for decoder (if bus provides one)
        self._mem = getattr(self.bus, "memory", None)
        if self._mem is None:
            # Fallback to a proxy if bus is completely opaque
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
        self._pc_modified = False
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

    def fetch_byte(self) -> int:
        """Fetch opcode byte and increment PC."""
        pc = self.regs.PC
        val = self._bus_read(pc, self.cycles)
        self.regs.PC = (pc + 1) & 0xFFFF
        return val

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def check_condition(self, condition: int) -> bool:
        """Check if condition CC is met."""
        return _COND_TABLE[self.regs.F & 0xFF][condition & 0x07]

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

    def _handle_nmi(self) -> int:
        self.nmi_pending = False
        regs = self.regs
        regs.IFF2 = regs.IFF1
        regs.IFF1 = False
        self.halted = False
        
        # Push PC to stack via bus
        cycles = self.cycles
        sp = (regs.SP - 2) & 0xFFFF
        pc = regs.PC
        self._bus_write(sp, pc & 0xFF, cycles)
        self._bus_write((sp + 1) & 0xFFFF, (pc >> 8) & 0xFF, cycles)
        
        regs.SP = sp
        regs.PC = 0x0066
        self._pc_modified = True
        return 11

    def _handle_maskable_interrupt(self) -> int:
        self.interrupt_pending = False
        regs = self.regs
        regs.IFF2 = regs.IFF1
        regs.IFF1 = False
        if self.halted:
            self.halted = False
            regs.PC = (regs.PC + 1) & 0xFFFF

        # Push PC to stack
        cycles = self.cycles
        sp = (regs.SP - 2) & 0xFFFF
        pc = regs.PC
        self._bus_write(sp, pc & 0xFF, cycles)
        self._bus_write((sp + 1) & 0xFFFF, (pc >> 8) & 0xFF, cycles)
        regs.SP = sp

        if regs.IM <= 1:
            regs.PC = 0x0038
            self._pc_modified = True
            return 13
        else:
            vector = (regs.I << 8) | self.interrupt_data
            low = self._bus_read(vector, cycles)
            high = self._bus_read((vector + 1) & 0xFFFF, cycles)
            regs.PC = low | (high << 8)
            self._pc_modified = True
            return 19

    def step(self) -> int:
        """Execute one instruction."""
        if self.bus_request:
            self.cycles += 1
            return 1

        regs = self.regs
        pc = regs.PC
        bus_read = self._bus_read
        cycles = self.cycles

        # Fetch first byte
        opcode = bus_read(pc, cycles)

        # Fast-path: no interrupt pending or deferred
        if not (self.interrupt_pending or self.nmi_pending or regs.EI_PENDING or regs.EI_JUST_RESOLVED):
            if self.halted:
                self._increment_r(1)
                self.cycles += 4
                return 4
        else:
            return self._step_interrupt(opcode, pc)

        # Dispatch pre-decoded opcode
        op = self._decode(self._mem, pc)

        if self._trace_enabled:
            self._write_trace(pc, opcode, op)

        self._pc_modified = False
        is_prefixed = opcode in (0xCB, 0xDD, 0xFD, 0xED)
        self._increment_r(2 if is_prefixed else 1)

        # Call the handler directly, avoiding MicroOp.__call__ overhead
        op_cycles = op.handler(self)

        if not self._pc_modified:
            regs.PC = (pc + op.length) & 0xFFFF

        self.cycles += op_cycles
        self.instruction_count += 1
        regs.UnresolvedPrefix = (op.length == 1 and is_prefixed)

        return op_cycles

    def _step_interrupt(self, opcode: int, pc: int) -> int:
        """Handle interrupts and deferrals (slow-path)."""
        regs = self.regs
        
        # Detect LD A,I or LD A,R for interrupt bug simulation
        self._is_ld_a_ir = (opcode == 0xED and self._bus_read((pc + 1) & 0xFFFF, self.cycles) in (0x57, 0x5F))

        interrupt_cycles = 0
        if regs.EI_PENDING and not regs.UnresolvedPrefix:
            regs.EI_PENDING = False
            regs.EI_JUST_RESOLVED = True
            if not regs.IFF1:
                regs.IFF1 = regs.IFF2 = True
            else:
                interrupt_cycles = self.handle_interrupt()
        elif regs.EI_JUST_RESOLVED:
            regs.EI_JUST_RESOLVED = False
            interrupt_cycles = self.handle_interrupt()
        else:
            interrupt_cycles = self.handle_interrupt()

        if interrupt_cycles:
            if self._is_ld_a_ir:
                regs.F &= ~FLAG_PV
            self.cycles += interrupt_cycles
            return interrupt_cycles

        # If we got here, no interrupt was actually taken, so continue with normal step
        if self.halted:
            self._increment_r(1)
            self.cycles += 4
            return 4
            
        op = self._decode(self._mem, pc)
        self._pc_modified = False
        is_prefixed = opcode in (0xCB, 0xDD, 0xFD, 0xED)
        self._increment_r(2 if is_prefixed else 1)
        op_cycles = op.handler(self)
        if not self._pc_modified:
            regs.PC = (pc + op.length) & 0xFFFF
        self.cycles += op_cycles
        self.instruction_count += 1
        regs.UnresolvedPrefix = (op.length == 1 and is_prefixed)
        return op_cycles

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
        if reg == 0: return regs.B
        if reg == 1: return regs.C
        if reg == 2: return regs.D
        if reg == 3: return regs.E
        if reg == 4: return regs.H
        if reg == 5: return regs.L
        if reg == 6: return self._bus_read(regs.HL, self.cycles)
        return regs.A

    def set_reg8(self, reg: int, value: int) -> None:
        """Set 8-bit register by index."""
        regs = self.regs
        value &= 0xFF
        if reg == 0: regs.B = value
        elif reg == 1: regs.C = value
        elif reg == 2: regs.D = value
        elif reg == 3: regs.E = value
        elif reg == 4: regs.H = value
        elif reg == 5: regs.L = value
        elif reg == 6: self._bus_write(regs.HL, value, self.cycles)
        else: regs.A = value

    def get_state(self) -> CPUState:
        """Return snapshot of current state."""
        s = self.regs.get_state()
        return CPUState(
            **s,
            halted=self.halted,
            cycles=self.cycles,
            instruction_count=self.instruction_count,
        )

    def set_state(self, state: CPUState) -> None:
        """Restore state from snapshot."""
        self.regs.set_state(state.__dict__)
        self.halted = state.halted
        self.cycles = state.cycles
        self.instruction_count = state.instruction_count
        self.decoder.invalidate_cache()

    def _write_trace(self, pc, opcode, op):
        if not self._trace_file: return
        regs = self.regs
        self._trace_file.write(
            f"{pc:04X} {opcode:02X} {op.mnemonic:<16} "
            f"A:{regs.A:02X} F:{regs.F:02X} BC:{regs.BC:04X} "
            f"DE:{regs.DE:04X} HL:{regs.HL:04X} IX:{regs.IX:04X} "
            f"IY:{regs.IY:04X} SP:{regs.SP:04X} CYC:{self.cycles}\n"
        )

    def enable_trace(self, filename: str = "cpu_trace.log") -> None:
        """Start logging CPU state to file."""
        if self._trace_file: self.disable_trace()
        self._trace_file = open(filename, "w")
        self._trace_enabled = True

    def disable_trace(self) -> None:
        """Stop logging CPU state."""
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
        self._trace_enabled = False
