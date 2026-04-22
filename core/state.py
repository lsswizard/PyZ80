"""
Z80 CPU State Module
Snapshot and restore CPU state for debugging, rewind, and testing.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CPUState:
    """
    Complete snapshot of CPU state for save/restore, debugging, and serialisation.

    Captures all register values, interrupt state, and execution counters
    needed to fully restore a CPU execution context.
    """

    # Main registers
    A: int = 0
    F: int = 0
    B: int = 0
    C: int = 0
    D: int = 0
    E: int = 0
    H: int = 0
    L: int = 0

    # Shadow registers
    Ap: int = 0
    Fp: int = 0
    Bp: int = 0
    Cp: int = 0
    Dp: int = 0
    Ep: int = 0
    Hp: int = 0
    Lp: int = 0

    # Index registers
    IX: int = 0
    IY: int = 0

    # Stack pointer and program counter
    SP: int = 0xFFFF
    PC: int = 0

    # Interrupt and refresh registers
    I: int = 0  # noqa: E741
    R: int = 0

    # Interrupt flip-flops and mode
    IFF1: bool = False
    IFF2: bool = False
    IM: int = 0

    # EI-related deferred-interrupt bookkeeping
    EI_PENDING: bool = False
    EI_JUST_RESOLVED: bool = False

    # Memory pointer (used by several flag-affecting instructions)
    Memptr: int = 0

    # Q factor tracking (Patrik Rak discovery for SCF/CCF undocumented behavior)
    Q: int = 0
    last_Q: int = 0

    # CPU execution status
    halted: bool = False
    cycles: int = 0
    instruction_count: int = 0

    # Interrupt / bus signals (needed for full round-trip snapshots)
    interrupt_pending: bool = False
    interrupt_data: int = 0xFF
    nmi_pending: bool = False
    bus_request: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialisation."""
        return {
            "A": self.A,
            "F": self.F,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "E": self.E,
            "H": self.H,
            "L": self.L,
            "Ap": self.Ap,
            "Fp": self.Fp,
            "Bp": self.Bp,
            "Cp": self.Cp,
            "Dp": self.Dp,
            "Ep": self.Ep,
            "Hp": self.Hp,
            "Lp": self.Lp,
            "IX": self.IX,
            "IY": self.IY,
            "SP": self.SP,
            "PC": self.PC,
            "I": self.I,
            "R": self.R,
            "IFF1": self.IFF1,
            "IFF2": self.IFF2,
            "IM": self.IM,
            "EI_PENDING": self.EI_PENDING,
            "EI_JUST_RESOLVED": self.EI_JUST_RESOLVED,
            "Memptr": self.Memptr,
            "Q": self.Q,
            "last_Q": self.last_Q,
            "halted": self.halted,
            "cycles": self.cycles,
            "instruction_count": self.instruction_count,
            "interrupt_pending": self.interrupt_pending,
            "interrupt_data": self.interrupt_data,
            "nmi_pending": self.nmi_pending,
            "bus_request": self.bus_request,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPUState":
        """Create state from dictionary, ignoring unknown keys for forward-compat."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:
        return (
            f"CPUState(A={self.A:02X} F={self.F:02X} "
            f"BC={self.B:02X}{self.C:02X} DE={self.D:02X}{self.E:02X} "
            f"HL={self.H:02X}{self.L:02X} IX={self.IX:04X} IY={self.IY:04X} "
            f"SP={self.SP:04X} PC={self.PC:04X} I={self.I:02X} R={self.R:02X} "
            f"IFF1={self.IFF1} IFF2={self.IFF2} IM={self.IM})"
        )

    def flags_str(self) -> str:
        """Return active flag names as a compact string."""
        parts = []
        if self.F & 0x80:
            parts.append("S")
        if self.F & 0x40:
            parts.append("Z")
        if self.F & 0x20:
            parts.append("5")
        if self.F & 0x10:
            parts.append("H")
        if self.F & 0x08:
            parts.append("3")
        if self.F & 0x04:
            parts.append("P")
        if self.F & 0x02:
            parts.append("N")
        if self.F & 0x01:
            parts.append("C")
        return "".join(parts) if parts else "----"
