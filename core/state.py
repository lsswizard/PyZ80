"""
Z80 CPU State Module
Snapshot and restore CPU state for debugging, rewind, and testing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import copy


@dataclass
class CPUState:
    """
    Snapshot of CPU state for debugging/serialization.

    Contains all register values, flags, and internal state
    needed to completely restore CPU execution context.
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

    # Interrupt flip-flops
    IFF1: bool = False
    IFF2: bool = False
    IM: int = 0  # Interrupt mode 0, 1, or 2
    EI_PENDING: bool = False

    # Memory pointer for special flag operations
    Memptr: int = 0

    # CPU status
    halted: bool = False
    cycles: int = 0
    instruction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
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
            "halted": self.halted,
            "cycles": self.cycles,
            "instruction_count": self.instruction_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPUState":
        """Create state from dictionary"""
        return cls(**data)

    def copy(self) -> "CPUState":
        """Create a deep copy of this state"""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Human-readable state representation"""
        return (
            f"CPUState(A={self.A:02X} F={self.F:02X} "
            f"BC={self.B:02X}{self.C:02X} DE={self.D:02X}{self.E:02X} "
            f"HL={self.H:02X}{self.L:02X} IX={self.IX:04X} IY={self.IY:04X} "
            f"SP={self.SP:04X} PC={self.PC:04X} I={self.I:02X} R={self.R:02X} "
            f"IFF1={self.IFF1} IFF2={self.IFF2} IM={self.IM})"
        )

    def flags_str(self) -> str:
        """Return flags as string representation"""
        flags = []
        if self.F & 0x80:
            flags.append("S")
        if self.F & 0x40:
            flags.append("Z")
        if self.F & 0x20:
            flags.append("5")
        if self.F & 0x10:
            flags.append("H")
        if self.F & 0x08:
            flags.append("3")
        if self.F & 0x04:
            flags.append("P")
        if self.F & 0x02:
            flags.append("N")
        if self.F & 0x01:
            flags.append("C")
        return "".join(flags) if flags else "----"


class StateManager:
    """
    Manages CPU state history for debugging and rewind functionality.

    Features:
    - Record state snapshots at key points
    - Rewind to previous states
    - Compare states for debugging
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: list[CPUState] = []
        self.current_index: int = -1

    def record(self, state: CPUState) -> None:
        """Record a state snapshot"""
        # If we're not at the end, truncate forward history
        if self.current_index < len(self.history) - 1:
            self.history = self.history[: self.current_index + 1]

        self.history.append(state.copy())
        self.current_index = len(self.history) - 1

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1

    def can_rewind(self) -> bool:
        """Check if rewind is possible"""
        return self.current_index > 0

    def can_forward(self) -> bool:
        """Check if forward is possible"""
        return self.current_index < len(self.history) - 1

    def rewind(self) -> Optional[CPUState]:
        """Rewind to previous state"""
        if self.can_rewind():
            self.current_index -= 1
            return self.history[self.current_index].copy()
        return None

    def forward(self) -> Optional[CPUState]:
        """Go forward to next state"""
        if self.can_forward():
            self.current_index += 1
            return self.history[self.current_index].copy()
        return None

    def current(self) -> Optional[CPUState]:
        """Get current state"""
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index].copy()
        return None

    def clear(self) -> None:
        """Clear all history"""
        self.history.clear()
        self.current_index = -1

    def compare(self, state1: CPUState, state2: CPUState) -> Dict[str, tuple]:
        """Compare two states and return differences"""
        diffs = {}
        d1 = state1.to_dict()
        d2 = state2.to_dict()

        for key in d1:
            if d1[key] != d2[key]:
                diffs[key] = (d1[key], d2[key])

        return diffs
