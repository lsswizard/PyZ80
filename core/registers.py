"""
Z80 Register Set Module

This module defines the Z80 register file, including both primary and shadow
register sets, as well as special function registers.

Z80 Register Overview:
    - Primary: A, B, C, D, E, H, L, F (flag register)
    - Shadow: A', B', C', D', E', H', L', F' (accessed via EX AF,AF' and EXX)
    - Index: IX, IY (16-bit index registers)
    - Stack: SP (stack pointer), PC (program counter)
    - Special: I (interrupt vector), R (refresh counter)
    - Control: IFF1, IFF2 (interrupt flip-flops), IM (interrupt mode)
"""

from typing import Dict, Any


class Registers:
    """
    Z80 register file with optimized property access.

    Uses __slots__ for memory efficiency and faster attribute access.
    Provides property-based access for 16-bit register combinations.
    """

    __slots__ = (
        "A",
        "F",
        "B",
        "C",
        "D",
        "E",
        "H",
        "L",
        "Ap",
        "Fp",
        "Bp",
        "Cp",
        "Dp",
        "Ep",
        "Hp",
        "Lp",
        "IX",
        "IY",
        "SP",
        "PC",
        "I",
        "R",
        "IFF1",
        "IFF2",
        "IM",
        "EI_PENDING",
        "EI_JUST_RESOLVED",
        "UnresolvedPrefix",
        "Memptr",
        "Q",
        "last_Q",
    )

    def __init__(self):
        self.reset()

    @property
    def BC(self) -> int:
        return (self.B << 8) | self.C

    @BC.setter
    def BC(self, value: int) -> None:
        self.B = (value >> 8) & 0xFF
        self.C = value & 0xFF

    @property
    def DE(self) -> int:
        return (self.D << 8) | self.E

    @DE.setter
    def DE(self, value: int) -> None:
        self.D = (value >> 8) & 0xFF
        self.E = value & 0xFF

    @property
    def HL(self) -> int:
        return (self.H << 8) | self.L

    @HL.setter
    def HL(self, value: int) -> None:
        self.H = (value >> 8) & 0xFF
        self.L = value & 0xFF

    @property
    def AF(self) -> int:
        return (self.A << 8) | self.F

    @AF.setter
    def AF(self, value: int) -> None:
        self.A = (value >> 8) & 0xFF
        self.F = value & 0xFF

    @property
    def IXh(self) -> int:
        return (self.IX >> 8) & 0xFF

    @IXh.setter
    def IXh(self, value: int) -> None:
        self.IX = (self.IX & 0x00FF) | ((value & 0xFF) << 8)

    @property
    def IXl(self) -> int:
        return self.IX & 0xFF

    @IXl.setter
    def IXl(self, value: int) -> None:
        self.IX = (self.IX & 0xFF00) | (value & 0xFF)

    @property
    def IYh(self) -> int:
        return (self.IY >> 8) & 0xFF

    @IYh.setter
    def IYh(self, value: int) -> None:
        self.IY = (self.IY & 0x00FF) | ((value & 0xFF) << 8)

    @property
    def IYl(self) -> int:
        return self.IY & 0xFF

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
        self.Q = 0
        self.last_Q = 0

    def get_state(self) -> Dict[str, Any]:
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
            "Memptr": self.Memptr,
        }

    _SLOT_SET = frozenset(__slots__)

    def set_state(self, state: Dict[str, Any]) -> None:
        slots = self._SLOT_SET
        for key, value in state.items():
            if key in slots:
                setattr(self, key, value)

    def get_reg16(self, pair: int) -> int:
        if pair == 0:
            return (self.B << 8) | self.C
        elif pair == 1:
            return (self.D << 8) | self.E
        elif pair == 2:
            return (self.H << 8) | self.L
        else:
            return self.SP

    def set_reg16(self, pair: int, value: int) -> None:
        # Inline to avoid lambda + setattr + property lookup chain.
        # Benchmarked ~28% faster than the lambda/setattr approach.
        v = value & 0xFFFF
        if pair == 0:
            self.B = v >> 8
            self.C = v & 0xFF
        elif pair == 1:
            self.D = v >> 8
            self.E = v & 0xFF
        elif pair == 2:
            self.H = v >> 8
            self.L = v & 0xFF
        else:
            self.SP = v

    def get_reg16_push(self, pair: int) -> int:
        if pair == 0:
            return (self.B << 8) | self.C
        elif pair == 1:
            return (self.D << 8) | self.E
        elif pair == 2:
            return (self.H << 8) | self.L
        else:
            return (self.A << 8) | self.F

    def set_reg16_push(self, pair: int, value: int) -> None:
        # Inline to avoid lambda + setattr + property lookup chain.
        v = value & 0xFFFF
        if pair == 0:
            self.B = v >> 8
            self.C = v & 0xFF
        elif pair == 1:
            self.D = v >> 8
            self.E = v & 0xFF
        elif pair == 2:
            self.H = v >> 8
            self.L = v & 0xFF
        else:
            self.A = v >> 8
            self.F = v & 0xFF
