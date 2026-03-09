"""
Z80 Core Module
High-performance Z80 CPU emulation core.

Exports:
- Z80CPU: Main CPU class
- Registers: Register set class
- CPUState: State snapshot class
- StateManager: State history manager
- InstructionDecoder: Opcode decoder
- MicroOp: Pre-decoded instruction
- Z80Bus: Bus protocol interface
"""

from .core import Z80CPU, Registers, Z80Bus, SimpleBus
from .state import CPUState, StateManager
from .decoder import InstructionDecoder
from .pipeline import MicroOp, PrefixedMicroOp
from .pipeline import read_byte, read_word, write_byte, write_word, push_word, pop_word

__all__ = [
    "Z80CPU",
    "Registers",
    "CPUState",
    "StateManager",
    "InstructionDecoder",
    "MicroOp",
    "PrefixedMicroOp",
    "Z80Bus",
    "SimpleBus",
    "read_byte",
    "read_word",
    "write_byte",
    "write_word",
    "push_word",
    "pop_word",
]
