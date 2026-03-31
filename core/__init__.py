"""
Z80 Core Module

This package provides a cycle-exact Z80 CPU emulator.

Exports:
    - Z80CPU: Main CPU class
    - Registers: Register file
    - InstructionDecoder: Opcode decoder with caching
    - MicroOp: Pre-decoded instruction
    - Z80Bus: Bus interface protocol
    - SimpleBus: Basic memory implementation
    - CPUState: CPU state snapshot
"""

from .cpu import Z80CPU
from .registers import Registers
from .bus import Z80Bus, SimpleBus
from .state import CPUState
from .decoder import InstructionDecoder
from .primitives import MicroOp

__all__ = [
    "Z80CPU",
    "Registers",
    "CPUState",
    "InstructionDecoder",
    "MicroOp",
    "Z80Bus",
    "SimpleBus",
]
