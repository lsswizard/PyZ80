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
"""

from .cpu import Z80CPU
from .registers import Registers
from .bus import Z80Bus, SimpleBus
from .state import CPUState, StateManager
from .decoder import InstructionDecoder
from .primitives import MicroOp
from .timing import TimingInfo, TimingEngine

__all__ = [
    "Z80CPU",
    "Registers",
    "CPUState",
    "StateManager",
    "InstructionDecoder",
    "MicroOp",
    "Z80Bus",
    "SimpleBus",
    "TimingInfo",
    "TimingEngine",
]
