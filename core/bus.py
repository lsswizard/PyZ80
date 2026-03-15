"""
Z80 Bus Interface Module

This module defines the interface between the Z80 CPU and external memory/I/O.
The Z80Bus protocol must be implemented by any class providing memory access.

The t_state parameter allows for cycle-exact bus emulation, enabling accurate
simulation of memory-mapped hardware that responds to specific CPU timing.

Protocol Requirements:
    - bus_read(addr, t_state): Read byte from memory
    - bus_write(addr, value, t_state): Write byte to memory
    - bus_io_read(port, t_state): Read from I/O port
    - bus_io_write(port, value, t_state): Write to I/O port
"""

from typing import Optional, Protocol


class Z80Bus(Protocol):
    """
    Protocol defining the interface between the Z80 CPU and the host system.

    Implement this protocol to connect the CPU to actual hardware, memory maps,
    or other emulation layers (e.g., video, audio).

    The t_state parameter represents the CPU's current T-state (time unit).
    Implementations can use this for:
    - Wait state simulation
    - Memory contention timing
    - Hardware register synchronization
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
    A basic implementation of Z80Bus for testing and simple emulation.

    Provides a flat 64KB memory model with 256 I/O ports.
    Does not implement timing-aware behavior (t_state is ignored).
    """

    def __init__(self, memory: Optional[bytearray] = None):
        """
        Initialize SimpleBus with optional memory array.

        Args:
            memory: Pre-allocated bytearray, or None for 64KB zero-filled
        """
        self.memory = memory if memory is not None else bytearray(65536)
        self.io_ports = bytearray([0xFF] * 256)

    def bus_read(self, addr: int, t_state: int) -> int:
        """Read a byte from memory (timing parameter ignored)."""
        return self.memory[addr & 0xFFFF]

    def bus_write(self, addr: int, value: int, t_state: int) -> None:
        """Write a byte to memory (timing parameter ignored)."""
        self.memory[addr & 0xFFFF] = value & 0xFF

    def bus_io_read(self, port: int, t_state: int) -> int:
        """Read from I/O port (timing parameter ignored)."""
        return self.io_ports[port & 0xFF]

    def bus_io_write(self, port: int, value: int, t_state: int) -> None:
        """Write to I/O port (timing parameter ignored)."""
        self.io_ports[port & 0xFF] = value & 0xFF

    def __getitem__(self, addr: int) -> int:
        """Direct memory access for debugging."""
        return self.bus_read(addr, 0)

    def __setitem__(self, addr: int, value: int) -> None:
        """Direct memory write for debugging."""
        self.bus_write(addr, value, 0)
