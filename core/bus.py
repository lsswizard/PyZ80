"""
Z80 Bus Interface Module
"""

from typing import Optional, Protocol

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
