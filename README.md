# PyZ80 - Cycle-Exact Z80 Emulator

A high-performance, cycle-exact Z80 CPU emulator written in Python.

## Features

- **Cycle-Exact Timing**: Accurate T-state tracking between M-cycles
- **Complete Instruction Set**: All 256 base opcodes + CB/ED/DD/FD prefixes
- **Cycle-Accurate Interrupts**: NMI and maskable interrupt handling
- **Pre-decoded Cache**: Optimized instruction execution with 64KB decode cache
- **Clean Architecture**: Machine-independent design via Z80Bus protocol

## Installation

```bash
pip install pyz80
```

## Quick Start

```python
from core import Z80CPU, SimpleBus

# Create CPU with 64KB RAM
bus = SimpleBus()
cpu = Z80CPU(bus)

# Load a simple program (Z80 NOP instruction at address 0)
cpu.bus[0] = 0x00  # NOP

# Execute one instruction
cycles = cpu.step()
print(f"Executed NOP in {cycles} cycles")

# Execute many instructions
cpu.execute(10000)  # Run for 10000 cycles
```

## Architecture

### Z80Bus Protocol

The CPU is completely isolated from memory via the `Z80Bus` protocol:

```python
class MyMachine(Z80Bus):
    def bus_read(self, addr: int, t_state: int) -> int:
        # Implement memory read with cycle-exact timing
        ...
    
    def bus_write(self, addr: int, value: int, t_state: int) -> None:
        # Implement memory write with cycle-exact timing
        ...
```

The `t_state` parameter allows implementing wait states, memory contention, or hardware synchronization.

### Cycle Counting

The emulator tracks T-states at every M-cycle boundary. After each instruction:

```python
cpu.step()
print(f"Total cycles: {cpu.cycles}")  # Accurate T-state count
```

### Interrupts

```python
# Trigger maskable interrupt
cpu.trigger_interrupt(data=0xFF)  # Data on data bus for IM 0

# Trigger non-maskable interrupt  
cpu.trigger_nmi()

# Check interrupt state
if cpu.interrupt_pending:
    cpu.handle_interrupt()
```

## Testing

Run the validation suite:

```bash
pytest tests/
```

Run specific timing tests:

```bash
pytest tests/test_validate_z80.py::TestTiming -v
```

## Performance Notes

- Pre-decoded instruction cache eliminates decode overhead
- Local attribute caching in hot paths
- Minimal object allocation during execution

## References

- [Z80 CPU User Manual](http://www.z80.info/zip/z80cpu_user_manual.pdf)
- [Z80 Instruction Set](http://www.z80.info/z80code.htm)
