"""
Z80 Core Benchmark
Measures emulated instructions per second and compares to real Z80 speed.
"""

import time
import sys

sys.path.insert(0, ".")

from core.cpu import Z80CPU
from core.bus import SimpleBus
from core.decoder import InstructionDecoder


def create_benchmark_program(memory: bytearray) -> None:
    """Write a benchmark program that exercises various instructions."""

    program = [
        0x06,
        0xFF,  # LD B, 255 (7 T)
        0x0E,
        0x00,  # LD C, 0 (7 T)
        0x0D,  # DEC C (4 T)
        0x3E,
        0x00,  # LD A, 0 (7 T)
        0x80,  # ADD A, B (4 T)
        0x05,  # DEC B (4 T)
        0xC2,
        0x04,
        0x00,  # JP NZ, 4 (10 T)
        0x76,  # HALT (4 T)
    ]

    for i, op in enumerate(program):
        memory[i] = op


def run_benchmark(cycles: int = 100000, verbose: bool = True) -> dict:
    """Run benchmark and return results."""

    bus = SimpleBus()
    create_benchmark_program(bus.memory)

    cpu = Z80CPU(bus)
    cpu.reset()
    cpu.regs.PC = 0

    start_time = time.perf_counter()

    while cpu.cycles < cycles:
        cpu.step()

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    instructions = cpu.instruction_count
    ips = instructions / elapsed
    mhz = (ips * 4) / 1_000_000

    results = {
        "cycles": cycles,
        "instructions": instructions,
        "elapsed_seconds": elapsed,
        "instructions_per_second": ips,
        "effective_mhz": mhz,
    }

    if verbose:
        print(f"Benchmark Results:")
        print(f"  Target cycles:     {cycles:,}")
        print(f"  Instructions:     {instructions:,}")
        print(f"  Elapsed time:     {elapsed:.3f} seconds")
        print(f"  Instructions/sec: {ips:,.0f}")
        print(f"  Effective MHz:   {mhz:.2f} MHz (Z80 @ 3.5MHz = ~875K instr/sec)")
        print(f"  Real-time factor: {ips / 875000:.1f}x real Z80 @ 3.5MHz")

    return results


def run_extended_benchmark() -> None:
    """Run more comprehensive benchmark with different instruction mixes."""

    print("=" * 60)
    print("Z80 Core Benchmark - Extended")
    print("=" * 60)

    # Benchmark 1: Basic arithmetic loop
    print("\n[1] Arithmetic Loop (DEC/JNZ)")
    program1 = [
        0x06,
        0x64,  # LD B, 100
        0x0D,  # DEC C
        0xC2,
        0x02,
        0x00,  # JP NZ, 2
    ]
    bus = SimpleBus()
    for i, op in enumerate(program1):
        bus.memory[i] = op

    cpu = Z80CPU(bus)
    cpu.reset()

    start = time.perf_counter()
    while cpu.cycles < 50_000:
        cpu.step()
    elapsed = time.perf_counter() - start
    print(f"    {cpu.instruction_count / elapsed:,.0f} instr/sec")

    # Benchmark 2: Memory operations
    print("\n[2] Memory Operations (LD (HL),A / LD A,(HL))")
    program2 = [
        0x3E,
        0x55,  # LD A, 0x55
        0x77,  # LD (HL), A
        0x7E,  # LD A, (HL)
        0x18,
        0xF9,  # JR -7
    ]
    bus = SimpleBus()
    for i, op in enumerate(program2):
        bus.memory[i] = op

    cpu = Z80CPU(bus)
    cpu.reset()

    start = time.perf_counter()
    while cpu.cycles < 50_000:
        cpu.step()
    elapsed = time.perf_counter() - start
    print(f"    {cpu.instruction_count / elapsed:,.0f} instr/sec")

    # Benchmark 3: Block transfers
    print("\n[3] Block Transfer (LDI)")
    program3 = [
        0xED,
        0xA0,  # LDI
        0x18,
        0xFC,  # JR -4
    ]
    bus = SimpleBus()
    for i, op in enumerate(program3):
        bus.memory[i] = op

    cpu = Z80CPU(bus)
    cpu.reset()

    start = time.perf_counter()
    while cpu.cycles < 50_000:
        cpu.step()
    elapsed = time.perf_counter() - start
    print(f"    {cpu.instruction_count / elapsed:,.0f} instr/sec")

    # Benchmark 4: Indexed operations (IX/IY)
    print("\n[4] Indexed Operations (ADD A,(IX+d))")
    program4 = [
        0xDD,
        0x86,
        0x00,  # ADD A, (IX+0)
        0x18,
        0xFA,  # JR -6
    ]
    bus = SimpleBus()
    for i, op in enumerate(program4):
        bus.memory[i] = op

    cpu = Z80CPU(bus)
    cpu.reset()
    cpu.regs.IX = 0x1000

    start = time.perf_counter()
    while cpu.cycles < 50_000:
        cpu.step()
    elapsed = time.perf_counter() - start
    print(f"    {cpu.instruction_count / elapsed:,.0f} instr/sec")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Z80 Core Benchmark")
    print("=" * 60)

    print("\nRunning quick benchmark...")
    run_benchmark(cycles=100_000)

    run_extended_benchmark()
