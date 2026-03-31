# PyZ80 C++ Core + Python Wrapper Plan

## Overview

Rewrite the `core/` emulation engine in pure C++ with a thin Python wrapper via nanobind.
The C++ library handles all hot-path execution. Python is used for test orchestration,
program loading, and result verification only.

## Why C++ over Cython

| Factor | Cython | C++ |
|--------|--------|-----|
| Per-instruction overhead | ~5-15ns (still some Python layer) | ~2-5ns (pure native) |
| Batch execution speed | ~50-100 MIPS | ~200-500 MIPS |
| Tooling (debugger, profiler) | Limited | Full (gdb, perf, valgrind) |
| Reusability | Python-only | C++ lib usable from any language |
| Build complexity | Simple | Moderate (CMake) |

## Expected Speedup

- `cpu.step()` from Python: **5-15x** (30ns binding overhead + 5ns C++ vs 100ns Python)
- `cpu.run(cycles)` batch: **50-100x** (no Python in hot loop, pure C++)
- Equivalent to ~200-500 MIPS vs current ~2 MIPS

---

## C++ Core Architecture

### Registers — Plain C Struct

```cpp
struct Registers {
    uint8_t  A, F, B, C, D, E, H, L;
    uint8_t  Ap, Fp, Bp, Cp, Dp, Ep, Hp, Lp;
    uint16_t IX, IY, SP, PC;
    uint8_t  I, R;
    bool     IFF1, IFF2;
    uint8_t  IM;
    bool     EI_PENDING, EI_JUST_RESOLVED, UNRESOLVED_PREFIX;
    uint16_t MEMPTR;
    uint8_t  Q, LAST_Q;

    uint16_t BC() const { return (B << 8) | C; }
    void set_BC(uint16_t v) { B = (v >> 8) & 0xFF; C = v & 0xFF; }
    uint16_t DE() const { return (D << 8) | E; }
    void set_DE(uint16_t v) { D = (v >> 8) & 0xFF; E = v & 0xFF; }
    uint16_t HL() const { return (H << 8) | L; }
    void set_HL(uint16_t v) { H = (v >> 8) & 0xFF; L = v & 0xFF; }
    uint16_t AF() const { return (A << 8) | F; }
    void set_AF(uint16_t v) { A = (v >> 8) & 0xFF; F = v & 0xFF; }

    uint8_t IXh() const { return (IX >> 8) & 0xFF; }
    void set_IXh(uint8_t v) { IX = (IX & 0x00FF) | (v << 8); }
    uint8_t IYh() const { return (IY >> 8) & 0xFF; }
    void set_IYh(uint8_t v) { IY = (IY & 0x00FF) | (v << 8); }
    uint8_t IXl() const { return IX & 0xFF; }
    void set_IXl(uint8_t v) { IX = (IX & 0xFF00) | v; }
    uint8_t IYl() const { return IY & 0xFF; }
    void set_IYl(uint8_t v) { IY = (IY & 0xFF00) | v; }

    void swap_shadow();
    void swap_shadow_all();
    void reset();
};
```

Total: ~38 bytes. Fits in a cache line. No GC pressure.

### Bus — Virtual Interface

```cpp
class Bus {
public:
    virtual ~Bus() = default;
    virtual uint8_t bus_read(uint16_t addr, int t_state) = 0;
    virtual void bus_write(uint16_t addr, uint8_t value, int t_state) = 0;
    virtual uint8_t bus_io_read(uint16_t port, int t_state) = 0;
    virtual void bus_io_write(uint16_t port, uint8_t value, int t_state) = 0;
};

class SimpleBus : public Bus {
public:
    uint8_t memory[65536];
    uint8_t io_ports[256];
    SimpleBus();
    // Virtual overrides: memory[addr & 0xFFFF], etc.
};
```

### Lookup Tables — Static Const Pre-computed

```cpp
namespace z80flags {
    extern const uint8_t PARITY_TABLE[256];
    extern const uint8_t SZ_TABLE[256];
    extern const uint8_t SZ53P_TABLE[256];
    extern const uint8_t SZHZP_TABLE[256];
    extern const uint8_t ADD_FLAGS[65536];   // 64KB
    extern const uint8_t ADC_FLAGS[65536];
    extern const uint8_t SUB_FLAGS[65536];
    extern const uint8_t SBC_FLAGS[65536];
    extern const uint8_t CP_FLAGS[65536];
    extern const uint8_t INC_FLAGS[256];
    extern const uint8_t DEC_FLAGS[256];
    extern const uint8_t ROT_RESULT[8][256]; // 2KB
    extern const uint8_t ROT_CARRY[8][256];
    extern const uint8_t RL_CARRY_0[256];
    extern const uint8_t RL_CARRY_1[256];
    extern const uint8_t RR_CARRY_0[256];
    extern const uint8_t RR_CARRY_1[256];
    extern const bool    COND_TABLE[2048];
    extern const uint8_t DAA_FULL_FLAGS[2048];

    uint8_t add16_flags(uint16_t hl, uint16_t reg, uint8_t current_f);
    uint8_t adc16_flags(uint16_t hl, uint16_t reg, uint8_t carry, uint8_t current_f);
    uint8_t sbc16_flags(uint16_t hl, uint16_t reg, uint8_t carry, uint8_t current_f);
    uint16_t daa_result(uint8_t a, uint8_t f);

    constexpr uint8_t FLAG_S=0x80, FLAG_Z=0x40, FLAG_F5=0x20;
    constexpr uint8_t FLAG_H=0x10, FLAG_F3=0x08, FLAG_PV=0x04;
    constexpr uint8_t FLAG_N=0x02, FLAG_C=0x01;
}
```

No Numba. No NumPy. ~320KB of lookup data, all `const uint8_t[]`.

### Handler Dispatch — C Function Pointer Tables

```cpp
typedef int (*OpHandler)(CPU& cpu);

struct OpcodeEntry {
    OpHandler handler;
    uint8_t cycles;
    uint8_t length;
    bool affects_f;
    bool is_ld_a_ir;
};

extern OpcodeEntry base_handlers[256];
extern OpcodeEntry cb_handlers[256];
extern OpcodeEntry ed_handlers[256];
extern OpcodeEntry dd_handlers[256];
extern OpcodeEntry fd_handlers[256];
extern OpcodeEntry ddcb_handlers[256];
extern OpcodeEntry fdcb_handlers[256];
```

Handlers decode params from `cpu.current_opcode`:
```cpp
int op_add_a_r(CPU& cpu) {
    uint8_t src = cpu.current_opcode & 0x07;
    uint8_t a = cpu.regs.A;
    uint8_t b = cpu.read_reg8(src);
    cpu.regs.A = (a + b) & 0xFF;
    cpu.regs.F = z80flags::ADD_FLAGS[(a << 8) | b];
    return 4;
}
```

### Decode Cache — Flat C Array

```cpp
struct DecodeSlot {
    OpHandler handler;
    uint8_t cycles;
    uint8_t length;
    bool affects_f;
    bool is_ld_a_ir;
};

class Decoder {
    DecodeSlot _cache[65536];
public:
    Decoder();
    const DecodeSlot& decode(uint8_t* mem, uint16_t addr);
};
```

### CPU — Main Execution Class

```cpp
class CPU {
public:
    Registers regs;
    Bus* bus;
    int cycles;
    int instruction_count;
    bool halted;
    uint8_t interrupt_data;
    bool interrupt_pending;
    bool nmi_pending;
    uint8_t current_opcode;
    bool _is_iy;

    uint8_t* _mem;          // direct pointer to SimpleBus::memory
    bool _is_simple_bus;

    CPU(Bus* bus = nullptr);
    void reset();
    int step();
    int run(int max_cycles);           // batch: KEY API for performance
    int run_instructions(int count);
    void trigger_interrupt(uint8_t data);
    void trigger_nmi();
    uint8_t read_reg8(int reg);
    void write_reg8(int reg, uint8_t value);
    Decoder decoder;
};
```

The critical `run()` method:
```cpp
int CPU::run(int max_cycles) {
    int start = cycles;
    while (cycles - start < max_cycles && !halted) {
        step();
    }
    return cycles - start;
}
```

`step()` — the tight inner loop:
```cpp
int CPU::step() {
    uint16_t pc = regs.PC;
    uint8_t opcode = _mem[pc];

    const DecodeSlot& slot = decoder.decode(_mem, pc);
    if (slot.handler == nullptr) {
        cycles += 4;
        regs.PC = (pc + 1) & 0xFFFF;
        return 4;
    }

    regs.R = (regs.R & 0x80) | ((regs.R + 1) & 0x7F);
    current_opcode = opcode;
    int t = slot.handler(*this);

    regs.LAST_Q = regs.Q;
    regs.Q = slot.affects_f ? regs.F : 0;

    if (!slot.is_ld_a_ir && !_pc_modified) {
        regs.PC = (pc + slot.length) & 0xFFFF;
    }
    _pc_modified = false;

    cycles += t;
    instruction_count++;
    return t;
}
```

---

## Instruction Handler Organization

### Handler Families (decode params from opcode)

| Family | Count | Param Source |
|--------|-------|-------------|
| LD r,r' | 1 | dest=(op>>3)&7, src=op&7 |
| LD r,n | 1 | dest=(op>>3)&7 |
| ADD/ADC/SUB/SBC/AND/OR/XOR/CP A,r | 8 | src=op&7 |
| INC r / DEC r | 2 | reg=(op>>3)&7 |
| RLC/RRC/RL/RR/SLA/SRA/SLL/SRL r | 8 | op_idx, reg (CB) |
| BIT/SET/RES n,r | 3 | bit=(op>>3)&7, reg=op&7 |
| PUSH/POP rr | 2 | pair=(op>>4)&3 |
| JP/CALL/RET/JR cc | 4 | cc=(op>>3)&7 |
| RST p | 1 | addr=((op>>3)&7)*8 |
| INC/DEC rr | 2 | pair=(op>>4)&3 |
| ADD HL,rr | 1 | pair=(op>>4)&3 |

### Dedicated Handlers (~50 unique)

NOP, HALT, DI, EI, RET, JP nn, JR e, DJNZ, CALL nn, EX DE,HL, EX AF,AF',
EXX, EX (SP),HL, DAA, CPL, SCF, CCF, NEG, IM 0/1/2, RLCA, RRCA, RLA, RRA,
JP (HL), LD SP,HL, LD A,I, LD A,R, RETI, RETN, LDI, LDD, LDIR, LDDR,
CPI, CPD, CPIR, CPDR, INI, IND, INIR, INDR, OUTI, OUTD, OTIR, OTDR,
RLD, RRD, etc.

Total unique handler functions: ~150-200.

---

## Python Wrapper via nanobind

```cpp
NB_MODULE(_pyz80, m) {
    nb::class_<Registers>(m, "Registers")
        .def_rw("A", &Registers::A)
        .def_rw("B", &Registers::B)
        // ... all 8-bit registers ...
        .def_prop_rw("BC", &Registers::BC, &Registers::set_BC)
        .def_prop_rw("DE", &Registers::DE, &Registers::set_DE)
        .def_prop_rw("HL", &Registers::HL, &Registers::set_HL)
        .def_prop_rw("AF", &Registers::AF, &Registers::set_AF)
        .def_prop_rw("IXh", &Registers::IXh, &Registers::set_IXh)
        .def_prop_rw("IXl", &Registers::IXl, &Registers::set_IXl)
        .def_prop_rw("IYh", &Registers::IYh, &Registers::set_IYh)
        .def_prop_rw("IYl", &Registers::IYl, &Registers::set_IYl)
        .def_rw("IFF1", &Registers::IFF1)
        .def_rw("IFF2", &Registers::IFF2)
        .def_rw("IM", &Registers::IM)
        .def("reset", &Registers::reset);

    nb::class_<SimpleBus, Bus>(m, "SimpleBus")
        .def(nb::init<>())
        .def("bus_read", &SimpleBus::bus_read)
        .def("bus_write", &SimpleBus::bus_write)
        .def("bus_io_read", &SimpleBus::bus_io_read)
        .def("bus_io_write", &SimpleBus::bus_io_write);

    nb::class_<CPU>(m, "Z80CPU")
        .def(nb::init<>())
        .def(nb::init<Bus*>())
        .def("step", &CPU::step)
        .def("run", &CPU::run)                      // NEW batch API
        .def("reset", &CPU::reset)
        .def("trigger_interrupt", &CPU::trigger_interrupt)
        .def("trigger_nmi", &CPU::trigger_nmi)
        .def_prop_ro("regs", [](CPU& c) -> Registers& { return c.regs; })
        .def_prop_ro("bus", [](CPU& c) -> Bus* { return c.bus; })
        .def_ro("cycles", &CPU::cycles)
        .def_ro("halted", &CPU::halted)
        .def_ro("instruction_count", &CPU::instruction_count);

    m.attr("FLAG_S") = (int)z80flags::FLAG_S;
    m.attr("FLAG_Z") = (int)z80flags::FLAG_Z;
    // ... all constants ...
    m.attr("PARITY_TABLE") = /* memoryview of 256 bytes */;
    m.attr("ADD_FLAGS") = /* memoryview of 64KB */;
    m.attr("SUB_FLAGS") = /* memoryview of 64KB */;
}
```

### Python Package

```python
# core/__init__.py
from ._pyz80 import Z80CPU, SimpleBus, Registers
from ._pyz80 import FLAG_S, FLAG_Z, FLAG_H, FLAG_PV, FLAG_N, FLAG_C
from ._pyz80 import PARITY_TABLE, ADD_FLAGS, SUB_FLAGS
```

All existing imports work unchanged.

---

## File Structure

```
cpp/
  core/
    flags.h / flags.cpp          # tables + flag helpers
    registers.h / registers.cpp  # Registers struct
    bus.h / bus.cpp              # Bus ABC + SimpleBus
    handlers.h / handlers.cpp    # ALL instruction handlers (~3000 lines)
    decoder.h / decoder.cpp      # Decoder + table building
    cpu.h / cpu.cpp              # CPU class + step()/run()
    pyz80.cpp                    # nanobind module

core/
  __init__.py                    # from ._pyz80 import ...
  _pyz80.so                      # compiled C++ extension
  state.py                       # unchanged (no perf concern)
  timing.py                      # unchanged
  instructions/
    __init__.py                  # unchanged

CMakeLists.txt                   # top-level CMake
pyproject.toml                   # scikit-build-core + nanobind
```

---

## Build System

### pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.8", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[project]
name = "PyZ80"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = []
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(pyz80)
set(CMAKE_CXX_STANDARD 20)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

add_library(z80_core STATIC
    cpp/core/flags.cpp
    cpp/core/registers.cpp
    cpp/core/bus.cpp
    cpp/core/handlers.cpp
    cpp/core/decoder.cpp
    cpp/core/cpu.cpp
)
target_include_directories(z80_core PUBLIC cpp/core)

nanobind_add_module(_pyz80 cpp/core/pyz80.cpp)
target_link_libraries(_pyz80 PRIVATE z80_core)
install(TARGETS _pyz80 LIBRARY DESTINATION pyz80)
```

---

## Implementation Phases

| Phase | What | Files |
|-------|------|-------|
| 1 | C++ foundation: lookup tables, Registers struct, Bus classes | flags.h/cpp, registers.h/cpp, bus.h/cpp |
| 2 | CPU core: all handlers, decoder, CPU class | handlers.h/cpp, decoder.h/cpp, cpu.h/cpp |
| 3 | C++ standalone tests: verify correctness + benchmark | tests/cpp/ |
| 4 | nanobind Python wrapper | pyz80.cpp |
| 5 | Build system integration + cleanup | CMakeLists.txt, pyproject.toml, core/__init__.py |

---

## step() Hot Path — Before vs After

**Python (current): ~50-100ns per instruction**
```python
def step(self):
    pc = regs.PC                           # Python __getattribute__
    opcode = mem[pc]                       # Python __getitem__
    op = cache_list[pc]                    # Python list lookup
    if op is None: op = self._decode(...)  # Python call
    regs.R = (r & 0x80) | ((r+1)&0x7F)    # Python arithmetic (boxed ints)
    op_cycles = op.handler(self)           # Lambda call
    regs.PC = (pc + op.length) & 0xFFFF    # Python assignment
```

**C++ (new): ~2-5ns per instruction**
```cpp
int CPU::step() {
    uint16_t pc = regs.PC;                 // register read
    uint8_t opcode = _mem[pc];             // array index
    const DecodeSlot& slot = _cache[pc];   // array index
    regs.R = (regs.R & 0x80) | ((regs.R+1)&0x7F);
    current_opcode = opcode;
    int t = slot.handler(*this);           // function pointer call (~1ns)
    regs.PC = (pc + slot.length) & 0xFFFF;
    cycles += t;
```

---

## Risk Areas

1. **nanobind uint8_t -> Python int**: automatic conversion, no issues expected.
2. **SimpleBus.memory access**: tests do `bus.memory[addr]`. Expose via buffer protocol.
3. **cpu.bus.io_ports access**: tests do `io_ports[port]`. Same buffer approach.
4. **decode_at() backward compat**: tools/check_opcodes.py needs `op.handler is not None`.
   Provide Python wrapper function returning a MicroOp dict/namedtuple.
5. **Handler count**: ~150-200 unique functions across 7 tables (1792 entries total).
   Many share functions via opcode bit decoding.
