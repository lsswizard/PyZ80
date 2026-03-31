# PyZ80 Cython Conversion Plan

## Design Decisions

- **Python 3.13** target
- **Integer dispatch** with C function pointer tables (replace 200+ lambdas)
- **Pure lookup tables** replace Numba JIT (drop numpy/numba/scipy)
- **Maximum performance** — full Cython conversion of all hot paths

## Architecture

### Handler Dispatch: C Function Pointers

```cython
ctypedef int (*handler_fn)(Z80CPU* cpu)

cdef struct OpcodeEntry:
    handler_fn handler
    unsigned char cycles
    unsigned char length
    bint affects_f
    bint is_ld_a_ir

cdef OpcodeEntry _base_handlers[256]
cdef OpcodeEntry _cb_handlers[256]
```

Every handler becomes `cdef int handler(Z80CPU* cpu)` — uniform signature.
Parameters decoded from opcode bits inside handler. DD/FD handlers use
`cpu._is_iy` flag set by dispatch layer.

### Cache: C Struct Array

```cython
cdef struct CacheSlot:
    handler_fn handler
    unsigned char cycles
    unsigned char length
    bint affects_f
    bint is_ld_a_ir

cdef CacheSlot _cache[65536]
```

### Registers: cdef class with typed attributes

All 30 fields -> `cdef public unsigned char/short`. `@property` wrappers
keep BC/DE/HL/AF working for external code.

### Flags: Drop Numba, keep tables + cdef inline for 16-bit

8-bit flags already table-driven. 16-bit flag helpers -> `cdef inline`
(compiled by Cython, same speed as Numba JIT).

---

## Implementation Phases

### Phase 1: Foundation

| Task | File | What |
|------|------|------|
| 1a | flags.py -> flags.pyx | Remove all Numba code, keep tables as cdef unsigned char[:], convert 16-bit helpers to cdef inline. Create flags.pxd. |
| 1b | registers.py -> registers.pyx | cdef class with typed attrs, cdef inline pair accessors, Python @property wrappers. Create registers.pxd. |
| 1c | bus.py -> bus.pyx | cdef class SimpleBus with cdef unsigned char[:] memory/io_ports, cdef inline bus_read/write. Create bus.pxd. |

### Phase 2: Core CPU

| Task | File | What |
|------|------|------|
| 2a | primitives.py -> primitives.pyx | cdef class MicroOp wrapping handler+metadata. Create primitives.pxd. |
| 2b | NEW instructions/handlers.pyx + handlers.pxd | Consolidate ALL handlers from ld8/ld16/alu8/alu16/bit/jump/block/misc into one .pyx. ~3000 lines. |
| 2c | instructions/opcodes.py -> opcodes.pyx | Rewrite table building: direct C function pointer assignment. Build OpcodeEntry[256] tables. Create opcodes.pxd. |
| 2d | decoder.py -> decoder.pyx | Cache -> CacheSlot[65536] C array. Create decoder.pxd. |
| 2e | cpu.py -> cpu.pyx | cdef class Z80CPU, fully typed step(). C-level dispatch through cache. Create cpu.pxd. |

### Phase 3: Integration

| Task | What |
|------|------|
| 3a | Create setup.py with cythonize for all .pyx files |
| 3b | Update pyproject.toml: remove numba/numpy/scipy, add cython>=3.0.0, python >=3.13 |
| 3c | Update .python-version to 3.13 |
| 3d | Delete old .py files replaced by .pyx |
| 3e | Run pytest tests/ + benchmark.py |

---

## step() Hot Path — Before vs After

**Before** (Python):
```python
def step(self):
    pc = regs.PC                          # Python attr
    opcode = mem[pc]                      # Python indexing
    op = cache_list[pc]                   # Python list lookup
    if op is None: op = self._decode(...)
    self.cycles += 4
    op_cycles = op.handler(self)          # PYTHON CALL through lambda
    regs.PC = (pc + op.length) & 0xFFFF
```

**After** (Cython):
```cython
def step(self):
    cdef unsigned int pc = self.regs.PC
    cdef unsigned char opcode = self._mem[pc]
    cdef CacheSlot slot = self._cache[pc]
    if slot.handler == NULL:
        slot = _decode_instruction(...)
        self._cache[pc] = slot
    self.cycles += 4
    cdef int op_cycles = slot.handler(self)   # C FUNCTION CALL
    self.regs.PC = (pc + slot.length) & 0xFFFF
```

---

## File Map

```
core/
  __init__.py              # unchanged
  registers.pyx + .pxd     # cdef class (NEW)
  flags.pyx + .pxd         # lookup tables, no Numba (NEW)
  bus.pyx + .pxd           # cdef class SimpleBus (NEW)
  primitives.pyx + .pxd    # cdef class MicroOp (NEW)
  decoder.pyx + .pxd       # C cache array (NEW)
  cpu.pyx + .pxd           # cdef class Z80CPU, typed step() (NEW)
  state.py                 # UNCHANGED
  timing.py                # UNCHANGED
  instructions/
    __init__.py            # unchanged
    handlers.pyx + .pxd    # ALL handlers consolidated (NEW)
    opcodes.pyx + .pxd     # C pointer tables (NEW)

setup.py                   # build script (NEW)
```

## API Compatibility

All public Python APIs preserved:
- Z80CPU(bus), .reset(), .step(), .regs, .cycles, .halted
- cpu.regs.A, cpu.regs.BC, cpu.regs.BC = val
- SimpleBus, .memory, .bus_read(), .bus_write()
- FLAG_S/Z/H/PV/N/C, PARITY_TABLE, ADD_FLAGS, SUB_FLAGS
- decoder.decode_at() -> MicroOp with .handler, .cycles, .length, .mnemonic

## Expected Speedup: 3-5x over current Python + Numba
