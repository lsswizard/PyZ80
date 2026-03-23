# Changelog

All notable changes to this project will be documented in this file.

## [1.6.0] - 2026-03-23

### Performance Improvements
- **Precomputed ALU flag tables**: Added ~320KB of lookup tables for 8-bit ALU operations (ADD, ADC, SUB, SBC, CP, INC, DEC), eliminating 5-8 branch operations per instruction
- **Register dispatch tables**: Created `_GET_REG8`/`_SET_REG8` function tables replacing if-chains in `get_reg8()`/`set_reg8()` with single table lookups
- **`_IS_PREFIXED` bytearray**: O(1) prefix detection replacing frozenset lookup
- **MicroOp `__slots__`**: Replaced dataclass with `__slots__` (~12% faster attribute access on hot path)
- **Slow path optimization**: Added `_needs_slow_step` flag to avoid checking 4 conditions per step
- **16-bit register inlining**: Inlined `get_reg16()`/`set_reg16()` operations (~28% faster)
- **Decoder cleanup**: Merged DD/FD fallback tables, removed unnecessary lambda wrappers

### Bug Fixes
- **instruction_count not incremented**: Fixed missing `self.instruction_count += 1` in execute path

### Verified
- Benchmark shows ~97x improvement on small test, 6-23% improvement on various instruction mixes

## [1.5.0] - 2026-03-22

### Bug Fixes
- **Block compare timing (CPD/CPDR)**: Fixed missing `+1` T-state offset in bus read for `cpd` and `cpdr`, matching `cpi`/`cpir` timing
- **CPD cycle accounting**: Added missing `cpu.cycles += 16` in `cpd` instruction
- **Undocumented LD IXH,IXH opcode**: Fixed DD 64 dispatching to `ld_ixh_ixh` instead of `ld_ixh_ixl` (same fix for FD 64)
- **Duplicate DEC r function**: Removed duplicate `dec_r` definition in alu8.py

### Performance Improvements
- **Bit operation masks**: Precomputed `_BIT_MASK` and `_RES_MASK` tuples for BIT/SET/RES instructions, replacing per-call `1 << bit` and `~(1 << bit)` operations
- **Opcode table O(1) access**: Changed all opcode tables (BASE, CB, ED, DD, FD, DDCB, FDCB) from dictionaries to fixed-size lists[256], eliminating hash overhead
- **Undocumented IXH/IXL self-copy**: Added `ld_ixh_ixh` and `ld_ixl_ixl` no-op functions (8 T-states) for completeness

### Testing (860 → 903 tests)
- **Block instruction timing**: Added 13 new timing tests for LDD, LDDR, CPI, CPD, CPIR, CPDR covering 16-cycle (final/match) and 21-cycle (repeating) cases
- **Undocumented IXH/IXL instructions**: Added 14 tests for self-copy, cross-copy, immediate load, register transfer, and IY variants
- **DDCB/FDCB timing**: Replaced single timing test with 18 parametrized cases covering RLC/RRC/RL/RR/SLA/SRA/SRL (23 cycles), SET/RES (23 cycles), and BIT (20 cycles) for both IX and IY indexed addressing

### Verified
- All 903 tests pass
- Cycle timing verified against Z80 CPU User Manual (UM0080)

## [1.4.0] - 2026-03-17

### Performance Improvements
- **Rotate/Shift lookup tables**: Added LUTs (~5KB) for all rotate/shift operations (RLC, RRC, RL, RR, SLA, SRA, SLL, SRL) replacing procedural shifts with single memory lookups
- **DAA lookup table**: Added 2KB LUT for Decimal Adjust Accumulator, precomputing all 2048 combinations of (N, H, C, A)
- **Explicit T-state timing**: Replaced `advance_cycles()` calls with explicit T-state offsets in bus operations, reducing function call overhead by ~70 calls
- **New CPU methods**: Added `read_at`, `write_at`, `read16_at`, `write16_at`, `push16_at`, `pop16_at` for cycle-exact bus timing

### Architecture
- Removed deprecated `advance_cycles` method
- All bus operations now pass explicit T-state positions matching Z80 UM0080 specification
- Core remains machine-independent (pure Python, no platform-specific code)

### Verified
- All 860 tests pass

## [1.3.0] - 2026-03-17

### New Features
- **ADC/SBC IX/IY instructions**: Implemented undocumented 16-bit arithmetic with index registers (ADC IX,BC, ADC IX,DE, SBC IX,BC, SBC IX,DE and IY equivalents)
- **Opcode verification tool**: Added `tools/check_opcodes.py` to verify all Z80 opcodes are properly handled

### Bug Fixes
- **DD/FD prefix fallthrough**: Fixed Bug8 - unknown DD/FD prefixes now correctly fall through to base opcode with proper PC handling for immediate values
- **DDCB/FDCB BIT operations**: Fixed opcode mapping for indexed BIT instructions
- **LD (nn),SP**: Fixed test and verified correct operation
- **Page boundary handling**: Fixed CALL/JP instructions at page boundaries

### Testing
- Added comprehensive test sections for:
  - Undocumented flags (F3/F5)
  - DD/FD prefix undocumented behavior
  - DDCB/FDCB indexed bit operations
  - Repeat I/O block instructions (INIR, INDR, OTIR, OTDR)
  - IX/IY 16-bit arithmetic
  - Comprehensive DAA tests
  - CCF H-flag behavior
  - 16-bit LD with SP
  - Page boundary edge cases
  - RST comprehensive tests
- All 860 tests pass

### Verified
- All Z80 opcode tables have 100% coverage (BASE, CB, ED, DD, FD, DDCB, FDCB)

## [1.2.0] - 2026-03-17

### Performance Improvements
- **Hot path optimization**: Added local variable caching in CPU execute loop to reduce attribute lookups
- **Cache-first dispatch**: Check pre-decoded cache before calling decoder for O(1) fast-path
- **Reduced JIT duplicates**: Removed duplicate Numba JIT function definitions

### Bug Fixes
- **DDCB/FDCB displacement bug**: Fixed Bug4 - handlers now correctly read displacement at execution time via `_get_indexed_addr()` instead of capturing it at decode time (which clobbered the bit-number field)
- **DD/FD prefix fallthrough**: Fixed Bug8 - unknown DD/FD prefixes now fall through to base opcode per Z80 undocumented behaviour
- **Bank switch cache invalidation**: Fixed incorrect optimization that silently skipped small memory ranges (<4KB) during bank switches - now correctly flushes all bank windows regardless of size

### Code Quality
- **Simplified flag computation**: Consolidated lookup tables (SZ53P_TABLE, SZHZP_TABLE) for cleaner single-lookup flag operations
- **Streamlined decoder**: Improved documentation and code clarity in decoder module
- **Complete state snapshots**: Added missing interrupt state fields (interrupt_pending, interrupt_data, nmi_pending, bus_request, EI_JUST_RESOLVED) to CPUState

### Verified
- All existing tests pass

### Critical Bug Fixes
- **IFF2 not cleared on maskable interrupt**: Now correctly clears both IFF1 and IFF2 (was only clearing IFF1)
- **IM 0 not implemented**: Now handles RST opcodes in IM 0 mode; falls back to RST 38h for non-RST opcodes
- **IM 2 cycle count**: Fixed to return 19 cycles (was incorrectly returning 20)
- **NMI + HALT handling**: Now correctly exits HALT and advances PC past HALT instruction on NMI

### Moderate Bug Fixes
- **LD A,I/R interrupt bug**: Now correctly checks previous instruction (was checking current instruction)
- **EI deferral**: Fixed to properly defer interrupt acceptance when IFF1 was already set
- **get_state/set_state**: Added missing fields (interrupt_pending, interrupt_data, nmi_pending, bus_request)
- **reset**: Now clears bus_request and _is_ld_a_ir flags

### Minor Bug Fixes
- **Spurious ED bus read**: Removed early check that read extra byte for ED-prefixed instructions
- **Cycle timestamps**: Improved timing accuracy in interrupt handlers
- **File handle leak**: Added __del__ to close trace file on object destruction

### Accuracy Improvements
- **SMC cache invalidation**: Added _internal_write wrapper to ensure decoder cache is always invalidated on memory writes
- **Cache address wrapping**: Fixed 16-bit wrap-around in invalidate_cache for instructions at page boundaries
- **Interrupt timing**: Updated push cycle timestamps for accurate bus timing

### Testing
- Added 7 new tests for interrupt handling edge cases:
  - test_maskable_interrupt_clears_both_iffs
  - test_nmi_exits_halt
  - test_nmi_returns_past_halt
  - test_maskable_interrupt_exits_halt
  - test_ei_when_already_enabled
  - test_ld_a_i_interrupt_bug
  - test_ld_a_r_interrupt_bug

### Verified
- All 802 tests pass
- Z80 hardware accuracy improved for interrupt handling

## [1.0.0] - 2026-03-09

### Added
- Complete Z80 CPU emulation core in `core/`
- Instruction set implementation covering all documented and undocumented Z80 opcodes
- Pre-decoding instruction cache for fast execution
- State management for debugging and rewind functionality
- Cycle-accurate timing engine
- Comprehensive test suite (795 tests)

### Fixed
- Fixed broken lambda functions in DD/FD opcode tables (LD IXH,IXH, LD IXL,IXL, LD IYH,IYH, LD IYL,IYL)
- Fixed JP (IX/IY) MEMPTR register - now correctly sets MEMPTR to IX/IY value
- Fixed EX (SP),IX/IY MEMPTR - now correctly sets MEMPTR to value read from stack
- Fixed INI/IND/OUTI/OUTD repeated execution timing (16/21 T-states)

### Refactored
- Consolidated 16-bit address reading into `_read_addr_from_pc()` helper
- Consolidated stack operations into `_push_word()` and `_pop_word()` helpers
- Consolidated INI/IND/OUTI/OUTD flag calculations into `_compute_in_out_flags()`
- Consolidated LDI/LDD block flags into `_compute_ld_block_flags()`
- Replaced if/elif chain in `_ixycb_rot()` with lookup table (`_ROT_OPS`)
- Removed unused imports and redundant code

### Verified
- All 795 tests pass
- Timing accuracy verified against Z80 documentation
- Machine independence maintained (uses Z80Bus protocol for all external interactions)

## [0.0.0] - 2026-03-09

### Added
- Initial project structure
