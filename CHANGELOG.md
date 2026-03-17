# Changelog

All notable changes to this project will be documented in this file.

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
