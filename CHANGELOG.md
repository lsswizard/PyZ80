# Changelog

All notable changes to this project will be documented in this file.

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
