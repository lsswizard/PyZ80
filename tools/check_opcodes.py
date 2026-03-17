#!/usr/bin/env python3
"""
Z80 Opcode Coverage Verification Script

Checks that all Z80 opcodes have handlers defined.
"""

import sys

sys.path.insert(0, "/home/lss/builds/PyZ80")

from core import Z80CPU
from core.decoder import InstructionDecoder
from core.bus import SimpleBus


def check_opcode_tables():
    """Check all opcode tables for defined handlers."""
    decoder = InstructionDecoder()

    results = {
        "base": [],
        "cb": [],
        "ed": [],
        "dd": [],
        "fd": [],
        "ddcb": [],
        "fdcb": [],
    }

    # Check BASE opcodes (0x00-0xFF)
    print("Checking BASE opcodes...")
    missing_base = []
    for opcode in range(256):
        # Skip prefix bytes
        if opcode in (0xCB, 0xDD, 0xED, 0xFD):
            continue
        try:
            # Write a simple instruction
            bus = SimpleBus()
            bus.bus_write(0, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_base.append(f"0x{opcode:02X}")
        except Exception as e:
            missing_base.append(f"0x{opcode:02X} ({e})")

    results["base"] = missing_base

    # Check CB prefix (0xCB 0x00-0xFF)
    print("Checking CB prefix opcodes...")
    missing_cb = []
    for opcode in range(256):
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xCB, 0)
            bus.bus_write(1, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_cb.append(f"0xCB 0x{opcode:02X}")
        except Exception as e:
            missing_cb.append(f"0xCB 0x{opcode:02X} ({e})")

    results["cb"] = missing_cb

    # Check ED prefix (0xED 0x00-0xFF)
    print("Checking ED prefix opcodes...")
    missing_ed = []
    for opcode in range(256):
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xED, 0)
            bus.bus_write(1, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            # ED opcodes that are NOPs are still handled
            if op.handler is None:
                missing_ed.append(f"0xED 0x{opcode:02X}")
        except Exception as e:
            missing_ed.append(f"0xED 0x{opcode:02X} ({e})")

    results["ed"] = missing_ed

    # Check DD prefix (0xDD followed by common opcodes)
    print("Checking DD prefix opcodes...")
    dd_opcodes = [
        0x00,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x12,
        0x13,
        0x14,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
        0x1F,
        0x20,
        0x21,
        0x22,
        0x23,
        0x24,
        0x25,
        0x26,
        0x27,
        0x28,
        0x29,
        0x2A,
        0x2B,
        0x2C,
        0x2D,
        0x2E,
        0x2F,
        0x30,
        0x31,
        0x32,
        0x33,
        0x34,
        0x35,
        0x36,
        0x37,
        0x38,
        0x39,
        0x3A,
        0x3B,
        0x3C,
        0x3D,
        0x3E,
        0x3F,
        0x40,
        0x41,
        0x42,
        0x43,
        0x44,
        0x45,
        0x46,
        0x47,
        0x48,
        0x49,
        0x4A,
        0x4B,
        0x4C,
        0x4D,
        0x4E,
        0x4F,
        0x50,
        0x51,
        0x52,
        0x53,
        0x54,
        0x55,
        0x56,
        0x57,
        0x58,
        0x59,
        0x5A,
        0x5B,
        0x5C,
        0x5D,
        0x5E,
        0x5F,
        0x60,
        0x61,
        0x62,
        0x63,
        0x64,
        0x65,
        0x66,
        0x67,
        0x68,
        0x69,
        0x6A,
        0x6B,
        0x6C,
        0x6D,
        0x6E,
        0x6F,
        0x70,
        0x71,
        0x72,
        0x73,
        0x74,
        0x75,
        0x76,
        0x77,
        0x78,
        0x79,
        0x7A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x80,
        0x81,
        0x82,
        0x83,
        0x84,
        0x85,
        0x86,
        0x87,
        0x88,
        0x89,
        0x8A,
        0x8B,
        0x8C,
        0x8D,
        0x8E,
        0x8F,
        0x90,
        0x91,
        0x92,
        0x93,
        0x94,
        0x95,
        0x96,
        0x97,
        0x98,
        0x99,
        0x9A,
        0x9B,
        0x9C,
        0x9D,
        0x9E,
        0x9F,
        0xA0,
        0xA1,
        0xA2,
        0xA3,
        0xA4,
        0xA5,
        0xA6,
        0xA7,
        0xA8,
        0xA9,
        0xAA,
        0xAB,
        0xAC,
        0xAD,
        0xAE,
        0xAF,
        0xB0,
        0xB1,
        0xB2,
        0xB3,
        0xB4,
        0xB5,
        0xB6,
        0xB7,
        0xB8,
        0xB9,
        0xBA,
        0xBB,
        0xBC,
        0xBD,
        0xBE,
        0xBF,
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC4,
        0xC5,
        0xC6,
        0xC7,
        0xC8,
        0xC9,
        0xCA,
        0xCB,
        0xCC,
        0xCD,
        0xCE,
        0xCF,
        0xD0,
        0xD1,
        0xD2,
        0xD3,
        0xD4,
        0xD5,
        0xD6,
        0xD7,
        0xD8,
        0xD9,
        0xDA,
        0xDB,
        0xDC,
        0xDD,
        0xDE,
        0xDF,
        0xE0,
        0xE1,
        0xE2,
        0xE3,
        0xE4,
        0xE5,
        0xE6,
        0xE7,
        0xE8,
        0xE9,
        0xEA,
        0xEB,
        0xEC,
        0xED,
        0xEE,
        0xEF,
        0xF0,
        0xF1,
        0xF2,
        0xF3,
        0xF4,
        0xF5,
        0xF6,
        0xF7,
        0xF8,
        0xF9,
        0xFA,
        0xFB,
        0xFC,
        0xFD,
        0xFE,
        0xFF,
    ]
    missing_dd = []
    for opcode in dd_opcodes:
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xDD, 0)
            bus.bus_write(1, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_dd.append(f"0xDD 0x{opcode:02X}")
        except Exception as e:
            missing_dd.append(f"0xDD 0x{opcode:02X} ({e})")

    results["dd"] = missing_dd

    # Same for FD
    print("Checking FD prefix opcodes...")
    missing_fd = []
    for opcode in dd_opcodes:
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xFD, 0)
            bus.bus_write(1, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_fd.append(f"0xFD 0x{opcode:02X}")
        except Exception as e:
            missing_fd.append(f"0xFD 0x{opcode:02X} ({e})")

    results["fd"] = missing_fd

    # Check DDCB prefix (0xDD 0xCB displacement opcode)
    print("Checking DDCB prefix opcodes...")
    missing_ddcb = []
    for opcode in range(256):
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xDD, 0)
            bus.bus_write(1, 0xCB, 0)
            bus.bus_write(2, 0x00, 0)  # displacement
            bus.bus_write(3, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_ddcb.append(f"0xDD 0xCB 0x00 0x{opcode:02X}")
        except Exception as e:
            missing_ddcb.append(f"0xDD 0xCB 0x00 0x{opcode:02X} ({e})")

    results["ddcb"] = missing_ddcb

    # Same for FDCB
    print("Checking FDCB prefix opcodes...")
    missing_fdcb = []
    for opcode in range(256):
        try:
            bus = SimpleBus()
            bus.bus_write(0, 0xFD, 0)
            bus.bus_write(1, 0xCB, 0)
            bus.bus_write(2, 0x00, 0)
            bus.bus_write(3, opcode, 0)
            op = decoder.decode_at(bus.memory, 0)
            if op.handler is None:
                missing_fdcb.append(f"0xFD 0xCB 0x00 0x{opcode:02X}")
        except Exception as e:
            missing_fdcb.append(f"0xFD 0xCB 0x00 0x{opcode:02X} ({e})")

    results["fdcb"] = missing_fdcb

    return results


def print_results(results):
    """Print the results in a readable format."""
    print("\n" + "=" * 60)
    print("Z80 OPCODE COVERAGE REPORT")
    print("=" * 60)

    total_missing = 0

    for table_name, missing in results.items():
        print(f"\n{table_name.upper()} Opcodes:")
        if missing:
            print(f"  MISSING: {len(missing)}")
            for m in missing[:10]:  # Show first 10
                print(f"    {m}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
            total_missing += len(missing)
        else:
            print(f"  OK: All handled")

    print("\n" + "=" * 60)
    if total_missing == 0:
        print("RESULT: ALL OPCODES ARE HANDLED!")
    else:
        print(f"RESULT: {total_missing} opcodes may be missing handlers")
    print("=" * 60)

    return total_missing


if __name__ == "__main__":
    results = check_opcode_tables()
    print_results(results)
