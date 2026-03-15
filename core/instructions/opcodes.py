from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from ..core import Z80CPU

from .ld8 import *
from .ld16 import *
from .alu8 import *
from .alu16 import *
from .bit import *
from .jump import *
from .misc import *
from .block import *

_REG_NAMES = ["B", "C", "D", "E", "H", "L", "(HL)", "A"]


def _reg_name(reg: int) -> str:
    """Get register name from index"""
    return _REG_NAMES[reg]


# Opcode tables
BASE_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = {}
CB_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = {}
ED_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = {}
DD_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = {}
FD_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = {}


def _build_base_opcode_table():
    global BASE_OPCODE_TABLE
    for dest in range(8):
        for src in range(8):
            opcode = 0x40 | (dest << 3) | src
            if opcode == 0x76:
                continue
            if dest == 6:
                BASE_OPCODE_TABLE[opcode] = (
                    lambda cpu, s=src: ld_hl_r(cpu, s),
                    7,
                    1,
                    f"LD (HL),{_reg_name(src)}",
                )
            elif src == 6:
                BASE_OPCODE_TABLE[opcode] = (
                    lambda cpu, d=dest: ld_r_hl(cpu, d),
                    7,
                    1,
                    f"LD {_reg_name(dest)},(HL)",
                )
            else:
                BASE_OPCODE_TABLE[opcode] = (
                    lambda cpu, d=dest, s=src: ld_r_r(cpu, d, s),
                    4,
                    1,
                    f"LD {_reg_name(dest)},{_reg_name(src)}",
                )

    for reg in range(8):
        opcode = 0x06 | (reg << 3)
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (ld_hl_n, 10, 2, "LD (HL),n")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: ld_r_n(cpu, r),
                7,
                2,
                f"LD {_reg_name(reg)},n",
            )

    BASE_OPCODE_TABLE[0x0A] = (ld_a_bc, 7, 1, "LD A,(BC)")
    BASE_OPCODE_TABLE[0x1A] = (ld_a_de, 7, 1, "LD A,(DE)")
    BASE_OPCODE_TABLE[0x3A] = (ld_a_nn, 13, 3, "LD A,(nn)")
    BASE_OPCODE_TABLE[0x02] = (ld_bc_a, 7, 1, "LD (BC),A")
    BASE_OPCODE_TABLE[0x12] = (ld_de_a, 7, 1, "LD (DE),A")
    BASE_OPCODE_TABLE[0x32] = (ld_nn_a, 13, 3, "LD (nn),A")

    BASE_OPCODE_TABLE[0x01] = (lambda cpu: ld_rr_nn(cpu, 0), 10, 3, "LD BC,nn")
    BASE_OPCODE_TABLE[0x11] = (lambda cpu: ld_rr_nn(cpu, 1), 10, 3, "LD DE,nn")
    BASE_OPCODE_TABLE[0x21] = (lambda cpu: ld_rr_nn(cpu, 2), 10, 3, "LD HL,nn")
    BASE_OPCODE_TABLE[0x31] = (lambda cpu: ld_rr_nn(cpu, 3), 10, 3, "LD SP,nn")

    BASE_OPCODE_TABLE[0x2A] = (ld_hl_nn, 16, 3, "LD HL,(nn)")
    BASE_OPCODE_TABLE[0x22] = (ld_nn_hl, 16, 3, "LD (nn),HL")
    BASE_OPCODE_TABLE[0xF9] = (ld_sp_hl, 6, 1, "LD SP,HL")

    BASE_OPCODE_TABLE[0xC5] = (lambda cpu: push_rr(cpu, 0), 11, 1, "PUSH BC")
    BASE_OPCODE_TABLE[0xD5] = (lambda cpu: push_rr(cpu, 1), 11, 1, "PUSH DE")
    BASE_OPCODE_TABLE[0xE5] = (lambda cpu: push_rr(cpu, 2), 11, 1, "PUSH HL")
    BASE_OPCODE_TABLE[0xF5] = (lambda cpu: push_rr(cpu, 3), 11, 1, "PUSH AF")

    BASE_OPCODE_TABLE[0xC1] = (lambda cpu: pop_rr(cpu, 0), 10, 1, "POP BC")
    BASE_OPCODE_TABLE[0xD1] = (lambda cpu: pop_rr(cpu, 1), 10, 1, "POP DE")
    BASE_OPCODE_TABLE[0xE1] = (lambda cpu: pop_rr(cpu, 2), 10, 1, "POP HL")
    BASE_OPCODE_TABLE[0xF1] = (lambda cpu: pop_rr(cpu, 3), 10, 1, "POP AF")

    BASE_OPCODE_TABLE[0xEB] = (ex_de_hl, 4, 1, "EX DE,HL")
    BASE_OPCODE_TABLE[0x08] = (ex_af_afp, 4, 1, "EX AF,AF'")
    BASE_OPCODE_TABLE[0xD9] = (exx, 4, 1, "EXX")
    BASE_OPCODE_TABLE[0xE3] = (ex_sp_hl, 19, 1, "EX (SP),HL")

    for reg in range(8):
        opcode = 0x80 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (add_a_hl, 7, 1, "ADD A,(HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: add_a_r(cpu, r),
                4,
                1,
                f"ADD A,{_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xC6] = (add_a_n, 7, 2, "ADD A,n")

    for reg in range(8):
        opcode = 0x88 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (adc_a_hl, 7, 1, "ADC A,(HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: adc_a_r(cpu, r),
                4,
                1,
                f"ADC A,{_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xCE] = (adc_a_n, 7, 2, "ADC A,n")

    for reg in range(8):
        opcode = 0x90 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (sub_hl, 7, 1, "SUB (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: sub_r(cpu, r),
                4,
                1,
                f"SUB {_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xD6] = (sub_n, 7, 2, "SUB n")

    for reg in range(8):
        opcode = 0x98 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (sbc_a_hl, 7, 1, "SBC A,(HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: sbc_a_r(cpu, r),
                4,
                1,
                f"SBC A,{_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xDE] = (sbc_a_n, 7, 2, "SBC A,n")

    for reg in range(8):
        opcode = 0xA0 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (and_hl, 7, 1, "AND (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: and_r(cpu, r),
                4,
                1,
                f"AND {_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xE6] = (and_n, 7, 2, "AND n")

    for reg in range(8):
        opcode = 0xA8 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (xor_hl, 7, 1, "XOR (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: xor_r(cpu, r),
                4,
                1,
                f"XOR {_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xEE] = (xor_n, 7, 2, "XOR n")

    for reg in range(8):
        opcode = 0xB0 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (or_hl, 7, 1, "OR (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: or_r(cpu, r),
                4,
                1,
                f"OR {_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xF6] = (or_n, 7, 2, "OR n")

    for reg in range(8):
        opcode = 0xB8 | reg
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (cp_hl, 7, 1, "CP (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: cp_r(cpu, r),
                4,
                1,
                f"CP {_reg_name(reg)}",
            )
    BASE_OPCODE_TABLE[0xFE] = (cp_n, 7, 2, "CP n")

    for reg in range(8):
        opcode = 0x04 | (reg << 3)
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (inc_hl, 11, 1, "INC (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: inc_r(cpu, r),
                4,
                1,
                f"INC {_reg_name(reg)}",
            )

    for reg in range(8):
        opcode = 0x05 | (reg << 3)
        if reg == 6:
            BASE_OPCODE_TABLE[opcode] = (dec_hl, 11, 1, "DEC (HL)")
        else:
            BASE_OPCODE_TABLE[opcode] = (
                lambda cpu, r=reg: dec_r(cpu, r),
                4,
                1,
                f"DEC {_reg_name(reg)}",
            )

    BASE_OPCODE_TABLE[0x09] = (lambda cpu: add_hl_rr(cpu, 0), 11, 1, "ADD HL,BC")
    BASE_OPCODE_TABLE[0x19] = (lambda cpu: add_hl_rr(cpu, 1), 11, 1, "ADD HL,DE")
    BASE_OPCODE_TABLE[0x29] = (lambda cpu: add_hl_rr(cpu, 2), 11, 1, "ADD HL,HL")
    BASE_OPCODE_TABLE[0x39] = (lambda cpu: add_hl_rr(cpu, 3), 11, 1, "ADD HL,SP")

    BASE_OPCODE_TABLE[0x03] = (lambda cpu: inc_rr(cpu, 0), 6, 1, "INC BC")
    BASE_OPCODE_TABLE[0x13] = (lambda cpu: inc_rr(cpu, 1), 6, 1, "INC DE")
    BASE_OPCODE_TABLE[0x23] = (lambda cpu: inc_rr(cpu, 2), 6, 1, "INC HL")
    BASE_OPCODE_TABLE[0x33] = (lambda cpu: inc_rr(cpu, 3), 6, 1, "INC SP")

    BASE_OPCODE_TABLE[0x0B] = (lambda cpu: dec_rr(cpu, 0), 6, 1, "DEC BC")
    BASE_OPCODE_TABLE[0x1B] = (lambda cpu: dec_rr(cpu, 1), 6, 1, "DEC DE")
    BASE_OPCODE_TABLE[0x2B] = (lambda cpu: dec_rr(cpu, 2), 6, 1, "DEC HL")
    BASE_OPCODE_TABLE[0x3B] = (lambda cpu: dec_rr(cpu, 3), 6, 1, "DEC SP")

    BASE_OPCODE_TABLE[0x27] = (daa, 4, 1, "DAA")
    BASE_OPCODE_TABLE[0x2F] = (cpl, 4, 1, "CPL")
    BASE_OPCODE_TABLE[0x3F] = (ccf, 4, 1, "CCF")
    BASE_OPCODE_TABLE[0x37] = (scf, 4, 1, "SCF")

    BASE_OPCODE_TABLE[0x07] = (rlca, 4, 1, "RLCA")
    BASE_OPCODE_TABLE[0x0F] = (rrca, 4, 1, "RRCA")
    BASE_OPCODE_TABLE[0x17] = (rla, 4, 1, "RLA")
    BASE_OPCODE_TABLE[0x1F] = (rra, 4, 1, "RRA")

    BASE_OPCODE_TABLE[0xC3] = (jp_nn, 10, 3, "JP nn")
    BASE_OPCODE_TABLE[0xC2] = (lambda cpu: jp_cc_nn(cpu, 0), 10, 3, "JP NZ,nn")
    BASE_OPCODE_TABLE[0xCA] = (lambda cpu: jp_cc_nn(cpu, 1), 10, 3, "JP Z,nn")
    BASE_OPCODE_TABLE[0xD2] = (lambda cpu: jp_cc_nn(cpu, 2), 10, 3, "JP NC,nn")
    BASE_OPCODE_TABLE[0xDA] = (lambda cpu: jp_cc_nn(cpu, 3), 10, 3, "JP C,nn")
    BASE_OPCODE_TABLE[0xE2] = (lambda cpu: jp_cc_nn(cpu, 4), 10, 3, "JP PO,nn")
    BASE_OPCODE_TABLE[0xEA] = (lambda cpu: jp_cc_nn(cpu, 5), 10, 3, "JP PE,nn")
    BASE_OPCODE_TABLE[0xF2] = (lambda cpu: jp_cc_nn(cpu, 6), 10, 3, "JP P,nn")
    BASE_OPCODE_TABLE[0xFA] = (lambda cpu: jp_cc_nn(cpu, 7), 10, 3, "JP M,nn")

    BASE_OPCODE_TABLE[0xE9] = (jp_hl, 4, 1, "JP (HL)")
    BASE_OPCODE_TABLE[0x18] = (jr_e, 12, 2, "JR e")
    BASE_OPCODE_TABLE[0x20] = (lambda cpu: jr_cc_e(cpu, 0), 7, 2, "JR NZ,e")
    BASE_OPCODE_TABLE[0x28] = (lambda cpu: jr_cc_e(cpu, 1), 7, 2, "JR Z,e")
    BASE_OPCODE_TABLE[0x30] = (lambda cpu: jr_cc_e(cpu, 2), 7, 2, "JR NC,e")
    BASE_OPCODE_TABLE[0x38] = (lambda cpu: jr_cc_e(cpu, 3), 7, 2, "JR C,e")
    BASE_OPCODE_TABLE[0x10] = (djnz_e, 8, 2, "DJNZ e")

    BASE_OPCODE_TABLE[0xCD] = (call_nn, 17, 3, "CALL nn")
    BASE_OPCODE_TABLE[0xC4] = (lambda cpu: call_cc_nn(cpu, 0), 10, 3, "CALL NZ,nn")
    BASE_OPCODE_TABLE[0xCC] = (lambda cpu: call_cc_nn(cpu, 1), 10, 3, "CALL Z,nn")
    BASE_OPCODE_TABLE[0xD4] = (lambda cpu: call_cc_nn(cpu, 2), 10, 3, "CALL NC,nn")
    BASE_OPCODE_TABLE[0xDC] = (lambda cpu: call_cc_nn(cpu, 3), 10, 3, "CALL C,nn")
    BASE_OPCODE_TABLE[0xE4] = (lambda cpu: call_cc_nn(cpu, 4), 10, 3, "CALL PO,nn")
    BASE_OPCODE_TABLE[0xEC] = (lambda cpu: call_cc_nn(cpu, 5), 10, 3, "CALL PE,nn")
    BASE_OPCODE_TABLE[0xF4] = (lambda cpu: call_cc_nn(cpu, 6), 10, 3, "CALL P,nn")
    BASE_OPCODE_TABLE[0xFC] = (lambda cpu: call_cc_nn(cpu, 7), 10, 3, "CALL M,nn")

    BASE_OPCODE_TABLE[0xC9] = (ret, 10, 1, "RET")
    BASE_OPCODE_TABLE[0xC0] = (lambda cpu: ret_cc(cpu, 0), 5, 1, "RET NZ")
    BASE_OPCODE_TABLE[0xC8] = (lambda cpu: ret_cc(cpu, 1), 5, 1, "RET Z")
    BASE_OPCODE_TABLE[0xD0] = (lambda cpu: ret_cc(cpu, 2), 5, 1, "RET NC")
    BASE_OPCODE_TABLE[0xD8] = (lambda cpu: ret_cc(cpu, 3), 5, 1, "RET C")
    BASE_OPCODE_TABLE[0xE0] = (lambda cpu: ret_cc(cpu, 4), 5, 1, "RET PO")
    BASE_OPCODE_TABLE[0xE8] = (lambda cpu: ret_cc(cpu, 5), 5, 1, "RET PE")
    BASE_OPCODE_TABLE[0xF0] = (lambda cpu: ret_cc(cpu, 6), 5, 1, "RET P")
    BASE_OPCODE_TABLE[0xF8] = (lambda cpu: ret_cc(cpu, 7), 5, 1, "RET M")

    for p in range(8):
        opcode = 0xC7 | (p << 3)
        addr = p * 8
        BASE_OPCODE_TABLE[opcode] = (
            lambda cpu, a=addr: rst_p(cpu, a),
            11,
            1,
            f"RST {addr:02X}h",
        )

    BASE_OPCODE_TABLE[0xDB] = (in_a_n, 11, 2, "IN A,(n)")
    BASE_OPCODE_TABLE[0xD3] = (out_n_a, 11, 2, "OUT (n),A")
    BASE_OPCODE_TABLE[0x00] = (nop, 4, 1, "NOP")
    BASE_OPCODE_TABLE[0x76] = (halt, 4, 1, "HALT")
    BASE_OPCODE_TABLE[0xF3] = (di, 4, 1, "DI")
    BASE_OPCODE_TABLE[0xFB] = (ei, 4, 1, "EI")


def _build_cb_opcode_table():
    global CB_OPCODE_TABLE
    rot_ops = ["RLC", "RRC", "RL", "RR", "SLA", "SRA", "SLL", "SRL"]
    for opcode in range(256):
        op_type = (opcode >> 6) & 0x03
        bit_or_op = (opcode >> 3) & 0x07
        reg = opcode & 0x07
        if op_type == 0:
            op_name = rot_ops[bit_or_op]
            if reg == 6:
                handlers = {
                    0: rlc_hl,
                    1: rrc_hl,
                    2: rl_hl,
                    3: rr_hl,
                    4: sla_hl,
                    5: sra_hl,
                    6: sll_hl,
                    7: srl_hl,
                }
                CB_OPCODE_TABLE[opcode] = (
                    handlers[bit_or_op],
                    15,
                    2,
                    f"{op_name} (HL)",
                )
            else:
                handlers = {
                    0: rlc_r,
                    1: rrc_r,
                    2: rl_r,
                    3: rr_r,
                    4: sla_r,
                    5: sra_r,
                    6: sll_r,
                    7: srl_r,
                }
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, h=handlers[bit_or_op], r=reg: h(cpu, r),
                    8,
                    2,
                    f"{op_name} {_reg_name(reg)}",
                )
        elif op_type == 1:
            if reg == 6:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op: bit_n_hl(cpu, b),
                    12,
                    2,
                    f"BIT {bit_or_op},(HL)",
                )
            else:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op, r=reg: bit_n_r(cpu, b, r),
                    8,
                    2,
                    f"BIT {bit_or_op},{_reg_name(reg)}",
                )
        elif op_type == 2:
            if reg == 6:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op: res_n_hl(cpu, b),
                    15,
                    2,
                    f"RES {bit_or_op},(HL)",
                )
            else:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op, r=reg: res_n_r(cpu, b, r),
                    8,
                    2,
                    f"RES {bit_or_op},{_reg_name(reg)}",
                )
        else:
            if reg == 6:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op: set_n_hl(cpu, b),
                    15,
                    2,
                    f"SET {bit_or_op},(HL)",
                )
            else:
                CB_OPCODE_TABLE[opcode] = (
                    lambda cpu, b=bit_or_op, r=reg: set_n_r(cpu, b, r),
                    8,
                    2,
                    f"SET {bit_or_op},{_reg_name(reg)}",
                )


def _build_ed_opcode_table():
    global ED_OPCODE_TABLE
    ED_OPCODE_TABLE[0xA0] = (ldi, 16, 2, "LDI")
    ED_OPCODE_TABLE[0xB0] = (ldir, 16, 2, "LDIR")
    ED_OPCODE_TABLE[0xA8] = (ldd, 16, 2, "LDD")
    ED_OPCODE_TABLE[0xB8] = (lddr, 16, 2, "LDDR")
    ED_OPCODE_TABLE[0xA1] = (cpi, 16, 2, "CPI")
    ED_OPCODE_TABLE[0xB1] = (cpir, 16, 2, "CPIR")
    ED_OPCODE_TABLE[0xA9] = (cpd, 16, 2, "CPD")
    ED_OPCODE_TABLE[0xB9] = (cpdr, 16, 2, "CPDR")
    ED_OPCODE_TABLE[0xA2] = (ini, 16, 2, "INI")
    ED_OPCODE_TABLE[0xB2] = (inir, 16, 2, "INIR")
    ED_OPCODE_TABLE[0xAA] = (ind, 16, 2, "IND")
    ED_OPCODE_TABLE[0xBA] = (indr, 16, 2, "INDR")
    ED_OPCODE_TABLE[0xA3] = (outi, 16, 2, "OUTI")
    ED_OPCODE_TABLE[0xB3] = (otir, 16, 2, "OTIR")
    ED_OPCODE_TABLE[0xAB] = (outd, 16, 2, "OUTD")
    ED_OPCODE_TABLE[0xBB] = (otdr, 16, 2, "OTDR")

    for i, (op, name) in enumerate(
        [(0x4A, "BC"), (0x5A, "DE"), (0x6A, "HL"), (0x7A, "SP")]
    ):
        ED_OPCODE_TABLE[op] = (
            lambda cpu, idx=i: adc_hl_rr(cpu, idx),
            15,
            2,
            f"ADC HL,{name}",
        )
    for i, (op, name) in enumerate(
        [(0x42, "BC"), (0x52, "DE"), (0x62, "HL"), (0x72, "SP")]
    ):
        ED_OPCODE_TABLE[op] = (
            lambda cpu, idx=i: sbc_hl_rr(cpu, idx),
            15,
            2,
            f"SBC HL,{name}",
        )

    for i, (op, name) in enumerate([(0x4B, "BC"), (0x5B, "DE"), (0x7B, "SP")]):
        ED_OPCODE_TABLE[op] = (
            lambda cpu, attr=name: ld_rr_nn_ind(cpu, attr),
            20,
            4,
            f"LD {name},(nn)",
        )
    ED_OPCODE_TABLE[0x6B] = (ld_hl_nn_ed, 20, 4, "LD HL,(nn)")

    for i, (op, name) in enumerate([(0x43, "BC"), (0x53, "DE"), (0x73, "SP")]):
        ED_OPCODE_TABLE[op] = (
            lambda cpu, attr=name: ld_nn_rr(cpu, attr),
            20,
            4,
            f"LD (nn),{name}",
        )
    ED_OPCODE_TABLE[0x63] = (ld_nn_hl_ed, 20, 4, "LD (nn),HL")

    for op in [0x44, 0x54, 0x64, 0x74, 0x4C, 0x5C, 0x6C, 0x7C]:
        ED_OPCODE_TABLE[op] = (neg, 8, 2, "NEG")

    ED_OPCODE_TABLE[0x4D] = (reti, 14, 2, "RETI")
    for op in [0x45, 0x55, 0x65, 0x75, 0x5D, 0x6D, 0x7D]:
        ED_OPCODE_TABLE[op] = (retn, 14, 2, "RETN")

    for op in [0x46, 0x66, 0x4E, 0x6E]:
        ED_OPCODE_TABLE[op] = (im_0, 8, 2, "IM 0")
    for op in [0x56, 0x76]:
        ED_OPCODE_TABLE[op] = (im_1, 8, 2, "IM 1")
    for op in [0x5E, 0x7E]:
        ED_OPCODE_TABLE[op] = (im_2, 8, 2, "IM 2")

    ED_OPCODE_TABLE[0x78] = (in_a_c, 12, 2, "IN A,(C)")
    ED_OPCODE_TABLE[0x40] = (in_b_c, 12, 2, "IN B,(C)")
    ED_OPCODE_TABLE[0x48] = (in_c_c, 12, 2, "IN C,(C)")
    ED_OPCODE_TABLE[0x50] = (in_d_c, 12, 2, "IN D,(C)")
    ED_OPCODE_TABLE[0x58] = (in_e_c, 12, 2, "IN E,(C)")
    ED_OPCODE_TABLE[0x60] = (in_h_c, 12, 2, "IN H,(C)")
    ED_OPCODE_TABLE[0x68] = (in_l_c, 12, 2, "IN L,(C)")
    ED_OPCODE_TABLE[0x70] = (in_f_c, 12, 2, "IN F,(C)")

    ED_OPCODE_TABLE[0x79] = (out_c_a, 12, 2, "OUT (C),A")
    ED_OPCODE_TABLE[0x41] = (out_c_b, 12, 2, "OUT (C),B")
    ED_OPCODE_TABLE[0x49] = (out_c_c, 12, 2, "OUT (C),C")
    ED_OPCODE_TABLE[0x51] = (out_c_d, 12, 2, "OUT (C),D")
    ED_OPCODE_TABLE[0x59] = (out_c_e, 12, 2, "OUT (C),E")
    ED_OPCODE_TABLE[0x61] = (out_c_h, 12, 2, "OUT (C),H")
    ED_OPCODE_TABLE[0x69] = (out_c_l, 12, 2, "OUT (C),L")
    ED_OPCODE_TABLE[0x71] = (out_c_0, 12, 2, "OUT (C),0")

    ED_OPCODE_TABLE[0x47] = (ld_i_a, 9, 2, "LD I,A")
    ED_OPCODE_TABLE[0x4F] = (ld_r_a, 9, 2, "LD R,A")
    ED_OPCODE_TABLE[0x57] = (ld_a_i, 9, 2, "LD A,I")
    ED_OPCODE_TABLE[0x5F] = (ld_a_r, 9, 2, "LD A,R")
    ED_OPCODE_TABLE[0x6F] = (rld, 18, 2, "RLD")
    ED_OPCODE_TABLE[0x67] = (rrd, 18, 2, "RRD")

    # Undefined ED opcodes - treat as NOPs with 8 cycles
    for op in range(0x00, 0x100):
        if op not in ED_OPCODE_TABLE:
            ED_OPCODE_TABLE[op] = (nop, 8, 2, f"NOP* (ED {op:02X})")


def _build_dd_opcode_table():
    global DD_OPCODE_TABLE
    DD_OPCODE_TABLE[0x21] = (lambda cpu: ld_ix_nn(cpu, False), 14, 4, "LD IX,nn")
    DD_OPCODE_TABLE[0x22] = (lambda cpu: ld_nn_ix(cpu, False), 20, 4, "LD (nn),IX")
    DD_OPCODE_TABLE[0x2A] = (lambda cpu: ld_ix_nn_ind(cpu, False), 20, 4, "LD IX,(nn)")
    DD_OPCODE_TABLE[0x23] = (lambda cpu: inc_ix(cpu, False), 10, 2, "INC IX")
    DD_OPCODE_TABLE[0x2B] = (lambda cpu: dec_ix(cpu, False), 10, 2, "DEC IX")
    DD_OPCODE_TABLE[0xF9] = (lambda cpu: ld_sp_ix(cpu, False), 10, 2, "LD SP,IX")
    DD_OPCODE_TABLE[0xE1] = (lambda cpu: pop_ix(cpu, False), 14, 2, "POP IX")
    DD_OPCODE_TABLE[0xE3] = (lambda cpu: ex_sp_ix(cpu, False), 23, 2, "EX (SP),IX")
    DD_OPCODE_TABLE[0xE5] = (lambda cpu: push_ix(cpu, False), 15, 2, "PUSH IX")
    DD_OPCODE_TABLE[0x09] = (lambda cpu: add_ix_rr(cpu, 0, False), 15, 2, "ADD IX,BC")
    DD_OPCODE_TABLE[0x19] = (lambda cpu: add_ix_rr(cpu, 1, False), 15, 2, "ADD IX,DE")
    DD_OPCODE_TABLE[0x29] = (lambda cpu: add_ix_rr(cpu, 2, False), 15, 2, "ADD IX,IX")
    DD_OPCODE_TABLE[0x39] = (lambda cpu: add_ix_rr(cpu, 3, False), 15, 2, "ADD IX,SP")
    DD_OPCODE_TABLE[0x24] = (lambda cpu: inc_ixh(cpu, False), 8, 2, "INC IXH")
    DD_OPCODE_TABLE[0x25] = (lambda cpu: dec_ixh(cpu, False), 8, 2, "DEC IXH")
    DD_OPCODE_TABLE[0x26] = (lambda cpu: ld_ixh_n(cpu, False), 11, 3, "LD IXH,n")
    DD_OPCODE_TABLE[0x2C] = (lambda cpu: inc_ixl(cpu, False), 8, 2, "INC IXL")
    DD_OPCODE_TABLE[0x2D] = (lambda cpu: dec_ixl(cpu, False), 8, 2, "DEC IXL")
    DD_OPCODE_TABLE[0x2E] = (lambda cpu: ld_ixl_n(cpu, False), 11, 3, "LD IXL,n")

    DD_OPCODE_TABLE[0x44] = (lambda cpu: ld_r_ixh(cpu, 0, False), 8, 2, "LD B,IXH")
    DD_OPCODE_TABLE[0x4C] = (lambda cpu: ld_r_ixh(cpu, 1, False), 8, 2, "LD C,IXH")
    DD_OPCODE_TABLE[0x54] = (lambda cpu: ld_r_ixh(cpu, 2, False), 8, 2, "LD D,IXH")
    DD_OPCODE_TABLE[0x5C] = (lambda cpu: ld_r_ixh(cpu, 3, False), 8, 2, "LD E,IXH")
    DD_OPCODE_TABLE[0x45] = (lambda cpu: ld_r_ixl(cpu, 0, False), 8, 2, "LD B,IXL")
    DD_OPCODE_TABLE[0x4D] = (lambda cpu: ld_r_ixl(cpu, 1, False), 8, 2, "LD C,IXL")
    DD_OPCODE_TABLE[0x55] = (lambda cpu: ld_r_ixl(cpu, 2, False), 8, 2, "LD D,IXL")
    DD_OPCODE_TABLE[0x5D] = (lambda cpu: ld_r_ixl(cpu, 3, False), 8, 2, "LD E,IXL")

    DD_OPCODE_TABLE[0x60] = (lambda cpu: ld_ixh_r(cpu, 0, False), 8, 2, "LD IXH,B")
    DD_OPCODE_TABLE[0x61] = (lambda cpu: ld_ixh_r(cpu, 1, False), 8, 2, "LD IXH,C")
    DD_OPCODE_TABLE[0x62] = (lambda cpu: ld_ixh_r(cpu, 2, False), 8, 2, "LD IXH,D")
    DD_OPCODE_TABLE[0x63] = (lambda cpu: ld_ixh_r(cpu, 3, False), 8, 2, "LD IXH,E")
    DD_OPCODE_TABLE[0x64] = (lambda cpu: ld_ixh_ixl(cpu, False), 8, 2, "LD IXH,IXH")
    DD_OPCODE_TABLE[0x65] = (lambda cpu: ld_ixh_ixl(cpu, False), 8, 2, "LD IXH,IXL")
    DD_OPCODE_TABLE[0x67] = (lambda cpu: ld_ixh_r(cpu, 7, False), 8, 2, "LD IXH,A")
    DD_OPCODE_TABLE[0x68] = (lambda cpu: ld_ixl_r(cpu, 0, False), 8, 2, "LD IXL,B")
    DD_OPCODE_TABLE[0x69] = (lambda cpu: ld_ixl_r(cpu, 1, False), 8, 2, "LD IXL,C")
    DD_OPCODE_TABLE[0x6A] = (lambda cpu: ld_ixl_r(cpu, 2, False), 8, 2, "LD IXL,D")
    DD_OPCODE_TABLE[0x6B] = (lambda cpu: ld_ixl_r(cpu, 3, False), 8, 2, "LD IXL,E")
    DD_OPCODE_TABLE[0x6C] = (lambda cpu: ld_ixl_ixh(cpu, False), 8, 2, "LD IXL,IXH")
    DD_OPCODE_TABLE[0x6D] = (lambda cpu: ld_ixl_ixl(cpu, False), 8, 2, "LD IXL,IXL")
    DD_OPCODE_TABLE[0x6F] = (lambda cpu: ld_ixl_r(cpu, 7, False), 8, 2, "LD IXL,A")
    DD_OPCODE_TABLE[0x7C] = (lambda cpu: ld_a_ixh(cpu, False), 8, 2, "LD A,IXH")
    DD_OPCODE_TABLE[0x7D] = (lambda cpu: ld_a_ixl(cpu, False), 8, 2, "LD A,IXL")

    DD_OPCODE_TABLE[0x34] = (lambda cpu: inc_ixd(cpu, False), 23, 3, "INC (IX+d)")
    DD_OPCODE_TABLE[0x35] = (lambda cpu: dec_ixd(cpu, False), 23, 3, "DEC (IX+d)")
    DD_OPCODE_TABLE[0x36] = (lambda cpu: ld_ixd_n(cpu, False), 19, 4, "LD (IX+d),n")
    DD_OPCODE_TABLE[0x84] = (lambda cpu: add_a_ixh(cpu, False), 8, 2, "ADD A,IXH")
    DD_OPCODE_TABLE[0x85] = (lambda cpu: add_a_ixl(cpu, False), 8, 2, "ADD A,IXL")
    DD_OPCODE_TABLE[0x86] = (lambda cpu: add_a_ixd(cpu, False), 19, 3, "ADD A,(IX+d)")
    DD_OPCODE_TABLE[0x8C] = (lambda cpu: adc_a_ixh(cpu, False), 8, 2, "ADC A,IXH")
    DD_OPCODE_TABLE[0x8D] = (lambda cpu: adc_a_ixl(cpu, False), 8, 2, "ADC A,IXL")
    DD_OPCODE_TABLE[0x8E] = (lambda cpu: adc_a_ixd(cpu, False), 19, 3, "ADC A,(IX+d)")
    DD_OPCODE_TABLE[0x94] = (lambda cpu: sub_ixh(cpu, False), 8, 2, "SUB IXH")
    DD_OPCODE_TABLE[0x95] = (lambda cpu: sub_ixl(cpu, False), 8, 2, "SUB IXL")
    DD_OPCODE_TABLE[0x96] = (lambda cpu: sub_ixd(cpu, False), 19, 3, "SUB (IX+d)")
    DD_OPCODE_TABLE[0x9C] = (lambda cpu: sbc_a_ixh(cpu, False), 8, 2, "SBC A,IXH")
    DD_OPCODE_TABLE[0x9D] = (lambda cpu: sbc_a_ixl(cpu, False), 8, 2, "SBC A,IXL")
    DD_OPCODE_TABLE[0x9E] = (lambda cpu: sbc_a_ixd(cpu, False), 19, 3, "SBC A,(IX+d)")
    DD_OPCODE_TABLE[0xA4] = (lambda cpu: and_ixh(cpu, False), 8, 2, "AND IXH")
    DD_OPCODE_TABLE[0xA5] = (lambda cpu: and_ixl(cpu, False), 8, 2, "AND IXL")
    DD_OPCODE_TABLE[0xA6] = (lambda cpu: and_ixd(cpu, False), 19, 3, "AND (IX+d)")
    DD_OPCODE_TABLE[0xAC] = (lambda cpu: xor_ixh(cpu, False), 8, 2, "XOR IXH")
    DD_OPCODE_TABLE[0xAD] = (lambda cpu: xor_ixl(cpu, False), 8, 2, "XOR IXL")
    DD_OPCODE_TABLE[0xAE] = (lambda cpu: xor_ixd(cpu, False), 19, 3, "XOR (IX+d)")
    DD_OPCODE_TABLE[0xB4] = (lambda cpu: or_ixh(cpu, False), 8, 2, "OR IXH")
    DD_OPCODE_TABLE[0xB5] = (lambda cpu: or_ixl(cpu, False), 8, 2, "OR IXL")
    DD_OPCODE_TABLE[0xB6] = (lambda cpu: or_ixd(cpu, False), 19, 3, "OR (IX+d)")
    DD_OPCODE_TABLE[0xBC] = (lambda cpu: cp_ixh(cpu, False), 8, 2, "CP IXH")
    DD_OPCODE_TABLE[0xBD] = (lambda cpu: cp_ixl(cpu, False), 8, 2, "CP IXL")
    DD_OPCODE_TABLE[0xBE] = (lambda cpu: cp_ixd(cpu, False), 19, 3, "CP (IX+d)")
    DD_OPCODE_TABLE[0xE9] = (lambda cpu: jp_ix(cpu, False), 8, 2, "JP (IX)")

    for reg in range(8):
        if reg != 6:
            DD_OPCODE_TABLE[0x46 | (reg << 3)] = (
                lambda cpu, r=reg: ld_r_ixd(cpu, r, False),
                19,
                3,
                f"LD {_reg_name(reg)},(IX+d)",
            )
            DD_OPCODE_TABLE[0x70 | reg] = (
                lambda cpu, r=reg: ld_ixd_r(cpu, r, False),
                19,
                3,
                f"LD (IX+d),{_reg_name(reg)}",
            )


def _build_fd_opcode_table():
    global FD_OPCODE_TABLE
    FD_OPCODE_TABLE[0x21] = (lambda cpu: ld_ix_nn(cpu, True), 14, 4, "LD IY,nn")
    FD_OPCODE_TABLE[0x22] = (lambda cpu: ld_nn_ix(cpu, True), 20, 4, "LD (nn),IY")
    FD_OPCODE_TABLE[0x2A] = (lambda cpu: ld_ix_nn_ind(cpu, True), 20, 4, "LD IY,(nn)")
    FD_OPCODE_TABLE[0x23] = (lambda cpu: inc_ix(cpu, True), 10, 2, "INC IY")
    FD_OPCODE_TABLE[0x2B] = (lambda cpu: dec_ix(cpu, True), 10, 2, "DEC IY")
    FD_OPCODE_TABLE[0xF9] = (lambda cpu: ld_sp_ix(cpu, True), 10, 2, "LD SP,IY")
    FD_OPCODE_TABLE[0xE1] = (lambda cpu: pop_ix(cpu, True), 14, 2, "POP IY")
    FD_OPCODE_TABLE[0xE3] = (lambda cpu: ex_sp_ix(cpu, True), 23, 2, "EX (SP),IY")
    FD_OPCODE_TABLE[0xE5] = (lambda cpu: push_ix(cpu, True), 15, 2, "PUSH IY")
    FD_OPCODE_TABLE[0x09] = (lambda cpu: add_ix_rr(cpu, 0, True), 15, 2, "ADD IY,BC")
    FD_OPCODE_TABLE[0x19] = (lambda cpu: add_ix_rr(cpu, 1, True), 15, 2, "ADD IY,DE")
    FD_OPCODE_TABLE[0x29] = (lambda cpu: add_ix_rr(cpu, 2, True), 15, 2, "ADD IY,IY")
    FD_OPCODE_TABLE[0x39] = (lambda cpu: add_ix_rr(cpu, 3, True), 15, 2, "ADD IY,SP")
    FD_OPCODE_TABLE[0x24] = (lambda cpu: inc_ixh(cpu, True), 8, 2, "INC IYH")
    FD_OPCODE_TABLE[0x25] = (lambda cpu: dec_ixh(cpu, True), 8, 2, "DEC IYH")
    FD_OPCODE_TABLE[0x26] = (lambda cpu: ld_ixh_n(cpu, True), 11, 3, "LD IYH,n")
    FD_OPCODE_TABLE[0x2C] = (lambda cpu: inc_ixl(cpu, True), 8, 2, "INC IYL")
    FD_OPCODE_TABLE[0x2D] = (lambda cpu: dec_ixl(cpu, True), 8, 2, "DEC IYL")
    FD_OPCODE_TABLE[0x2E] = (lambda cpu: ld_ixl_n(cpu, True), 11, 3, "LD IYL,n")

    FD_OPCODE_TABLE[0x44] = (lambda cpu: ld_r_ixh(cpu, 0, True), 8, 2, "LD B,IYH")
    FD_OPCODE_TABLE[0x4C] = (lambda cpu: ld_r_ixh(cpu, 1, True), 8, 2, "LD C,IYH")
    FD_OPCODE_TABLE[0x54] = (lambda cpu: ld_r_ixh(cpu, 2, True), 8, 2, "LD D,IYH")
    FD_OPCODE_TABLE[0x5C] = (lambda cpu: ld_r_ixh(cpu, 3, True), 8, 2, "LD E,IYH")
    FD_OPCODE_TABLE[0x45] = (lambda cpu: ld_r_ixl(cpu, 0, True), 8, 2, "LD B,IYL")
    FD_OPCODE_TABLE[0x4D] = (lambda cpu: ld_r_ixl(cpu, 1, True), 8, 2, "LD C,IYL")
    FD_OPCODE_TABLE[0x55] = (lambda cpu: ld_r_ixl(cpu, 2, True), 8, 2, "LD D,IYL")
    FD_OPCODE_TABLE[0x5D] = (lambda cpu: ld_r_ixl(cpu, 3, True), 8, 2, "LD E,IYL")

    FD_OPCODE_TABLE[0x60] = (lambda cpu: ld_ixh_r(cpu, 0, True), 8, 2, "LD IYH,B")
    FD_OPCODE_TABLE[0x61] = (lambda cpu: ld_ixh_r(cpu, 1, True), 8, 2, "LD IYH,C")
    FD_OPCODE_TABLE[0x62] = (lambda cpu: ld_ixh_r(cpu, 2, True), 8, 2, "LD IYH,D")
    FD_OPCODE_TABLE[0x63] = (lambda cpu: ld_ixh_r(cpu, 3, True), 8, 2, "LD IYH,E")
    FD_OPCODE_TABLE[0x64] = (lambda cpu: ld_ixh_ixl(cpu, True), 8, 2, "LD IYH,IYH")
    FD_OPCODE_TABLE[0x65] = (lambda cpu: ld_ixh_ixl(cpu, True), 8, 2, "LD IYH,IYL")
    FD_OPCODE_TABLE[0x67] = (lambda cpu: ld_ixh_r(cpu, 7, True), 8, 2, "LD IYH,A")
    FD_OPCODE_TABLE[0x68] = (lambda cpu: ld_ixl_r(cpu, 0, True), 8, 2, "LD IYL,B")
    FD_OPCODE_TABLE[0x69] = (lambda cpu: ld_ixl_r(cpu, 1, True), 8, 2, "LD IYL,C")
    FD_OPCODE_TABLE[0x6A] = (lambda cpu: ld_ixl_r(cpu, 2, True), 8, 2, "LD IYL,D")
    FD_OPCODE_TABLE[0x6B] = (lambda cpu: ld_ixl_r(cpu, 3, True), 8, 2, "LD IYL,E")
    FD_OPCODE_TABLE[0x6C] = (lambda cpu: ld_ixl_ixh(cpu, True), 8, 2, "LD IYL,IYH")
    FD_OPCODE_TABLE[0x6D] = (lambda cpu: ld_ixl_ixl(cpu, True), 8, 2, "LD IYL,IYL")
    FD_OPCODE_TABLE[0x6F] = (lambda cpu: ld_ixl_r(cpu, 7, True), 8, 2, "LD IYL,A")
    FD_OPCODE_TABLE[0x7C] = (lambda cpu: ld_a_ixh(cpu, True), 8, 2, "LD A,IYH")
    FD_OPCODE_TABLE[0x7D] = (lambda cpu: ld_a_ixl(cpu, True), 8, 2, "LD A,IYL")

    FD_OPCODE_TABLE[0x34] = (lambda cpu: inc_ixd(cpu, True), 23, 3, "INC (IY+d)")
    FD_OPCODE_TABLE[0x35] = (lambda cpu: dec_ixd(cpu, True), 23, 3, "DEC (IY+d)")
    FD_OPCODE_TABLE[0x36] = (lambda cpu: ld_ixd_n(cpu, True), 19, 4, "LD (IY+d),n")
    FD_OPCODE_TABLE[0x84] = (lambda cpu: add_a_ixh(cpu, True), 8, 2, "ADD A,IYH")
    FD_OPCODE_TABLE[0x85] = (lambda cpu: add_a_ixl(cpu, True), 8, 2, "ADD A,IYL")
    FD_OPCODE_TABLE[0x8C] = (lambda cpu, iy=True: adc_a_ixh(cpu, iy), 8, 2, "ADC A,IYH")
    FD_OPCODE_TABLE[0x8D] = (lambda cpu, iy=True: adc_a_ixl(cpu, iy), 8, 2, "ADC A,IYL")
    FD_OPCODE_TABLE[0x94] = (lambda cpu: sub_ixh(cpu, True), 8, 2, "SUB IYH")
    FD_OPCODE_TABLE[0x95] = (lambda cpu: sub_ixl(cpu, True), 8, 2, "SUB IYL")
    FD_OPCODE_TABLE[0x9C] = (lambda cpu, iy=True: sbc_a_ixh(cpu, iy), 8, 2, "SBC A,IYH")
    FD_OPCODE_TABLE[0x9D] = (lambda cpu, iy=True: sbc_a_ixl(cpu, iy), 8, 2, "SBC A,IYL")
    FD_OPCODE_TABLE[0xA4] = (lambda cpu: and_ixh(cpu, True), 8, 2, "AND IYH")
    FD_OPCODE_TABLE[0xA5] = (lambda cpu: and_ixl(cpu, True), 8, 2, "AND IYL")
    FD_OPCODE_TABLE[0xAC] = (lambda cpu: xor_ixh(cpu, True), 8, 2, "XOR IYH")
    FD_OPCODE_TABLE[0xAD] = (lambda cpu: xor_ixl(cpu, True), 8, 2, "XOR IYL")
    FD_OPCODE_TABLE[0xB4] = (lambda cpu: or_ixh(cpu, True), 8, 2, "OR IYH")
    FD_OPCODE_TABLE[0xB5] = (lambda cpu: or_ixl(cpu, True), 8, 2, "OR IYL")
    FD_OPCODE_TABLE[0xBC] = (lambda cpu: cp_ixh(cpu, True), 8, 2, "CP IYH")
    FD_OPCODE_TABLE[0xBD] = (lambda cpu: cp_ixl(cpu, True), 8, 2, "CP IYL")
    FD_OPCODE_TABLE[0x86] = (lambda cpu: add_a_ixd(cpu, True), 19, 3, "ADD A,(IY+d)")
    FD_OPCODE_TABLE[0x8E] = (lambda cpu: adc_a_ixd(cpu, True), 19, 3, "ADC A,(IY+d)")
    FD_OPCODE_TABLE[0x96] = (lambda cpu: sub_ixd(cpu, True), 19, 3, "SUB (IY+d)")
    FD_OPCODE_TABLE[0x9E] = (lambda cpu: sbc_a_ixd(cpu, True), 19, 3, "SBC A,(IY+d)")
    FD_OPCODE_TABLE[0xA6] = (lambda cpu: and_ixd(cpu, True), 19, 3, "AND (IY+d)")
    FD_OPCODE_TABLE[0xAE] = (lambda cpu: xor_ixd(cpu, True), 19, 3, "XOR (IY+d)")
    FD_OPCODE_TABLE[0xB6] = (lambda cpu: or_ixd(cpu, True), 19, 3, "OR (IY+d)")
    FD_OPCODE_TABLE[0xBE] = (lambda cpu: cp_ixd(cpu, True), 19, 3, "CP (IY+d)")
    FD_OPCODE_TABLE[0xE9] = (lambda cpu: jp_ix(cpu, True), 8, 2, "JP (IY)")

    for reg in range(8):
        if reg != 6:
            FD_OPCODE_TABLE[0x46 | (reg << 3)] = (
                lambda cpu, r=reg: ld_r_ixd(cpu, r, True),
                19,
                3,
                f"LD {_reg_name(reg)},(IY+d)",
            )
            FD_OPCODE_TABLE[0x70 | reg] = (
                lambda cpu, r=reg: ld_ixd_r(cpu, r, True),
                19,
                3,
                f"LD (IY+d),{_reg_name(reg)}",
            )


def _build_ixycb_opcode_table(is_iy: bool) -> dict:
    table: dict = {}
    prefix = "IY" if is_iy else "IX"
    rot_names = ["RLC", "RRC", "RL", "RR", "SLA", "SRA", "SLL", "SRL"]
    for opcode in range(256):
        op_type = (opcode >> 6) & 0x03
        op_idx = (opcode >> 3) & 0x07
        reg = opcode & 0x07
        if op_type == 0:
            suffix = "" if reg == 6 else f",{_reg_name(reg)}"
            table[opcode] = (
                lambda cpu, r=reg, o=op_idx, iy=is_iy: _ixycb_rot(cpu, r, o, iy),
                23,
                4,
                f"{rot_names[op_idx]} ({prefix}+d){suffix}",
            )
        elif op_type == 1:
            table[opcode] = (
                lambda cpu, b=op_idx, iy=is_iy: _ixycb_bit_n(cpu, b, iy),
                20,
                4,
                f"BIT {op_idx},({prefix}+d)",
            )
        elif op_type == 2:
            suffix = "" if reg == 6 else f",{_reg_name(reg)}"
            table[opcode] = (
                lambda cpu, b=op_idx, r=reg, iy=is_iy: _ixycb_res_n(cpu, b, r, iy),
                23,
                4,
                f"RES {op_idx},({prefix}+d){suffix}",
            )
        else:
            suffix = "" if reg == 6 else f",{_reg_name(reg)}"
            table[opcode] = (
                lambda cpu, b=op_idx, r=reg, iy=is_iy: _ixycb_set_n(cpu, b, r, iy),
                23,
                4,
                f"SET {op_idx},({prefix}+d){suffix}",
            )
    return table


DDCB_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = (
    _build_ixycb_opcode_table(False)
)
FDCB_OPCODE_TABLE: dict[int, tuple[Callable, int, int, str]] = (
    _build_ixycb_opcode_table(True)
)

_build_base_opcode_table()
_build_cb_opcode_table()
_build_ed_opcode_table()
_build_dd_opcode_table()
_build_fd_opcode_table()


def get_base_opcode(opcode: int) -> Optional[tuple]:
    return BASE_OPCODE_TABLE.get(opcode)


def get_cb_opcode(opcode: int) -> Optional[tuple]:
    return CB_OPCODE_TABLE.get(opcode)


def get_ed_opcode(opcode: int) -> Optional[tuple]:
    return ED_OPCODE_TABLE.get(opcode)


def get_dd_opcode(opcode: int) -> Optional[tuple]:
    return DD_OPCODE_TABLE.get(opcode)


def get_fd_opcode(opcode: int) -> Optional[tuple]:
    return FD_OPCODE_TABLE.get(opcode)


def get_ddcb_opcode(opcode: int) -> Optional[tuple]:
    return DDCB_OPCODE_TABLE.get(opcode)


def get_fdcb_opcode(opcode: int) -> Optional[tuple]:
    return FDCB_OPCODE_TABLE.get(opcode)
