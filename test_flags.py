import sys
sys.path.insert(0, '.')
from core.cpu import Z80CPU
from core.bus import SimpleBus

bus = SimpleBus()
# Program: DEC C at address 0
bus.memory[0] = 0x0D
cpu = Z80CPU(bus)
cpu.reset()
cpu.regs.PC = 0
print('Before: C=', cpu.regs.C, 'Z=', cpu.regs.Z, 'F=', cpu.regs.F)
cpu.step()
print('After: C=', cpu.regs.C, 'Z=', cpu.regs.Z, 'F=', cpu.regs.F)
print('Cycles:', cpu.cycles, 'Instructions:', cpu.instruction_count)
