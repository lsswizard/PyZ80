#!/usr/bin/env python3
"""
CPU Usage Benchmark for Z80 Emulator
Tests the emulator's CPU usage without requiring SDL display
"""

import time
import sys
import os

sys.path.insert(0, "emulator")
sys.path.insert(0, ".")

# Disable SDL for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from machines.spectrum.models.spectrum48 import Spectrum48
from machines.spectrum.models.spectrum128 import Spectrum128


def benchmark_cpu(model_class, rom_path, num_frames=100):
    """Run emulator for N frames and measure time"""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model_class.__name__}")
    print(f"ROM: {rom_path}")
    print(f"Frames: {num_frames}")
    print(f"{'=' * 60}")

    # Create machine
    machine = model_class(rom_path)
    machine.reset()

    # Get timing info
    timing = machine.get_timing_info()
    tstates_per_frame = timing.t_states_per_frame
    target_fps = timing.cpu_clock_hz / tstates_per_frame

    print(f"T-states/frame: {tstates_per_frame}")
    print(f"Target FPS: {target_fps:.2f}")

    # Warmup - run a few frames first
    for _ in range(5):
        machine.step_frame()

    # Benchmark
    start_time = time.perf_counter()

    for i in range(num_frames):
        machine.step_frame()

        # Progress indicator every 10 frames
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            current_fps = (i + 1) / elapsed
            print(f"  Frame {i + 1}/{num_frames}: {current_fps:.1f} emulated fps")

    end_time = time.perf_counter()

    # Calculate results
    total_time = end_time - start_time
    emulated_fps = num_frames / total_time
    real_fps = 1.0 / total_time * num_frames

    # Calculate how fast the emulator runs compared to real hardware
    speed_multiplier = emulated_fps / target_fps

    # CPU usage estimation
    # If running at 1x speed, we'd expect ~20ms per frame (50fps)
    # Any time less than that means we're running faster than real-time
    time_per_frame_ms = (total_time / num_frames) * 1000

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Emulated FPS: {emulated_fps:.1f}")
    print(f"  Target FPS: {target_fps:.1f}")
    print(f"  Speed multiplier: {speed_multiplier:.2f}x")
    print(f"  Time per frame: {time_per_frame_ms:.2f}ms")
    print(f"{'=' * 60}")

    # CPU usage interpretation
    # On a typical modern CPU, 100% of one core = full speed emulation at target FPS
    # If we run at 10x real-time on a fast CPU, that's still efficient
    if speed_multiplier >= 1.0:
        print(f"\n✓ Emulator runs at {speed_multiplier:.1f}x real-time speed")
        print(f"  CPU usage estimate: ~{(100 / speed_multiplier):.0f}% of one core")
    else:
        print(f"\n✗ Emulator runs at {speed_multiplier:.2f}x - slower than real-time")
        print(f"  This may cause emulation to run slow")

    return {
        "emulated_fps": emulated_fps,
        "target_fps": target_fps,
        "speed_multiplier": speed_multiplier,
        "time_per_frame_ms": time_per_frame_ms,
        "total_time": total_time,
    }


def main():
    print("Z80 Emulator CPU Benchmark")
    print("=" * 60)

    # Find ROMs
    import glob

    roms = {
        "48k": "emulator/machines/spectrum/roms/48.rom",
        "128k": "emulator/machines/spectrum/roms/128.rom",
    }

    results = {}

    # Test 48K
    if os.path.exists(roms["48k"]):
        results["48k"] = benchmark_cpu(Spectrum48, roms["48k"], num_frames=100)

    # Test 128K
    if os.path.exists(roms["128k"]):
        results["128k"] = benchmark_cpu(Spectrum128, roms["128k"], num_frames=100)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, res in results.items():
        print(
            f"{model}: {res['speed_multiplier']:.1f}x speed ({res['emulated_fps']:.0f} fps)"
        )


if __name__ == "__main__":
    main()
