"""
Timing Module
T-state timing and frame synchronization utilities.
"""

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class TimingInfo:
    """Timing information for the emulator.

    All machines should provide their own TimingInfo with appropriate values.
    """

    t_states_per_frame: int = 0
    t_states_per_line: int = 0
    lines_per_frame: int = 0
    cpu_clock_hz: int = 0

    @property
    def frame_rate(self) -> float:
        """Derived frame rate."""
        if self.t_states_per_frame == 0:
            return 0.0
        return self.cpu_clock_hz / self.t_states_per_frame

    @property
    def cpu_clock_mhz(self) -> float:
        return self.cpu_clock_hz / 1_000_000

    @property
    def frame_time_ms(self) -> float:
        rate = self.frame_rate
        if rate == 0:
            return 0.0
        return 1000.0 / rate

    @property
    def line_time_us(self) -> float:
        return (self.t_states_per_line / self.cpu_clock_hz) * 1_000_000

    @property
    def t_states_per_ms(self) -> float:
        return self.cpu_clock_hz / 1000.0

    def __post_init__(self):
        """Validate timing consistency."""
        if (
            self.t_states_per_frame > 0
            and self.t_states_per_line > 0
            and self.lines_per_frame > 0
        ):
            expected = self.t_states_per_line * self.lines_per_frame
            if self.t_states_per_frame != expected:
                raise ValueError(
                    f"Timing inconsistency: t_states_per_line * lines_per_frame = "
                    f"{expected}, but t_states_per_frame = {self.t_states_per_frame}"
                )


class TimingEngine:
    """
    Cycle-accurate timing engine for the Z80 emulator.

    Coordinates:
    - CPU T-state counting
    - Frame timing
    - Interrupt generation
    """

    def __init__(self, timing: Optional[TimingInfo] = None):
        self.timing = timing
        self.t_states = 0
        self.frame_count = 0

        self.on_frame_complete: Optional[Callable[[int], None]] = None

        self.frame_skip = 0
        self.current_skip = 0

    def reset(self) -> None:
        """Reset timing state"""
        self.t_states = 0
        self.frame_count = 0
        self.current_skip = 0

    def advance(self, cycles: int) -> int:
        """
        Advance timing by specified cycles.

        Returns:
            Number of frames completed
        """
        if self.timing is None or self.timing.t_states_per_frame == 0:
            return 0

        self.t_states += cycles
        frames_elapsed = self.t_states // self.timing.t_states_per_frame

        if frames_elapsed > 0:
            self.t_states = self.t_states % self.timing.t_states_per_frame
            self.frame_count += frames_elapsed

            if self.on_frame_complete:
                for _ in range(frames_elapsed):
                    self.on_frame_complete(1)

        return frames_elapsed

    def get_current_frame(self) -> int:
        """Get current frame number"""
        return self.frame_count

    def get_t_states_in_frame(self) -> int:
        """Get T-states elapsed in current frame"""
        return self.t_states

    def get_current_scanline(self) -> int:
        """Get current scanline (0 to lines_per_frame-1)"""
        if self.timing is None or self.timing.t_states_per_line == 0:
            return 0
        return self.t_states // self.timing.t_states_per_line

    def get_t_states_in_line(self) -> int:
        """Get T-states elapsed in current scanline"""
        if self.timing is None:
            return 0
        return self.t_states % self.timing.t_states_per_line

    def is_interrupt_due(self) -> bool:
        """Check if interrupt should fire (at start of frame)"""
        if self.timing is None:
            return False
        return self.t_states < self.timing.t_states_per_line

    def get_cycles_to_next_interrupt(self) -> int:
        """Get cycles until next interrupt."""
        if self.timing is None or self.timing.t_states_per_frame == 0:
            return 0
        return self.timing.t_states_per_frame - self.t_states

    def set_frame_skip(self, skip: int) -> None:
        """Set frame skip count (0 = no skip)"""
        self.frame_skip = max(0, skip)

    def advance_frame_skip(self) -> bool:
        """
        Advance frame skip counter and check if current frame should be rendered.

        Note: This method has a side effect (advances the counter).
        Returns True if this frame should be rendered.
        """
        if self.frame_skip == 0:
            return True
        self.current_skip = (self.current_skip + 1) % (self.frame_skip + 1)
        return self.current_skip == 0

    def should_render_frame(self) -> bool:
        """Check if current frame should be rendered without advancing counter."""
        if self.frame_skip == 0:
            return True
        return self.current_skip == 0


def t_states_to_ms(t_states: int, clock_hz: int) -> float:
    """Convert T-states to milliseconds."""
    return (t_states / clock_hz) * 1000.0


def ms_to_t_states(ms: float, clock_hz: int) -> int:
    """Convert milliseconds to T-states."""
    return int((ms / 1000.0) * clock_hz)


def t_states_to_us(t_states: int, clock_hz: int) -> float:
    """Convert T-states to microseconds."""
    return (t_states / clock_hz) * 1_000_000.0
