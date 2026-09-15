"""Intel i915 GPU utilization through the kernel perf PMU."""

import ctypes
import errno
import os
import platform
import re
import struct
import threading
from typing import Optional


class PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("bp_addr", ctypes.c_uint64),
    ]


class IntelGpuMetrics:
    """Read i915 rcs0 busy time without depending on the perf command."""

    PMU_PATH = "/sys/bus/event_source/devices/i915"
    EVENT_NAME = "rcs0-busy"
    PERF_SAMPLE_IDENTIFIER = 1 << 16
    PERF_FORMAT_TOTAL_TIME_ENABLED = 1
    PERF_FORMAT_TOTAL_TIME_RUNNING = 2
    PERF_EVENT_IOC_ENABLE = 0x2400
    PERF_FLAG_FD_CLOEXEC = 1
    PERF_ATTR_DISABLED = 1 << 0
    PERF_ATTR_INHERIT = 1 << 1
    SYSCALL_NUMBERS = {
        "x86_64": 298,
        "i386": 336,
        "i686": 336,
        "x86": 336,
        "aarch64": 241,
        "armv7l": 364,
        "ppc64le": 319,
        "s390x": 331,
        "riscv64": 241,
    }

    def __init__(self, logger):
        self._logger = logger
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._fd = -1
        self._lock = threading.Lock()
        self._has_previous_sample = False
        self._previous_busy = 0
        self._previous_time_enabled = 0
        self._previous_time_running = 0
        self._read_error_logged = False
        self.available = False

        try:
            self._fd = self._open_event()
            self.available = True
            self._logger.info("Intel i915 GPU utilization monitoring enabled")
        except (OSError, ValueError) as error:
            self._logger.warning("Intel i915 GPU monitoring unavailable: %s", error)

    @staticmethod
    def parse_cpu_mask(mask: str) -> Optional[int]:
        """Return the first CPU represented by a Linux hexadecimal cpumask."""
        value = mask.strip()
        if not value:
            raise ValueError("empty PMU CPU mask")
        if value == "0":
            return 0

        # Some kernel interfaces use a CPU-list spelling; accept it as well.
        if any(character in value for character in "- "):
            first = value.split(",", 1)[0].strip().split("-", 1)[0]
            return int(first)

        groups = value.split(",")
        for group_index, group in enumerate(reversed(groups)):
            bits = int(group, 16)
            if bits:
                return group_index * 32 + ((bits & -bits).bit_length() - 1)
        raise ValueError("PMU CPU mask does not select a CPU")

    @staticmethod
    def parse_event_config(definition: str) -> int:
        match = re.search(r"(?:^|\s)config=0x([0-9a-fA-F]+)(?:\s|$)", definition.strip())
        if not match:
            raise ValueError("i915 event has no hexadecimal config")
        return int(match.group(1), 16)

    def _read_text(self, path: str) -> str:
        with open(path, "r", encoding="ascii") as file:
            return file.read().strip()

    def _open_event(self) -> int:
        syscall_number = self.SYSCALL_NUMBERS.get(platform.machine())
        if syscall_number is None:
            raise OSError(errno.ENOSYS, f"unsupported architecture: {platform.machine()}")

        pmu_type = int(self._read_text(os.path.join(self.PMU_PATH, "type")))
        event_path = os.path.join(self.PMU_PATH, "events", self.EVENT_NAME)
        unit = self._read_text(f"{event_path}.unit")
        if unit != "ns":
            raise ValueError(f"unexpected i915 event unit: {unit!r}")
        event_config = self.parse_event_config(self._read_text(event_path))
        pmu_cpu = self.parse_cpu_mask(self._read_text(os.path.join(self.PMU_PATH, "cpumask")))
        if pmu_cpu is None:
            raise ValueError("i915 PMU has no monitoring CPU")

        # The GUI is intentionally not allowed to open perf events directly:
        # the daemon already runs with the privileges required by perf_event_open.
        attributes = PerfEventAttr(
            type=pmu_type,
            size=ctypes.sizeof(PerfEventAttr),
            config=event_config,
            sample_type=self.PERF_SAMPLE_IDENTIFIER,
            read_format=(
                self.PERF_FORMAT_TOTAL_TIME_ENABLED
                | self.PERF_FORMAT_TOTAL_TIME_RUNNING
            ),
            flags=self.PERF_ATTR_DISABLED | self.PERF_ATTR_INHERIT,
        )
        self._libc.syscall.restype = ctypes.c_long
        fd = self._libc.syscall(
            syscall_number,
            ctypes.byref(attributes),
            -1,
            pmu_cpu,
            -1,
            self.PERF_FLAG_FD_CLOEXEC,
        )
        if fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))

        if self._libc.ioctl(fd, self.PERF_EVENT_IOC_ENABLE, 0) != 0:
            error_number = ctypes.get_errno()
            os.close(fd)
            raise OSError(error_number, os.strerror(error_number))
        return fd

    def get_usage(self) -> float:
        """Return the percentage of enabled time spent busy since the last read."""
        if not self.available:
            return 0.0

        try:
            with self._lock:
                data = os.read(self._fd, 24)
        except OSError as error:
            if not self._read_error_logged:
                self._logger.warning("Unable to read Intel i915 GPU metrics: %s", error)
                self._read_error_logged = True
            return 0.0

        if len(data) != 24:
            return 0.0

        busy, time_enabled, time_running = struct.unpack("3Q", data)
        with self._lock:
            if not self._has_previous_sample:
                # A single cumulative counter read has no interval to measure.
                self._previous_busy = busy
                self._previous_time_enabled = time_enabled
                self._previous_time_running = time_running
                self._has_previous_sample = True
                return 0.0

            busy_delta = busy - self._previous_busy
            enabled_delta = time_enabled - self._previous_time_enabled
            running_delta = time_running - self._previous_time_running
            self._previous_busy = busy
            self._previous_time_enabled = time_enabled
            self._previous_time_running = time_running

        if enabled_delta == 0 or running_delta == 0:
            return 0.0

        scaled_busy = busy_delta * enabled_delta / running_delta
        return max(0.0, min(100.0, scaled_busy / enabled_delta * 100.0))

    def close(self):
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
                self.available = False

    def __enter__(self):
        return self

    def __exit__(self, _exception_type, _exception_value, _traceback):
        self.close()
