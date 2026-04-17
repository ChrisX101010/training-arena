"""GPU Manager - Temperature monitoring and adaptive throttling."""

import subprocess
import time
import threading
from typing import Optional


class GPUManager:
    def __init__(self, target_temp: int = 70, max_temp: int = 78, check_interval: float = 2.0):
        self.target_temp = target_temp
        self.max_temp = max_temp
        self.check_interval = check_interval
        self.throttle_factor = 1.0
        self._monitoring = False
        self._thread = None
        self._lock = threading.Lock()

    def start_monitoring(self):
        if self._monitoring:
            return
        self._monitoring = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"🌡️ GPU monitor started (target {self.target_temp}°C, max {self.max_temp}°C)")

    def stop_monitoring(self):
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_sleep_time(self, base: float) -> float:
        with self._lock:
            return base if self.throttle_factor >= 1.0 else base / self.throttle_factor

    def _get_temp(self) -> Optional[float]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                return float(r.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return None

    def _loop(self):
        while self._monitoring:
            temp = self._get_temp()
            if temp is not None:
                with self._lock:
                    if temp > self.max_temp:
                        self.throttle_factor = 0.5
                    elif temp > self.target_temp:
                        excess = (temp - self.target_temp) / (self.max_temp - self.target_temp)
                        self.throttle_factor = max(0.6, 1.0 - excess * 0.5)
                    else:
                        self.throttle_factor = min(1.0, self.throttle_factor + 0.05)
            time.sleep(self.check_interval)
