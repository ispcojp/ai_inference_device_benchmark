import pynvml
import threading
import time
from typing import Dict
from pynvml import c_nvmlDevice_t


def get_used_vram(handle: c_nvmlDevice_t) -> int:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used


def monitor_vram(
    vram_holder: Dict[str, int], handle: c_nvmlDevice_t, stop_event: threading.Event
):
    while not stop_event.is_set():
        used = get_used_vram(handle)
        if used > vram_holder["value"]:
            vram_holder["value"] = used

        time.sleep(0.1)


def vram_monitor_start(
    handle: c_nvmlDevice_t,
) -> tuple[Dict[str, int], threading.Thread, threading.Event]:
    baseline = get_used_vram(handle)
    # モニター開始
    vram_holder = {"value": baseline}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_vram, args=(vram_holder, handle, stop_event)
    )
    monitor_thread.start()

    return vram_holder, monitor_thread, stop_event


def first_vram_setting() -> c_nvmlDevice_t:
    pynvml.nvmlInit()  # NVMLの初期化
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    return handle


def vram_monitor_end(
    stop_event: threading.Event,
    monitor_thread: threading.Thread,
    vram_holder: Dict[str, int],
    first_peak_vram: int,
) -> int:
    stop_event.set()
    monitor_thread.join()
    peak_vram = vram_holder["value"] - first_peak_vram
    pynvml.nvmlShutdown()

    return peak_vram
