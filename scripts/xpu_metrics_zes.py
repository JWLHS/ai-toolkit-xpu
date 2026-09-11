#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intel Arc / XPU metrics straight from Level Zero Sysman — no xpu-smi needed.

``ze_loader.dll`` ships with the Intel graphics driver, so this works on a
fresh clone where Intel's XPU Manager (``xpu-smi``) was never installed.

    python scripts/xpu_metrics_zes.py        # JSON array, one object per GPU

Fields: index, name, temperature (C), gpuUtil (%), memUtil (%), memTotalMiB,
memUsedMiB, memFreeMiB, powerW, freqMHz. Anything the driver refuses (power
usually needs an elevated shell) comes back as 0 rather than failing the run.

The device name comes straight from the driver: zeDeviceGetProperties() returns
the marketing name ("Intel(R) Arc(TM) A770 Graphics") at offset 112, so the UI
does not have to shell out to WMI.

Struct offsets and the general approach follow the Level Zero usage in
allanmeng/ComfyUI-XPUSYS-Monitor (providers/intel.py, MIT) — thanks to that
project for mapping them out; this file is trimmed to what the ai-toolkit UI
shows. See the credits section of README.md.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time

ZE_RESULT_SUCCESS = 0
ZES_STYPE_MEM_STATE = 0x18
ZES_STYPE_FREQ_PROPS = 0x10
ZES_FREQ_DOMAIN_GPU = 0
ZES_TEMP_SENSORS_GPU = 1
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3

# Engine activity needs two samples to turn counters into a percentage.
LOAD_SAMPLE_SECONDS = 0.05
# Power is an energy counter; two reads give watts (uJ / us == W).
POWER_SAMPLE_SECONDS = 0.2


def _load_library():
    for name in ("ze_loader.dll", "libze_loader.so.1", "libze_loader.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _enumerate(lib, init_name, driver_get, device_get):
    """Enumerate devices through one of the Level Zero entry points."""
    try:
        init = getattr(lib, init_name)
        get_drivers = getattr(lib, driver_get)
        get_devices = getattr(lib, device_get)
    except AttributeError:
        return []

    init.restype = ctypes.c_int
    init.argtypes = [ctypes.c_uint32]
    if init(1 << 0) != ZE_RESULT_SUCCESS:  # ZE_INIT_FLAG_GPU_ONLY
        return []

    get_drivers.restype = ctypes.c_int
    get_drivers.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    count = ctypes.c_uint32(0)
    if get_drivers(ctypes.byref(count), None) != ZE_RESULT_SUCCESS or count.value == 0:
        return []
    drivers = (ctypes.c_void_p * count.value)()
    get_drivers(ctypes.byref(count), drivers)

    get_devices.restype = ctypes.c_int
    get_devices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    dev_count = ctypes.c_uint32(0)
    if get_devices(drivers[0], ctypes.byref(dev_count), None) != ZE_RESULT_SUCCESS:
        return []
    if dev_count.value == 0:
        return []
    devices = (ctypes.c_void_p * dev_count.value)()
    get_devices(drivers[0], ctypes.byref(dev_count), devices)
    return list(devices)


def _sysman_devices(lib):
    """Handles for the zes* metrics calls (Sysman must be initialized first)."""
    return _enumerate(lib, "zesInit", "zesDriverGet", "zesDeviceGet")


def _core_devices(lib):
    """Handles for ze* calls — the ones that know the marketing name."""
    return _enumerate(lib, "zeInit", "zeDriverGet", "zeDeviceGet")


def _enum_handles(lib, fn_name, device):
    try:
        fn = getattr(lib, fn_name)
    except AttributeError:
        return []
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    count = ctypes.c_uint32(0)
    if fn(device, ctypes.byref(count), None) != ZE_RESULT_SUCCESS or count.value == 0:
        return []
    handles = (ctypes.c_void_p * count.value)()
    fn(device, ctypes.byref(count), handles)
    return list(handles)


def _read_memory(lib, device):
    """(free_mib, total_mib) via zesMemoryGetState."""
    total_bytes = free_bytes = 0
    try:
        fn = lib.zesMemoryGetState
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    except AttributeError:
        return 0, 0
    for handle in _enum_handles(lib, "zesDeviceEnumMemoryModules", device):
        buf = (ctypes.c_uint8 * 48)()
        # zes_mem_state_t: stype@0, free@24, size@32
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))[0] = ZES_STYPE_MEM_STATE
        if fn(handle, buf) != ZE_RESULT_SUCCESS:
            continue
        base = ctypes.addressof(buf)
        free_bytes += ctypes.cast(base + 24, ctypes.POINTER(ctypes.c_uint64))[0]
        total_bytes += ctypes.cast(base + 32, ctypes.POINTER(ctypes.c_uint64))[0]
    mib = 1024 * 1024
    return round(free_bytes / mib), round(total_bytes / mib)


def _read_device_name(lib, device):
    """Driver-reported device name via zeDeviceGetProperties (name @ +112)."""
    try:
        fn = lib.zeDeviceGetProperties
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        buf = (ctypes.c_uint8 * 4096)()
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))[0] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
        if fn(device, buf) != ZE_RESULT_SUCCESS:
            return "", 0, 0
        base = ctypes.addressof(buf)
        vendor_id = ctypes.cast(base + 20, ctypes.POINTER(ctypes.c_uint32))[0]
        device_id = ctypes.cast(base + 24, ctypes.POINTER(ctypes.c_uint32))[0]
        name = ctypes.string_at(base + 112, 256).split(b"\0", 1)[0].decode("utf-8", "replace")
        return name.strip(), vendor_id, device_id
    except (AttributeError, OSError):
        # A device handle that the core API rejects is not fatal — the UI just
        # falls back to the Windows/WMI name.
        return "", 0, 0


def _read_load(lib, device):
    """Compute-engine busy percentage from two zesEngineGetActivity samples."""
    engines = _enum_handles(lib, "zesDeviceEnumEngineGroups", device)
    if not engines:
        return 0.0
    try:
        fn = lib.zesEngineGetActivity
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64 * 2)]
    except AttributeError:
        return 0.0
    first = (ctypes.c_uint64 * 2)()
    second = (ctypes.c_uint64 * 2)()
    if fn(engines[0], ctypes.byref(first)) != ZE_RESULT_SUCCESS:
        return 0.0
    time.sleep(LOAD_SAMPLE_SECONDS)
    if fn(engines[0], ctypes.byref(second)) != ZE_RESULT_SUCCESS:
        return 0.0
    active = second[0] - first[0]
    elapsed = second[1] - first[1]
    if elapsed <= 0:
        return 0.0
    return min(100.0, active / elapsed * 100.0)


def _read_frequency(lib, device):
    """Actual GPU core clock in MHz via zesFrequencyGetState.

    Devices expose several frequency domains (core / memory / media). Only the
    ZES_FREQ_DOMAIN_GPU domains are clocks; domains the driver lets us control
    are the real core ones, so those win.
    """
    domains = _enum_handles(lib, "zesDeviceEnumFrequencyDomains", device)
    if not domains:
        return 0.0
    try:
        fn = lib.zesFrequencyGetState
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        props_fn = lib.zesFrequencyGetProperties
        props_fn.restype = ctypes.c_int
        props_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    except AttributeError:
        props_fn = None

    core = 0.0
    other = 0.0
    for handle in domains:
        is_core = False
        if props_fn is not None:
            props = (ctypes.c_uint8 * 64)()
            # zes_freq_properties_t: stype@0, type@16, canControl@28
            ctypes.cast(props, ctypes.POINTER(ctypes.c_uint32))[0] = ZES_STYPE_FREQ_PROPS
            if props_fn(handle, props) == ZE_RESULT_SUCCESS:
                base = ctypes.addressof(props)
                domain_type = ctypes.cast(base + 16, ctypes.POINTER(ctypes.c_uint32))[0]
                can_control = ctypes.cast(base + 28, ctypes.POINTER(ctypes.c_uint32))[0]
                is_core = domain_type == ZES_FREQ_DOMAIN_GPU and bool(can_control)
                if domain_type != ZES_FREQ_DOMAIN_GPU:
                    continue  # memory / media domain, not the core clock

        buf = (ctypes.c_uint8 * 64)()
        if fn(handle, buf) != ZE_RESULT_SUCCESS:
            continue
        base = ctypes.addressof(buf)
        # request@24, actual@48 (Level Zero spec, see module docstring)
        request = ctypes.cast(base + 24, ctypes.POINTER(ctypes.c_double))[0]
        actual = ctypes.cast(base + 48, ctypes.POINTER(ctypes.c_double))[0]
        value = actual or request
        if is_core:
            core = max(core, value)
        else:
            other = max(other, value)
    return core or other


def _read_temperature(lib, device):
    """GPU core temperature in C via zesTemperatureGetState."""
    sensors = _enum_handles(lib, "zesDeviceEnumTemperatureSensors", device)
    if not sensors:
        return 0.0
    try:
        fn = lib.zesTemperatureGetState
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    except AttributeError:
        return 0.0
    for handle in sensors:
        value = ctypes.c_double(0.0)
        if fn(handle, ctypes.byref(value)) == ZE_RESULT_SUCCESS and value.value > 0:
            return value.value
    return 0.0


def _read_power(lib, device):
    """Watts from the energy counter (usually needs an elevated shell)."""
    domains = _enum_handles(lib, "zesDeviceEnumPowerDomains", device)
    if not domains:
        return 0.0
    try:
        fn = lib.zesPowerGetEnergyCounter
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64 * 2)]
    except AttributeError:
        return 0.0
    first = (ctypes.c_uint64 * 2)()
    if fn(domains[0], ctypes.byref(first)) != ZE_RESULT_SUCCESS:
        return 0.0
    time.sleep(POWER_SAMPLE_SECONDS)
    second = (ctypes.c_uint64 * 2)()
    if fn(domains[0], ctypes.byref(second)) != ZE_RESULT_SUCCESS:
        return 0.0
    energy = second[0] - first[0]        # microjoules
    elapsed = second[1] - first[1]       # microseconds -> uJ/us == W
    if elapsed <= 0:
        return 0.0
    return max(0.0, energy / elapsed)


def main() -> int:
    lib = _load_library()
    if lib is None:
        print("ze_loader.dll not found — Intel graphics driver missing?", file=sys.stderr)
        return 2

    core_devices = _core_devices(lib)
    devices = _sysman_devices(lib) or core_devices
    if not devices:
        print("no Level Zero devices", file=sys.stderr)
        return 3

    out = []
    for index, device in enumerate(devices):
        try:
            # the name comes from the core handle; metrics from the Sysman one
            name_handle = core_devices[index] if index < len(core_devices) else device
            name, vendor_id, device_id = _read_device_name(lib, name_handle)
        except OSError:
            name, vendor_id, device_id = "", 0, 0
        free_mib, total_mib = _read_memory(lib, device)
        used_mib = max(0, total_mib - free_mib)
        out.append(
            {
                "index": index,
                "name": name,
                "vendorId": hex(vendor_id) if vendor_id else "",
                "deviceId": hex(device_id) if device_id else "",
                "temperature": _read_temperature(lib, device),
                "gpuUtil": _read_load(lib, device),
                "memUtil": round(used_mib / total_mib * 100) if total_mib else 0,
                "memTotalMiB": total_mib,
                "memUsedMiB": used_mib,
                "memFreeMiB": free_mib,
                "powerW": _read_power(lib, device),
                "freqMHz": _read_frequency(lib, device),
            }
        )

    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
