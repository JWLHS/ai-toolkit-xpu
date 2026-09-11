import { exec } from 'child_process';
import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify as promisifyCb } from 'util';
import { promisify } from 'util';
import { GpuInfo } from '@/types';

/* ---------------------------------------------------------------------------
   Intel Arc / XPU (xpu-smi)

   Upstream targets NVIDIA (nvidia-smi) and Apple GPUs; on Intel Arc the same
   stats come from Intel's xpu-smi (XPU Manager). Both /api/gpu and the SSE
   system monitor use this module so the two panels can never disagree.
   --------------------------------------------------------------------------- */

const execAsync = promisify(exec);
const execFileAsync = promisifyCb(execFile);

// xpu-smi only reports the PCI id ("Intel(R) Graphics [0x56a0]"). Windows knows
// the marketing name, and these ids cover the DG2 cards when that lookup fails.
const ARC_PCI_NAMES: Record<string, string> = {
  '0x56a0': 'Intel Arc A770',
  '0x56a1': 'Intel Arc A750',
  '0x56a2': 'Intel Arc A580',
  '0x56a5': 'Intel Arc A380',
  '0x56a6': 'Intel Arc A310',
};

/** Friendly GPU names from Windows (WMI); [] when unavailable. */
async function windowsGpuNames(): Promise<string[]> {
  if (process.platform !== 'win32') return [];
  const ps = process.env.SystemRoot
    ? `${process.env.SystemRoot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`
    : 'powershell';
  try {
    const { stdout } = await execAsync(
      `"${ps}" -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"`,
      { timeout: 10000 },
    );
    return stdout
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * "Intel(R) Graphics [0x56a0]" -> "Intel(R) Arc(TM) A770 Graphics" when Windows
 * knows it, else the PCI-id table, else the raw xpu-smi string.
 */
function friendlyName(rawName: string, pciDeviceId: string | undefined, fromWmi: string[]): string {
  const isGeneric = /\[0x[0-9a-f]+\]/i.test(rawName);
  if (!isGeneric) return rawName;

  const wmiArc = fromWmi.find(name => /arc/i.test(name)) ?? fromWmi.find(name => /intel/i.test(name));
  if (wmiArc) return wmiArc;

  const key = (pciDeviceId ?? rawName.match(/\[(0x[0-9a-f]+)\]/i)?.[1] ?? '').toLowerCase();
  return ARC_PCI_NAMES[key] ?? rawName;
}

export interface XpuDevice {
  index: number;
  name: string;
  driverVersion: string;
  memoryTotalMiB: number;
}

/** True when Intel's xpu-smi is on PATH (shipped with the driver / XPU Manager). */
export async function checkXpuSmi(): Promise<boolean> {
  try {
    await execAsync('xpu-smi -h', { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
}

/** Static per-device info (name / driver / total VRAM). Cheap to cache. */
export async function listXpuDevices(): Promise<XpuDevice[]> {
  const { stdout: listOut } = await execAsync('xpu-smi discovery -j', {
    timeout: 10000,
  });
  const deviceList: any[] = JSON.parse(listOut).device_list ?? [];
  const devices: XpuDevice[] = [];
  const wmiNames = await windowsGpuNames();

  for (const dev of deviceList) {
    const id = dev.device_id;

    let fullInfo: any = {};
    try {
      const { stdout } = await execAsync(`xpu-smi discovery -d ${id} -j`, {
        timeout: 10000,
      });
      fullInfo = JSON.parse(stdout);
    } catch {
      // keep defaults; the dump below may still work
    }

    const totalBytes = Number(fullInfo.memory_physical_size_byte ?? 0);
    devices.push({
      index: typeof id === 'string' ? parseInt(id) || 0 : Number(id) || 0,
      name: friendlyName(
        fullInfo.device_name ?? dev.device_name ?? 'Intel XPU',
        dev.pci_device_id ?? fullInfo.pci_device_id,
        wmiNames,
      ),
      driverVersion: fullInfo.driver_version ?? dev.driver_version ?? 'unknown',
      memoryTotalMiB: totalBytes ? Math.round(totalBytes / 1024 / 1024) : 0,
    });
  }
  return devices;
}

/**
 * One live sample. xpu-smi dump columns (with -m 0,1,2,3,5,17,18):
 *   Timestamp, DeviceId, GPU Util%, Power W, Freq MHz, Core Temp C,
 *   Mem Util%, Mem BW Util%, Mem Used MiB
 */
export async function sampleXpuDevice(device: XpuDevice): Promise<GpuInfo> {
  let gpuUtil = 0;
  let power = 0;
  let freq = 0;
  let temperature = 0;
  let memUtil = 0;
  let memUsed = 0;
  try {
    const { stdout } = await execAsync(
      `xpu-smi dump -d ${device.index} -m 0,1,2,3,5,17,18 -n 1`,
      { timeout: 10000 },
    );
    const line = stdout.trim().split('\n').pop() ?? '';
    const f = line.split(',').map(x => x.trim());
    const num = (v: string | undefined) =>
      !v || v.toUpperCase() === 'N/A' ? 0 : parseFloat(v) || 0;
    gpuUtil = num(f[2]);
    power = num(f[3]);
    freq = num(f[4]);
    temperature = num(f[5]);
    memUtil = num(f[6]);
    memUsed = num(f[8]);
  } catch {
    // stats stay zero; the panel still shows name + total memory
  }

  const used = Math.round(memUsed);
  return {
    index: device.index,
    name: device.name,
    driverVersion: device.driverVersion,
    temperature: Math.round(temperature),
    utilization: { gpu: Math.round(gpuUtil), memory: Math.round(memUtil) },
    memory: {
      total: device.memoryTotalMiB,
      used,
      free: Math.max(0, device.memoryTotalMiB - used),
    },
    power: { draw: power, limit: 0 },
    clocks: { graphics: freq, memory: 0 },
  };
}

/** All devices, sampled once. Returns [] when nothing is detectable. */
export async function sampleXpuGpus(devices?: XpuDevice[]): Promise<GpuInfo[]> {
  const list = devices ?? (await listXpuDevices());
  const gpus: GpuInfo[] = [];
  for (const device of list) {
    gpus.push(await sampleXpuDevice(device));
  }
  return gpus.sort((a, b) => a.index - b.index);
}

/* ---------------------------------------------------------------------------
   Fallback without xpu-smi

   xpu-smi ships with Intel's XPU Manager, which is a separate download — a
   fresh clone may not have it. The training venv always has torch+xpu though,
   and torch can report the device name and VRAM. Temperature / power / clocks
   are only available through xpu-smi, so those stay 0 in this mode.
   --------------------------------------------------------------------------- */

const TORCH_TTL_MS = 15_000;
let torchCache: { at: number; gpus: GpuInfo[] } | null = null;
let torchPython: string | null | undefined;

/** The training venv's python (…/.venv/Scripts/python.exe), if we can find it. */
export function findVenvPython(): string | null {
  if (torchPython !== undefined) return torchPython;
  const exe = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python'];
  const roots = [process.env.AI_TOOLKIT_ROOT, path.resolve(process.cwd(), '..'), process.cwd()].filter(
    (r): r is string => !!r,
  );
  const candidates = [
    process.env.AI_TOOLKIT_PYTHON,
    ...roots.flatMap(root =>
      ['.venv', '.venv-xpu'].map(venv => path.join(root, venv, ...exe)),
    ),
  ].filter((p): p is string => !!p);
  torchPython = candidates.find(candidate => fs.existsSync(candidate)) ?? null;
  return torchPython;
}

const TORCH_SNIPPET = [
  'import json, sys',
  'try:',
  '    import torch',
  '    if not torch.xpu.is_available():',
  '        sys.exit(3)',
  '    out = []',
  '    for i in range(torch.xpu.device_count()):',
  '        free, total = torch.xpu.mem_get_info(i)',
  '        out.append({"index": i, "name": torch.xpu.get_device_name(i),',
  '                    "total": total, "free": free, "used": total - free,',
  '                    "torch": torch.__version__})',
  '    print(json.dumps(out))',
  'except Exception:',
  '    sys.exit(4)',
].join('\n');

/**
 * Sample through torch (cached for TORCH_TTL_MS — importing torch takes ~2s).
 * Returns null when there is no usable venv, so callers can fall back further.
 */
export async function sampleXpuViaTorch(force = false): Promise<GpuInfo[] | null> {
  if (!force && torchCache && Date.now() - torchCache.at < TORCH_TTL_MS) {
    return torchCache.gpus;
  }
  const python = findVenvPython();
  if (!python) return null;
  try {
    const { stdout } = await execFileAsync(python, ['-c', TORCH_SNIPPET], {
      timeout: 30_000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });
    const line = stdout.trim().split('\n').pop() ?? '[]';
    const parsed: any[] = JSON.parse(line);
    const gpus: GpuInfo[] = parsed.map(dev => {
      const mib = (bytes: number) => Math.round(bytes / 1024 / 1024);
      const total = mib(dev.total);
      const used = mib(dev.used);
      return {
        index: dev.index,
        name: dev.name,
        driverVersion: 'torch ' + (dev.torch ?? ''),
        temperature: 0,
        utilization: { gpu: 0, memory: total ? Math.round((used / total) * 100) : 0 },
        memory: { total, used, free: mib(dev.free) },
        power: { draw: 0, limit: 0 },
        clocks: { graphics: 0, memory: 0 },
      };
    });
    torchCache = { at: Date.now(), gpus };
    return gpus;
  } catch {
    return null;
  }
}

/**
 * Level Zero Sysman (zes) path — no xpu-smi needed.
 *
 * ``ze_loader.dll`` comes with the Intel graphics driver, so a fresh clone can
 * read temp / power / clock / VRAM through scripts/xpu_metrics_zes.py.
 */
const ZES_TTL_MS = 5_000;
let zesCache: { at: number; gpus: GpuInfo[] } | null = null;

function zesScriptPath(): string | null {
  const roots = [process.env.AI_TOOLKIT_ROOT, path.resolve(process.cwd(), '..'), process.cwd()].filter(
    (r): r is string => !!r,
  );
  for (const root of roots) {
    const candidate = path.join(root, 'scripts', 'xpu_metrics_zes.py');
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

export async function sampleXpuViaZes(force = false): Promise<GpuInfo[] | null> {
  if (!force && zesCache && Date.now() - zesCache.at < ZES_TTL_MS) return zesCache.gpus;
  const python = findVenvPython();
  const script = zesScriptPath();
  if (!python || !script) return null;
  try {
    const { stdout } = await execFileAsync(python, [script], {
      timeout: 30_000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });
    const parsed: any[] = JSON.parse(stdout.trim().split('\n').pop() ?? '[]');
    const wmiNames = await windowsGpuNames();
    const gpus: GpuInfo[] = parsed.map(dev => ({
      index: dev.index,
      name: friendlyName(dev.name || `Intel(R) Graphics [${dev.index}]`, undefined, wmiNames),
      driverVersion: 'Level Zero Sysman',
      temperature: Math.round(dev.temperature ?? 0),
      utilization: {
        gpu: Math.round(dev.gpuUtil ?? 0),
        memory: Math.round(dev.memUtil ?? 0),
      },
      memory: {
        total: dev.memTotalMiB ?? 0,
        used: dev.memUsedMiB ?? 0,
        free: dev.memFreeMiB ?? 0,
      },
      power: { draw: Math.round((dev.powerW ?? 0) * 10) / 10, limit: 0 },
      clocks: { graphics: Math.round(dev.freqMHz ?? 0), memory: 0 },
    }));
    if (gpus.length === 0) return null;
    zesCache = { at: Date.now(), gpus };
    return gpus;
  } catch {
    return null;
  }
}

/**
 * Which Intel monitor to use.
 *
 * Default is Level Zero Sysman: it reads name / VRAM / load / temp / power /
 * clock straight from the driver (ze_loader.dll ships with the Intel graphics
 * driver), so nothing extra has to be installed. xpu-smi is only a fallback
 * for drivers where Sysman misbehaves, and torch is the last resort.
 *
 * Override with AI_TOOLKIT_XPU_MONITOR=auto|zes|xpu-smi|torch.
 */
export function preferredXpuSource(): 'auto' | 'zes' | 'xpu-smi' | 'torch' {
  const value = (process.env.AI_TOOLKIT_XPU_MONITOR ?? 'auto').toLowerCase();
  return value === 'zes' || value === 'xpu-smi' || value === 'torch' ? value : 'auto';
}

export async function sampleXpuAny(): Promise<{
  gpus: GpuInfo[];
  source: 'xpu-smi' | 'zes' | 'torch';
} | null> {
  const preference = preferredXpuSource();
  const tryZes = async () => {
    const gpus = await sampleXpuViaZes(true);
    return gpus && gpus.length > 0 ? { gpus, source: 'zes' as const } : null;
  };
  const tryXpuSmi = async () => {
    if (!(await checkXpuSmi())) return null;
    try {
      const gpus = await sampleXpuGpus();
      return gpus.length > 0 ? { gpus, source: 'xpu-smi' as const } : null;
    } catch {
      return null;
    }
  };
  const tryTorch = async () => {
    const gpus = await sampleXpuViaTorch(true);
    return gpus && gpus.length > 0 ? { gpus, source: 'torch' as const } : null;
  };

  switch (preference) {
    case 'zes':
      return (await tryZes()) ?? null;
    case 'xpu-smi':
      return (await tryXpuSmi()) ?? null;
    case 'torch':
      return (await tryTorch()) ?? null;
    default:
      break;
  }

  return (await tryZes()) ?? (await tryXpuSmi()) ?? (await tryTorch()) ?? null;
}
