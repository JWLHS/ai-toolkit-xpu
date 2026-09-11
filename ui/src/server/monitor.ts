import { spawn, execFile, ChildProcess } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import si from 'systeminformation';
import { loadMacstats } from '@/server/macstats';
import { createLoadSampler, readCpuTemperature, readMemory } from '@/server/cpuStats';
import { CpuInfo, GpuInfo, GPUApiResponse, MonitorHistoryPoint, MonitorInit, MonitorSample } from '@/types';
import { historyPointFromSample, MONITOR_HISTORY_LENGTH, MONITOR_TICK_MS } from '@/utils/monitorSample';
import {
  listXpuDevices,
  preferredXpuSource,
  sampleXpuDevice,
  sampleXpuViaTorch,
  sampleXpuViaZes,
  XpuDevice,
} from '@/server/xpuSmi';

const execFileAsync = promisify(execFile);

// xpu-smi spawns a process per call (~200ms on Arc), so the XPU path samples
// every Nth tick and reuses the last sample in between. 500ms * 4 = 2s.
const XPU_SAMPLE_EVERY_TICKS = 4;

/**
 * Always-on system monitor. Samples CPU + GPU every MONITOR_TICK_MS, keeps a
 * 2-minute rolling history (load + memory only), and pushes every full sample
 * to subscribers (the SSE route at /api/monitor).
 *
 * GPU stats come from a single resident `nvidia-smi ... -lms` child process —
 * NVML stays initialized between samples, which is what made the old
 * spawn-per-request /api/gpu route slow. If loop mode never produces output
 * (or nvidia-smi keeps hanging), we fall back to a one-shot spawn per tick,
 * which matches the old route's behavior exactly.
 */

const NV_QUERY_ARGS = [
  '--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory',
  '--format=csv,noheader,nounits',
];
const NV_ENV = { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' };
// No stdout line for this long means the loop child is buffering or hung.
const NV_WATCHDOG_MS = 15_000;
// A line-less gap this long after the last line closes out a batch (all
// lines of one iteration arrive together; iterations are MONITOR_TICK_MS
// apart, so this can never bleed into the next batch).
const NV_BATCH_FLUSH_MS = 100;
// Temperature refresh is decoupled from the tick (see refreshCpuTemp)
const CPU_TEMP_REFRESH_MS = 5000;

/** Average current core clock (MHz). os.cpus() is a cheap syscall — no child. */
function averageCpuSpeedMHz(): number {
  const speeds = os
    .cpus()
    .map(core => core.speed)
    .filter(speed => typeof speed === 'number' && speed > 0);
  if (speeds.length === 0) return 0;
  return Math.round(speeds.reduce((sum, speed) => sum + speed, 0) / speeds.length);
}

function parseGpuLine(line: string): GpuInfo | null {
  const [
    index,
    name,
    driverVersion,
    temperature,
    gpuUtil,
    memoryUtil,
    memoryTotal,
    memoryFree,
    memoryUsed,
    powerDraw,
    powerLimit,
    clockGraphics,
    clockMemory,
  ] = line.split(', ').map(item => item.trim());
  if (isNaN(parseInt(index))) return null;
  return {
    index: parseInt(index),
    name,
    driverVersion,
    temperature: parseInt(temperature),
    utilization: {
      gpu: parseInt(gpuUtil),
      memory: parseInt(memoryUtil),
    },
    memory: {
      total: parseInt(memoryTotal),
      free: parseInt(memoryFree),
      used: parseInt(memoryUsed),
    },
    power: {
      draw: parseFloat(powerDraw),
      limit: parseFloat(powerLimit),
    },
    clocks: {
      graphics: parseInt(clockGraphics),
      memory: parseInt(clockMemory),
    },
  };
}

// Serialized once per tick and shared, so N subscribers don't stringify N times
type Subscriber = (sample: MonitorSample, serialized: string) => void;

class SystemMonitor {
  private isMac = os.platform() === 'darwin';
  private history: MonitorHistoryPoint[] = [];
  private subscribers = new Set<Subscriber>();
  private latestCpu: CpuInfo | null = null;
  private latestGpu: GPUApiResponse = { hasNvidiaSmi: false, isMac: this.isMac, gpus: [] };
  private macGpuName = 'Apple GPU';
  // GPU backend: NVIDIA by default, Intel XPU when xpu-smi answers.
  private gpuBackend: 'nvidia' | 'xpu' = 'nvidia';
  private xpuDevices: XpuDevice[] | null = null;
  private xpuTickCounter = 0;
  private xpuSampleInFlight = false;
  // true when the stats come from torch (no xpu-smi installed)
  private xpuViaTorch = false;
  // true when the stats come from Level Zero Sysman (no xpu-smi installed)
  private xpuViaZes = false;
  private nvChild: ChildProcess | null = null;
  private nvBatch: GpuInfo[] = [];
  private nvStdoutBuffer = '';
  private nvFlushTimer: NodeJS.Timeout | null = null;
  private nvUnavailable = false;
  private nvReaped = false;
  private nvEverGotLine = false;
  private nvOneShotMode = false;
  private nvOneShotInFlight = false;
  private lastNvLineAt = 0;
  private tickInFlight = false;
  private lastCpuTemp = 0;
  private cpuTempInFlight = false;
  private lastCpuTempAt = 0;
  private cpuStatic: { name: string; cores: number } | null = null;
  // Differences os.cpus() times between ticks; no child process involved.
  private sampleLoad = createLoadSampler();

  start(): void {
    if (this.isMac) {
      this.initMacGpuName();
    } else {
      this.startGpuBackend();
    }
    // Never leave a resident nvidia-smi behind.
    process.once('exit', () => {
      try {
        this.nvChild?.kill('SIGKILL');
      } catch {
        // already gone
      }
    });
    // Fixed cadence; a tick that overruns the interval (slow temperature
    // read, hung nvidia-smi one-shot) just skips beats instead of stacking.
    this.tick();
    setInterval(() => this.tick(), MONITOR_TICK_MS);
  }

  /**
   * Intel Arc boxes have no nvidia-smi. Order: Level Zero Sysman (no extra
   * install), then xpu-smi if the user has Intel's XPU Manager, then torch,
   * and finally the NVIDIA loop for machines that really do have an NVIDIA GPU.
   * AI_TOOLKIT_XPU_MONITOR=zes|xpu-smi|torch|auto forces one of them.
   */
  private startGpuBackend(): void {
    const preference = preferredXpuSource();
    if (preference === 'xpu-smi') return this.startXpuSmiOrNv();
    if (preference === 'torch') return this.startTorchOrNv();
    // default: Sysman first
    this.startZesOrTorchOrNv();
  }

  private startXpuSmiOrNv(): void {
    execFile('xpu-smi', ['-h'], { timeout: 5000 }, err => {
      if (err) return this.startZesOrTorchOrNv();
      this.gpuBackend = 'xpu';
      this.latestGpu = { backend: 'xpu', hasNvidiaSmi: false, isMac: false, gpus: [] };
      listXpuDevices()
        .then(devices => {
          this.xpuDevices = devices;
          if (devices.length === 0) {
            console.warn('Monitor: xpu-smi found no Intel devices');
          }
        })
        .catch(error => {
          console.warn('Monitor: xpu-smi discovery failed:', error);
          this.xpuDevices = [];
        });
    });
  }

  /**
   * No xpu-smi: Level Zero Sysman gives temp/power/clock/VRAM (ze_loader.dll
   * ships with the driver), torch at least gives the name and VRAM. Anything
   * beats showing "no GPU" on an Intel box without Intel's XPU Manager.
   */
  private startZesOrTorchOrNv(): void {
    sampleXpuViaZes(true)
      .then(gpus => {
        if (gpus && gpus.length > 0) {
          this.gpuBackend = 'xpu';
          this.xpuViaZes = true;
          this.latestGpu = { backend: 'xpu', hasNvidiaSmi: false, isMac: false, gpus };
          return;
        }
        this.startTorchOrNv();
      })
      .catch(() => this.startTorchOrNv());
  }

  private startTorchOrNv(): void {
    sampleXpuViaTorch(true)
      .then(gpus => {
        if (!gpus || gpus.length === 0) {
          this.startNvLoop();
          return;
        }
        this.gpuBackend = 'xpu';
        this.xpuViaTorch = true;
        this.latestGpu = {
          backend: 'xpu',
          hasNvidiaSmi: false,
          isMac: false,
          gpus,
          error: 'xpu-smi 未安装：温度/功耗/频率不可用',
        };
      })
      .catch(() => this.startNvLoop());
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  getInit(): MonitorInit {
    return {
      t: Date.now(),
      cpu: this.latestCpu,
      gpu: this.latestGpu,
      history: [...this.history],
    };
  }

  // -------------------------------------------------------------------------
  // Tick loop
  // -------------------------------------------------------------------------
  private async tick(): Promise<void> {
    if (this.tickInFlight) return;
    this.tickInFlight = true;
    try {
      await this.doTick();
    } finally {
      this.tickInFlight = false;
    }
  }

  private async doTick(): Promise<void> {
    const t = Date.now();
    try {
      this.latestCpu = await this.sampleCpu();
    } catch (error) {
      console.error('Monitor: CPU sample failed:', error);
    }
    try {
      if (this.isMac) {
        this.latestGpu = this.sampleMacGpu();
      } else if (this.gpuBackend === 'xpu') {
        await this.sampleXpuTick();
      } else if (this.nvOneShotMode) {
        await this.sampleNvOneShot();
      } else {
        this.nvWatchdog();
      }
    } catch (error) {
      console.error('Monitor: GPU sample failed:', error);
    }

    const sample: MonitorSample = { t, cpu: this.latestCpu, gpu: this.latestGpu };
    this.history.push(historyPointFromSample(sample));
    if (this.history.length > MONITOR_HISTORY_LENGTH) {
      this.history.splice(0, this.history.length - MONITOR_HISTORY_LENGTH);
    }
    const serialized = JSON.stringify(sample);
    for (const subscriber of this.subscribers) {
      try {
        subscriber(sample, serialized);
      } catch (error) {
        console.error('Monitor: subscriber failed:', error);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Intel XPU GPU (xpu-smi)
  // -------------------------------------------------------------------------
  private async sampleXpuTick(): Promise<void> {
    if (this.xpuViaZes) {
      const gpus = await sampleXpuViaZes();
      if (gpus && gpus.length > 0) {
        this.latestGpu = { backend: 'xpu', hasNvidiaSmi: false, isMac: false, gpus };
      }
      return;
    }
    if (this.xpuViaTorch) {
      const gpus = await sampleXpuViaTorch();
      if (gpus && gpus.length > 0) {
        this.latestGpu = {
          backend: 'xpu',
          hasNvidiaSmi: false,
          isMac: false,
          gpus,
          error: 'xpu-smi 未安装：温度/功耗/频率不可用',
        };
      }
      return;
    }
    if (this.xpuSampleInFlight) return;
    const devices = this.xpuDevices;
    if (!devices || devices.length === 0) return; // still probing, or none found

    this.xpuTickCounter++;
    const haveSample = this.latestGpu.gpus.length > 0;
    if (haveSample && this.xpuTickCounter % XPU_SAMPLE_EVERY_TICKS !== 0) return;

    this.xpuSampleInFlight = true;
    try {
      const gpus: GpuInfo[] = [];
      for (const device of devices) {
        gpus.push(await sampleXpuDevice(device));
      }
      this.latestGpu = {
        backend: 'xpu',
        hasNvidiaSmi: false,
        isMac: false,
        gpus: gpus.sort((a, b) => a.index - b.index),
      };
    } catch (error) {
      console.error('Monitor: XPU sample failed:', error);
    } finally {
      this.xpuSampleInFlight = false;
    }
  }

  // -------------------------------------------------------------------------
  // CPU (mirrors /api/cpu exactly)
  // -------------------------------------------------------------------------
  private async sampleCpu(): Promise<CpuInfo> {
    // si.cpu() shells out to lscpu/dmidecode on every call — at tick cadence
    // that alone was eating a measurable slice of a core. Name and core count
    // never change, so resolve them exactly once.
    if (!this.cpuStatic) {
      const cpuInfoRaw = await si.cpu();
      this.cpuStatic = { name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`, cores: cpuInfoRaw.cores };
    }
    const cpuStatic = this.cpuStatic;

    if (this.isMac) {
      try {
        const ms = loadMacstats();
        if (!ms) throw new Error('macstats unavailable');
        const ramData = ms.getRAMUsageSync();
        const cpuData = ms.getCpuDataSync();
        return {
          name: cpuStatic.name,
          cores: cpuStatic.cores,
          temperature: cpuData.temperature || 0,
          totalMemory: ramData.total / (1024 * 1024),
          availableMemory: ramData.free / (1024 * 1024),
          freeMemory: ramData.free / (1024 * 1024),
          currentLoad: this.sampleLoad(),
          speedMHz: averageCpuSpeedMHz(),
        };
      } catch {
        // Fallback to the generic path if macstats fails
      }
    }

    // The temperature read can take >1s on some machines and would make the
    // tick skip beats, so it refreshes concurrently and we use the cached
    // value (at most a tick or two old).
    this.refreshCpuTemp();
    // Nothing below spawns a process. Every child process forks this server
    // first, and forking a multi-GB Node heap freezes the event loop for
    // hundreds of ms — at tick cadence that stalled every request in the UI.
    const memoryData = await readMemory();
    return {
      name: cpuStatic.name,
      cores: cpuStatic.cores,
      temperature: this.lastCpuTemp,
      totalMemory: memoryData.total / (1024 * 1024),
      availableMemory: memoryData.available / (1024 * 1024),
      freeMemory: memoryData.free / (1024 * 1024),
      currentLoad: this.sampleLoad(),
      speedMHz: averageCpuSpeedMHz(),
    };
  }

  private refreshCpuTemp(): void {
    // On Linux this is a sysfs read; elsewhere it may still shell out (via
    // systeminformation), and e.g. `sensors` can take ~1s of slow SMBus
    // polling — refreshed back-to-back it becomes a permanently resident
    // child. Every few seconds is plenty for a temperature.
    if (this.cpuTempInFlight || Date.now() - this.lastCpuTempAt < CPU_TEMP_REFRESH_MS) return;
    this.lastCpuTempAt = Date.now();
    this.cpuTempInFlight = true;
    readCpuTemperature()
      .then(t => {
        this.lastCpuTemp = t ?? 0;
      })
      .catch(() => {
        // keep the previous value
      })
      .finally(() => {
        this.cpuTempInFlight = false;
      });
  }

  // -------------------------------------------------------------------------
  // Mac GPU (mirrors /api/gpu's mac path)
  // -------------------------------------------------------------------------
  private initMacGpuName(): void {
    execFileAsync('sh', ['-c', 'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"'], {
      timeout: 5000,
    })
      .then(({ stdout }) => {
        const nameMatch = stdout.match(/Chipset Model:\s*(.+)/);
        const coresMatch = stdout.match(/Total Number of Cores:\s*(\d+)/);
        if (nameMatch) {
          this.macGpuName = nameMatch[1].trim();
          if (coresMatch) {
            this.macGpuName += ` GPU (${coresMatch[1]} cores)`;
          }
        }
      })
      .catch(() => {
        // fallback to generic name
      });
  }

  private sampleMacGpu(): GPUApiResponse {
    let temperature = 0;
    let gpuLoad = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = os.totalmem() / (1024 * 1024);

    const ms = loadMacstats();
    if (ms) {
      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }
      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }
      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    }

    return {
      backend: 'mac',
      hasNvidiaSmi: false,
      isMac: true,
      gpus: [
        {
          index: 0,
          name: this.macGpuName,
          driverVersion: 'macOS',
          temperature: Math.round(temperature),
          utilization: {
            gpu: gpuLoad,
            memory: memTotal > 0 ? Math.round((memUsed / memTotal) * 100) : 0,
          },
          memory: {
            total: Math.round(memTotal),
            free: Math.round(memTotal - memUsed),
            used: Math.round(memUsed),
          },
          power: { draw: powerDraw, limit: 0 },
          clocks: { graphics: 0, memory: 0 },
        },
      ],
    };
  }

  // -------------------------------------------------------------------------
  // NVIDIA loop-mode child
  // -------------------------------------------------------------------------
  private startNvLoop(): void {
    if (this.nvUnavailable || this.nvOneShotMode || this.nvChild) return;

    // A hard-killed server (SIGKILL never runs the exit hook) can orphan the
    // resident loop child. Our exact query string only ever appears in
    // children we spawned, so reap any stray once at boot — after it
    // completes, to not race the kill against our own fresh child.
    if (!this.nvReaped && process.platform !== 'win32') {
      this.nvReaped = true;
      // No loop flag in the pattern so strays from any past cadence match
      execFile('pkill', ['-9', '-f', `nvidia-smi ${NV_QUERY_ARGS.join(' ')}`], () => this.startNvLoop());
      return;
    }
    this.nvReaped = true;

    let child: ChildProcess;
    try {
      child = spawn('nvidia-smi', [...NV_QUERY_ARGS, '-lms', String(MONITOR_TICK_MS)], {
        env: NV_ENV,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch {
      this.markNvUnavailable();
      return;
    }
    this.nvChild = child;
    this.nvStdoutBuffer = '';
    this.nvBatch = [];
    this.lastNvLineAt = Date.now();

    child.stdout!.on('data', (chunk: Buffer) => this.onNvData(chunk.toString()));
    child.stderr!.on('data', () => {
      // nvidia-smi warnings are not actionable here
    });
    child.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'ENOENT') {
        this.markNvUnavailable();
      }
    });
    child.on('exit', () => {
      if (this.nvChild !== child) return;
      this.nvChild = null;
      if (this.nvFlushTimer) {
        clearTimeout(this.nvFlushTimer);
        this.nvFlushTimer = null;
      }
      if (!this.nvUnavailable && !this.nvOneShotMode) {
        setTimeout(() => this.startNvLoop(), 5000);
      }
    });
  }

  private onNvData(text: string): void {
    this.nvStdoutBuffer += text;
    const lines = this.nvStdoutBuffer.split('\n');
    this.nvStdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const gpu = parseGpuLine(line);
      if (!gpu) continue;
      this.nvEverGotLine = true;
      this.lastNvLineAt = Date.now();
      this.nvBatch.push(gpu);
    }
    if (this.nvFlushTimer) clearTimeout(this.nvFlushTimer);
    this.nvFlushTimer = setTimeout(() => this.flushNvBatch(), NV_BATCH_FLUSH_MS);
  }

  private flushNvBatch(): void {
    this.nvFlushTimer = null;
    if (this.nvBatch.length === 0) return;
    this.latestGpu = {
      hasNvidiaSmi: true,
      isMac: false,
      gpus: this.nvBatch.sort((a, b) => a.index - b.index),
    };
    this.nvBatch = [];
  }

  private nvWatchdog(): void {
    if (this.nvUnavailable || !this.nvChild) return;
    if (Date.now() - this.lastNvLineAt <= NV_WATCHDOG_MS) return;
    console.warn('Monitor: no output from nvidia-smi loop, restarting it');
    // Loop mode that never produced a single line isn't going to start —
    // switch to a one-shot spawn per tick instead of kill/respawn forever.
    if (!this.nvEverGotLine) {
      this.nvOneShotMode = true;
    }
    this.nvChild.kill('SIGKILL');
  }

  private async sampleNvOneShot(): Promise<void> {
    if (this.nvUnavailable || this.nvOneShotInFlight) return;
    this.nvOneShotInFlight = true;
    try {
      const { stdout } = await execFileAsync('nvidia-smi', NV_QUERY_ARGS, { env: NV_ENV });
      const gpus = stdout
        .trim()
        .split('\n')
        .map(parseGpuLine)
        .filter((gpu): gpu is GpuInfo => gpu !== null)
        .sort((a, b) => a.index - b.index);
      this.latestGpu = { hasNvidiaSmi: true, isMac: false, gpus };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        this.markNvUnavailable();
      } else {
        console.error('Monitor: one-shot nvidia-smi failed:', err);
      }
    } finally {
      this.nvOneShotInFlight = false;
    }
  }

  private markNvUnavailable(): void {
    this.nvUnavailable = true;
    this.latestGpu = {
      backend: 'nvidia',
      hasNvidiaSmi: false,
      isMac: false,
      gpus: [],
      error: 'nvidia-smi not found or not accessible',
    };
  }
}

/**
 * Idempotent starter. Guarded on globalThis so dev-mode module reloads never
 * stack a second sampler (and a second resident nvidia-smi) in the same
 * process.
 */
export function startMonitor(): SystemMonitor {
  const g = globalThis as unknown as { __aiToolkitSystemMonitor?: SystemMonitor };
  if (!g.__aiToolkitSystemMonitor) {
    g.__aiToolkitSystemMonitor = new SystemMonitor();
    // Escape hatch: with AI_TOOLKIT_DISABLE_MONITOR=1 the monitor object
    // exists (the SSE route stays functional) but never samples anything.
    if (process.env.AI_TOOLKIT_DISABLE_MONITOR !== '1') {
      g.__aiToolkitSystemMonitor.start();
    }
  }
  return g.__aiToolkitSystemMonitor;
}
