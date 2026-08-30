import { NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

/* ============================================================
   GET /api/gpu  ——  支持多 GPU / XPU 显存实时监控
   ============================================================ */
export async function GET() {
  try {
    // 检查 xpu-smi 是否可用
    const hasXpu = await checkXpuSmi();
    if (!hasXpu) {
      return NextResponse.json({
        backend: "none",
        hasNvidiaSmi: false,
        gpus: [],
        error: "xpu-smi not found",
      });
    }

    // STEP 1 —— 获取所有 GPU 列表
    const discovery = await getDeviceList();
    const deviceList = discovery.device_list ?? [];

    if (deviceList.length === 0) {
      return NextResponse.json({
        backend: "xpu",
        hasNvidiaSmi: true,
        gpus: [],
        error: "No XPU devices found",
      });
    }

    // STEP 2 —— 遍历每个 GPU，读取完整信息
    const result = [];

    for (const dev of deviceList) {
      const id = dev.device_id;

      // 获取含显存信息的详细 JSON
      const fullInfo = await getDeviceDetails(id);

      // 解析显存总量/剩余量
      const memTotalMiB = fullInfo.memory_physical_size_byte
        ? Math.round(Number(fullInfo.memory_physical_size_byte) / 1024 / 1024)
        : 0;

      const memFreeMiB = fullInfo.memory_free_size_byte
        ? Math.round(Number(fullInfo.memory_free_size_byte) / 1024 / 1024)
        : Math.max(0, memTotalMiB - 1024); // fallback

      // STEP 3 —— 解析 dump 数据（实时 GPU 性能）
      const stats = await getDumpStats(id);

      // 组合输出（保持 UI 兼容）
      result.push({
        index: id,
        name: fullInfo.device_name ?? "Intel XPU",
        driverVersion: fullInfo.driver_version ?? "unknown",

        temperature: stats.temperature,
        utilization: {
          gpu: stats.gpuUtil,
          memory: stats.memUtil,
        },

        memory: {
          total: memTotalMiB,
          used: stats.memUsed, // dump 中的实时显存使用（MiB）
          free: memTotalMiB > 0 ? memTotalMiB - stats.memUsed : memFreeMiB,
        },

        power: stats.power,
        frequency: stats.freq,
      });
    }

    return NextResponse.json({
      backend: "xpu",
      hasNvidiaSmi: true,
      gpus: result,
    });
  } catch (err) {
    return NextResponse.json(
      {
        backend: "error",
        hasNvidiaSmi: false,
        gpus: [],
        error: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

/* ============================================================
   工具函数区域
   ============================================================ */

// --- 判断 xpu-smi 是否可用 ---
async function checkXpuSmi() {
  try {
    await execAsync("xpu-smi -h");
    return true;
  } catch {
    return false;
  }
}

// --- 获取设备列表（不含显存）---
async function getDeviceList() {
  const { stdout } = await execAsync("xpu-smi discovery -j");
  return JSON.parse(stdout);
}

// --- 获取单个 GPU 的完整信息（含显存）---
async function getDeviceDetails(id: number) {
  const { stdout } = await execAsync(`xpu-smi discovery -d ${id} -j`);
  return JSON.parse(stdout);
}

// --- dump 实时性能数据 ---
async function getDumpStats(id: number) {
  const { stdout } = await execAsync(
    `xpu-smi dump -d ${id} -m 0,1,2,3,5,17,18 -n 1`
  );

  const lines = stdout.trim().split("\n");
  const csvLine = lines[lines.length - 1]; // 最后一行是数据

  const fields = csvLine.split(",").map((x) => x.trim());

  const safe = (v: any) =>
    !v || v.toLowerCase?.() === "n/a" ? 0 : parseFloat(v);

  return {
    gpuUtil: safe(fields[2]),
    power: safe(fields[3]),
    freq: safe(fields[4]),
    temperature: safe(fields[5]),
    memUtil: safe(fields[6]),
    memUsed: safe(fields[8]) || 0, // MiB
  };
}
