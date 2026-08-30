import React from 'react';
import { Thermometer, Zap, Clock, HardDrive, Fan, Cpu } from 'lucide-react';

interface GPUWidgetProps {
  gpu: any;          // 更宽松，支持多后端
  backend?: string;  // 由 GpuMonitor 传入
}

export default function GPUWidget({ gpu, backend = "unknown" }: GPUWidgetProps) {
  const safe = (v: any, def = 0) => (v === undefined || v === null || isNaN(v) ? def : v);

  const memoryUsed = safe(gpu?.memory?.used);
  const memoryTotal = safe(gpu?.memory?.total);
  const memPercent = memoryTotal ? (memoryUsed / memoryTotal) * 100 : 0;

  const gpuUtil = safe(gpu?.utilization?.gpu);
  const temperature = safe(gpu?.temperature);

  // XPU 没有 fan/clocks/power.limit → 自动 fallback
  const fanSpeed = safe(gpu?.fan?.speed, 0);
  const clockSpeed = safe(gpu?.clocks?.graphics, gpu?.frequency ?? 0); // Arc 有 freq
  const powerDraw = safe(gpu?.power?.draw, gpu?.power ?? 0); // Arc 返回 power 字段
  const powerLimit = safe(gpu?.power?.limit, 0);

  const formatMemory = (mb: number) =>
    mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;

  const getUtilColor = (value: number) =>
    value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';

  const getTempColor = (temp: number) =>
    temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h2 className="font-semibold text-gray-100">{gpu?.name}</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">
            #{gpu?.index}
          </span>
        </div>

        <span className="text-xs text-gray-400 border px-2 py-0.5 rounded">
          {backend.toUpperCase()}
        </span>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">

        {/* Temperature + Fan + Util */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            {/* Temp */}
            <div className="flex items-center space-x-2">
              <Thermometer className={`w-4 h-4 ${getTempColor(temperature)}`} />
              <div>
                <p className="text-xs text-gray-400">核心温度</p>
                <p className={`text-sm font-medium ${getTempColor(temperature)}`}>
                  {temperature}°C
                </p>
              </div>
            </div>

            {/* Fan (XPU fallback 0%) */}
            <div className="flex items-center space-x-2">
              <Fan className="w-4 h-4 text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">风扇转速（无法检测）</p>
                <p className="text-sm font-medium text-blue-400">
                  {fanSpeed}%
                </p>
              </div>
            </div>
          </div>

          {/* Util + Memory */}
          <div>
            <div className="flex items-center space-x-2 mb-1">
              <Cpu className="w-4 h-4 text-gray-400" />
              <p className="text-xs text-gray-400">核心负载</p>
              <span className="text-xs text-gray-300 ml-auto">{gpuUtil}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className={`h-1 rounded-full transition-all ${getUtilColor(gpuUtil)}`}
                style={{ width: `${gpuUtil}%` }}
              />
            </div>

            {/* Memory */}
            <div className="flex items-center space-x-2 mb-1 mt-3">
              <HardDrive className="w-4 h-4 text-blue-400" />
              <p className="text-xs text-gray-400">显存使用</p>
              <span className="text-xs text-gray-300 ml-auto">
                {memPercent.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className="h-1 rounded-full bg-blue-500 transition-all"
                style={{ width: `${memPercent}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              {formatMemory(memoryUsed)} / {formatMemory(memoryTotal)}
            </p>
          </div>
        </div>

        {/* Power + Clocks */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-800">
          <div className="flex items-start space-x-2">
            <Clock className="w-4 h-4 text-purple-400" />
            <div>
              <p className="text-xs text-gray-400">核心频率</p>
              <p className="text-sm text-gray-200">{clockSpeed} MHz</p>
            </div>
          </div>

          <div className="flex items-start space-x-2">
            <Zap className="w-4 h-4 text-amber-400" />
            <div>
              <p className="text-xs text-gray-400">实时功耗</p>
              <p className="text-sm text-gray-200">
                {powerDraw} W
                <span className="text-gray-400 text-xs">
                  {powerLimit ? ` / ${powerLimit} W` : ""}
                </span>
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
