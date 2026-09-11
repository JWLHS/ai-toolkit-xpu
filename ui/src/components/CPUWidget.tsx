import React from 'react';
import { CpuInfo } from '@/types';
import { Activity, Gauge, HardDrive, Layers, Thermometer, Zap } from 'lucide-react';

interface CPUWidgetProps {
  cpu: CpuInfo | null;
}

const formatMemory = (mb: number): string =>
  mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;

const formatMHz = (mhz: number): string => (mhz >= 1000 ? `${(mhz / 1000).toFixed(2)} GHz` : `${Math.round(mhz)} MHz`);

const loadColor = (value: number): string => (value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500');

const tempColor = (temp: number): string =>
  temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';

/** CPU / 内存资源卡片：负载、内存占用、频率、温度、线程数。 */
export default function CPUWidget({ cpu }: CPUWidgetProps) {
  if (!cpu) {
    return (
      <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
        <div className="bg-gray-800 px-4 py-3">
          <h2 className="font-semibold text-gray-100">CPU 信息</h2>
        </div>
        <div className="p-4">
          <p className="text-sm text-gray-400">暂无 CPU 数据</p>
        </div>
      </div>
    );
  }

  const memoryUsed = Math.max(0, cpu.totalMemory - cpu.availableMemory);
  const memoryPercent = cpu.totalMemory > 0 ? (memoryUsed / cpu.totalMemory) * 100 : 0;
  const speed = cpu.speedMHz ?? 0;
  const cached = Math.max(0, cpu.availableMemory - cpu.freeMemory);

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <h2 className="font-semibold text-gray-100">{cpu.name}</h2>
        </div>
        <span className="text-xs text-gray-400">{cpu.cores} 线程</span>
      </div>

      <div className="p-4 space-y-3">
        <div>
          <div className="flex items-center space-x-2 mb-1">
            <Activity className="w-4 h-4 text-gray-400" />
            <p className="text-xs text-gray-400">CPU 负载</p>
            <span className="text-xs text-gray-300 ml-auto">{cpu.currentLoad.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1">
            <div
              className={`h-1 rounded-full transition-all ${loadColor(cpu.currentLoad)}`}
              style={{ width: `${Math.min(100, cpu.currentLoad)}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex items-center space-x-2 mb-1">
            <HardDrive className="w-4 h-4 text-blue-500" />
            <p className="text-xs text-gray-400">内存占用</p>
            <span className="text-xs text-gray-300 ml-auto">{memoryPercent.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1">
            <div
              className={`h-1 rounded-full transition-all ${loadColor(memoryPercent)}`}
              style={{ width: `${Math.min(100, memoryPercent)}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-0.5">
            {formatMemory(memoryUsed)} / {formatMemory(cpu.totalMemory)}（可用 {formatMemory(cpu.availableMemory)}）
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-1">
          <div className="flex items-center space-x-2">
            <Gauge className="w-4 h-4 text-emerald-500" />
            <div>
              <p className="text-xs text-gray-400">频率</p>
              <p className="text-sm font-medium text-emerald-500">{speed > 0 ? formatMHz(speed) : '未提供'}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Thermometer className={`w-4 h-4 ${cpu.temperature > 0 ? tempColor(cpu.temperature) : 'text-gray-500'}`} />
            <div>
              <p className="text-xs text-gray-400">温度</p>
              <p className={`text-sm font-medium ${cpu.temperature > 0 ? tempColor(cpu.temperature) : 'text-gray-500'}`}>
                {cpu.temperature > 0 ? `${cpu.temperature.toFixed(0)}°C` : '未提供'}
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2 pt-2 border-t border-gray-800">
          <Layers className="w-3.5 h-3.5 text-gray-500" />
          <p className="text-xs text-gray-500">
            空闲 {formatMemory(cpu.freeMemory)} · 缓存/其他 {formatMemory(cached)}
          </p>
        </div>
      </div>
    </div>
  );
}
