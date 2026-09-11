import React, { useMemo } from 'react';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import useMonitorStream from '@/hooks/useMonitorStream';

const GpuMonitor: React.FC = () => {
  // Live samples arrive every MONITOR_TICK_MS over the shared /api/monitor SSE stream
  const { gpu: gpuData, lastUpdated } = useMonitorStream();
  const loading = gpuData === null;
  const error = null;

  const getGridClasses = (gpuCount: number): string => {
    switch (gpuCount) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-1 sm:grid-cols-2';
      case 3:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
      case 4:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4';
      case 5:
      case 6:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
      case 7:
      case 8:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4';
      case 9:
      case 10:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-5';
      default:
        return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
    }
  };

  console.log('state', {
    loading,
    gpuData,
    error,
    lastUpdated,
  });

  const content = useMemo(() => {
    if (loading && !gpuData) {
      return <Loading />;
    }

    if (error) {
      return (
        <div className="bg-red-900 border border-red-600 text-red-200 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">出错了！</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      );
    }

    if (!gpuData) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">暂无显卡监控数据。</span>
        </div>
      );
    }

    // Intel Arc reports through xpu-smi, NVIDIA through nvidia-smi. Only when
    // neither produced a device is there really nothing to show.
    if (gpuData.gpus.length === 0) {
      const hint =
        gpuData.backend === 'xpu'
          ? '已启用 Intel XPU 监控（xpu-smi），但没读到显卡。请确认 Intel 显卡驱动与 XPU Manager（xpu-smi）已安装。'
          : gpuData.hasNvidiaSmi
            ? 'nvidia-smi 可用，但没有检测到 NVIDIA 显卡。'
            : '未检测到可用的显卡监控工具：NVIDIA 需要 nvidia-smi；Intel Arc 需要 xpu-smi（随 Intel 显卡驱动 / XPU Manager 安装）。';
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">未检测到显卡监控数据</strong>
          <span className="block sm:inline"> {hint}</span>
          {gpuData.error && <p className="mt-2 text-sm">{gpuData.error}</p>}
        </div>
      );
    }

    const gridClass = getGridClasses(gpuData?.gpus?.length || 1);

    return (
      <div className={`grid ${gridClass} gap-3`}>
        {gpuData.gpus.map((gpu, idx) => (
          <GPUWidget key={idx} gpu={gpu} />
        ))}
      </div>
    );
  }, [loading, gpuData, error]);

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <h1 className="text-md">显卡监控</h1>
        <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString()}</div>
      </div>
      {content}
    </div>
  );
};

export default GpuMonitor;
