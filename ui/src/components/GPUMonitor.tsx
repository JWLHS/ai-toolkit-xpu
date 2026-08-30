import React, { useState, useEffect, useRef, useMemo } from 'react';
import { GPUApiResponse } from '@/types';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import { apiClient } from '@/utils/api';

const GpuMonitor: React.FC = () => {
  const [gpuData, setGpuData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const isFetchingRef = useRef(false);

  useEffect(() => {
    const fetchGpuInfo = async () => {
      if (isFetchingRef.current) return;
      isFetchingRef.current = true;
      setLoading(true);

      try {
        const res = await apiClient.get('/api/gpu');
        setGpuData(res.data || {});
        setLastUpdated(new Date());
        setError(null);
      } catch (err: any) {
        setError('无法获取 GPU 数据：' + (err?.message || String(err)));
      } finally {
        setLoading(false);
        isFetchingRef.current = false;
      }
    };

    fetchGpuInfo();
    const timer = setInterval(fetchGpuInfo, 1000);
    return () => clearInterval(timer);
  }, []);

  // 智能响应式网格：手机1列 → 平板2列 → 电脑多列自动适配
  const getGridClass = (count: number) => {
    if (count === 1) return 'grid-cols-1';
    if (count === 2) return 'grid-cols-1 sm:grid-cols-2';
    if (count <= 4) return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4';
    if (count <= 6) return 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6';
    return 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6';
  };

  const content = useMemo(() => {
    // 首次加载
    if (loading && !gpuData) return <Loading />;

    // 错误
    if (error) {
      return (
        <div className="bg-red-900/60 border border-red-700 text-red-200 px-5 py-4 rounded-xl backdrop-blur">
          <strong className="font-bold">出错了！</strong> {error}
        </div>
      );
    }

    // 无数据
    if (!gpuData || !gpuData.gpus || gpuData.gpus.length === 0) {
      const backend = gpuData?.backend || '未知';
      return (
        <div className="bg-amber-900/60 border border-amber-700 text-amber-200 px-5 py-4 rounded-xl backdrop-blur text-center">
          未检测到 GPU（后端：{backend}）
          {gpuData?.error && <p className="text-sm mt-2 opacity-80">{gpuData.error}</p>}
        </div>
      );
    }

    const gpus = gpuData.gpus;
    const backend = gpuData.backend || (gpuData.hasNvidiaSmi ? 'CUDA' : '其他');

    return (
      <>
        {/* 标题栏 + 在线数量 + 刷新状态 */}
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 mb-5">
          <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-3">
            GPU 实时监控
            <span className="px-3 py-1 bg-emerald-900/80 text-emerald-300 text-sm rounded-full border border-emerald-700">
              {gpus.length} 张在线
            </span>
          </h1>

          <div className="flex items-center gap-4 text-sm text-gray-400">
            {loading && (
              <span className="flex items-center gap-1 text-amber-400">
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                刷新中...
              </span>
            )}
            <span>更新时间：{lastUpdated?.toLocaleTimeString() || '--'}</span>
          </div>
        </div>

        {/* 响应式卡片网格 */}
        <div className={`grid ${getGridClass(gpus.length)} gap-4`}>
          {gpus.map((gpu: any, i: number) => (
            <GPUWidget key={gpu.uuid || gpu.index || i} gpu={gpu} backend={backend} />
          ))}
        </div>
      </>
    );
  }, [loading, gpuData, error, lastUpdated]);

  return (
    <div className="w-full max-w-screen-2xl mx-auto p-4 sm:p-6">
      {content}
    </div>
  );
};

export default GpuMonitor;