'use client';

import GpuMonitor from '@/components/GPUMonitor';
import CPUWidget from '@/components/CPUWidget';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import useMonitorStream from '@/hooks/useMonitorStream';
import Link from 'next/link';

export default function Dashboard() {
  // Same SSE stream the GPU monitor uses — no extra polling.
  const { cpu } = useMonitorStream();

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">仪表盘</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <GpuMonitor />
          <CPUWidget cpu={cpu} />
        </div>
        <div className="w-full mt-4">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-md">仪表盘</h1>
            <div className="text-xs text-gray-500">
              <Link href="/jobs">查看全部</Link>
            </div>
          </div>
          <JobsTable onlyActive />
        </div>
      </MainContent>
    </>
  );
}
