import type { NextConfig } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';

/** Sidebar 显示的版本号：直接读仓库根目录的 version.py，避免两处不同步。 */
function appVersion(): string {
  try {
    const source = readFileSync(join(process.cwd(), '..', 'version.py'), 'utf8');
    return /VERSION\s*=\s*["']([^"']+)["']/.exec(source)?.[1] ?? 'unknown';
  } catch {
    return 'unknown';
  }
}

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_APP_VERSION: appVersion(),
  },
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
};

export default nextConfig;
