'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">设置</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face 令牌
                    <div className="text-gray-500 text-sm ml-1">
                      访问受限或私有模型前，请先到{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        Huggingface
                      </a>{' '}
                      上创建一个 Read 令牌。
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入 Hugging Face 令牌"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    训练输出目录
                    <div className="text-gray-500 text-sm ml-1">
                      训练相关的信息都会保存在这里。必须是绝对路径。留空则默认使用项目根目录下的 output 文件夹。
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入训练输出路径"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    数据集目录
                    <div className="text-gray-500 text-sm ml-1">
                      数据集的存放与查找位置。{' '}
                      <span className="text-orange-800">
                        警告：本软件可能会修改数据集，建议另外保留一份备份，或专门指定一个目录供它使用。
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入数据集目录路径"
                  />
                </div>

                <div>
                  <label htmlFor="MODELS_PATH" className="block text-sm font-medium mb-2">
                    模型目录
                    <div className="text-gray-500 text-sm ml-1">
                      部分模型支持直接加载 ComfyUI 的模型权重，这类模型会从这个路径加载或下载到这里。必须是绝对路径。留空则默认使用项目根目录下的 models 文件夹。
                    </div>
                  </label>
                  <input
                    type="text"
                    id="MODELS_PATH"
                    name="MODELS_PATH"
                    value={settings.MODELS_PATH}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="输入模型目录路径"
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? '保存中...' : '保存设置'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">设置保存成功！</p>}
          {status === 'error' && <p className="text-red-500 text-center">保存失败，请重试。</p>}
        </form>
      </MainContent>
    </>
  );
}
