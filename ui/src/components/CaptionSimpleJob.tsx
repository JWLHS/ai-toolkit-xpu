import React from 'react';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  SelectInput,
  SliderInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig } from '@/types';
import { handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import {
  batchSizeOptions,
  captionerTypes,
  defaultQtype,
  groupedCaptionerTypes,
  maxNewTokensOptions,
  maxResOptions,
  quantizationOptions,
} from '@/helpers/captionOptions';

type Props = {
  jobConfig: CaptionJobConfig;
  setJobConfig: (value: any, key?: string) => void;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  showGPUSelect: boolean;
};

const CaptionSimpleJob: React.FC<Props> = ({ jobConfig, setJobConfig, gpuIDs, setGpuIDs, gpuList, showGPUSelect }) => {
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const additionalSections = selectedCaptionOption?.additionalSections || [];
  const captionPrompts = selectedCaptionOption?.captionPrompts || {};
  const promptPresetNames = Object.keys(captionPrompts);
  const minNewTokens = selectedCaptionOption?.minNewTokens ?? 0;
  const newTokensOptions = maxNewTokensOptions.filter(option => parseInt(option.value) >= minNewTokens);

  return (
    <div className="text-sm text-gray-400">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="标注器类型"
            value={jobConfig.config.process[0].type}
            onChange={value => {
              handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
            }}
            options={groupedCaptionerTypes}
          />
        </div>
        {showGPUSelect && (
          <div>
            <SelectInput
              label="GPU 编号"
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
      </div>
      <div className="mt-4">
        <CreatableSelectInput
          label="名称或路径"
          value={jobConfig.config.process[0].caption.model_name_or_path}
          docKey="config.process[0].caption.model_name_or_path"
          onChange={(value: string | null) => {
            if (value?.trim() === '') {
              value = null;
            }
            setJobConfig(value, 'config.process[0].caption.model_name_or_path');
          }}
          placeholder=""
          options={selectedCaptionOption?.name_or_path_options || []}
          required
        />
      </div>
      {additionalSections.includes('caption.model_name_or_path2') && (
        <div className="mt-4">
          <CreatableSelectInput
            label="名称或路径 2"
            value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
            onChange={(value: string | null) => {
              if (value?.trim() === '') {
                value = null;
              }
              setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
            }}
            placeholder=""
            options={selectedCaptionOption?.name_or_path2_options || []}
          />
        </div>
      )}
      {additionalSections.includes('caption.fixed_caption') && (
        <div className="mt-4">
          <TextInput
            label="固定标注"
            value={jobConfig.config.process[0].caption.fixed_caption || ''}
            onChange={value => {
              if (value?.trim() === '') {
                //@ts-ignore
                value = undefined;
              }
              setJobConfig(value, 'config.process[0].caption.fixed_caption');
            }}
            placeholder="输入固定标注（若所有音频共用同一条标注）"
          />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="量化"
            value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
            onChange={value => {
              if (value === '') {
                setJobConfig(false, 'config.process[0].caption.quantize');
                value = defaultQtype;
              } else {
                setJobConfig(true, 'config.process[0].caption.quantize');
              }
              setJobConfig(value, 'config.process[0].caption.qtype');
            }}
            options={quantizationOptions}
          />
          <div className="mt-4">
            <CreatableSelectInput
              label="标注扩展名"
              value={jobConfig.config.process[0].caption.caption_extension || 'txt'}
              onChange={value => {
                setJobConfig(value, 'config.process[0].caption.caption_extension');
              }}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          {additionalSections.includes('caption.max_res') && (
            <div className="mt-4">
              <SelectInput
                label="最大分辨率"
                value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_res');
                  }
                }}
                options={maxResOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.max_new_tokens') && (
            <div className="mt-4">
              <SelectInput
                label="最大新 token 数"
                value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                  }
                }}
                options={newTokensOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.batch_size') && (
            <div className="mt-4">
              <SelectInput
                label="批大小"
                value={`${jobConfig.config.process[0].caption.batch_size || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.batch_size');
                  }
                }}
                options={batchSizeOptions}
              />
            </div>
          )}
        </div>
        <div>
          <FormGroup label="选项">
            <Checkbox
              label="低显存"
              checked={jobConfig.config.process[0].caption.low_vram}
              onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
            />
            <Checkbox
              label="重新打标"
              checked={jobConfig.config.process[0].caption.recaption}
              onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
            />
            <Checkbox
              label="编译模型"
              checked={jobConfig.config.process[0].caption.compile || false}
              onChange={value => setJobConfig(value, 'config.process[0].caption.compile')}
            />
            {additionalSections.includes('caption.thinking') && (
              <Checkbox
                label="思考"
                checked={jobConfig.config.process[0].caption.thinking || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.thinking')}
              />
            )}
            {additionalSections.includes('caption.layer_offloading') && (
              <>
                <Checkbox
                  label="层级卸载"
                  checked={jobConfig.config.process[0].caption.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.layer_offloading')}
                />
                {jobConfig.config.process[0].caption.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="卸载比例 %"
                      value={Math.round((jobConfig.config.process[0].caption.layer_offloading_percent ?? 1) * 100)}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].caption.layer_offloading_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </FormGroup>
        </div>
      </div>
      {additionalSections.includes('caption.caption_prompt') && (
        <div className="mt-4">
          {promptPresetNames.length > 1 && (
            <div className="mb-4">
              <SelectInput
                label="提示词预设"
                value={
                  promptPresetNames.find(
                    name => captionPrompts[name] === jobConfig.config.process[0].caption.caption_prompt,
                  ) || ''
                }
                onChange={value => {
                  if (captionPrompts[value] !== undefined) {
                    setJobConfig(captionPrompts[value], 'config.process[0].caption.caption_prompt');
                  }
                }}
                options={[
                  { value: '', label: '- Custom -' },
                  ...promptPresetNames.map(name => ({ value: name, label: name })),
                ]}
              />
            </div>
          )}
          <TextAreaInput
            label="标注提示词"
            value={jobConfig.config.process[0].caption.caption_prompt || ''}
            onChange={value => {
              setJobConfig(value, 'config.process[0].caption.caption_prompt');
            }}
            placeholder="输入标注提示词"
          />
        </div>
      )}
    </div>
  );
};

export default CaptionSimpleJob;
