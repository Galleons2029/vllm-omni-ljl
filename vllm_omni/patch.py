# ============================================================================
# vllm_omni/patch.py  —  vLLM-Omni 启动时对 vLLM 的全局 monkey-patch
#
# 【执行时机】
#   `import vllm_omni` 时由 __init__.py 自动执行，发生在任何模型代码之前。
#   因此这里的改动对整个进程全局生效。
#
# 【patch 策略】
#   vLLM-Omni 需要扩展 vLLM 的核心数据类（Request、EngineCoreRequest 等），
#   让它们携带 Omni 特有的字段（prompt_embeds、additional_information 等）。
#   做法是定义 Omni 子类，然后在 sys.modules 中把所有 vllm 模块里对原始类的
#   引用替换成 Omni 子类——这样已经导入 vLLM 的代码也会透明地用上新类型。
# ============================================================================

import sys
from functools import cached_property

from aenum import extend_enum
from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus
from vllm.v1.request import StreamingUpdate as _OriginalStreamingUpdate

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

# =============================================================================
# Patch 1：ModelConfig.is_mm_prefix_lm
# =============================================================================
# 【背景】
#   HunyuanImage-3.0 的图像 token 需要双向注意力（bidirectional attention）。
#   vLLM 通过 ModelConfig.is_mm_prefix_lm 属性来决定是否启用双向注意力，
#   但该检查依赖一个内部白名单 MM_PREFIX_LM_MODELS，其中不包含
#   "hunyuan_image_3_moe"（HuggingFace config.json 里的 model_type）。
#
# 【为什么不在模型层修复】
#   is_mm_prefix_lm 在 vLLM 核心的调度器和 attention backend 选择阶段就被
#   检查，此时模型代码还没执行，模型层没有插入点。
#
# 【patch 技巧】
#   cached_property 存在 pydantic dataclass 的兼容问题（vllm 0.19.0+），
#   必须通过 __dict__ 直接访问描述符，绕过 cached_property.__get__ 的触发。
#   __set_name__ 是 Python descriptor protocol 的必要调用。
#
# 【脆弱性警告】
#   如果 vLLM 把 is_mm_prefix_lm 改成普通方法或删掉，此 patch 会在 import
#   时断言失败，而不是静默回退——fail-fast 设计，便于发现兼容性问题。
# =============================================================================
_OMNI_MM_PREFIX_LM_MODELS = ("hunyuan_image_3_moe",)
# 通过 __dict__ 访问，避免触发 cached_property.__get__（pydantic dataclass 下会抛异常）
_cp = _OriginalModelConfig.__dict__["is_mm_prefix_lm"]
_original_is_mm_prefix_lm = _cp.func if hasattr(_cp, "func") else _cp.fget


def _patched_is_mm_prefix_lm(self):
    # 先走原始逻辑；只有原始逻辑返回 False 时才检查 Omni 扩展名单
    if _original_is_mm_prefix_lm(self):
        return True
    model_type = getattr(self.hf_config, "model_type", "")
    return model_type in _OMNI_MM_PREFIX_LM_MODELS


_patched_cp = cached_property(_patched_is_mm_prefix_lm)
_patched_cp.__set_name__(_OriginalModelConfig, "is_mm_prefix_lm")
_OriginalModelConfig.is_mm_prefix_lm = _patched_cp

# 安全断言：确认 patch 已生效，任何 vLLM 内部变更都会在 import 时立即暴露
_installed = _OriginalModelConfig.__dict__.get("is_mm_prefix_lm")
assert _installed is _patched_cp, (
    "is_mm_prefix_lm patch failed to install — bidirectional attention "
    "for HunyuanImage3 will not work. Check vLLM ModelConfig changes."
)

# =============================================================================
# Patch 2：GlmImageTextConfig.rope_parameters
# =============================================================================
# 【背景】
#   GLM-Image 使用多模态旋转位置编码（M-RoPE），需要 mrope_section: [8,12,12]
#   描述时间/高/宽三个维度的 head 分配。
#   transformers 的 GlmImageTextConfig.__init__ 不把这个字段写入 rope_parameters，
#   导致 vLLM 的 uses_mrope 检测（依赖 "mrope_section" in rope_parameters）失败。
# =============================================================================
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig

    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        # Ensure rope_parameters exists and contains mrope_section
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            # GLM-Image uses mrope_section: [8, 12, 12] for T/H/W dimensions
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    # GlmImageTextConfig not available, skip patching
    pass

# =============================================================================
# Patch 3：RequestStatus 枚举扩展
# =============================================================================
# 【背景】
#   流式 TTS（async_chunk 模式）中，一个请求完成一个音频 chunk 后进入"等待
#   下一个 chunk"状态，不应被 vLLM 调度器当作已完成。
#   vLLM 的 RequestStatus 枚举是固定的，用 aenum.extend_enum 在运行时追加新成员。
#   值选 -1 是为了让它不满足 vLLM 现有的"完成状态"判断条件（通常检查 value > 0）。
# =============================================================================
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)

# =============================================================================
# Patch 4：sys.modules 全局替换（最核心的 patch）
# =============================================================================
# 【原理】
#   Python import 系统把所有已加载模块放在 sys.modules 字典里。
#   vLLM 各模块在 import 时把原始类绑定到模块的全局命名空间（如
#   `from vllm.v1.engine import EngineCoreRequest`），这些绑定独立于
#   原始类所在的模块——仅替换源模块的类定义不够，还要替换所有引用点。
#
# 【遍历策略】
#   只处理名称含 "vllm" 的模块（跳过第三方），对每个已知的原始类做
#   "身份相等"检查（`module.Foo == _OriginalFoo`），只替换真正持有原始类
#   的绑定，避免误替换同名但不相关的属性。
#
# 【被替换的类及其扩展内容】
#   - EngineCoreOutput / EngineCoreOutputs：增加 omni 多模态输出字段
#   - EngineCoreRequest：增加 prompt_embeds、additional_information 字段
#   - TokensPrompt：增加 prompt_embeds、additional_information 字段
#   - MRotaryEmbedding：支持 Omni 扩展的 M-RoPE 实现
#   - Request：增加 Omni 状态字段（WAITING_FOR_CHUNK 等）
#   - StreamingUpdate：增加流式 TTS 所需的增量更新字段
# =============================================================================
for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "StreamingUpdate") and module.StreamingUpdate == _OriginalStreamingUpdate:
        module.StreamingUpdate = OmniStreamingUpdate
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
