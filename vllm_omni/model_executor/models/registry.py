# ============================================================================
# vllm_omni/model_executor/models/registry.py  —  Omni 模型注册表
#
# 【作用】
#   vLLM 通过模型注册表把 HuggingFace config.json 中的 "model_type" 字符串
#   映射到具体的 Python 模型类。vLLM-Omni 在 vLLM 原有注册表之上追加
#   _OMNI_MODELS，使得多模态/非自回归模型也能被 vLLM 的模型加载器找到。
#
# 【字典格式】
#   _OMNI_MODELS = {
#       "HF_model_type_str": (
#           "models子目录名",     # vllm_omni/model_executor/models/<子目录>/
#           "模块文件名（不含.py）",
#           "类名",
#       ),
#       ...
#   }
#
#   最终映射路径：
#     vllm_omni.model_executor.models.<子目录>.<模块文件名>.<类名>
#
# 【如何新增一个模型】
#   1. 在 vllm_omni/model_executor/models/ 下创建新子目录
#   2. 在子目录中实现模型类（继承 vLLM 的 ModelBase）
#   3. 在 _OMNI_MODELS 中添加一条 "HF_model_type" → (子目录, 模块, 类) 的映射
#   4. 在 vllm_omni/transformers_utils/configs/ 中注册对应的 HF 配置类（如需要）
# ============================================================================

from vllm.model_executor.models.registry import (
    _VLLM_MODELS,
    _LazyRegisteredModel,
    _ModelRegistry,
)

# Omni 特有模型映射表
# key   = HuggingFace config.json 中的 "model_type" 字段值
# value = (models子目录, 模块文件名, 类名)
_OMNI_MODELS = {
    # -------------------------------------------------------------------------
    # Qwen2.5-Omni：多模态全家桶（Thinker LLM → Talker TTS → Token2Wav 解码器）
    # -------------------------------------------------------------------------
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni",
        "qwen2_5_omni",
        "Qwen2_5OmniForConditionalGeneration",
    ),
    "Qwen2_5OmniThinkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen2_5OmniTalkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_talker",
        "Qwen2_5OmniTalkerForConditionalGeneration",
    ),
    "Qwen2_5OmniToken2WavModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavForConditionalGenerationVLLM",
    ),
    "Qwen2_5OmniToken2WavDiTModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavModel",
    ),
    "Qwen2ForCausalLM_old": ("qwen2_5_omni", "qwen2_old", "Qwen2ForCausalLM"),  # need to discuss
    # -------------------------------------------------------------------------
    # Qwen3-Omni MoE：同上，MoE 架构版本
    # -------------------------------------------------------------------------
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni",
        "Qwen3OmniMoeForConditionalGeneration",
    ),
    "Qwen3OmniMoeThinkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeTalkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_talker",
        "Qwen3OmniMoeTalkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeCode2Wav": (
        "qwen3_omni",
        "qwen3_omni_code2wav",
        "Qwen3OmniMoeCode2Wav",
    ),
    # -------------------------------------------------------------------------
    # CosyVoice3：零样本语音克隆 TTS（Stage 0: AR Talker → Stage 1: Code2Wav）
    # -------------------------------------------------------------------------
    "CosyVoice3Model": (
        "cosyvoice3",
        "cosyvoice3",
        "CosyVoice3Model",
    ),
    # -------------------------------------------------------------------------
    # OmniVoice：端到端全模态语音模型
    # -------------------------------------------------------------------------
    "OmniVoiceModel": (
        "omnivoice",
        "omnivoice",
        "OmniVoiceModel",
    ),
    # -------------------------------------------------------------------------
    # MammothModa2：多模态生成（含 DiT pipeline）
    # -------------------------------------------------------------------------
    "MammothModa2Qwen2ForCausalLM": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2Qwen2ForCausalLM",
    ),
    "MammothModa2ARForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ARForConditionalGeneration",
    ),
    "MammothModa2DiTPipeline": (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    ),
    "MammothModa2ForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    "Mammothmoda2Model": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    # -------------------------------------------------------------------------
    # Qwen3-TTS：文本转语音（AR Talker → Code2Wav 两阶段）
    # -------------------------------------------------------------------------
    "Qwen3TTSForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSTalkerForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSCode2Wav": (
        "qwen3_tts",
        "qwen3_tts_code2wav",
        "Qwen3TTSCode2Wav",
    ),
    ## mimo_audio
    "MiMoAudioModel": (
        "mimo_audio",
        "mimo_audio",
        "MiMoAudioForConditionalGeneration",
    ),
    "MiMoV2ASRForCausalLM": (
        "mimo_audio",
        "mimo_audio",
        "MiMoAudioForConditionalGeneration",
    ),
    "MiMoAudioLLMModel": (
        "mimo_audio",
        "mimo_audio_llm",
        "MiMoAudioLLMForConditionalGeneration",
    ),
    "MiMoAudioToken2WavModel": (
        "mimo_audio",
        "mimo_audio_code2wav",
        "MiMoAudioToken2WavForConditionalGenerationVLLM",
    ),
    # -------------------------------------------------------------------------
    # GLM-Image：图文混合生成（AR + M-RoPE，需要 patch 2 中的 mrope_section 修复）
    # -------------------------------------------------------------------------
    ## glm_image
    "GlmImageForConditionalGeneration": (
        "glm_image",
        "glm_image_ar",
        "GlmImageForConditionalGeneration",
    ),
    # -------------------------------------------------------------------------
    # BAGEL：多模态理解与生成统一模型
    # -------------------------------------------------------------------------
    "OmniBagelForConditionalGeneration": (
        "bagel",
        "bagel",
        "OmniBagelForConditionalGeneration",
    ),
    # -------------------------------------------------------------------------
    # HunyuanImage3：混元图像生成（需要 patch 1 中的 is_mm_prefix_lm 修复）
    # -------------------------------------------------------------------------
    "HunyuanImage3ForCausalMM": (
        "hunyuan_image3",
        "hunyuan_image3",
        "HunyuanImage3ForConditionalGeneration",
    ),
    # -------------------------------------------------------------------------
    # Fish Speech：高质量 TTS（SlowAR + DAC 解码器）
    # -------------------------------------------------------------------------
    ## fish_speech (Fish Speech S2 Pro)
    "FishSpeechSlowARForConditionalGeneration": (
        "fish_speech",
        "fish_speech_slow_ar",
        "FishSpeechSlowARForConditionalGeneration",
    ),
    "FishSpeechDACDecoder": (
        "fish_speech",
        "fish_speech_dac_decoder",
        "FishSpeechDACDecoder",
    ),
    ## VoxCPM
    "VoxCPMForConditionalGeneration": (
        "voxcpm",
        "voxcpm",
        "VoxCPMForConditionalGeneration",
    ),
    ## VoxCPM2
    "VoxCPM2TalkerForConditionalGeneration": (
        "voxcpm2",
        "voxcpm2_talker",
        "VoxCPM2TalkerForConditionalGeneration",
    ),
    ## Voxtral TTS
    "VoxtralTTSForConditionalGeneration": (
        "voxtral_tts",
        "voxtral_tts",
        "VoxtralTTSForConditionalGeneration",
    ),
    "VoxtralTTSAudioGeneration": (
        "voxtral_tts",
        "voxtral_tts_audio_generation",
        "VoxtralTTSAudioGenerationForConditionalGeneration",
    ),
    "VoxtralTTSAudioTokenizer": ("voxtral_tts", "voxtral_tts_audio_tokenizer", "VoxtralTTSAudioTokenizer"),
    ## MOSS-TTS-Nano
    "MossTTSNanoForCausalLM": (
        "moss_tts_nano",
        "modeling_moss_tts_nano",
        "MossTTSNanoForGeneration",
    ),
    "DyninOmniForConditionalGeneration": (
        "dynin_omni",
        "dynin_omni",
        "DyninOmniForConditionalGeneration",
    ),
    ## Ming-flash-omni-2.0
    "MingFlashOmniForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni",
        "MingFlashOmniForConditionalGeneration",
    ),
    "MingFlashOmniThinkerForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni_thinker",
        "MingFlashOmniThinkerForConditionalGeneration",
    ),
    "MingFlashOmniTalkerForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni_talker",
        "MingFlashOmniTalkerForConditionalGeneration",
    ),
    # Alias: HF repo currently ships this architecture name in config.json
    "BailingMM2NativeForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni",
        "MingFlashOmniForConditionalGeneration",
    ),
}


# ============================================================================
# 最终注册表：vLLM 原有模型 + Omni 扩展模型
#
# _LazyRegisteredModel 是懒加载包装器：
#   - vLLM 原有模型：module_name = "vllm.model_executor.models.<模块>"
#   - Omni 模型：    module_name = "vllm_omni.model_executor.models.<子目录>.<模块>"
#
# 懒加载意味着模型类只在真正需要时才 import，避免启动时加载所有模型的重依赖。
# ============================================================================
_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,
    **_OMNI_MODELS,
}

OmniModelRegistry = _ModelRegistry(
    {
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
        },
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
